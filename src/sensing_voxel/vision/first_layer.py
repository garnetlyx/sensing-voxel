"""
First-layer quality classifier using OpenCV image analysis.

Detects gaps (Z too high) and ridges/scraping (Z too low) in first-layer
extrusion to drive Z-offset corrections.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from sensing_voxel.core.models import (
    FirstLayerAnalysisResult,
    FirstLayerBounds,
    FirstLayerQuality,
    HomographyMatrix,
    VisionFrame,
)

logger = logging.getLogger(__name__)


class FirstLayerClassifier:
    """Classify first-layer quality from camera frames."""

    MIN_COVERAGE_RATIO = 0.20  # Skip frames with < 20% printable content

    @staticmethod
    def compute_roi(
        bounds: FirstLayerBounds,
        homography: HomographyMatrix,
        margin_mm: float = 5.0,
    ) -> tuple[int, int, int, int]:
        """
        Map bed coordinates to pixel ROI using inverse homography.

        Args:
            bounds: First-layer XY bounds in mm
            homography: Camera-to-bed homography matrix
            margin_mm: Extra margin around bounds

        Returns:
            (x, y, w, h) pixel ROI tuple
        """
        H = np.array(homography.matrix)
        H_inv = np.linalg.inv(H)

        # Bed corners with margin
        bed_corners = np.array([
            [bounds.min_x - margin_mm, bounds.min_y - margin_mm],
            [bounds.max_x + margin_mm, bounds.min_y - margin_mm],
            [bounds.max_x + margin_mm, bounds.max_y + margin_mm],
            [bounds.min_x - margin_mm, bounds.max_y + margin_mm],
        ], dtype=np.float32).reshape(-1, 1, 2)

        # Transform bed coords → pixel coords
        pixel_corners = cv2.perspectiveTransform(bed_corners, H_inv).reshape(-1, 2)

        x_min = max(0, int(np.min(pixel_corners[:, 0])))
        y_min = max(0, int(np.min(pixel_corners[:, 1])))
        x_max = int(np.max(pixel_corners[:, 0]))
        y_max = int(np.max(pixel_corners[:, 1]))

        return (x_min, y_min, x_max - x_min, y_max - y_min)

    def analyze(
        self,
        frame: VisionFrame,
        roi: tuple[int, int, int, int],
    ) -> FirstLayerAnalysisResult:
        """
        Analyze first-layer quality in the ROI.

        Uses edge detection and intensity analysis to classify quality.

        Args:
            frame: Camera frame
            roi: (x, y, w, h) pixel region of interest

        Returns:
            FirstLayerAnalysisResult
        """
        x, y, w, h = roi
        img = frame.frame_data

        # Bounds check
        img_h, img_w = img.shape[:2]
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        if w < 10 or h < 10:
            return FirstLayerAnalysisResult(
                quality=FirstLayerQuality.ACCEPTABLE,
                confidence=0.0,
                coverage_ratio=0.0,
                defects=["roi_too_small"],
                roi=(x, y, w, h),
            )

        crop = img[y : y + h, x : x + w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Compute coverage: non-background pixels (above threshold)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        coverage_ratio = float(np.count_nonzero(binary)) / binary.size

        if coverage_ratio < self.MIN_COVERAGE_RATIO:
            return FirstLayerAnalysisResult(
                quality=FirstLayerQuality.ACCEPTABLE,
                confidence=0.1,
                coverage_ratio=coverage_ratio,
                defects=["low_coverage"],
                roi=(x, y, w, h),
            )

        # Edge detection for gap/ridge analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / edges.size

        # Intensity analysis
        mean_intensity = float(np.mean(gray))
        std_intensity = float(np.std(gray))

        # Classification logic
        defects = []
        quality = FirstLayerQuality.GOOD
        confidence = 0.7

        # High edge density + high intensity variation → gaps between extrusion lines (Z too high)
        if edge_density > 0.15 and std_intensity > 40:
            defects.append("gaps")
            quality = FirstLayerQuality.POOR
            confidence = min(0.9, 0.5 + edge_density * 2)

        # Very low std (uniform bright) + high mean → ridging/scraping (Z too low)
        elif std_intensity < 15 and mean_intensity > 180:
            defects.append("ridges")
            quality = FirstLayerQuality.POOR
            confidence = 0.75

        # Moderate variation → acceptable
        elif edge_density > 0.08 or std_intensity > 30:
            quality = FirstLayerQuality.ACCEPTABLE
            confidence = 0.6

        # Very uniform, good coverage → excellent
        elif std_intensity < 20 and coverage_ratio > 0.6:
            quality = FirstLayerQuality.EXCELLENT
            confidence = 0.85

        return FirstLayerAnalysisResult(
            quality=quality,
            confidence=confidence,
            coverage_ratio=coverage_ratio,
            defects=defects,
            roi=(x, y, w, h),
        )
