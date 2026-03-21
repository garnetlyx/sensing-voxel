"""
Vision-based scoring for calibration tower zones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any

import numpy as np


class SurfaceQuality(Enum):
    EXCELLENT = auto()
    GOOD = auto()
    FAIR = auto()
    POOR = auto()


class StringingLevel(Enum):
    NONE = auto()
    MINIMAL = auto()
    MODERATE = auto()
    SEVERE = auto()


@dataclass
class TowerZone:
    section_index: int
    value: int
    center_x_mm: float
    center_y_mm: float
    width_mm: float
    height_mm: float
    layer_start: int
    layer_end: int


@dataclass
class ZoneScore:
    zone: TowerZone
    surface_quality: SurfaceQuality
    surface_score: float
    stringing_level: StringingLevel
    stringing_score: float
    adhesion_score: float
    overall_score: float
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def value(self) -> int:
        return self.zone.value


@dataclass
class ScoringConfig:
    surface_weight: float = 0.4
    stringing_weight: float = 0.35
    adhesion_weight: float = 0.25
    laplacian_threshold: float = 100.0
    sobel_threshold: float = 50.0
    stringing_brightness_threshold: int = 200
    min_surface_variance: float = 500.0
    max_surface_variance: float = 5000.0

    def __post_init__(self):
        total = self.surface_weight + self.stringing_weight + self.adhesion_weight
        if not 0.99 <= total <= 1.01:
            raise ValueError("weights must sum to 1.0")


class VisualScorer:
    def __init__(self, config: ScoringConfig | None = None):
        self.config = config or ScoringConfig()

    def score_zone(
        self,
        zone: TowerZone,
        image: np.ndarray,
        calibration: Any | None = None,
    ) -> ZoneScore:
        roi = self._extract_roi(image, zone, calibration)

        if roi is None or roi.size == 0:
            return self._create_error_score(zone, "failed_to_extract_roi")

        surface_quality, surface_score = self._score_surface(roi)
        stringing_level, stringing_score = self._score_stringing(roi)
        adhesion_score = self._score_adhesion(roi)

        overall = (
            surface_score * self.config.surface_weight
            + stringing_score * self.config.stringing_weight
            + adhesion_score * self.config.adhesion_weight
        )

        details = {
            "roi_shape": roi.shape,
            "mean_brightness": float(np.mean(roi)),
            "brightness_std": float(np.std(roi)),
        }

        return ZoneScore(
            zone=zone,
            surface_quality=surface_quality,
            surface_score=surface_score,
            stringing_level=stringing_level,
            stringing_score=stringing_score,
            adhesion_score=adhesion_score,
            overall_score=overall,
            details=details,
        )

    def score_all_zones(
        self,
        zones: list[TowerZone],
        images: list[np.ndarray],
        calibration: Any | None = None,
    ) -> list[ZoneScore]:
        return [self.score_zone(zone, img, calibration) for zone, img in zip(zones, images)]

    def recommend_optimal(
        self,
        scores: list[ZoneScore],
        min_acceptable_score: float = 0.6,
    ) -> tuple[int, ZoneScore | None, list[dict]]:
        if not scores:
            return 0, None, []

        sorted_scores = sorted(scores, key=lambda s: s.overall_score, reverse=True)
        best = sorted_scores[0]

        if best.overall_score < min_acceptable_score:
            return best.value, best, self._create_comparison(sorted_scores)

        candidates = [s for s in sorted_scores if s.overall_score >= min_acceptable_score]

        best_balanced = self._select_balanced(candidates)

        return best_balanced.value, best_balanced, self._create_comparison(sorted_scores)

    def _extract_roi(
        self,
        image: np.ndarray,
        zone: TowerZone,
        calibration: Any | None,
    ) -> np.ndarray | None:
        if image is None or image.size == 0:
            return None

        if len(image.shape) < 2:
            return None

        h, w = image.shape[:2]

        if calibration and hasattr(calibration, "mm_to_pixels"):
            px, py = calibration.mm_to_pixels(zone.center_x_mm, zone.center_y_mm)
            pw = zone.width_mm * calibration.mm_per_pixel
            ph = zone.height_mm * calibration.mm_per_pixel
        else:
            px = int(zone.center_x_mm * 10)
            py = int(zone.center_y_mm * 10)
            pw = int(zone.width_mm * 10)
            ph = int(zone.height_mm * 10)

        x1 = max(0, int(px - pw / 2))
        y1 = max(0, int(py - ph / 2))
        x2 = min(w, int(px + pw / 2))
        y2 = min(h, int(py + ph / 2))

        if x2 <= x1 or y2 <= y1:
            return None

        roi = image[y1:y2, x1:x2]

        if len(roi.shape) == 3:
            roi = np.mean(roi, axis=2).astype(np.uint8)

        return roi

    def _score_surface(self, roi: np.ndarray) -> tuple[SurfaceQuality, float]:
        laplacian = cv2.Laplacian(roi, cv2.CV_64F)
        laplacian_var = laplacian.var()

        sobelx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2).mean()

        sharpness_score = min(1.0, laplacian_var / self.config.laplacian_threshold)
        edge_score = min(1.0, sobel_magnitude / self.config.sobel_threshold)

        surface_score = 0.7 * sharpness_score + 0.3 * edge_score

        variance = np.var(roi.astype(float))
        variance_score = 1.0 - min(
            1.0,
            max(0, variance - self.config.min_surface_variance)
            / (self.config.max_surface_variance - self.config.min_surface_variance),
        )
        surface_score = 0.6 * surface_score + 0.4 * variance_score

        if surface_score >= 0.85:
            quality = SurfaceQuality.EXCELLENT
        elif surface_score >= 0.7:
            quality = SurfaceQuality.GOOD
        elif surface_score >= 0.5:
            quality = SurfaceQuality.FAIR
        else:
            quality = SurfaceQuality.POOR

        return quality, float(surface_score)

    def _score_stringing(self, roi: np.ndarray) -> tuple[StringingLevel, float]:
        bright_pixels = np.sum(roi > self.config.stringing_brightness_threshold)
        bright_ratio = bright_pixels / roi.size

        edges = cv2.Canny(roi, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=20, maxLineGap=5)

        line_score = 1.0
        if lines is not None:
            horizontal_lines = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
                if angle < 20 or angle > 160:
                    horizontal_lines += 1

            if horizontal_lines > 5:
                line_score = max(0.0, 1.0 - (horizontal_lines - 5) * 0.1)

        stringing_score = 0.6 * (1.0 - min(1.0, bright_ratio * 10)) + 0.4 * line_score

        if stringing_score >= 0.9:
            level = StringingLevel.NONE
        elif stringing_score >= 0.7:
            level = StringingLevel.MINIMAL
        elif stringing_score >= 0.4:
            level = StringingLevel.MODERATE
        else:
            level = StringingLevel.SEVERE

        return level, stringing_score

    def _score_adhesion(self, roi: np.ndarray) -> float:
        mean_brightness = np.mean(roi)
        brightness_score = 1.0 - abs(mean_brightness - 128) / 128

        edges = cv2.Canny(roi, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        edge_score = 1.0 - min(1.0, abs(edge_density - 0.05) * 10)

        return float(0.5 * brightness_score + 0.5 * edge_score)

    def _select_balanced(self, candidates: list[ZoneScore]) -> ZoneScore:
        def balance_score(s: ZoneScore) -> float:
            return s.surface_score * 0.35 + s.stringing_score * 0.35 + s.adhesion_score * 0.3

        return max(candidates, key=balance_score)

    def _create_comparison(self, sorted_scores: list[ZoneScore]) -> list[dict]:
        return [
            {
                "value": s.zone.value,
                "overall": round(s.overall_score, 3),
                "surface": round(s.surface_score, 3),
                "stringing": round(s.stringing_score, 3),
                "adhesion": round(s.adhesion_score, 3),
                "quality": s.surface_quality.name,
                "stringing_level": s.stringing_level.name,
            }
            for s in sorted_scores[:5]
        ]

    def _create_error_score(self, zone: TowerZone, error: str) -> ZoneScore:
        return ZoneScore(
            zone=zone,
            surface_quality=SurfaceQuality.POOR,
            surface_score=0.0,
            stringing_level=StringingLevel.SEVERE,
            stringing_score=0.0,
            adhesion_score=0.0,
            overall_score=0.0,
            details={"error": error},
        )


import cv2
