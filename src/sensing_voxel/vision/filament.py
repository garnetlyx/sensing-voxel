"""Filament spool monitoring with volume estimation and tangle detection."""

from __future__ import annotations

import asyncio
import logging
import math
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import cv2
import numpy as np

from sensing_voxel.config import FilamentMonitorConfig
from sensing_voxel.core.models import (
    FilamentCheckResult,
    FilamentSpoolMeasurement,
    FilamentSpoolStatus,
    TangleDecision,
    TangleDetectionResult,
    VisionFrame,
)
from sensing_voxel.vision.usb_camera import USBCameraStream

logger = logging.getLogger(__name__)


@dataclass
class SpoolGeometry:
    """Calculated spool geometry from image analysis."""

    spool_diameter_px: float
    filament_diameter_px: float
    center_x: float
    center_y: float
    confidence: float


class SpoolAnalyzer:
    """Analyze spool images to estimate remaining filament."""

    SPOOL_DIAMETER_FULL_MM = 200.0
    SPOOL_DIAMETER_EMPTY_MM = 80.0
    HUB_DIAMETER_MM = 55.0

    def __init__(self, config: FilamentMonitorConfig):
        self.config = config

    def analyze(self, frame: VisionFrame) -> FilamentSpoolMeasurement:
        """Analyze frame and estimate remaining filament."""
        img = frame.frame_data
        if img is None or img.size == 0:
            return FilamentSpoolMeasurement(
                estimated_diameter_mm=0.0,
                estimated_weight_g=0.0,
                confidence=0.0,
                timestamp=frame.timestamp,
                frame_id=frame.frame_id,
            )

        geometry = self._detect_spool_geometry(img)

        if geometry.confidence < 0.3:
            return FilamentSpoolMeasurement(
                estimated_diameter_mm=0.0,
                estimated_weight_g=0.0,
                confidence=geometry.confidence,
                timestamp=frame.timestamp,
                frame_id=frame.frame_id,
            )

        diameter_mm = self._px_to_mm(geometry.filament_diameter_px, img.shape[1])
        weight_g = self._estimate_weight(diameter_mm)

        return FilamentSpoolMeasurement(
            estimated_diameter_mm=diameter_mm,
            estimated_weight_g=weight_g,
            confidence=geometry.confidence,
            timestamp=frame.timestamp,
            frame_id=frame.frame_id,
        )

    def _detect_spool_geometry(self, img: np.ndarray) -> SpoolGeometry:
        """Detect spool boundaries using computer vision."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=100,
            param1=50,
            param2=30,
            minRadius=50,
            maxRadius=300,
        )

        if circles is None or len(circles) == 0:
            return self._fallback_detection(gray)

        circles = np.uint16(np.around(circles))
        largest = circles[0][0]
        center_x, center_y, radius = largest

        spool_diameter_px = radius * 2

        inner_edge = self._detect_inner_edge(gray, center_x, center_y, radius)
        filament_diameter_px = max(0, spool_diameter_px - inner_edge * 2)

        confidence = self._calculate_confidence(gray, center_x, center_y, radius)

        return SpoolGeometry(
            spool_diameter_px=spool_diameter_px,
            filament_diameter_px=filament_diameter_px,
            center_x=float(center_x),
            center_y=float(center_y),
            confidence=confidence,
        )

    def _fallback_detection(self, gray: np.ndarray) -> SpoolGeometry:
        """Fallback detection using contour analysis."""
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return SpoolGeometry(
                spool_diameter_px=0.0,
                filament_diameter_px=0.0,
                center_x=0.0,
                center_y=0.0,
                confidence=0.0,
            )

        largest = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest)

        if radius < 30:
            return SpoolGeometry(
                spool_diameter_px=0.0,
                filament_diameter_px=0.0,
                center_x=0.0,
                center_y=0.0,
                confidence=0.0,
            )

        return SpoolGeometry(
            spool_diameter_px=radius * 2,
            filament_diameter_px=radius * 1.5,
            center_x=x,
            center_y=y,
            confidence=0.4,
        )

    def _detect_inner_edge(
        self, gray: np.ndarray, center_x: int, center_y: int, outer_radius: int
    ) -> float:
        """Detect the inner edge of filament wound on spool."""
        line_length = int(outer_radius * 1.5)
        angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)

        edge_distances = []
        for angle in angles:
            x_end = int(center_x + line_length * np.cos(angle))
            y_end = int(center_y + line_length * np.sin(angle))

            line = self._get_line_profile(gray, center_x, center_y, x_end, y_end)
            edge = self._find_edge_in_profile(line, outer_radius)
            if edge > 0:
                edge_distances.append(edge)

        if not edge_distances:
            return outer_radius * 0.3

        return np.median(edge_distances)

    def _get_line_profile(self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """Extract intensity profile along a line."""
        length = int(np.hypot(x2 - x1, y2 - y1))
        if length == 0:
            return np.array([])

        x_vals = np.linspace(x1, x2, length).astype(int)
        y_vals = np.linspace(y1, y2, length).astype(int)

        x_vals = np.clip(x_vals, 0, img.shape[1] - 1)
        y_vals = np.clip(y_vals, 0, img.shape[0] - 1)

        return img[y_vals, x_vals]

    def _find_edge_in_profile(self, profile: np.ndarray, outer_radius: int) -> float:
        """Find the inner edge of filament in intensity profile."""
        if len(profile) < 10:
            return 0.0

        gradient = np.gradient(profile.astype(float))

        search_start = int(outer_radius * 0.2)
        search_end = int(outer_radius * 0.8)

        if search_end > len(gradient):
            search_end = len(gradient)

        if search_start >= search_end:
            search_start = 0

        search_region = gradient[search_start:search_end]
        if len(search_region) == 0:
            return 0.0

        edge_idx = np.argmax(np.abs(search_region))
        return float(search_start + edge_idx)

    def _calculate_confidence(
        self, gray: np.ndarray, center_x: int, center_y: int, radius: int
    ) -> float:
        """Calculate confidence score for detection."""
        h, w = gray.shape

        x1 = max(0, center_x - radius)
        y1 = max(0, center_y - radius)
        x2 = min(w, center_x + radius)
        y2 = min(h, center_y + radius)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0

        variance = np.var(roi.astype(float))
        contrast = np.max(roi) - np.min(roi)

        variance_score = min(1.0, variance / 1000.0)
        contrast_score = min(1.0, contrast / 100.0)

        center_x_norm = center_x / w
        center_y_norm = center_y / h
        center_score = 1.0 - abs(center_x_norm - 0.5) - abs(center_y_norm - 0.5)
        center_score = max(0.0, center_score)

        return float(variance_score * 0.3 + contrast_score * 0.4 + center_score * 0.3)

    def _px_to_mm(self, px_value: float, img_width_px: int) -> float:
        """Convert pixels to millimeters based on known spool size."""
        if img_width_px == 0:
            return 0.0
        mm_per_px = self.SPOOL_DIAMETER_FULL_MM / img_width_px
        return px_value * mm_per_px

    def _estimate_weight(self, filament_diameter_mm: float) -> float:
        """Estimate remaining filament weight from diameter."""
        full_diameter = self.SPOOL_DIAMETER_FULL_MM
        hub_diameter = self.HUB_DIAMETER_MM

        if filament_diameter_mm <= 0:
            return self.config.spool_empty_weight_g

        remaining_ratio = (filament_diameter_mm - hub_diameter) / (full_diameter - hub_diameter)
        remaining_ratio = max(0.0, min(1.0, remaining_ratio))

        filament_weight = (
            self.config.spool_full_weight_g - self.config.spool_empty_weight_g
        ) * remaining_ratio

        return self.config.spool_empty_weight_g + filament_weight


class TangleDetector:
    """Detect filament tangles on spool surface."""

    def __init__(self, config: FilamentMonitorConfig):
        self.config = config
        self._recent_detections: deque[bool] = deque(maxlen=10)

    def detect(self, frame: VisionFrame) -> TangleDetectionResult:
        """Detect tangles in spool image."""
        img = frame.frame_data
        if img is None or img.size == 0:
            return TangleDetectionResult(
                detected=False,
                confidence=0.0,
                severity=TangleDecision.NONE,
                bounding_boxes=[],
                timestamp=frame.timestamp,
                frame_id=frame.frame_id,
            )

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(gray, 50, 150)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=30,
            maxLineGap=10,
        )

        tangle_score = 0.0
        bounding_boxes = []

        if lines is not None:
            line_groups = self._group_intersecting_lines(lines)
            tangle_score, bounding_boxes = self._analyze_line_intersections(line_groups)

        self._recent_detections.append(tangle_score > self.config.tangle_detection_threshold)

        confirmation_count = sum(self._recent_detections)
        detected = confirmation_count >= self.config.tangle_confirmation_frames

        if detected:
            severity = (
                TangleDecision.CRITICAL if confirmation_count >= 5 else TangleDecision.WARNING
            )
        else:
            severity = TangleDecision.NONE

        confidence = min(1.0, tangle_score)

        return TangleDetectionResult(
            detected=detected,
            confidence=confidence,
            severity=severity,
            bounding_boxes=bounding_boxes[:5],
            timestamp=frame.timestamp,
            frame_id=frame.frame_id,
        )

    def _group_intersecting_lines(self, lines: np.ndarray) -> list[list[tuple[int, int, int, int]]]:
        """Group lines that intersect with each other."""
        if len(lines) == 0:
            return []

        line_list = [(int(l[0][0]), int(l[0][1]), int(l[0][2]), int(l[0][3])) for l in lines]
        groups = []
        used = set()

        for i, line1 in enumerate(line_list):
            if i in used:
                continue

            group = [line1]
            used.add(i)

            for j, line2 in enumerate(line_list[i + 1 :], start=i + 1):
                if j in used:
                    continue
                if self._lines_intersect(line1, line2):
                    group.append(line2)
                    used.add(j)

            if len(group) >= 2:
                groups.append(group)

        return groups

    def _lines_intersect(
        self, line1: tuple[int, int, int, int], line2: tuple[int, int, int, int]
    ) -> bool:
        """Check if two line segments intersect."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        def ccw(ax: int, ay: int, bx: int, by: int, cx: int, cy: int) -> bool:
            return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)

        return ccw(x1, y1, x3, y3, x4, y4) != ccw(x2, y2, x3, y3, x4, y4) and ccw(
            x1, y1, x2, y2, x3, y3
        ) != ccw(x1, y1, x2, y2, x4, y4)

    def _analyze_line_intersections(
        self, groups: list[list[tuple[int, int, int, int]]]
    ) -> tuple[float, list[tuple[float, float, float, float]]]:
        """Analyze line groups to calculate tangle score and bounding boxes."""
        if not groups:
            return 0.0, []

        total_score = 0.0
        bounding_boxes = []

        for group in groups:
            if len(group) < 2:
                continue

            intersection_score = len(group) * 0.15

            min_x = min(min(l[0], l[2]) for l in group)
            max_x = max(max(l[0], l[2]) for l in group)
            min_y = min(min(l[1], l[3]) for l in group)
            max_y = max(max(l[1], l[3]) for l in group)

            box_area = (max_x - min_x) * (max_y - min_y)
            density_score = min(0.5, len(group) / box_area * 1000) if box_area > 0 else 0

            group_score = min(1.0, intersection_score + density_score)
            total_score += group_score

            bounding_boxes.append(
                (float(min_x), float(min_y), float(max_x - min_x), float(max_y - min_y))
            )

        return min(1.0, total_score), bounding_boxes


class FilamentMonitor:
    """Main filament monitor combining volume estimation and tangle detection."""

    def __init__(self, config: FilamentMonitorConfig):
        self.config = config
        self.camera = USBCameraStream(config)
        self.spool_analyzer = SpoolAnalyzer(config)
        self.tangle_detector = TangleDetector(config)
        self._running = False
        self._monitor_task: asyncio.Task | None = None
        self._on_tangle: Callable[[TangleDetectionResult], Awaitable[None]] | None = None

    async def start(
        self,
        on_tangle: Callable[[TangleDetectionResult], Awaitable[None]] | None = None,
    ) -> bool:
        """Start filament monitoring."""
        if self._running:
            return True

        if not self.config.enabled:
            logger.info("Filament monitor disabled in config")
            return False

        if not await self.camera.start():
            logger.error("Failed to start USB camera")
            return False

        self._on_tangle = on_tangle
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Filament monitoring started")
        return True

    async def stop(self):
        """Stop filament monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        await self.camera.stop()
        logger.info("Filament monitoring stopped")

    async def measure_spool(self) -> FilamentSpoolMeasurement | None:
        """Take a single measurement of the spool."""
        frame = await self.camera.get_frame(timeout=5.0)
        if frame is None:
            return None
        return self.spool_analyzer.analyze(frame)

    async def check_tangle(self) -> TangleDetectionResult | None:
        """Check for tangles."""
        frame = await self.camera.get_frame(timeout=5.0)
        if frame is None:
            return None
        return self.tangle_detector.detect(frame)

    def check_filament_sufficient(
        self,
        measurement: FilamentSpoolMeasurement,
        estimated_required_g: float,
    ) -> FilamentCheckResult:
        """Check if remaining filament is sufficient for print job."""
        safety_margin = 1.0 + self.config.min_filament_safety_margin_percent / 100.0
        required_with_margin = estimated_required_g * safety_margin

        remaining = measurement.estimated_weight_g

        if remaining >= required_with_margin:
            status = FilamentSpoolStatus.SUFFICIENT
            message = (
                f"Filament sufficient: {remaining:.0f}g available, "
                f"{estimated_required_g:.0f}g required ({self.config.min_filament_safety_margin_percent}% margin)"
            )
        elif remaining >= estimated_required_g:
            status = FilamentSpoolStatus.LOW
            message = (
                f"Filament low: {remaining:.0f}g available, "
                f"{estimated_required_g:.0f}g required (below {self.config.min_filament_safety_margin_percent}% margin)"
            )
        else:
            status = FilamentSpoolStatus.INSUFFICIENT
            message = (
                f"Filament insufficient: {remaining:.0f}g available, "
                f"{estimated_required_g:.0f}g required"
            )

        return FilamentCheckResult(
            status=status,
            estimated_remaining_g=remaining,
            estimated_required_g=estimated_required_g,
            safety_margin_percent=self.config.min_filament_safety_margin_percent,
            message=message,
        )

    async def _monitor_loop(self):
        """Background monitoring loop for tangles."""
        while self._running:
            try:
                result = await self.check_tangle()
                if result and result.detected and result.severity == TangleDecision.CRITICAL:
                    logger.warning(f"Tangle detected: confidence={result.confidence:.2f}")
                    if self._on_tangle:
                        await self._on_tangle(result)

                await asyncio.sleep(self.config.check_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Filament monitor loop error: {e}")
                await asyncio.sleep(self.config.check_interval_seconds)


# =============================================================================
# Simple API Functions (Standalone)
# =============================================================================

# Default filament parameters (1.75mm PLA)
DEFAULT_FILAMENT_DIAMETER_MM = 1.75
DEFAULT_FILAMENT_DENSITY_G_CM3 = 1.24
DEFAULT_SPOOL_WIDTH_MM = 70.0


def capture_spool_image(camera_index: int = 0) -> np.ndarray | None:
    """
    Capture a single image from the secondary USB camera.

    Args:
        camera_index: USB camera index (default 0)

    Returns:
        OpenCV image array or None if capture failed
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.warning(f"Failed to open USB camera (index={camera_index})")
        return None

    try:
        ret, frame = cap.read()
        if not ret or frame is None:
            logger.warning("Failed to capture frame from USB camera")
            return None
        return frame
    finally:
        cap.release()


def estimate_remaining_filament(
    diameter_mm: float,
    filament_diameter_mm: float = DEFAULT_FILAMENT_DIAMETER_MM,
    spool_width_mm: float = DEFAULT_SPOOL_WIDTH_MM,
    filament_density_g_cm3: float = DEFAULT_FILAMENT_DENSITY_G_CM3,
) -> float:
    """
    Estimate remaining filament length from spool diameter.

    Args:
        diameter_mm: Current spool outer diameter in mm (measured)
        filament_diameter_mm: Filament diameter in mm (default 1.75)
        spool_width_mm: Width of the spool in mm (default 70)
        filament_density_g_cm3: Filament density in g/cm³ (default 1.24 for PLA)

    Returns:
        Estimated remaining filament length in meters
    """
    if diameter_mm <= 0:
        return 0.0

    # Spool hub diameter (typically ~55mm)
    HUB_DIAMETER_MM = 55.0

    if diameter_mm <= HUB_DIAMETER_MM:
        return 0.0

    # Calculate cross-sectional area of filament on spool
    spool_radius_outer = diameter_mm / 2.0
    spool_radius_inner = HUB_DIAMETER_MM / 2.0

    # Area of filament wound on spool (annulus)
    filament_area_mm2 = np.pi * (spool_radius_outer**2 - spool_radius_inner**2)

    # Convert to volume (account for packing efficiency ~0.7)
    packing_efficiency = 0.7
    filament_volume_mm3 = filament_area_mm2 * spool_width_mm * packing_efficiency

    # Convert to cm³
    filament_volume_cm3 = filament_volume_mm3 / 1000.0

    # Calculate filament cross-sectional area
    filament_cross_section_mm2 = np.pi * (filament_diameter_mm / 2.0) ** 2
    filament_cross_section_cm2 = filament_cross_section_mm2 / 100.0

    # Length = Volume / Cross-sectional area
    if filament_cross_section_cm2 <= 0:
        return 0.0

    length_cm = filament_volume_cm3 / filament_cross_section_cm2
    length_m = length_cm / 100.0

    return max(0.0, length_m)


def check_adequate_filament(
    remaining_diameter_mm: float,
    job_requirement_mm: float,
    filament_diameter_mm: float = DEFAULT_FILAMENT_DIAMETER_MM,
    safety_margin: float = 1.1,  # 10% safety margin
) -> bool:
    """
    Check if remaining filament is adequate for the job.

    Args:
        remaining_diameter_mm: Current spool diameter in mm
        job_requirement_mm: Required filament length in mm
        filament_diameter_mm: Filament diameter in mm
        safety_margin: Multiplier for safety margin (default 1.1 = 10%)

    Returns:
        True if filament is sufficient, False otherwise
    """
    remaining_m = estimate_remaining_filament(remaining_diameter_mm, filament_diameter_mm)
    remaining_mm = remaining_m * 1000.0

    required_with_margin = job_requirement_mm * safety_margin

    return remaining_mm >= required_with_margin


def detect_tangle(image: np.ndarray, threshold: float = 0.7) -> bool:
    """
    Detect crossed filament (tangle) using line intersection detection.

    Args:
        image: OpenCV image array (BGR format)
        threshold: Confidence threshold for tangle detection

    Returns:
        True if tangle detected, False otherwise
    """
    if image is None or image.size == 0:
        return False

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=30,
        maxLineGap=10,
    )

    if lines is None or len(lines) < 2:
        return False

    # Check for intersecting lines (indicates crossed filament)
    line_list = [(int(l[0][0]), int(l[0][1]), int(l[0][2]), int(l[0][3])) for l in lines]

    intersection_count = 0
    for i, line1 in enumerate(line_list):
        for line2 in line_list[i + 1 :]:
            if _line_segments_intersect(line1, line2):
                intersection_count += 1

    # Normalize by number of lines to get a score
    max_possible = len(line_list) * (len(line_list) - 1) / 2
    if max_possible == 0:
        return False

    tangle_score = intersection_count / max_possible

    return tangle_score > threshold


def _line_segments_intersect(
    line1: tuple[int, int, int, int], line2: tuple[int, int, int, int]
) -> bool:
    """
    Check if two line segments intersect using counter-clockwise test.

    Args:
        line1: (x1, y1, x2, y2) first line segment
        line2: (x3, y3, x4, y4) second line segment

    Returns:
        True if lines intersect
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    def ccw(ax: int, ay: int, bx: int, by: int, cx: int, cy: int) -> bool:
        return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)

    return ccw(x1, y1, x3, y3, x4, y4) != ccw(x2, y2, x3, y3, x4, y4) and ccw(
        x1, y1, x2, y2, x3, y3
    ) != ccw(x1, y1, x2, y2, x4, y4)


async def pause_on_tangle(
    printer_adapter, notification_callback: Callable[[str], Awaitable[None]] | None = None
) -> bool:
    """
    Pause print and notify when tangle is detected.

    Args:
        printer_adapter: Printer adapter with pause_print() method
        notification_callback: Optional async callback for notifications

    Returns:
        True if pause was successful, False otherwise
    """
    from sensing_voxel.core.models import CommandResultStatus

    try:
        result = await printer_adapter.pause_print()
        success = result.status == CommandResultStatus.SUCCESS

        if success:
            msg = "Print paused due to filament tangle detection"
            logger.warning(msg)
            if notification_callback:
                await notification_callback(msg)
        else:
            logger.error(f"Failed to pause print: {result.message}")

        return success
    except Exception as e:
        logger.error(f"Error pausing print on tangle: {e}")
        return False
