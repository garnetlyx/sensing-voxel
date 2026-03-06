"""
Camera calibration via checkerboard detection.

Computes homography matrix mapping camera pixels to bed coordinates.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from sensing_voxel.core.models import (
    CalibrationStatus,
    CameraCalibration,
    HomographyMatrix,
    VisionFrame,
)

logger = logging.getLogger(__name__)


class CameraCalibrator:
    """Checkerboard-based camera calibration for bed coordinate mapping."""

    def __init__(
        self,
        checkerboard_size: tuple[int, int] = (9, 6),
        square_size_mm: float = 25.0,
        calibration_expiry_days: int = 30,
    ):
        self.checkerboard_size = checkerboard_size
        self.square_size_mm = square_size_mm
        self.calibration_expiry_days = calibration_expiry_days

    def calibrate(
        self,
        frame: VisionFrame,
        camera_id: str | None = None,
    ) -> CameraCalibration | None:
        """
        Perform calibration from a single frame with checkerboard.

        Args:
            frame: Camera frame containing checkerboard pattern
            camera_id: Optional camera identifier

        Returns:
            CameraCalibration if checkerboard found, None otherwise
        """
        gray = cv2.cvtColor(frame.frame_data, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(
            gray, self.checkerboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if not found:
            logger.warning("Checkerboard not found in frame")
            return None

        # Refine corner positions
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Build real-world points (on bed surface, Z=0)
        rows, cols = self.checkerboard_size
        obj_points = np.zeros((rows * cols, 2), dtype=np.float32)
        for i in range(rows):
            for j in range(cols):
                obj_points[i * cols + j] = [
                    j * self.square_size_mm,
                    i * self.square_size_mm,
                ]

        img_points = corners.reshape(-1, 2).astype(np.float32)

        # Compute homography: image → bed coordinates
        H, mask = cv2.findHomography(img_points, obj_points, cv2.RANSAC, 5.0)

        if H is None:
            logger.error("Homography computation failed")
            return None

        # Calculate mm_per_pixel from homography
        # Use top-left pair of points
        dx_px = np.linalg.norm(img_points[1] - img_points[0])
        mm_per_pixel = self.square_size_mm / dx_px if dx_px > 0 else 1.0

        # Compute reprojection error
        projected = cv2.perspectiveTransform(
            img_points.reshape(-1, 1, 2), H
        ).reshape(-1, 2)
        error = float(np.mean(np.linalg.norm(projected - obj_points, axis=1)))

        now = datetime.now()
        valid_until = now + timedelta(days=self.calibration_expiry_days)

        homography = HomographyMatrix(
            matrix=H.tolist(),
            mm_per_pixel=mm_per_pixel,
            camera_id=camera_id,
            calibration_timestamp=now,
            valid_until=valid_until,
        )

        roi_points = [(float(c[0]), float(c[1])) for c in corners.reshape(-1, 2)]

        calibration = CameraCalibration(
            homography=homography,
            roi_points=roi_points,
            error_mm=error,
        )

        logger.info(
            f"Calibration complete: mm_per_pixel={mm_per_pixel:.4f}, "
            f"reprojection_error={error:.3f}mm"
        )
        return calibration

    @staticmethod
    def check_validity(calibration: CameraCalibration | None) -> CalibrationStatus:
        """Check if calibration is valid."""
        if calibration is None:
            return CalibrationStatus.MISSING

        h = calibration.homography
        if h.matrix is None or len(h.matrix) != 3:
            return CalibrationStatus.INVALID

        if not h.is_valid():
            return CalibrationStatus.EXPIRED

        return CalibrationStatus.VALID

    @staticmethod
    def save_calibration(calibration: CameraCalibration, output_dir: Path):
        """Save calibration to JSON file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        h = calibration.homography
        camera_id = h.camera_id or "default"

        data = {
            "camera_id": camera_id,
            "matrix": h.matrix,
            "mm_per_pixel": h.mm_per_pixel,
            "calibration_timestamp": h.calibration_timestamp.isoformat(),
            "valid_until": h.valid_until.isoformat() if h.valid_until else None,
            "error_mm": calibration.error_mm,
            "roi_points": calibration.roi_points,
        }

        path = output_dir / f"calibration_{camera_id}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Calibration saved to {path}")

    @staticmethod
    def load_calibration(
        calibration_dir: Path, camera_id: str = "default"
    ) -> CameraCalibration | None:
        """Load calibration from JSON file."""
        path = calibration_dir / f"calibration_{camera_id}.json"
        if not path.exists():
            return None

        try:
            with open(path) as f:
                data = json.load(f)

            homography = HomographyMatrix(
                matrix=data["matrix"],
                mm_per_pixel=data["mm_per_pixel"],
                camera_id=data.get("camera_id"),
                calibration_timestamp=datetime.fromisoformat(
                    data["calibration_timestamp"]
                ),
                valid_until=(
                    datetime.fromisoformat(data["valid_until"])
                    if data.get("valid_until")
                    else None
                ),
            )

            return CameraCalibration(
                homography=homography,
                roi_points=data.get("roi_points"),
                error_mm=data.get("error_mm"),
            )
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return None
