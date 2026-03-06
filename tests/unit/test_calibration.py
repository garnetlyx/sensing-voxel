"""
Unit tests for camera calibration.
"""

import pytest
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from sensing_voxel.core.models import (
    CalibrationStatus,
    CameraCalibration,
    HomographyMatrix,
)
from sensing_voxel.vision.calibration import CameraCalibrator


class TestCameraCalibrator:

    @pytest.fixture
    def calibrator(self):
        return CameraCalibrator(
            checkerboard_size=(9, 6),
            square_size_mm=25.0,
            calibration_expiry_days=30,
        )

    @pytest.fixture
    def valid_calibration(self):
        return CameraCalibration(
            homography=HomographyMatrix(
                matrix=np.eye(3).tolist(),
                mm_per_pixel=0.5,
                camera_id="test_cam",
                calibration_timestamp=datetime.now(),
                valid_until=datetime.now() + timedelta(days=30),
            ),
            error_mm=0.1,
        )

    @pytest.fixture
    def expired_calibration(self):
        return CameraCalibration(
            homography=HomographyMatrix(
                matrix=np.eye(3).tolist(),
                mm_per_pixel=0.5,
                camera_id="test_cam",
                calibration_timestamp=datetime.now() - timedelta(days=60),
                valid_until=datetime.now() - timedelta(days=30),
            ),
            error_mm=0.1,
        )

    def test_check_validity_valid(self, valid_calibration):
        status = CameraCalibrator.check_validity(valid_calibration)
        assert status == CalibrationStatus.VALID

    def test_check_validity_expired(self, expired_calibration):
        status = CameraCalibrator.check_validity(expired_calibration)
        assert status == CalibrationStatus.EXPIRED

    def test_check_validity_missing(self):
        status = CameraCalibrator.check_validity(None)
        assert status == CalibrationStatus.MISSING

    def test_check_validity_invalid_matrix(self):
        cal = CameraCalibration(
            homography=HomographyMatrix(
                matrix=[[1, 0]],  # Wrong size
                mm_per_pixel=0.5,
            ),
            error_mm=0.1,
        )
        status = CameraCalibrator.check_validity(cal)
        assert status == CalibrationStatus.INVALID

    def test_save_and_load_calibration(self, valid_calibration, tmp_path):
        CameraCalibrator.save_calibration(valid_calibration, tmp_path)

        loaded = CameraCalibrator.load_calibration(tmp_path, "test_cam")
        assert loaded is not None
        assert loaded.homography.mm_per_pixel == 0.5
        assert loaded.homography.camera_id == "test_cam"
        assert loaded.error_mm == 0.1

    def test_load_calibration_nonexistent(self, tmp_path):
        loaded = CameraCalibrator.load_calibration(tmp_path, "missing")
        assert loaded is None

    def test_save_creates_directory(self, valid_calibration, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c"
        CameraCalibrator.save_calibration(valid_calibration, deep_path)
        assert deep_path.exists()

    def test_load_calibration_preserves_timestamp(self, valid_calibration, tmp_path):
        CameraCalibrator.save_calibration(valid_calibration, tmp_path)
        loaded = CameraCalibrator.load_calibration(tmp_path, "test_cam")
        assert loaded is not None
        assert loaded.homography.valid_until is not None
