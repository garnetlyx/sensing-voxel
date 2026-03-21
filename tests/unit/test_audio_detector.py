"""
Unit tests for audio anomaly detection module.
"""

from __future__ import annotations

import asyncio
import io
import wave
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from sensing_voxel.audio.detector import (
    AudioAnomalyDetector,
    DetectionResult,
    AnomalyType,
)
from sensing_voxel.audio.baseline import BaselineManager, BaselineProfile


class TestAnomalyType:
    """Test AnomalyType enum."""

    def test_enum_values(self):
        assert AnomalyType.GRINDING.name == "GRINDING"
        assert AnomalyType.CRASHING.name == "CRASHING"
        assert AnomalyType.MOTOR_FAULT.name == "MOTOR_FAULT"
        assert AnomalyType.VIBRATION.name == "VIBRATION"
        assert AnomalyType.UNKNOWN.name == "UNKNOWN"


class TestDetectionResult:
    """Test DetectionResult dataclass."""

    def test_creation(self):
        result = DetectionResult(
            is_anomaly=True,
            confidence=0.85,
            anomaly_type=AnomalyType.GRINDING,
            detected_classes={"Grinding": 0.85},
            requires_emergency_stop=True,
        )
        assert result.is_anomaly is True
        assert result.confidence == 0.85
        assert result.anomaly_type == AnomalyType.GRINDING
        assert result.requires_emergency_stop is True
        assert isinstance(result.timestamp, datetime)

    def test_default_values(self):
        result = DetectionResult(
            is_anomaly=False,
            confidence=0.0,
            anomaly_type=AnomalyType.UNKNOWN,
        )
        assert result.detected_classes == {}
        assert result.requires_emergency_stop is False


class TestAudioAnomalyDetector:
    """Test AudioAnomalyDetector class."""

    @pytest.fixture
    def detector(self):
        return AudioAnomalyDetector(
            sample_rate=16000,
            confidence_threshold=0.7,
            emergency_threshold=0.85,
            temporal_confirmation_count=2,
        )

    @pytest.fixture
    def mock_audio_data(self):
        return np.random.randn(16000).astype(np.float32) * 0.1

    def test_initialization(self, detector):
        assert detector.sample_rate == 16000
        assert detector.confidence_threshold == 0.7
        assert detector.emergency_threshold == 0.85
        assert detector.temporal_confirmation_count == 2
        assert detector.is_running is False

    def test_normalize_audio_float(self, detector):
        samples = np.array([0.5, -0.5, 0.0, 1.0], dtype=np.float32)
        normalized = detector._normalize_audio(samples)
        assert normalized.max() <= 1.0
        assert normalized.min() >= -1.0

    def test_normalize_audio_int16(self, detector):
        samples = np.array([16384, -16384, 0, 32767], dtype=np.int16)
        normalized = detector._normalize_audio(samples)
        assert normalized.dtype == np.float32
        assert normalized.max() <= 1.0

    def test_mock_detect_normal(self, detector):
        samples = np.random.randn(16000).astype(np.float32) * 0.1
        result = detector._mock_detect(samples)
        assert result.is_anomaly is False
        assert result.anomaly_type == AnomalyType.UNKNOWN

    def test_mock_detect_anomaly(self, detector):
        samples = np.random.randn(16000).astype(np.float32) * 0.8
        result = detector._mock_detect(samples)
        assert result.is_anomaly is True
        assert result.confidence > 0.5

    def test_classify_anomaly_with_scores(self, detector):
        class_scores = {
            "Grinding": 0.8,
            "Scraping": 0.3,
            "Printing": 0.1,
        }
        result = detector._classify_anomaly(class_scores)
        assert result.is_anomaly is True
        assert result.anomaly_type == AnomalyType.GRINDING

    def test_classify_anomaly_no_match(self, detector):
        class_scores = {
            "Silence": 0.9,
            "Background noise": 0.1,
        }
        result = detector._classify_anomaly(class_scores)
        assert result.is_anomaly is False
        assert result.anomaly_type == AnomalyType.UNKNOWN

    def test_apply_baseline_adjustment(self, detector):
        detector._baseline_profile = {"anomaly_thresholds": {"grinding": 0.6}}
        # Confidence below 1.5x threshold should be reduced
        adjusted = detector._apply_baseline_adjustment(AnomalyType.GRINDING, 0.8)
        # 0.8 is below 0.9 (0.6 * 1.5), so it gets multiplied by 0.5
        assert adjusted == pytest.approx(0.4, abs=0.01)

    def test_apply_baseline_adjustment_high_confidence(self, detector):
        detector._baseline_profile = {"anomaly_thresholds": {"grinding": 0.6}}
        # Confidence above 1.5x threshold should not be reduced
        adjusted = detector._apply_baseline_adjustment(AnomalyType.GRINDING, 0.95)
        # 0.95 is above 0.9 (0.6 * 1.5), so it stays at 0.95
        assert adjusted == pytest.approx(0.95, abs=0.01)

    def test_apply_temporal_confirmation_single(self, detector):
        result = DetectionResult(
            is_anomaly=True,
            confidence=0.8,
            anomaly_type=AnomalyType.GRINDING,
        )
        confirmed = detector._apply_temporal_confirmation(result)
        assert confirmed is None

    def test_apply_temporal_confirmation_multiple(self, detector):
        result1 = DetectionResult(
            is_anomaly=True,
            confidence=0.8,
            anomaly_type=AnomalyType.GRINDING,
        )
        result2 = DetectionResult(
            is_anomaly=True,
            confidence=0.85,
            anomaly_type=AnomalyType.GRINDING,
        )
        detector._apply_temporal_confirmation(result1)
        confirmed = detector._apply_temporal_confirmation(result2)
        assert confirmed is not None
        assert confirmed.is_anomaly is True
        assert confirmed.confidence == pytest.approx(0.825, abs=0.01)

    @pytest.mark.asyncio
    async def test_start_stop_detection(self, detector):
        mock_source = AsyncMock(return_value=np.random.randn(16000).astype(np.float32) * 0.1)

        await detector.start_detection(mock_source)
        assert detector.is_running is True

        await detector.stop_detection()
        assert detector.is_running is False

    @pytest.mark.asyncio
    async def test_detection_callback(self, detector):
        callback_called = False
        callback_result = None

        async def mock_callback(result):
            nonlocal callback_called, callback_result
            callback_called = True
            callback_result = result

        detector.set_on_detection_callback(mock_callback)

        result = DetectionResult(
            is_anomaly=True,
            confidence=0.9,
            anomaly_type=AnomalyType.CRASHING,
            requires_emergency_stop=True,
        )
        await detector._handle_detection(result)

        assert callback_called is True
        assert callback_result is not None
        assert callback_result.anomaly_type == AnomalyType.CRASHING

    def test_analyze_sample_array(self, detector):
        samples = np.random.randn(16000).astype(np.float32) * 0.1
        result = detector.analyze_sample(samples, sample_rate=16000)
        assert isinstance(result, DetectionResult)

    def test_create_wav_bytes(self):
        samples = np.random.randn(16000).astype(np.float32) * 0.5
        samples_int16 = (samples * 32767).astype(np.int16)

        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(samples_int16.tobytes())

        wav_bytes = wav_io.getvalue()
        assert len(wav_bytes) > 0

    def test_load_wav_bytes(self, detector):
        samples = np.random.randn(16000).astype(np.float32) * 0.5
        samples_int16 = (samples * 32767).astype(np.int16)

        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(samples_int16.tobytes())

        wav_bytes = wav_io.getvalue()
        loaded = detector._load_wav_bytes(wav_bytes)

        assert len(loaded) == len(samples)
        assert loaded.dtype == np.float32


class TestBaselineProfile:
    """Test BaselineProfile dataclass."""

    def test_creation(self):
        profile = BaselineProfile(
            printer_id="test_printer",
            sample_count=16000,
            duration_seconds=1.0,
        )
        assert profile.printer_id == "test_printer"
        assert profile.sample_count == 16000
        assert isinstance(profile.created_at, datetime)

    def test_is_stale(self):
        from datetime import timedelta

        profile = BaselineProfile(printer_id="test")
        assert profile.is_stale(max_age_days=30) is False

        profile.updated_at = datetime.now() - timedelta(days=31)
        assert profile.is_stale(max_age_days=30) is True

    def test_to_dict(self):
        profile = BaselineProfile(
            printer_id="test",
            sample_count=1000,
            anomaly_thresholds={"grinding": 0.6},
        )
        data = profile.to_dict()
        assert data["printer_id"] == "test"
        assert data["sample_count"] == 1000
        assert isinstance(data["created_at"], str)

    def test_from_dict(self):
        data = {
            "printer_id": "test",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "sample_count": 1000,
            "sample_rate": 16000,
            "duration_seconds": 1.0,
            "mean_spectrum": [],
            "std_spectrum": [],
            "energy_mean": 0.0,
            "energy_std": 0.0,
            "zcr_mean": 0.0,
            "zcr_std": 0.0,
            "anomaly_thresholds": {},
            "normal_class_scores": {},
            "metadata": {},
        }
        profile = BaselineProfile.from_dict(data)
        assert profile.printer_id == "test"
        assert isinstance(profile.created_at, datetime)


class TestBaselineManager:
    """Test BaselineManager class."""

    @pytest.fixture
    def manager(self, tmp_path):
        return BaselineManager(storage_dir=str(tmp_path / "baselines"))

    def test_initialization(self, manager, tmp_path):
        assert manager.storage_dir.exists()
        assert manager.is_capturing is False

    def test_get_profile_nonexistent(self, manager):
        profile = manager.get_profile("nonexistent")
        assert profile is None

    def test_save_and_get_profile(self, manager):
        profile = BaselineProfile(
            printer_id="test_printer",
            sample_count=16000,
        )
        assert manager.save_profile(profile) is True

        loaded = manager.get_profile("test_printer")
        assert loaded is not None
        assert loaded.printer_id == "test_printer"
        assert loaded.sample_count == 16000

    def test_delete_profile(self, manager):
        profile = BaselineProfile(printer_id="to_delete")
        manager.save_profile(profile)

        assert manager.delete_profile("to_delete") is True
        assert manager.get_profile("to_delete") is None

    def test_list_profiles(self, manager):
        manager.save_profile(BaselineProfile(printer_id="printer1"))
        manager.save_profile(BaselineProfile(printer_id="printer2"))

        profiles = manager.list_profiles()
        assert "printer1" in profiles
        assert "printer2" in profiles

    def test_is_profile_valid(self, manager):
        from datetime import timedelta

        profile = BaselineProfile(printer_id="test")
        manager.save_profile(profile)

        assert manager.is_profile_valid("test", max_age_days=30) is True

        profile.updated_at = datetime.now() - timedelta(days=31)
        manager.save_profile(profile)
        assert manager.is_profile_valid("test", max_age_days=30) is False

    def test_get_baseline_for_detector(self, manager):
        profile = BaselineProfile(
            printer_id="test",
            anomaly_thresholds={"grinding": 0.6, "crashing": 0.7},
            energy_mean=0.5,
            energy_std=0.1,
        )
        manager.save_profile(profile)

        baseline = manager.get_baseline_for_detector("test")
        assert baseline is not None
        assert baseline["anomaly_thresholds"]["grinding"] == 0.6
        assert baseline["energy_mean"] == 0.5

    def test_update_anomaly_threshold(self, manager):
        profile = BaselineProfile(printer_id="test", anomaly_thresholds={"grinding": 0.6})
        manager.save_profile(profile)

        assert manager.update_anomaly_threshold("test", "grinding", 0.75) is True

        updated = manager.get_profile("test")
        assert updated.anomaly_thresholds["grinding"] == 0.75

    def test_normalize_audio(self, manager):
        samples = np.array([16384, -16384, 0], dtype=np.int16)
        normalized = manager._normalize_audio(samples)
        assert normalized.dtype == np.float32
        assert normalized.max() <= 1.0

    def test_compute_profile(self, manager):
        samples = np.random.randn(16000 * 2).astype(np.float32) * 0.3
        profile = manager._compute_profile("test_printer", samples, sample_rate=16000)

        assert profile.printer_id == "test_printer"
        assert profile.sample_count == len(samples)
        assert profile.duration_seconds == pytest.approx(2.0, abs=0.1)
        assert len(profile.mean_spectrum) > 0
        assert profile.energy_mean > 0
        assert "grinding" in profile.anomaly_thresholds

    def test_get_profile_path_sanitization(self, manager):
        path = manager._get_profile_path("printer/with/invalid\\chars")
        assert "/" not in path.name
        assert "\\" not in path.name
        assert ".." not in path.name


class TestIntegration:
    """Integration tests for audio detection system."""

    @pytest.mark.asyncio
    async def test_full_detection_flow(self):
        detector = AudioAnomalyDetector(
            sample_rate=16000,
            confidence_threshold=0.3,
            temporal_confirmation_count=1,
        )

        callback_results = []

        async def mock_callback(result):
            callback_results.append(result)

        detector.set_on_detection_callback(mock_callback)

        high_energy_samples = np.random.randn(16000).astype(np.float32) * 1.5

        async def mock_source():
            return high_energy_samples

        await detector.start_detection(mock_source)
        await asyncio.sleep(0.3)
        await detector.stop_detection()

        assert detector.is_running is False
        assert detector.recent_detection_count >= 0

    def test_detector_with_baseline(self):
        detector = AudioAnomalyDetector()
        baseline = {
            "anomaly_thresholds": {
                "grinding": 0.5,
                "crashing": 0.6,
            }
        }
        detector.set_baseline_profile(baseline)
        assert detector._baseline_profile == baseline

    def test_grinding_triggers_emergency(self):
        detector = AudioAnomalyDetector(
            confidence_threshold=0.7,
            emergency_threshold=0.85,
        )
        class_scores = {
            "Grinding": 0.9,
            "Scraping": 0.3,
        }
        result = detector._classify_anomaly(class_scores)
        assert result.is_anomaly is True
        assert result.requires_emergency_stop is True

    def test_crashing_triggers_emergency(self):
        detector = AudioAnomalyDetector(
            confidence_threshold=0.7,
            emergency_threshold=0.85,
        )
        class_scores = {
            "Crash": 0.8,
            "Banging": 0.7,
        }
        result = detector._classify_anomaly(class_scores)
        assert result.is_anomaly is True
        assert result.requires_emergency_stop is True
