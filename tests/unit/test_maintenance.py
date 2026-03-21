"""
Unit tests for predictive maintenance module

Tests SHAPER_CALIBRATE tracking, drift detection, diagnosis generation,
and proactive calibration triggers.
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path

from sensing_voxel.monitoring.maintenance import (
    MaintenanceAlertLevel,
    MaintenanceDiagnosis,
    MaintenanceStatus,
    MaintenanceTracker,
    PredictiveMaintenanceConfig,
    ShaperCalibrationResult,
    ShaperType,
    DriftAnalysis,
)
from sensing_voxel.storage import (
    Database,
    DatabaseConfig,
    CalibrationType,
)


@pytest.fixture
def temp_db_path(tmp_path):
    return tmp_path / "maintenance_test.db"


@pytest.fixture
def database(temp_db_path):
    config = DatabaseConfig(db_path=temp_db_path)
    db = Database(config)
    db.initialize()
    return db


@pytest.fixture
def tracker(database):
    config = PredictiveMaintenanceConfig(
        calibration_interval_days=30,
        calibration_max_age_days=60,
        drift_warning_threshold=10.0,
        drift_critical_threshold=20.0,
        proactive_trigger_threshold=15.0,
    )
    return MaintenanceTracker(
        database=database,
        printer_id="test_printer",
        config=config,
    )


@pytest.fixture
def sample_shaper_result():
    return ShaperCalibrationResult(
        x_resonance=50.0,
        y_resonance=45.0,
        x_shaper=ShaperType.ZV,
        y_shaper=ShaperType.ZV,
        x_vibration_reduction=0.85,
        y_vibration_reduction=0.82,
        x_smoothing=0.05,
        y_smoothing=0.06,
    )


class TestShaperCalibrationResult:
    def test_to_dict_conversion(self, sample_shaper_result):
        data = sample_shaper_result.to_dict()

        assert data["x_resonance"] == 50.0
        assert data["y_resonance"] == 45.0
        assert data["x_shaper"] == "zv"
        assert data["y_shaper"] == "zv"
        assert data["x_vibration_reduction"] == 0.85
        assert data["raw_data"] == {}

    def test_from_dict_conversion(self):
        data = {
            "x_resonance": 52.0,
            "y_resonance": 48.0,
            "x_shaper": "ei",
            "y_shaper": "mzv",
            "x_vibration_reduction": 0.90,
            "y_vibration_reduction": 0.88,
            "x_smoothing": 0.03,
            "y_smoothing": 0.04,
            "raw_data": {"extra": "info"},
        }

        result = ShaperCalibrationResult.from_dict(data)

        assert result.x_resonance == 52.0
        assert result.y_resonance == 48.0
        assert result.x_shaper == ShaperType.EI
        assert result.y_shaper == ShaperType.MZV
        assert result.raw_data == {"extra": "info"}

    def test_from_dict_with_unknown_shaper(self):
        data = {
            "x_resonance": 50.0,
            "y_resonance": 45.0,
            "x_shaper": "unknown",
            "y_shaper": "unknown",
        }

        result = ShaperCalibrationResult.from_dict(data)
        assert result.x_shaper == ShaperType.UNKNOWN
        assert result.y_shaper == ShaperType.UNKNOWN


class TestDriftAnalysis:
    def test_critical_severity(self):
        analysis = DriftAnalysis(
            axis="X",
            frequency_drift_hz=10.0,
            frequency_drift_percent=25.0,
            shaper_changed=False,
        )
        assert analysis.severity == MaintenanceAlertLevel.CRITICAL

    def test_warning_severity(self):
        analysis = DriftAnalysis(
            axis="X",
            frequency_drift_hz=5.0,
            frequency_drift_percent=15.0,
            shaper_changed=False,
        )
        assert analysis.severity == MaintenanceAlertLevel.WARNING

    def test_info_severity(self):
        analysis = DriftAnalysis(
            axis="X",
            frequency_drift_hz=2.0,
            frequency_drift_percent=5.0,
            shaper_changed=False,
        )
        assert analysis.severity == MaintenanceAlertLevel.INFO

    def test_diagnosis_without_shaper_change(self):
        analysis = DriftAnalysis(
            axis="X",
            frequency_drift_hz=5.0,
            frequency_drift_percent=10.0,
            shaper_changed=False,
            time_since_last=timedelta(days=30),
        )
        diagnosis = analysis.generate_diagnosis()

        assert "X-axis resonance shifted 10.0%" in diagnosis
        assert "5.0 Hz" in diagnosis
        assert "since last month" in diagnosis

    def test_diagnosis_with_shaper_change(self):
        analysis = DriftAnalysis(
            axis="Y",
            frequency_drift_hz=8.0,
            frequency_drift_percent=18.0,
            shaper_changed=True,
            previous_shaper=ShaperType.ZV,
            current_shaper=ShaperType.EI,
            time_since_last=timedelta(days=14),
        )
        diagnosis = analysis.generate_diagnosis()

        assert "Y-axis shaper changed from ZV to EI" in diagnosis
        assert "over 2 weeks" in diagnosis

    def test_diagnosis_with_days_timeframe(self):
        analysis = DriftAnalysis(
            axis="X",
            frequency_drift_hz=3.0,
            frequency_drift_percent=6.0,
            shaper_changed=False,
            time_since_last=timedelta(days=5),
        )
        diagnosis = analysis.generate_diagnosis()

        assert "over 5 days" in diagnosis


class TestPredictiveMaintenanceConfig:
    def test_default_values(self):
        config = PredictiveMaintenanceConfig()

        assert config.drift_warning_threshold == 10.0
        assert config.drift_critical_threshold == 20.0
        assert config.calibration_interval_days == 30
        assert config.calibration_max_age_days == 60
        assert config.enable_proactive_trigger is True
        assert config.proactive_trigger_threshold == 15.0

    def test_to_dict(self):
        config = PredictiveMaintenanceConfig(
            drift_warning_threshold=8.0,
            enable_proactive_trigger=False,
        )
        data = config.to_dict()

        assert data["drift_warning_threshold"] == 8.0
        assert data["enable_proactive_trigger"] is False

    def test_from_dict(self):
        data = {
            "drift_warning_threshold": 12.0,
            "calibration_interval_days": 45,
            "enable_proactive_trigger": True,
        }
        config = PredictiveMaintenanceConfig.from_dict(data)

        assert config.drift_warning_threshold == 12.0
        assert config.calibration_interval_days == 45
        assert config.drift_critical_threshold == 20.0  # Default


class TestMaintenanceTracker:
    def test_record_shaper_calibration(self, tracker, sample_shaper_result):
        tracker.record_shaper_calibration(
            calibration_id="cal_001",
            result=sample_shaper_result,
            notes="Test calibration",
        )

        history = tracker.get_calibration_history()
        assert len(history) == 1
        assert history[0][1].x_resonance == 50.0
        assert history[0][1].y_resonance == 45.0

    def test_record_multiple_calibrations(self, tracker, sample_shaper_result):
        for i in range(3):
            result = ShaperCalibrationResult(
                x_resonance=50.0 + i,
                y_resonance=45.0 + i,
                x_shaper=ShaperType.ZV,
                y_shaper=ShaperType.ZV,
            )
            tracker.record_shaper_calibration(
                calibration_id=f"cal_{i}",
                result=result,
            )

        history = tracker.get_calibration_history()
        assert len(history) == 3

    def test_get_calibration_history_sorted(self, tracker, sample_shaper_result):
        old_time = datetime.now() - timedelta(days=30)
        recent_time = datetime.now() - timedelta(days=1)

        from sensing_voxel.storage.calibration_repository import CalibrationRepository

        repo = CalibrationRepository(tracker.db)

        # Create old calibration directly with specific timestamp
        result1 = ShaperCalibrationResult(
            x_resonance=50.0,
            y_resonance=45.0,
            x_shaper=ShaperType.ZV,
            y_shaper=ShaperType.ZV,
        )
        repo.create(
            calibration_id="cal_old",
            printer_id="test_printer",
            calibration_type=CalibrationType.SHAPER,
            values=result1.to_dict(),
            performed_at=old_time,
        )

        # Create recent calibration
        result2 = ShaperCalibrationResult(
            x_resonance=52.0,
            y_resonance=47.0,
            x_shaper=ShaperType.ZV,
            y_shaper=ShaperType.ZV,
        )
        repo.create(
            calibration_id="cal_recent",
            printer_id="test_printer",
            calibration_type=CalibrationType.SHAPER,
            values=result2.to_dict(),
            performed_at=recent_time,
        )

        history = tracker.get_calibration_history()
        assert len(history) == 2
        # Should be sorted oldest first
        assert history[0][1].x_resonance == 50.0
        assert history[1][1].x_resonance == 52.0

    def test_get_empty_calibration_history(self, tracker):
        history = tracker.get_calibration_history()
        assert history == []


class TestDriftDetection:
    def test_analyze_drift_with_sufficient_history(self, tracker):
        # Record initial calibration
        result1 = ShaperCalibrationResult(
            x_resonance=50.0,
            y_resonance=45.0,
            x_shaper=ShaperType.ZV,
            y_shaper=ShaperType.ZV,
        )
        tracker.record_shaper_calibration("cal_001", result1)

        # Record later calibration with drift
        result2 = ShaperCalibrationResult(
            x_resonance=55.0,  # 10% drift
            y_resonance=49.5,  # 10% drift
            x_shaper=ShaperType.ZV,
            y_shaper=ShaperType.MZV,  # Shaper changed
        )
        tracker.record_shaper_calibration("cal_002", result2)

        analyses = tracker.analyze_drift()

        assert len(analyses) == 2  # X and Y axes

        x_analysis = next(a for a in analyses if a.axis == "X")
        assert x_analysis.frequency_drift_hz == 5.0
        assert x_analysis.frequency_drift_percent == 10.0
        assert not x_analysis.shaper_changed

        y_analysis = next(a for a in analyses if a.axis == "Y")
        assert y_analysis.frequency_drift_percent == 10.0
        assert y_analysis.shaper_changed
        assert y_analysis.previous_shaper == ShaperType.ZV
        assert y_analysis.current_shaper == ShaperType.MZV

    def test_analyze_drift_with_insufficient_history(self, tracker):
        result = ShaperCalibrationResult(
            x_resonance=50.0,
            y_resonance=45.0,
            x_shaper=ShaperType.ZV,
            y_shaper=ShaperType.ZV,
        )
        tracker.record_shaper_calibration("cal_001", result)

        analyses = tracker.analyze_drift()
        assert analyses == []

    def test_analyze_drift_critical_threshold(self, tracker):
        result1 = ShaperCalibrationResult(
            x_resonance=50.0,
            y_resonance=45.0,
            x_shaper=ShaperType.ZV,
            y_shaper=ShaperType.ZV,
        )
        tracker.record_shaper_calibration("cal_001", result1)

        result2 = ShaperCalibrationResult(
            x_resonance=65.0,  # 30% drift - critical
            y_resonance=45.0,
            x_shaper=ShaperType.ZV,
            y_shaper=ShaperType.ZV,
        )
        tracker.record_shaper_calibration("cal_002", result2)

        analyses = tracker.analyze_drift()
        x_analysis = next(a for a in analyses if a.axis == "X")
        assert x_analysis.severity == MaintenanceAlertLevel.CRITICAL


class TestTrendAnalysis:
    def test_analyze_trend_with_multiple_calibrations(self, tracker):
        for i in range(5):
            result = ShaperCalibrationResult(
                x_resonance=50.0 + i * 2,  # Increasing trend
                y_resonance=45.0,
                x_shaper=ShaperType.ZV,
                y_shaper=ShaperType.ZV,
            )
            tracker.record_shaper_calibration(f"cal_{i}", result)

        trends = tracker.analyze_trend()

        assert "X" in trends
        assert "Y" in trends

        x_stats = trends["X"]
        assert x_stats["mean"] == 54.0  # Average of 50, 52, 54, 56, 58
        assert x_stats["min"] == 50.0
        assert x_stats["max"] == 58.0
        assert x_stats["range"] == 8.0
        assert x_stats["trend_slope"] > 0  # Positive trend

    def test_analyze_trend_insufficient_data(self, tracker):
        trends = tracker.analyze_trend()
        assert trends == {}

        result = ShaperCalibrationResult(
            x_resonance=50.0,
            y_resonance=45.0,
            x_shaper=ShaperType.ZV,
            y_shaper=ShaperType.ZV,
        )
        tracker.record_shaper_calibration("cal_001", result)

        trends = tracker.analyze_trend()
        assert trends == {}


class TestDiagnosisGeneration:
    def test_healthy_diagnosis(self, tracker):
        result = ShaperCalibrationResult(
            x_resonance=50.0,
            y_resonance=45.0,
            x_shaper=ShaperType.ZV,
            y_shaper=ShaperType.ZV,
        )
        tracker.record_shaper_calibration("cal_001", result)

        diagnosis = tracker.generate_diagnosis()

        assert diagnosis.status == MaintenanceStatus.HEALTHY
        assert diagnosis.printer_id == "test_printer"
        assert not diagnosis.calibration_overdue

    def test_calibration_overdue_diagnosis(self, tracker):
        from sensing_voxel.storage.calibration_repository import CalibrationRepository

        repo = CalibrationRepository(tracker.db)

        old_time = datetime.now() - timedelta(days=70)  # Beyond max_age_days (60)

        result = ShaperCalibrationResult(
            x_resonance=50.0,
            y_resonance=45.0,
            x_shaper=ShaperType.ZV,
            y_shaper=ShaperType.ZV,
        )
        repo.create(
            calibration_id="cal_old",
            printer_id="test_printer",
            calibration_type=CalibrationType.SHAPER,
            values=result.to_dict(),
            performed_at=old_time,
        )

        diagnosis = tracker.generate_diagnosis()

        assert diagnosis.status == MaintenanceStatus.NEEDS_ATTENTION
        assert diagnosis.calibration_overdue
        assert diagnosis.days_since_calibration >= 70
        assert any("Calibration overdue" in r for r in diagnosis.recommendations)

    def test_critical_drift_diagnosis(self, tracker):
        result1 = ShaperCalibrationResult(
            x_resonance=50.0,
            y_resonance=45.0,
            x_shaper=ShaperType.ZV,
            y_shaper=ShaperType.ZV,
        )
        tracker.record_shaper_calibration("cal_001", result1)

        result2 = ShaperCalibrationResult(
            x_resonance=70.0,  # 40% drift - critical
            y_resonance=45.0,
            x_shaper=ShaperType.ZV,
            y_shaper=ShaperType.ZV,
        )
        tracker.record_shaper_calibration("cal_002", result2)

        diagnosis = tracker.generate_diagnosis()

        assert diagnosis.status == MaintenanceStatus.CRITICAL
        assert len(diagnosis.critical_issues) == 1
        assert any("CRITICAL" in r for r in diagnosis.recommendations)

    def test_degraded_status_with_minor_drift(self, tracker):
        result1 = ShaperCalibrationResult(
            x_resonance=50.0,
            y_resonance=45.0,
            x_shaper=ShaperType.ZV,
            y_shaper=ShaperType.ZV,
        )
        tracker.record_shaper_calibration("cal_001", result1)

        result2 = ShaperCalibrationResult(
            x_resonance=52.5,  # 5% drift
            y_resonance=45.0,
            x_shaper=ShaperType.ZV,
            y_shaper=ShaperType.ZV,
        )
        tracker.record_shaper_calibration("cal_002", result2)

        diagnosis = tracker.generate_diagnosis()

        assert diagnosis.status == MaintenanceStatus.DEGRADED

    def test_diagnosis_summary(self, tracker):
        result = ShaperCalibrationResult(
            x_resonance=50.0,
            y_resonance=45.0,
            x_shaper=ShaperType.ZV,
            y_shaper=ShaperType.ZV,
        )
        tracker.record_shaper_calibration("cal_001", result)

        diagnosis = tracker.generate_diagnosis()

        assert "operating normally" in diagnosis.summary

    def test_high_variability_recommendation(self, tracker):
        # Create calibrations with high variability
        frequencies = [50.0, 65.0, 45.0, 70.0, 52.0]  # High variance (CV ~18%)
        for i, freq in enumerate(frequencies):
            result = ShaperCalibrationResult(
                x_resonance=freq,
                y_resonance=45.0,
                x_shaper=ShaperType.ZV,
                y_shaper=ShaperType.ZV,
            )
            tracker.record_shaper_calibration(f"cal_{i}", result)

        diagnosis = tracker.generate_diagnosis()

        assert any("high variability" in r.lower() for r in diagnosis.recommendations)


class TestProactiveCalibrationTrigger:
    def test_trigger_when_interval_exceeded(self, tracker):
        from sensing_voxel.storage.calibration_repository import CalibrationRepository

        repo = CalibrationRepository(tracker.db)

        old_time = datetime.now() - timedelta(days=65)  # Beyond max_age_days

        result = ShaperCalibrationResult(
            x_resonance=50.0,
            y_resonance=45.0,
            x_shaper=ShaperType.ZV,
            y_shaper=ShaperType.ZV,
        )
        repo.create(
            calibration_id="cal_old",
            printer_id="test_printer",
            calibration_type=CalibrationType.SHAPER,
            values=result.to_dict(),
            performed_at=old_time,
        )

        should_trigger, reason = tracker.should_trigger_proactive_calibration()

        assert should_trigger is True
        assert reason is not None
        assert "interval exceeded" in reason.lower()

    def test_trigger_when_drift_exceeds_threshold(self, tracker):
        result1 = ShaperCalibrationResult(
            x_resonance=50.0,
            y_resonance=45.0,
            x_shaper=ShaperType.ZV,
            y_shaper=ShaperType.ZV,
        )
        tracker.record_shaper_calibration("cal_001", result1)

        result2 = ShaperCalibrationResult(
            x_resonance=60.0,  # 20% drift, exceeds 15% threshold
            y_resonance=45.0,
            x_shaper=ShaperType.ZV,
            y_shaper=ShaperType.ZV,
        )
        tracker.record_shaper_calibration("cal_002", result2)

        should_trigger, reason = tracker.should_trigger_proactive_calibration()

        assert should_trigger is True
        assert reason is not None
        assert "X-axis" in reason
        assert "20.0%" in reason or "exceeds" in reason.lower()

    def test_no_trigger_when_healthy(self, tracker):
        result1 = ShaperCalibrationResult(
            x_resonance=50.0,
            y_resonance=45.0,
            x_shaper=ShaperType.ZV,
            y_shaper=ShaperType.ZV,
        )
        tracker.record_shaper_calibration("cal_001", result1)

        result2 = ShaperCalibrationResult(
            x_resonance=51.0,  # 2% drift, below threshold
            y_resonance=45.0,
            x_shaper=ShaperType.ZV,
            y_shaper=ShaperType.ZV,
        )
        tracker.record_shaper_calibration("cal_002", result2)

        should_trigger, reason = tracker.should_trigger_proactive_calibration()

        assert should_trigger is False
        assert reason is None

    def test_no_trigger_when_disabled(self, tracker):
        tracker.config.enable_proactive_trigger = False

        result1 = ShaperCalibrationResult(
            x_resonance=50.0,
            y_resonance=45.0,
            x_shaper=ShaperType.ZV,
            y_shaper=ShaperType.ZV,
        )
        tracker.record_shaper_calibration("cal_001", result1)

        result2 = ShaperCalibrationResult(
            x_resonance=70.0,  # 40% drift
            y_resonance=45.0,
            x_shaper=ShaperType.ZV,
            y_shaper=ShaperType.ZV,
        )
        tracker.record_shaper_calibration("cal_002", result2)

        should_trigger, _ = tracker.should_trigger_proactive_calibration()

        assert should_trigger is False

    def test_no_trigger_when_no_history(self, tracker):
        should_trigger, reason = tracker.should_trigger_proactive_calibration()

        assert should_trigger is True  # No calibration history = overdue
        assert reason is not None


class TestGetLastCalibration:
    def test_get_last_calibration_with_history(self, tracker, sample_shaper_result):
        tracker.record_shaper_calibration("cal_001", sample_shaper_result)

        last = tracker.get_last_calibration()

        assert last is not None
        timestamp, result = last
        assert isinstance(timestamp, datetime)
        assert result.x_resonance == 50.0

    def test_get_last_calibration_empty(self, tracker):
        last = tracker.get_last_calibration()
        assert last is None


class TestGetStats:
    def test_get_stats(self, tracker, sample_shaper_result):
        tracker.record_shaper_calibration("cal_001", sample_shaper_result)

        stats = tracker.get_stats()

        assert stats["printer_id"] == "test_printer"
        assert stats["total_calibrations"] == 1
        assert stats["current_status"] == "HEALTHY"
        assert stats["active_alerts"] == 0

    def test_get_stats_with_alerts(self, tracker):
        result1 = ShaperCalibrationResult(
            x_resonance=50.0,
            y_resonance=45.0,
            x_shaper=ShaperType.ZV,
            y_shaper=ShaperType.ZV,
        )
        tracker.record_shaper_calibration("cal_001", result1)

        result2 = ShaperCalibrationResult(
            x_resonance=62.0,  # 24% drift - critical
            y_resonance=45.0,
            x_shaper=ShaperType.ZV,
            y_shaper=ShaperType.ZV,
        )
        tracker.record_shaper_calibration("cal_002", result2)

        stats = tracker.get_stats()

        assert stats["current_status"] == "CRITICAL"
        assert stats["critical_issues"] == 1


class TestShaperTypeEnum:
    def test_shaper_type_values(self):
        assert ShaperType.ZV.value == "zv"
        assert ShaperType.MZV.value == "mzv"
        assert ShaperType.EI.value == "ei"
        assert ShaperType.UNKNOWN.value == "unknown"


class TestMaintenanceStatusEnum:
    def test_status_values(self):
        assert MaintenanceStatus.HEALTHY.name == "HEALTHY"
        assert MaintenanceStatus.DEGRADED.name == "DEGRADED"
        assert MaintenanceStatus.NEEDS_ATTENTION.name == "NEEDS_ATTENTION"
        assert MaintenanceStatus.CRITICAL.name == "CRITICAL"


class TestMaintenanceAlertLevelEnum:
    def test_alert_levels(self):
        assert MaintenanceAlertLevel.INFO.name == "INFO"
        assert MaintenanceAlertLevel.WARNING.name == "WARNING"
        assert MaintenanceAlertLevel.CRITICAL.name == "CRITICAL"


# Tests for simplified API functions


class TestRecordShaperResult:
    def test_record_x_axis_result(self, database):
        from sensing_voxel.monitoring.maintenance import record_shaper_result

        record_shaper_result(
            database=database,
            printer_id="test_printer",
            axis="X",
            frequency=52.5,
            recommended_shaper="zv",
            calibration_id="cal_x_001",
        )

        # Verify by checking history
        from sensing_voxel.monitoring.maintenance import MaintenanceTracker

        tracker = MaintenanceTracker(database, "test_printer")
        history = tracker.get_calibration_history()

        assert len(history) == 1
        _, result = history[0]
        assert result.x_resonance == 52.5
        assert result.x_shaper == ShaperType.ZV

    def test_record_y_axis_result(self, database):
        from sensing_voxel.monitoring.maintenance import record_shaper_result

        record_shaper_result(
            database=database,
            printer_id="test_printer",
            axis="Y",
            frequency=47.0,
            recommended_shaper="mzv",
            notes="Y-axis calibration",
        )

        from sensing_voxel.monitoring.maintenance import MaintenanceTracker

        tracker = MaintenanceTracker(database, "test_printer")
        history = tracker.get_calibration_history()

        assert len(history) == 1
        _, result = history[0]
        assert result.y_resonance == 47.0
        assert result.y_shaper == ShaperType.MZV

    def test_record_invalid_axis(self, database):
        from sensing_voxel.monitoring.maintenance import record_shaper_result

        with pytest.raises(ValueError, match="Invalid axis"):
            record_shaper_result(
                database=database,
                printer_id="test_printer",
                axis="Z",  # Invalid axis
                frequency=50.0,
                recommended_shaper="zv",
            )


class TestGetBaseline:
    def test_get_baseline_x_axis(self, database):
        from sensing_voxel.monitoring.maintenance import record_shaper_result, get_baseline

        # Record first calibration
        record_shaper_result(
            database=database,
            printer_id="test_printer",
            axis="X",
            frequency=50.0,
            recommended_shaper="zv",
            calibration_id="cal_001",
        )

        # Record second calibration
        record_shaper_result(
            database=database,
            printer_id="test_printer",
            axis="X",
            frequency=55.0,
            recommended_shaper="zv",
            calibration_id="cal_002",
        )

        baseline = get_baseline(database, "test_printer", "X")
        assert baseline == 50.0  # First recorded value

    def test_get_baseline_y_axis(self, database):
        from sensing_voxel.monitoring.maintenance import record_shaper_result, get_baseline

        record_shaper_result(
            database=database,
            printer_id="test_printer",
            axis="Y",
            frequency=45.0,
            recommended_shaper="zv",
            calibration_id="cal_001",
        )

        baseline = get_baseline(database, "test_printer", "Y")
        assert baseline == 45.0

    def test_get_baseline_no_history(self, database):
        from sensing_voxel.monitoring.maintenance import get_baseline

        baseline = get_baseline(database, "test_printer", "X")
        assert baseline is None

    def test_get_baseline_invalid_axis(self, database):
        from sensing_voxel.monitoring.maintenance import get_baseline

        with pytest.raises(ValueError, match="Invalid axis"):
            get_baseline(database, "test_printer", "Z")


class TestDetectDrift:
    def test_detect_drift_positive(self, database):
        from sensing_voxel.monitoring.maintenance import record_shaper_result, detect_drift

        # Baseline at 50 Hz
        record_shaper_result(
            database=database,
            printer_id="test_printer",
            axis="X",
            frequency=50.0,
            recommended_shaper="zv",
            calibration_id="cal_001",
        )

        # Current at 55 Hz = 10% increase
        drift = detect_drift(database, "test_printer", 55.0, "X")
        assert drift == 10.0

    def test_detect_drift_negative(self, database):
        from sensing_voxel.monitoring.maintenance import record_shaper_result, detect_drift

        record_shaper_result(
            database=database,
            printer_id="test_printer",
            axis="Y",
            frequency=50.0,
            recommended_shaper="zv",
            calibration_id="cal_001",
        )

        # Current at 45 Hz = 10% decrease
        drift = detect_drift(database, "test_printer", 45.0, "Y")
        assert drift == -10.0

    def test_detect_drift_no_baseline(self, database):
        from sensing_voxel.monitoring.maintenance import detect_drift

        drift = detect_drift(database, "test_printer", 50.0, "X")
        assert drift is None

    def test_detect_drift_invalid_axis(self, database):
        from sensing_voxel.monitoring.maintenance import detect_drift

        with pytest.raises(ValueError, match="Invalid axis"):
            detect_drift(database, "test_printer", 50.0, "Z")


class TestCheckCalibrationInterval:
    def test_check_calibration_interval_with_history(self, database):
        from sensing_voxel.monitoring.maintenance import (
            record_shaper_result,
            check_calibration_interval,
        )

        record_shaper_result(
            database=database,
            printer_id="test_printer",
            axis="X",
            frequency=50.0,
            recommended_shaper="zv",
        )

        days_since = check_calibration_interval(database, "test_printer")
        assert days_since is not None
        assert days_since >= 0
        assert days_since < 1  # Just recorded

    def test_check_calibration_interval_no_history(self, database):
        from sensing_voxel.monitoring.maintenance import check_calibration_interval

        days_since = check_calibration_interval(database, "test_printer")
        assert days_since is None


class TestShouldCalibrate:
    def test_should_calibrate_no_history(self, database):
        from sensing_voxel.monitoring.maintenance import should_calibrate

        # No calibration history should trigger calibration
        result = should_calibrate(database, "test_printer")
        assert result is True

    def test_should_calibrate_interval_exceeded(self, database):
        from sensing_voxel.monitoring.maintenance import should_calibrate
        from sensing_voxel.storage.calibration_repository import CalibrationRepository
        from sensing_voxel.storage import CalibrationType

        repo = CalibrationRepository(database)

        # Create old calibration
        from sensing_voxel.monitoring.maintenance import ShaperCalibrationResult, ShaperType

        old_time = datetime.now() - timedelta(days=35)
        result = ShaperCalibrationResult(
            x_resonance=50.0,
            y_resonance=45.0,
            x_shaper=ShaperType.ZV,
            y_shaper=ShaperType.ZV,
        )
        repo.create(
            calibration_id="cal_old",
            printer_id="test_printer",
            calibration_type=CalibrationType.SHAPER,
            values=result.to_dict(),
            performed_at=old_time,
        )

        # Should recommend calibration (35 days > 30 default)
        should = should_calibrate(database, "test_printer", interval_days=30)
        assert should is True

    def test_should_calibrate_drift_exceeded(self, database):
        from sensing_voxel.monitoring.maintenance import (
            record_shaper_result,
            should_calibrate,
        )

        # Record baseline
        record_shaper_result(
            database=database,
            printer_id="test_printer",
            axis="X",
            frequency=50.0,
            recommended_shaper="zv",
            calibration_id="cal_001",
        )

        # Current frequency shows 15% drift (above 10% threshold)
        should = should_calibrate(database, "test_printer", drift_threshold=10.0, interval_days=60)
        # This will check against the last recorded value
        # Since we only have one record, drift check won't trigger
        # But let's add a second record to test drift

        record_shaper_result(
            database=database,
            printer_id="test_printer",
            axis="X",
            frequency=58.0,  # 16% drift from 50
            recommended_shaper="zv",
            calibration_id="cal_002",
        )

        should = should_calibrate(database, "test_printer", drift_threshold=10.0, interval_days=60)
        assert should is True

    def test_should_calibrate_healthy(self, database):
        from sensing_voxel.monitoring.maintenance import (
            record_shaper_result,
            should_calibrate,
        )

        # Record baseline
        record_shaper_result(
            database=database,
            printer_id="test_printer",
            axis="X",
            frequency=50.0,
            recommended_shaper="zv",
            calibration_id="cal_001",
        )

        # Record recent with minimal drift
        record_shaper_result(
            database=database,
            printer_id="test_printer",
            axis="X",
            frequency=51.0,  # 2% drift (below 10% threshold)
            recommended_shaper="zv",
            calibration_id="cal_002",
        )

        # Should not recommend calibration
        should = should_calibrate(database, "test_printer", drift_threshold=10.0, interval_days=30)
        assert should is False
