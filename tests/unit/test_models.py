"""
Unit tests for core data models
"""

import pytest
from datetime import datetime, timedelta
from sensing_voxel.core.models import (
    # Enums
    AgentState,
    NormalizedPrinterState,
    DecisionAction,
    ConsumeResult,
    FileValidationStatus,
    SlicingStatus,
    VisionConfidenceLevel,
    FirstLayerQuality,
    CalibrationStatus,
    ZOffsetStatus,
    CommandResultStatus,
    # Data classes
    JobContext,
    PrintProgress,
    TelemetryData,
    FirstLayerParams,
    FirstLayerBounds,
    CommandResult,
    HomographyMatrix,
    CameraCalibration,
    FirstLayerAnalysisResult,
    VisionFrame,
    ZOffsetState,
    MonitoringConfig,
    MonitoringStats,
    SlicingResult,
    GeometryAnalysis,
    TelegramUser,
    BotMessage,
    DecisionTimeout,
    # Helper functions
    generate_job_id,
    generate_decision_token,
)


class TestEnums:
    """Test enum values"""

    def test_agent_state_enum(self):
        """Test AgentState has all required states"""
        required_states = [
            "IDLE",
            "RECEIVING",
            "SLICING",
            "AWAITING_APPROVAL",
            "UPLOADING",
            "PRINTING",
            "MONITORING",
            "PAUSED_PENDING_USER",
            "COMPLETE",
            "FAILED",
            "EMERGENCY_FAILED",
            "ERROR",
        ]
        actual_states = [s.name for s in AgentState]
        assert set(required_states) == set(actual_states)

    def test_normalized_printer_state_enum(self):
        """Test NormalizedPrinterState has all required states"""
        required_states = [
            "PRINTING",
            "PAUSED",
            "IDLE_NO_ACTIVE_PRINT",
            "TERMINAL",
            "ERROR",
            "UNKNOWN",
        ]
        actual_states = [s.name for s in NormalizedPrinterState]
        assert set(required_states) == set(actual_states)

    def test_decision_action_enum(self):
        """Test DecisionAction has all required actions"""
        required_actions = [
            "APPROVE",
            "REJECT",
            "RESUME",
            "CANCEL",
            "TIMEOUT_APPROVAL",
        ]
        actual_actions = [a.name for a in DecisionAction]
        assert set(required_actions) == set(actual_actions)


class TestJobContext:
    """Test JobContext data class"""

    def test_job_context_creation(self):
        """Test JobContext can be created"""
        job = JobContext(
            job_id="test_job",
            job_file="/path/to/file.stl",
            uploaded_at=datetime.now(),
        )
        assert job.job_id == "test_job"
        assert job.job_file == "/path/to/file.stl"
        assert job.slicing_status == SlicingStatus.PENDING
        assert job.file_validation_status == FileValidationStatus.VALID
        assert job.is_external is False
        assert job.external_monitoring is False

    def test_job_context_with_optional_fields(self):
        """Test JobContext with optional fields"""
        now = datetime.now()
        job = JobContext(
            job_id="test_job",
            job_file="/path/to/file.stl",
            uploaded_at=now,
            expires_at=now + timedelta(hours=1),
            decision_token="token123",
            decision_consumed=True,
            decision_action=DecisionAction.APPROVE,
            decision_consumed_at=now + timedelta(minutes=5),
            gcode_path="/path/to/output.gcode",
            gcode_hash="abc123",
            first_layer_params=FirstLayerParams(
                layer_height=0.2,
                nozzle_temp=200.0,
                bed_temp=60.0,
                speed=20.0,
            ),
            first_layer_bounds=FirstLayerBounds(
                min_x=0.0,
                max_x=100.0,
                min_y=0.0,
                max_y=100.0,
            ),
            is_external=True,
            external_monitoring=True,
        )
        assert job.expires_at is not None
        assert job.decision_token == "token123"
        assert job.decision_consumed is True
        assert job.decision_action == DecisionAction.APPROVE
        assert job.gcode_path == "/path/to/output.gcode"
        assert job.first_layer_params is not None
        assert job.first_layer_bounds is not None
        assert job.is_external is True
        assert job.external_monitoring is True


class TestFirstLayerBounds:
    """Test FirstLayerBounds properties"""

    def test_bounds_properties(self):
        """Test width, height, and area properties"""
        bounds = FirstLayerBounds(
            min_x=10.0,
            max_x=50.0,
            min_y=20.0,
            max_y=70.0,
        )
        assert bounds.width == 40.0
        assert bounds.height == 50.0
        assert bounds.area == 2000.0


class TestHomographyMatrix:
    """Test HomographyMatrix validity"""

    def test_valid_calibration_no_expiry(self):
        """Test calibration valid when no expiry set"""
        homography = HomographyMatrix(
            matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            mm_per_pixel=0.1,
            valid_until=None,
        )
        assert homography.is_valid() is True

    def test_valid_calibration_before_expiry(self):
        """Test calibration valid before expiry"""
        future = datetime.now() + timedelta(days=1)
        homography = HomographyMatrix(
            matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            mm_per_pixel=0.1,
            valid_until=future,
        )
        assert homography.is_valid() is True

    def test_expired_calibration(self):
        """Test calibration expired after expiry"""
        past = datetime.now() - timedelta(days=1)
        homography = HomographyMatrix(
            matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            mm_per_pixel=0.1,
            valid_until=past,
        )
        assert homography.is_valid() is False


class TestZOffsetState:
    """Test ZOffsetState properties"""

    def test_is_dirty_property(self):
        """Test is_dirty property"""
        clean_state = ZOffsetState(status=ZOffsetStatus.CLEAN)
        dirty_state = ZOffsetState(status=ZOffsetStatus.DIRTY)

        assert clean_state.is_dirty is False
        assert dirty_state.is_dirty is True

    def test_magnitude_property(self):
        """Test magnitude property"""
        state = ZOffsetState(current_offset=-0.03)
        assert state.magnitude == 0.03


class TestMonitoringConfig:
    """Test MonitoringConfig defaults"""

    def test_default_values(self):
        """Test default configuration values"""
        config = MonitoringConfig()
        assert config.enable_first_layer is True
        assert config.evaluation_interval_seconds == 5
        assert config.cooldown_seconds == 10
        assert config.min_consecutive_evaluations == 2
        assert config.z_offset_step_mm == 0.01
        assert config.z_offset_max_correction_mm == 0.05
        assert config.z_offset_saturation_count == 3


class TestHelperFunctions:
    """Test helper functions"""

    def test_generate_job_id(self):
        """Test job ID generation"""
        job_id = generate_job_id()
        assert job_id.startswith("job_")
        assert len(job_id) > 10  # Should have timestamp + UUID fragment

    def test_generate_decision_token(self):
        """Test decision token generation"""
        token1 = generate_decision_token()
        token2 = generate_decision_token()
        assert len(token1) == 32  # UUID hex is 32 chars
        assert token1 != token2  # Should be unique


class TestDataClassDefaults:
    """Test data class default values"""

    def test_print_progress_defaults(self):
        """Test PrintProgress default values"""
        progress = PrintProgress(state=NormalizedPrinterState.PRINTING)
        assert progress.filename is None
        assert progress.progress is None
        assert progress.current_layer is None

    def test_telemetry_data_defaults(self):
        """Test TelemetryData default values"""
        telemetry = TelemetryData(timestamp=datetime.now())
        assert telemetry.temperatures == {}
        assert telemetry.position is None

    def test_command_result_defaults(self):
        """Test CommandResult default values"""
        result = CommandResult(status=CommandResultStatus.SUCCESS)
        assert result.message is None
        assert result.data is None

    def test_monitoring_stats_defaults(self):
        """Test MonitoringStats default values"""
        stats = MonitoringStats()
        assert stats.frames_processed == 0
        assert stats.z_offset_corrections == 0
