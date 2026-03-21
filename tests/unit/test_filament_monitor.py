"""Unit tests for filament spool monitoring."""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from sensing_voxel.config import FilamentMonitorConfig
from sensing_voxel.core.models import (
    FilamentCheckResult,
    FilamentSpoolMeasurement,
    FilamentSpoolStatus,
    TangleDecision,
    TangleDetectionResult,
    VisionFrame,
)
from sensing_voxel.vision.filament import (
    FilamentMonitor,
    SpoolAnalyzer,
    SpoolGeometry,
    TangleDetector,
    capture_spool_image,
    estimate_remaining_filament,
    check_adequate_filament,
    detect_tangle,
    _line_segments_intersect,
    pause_on_tangle,
    DEFAULT_FILAMENT_DIAMETER_MM,
    DEFAULT_FILAMENT_DENSITY_G_CM3,
    DEFAULT_SPOOL_WIDTH_MM,
)
from sensing_voxel.core.models import CommandResult, CommandResultStatus
from sensing_voxel.vision.usb_camera import USBCameraStream


class TestSpoolAnalyzer:
    """Tests for spool analyzer."""

    @pytest.fixture
    def config(self):
        return FilamentMonitorConfig(
            enabled=True,
            usb_camera_index=0,
            spool_full_weight_g=1000.0,
            spool_empty_weight_g=250.0,
        )

    @pytest.fixture
    def analyzer(self, config):
        return SpoolAnalyzer(config)

    def test_analyze_empty_frame(self, analyzer):
        """Test analyzing empty/none frame data."""
        frame = VisionFrame(
            frame_data=None,
            timestamp=datetime.now(),
            frame_id=1,
            width=0,
            height=0,
        )
        result = analyzer.analyze(frame)
        assert result.confidence == 0.0
        assert result.estimated_weight_g == 0.0

    def test_analyze_black_image(self, analyzer):
        """Test analyzing completely black image."""
        black_img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = VisionFrame(
            frame_data=black_img,
            timestamp=datetime.now(),
            frame_id=1,
            width=640,
            height=480,
        )
        result = analyzer.analyze(frame)
        assert result.confidence < 0.3  # Low confidence on black image

    def test_fallback_detection(self, analyzer):
        """Test fallback contour detection."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2 = pytest.importorskip("cv2")
        # Draw a circle to simulate spool
        cv2.circle(img, (320, 240), 100, (255, 255, 255), -1)

        frame = VisionFrame(
            frame_data=img,
            timestamp=datetime.now(),
            frame_id=1,
            width=640,
            height=480,
        )
        result = analyzer.analyze(frame)
        assert result.confidence > 0.0

    def test_estimate_weight_full(self, analyzer):
        """Test weight estimation for full spool."""
        weight = analyzer._estimate_weight(200.0)  # Full diameter
        assert weight == pytest.approx(1000.0, rel=0.01)

    def test_estimate_weight_empty(self, analyzer):
        """Test weight estimation for empty spool."""
        weight = analyzer._estimate_weight(55.0)  # Hub diameter
        assert weight == pytest.approx(250.0, rel=0.01)

    def test_estimate_weight_zero(self, analyzer):
        """Test weight estimation for zero/negative diameter."""
        weight = analyzer._estimate_weight(0.0)
        assert weight == 250.0  # Returns empty weight

    def test_px_to_mm_conversion(self, analyzer):
        """Test pixel to millimeter conversion."""
        mm = analyzer._px_to_mm(100.0, 640)
        expected_mm = 100.0 * 200.0 / 640
        assert mm == pytest.approx(expected_mm, rel=0.01)

    def test_px_to_mm_zero_width(self, analyzer):
        """Test px_to_mm with zero width."""
        mm = analyzer._px_to_mm(100.0, 0)
        assert mm == 0.0


class TestTangleDetector:
    """Tests for tangle detector."""

    @pytest.fixture
    def config(self):
        return FilamentMonitorConfig(
            enabled=True,
            tangle_detection_threshold=0.7,
            tangle_confirmation_frames=3,
        )

    @pytest.fixture
    def detector(self, config):
        return TangleDetector(config)

    def test_detect_no_tangle(self, detector):
        """Test detection on image with no tangles."""
        cv2 = pytest.importorskip("cv2")
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw parallel lines (no intersection)
        for i in range(0, 400, 20):
            cv2.line(img, (100, i), (500, i), (255, 255, 255), 2)

        frame = VisionFrame(
            frame_data=img,
            timestamp=datetime.now(),
            frame_id=1,
            width=640,
            height=480,
        )
        result = detector.detect(frame)
        assert result.severity == TangleDecision.NONE
        assert not result.detected

    def test_detect_empty_frame(self, detector):
        """Test detection on empty frame."""
        frame = VisionFrame(
            frame_data=None,
            timestamp=datetime.now(),
            frame_id=1,
            width=0,
            height=0,
        )
        result = detector.detect(frame)
        assert result.severity == TangleDecision.NONE
        assert result.confidence == 0.0

    def test_lines_intersect(self, detector):
        """Test line intersection detection."""
        # Intersecting lines
        line1 = (0, 0, 100, 100)
        line2 = (0, 100, 100, 0)
        assert detector._lines_intersect(line1, line2)

        # Parallel lines
        line3 = (0, 0, 100, 0)
        line4 = (0, 50, 100, 50)
        assert not detector._lines_intersect(line3, line4)

    def test_temporal_confirmation(self, detector):
        """Test temporal confirmation requires multiple frames."""
        cv2 = pytest.importorskip("cv2")
        # Create image with intersecting lines (high tangle score)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(10):
            x1, y1 = np.random.randint(0, 640), np.random.randint(0, 480)
            x2, y2 = np.random.randint(0, 640), np.random.randint(0, 480)
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

        frame = VisionFrame(
            frame_data=img,
            timestamp=datetime.now(),
            frame_id=1,
            width=640,
            height=480,
        )

        # First detection - not confirmed yet
        result = detector.detect(frame)
        assert not result.detected  # Need 3 confirmations


class TestFilamentMonitor:
    """Tests for filament monitor."""

    @pytest.fixture
    def config(self):
        return FilamentMonitorConfig(
            enabled=True,
            usb_camera_index=0,
            check_interval_seconds=1.0,
            min_filament_safety_margin_percent=10.0,
        )

    @pytest.fixture
    def monitor(self, config):
        return FilamentMonitor(config)

    def test_check_filament_sufficient(self, monitor):
        """Test filament sufficiency check - sufficient case."""
        measurement = FilamentSpoolMeasurement(
            estimated_diameter_mm=150.0,
            estimated_weight_g=700.0,
            confidence=0.8,
        )
        result = monitor.check_filament_sufficient(measurement, 500.0)
        assert result.status == FilamentSpoolStatus.SUFFICIENT
        assert "sufficient" in result.message.lower()

    def test_check_filament_low(self, monitor):
        """Test filament sufficiency check - low case."""
        measurement = FilamentSpoolMeasurement(
            estimated_diameter_mm=100.0,
            estimated_weight_g=520.0,
            confidence=0.8,
        )
        result = monitor.check_filament_sufficient(measurement, 500.0)
        assert result.status == FilamentSpoolStatus.LOW
        assert "low" in result.message.lower()

    def test_check_filament_insufficient(self, monitor):
        """Test filament sufficiency check - insufficient case."""
        measurement = FilamentSpoolMeasurement(
            estimated_diameter_mm=80.0,
            estimated_weight_g=300.0,
            confidence=0.8,
        )
        result = monitor.check_filament_sufficient(measurement, 500.0)
        assert result.status == FilamentSpoolStatus.INSUFFICIENT
        assert "insufficient" in result.message.lower()

    def test_check_result_with_custom_margin(self, monitor):
        """Test filament check with custom safety margin."""
        monitor.config.min_filament_safety_margin_percent = 50.0
        measurement = FilamentSpoolMeasurement(
            estimated_diameter_mm=150.0,
            estimated_weight_g=700.0,
            confidence=0.8,
        )
        # With 50% margin, 500g job needs 750g
        result = monitor.check_filament_sufficient(measurement, 500.0)
        assert result.status == FilamentSpoolStatus.LOW

    @pytest.mark.asyncio
    async def test_start_disabled_config(self):
        """Test start when disabled in config."""
        config = FilamentMonitorConfig(enabled=False)
        monitor = FilamentMonitor(config)
        result = await monitor.start()
        assert not result

    @pytest.mark.asyncio
    async def test_stop_not_running(self, monitor):
        """Test stop when not running."""
        await monitor.stop()  # Should not raise

    @pytest.mark.asyncio
    async def test_measure_spool_no_camera(self, monitor):
        """Test measure spool when camera not connected."""
        result = await monitor.measure_spool()
        assert result is None  # Camera not started

    @pytest.mark.asyncio
    async def test_check_tangle_no_camera(self, monitor):
        """Test check tangle when camera not connected."""
        result = await monitor.check_tangle()
        assert result is None  # Camera not started


class TestFilamentCheckResult:
    """Tests for FilamentCheckResult model."""

    def test_result_creation(self):
        """Test creating a filament check result."""
        result = FilamentCheckResult(
            status=FilamentSpoolStatus.SUFFICIENT,
            estimated_remaining_g=750.0,
            estimated_required_g=500.0,
            safety_margin_percent=10.0,
            message="Filament sufficient",
        )
        assert result.status == FilamentSpoolStatus.SUFFICIENT
        assert result.estimated_remaining_g == 750.0
        assert result.estimated_required_g == 500.0


class TestSpoolGeometry:
    """Tests for SpoolGeometry dataclass."""

    def test_geometry_creation(self):
        """Test creating spool geometry."""
        geom = SpoolGeometry(
            spool_diameter_px=200.0,
            filament_diameter_px=150.0,
            center_x=320.0,
            center_y=240.0,
            confidence=0.85,
        )
        assert geom.spool_diameter_px == 200.0
        assert geom.filament_diameter_px == 150.0
        assert geom.center_x == 320.0
        assert geom.confidence == 0.85


class TestTangleDetectionResult:
    """Tests for TangleDetectionResult model."""

    def test_result_creation(self):
        """Test creating tangle detection result."""
        result = TangleDetectionResult(
            detected=True,
            confidence=0.85,
            severity=TangleDecision.CRITICAL,
            bounding_boxes=[(10.0, 10.0, 50.0, 50.0)],
        )
        assert result.detected
        assert result.confidence == 0.85
        assert result.severity == TangleDecision.CRITICAL
        assert len(result.bounding_boxes) == 1


class TestUSBCameraStream:
    """Tests for USB camera stream."""

    @pytest.fixture
    def config(self):
        return FilamentMonitorConfig(
            enabled=True,
            usb_camera_index=0,
            fps=5,
            frame_width=640,
            frame_height=480,
        )

    @pytest.fixture
    def stream(self, config):
        return USBCameraStream(config)

    def test_initial_state(self, stream):
        """Test initial state of USB camera stream."""
        assert not stream.is_connected
        assert not stream._running

    @pytest.mark.asyncio
    async def test_stop_not_started(self, stream):
        """Test stop when not started."""
        await stream.stop()  # Should not raise

    def test_is_connected_no_camera(self, stream):
        """Test is_connected when no camera."""
        assert not stream.is_connected

    @pytest.mark.asyncio
    async def test_get_frame_timeout(self, stream):
        """Test get frame with timeout."""
        result = await stream.get_frame(timeout=0.1)
        assert result is None  # No frame available, timeout


class TestIntegrationFilamentFlow:
    """Integration tests for filament monitoring flow."""

    @pytest.mark.asyncio
    async def test_full_monitoring_flow(self):
        """Test full monitoring flow."""
        config = FilamentMonitorConfig(
            enabled=False,  # Don't try to start real camera
            min_filament_safety_margin_percent=10.0,
        )
        monitor = FilamentMonitor(config)

        # Mock the measurement
        measurement = FilamentSpoolMeasurement(
            estimated_diameter_mm=150.0,
            estimated_weight_g=700.0,
            confidence=0.8,
        )

        # Check filament sufficiency for a 500g job
        result = monitor.check_filament_sufficient(measurement, 500.0)

        assert result.status == FilamentSpoolStatus.SUFFICIENT
        assert result.estimated_remaining_g == 700.0
        assert result.estimated_required_g == 500.0

    def test_spool_analyzer_line_profile(self):
        """Test line profile extraction."""
        config = FilamentMonitorConfig()
        analyzer = SpoolAnalyzer(config)

        img = np.zeros((480, 640), dtype=np.uint8)
        for i in range(640):
            img[:, i] = (i * 255) // 640

        profile = analyzer._get_line_profile(img, 0, 240, 639, 240)
        assert len(profile) > 0
        assert abs(len(profile) - 640) <= 1  # May be off by 1 due to int conversion

    def test_spool_analyzer_edge_detection(self):
        """Test edge detection in profile."""
        config = FilamentMonitorConfig()
        analyzer = SpoolAnalyzer(config)

        # Create profile with sharp edge
        profile = np.zeros(100, dtype=np.uint8)
        profile[50:] = 255

        edge = analyzer._find_edge_in_profile(profile, 80)
        assert edge > 0  # Should find an edge


class TestSimpleAPIFunctions:
    """Tests for simple standalone API functions."""

    def test_estimate_remaining_filament_empty(self):
        """Test with spool diameter at or below hub diameter."""
        result = estimate_remaining_filament(50.0)
        assert result == 0.0

    def test_estimate_remaining_filament_zero(self):
        """Test with zero diameter."""
        result = estimate_remaining_filament(0.0)
        assert result == 0.0

    def test_estimate_remaining_filament_negative(self):
        """Test with negative diameter."""
        result = estimate_remaining_filament(-10.0)
        assert result == 0.0

    def test_estimate_remaining_filament_full(self):
        """Test with full spool (~200mm diameter)."""
        result = estimate_remaining_filament(200.0)
        # Full spool calculation
        assert result > 300.0  # More than 300 meters

    def test_estimate_remaining_different_filament_diameter(self):
        """Test with 2.85mm filament."""
        result_175 = estimate_remaining_filament(150.0, filament_diameter_mm=1.75)
        result_285 = estimate_remaining_filament(150.0, filament_diameter_mm=2.85)
        # 2.85mm filament should have less length for same volume
        assert result_285 < result_175

    def test_check_adequate_sufficient(self):
        """Test when filament is clearly sufficient."""
        result = check_adequate_filament(
            remaining_diameter_mm=200.0,
            job_requirement_mm=100000.0,  # 100 meters
        )
        assert result is True

    def test_check_adequate_insufficient(self):
        """Test when filament is clearly insufficient."""
        result = check_adequate_filament(
            remaining_diameter_mm=80.0,
            job_requirement_mm=100000.0,  # 100 meters
        )
        assert result is False

    def test_check_adequate_with_safety_margin(self):
        """Test with safety margin."""
        remaining_m = estimate_remaining_filament(150.0)
        remaining_mm = remaining_m * 1000.0

        # With 10% margin, should fail at exactly remaining
        result = check_adequate_filament(
            remaining_diameter_mm=150.0,
            job_requirement_mm=remaining_mm,
            safety_margin=1.1,
        )
        assert result is False

    def test_check_adequate_zero_requirement(self):
        """Test with zero job requirement."""
        result = check_adequate_filament(
            remaining_diameter_mm=100.0,
            job_requirement_mm=0.0,
        )
        assert result is True

    def test_line_segments_intersect_crossing(self):
        """Test two lines that clearly intersect."""
        line1 = (0, 0, 10, 10)
        line2 = (0, 10, 10, 0)
        assert _line_segments_intersect(line1, line2) is True

    def test_line_segments_intersect_parallel(self):
        """Test parallel lines."""
        line1 = (0, 0, 10, 0)
        line2 = (0, 5, 10, 5)
        assert _line_segments_intersect(line1, line2) is False

    def test_detect_tangle_empty(self):
        """Test with None image."""
        result = detect_tangle(np.array([]))  # type: ignore
        assert result is False

    def test_detect_tangle_empty_array(self):
        """Test with empty array."""
        result = detect_tangle(np.array([]))
        assert result is False

    def test_detect_tangle_no_lines(self):
        """Test with blank image (no lines to detect)."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        result = detect_tangle(image)
        assert result is False

    def test_detect_tangle_crossing_lines(self):
        """Test with crossing lines (should detect tangle)."""
        cv2 = pytest.importorskip("cv2")
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.line(image, (50, 50), (150, 150), (255, 255, 255), 5)
        cv2.line(image, (50, 150), (150, 50), (255, 255, 255), 5)
        result = detect_tangle(image, threshold=0.3)
        assert result is True

    @patch("sensing_voxel.vision.filament.cv2.VideoCapture")
    def test_capture_spool_image_success(self, mock_capture_class):
        """Test successful image capture."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
        mock_capture_class.return_value = mock_cap

        result = capture_spool_image(camera_index=0)

        assert result is not None
        assert isinstance(result, np.ndarray)
        mock_cap.release.assert_called_once()

    @patch("sensing_voxel.vision.filament.cv2.VideoCapture")
    def test_capture_spool_image_failure(self, mock_capture_class):
        """Test when camera fails to open."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_capture_class.return_value = mock_cap

        result = capture_spool_image(camera_index=0)
        assert result is None

    @pytest.mark.asyncio
    async def test_pause_on_tangle_success(self):
        """Test successful pause on tangle."""
        mock_adapter = AsyncMock()
        mock_adapter.pause_print.return_value = CommandResult(
            status=CommandResultStatus.SUCCESS,
            message="Print paused",
        )

        result = await pause_on_tangle(mock_adapter)
        assert result is True
        mock_adapter.pause_print.assert_called_once()

    @pytest.mark.asyncio
    async def test_pause_on_tangle_failure(self):
        """Test failed pause on tangle."""
        mock_adapter = AsyncMock()
        mock_adapter.pause_print.return_value = CommandResult(
            status=CommandResultStatus.FAILURE,
            message="Failed to pause",
        )

        result = await pause_on_tangle(mock_adapter)
        assert result is False

    @pytest.mark.asyncio
    async def test_pause_on_tangle_with_notification(self):
        """Test pause with notification callback."""
        mock_adapter = AsyncMock()
        mock_adapter.pause_print.return_value = CommandResult(
            status=CommandResultStatus.SUCCESS,
            message="Print paused",
        )
        mock_notify = AsyncMock()

        result = await pause_on_tangle(mock_adapter, notification_callback=mock_notify)

        assert result is True
        mock_notify.assert_called_once()
        assert "tangle" in mock_notify.call_args[0][0].lower()

    @pytest.mark.asyncio
    async def test_pause_on_tangle_exception(self):
        """Test pause when adapter raises exception."""
        mock_adapter = AsyncMock()
        mock_adapter.pause_print.side_effect = Exception("Connection lost")

        result = await pause_on_tangle(mock_adapter)
        assert result is False

    def test_constants(self):
        """Test module constants."""
        assert DEFAULT_FILAMENT_DIAMETER_MM == 1.75
        assert DEFAULT_FILAMENT_DENSITY_G_CM3 == 1.24
        assert DEFAULT_SPOOL_WIDTH_MM == 70.0
