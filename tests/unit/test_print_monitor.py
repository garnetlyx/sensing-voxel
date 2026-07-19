"""
Unit tests for PrintMonitor — background print monitoring with LLM vision analysis.
"""

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sensing_voxel.config import PrintMonitorConfig
from sensing_voxel.monitoring.print_monitor import (
    AnalysisResult,
    MonitorSeverity,
    MonitorState,
    PrintMonitor,
)
from sensing_voxel.safety.models import MonitorStartContract


@pytest.fixture
def mock_adapter():
    adapter = MagicMock()
    adapter.is_connected = AsyncMock(return_value=True)
    adapter.get_print_progress = AsyncMock(
        return_value=MagicMock(data={"state": "printing", "extruder_temp": 210, "bed_temp": 60})
    )
    adapter.pause_print = AsyncMock(return_value=MagicMock(status="success"))
    adapter.upload_file = AsyncMock(return_value=MagicMock(status="success"))
    adapter.start_print = AsyncMock(return_value=MagicMock(status="success"))
    return adapter


@pytest.fixture
def mock_llm():
    return AsyncMock()


@pytest.fixture
def mock_notifier():
    return AsyncMock()


@pytest.fixture
def monitor(mock_adapter, mock_llm, mock_notifier):
    return PrintMonitor(
        printer_id="test-printer",
        adapter=mock_adapter,
        rtsp_url="rtsp://fake/stream",
        llm_send_fn=mock_llm,
        notifier_fn=mock_notifier,
        config=PrintMonitorConfig(
            enabled=True,
            check_interval_seconds=0.1,
            anomaly_response_timeout_seconds=0.5,
            camera_search_enabled=False,
            auto_pause_enabled=True,
        ),
    )


class TestPrintMonitorParsing:
    def test_parse_ok_response(self, monitor):
        response = json.dumps({
            "severity": "ok",
            "anomaly_type": None,
            "confidence": 0.95,
            "printer_visible": True,
            "summary": "Print looks normal",
        })
        result = monitor._parse_analysis(response)
        assert result.severity == MonitorSeverity.OK
        assert result.printer_visible is True
        assert result.confidence == 0.95

    def test_parse_critical_anomaly(self, monitor):
        response = json.dumps({
            "severity": "critical",
            "anomaly_type": "spaghetti",
            "confidence": 0.9,
            "printer_visible": True,
            "summary": "Massive spaghetti failure",
            "recommended_action": "pause immediately",
        })
        result = monitor._parse_analysis(response)
        assert result.severity == MonitorSeverity.CRITICAL
        assert result.anomaly_type == "spaghetti"
        assert result.recommended_action == "pause immediately"

    def test_parse_invalid_json_is_warning(self, monitor):
        result = monitor._parse_analysis("not json at all")
        assert result.severity == MonitorSeverity.WARNING
        assert result.anomaly_type == "parse_error"
        assert monitor._is_anomaly(result) is True

    def test_parse_json_embedded_in_text(self, monitor):
        response = "Here is the analysis:\n" + json.dumps({
            "severity": "warning",
            "anomaly_type": "air_printing",
            "confidence": 0.7,
            "printer_visible": True,
            "summary": "No filament coming out",
        })
        result = monitor._parse_analysis(response)
        assert result.severity == MonitorSeverity.WARNING
        assert result.anomaly_type == "air_printing"


class TestPrintMonitorAnomalyDetection:
    def test_ok_result_is_not_anomaly(self, monitor):
        result = AnalysisResult(
            severity=MonitorSeverity.OK,
            anomaly_type=None,
            confidence=0.95,
            printer_visible=True,
            summary="All good",
        )
        assert monitor._is_anomaly(result) is False

    def test_warning_is_anomaly(self, monitor):
        result = AnalysisResult(
            severity=MonitorSeverity.WARNING,
            anomaly_type="spaghetti",
            confidence=0.8,
            printer_visible=True,
            summary="Spaghetti detected",
        )
        assert monitor._is_anomaly(result) is True

    def test_critical_is_anomaly(self, monitor):
        result = AnalysisResult(
            severity=MonitorSeverity.CRITICAL,
            anomaly_type="fire",
            confidence=0.9,
            printer_visible=True,
            summary="Fire detected",
        )
        assert monitor._is_anomaly(result) is True

    def test_low_confidence_warning_is_anomaly(self, monitor):
        result = AnalysisResult(
            severity=MonitorSeverity.WARNING,
            anomaly_type="spaghetti",
            confidence=0.4,
            printer_visible=True,
            summary="Maybe spaghetti",
        )
        assert monitor._is_anomaly(result) is True

    def test_unknown_result_is_anomaly(self, monitor):
        result = AnalysisResult(
            severity=MonitorSeverity.UNKNOWN,
            anomaly_type=None,
            confidence=0.0,
            printer_visible=True,
            summary="Analysis unavailable",
        )
        assert monitor._is_anomaly(result) is True

    def test_printer_not_visible_is_anomaly(self, monitor):
        result = AnalysisResult(
            severity=MonitorSeverity.WARNING,
            anomaly_type="printer_not_visible",
            confidence=0.8,
            printer_visible=False,
            summary="Can't see printer",
        )
        assert monitor._is_anomaly(result) is True


class TestPrintMonitorAnalysisFailures:
    @pytest.mark.asyncio
    async def test_llm_exception_returns_warning_anomaly(self, monitor, mock_llm):
        mock_llm.side_effect = RuntimeError("provider unavailable")

        result = await monitor._analyze("encoded-image", {"filename": "test.gcode"})

        assert result.severity == MonitorSeverity.WARNING
        assert result.anomaly_type == "llm_error"
        assert result.printer_visible is False
        assert monitor._is_anomaly(result) is True

    @pytest.mark.asyncio
    async def test_analysis_timeout_becomes_warning_anomaly(self, monitor):
        async def timeout_analyze(*args, **kwargs):
            raise asyncio.TimeoutError

        monitor._analyze = timeout_analyze
        monitor._capture_rtsp = AsyncMock(return_value="encoded-image")
        monitor._state = MagicMock()
        monitor._state.check_count = 1
        monitor._state.filename = "test.gcode"
        monitor._state.camera_failures = 0
        monitor._state.camera_notified = False
        progress = MagicMock()
        progress.state = "PRINTING"
        progress.current_layer = 1
        progress.data = {"state": "printing"}
        monitor._adapter.get_print_progress = AsyncMock(return_value=progress)
        monitor._handle_anomaly = AsyncMock()

        await monitor._check_once()

        result = monitor._state.last_analysis
        assert result.severity == MonitorSeverity.WARNING
        assert result.anomaly_type == "llm_timeout"
        assert monitor._is_anomaly(result) is True
        monitor._handle_anomaly.assert_awaited_once_with(result)


class TestPrintMonitorCadence:
    @pytest.mark.asyncio
    async def test_first_layer_uses_60_seconds_then_normal_interval(self, mock_adapter, mock_llm, mock_notifier):
        intervals: list[float] = []
        checks = 0
        monitor: PrintMonitor

        async def wait_once(stop_event: asyncio.Event, timeout: float) -> bool:
            nonlocal checks
            intervals.append(timeout)
            if len(intervals) == 2:
                stop_event.set()
                return True
            return False

        monitor = PrintMonitor(
            printer_id="test-printer",
            adapter=mock_adapter,
            rtsp_url="rtsp://fake/stream",
            llm_send_fn=mock_llm,
            notifier_fn=mock_notifier,
            config=PrintMonitorConfig(check_interval_seconds=1800),
            wait_fn=wait_once,
        )

        async def check_once() -> None:
            nonlocal checks
            checks += 1
            assert monitor._state is not None
            if checks == 2:
                monitor._update_first_layer_state(MagicMock(current_layer=2))

        monitor._check_once = check_once
        monitor._state = MonitorState(
            printer_id="test-printer", first_layer_z=0.2, required=True
        )

        await monitor._loop()

        assert intervals == [60.0, 1800]
        assert checks == 2

    @pytest.mark.asyncio
    async def test_unknown_current_layer_fails_closed_when_required(self, monitor, mock_adapter):
        monitor._state = MonitorState(
            printer_id="test-printer", first_layer_z=0.2, required=True
        )
        progress = MagicMock(state="PRINTING", current_layer=None)
        mock_adapter.get_print_progress.return_value = progress
        monitor._fail_closed = AsyncMock()

        await monitor._check_once()

        monitor._fail_closed.assert_awaited_once_with("current_layer_unknown")
        monitor._capture_rtsp = AsyncMock()
        monitor._capture_rtsp.assert_not_awaited()


class TestPrintMonitorRequiredReadiness:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "failure",
        ["progress", "camera", "llm"],
    )
    async def test_required_monitor_rejects_unavailable_dependency(
        self, monitor, mock_adapter, failure
    ):
        progress = MagicMock(state="IDLE_NO_ACTIVE_PRINT", current_layer=None)
        mock_adapter.get_print_progress.return_value = progress
        monitor._capture_rtsp = AsyncMock(return_value="encoded-image")
        monitor._analyze = AsyncMock(
            return_value=AnalysisResult(
                severity=MonitorSeverity.OK,
                anomaly_type=None,
                confidence=1.0,
                printer_visible=True,
                summary="ready",
            )
        )
        if failure == "progress":
            mock_adapter.get_print_progress.side_effect = RuntimeError("offline")
        elif failure == "camera":
            monitor._capture_rtsp.return_value = None
        else:
            monitor._analyze.return_value = AnalysisResult(
                severity=MonitorSeverity.WARNING,
                anomaly_type="llm_error",
                confidence=0.0,
                printer_visible=False,
                summary="unavailable",
            )

        contract = MonitorStartContract(
            filename="validated.gcode",
            source_trust="user_gcode",
            first_layer_z=0.2,
        )
        with pytest.raises(RuntimeError, match="required monitor"):
            await monitor.start_monitoring(contract=contract)

        assert not monitor.is_monitoring


    @pytest.mark.asyncio
    async def test_required_restart_does_not_create_duplicate_loop(self, monitor, mock_adapter):
        mock_adapter.get_print_progress.return_value = MagicMock(
            state="IDLE_NO_ACTIVE_PRINT", current_layer=None
        )
        monitor._capture_rtsp = AsyncMock(return_value="encoded-image")
        monitor._analyze = AsyncMock(
            return_value=AnalysisResult(
                severity=MonitorSeverity.OK,
                anomaly_type=None,
                confidence=1.0,
                printer_visible=True,
                summary="ready",
            )
        )
        contract = MonitorStartContract(
            filename="validated.gcode",
            source_trust="sva_generated",
            first_layer_z=0.2,
        )

        await monitor.start_monitoring(contract=contract)
        first_task = monitor._task
        await monitor.start_monitoring(contract=contract)

        assert monitor._task is first_task
        assert monitor._capture_rtsp.await_count == 1
        await monitor.stop_monitoring(reason="test")


class TestPrintMonitorLifecycle:
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, monitor):
        await monitor.start_monitoring(filename="test.gcode")
        assert monitor.is_monitoring
        assert monitor._state is not None
        assert monitor._state.filename == "test.gcode"

        await monitor.stop_monitoring(reason="test")
        assert not monitor.is_monitoring
        assert monitor._state is None

    @pytest.mark.asyncio
    async def test_monitor_stops_from_its_own_loop_on_completion(
        self, monitor, mock_adapter, mock_notifier
    ):
        mock_adapter.get_print_progress.return_value = MagicMock(
            state="COMPLETE", current_layer=2
        )
        monitor._config.notify_on_completion = True

        await monitor.start_monitoring(filename="test.gcode")
        task = monitor._task
        assert task is not None
        await asyncio.wait_for(task, timeout=1)

        assert not monitor.is_monitoring
        assert monitor._state is None
        mock_notifier.assert_awaited_once_with(
            "test-printer", "success", "Print completed on test-printer: test.gcode"
        )

    @pytest.mark.asyncio
    async def test_acknowledge_clears_anomaly(self, monitor):
        monitor._state = MagicMock()
        monitor._state.anomaly_pending = True
        await monitor.acknowledge()
        assert monitor._ack_event.is_set()
        assert monitor._state.anomaly_pending is False


class TestPrintMonitorAutoPause:
    @pytest.mark.asyncio
    async def test_auto_pause_on_unacknowledged_anomaly(self, monitor, mock_adapter, mock_notifier):
        monitor._state = MagicMock()
        monitor._state.anomaly_pending = False
        monitor._state.printer_id = "test-printer"

        result = AnalysisResult(
            severity=MonitorSeverity.CRITICAL,
            anomaly_type="fire",
            confidence=0.95,
            printer_visible=True,
            summary="FIRE DETECTED",
            recommended_action="pause now",
        )

        await monitor._handle_anomaly(result)
        assert monitor._state.anomaly_pending is True
        assert monitor._pause_task is not None

        await asyncio.sleep(0.8)

        mock_adapter.pause_print.assert_called_once()
        mock_notifier.assert_called()
        assert monitor._pause_task.done()

    @pytest.mark.asyncio
    async def test_acknowledge_prevents_pause(self, monitor, mock_adapter):
        monitor._state = MagicMock()
        monitor._state.anomaly_pending = False
        monitor._state.printer_id = "test-printer"

        result = AnalysisResult(
            severity=MonitorSeverity.WARNING,
            anomaly_type="spaghetti",
            confidence=0.8,
            printer_visible=True,
            summary="Mild spaghetti",
        )

        await monitor._handle_anomaly(result)

        await asyncio.sleep(0.1)
        await monitor.acknowledge()

        await asyncio.sleep(0.8)
        mock_adapter.pause_print.assert_not_called()
