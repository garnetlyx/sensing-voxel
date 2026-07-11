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
    PrintMonitor,
)


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

    def test_parse_invalid_json(self, monitor):
        result = monitor._parse_analysis("not json at all")
        assert result.severity == MonitorSeverity.UNKNOWN
        assert result.anomaly_type == "parse_error"

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

    def test_low_confidence_not_anomaly(self, monitor):
        result = AnalysisResult(
            severity=MonitorSeverity.WARNING,
            anomaly_type="spaghetti",
            confidence=0.4,
            printer_visible=True,
            summary="Maybe spaghetti",
        )
        assert monitor._is_anomaly(result) is False

    def test_printer_not_visible_not_anomaly(self, monitor):
        result = AnalysisResult(
            severity=MonitorSeverity.WARNING,
            anomaly_type="printer_not_visible",
            confidence=0.8,
            printer_visible=False,
            summary="Can't see printer",
        )
        assert monitor._is_anomaly(result) is False


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
