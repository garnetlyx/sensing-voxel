"""
Unit tests for Phase 4.1 Real-World Test scripts.

Tests for:
- scripts/verify_hardware.py
- scripts/monitor_print.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from verify_hardware import HardwareVerifier, LLM_PROXY_URL, DEFAULT_PRINTERS
from monitor_print import PrintMonitor, TelemetryRecord


class TestHardwareVerifier:
    """Tests for HardwareVerifier class."""

    @pytest.fixture
    def verifier(self, tmp_path):
        output_path = tmp_path / "hardware_status.json"
        return HardwareVerifier(output_path=str(output_path))

    @pytest.fixture
    def mock_aiohttp_response(self):
        """Mock aiohttp response."""
        response = AsyncMock()
        response.status = 200
        response.read = AsyncMock(return_value=b"test data")
        return response

    def test_init(self, verifier, tmp_path):
        """Test verifier initialization."""
        assert verifier.output_path == tmp_path / "hardware_status.json"
        assert verifier.results["overall_status"] == "unknown"
        assert "components" in verifier.results
        assert "timestamp" in verifier.results

    def test_print_summary(self, verifier, capsys):
        """Test summary printing."""
        verifier.results["overall_status"] = "ok"
        verifier.results["components"] = {
            "printers": {
                "status": "ok",
                "printers": [
                    {"name": "test_printer", "host": "localhost", "port": 80, "status": "ok"}
                ],
            },
            "camera": {"status": "ok", "host": "192.168.1.1", "port": 2020},
            "llm_proxy": {"status": "ok", "url": LLM_PROXY_URL},
        }

        verifier.print_summary()
        output = capsys.readouterr().out

        assert "HARDWARE VERIFICATION SUMMARY" in output
        assert "overall" in output.lower() or "Overall" in output
        assert "Printers:" in output
        assert "Camera:" in output
        assert "LLM Proxy:" in output

    @pytest.mark.asyncio
    async def test_verify_llm_proxy_success(self, verifier):
        """Test LLM proxy verification - success case."""
        mock_response = AsyncMock()
        mock_response.status = 200

        mock_get_cm = MagicMock()
        mock_get_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_get_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_get_cm)

        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_cm):
            result = await verifier._verify_llm_proxy()

            assert result["status"] == "ok"
            assert result["http_status"] == 200
            assert result["error"] is None

    @pytest.mark.asyncio
    async def test_verify_llm_proxy_failure(self, verifier):
        """Test LLM proxy verification - failure case."""
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(side_effect=Exception("Connection refused"))

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await verifier._verify_llm_proxy()

            assert result["status"] == "failed"
            assert "Connection refused" in result["error"]

    @pytest.mark.asyncio
    async def test_save_results(self, verifier, tmp_path):
        """Test saving results to JSON file."""
        verifier.results = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "ok",
            "components": {},
        }

        await verifier._save_results()

        assert verifier.output_path.exists()

        with open(verifier.output_path) as f:
            data = json.load(f)
            assert data["overall_status"] == "ok"
            assert "timestamp" in data


class TestPrintMonitor:
    """Tests for PrintMonitor class."""

    @pytest.fixture
    def monitor(self, tmp_path):
        log_file = tmp_path / "telemetry.jsonl"
        snapshot_dir = tmp_path / "snapshots"

        return PrintMonitor(
            printer_host="printer.example",
            printer_port=80,
            camera_url=None,
            log_file=str(log_file),
            snapshot_dir=str(snapshot_dir),
            snapshot_interval=60,
            poll_interval=5,
        )

    def test_init(self, monitor, tmp_path):
        """Test monitor initialization."""
        assert monitor.printer_host == "printer.example"
        assert monitor.printer_port == 80
        assert monitor.camera_url is None
        assert str(monitor.log_file).endswith("telemetry.jsonl")
        assert str(monitor.snapshot_dir).endswith("snapshots")
        assert monitor.snapshot_interval == 60
        assert monitor.poll_interval == 5
        assert monitor._running is False

    def test_telemetry_record_creation(self):
        """Test TelemetryRecord dataclass."""
        record = TelemetryRecord(
            timestamp=datetime.now().isoformat(),
            state="PRINTING",
            filename="test.gcode",
            progress=0.5,
            print_time=3600.0,
            current_layer=10,
            total_layers=100,
            nozzle_temp=200.0,
            nozzle_target=200.0,
            bed_temp=60.0,
            bed_target=60.0,
            snapshot_path=None,
        )

        assert record.state == "PRINTING"
        assert record.progress == 0.5
        assert record.filename == "test.gcode"

    @pytest.mark.asyncio
    async def test_detect_completion(self, monitor):
        """Test completion detection."""
        from sensing_voxel.core.models import NormalizedPrinterState, PrintProgress

        # Was printing, now terminal
        monitor._last_state = NormalizedPrinterState.PRINTING
        progress = PrintProgress(state=NormalizedPrinterState.TERMINAL)

        assert monitor._detect_completion(progress) is True

        # Was printing, still printing
        monitor._last_state = NormalizedPrinterState.PRINTING
        progress = PrintProgress(state=NormalizedPrinterState.PRINTING)

        assert monitor._detect_completion(progress) is False

    @pytest.mark.asyncio
    async def test_detect_failure(self, monitor):
        """Test failure detection."""
        from sensing_voxel.core.models import NormalizedPrinterState, PrintProgress

        # Error state
        progress = PrintProgress(state=NormalizedPrinterState.ERROR)
        assert monitor._detect_failure(progress) is True

        # Non-error state
        progress = PrintProgress(state=NormalizedPrinterState.PRINTING)
        assert monitor._detect_failure(progress) is False

    @pytest.mark.asyncio
    async def test_log_telemetry(self, monitor, tmp_path):
        """Test telemetry logging."""
        from sensing_voxel.core.models import NormalizedPrinterState, PrintProgress

        monitor.log_file.parent.mkdir(parents=True, exist_ok=True)

        progress = PrintProgress(
            state=NormalizedPrinterState.PRINTING,
            filename="test.gcode",
            progress=0.25,
            print_time=900.0,
            current_layer=5,
            total_layers=20,
        )

        await monitor._log_telemetry(progress)

        assert monitor.log_file.exists()

        with open(monitor.log_file) as f:
            line = f.readline()
            record = json.loads(line)
            assert record["state"] == "PRINTING"
            assert record["filename"] == "test.gcode"
            assert record["progress"] == 0.25


class TestScriptHelpers:
    """Tests for script helper functions."""

    def test_default_printers_configuration(self):
        """Test default printers configuration."""
        assert len(DEFAULT_PRINTERS) == 2

        neptune = DEFAULT_PRINTERS[0]
        assert neptune["name"] == "neptune_4_max"
        assert neptune["host"] == "printer.example"
        assert neptune["port"] == 80

        ender = DEFAULT_PRINTERS[1]
        assert ender["name"] == "ender_3"
        assert ender["host"] == "printer.example"
        assert ender["port"] == 80

    def test_llm_proxy_url(self):
        """Test LLM proxy URL constant."""
        assert LLM_PROXY_URL == "https://api.minimaxi.com/v1/models"
        assert LLM_PROXY_URL.startswith("https://")


class TestIntegration:
    """Integration tests for script workflows."""

    @pytest.mark.asyncio
    async def test_verifier_output_format(self, tmp_path):
        """Test verifier produces valid JSON output."""
        output_path = tmp_path / "status.json"
        verifier = HardwareVerifier(output_path=str(output_path))

        # Manually set results (without actual hardware calls)
        verifier.results["timestamp"] = datetime.now().isoformat()
        verifier.results["overall_status"] = "ok"
        verifier.results["components"] = {
            "printers": {"status": "ok", "printers": []},
            "camera": {"status": "ok"},
            "llm_proxy": {"status": "ok"},
        }

        await verifier._save_results()

        # Verify JSON is valid
        with open(output_path) as f:
            data = json.load(f)

        assert "timestamp" in data
        assert "overall_status" in data
        assert "components" in data

        # Verify structure
        components = data["components"]
        assert "printers" in components
        assert "camera" in components
        assert "llm_proxy" in components

    @pytest.mark.asyncio
    async def test_monitor_telemetry_format(self, tmp_path):
        """Test monitor produces valid JSONL output."""
        log_file = tmp_path / "telemetry.jsonl"
        monitor = PrintMonitor(
            printer_host="localhost",
            printer_port=80,
            camera_url=None,
            log_file=str(log_file),
            snapshot_dir=str(tmp_path / "snapshots"),
            snapshot_interval=60,
            poll_interval=5,
        )

        from sensing_voxel.core.models import NormalizedPrinterState, PrintProgress

        monitor.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Log multiple records
        for i in range(3):
            progress = PrintProgress(
                state=NormalizedPrinterState.PRINTING,
                filename=f"print_{i}.gcode",
                progress=i * 0.1,
            )
            await monitor._log_telemetry(progress)

        # Verify JSONL format
        with open(log_file) as f:
            lines = f.readlines()

        assert len(lines) == 3

        for i, line in enumerate(lines):
            record = json.loads(line)
            assert "timestamp" in record
            assert "state" in record
            assert record["filename"] == f"print_{i}.gcode"
