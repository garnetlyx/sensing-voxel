"""
Unit tests for Klipper/Moonraker adapter
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch, mock_open
from sensing_voxel.adapters.klipper import KlipperAdapter
from sensing_voxel.core.models import (
    NormalizedPrinterState,
    CommandResultStatus,
    AdapterConnectionError,
)
import aiohttp


class TestKlipperAdapter:
    """Test Klipper/Moonraker adapter"""

    @pytest.fixture
    def adapter(self):
        """Create adapter instance"""
        return KlipperAdapter(
            host="localhost",
            port=7125,
            api_key="test_key",
        )

    @pytest.mark.asyncio
    async def test_connect(self, adapter):
        """Test adapter connection"""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value = AsyncMock()
            await adapter.connect()

            assert adapter._connected
            assert adapter._session is not None
            mock_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect(self, adapter):
        """Test adapter disconnection"""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_ws = AsyncMock()
            mock_session_obj = AsyncMock()
            mock_session_obj.ws_connect.return_value.__aenter__.return_value = mock_ws
            mock_session.return_value = mock_session_obj

            await adapter.connect()
            await adapter.disconnect()

            assert not adapter._connected
            # Session is closed but reference may remain until garbage collection

    @pytest.mark.asyncio
    async def test_get_printer_state(self, adapter):
        """Test getting printer state"""
        with patch.object(adapter, "_rest_request") as mock_request:
            mock_request.return_value = Mock(
                status=CommandResultStatus.SUCCESS,
                data={"print_stats": {"state": "printing"}},
            )

            state = await adapter.get_printer_state()
            assert state == NormalizedPrinterState.PRINTING

    @pytest.mark.asyncio
    async def test_get_printer_state_unknown(self, adapter):
        """Test getting unknown printer state"""
        with patch.object(adapter, "_rest_request") as mock_request:
            mock_request.return_value = Mock(
                status=CommandResultStatus.SUCCESS,
                data={"print_stats": {"state": "unknown_state"}},
            )

            state = await adapter.get_printer_state()
            assert state == NormalizedPrinterState.UNKNOWN

    @pytest.mark.asyncio
    async def test_start_print(self, adapter):
        """Test starting print"""
        with patch.object(adapter, "_rest_request") as mock_request:
            mock_request.return_value = Mock(
                status=CommandResultStatus.SUCCESS,
                data={},
            )

            result = await adapter.start_print("test.gcode")
            assert result.status == CommandResultStatus.SUCCESS
            mock_request.assert_called_once_with(
                "POST",
                "/printer/print/start",
                json={"filename": "test.gcode"},
            )

    @pytest.mark.asyncio
    async def test_pause_print(self, adapter):
        """Test pausing print"""
        with patch.object(adapter, "_rest_request") as mock_request:
            mock_request.return_value = Mock(
                status=CommandResultStatus.SUCCESS,
                data={},
            )

            result = await adapter.pause_print()
            assert result.status == CommandResultStatus.SUCCESS
            mock_request.assert_called_once_with(
                "POST", "/printer/print/pause"
            )

    @pytest.mark.asyncio
    async def test_resume_print(self, adapter):
        """Test resuming print"""
        with patch.object(adapter, "_rest_request") as mock_request:
            mock_request.return_value = Mock(
                status=CommandResultStatus.SUCCESS,
                data={},
            )

            result = await adapter.resume_print()
            assert result.status == CommandResultStatus.SUCCESS
            mock_request.assert_called_once_with(
                "POST", "/printer/print/resume"
            )

    @pytest.mark.asyncio
    async def test_cancel_print(self, adapter):
        """Test canceling print"""
        with patch.object(adapter, "_rest_request") as mock_request:
            mock_request.return_value = Mock(
                status=CommandResultStatus.SUCCESS,
                data={},
            )

            result = await adapter.cancel_print()
            assert result.status == CommandResultStatus.SUCCESS
            mock_request.assert_called_once_with(
                "POST", "/printer/print/cancel"
            )

    @pytest.mark.asyncio
    async def test_emergency_cancel(self, adapter):
        """Test emergency cancel uses dedicated session with aggressive timeout"""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="OK")

        mock_session_instance = AsyncMock()
        mock_session_instance.post = Mock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=False),
        ))
        mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session_instance.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session_instance):
            result = await adapter.emergency_cancel()
            assert result.status == CommandResultStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_get_z_offset(self, adapter):
        """Test getting Z offset"""
        with patch.object(adapter, "_rest_request") as mock_request:
            mock_request.return_value = Mock(
                status=CommandResultStatus.SUCCESS,
                data={
                    "gcode_move": {
                        "homing_origin": [0.0, 0.0, 0.5]
                    }
                },
            )

            z_offset = await adapter.get_z_offset()
            assert z_offset == 0.5

    @pytest.mark.asyncio
    async def test_set_z_offset(self, adapter):
        """Test setting Z offset"""
        with patch.object(adapter, "_rest_request") as mock_request:
            mock_request.return_value = Mock(
                status=CommandResultStatus.SUCCESS,
                data={},
            )

            result = await adapter.set_z_offset(0.1)
            assert result.status == CommandResultStatus.SUCCESS
            mock_request.assert_called_once_with(
                "POST",
                "/printer/gcode/script",
                json={"script": "SET_GCODE_OFFSET Z_ADJUST=0.1000 MOVE=1"},
            )

    @pytest.mark.asyncio
    async def test_upload_file_success(self, adapter):
        """Test uploading file successfully"""
        with patch.object(adapter, "_rest_request") as mock_request:
            with patch("pathlib.Path.exists") as mock_exists:
                with patch("builtins.open", mock_open(read_data=b"test")):
                    mock_exists.return_value = True
                    mock_request.return_value = Mock(
                        status=CommandResultStatus.SUCCESS,
                        data={},
                    )

                    result = await adapter.upload_file(
                        "/tmp/test.gcode", "test.gcode"
                    )
                    assert result.status == CommandResultStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_upload_file_not_found(self, adapter):
        """Test uploading non-existent file"""
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = False

            result = await adapter.upload_file(
                "/tmp/nonexistent.gcode", "test.gcode"
            )
            assert result.status == CommandResultStatus.FAILURE

    @pytest.mark.asyncio
    async def test_handle_status_update(self, adapter):
        """Test handling status update"""
        # First call - sets initial state, doesn't emit change
        status1 = {
            "print_stats": {"state": "printing"},
            "heater_bed": {"temperature": 60.0, "target": 60.0},
            "extruder": {"temperature": 200.0, "target": 200.0},
            "toolhead": {"position": [10.0, 20.0, 5.0]},
        }

        adapter._emit_state_changed = AsyncMock()
        adapter._emit_telemetry_update = AsyncMock()

        await adapter._handle_status_update(status1)

        # Should emit telemetry but not state change on first call
        adapter._emit_telemetry_update.assert_called_once()
        adapter._emit_state_changed.assert_not_called()

        # Second call with different state - should emit state change
        status2 = {
            "print_stats": {"state": "paused"},
            "heater_bed": {"temperature": 60.0, "target": 60.0},
            "extruder": {"temperature": 200.0, "target": 200.0},
        }

        adapter._emit_state_changed.reset_mock()
        adapter._emit_telemetry_update.reset_mock()

        await adapter._handle_status_update(status2)

        # Should emit both state change and telemetry
        adapter._emit_state_changed.assert_called_once()
        adapter._emit_telemetry_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_parse_telemetry(self, adapter):
        """Test parsing telemetry data"""
        status = {
            "heater_bed": {"temperature": 60.0, "target": 60.0},
            "extruder": {"temperature": 200.0, "target": 200.0},
            "toolhead": {"position": [10.0, 20.0, 5.0]},
        }
        print_stats = {
            "filename": "test.gcode",
            "progress": 0.5,
            "print_duration": 300.0,
            "state": "printing",
        }

        telemetry = adapter._parse_telemetry(status, print_stats)

        assert telemetry is not None
        assert telemetry.temperatures["bed"] == 60.0
        assert telemetry.temperatures["extruder"] == 200.0
        assert telemetry.position["x"] == 10.0
        assert telemetry.metadata["filename"] == "test.gcode"
        assert telemetry.metadata["progress"] == 0.5

    @pytest.mark.asyncio
    async def test_parse_telemetry_minimal(self, adapter):
        """Test parsing minimal telemetry"""
        status = {}
        print_stats = {}

        telemetry = adapter._parse_telemetry(status, print_stats)
        assert telemetry is None

    @pytest.mark.asyncio
    async def test_is_connected(self, adapter):
        """Test connection status check"""
        assert not await adapter.is_connected()

        with patch("aiohttp.ClientSession"):
            await adapter.connect()
            assert await adapter.is_connected()

    @pytest.mark.asyncio
    async def test_rest_request_success(self, adapter):
        """Test successful REST request"""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(
                return_value={"result": {"success": True}}
            )
            mock_session.return_value.request.return_value.__aenter__.return_value = (
                mock_response
            )

            adapter._session = mock_session.return_value
            result = await adapter._rest_request("GET", "/test")

            assert result.status == CommandResultStatus.SUCCESS
            assert result.data == {"success": True}

    @pytest.mark.asyncio
    async def test_rest_request_failure(self, adapter):
        """Test failed REST request"""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.text = AsyncMock(return_value="Internal Server Error")
            mock_session.return_value.request.return_value.__aenter__.return_value = (
                mock_response
            )

            adapter._session = mock_session.return_value
            result = await adapter._rest_request("GET", "/test")

            assert result.status == CommandResultStatus.FAILURE

    @pytest.mark.asyncio
    async def test_rest_request_exception(self, adapter):
        """Test REST request exception"""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value.request.side_effect = Exception(
                "Connection error"
            )

            adapter._session = mock_session.return_value
            result = await adapter._rest_request("GET", "/test")

            assert result.status == CommandResultStatus.FAILURE

    @pytest.mark.asyncio
    async def test_get_print_progress(self, adapter):
        """Test getting print progress"""
        with patch.object(adapter, "_rest_request") as mock_request:
            mock_request.return_value = Mock(
                status=CommandResultStatus.SUCCESS,
                data={
                    "print_stats": {
                        "state": "printing",
                        "filename": "test.gcode",
                        "progress": 0.5,
                        "print_duration": 300.0,
                        "info": {"current_layer": 10, "total_layer": 100},
                    },
                    "heater_bed": {"temperature": 60.0, "target": 60.0},
                    "extruder": {"temperature": 200.0, "target": 200.0},
                },
            )

            progress = await adapter.get_print_progress()

            assert progress.state == NormalizedPrinterState.PRINTING
            assert progress.filename == "test.gcode"
            assert progress.progress == 0.5
            assert progress.print_time == 300.0
            assert progress.current_layer == 10
            assert progress.total_layers == 100
            assert progress.bed_temp == 60.0
            assert progress.nozzle_temp == 200.0

    def test_handle_gcode_output(self, adapter):
        """Test handling G-code output"""
        params = {"message": "G-code output"}
        adapter._handle_gcode_output(params)
        # Should not raise any exceptions

    def test_state_map_completeness(self, adapter):
        """Test that all expected states are mapped"""
        expected_states = {
            "printing",
            "paused",
            "standby",
            "idle",
            "complete",
            "cancelled",
            "error",
            "ready",
            "startup",
            "shutdown",
            "disconnecting",
        }

        assert set(adapter.STATE_MAP.keys()) == expected_states
