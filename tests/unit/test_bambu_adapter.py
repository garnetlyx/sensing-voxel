"""
Unit tests for Bambu Lab printer adapter (MQTT + FTP/FTPS)

Full test coverage for BambuAdapter including:
- Connection management
- State normalization
- Print lifecycle commands
- FTP file operations
- Z-offset control
- Temperature control
- Telemetry parsing
- Error handling
"""

import json
from unittest.mock import AsyncMock, MagicMock, Mock, mock_open, patch

import ftplib
import pytest

from sensing_voxel.adapters.bambu import (
    BambuAdapter,
    BambuFilamentInfo,
    BambuPrintStats,
)
from sensing_voxel.adapters.base import NotSupportedError
from sensing_voxel.core.models import (
    CommandResultStatus,
    NormalizedPrinterState,
)


class TestBambuAdapter:
    """Test Bambu Lab printer adapter"""

    SERIAL = "01P00A000000001"
    ACCESS_CODE = "12345678"

    @pytest.fixture
    def adapter(self):
        """Create adapter instance with typical config"""
        return BambuAdapter(
            host="192.168.1.100",
            mqtt_port=8883,
            ftp_port=990,
            serial_number=self.SERIAL,
            access_code=self.ACCESS_CODE,
            use_tls=True,
        )

    @pytest.fixture
    def adapter_no_tls(self):
        """Create adapter instance without TLS"""
        return BambuAdapter(
            host="192.168.1.100",
            mqtt_port=1883,
            ftp_port=21,
            serial_number=self.SERIAL,
            access_code=self.ACCESS_CODE,
            use_tls=False,
        )

    @pytest.fixture
    def connected_adapter(self, adapter):
        """Create adapter with mocked MQTT client in connected state"""
        mock_mqtt = MagicMock()
        mock_result = MagicMock()
        mock_result.rc = 0
        mock_mqtt.publish.return_value = mock_result
        adapter._mqtt_client = mock_mqtt
        adapter._mqtt_connected = True
        adapter._connected = True
        return adapter

    # --- Connection Tests ---

    @pytest.mark.asyncio
    async def test_connect_creates_mqtt_client_with_tls(self, adapter):
        """Test that connect creates MQTT client with TLS and credentials"""
        with patch("sensing_voxel.adapters.bambu.MqttClient") as MockClient:
            mock_mqtt = MagicMock()
            MockClient.return_value = mock_mqtt

            with patch("asyncio.create_task"):
                await adapter.connect()

            MockClient.assert_called_once()
            call_kwargs = MockClient.call_args[1]
            assert f"bambu_adapter_{self.SERIAL}" in str(call_kwargs)

            mock_mqtt.username_pw_set.assert_called_once_with("bblp", self.ACCESS_CODE)
            mock_mqtt.tls_set.assert_called_once()
            assert mock_mqtt.on_connect is not None
            assert mock_mqtt.on_disconnect is not None
            assert mock_mqtt.on_message is not None
            assert adapter._connected

    @pytest.mark.asyncio
    async def test_connect_without_tls(self, adapter_no_tls):
        """Test that connect skips TLS when use_tls is False"""
        with patch("sensing_voxel.adapters.bambu.MqttClient") as MockClient:
            mock_mqtt = MagicMock()
            MockClient.return_value = mock_mqtt

            with patch("asyncio.create_task"):
                await adapter_no_tls.connect()

            mock_mqtt.tls_set.assert_not_called()
            mock_mqtt.username_pw_set.assert_called_once_with("bblp", self.ACCESS_CODE)

    @pytest.mark.asyncio
    async def test_connect_without_access_code(self):
        """Test that connect skips credentials when no access code"""
        adapter = BambuAdapter(
            host="192.168.1.100",
            serial_number=self.SERIAL,
            access_code="",
        )

        with patch("sensing_voxel.adapters.bambu.MqttClient") as MockClient:
            mock_mqtt = MagicMock()
            MockClient.return_value = mock_mqtt

            with patch("asyncio.create_task"):
                await adapter.connect()

            mock_mqtt.username_pw_set.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_idempotent(self, adapter):
        """Test that calling connect twice does not create a second client"""
        with patch("sensing_voxel.adapters.bambu.MqttClient") as MockClient:
            mock_mqtt = MagicMock()
            MockClient.return_value = mock_mqtt

            with patch("asyncio.create_task"):
                await adapter.connect()
                await adapter.connect()

            MockClient.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect(self, connected_adapter):
        """Test that disconnect stops MQTT loop and disconnects"""
        mock_mqtt = connected_adapter._mqtt_client

        await connected_adapter.disconnect()

        mock_mqtt.loop_stop.assert_called_once()
        mock_mqtt.disconnect.assert_called_once()
        assert not connected_adapter._connected

    @pytest.mark.asyncio
    async def test_disconnect_with_ftp(self, connected_adapter):
        """Test that disconnect also closes FTP connection"""
        mock_ftp = MagicMock()
        connected_adapter._ftp_client = mock_ftp
        connected_adapter._ftp_connected = True

        await connected_adapter.disconnect()

        mock_ftp.quit.assert_called_once()
        assert connected_adapter._ftp_client is None

    @pytest.mark.asyncio
    async def test_disconnect_ftp_error_handled(self, connected_adapter):
        """Test that FTP quit() error is handled gracefully"""
        mock_ftp = MagicMock()
        mock_ftp.quit.side_effect = Exception("FTP timeout")
        connected_adapter._ftp_client = mock_ftp
        connected_adapter._ftp_connected = True

        await connected_adapter.disconnect()
        assert connected_adapter._ftp_client is None

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self, adapter):
        """Test that disconnect is safe when already disconnected"""
        await adapter.disconnect()

    # --- is_connected ---

    @pytest.mark.asyncio
    async def test_is_connected_false_initially(self, adapter):
        """Test that is_connected returns False before connect"""
        assert not await adapter.is_connected()

    @pytest.mark.asyncio
    async def test_is_connected_requires_both_flags(self, adapter):
        """Test that is_connected requires both _connected and _mqtt_connected"""
        adapter._connected = True
        adapter._mqtt_connected = False
        assert not await adapter.is_connected()

        adapter._connected = False
        adapter._mqtt_connected = True
        assert not await adapter.is_connected()

        adapter._connected = True
        adapter._mqtt_connected = True
        assert await adapter.is_connected()

    # --- State Normalization ---

    def test_state_map_completeness(self, adapter):
        """Test that all Bambu status codes 0-18 are mapped"""
        expected_codes = set(range(19))
        assert set(BambuAdapter.STATE_MAP.keys()) == expected_codes

    def test_state_map_idle_codes(self, adapter):
        """Test idle status codes"""
        assert BambuAdapter.STATE_MAP[0] == NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT
        assert BambuAdapter.STATE_MAP[8] == NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT
        assert BambuAdapter.STATE_MAP[9] == NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT
        assert BambuAdapter.STATE_MAP[17] == NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT

    def test_state_map_printing_codes(self, adapter):
        """Test printing status codes"""
        printing_codes = [1, 3, 4, 10, 12, 16, 18]
        for code in printing_codes:
            assert BambuAdapter.STATE_MAP[code] == NormalizedPrinterState.PRINTING, (
                f"Code {code} should map to PRINTING"
            )

    def test_state_map_paused_codes(self, adapter):
        """Test paused status codes"""
        paused_codes = [2, 11, 14]
        for code in paused_codes:
            assert BambuAdapter.STATE_MAP[code] == NormalizedPrinterState.PAUSED, (
                f"Code {code} should map to PAUSED"
            )

    def test_state_map_terminal_codes(self, adapter):
        """Test terminal (complete/failed/finishing) status codes"""
        terminal_codes = [5, 6, 13]
        for code in terminal_codes:
            assert BambuAdapter.STATE_MAP[code] == NormalizedPrinterState.TERMINAL, (
                f"Code {code} should map to TERMINAL"
            )

    def test_state_map_error_codes(self, adapter):
        """Test error status codes"""
        assert BambuAdapter.STATE_MAP[7] == NormalizedPrinterState.ERROR
        assert BambuAdapter.STATE_MAP[15] == NormalizedPrinterState.ERROR

    # --- get_printer_state ---

    @pytest.mark.asyncio
    async def test_get_printer_state_returns_last_known(self, adapter):
        """Test that get_printer_state returns last known state"""
        adapter._last_state = NormalizedPrinterState.PRINTING
        state = await adapter.get_printer_state()
        assert state == NormalizedPrinterState.PRINTING

    @pytest.mark.asyncio
    async def test_get_printer_state_unknown_when_no_state(self, adapter):
        """Test that get_printer_state returns UNKNOWN when no state received"""
        state = await adapter.get_printer_state()
        assert state == NormalizedPrinterState.UNKNOWN

    # --- Print Lifecycle Commands ---

    @pytest.mark.asyncio
    async def test_start_print(self, connected_adapter):
        """Test start_print sends correct MQTT payload"""
        result = await connected_adapter.start_print("plate_1.gcode.3mf")

        assert result.status == CommandResultStatus.SUCCESS
        expected_topic = f"device/{self.SERIAL}/request"
        expected_payload = json.dumps(
            {
                "print": {
                    "command": "project_file",
                    "name": "plate_1.gcode.3mf",
                }
            }
        )

        connected_adapter._mqtt_client.publish.assert_called_once_with(
            expected_topic, expected_payload, qos=1
        )

    @pytest.mark.asyncio
    async def test_pause_print(self, connected_adapter):
        """Test pause_print sends correct MQTT payload"""
        result = await connected_adapter.pause_print()

        assert result.status == CommandResultStatus.SUCCESS
        expected_topic = f"device/{self.SERIAL}/request"
        expected_payload = json.dumps({"print": {"command": "pause"}})

        connected_adapter._mqtt_client.publish.assert_called_once_with(
            expected_topic, expected_payload, qos=1
        )

    @pytest.mark.asyncio
    async def test_resume_print(self, connected_adapter):
        """Test resume_print sends correct MQTT payload"""
        result = await connected_adapter.resume_print()

        assert result.status == CommandResultStatus.SUCCESS
        expected_topic = f"device/{self.SERIAL}/request"
        expected_payload = json.dumps({"print": {"command": "resume"}})

        connected_adapter._mqtt_client.publish.assert_called_once_with(
            expected_topic, expected_payload, qos=1
        )

    @pytest.mark.asyncio
    async def test_cancel_print(self, connected_adapter):
        """Test cancel_print sends correct MQTT payload"""
        result = await connected_adapter.cancel_print()

        assert result.status == CommandResultStatus.SUCCESS
        expected_topic = f"device/{self.SERIAL}/request"
        expected_payload = json.dumps({"print": {"command": "stop"}})

        connected_adapter._mqtt_client.publish.assert_called_once_with(
            expected_topic, expected_payload, qos=1
        )

    @pytest.mark.asyncio
    async def test_emergency_cancel_uses_qos2(self, connected_adapter):
        """Test emergency_cancel publishes with QoS 2 for highest reliability"""
        result = await connected_adapter.emergency_cancel()

        assert result.status == CommandResultStatus.SUCCESS
        expected_topic = f"device/{self.SERIAL}/request"
        expected_payload = json.dumps({"print": {"command": "stop"}})

        connected_adapter._mqtt_client.publish.assert_called_once_with(
            expected_topic, expected_payload, qos=2
        )

    @pytest.mark.asyncio
    async def test_start_print_no_mqtt_client(self, adapter):
        """Test start_print fails gracefully when MQTT client is None"""
        adapter._mqtt_client = None

        result = await adapter.start_print("test.gcode")
        assert result.status == CommandResultStatus.FAILURE
        assert "not connected" in result.message

    @pytest.mark.asyncio
    async def test_pause_print_no_mqtt_client(self, adapter):
        """Test pause_print fails when MQTT client is None"""
        adapter._mqtt_client = None

        result = await adapter.pause_print()
        assert result.status == CommandResultStatus.FAILURE

    @pytest.mark.asyncio
    async def test_resume_print_no_mqtt_client(self, adapter):
        """Test resume_print fails when MQTT client is None"""
        adapter._mqtt_client = None

        result = await adapter.resume_print()
        assert result.status == CommandResultStatus.FAILURE

    @pytest.mark.asyncio
    async def test_cancel_print_no_mqtt_client(self, adapter):
        """Test cancel_print fails when MQTT client is None"""
        adapter._mqtt_client = None

        result = await adapter.cancel_print()
        assert result.status == CommandResultStatus.FAILURE

    @pytest.mark.asyncio
    async def test_emergency_cancel_no_mqtt_client(self, adapter):
        """Test emergency_cancel fails when MQTT client is None"""
        adapter._mqtt_client = None

        result = await adapter.emergency_cancel()
        assert result.status == CommandResultStatus.FAILURE

    # --- FTP Upload ---

    @pytest.mark.asyncio
    async def test_upload_file_tls(self, adapter):
        """Test file upload via FTP_TLS"""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("ftplib.FTP_TLS") as MockFtpTls:
                mock_ftp = MagicMock()
                MockFtpTls.return_value = mock_ftp

                with patch("builtins.open", mock_open(read_data=b"gcode data")):
                    result = await adapter.upload_file("/tmp/test.gcode", "test.gcode")

                assert result.status == CommandResultStatus.SUCCESS
                MockFtpTls.assert_called_once()
                mock_ftp.connect.assert_called_once_with("192.168.1.100", 990)
                mock_ftp.login.assert_called_once_with(self.SERIAL, self.ACCESS_CODE)
                mock_ftp.prot_p.assert_called_once()
                mock_ftp.storbinary.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_file_plain_ftp(self, adapter_no_tls):
        """Test file upload via plain FTP (no TLS)"""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("ftplib.FTP") as MockFtp:
                mock_ftp = MagicMock()
                MockFtp.return_value = mock_ftp

                with patch("builtins.open", mock_open(read_data=b"gcode data")):
                    result = await adapter_no_tls.upload_file("/tmp/test.gcode", "test.gcode")

                assert result.status == CommandResultStatus.SUCCESS
                MockFtp.assert_called_once()
                mock_ftp.connect.assert_called_once_with("192.168.1.100", 21)
                mock_ftp.login.assert_called_once_with(self.SERIAL, self.ACCESS_CODE)
                mock_ftp.prot_p.assert_not_called()

    @pytest.mark.asyncio
    async def test_upload_file_not_found(self, adapter):
        """Test upload fails when local file does not exist"""
        with patch("pathlib.Path.exists", return_value=False):
            result = await adapter.upload_file("/tmp/nonexistent.gcode", "test.gcode")
            assert result.status == CommandResultStatus.FAILURE
            assert "not found" in result.message.lower()

    @pytest.mark.asyncio
    async def test_upload_file_reuses_ftp_connection(self, adapter):
        """Test that upload reuses existing FTP connection"""
        mock_ftp = MagicMock()
        mock_ftp.voidcmd = MagicMock()
        adapter._ftp_client = mock_ftp
        adapter._ftp_connected = True

        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=b"data")):
                result = await adapter.upload_file("/tmp/test.gcode", "test.gcode")

        assert result.status == CommandResultStatus.SUCCESS
        mock_ftp.connect.assert_not_called()
        mock_ftp.storbinary.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_file_ftp_error(self, adapter):
        """Test upload handles FTP errors gracefully"""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("ftplib.FTP_TLS") as MockFtpTls:
                MockFtpTls.return_value.connect.side_effect = Exception("FTP refused")

                result = await adapter.upload_file("/tmp/test.gcode", "test.gcode")
                assert result.status == CommandResultStatus.FAILURE
                assert "Upload failed" in result.message

    # --- File Management ---

    @pytest.mark.asyncio
    async def test_list_files_success(self, adapter):
        """Test listing files via FTP"""
        captured = []

        def mock_dir(callback):
            callback("-rw-rw-rw- 1 user group 123456 Jan 01 12:00 test.gcode")
            callback("-rw-rw-rw- 1 user group 789012 Jan 01 12:01 another.3mf")

        mock_ftp = MagicMock()
        mock_ftp.voidcmd = MagicMock()
        mock_ftp.dir = mock_dir
        adapter._ftp_client = mock_ftp
        adapter._ftp_connected = True

        result = await adapter.list_files()

        assert result.status == CommandResultStatus.SUCCESS
        assert "files" in result.data
        assert len(result.data["files"]) == 2
        assert result.data["files"][0]["name"] == "test.gcode"
        assert result.data["files"][0]["size"] == 123456

    @pytest.mark.asyncio
    async def test_list_files_ftp_error(self, adapter):
        """Test list_files handles FTP errors"""
        with patch("ftplib.FTP_TLS") as MockFtpTls:
            MockFtpTls.return_value.connect.side_effect = Exception("FTP error")

            result = await adapter.list_files()
            assert result.status == CommandResultStatus.FAILURE
            assert "Failed to list files" in result.message

    @pytest.mark.asyncio
    async def test_delete_file_success(self, adapter):
        """Test deleting a file via FTP"""
        mock_ftp = MagicMock()
        mock_ftp.voidcmd = MagicMock()
        adapter._ftp_client = mock_ftp
        adapter._ftp_connected = True

        result = await adapter.delete_file("test.gcode")

        assert result.status == CommandResultStatus.SUCCESS
        mock_ftp.delete.assert_called_once_with("test.gcode")

    @pytest.mark.asyncio
    async def test_delete_file_ftp_error(self, adapter):
        """Test delete_file handles FTP errors"""
        with patch("ftplib.FTP_TLS") as MockFtpTls:
            MockFtpTls.return_value.connect.side_effect = Exception("FTP error")

            result = await adapter.delete_file("test.gcode")
            assert result.status == CommandResultStatus.FAILURE
            assert "Failed to delete file" in result.message

    @pytest.mark.asyncio
    async def test_get_file_metadata_success(self, adapter):
        """Test getting file metadata via FTP"""
        mock_ftp = MagicMock()
        mock_ftp.voidcmd = MagicMock()
        mock_ftp.size = MagicMock(return_value=1024000)

        gcode_header = b"; total layers = 100\n; layer_height = 0.2\n"

        def mock_retrbinary(cmd, callback, blocksize=8192):
            callback(gcode_header)
            raise ftplib.error_temp("Stop reading")

        mock_ftp.retrbinary = mock_retrbinary
        adapter._ftp_client = mock_ftp
        adapter._ftp_connected = True

        result = await adapter.get_file_metadata("test.gcode")

        assert result.status == CommandResultStatus.SUCCESS
        assert result.data["filename"] == "test.gcode"
        assert result.data["size"] == 1024000
        assert result.data["total_layers"] == 100
        assert result.data["layer_height"] == 0.2

    # --- Z-Offset Control ---

    @pytest.mark.asyncio
    async def test_get_z_offset_returns_cached_value(self, adapter):
        """Test get_z_offset returns cached Z offset"""
        adapter._z_offset = 0.05
        result = await adapter.get_z_offset()
        assert result == 0.05

    @pytest.mark.asyncio
    async def test_set_z_offset_success(self, connected_adapter):
        """Test set_z_offset sends correct MQTT command"""
        result = await connected_adapter.set_z_offset(0.1)

        assert result.status == CommandResultStatus.SUCCESS
        expected_topic = f"device/{self.SERIAL}/request"
        expected_payload = json.dumps(
            {
                "print": {
                    "command": "set_z_offset",
                    "z_offset": 0.1,
                }
            }
        )

        connected_adapter._mqtt_client.publish.assert_called_once_with(
            expected_topic, expected_payload, qos=1
        )
        assert connected_adapter._z_offset == 0.1

    @pytest.mark.asyncio
    async def test_set_z_offset_out_of_range(self, adapter):
        """Test set_z_offset validates offset range"""
        result = await adapter.set_z_offset(1.0)  # Too high
        assert result.status == CommandResultStatus.FAILURE
        assert "out of range" in result.message

    @pytest.mark.asyncio
    async def test_set_z_offset_invalid_value(self, adapter):
        """Test set_z_offset validates offset type"""
        result = await adapter.set_z_offset("invalid")
        assert result.status == CommandResultStatus.FAILURE
        assert "Invalid offset value" in result.message

    # --- Temperature Control ---

    @pytest.mark.asyncio
    async def test_set_nozzle_temp_success(self, connected_adapter):
        """Test set_nozzle_temp sends correct MQTT command"""
        result = await connected_adapter.set_nozzle_temp(200)

        assert result.status == CommandResultStatus.SUCCESS
        expected_topic = f"device/{self.SERIAL}/request"
        expected_payload = json.dumps(
            {
                "print": {
                    "command": "set_nozzle_temp",
                    "nozzle_temp": 200,
                }
            }
        )

        connected_adapter._mqtt_client.publish.assert_called_once_with(
            expected_topic, expected_payload, qos=1
        )

    @pytest.mark.asyncio
    async def test_set_nozzle_temp_out_of_range(self, adapter):
        """Test set_nozzle_temp validates temperature range"""
        result = await adapter.set_nozzle_temp(400)  # Too high
        assert result.status == CommandResultStatus.FAILURE
        assert "out of range" in result.message

    @pytest.mark.asyncio
    async def test_set_bed_temp_success(self, connected_adapter):
        """Test set_bed_temp sends correct MQTT command"""
        result = await connected_adapter.set_bed_temp(60)

        assert result.status == CommandResultStatus.SUCCESS
        expected_topic = f"device/{self.SERIAL}/request"
        expected_payload = json.dumps(
            {
                "print": {
                    "command": "set_bed_temp",
                    "bed_temp": 60,
                }
            }
        )

        connected_adapter._mqtt_client.publish.assert_called_once_with(
            expected_topic, expected_payload, qos=1
        )

    @pytest.mark.asyncio
    async def test_set_bed_temp_out_of_range(self, adapter):
        """Test set_bed_temp validates temperature range"""
        result = await adapter.set_bed_temp(150)  # Too high
        assert result.status == CommandResultStatus.FAILURE
        assert "out of range" in result.message

    # --- Motion Control ---

    @pytest.mark.asyncio
    async def test_home_axes_success(self, connected_adapter):
        """Test home_axes sends correct MQTT command"""
        result = await connected_adapter.home_axes()

        assert result.status == CommandResultStatus.SUCCESS
        expected_topic = f"device/{self.SERIAL}/request"
        expected_payload = json.dumps({"print": {"command": "home"}})

        connected_adapter._mqtt_client.publish.assert_called_once_with(
            expected_topic, expected_payload, qos=1
        )

    @pytest.mark.asyncio
    async def test_home_axes_invalid_axis(self, adapter):
        """Test home_axes validates axis parameter"""
        result = await adapter.home_axes("ABC")
        assert result.status == CommandResultStatus.FAILURE
        assert "Invalid axis" in result.message

    @pytest.mark.asyncio
    async def test_move_axis_not_supported(self, adapter):
        """Test move_axis returns not supported"""
        result = await adapter.move_axis("X", 10.0)
        assert result.status == CommandResultStatus.FAILURE
        assert "not supported" in result.message.lower()

    @pytest.mark.asyncio
    async def test_extrude_load(self, connected_adapter):
        """Test extrude with positive length triggers load"""
        result = await connected_adapter.extrude(10.0)

        assert result.status == CommandResultStatus.SUCCESS
        expected_topic = f"device/{self.SERIAL}/request"
        expected_payload = json.dumps({"print": {"command": "load_filament"}})

        connected_adapter._mqtt_client.publish.assert_called_once_with(
            expected_topic, expected_payload, qos=1
        )

    @pytest.mark.asyncio
    async def test_extrude_unload(self, connected_adapter):
        """Test extrude with negative length triggers unload"""
        result = await connected_adapter.extrude(-10.0)

        assert result.status == CommandResultStatus.SUCCESS
        expected_topic = f"device/{self.SERIAL}/request"
        expected_payload = json.dumps({"print": {"command": "unload_filament"}})

        connected_adapter._mqtt_client.publish.assert_called_once_with(
            expected_topic, expected_payload, qos=1
        )

    # --- Fan Control ---

    @pytest.mark.asyncio
    async def test_set_fan_speed_success(self, connected_adapter):
        """Test set_fan_speed sends correct MQTT command"""
        result = await connected_adapter.set_fan_speed(128)

        assert result.status == CommandResultStatus.SUCCESS
        expected_topic = f"device/{self.SERIAL}/request"
        expected_payload = json.dumps(
            {
                "print": {
                    "command": "set_cooling_fan_speed",
                    "fan_speed": 50,  # 128/255 * 100
                }
            }
        )

        connected_adapter._mqtt_client.publish.assert_called_once_with(
            expected_topic, expected_payload, qos=1
        )

    @pytest.mark.asyncio
    async def test_set_fan_speed_out_of_range(self, adapter):
        """Test set_fan_speed validates speed range"""
        result = await adapter.set_fan_speed(300)  # Too high
        assert result.status == CommandResultStatus.FAILURE
        assert "0-255" in result.message

    # --- Speed/Flow Overrides ---

    @pytest.mark.asyncio
    async def test_set_speed_override_success(self, connected_adapter):
        """Test set_speed_override sends correct MQTT command"""
        result = await connected_adapter.set_speed_override(150)

        assert result.status == CommandResultStatus.SUCCESS
        expected_topic = f"device/{self.SERIAL}/request"
        expected_payload = json.dumps(
            {
                "print": {
                    "command": "set_print_speed",
                    "speed": 150,
                }
            }
        )

        connected_adapter._mqtt_client.publish.assert_called_once_with(
            expected_topic, expected_payload, qos=1
        )

    @pytest.mark.asyncio
    async def test_set_speed_override_out_of_range(self, adapter):
        """Test set_speed_override validates percentage range"""
        result = await adapter.set_speed_override(600)  # Too high
        assert result.status == CommandResultStatus.FAILURE
        assert "10-500" in result.message

    @pytest.mark.asyncio
    async def test_set_flow_override_not_supported(self, adapter):
        """Test set_flow_override returns not supported"""
        result = await adapter.set_flow_override(100)
        assert result.status == CommandResultStatus.FAILURE
        assert "not supported" in result.message.lower()

    # --- G-Code Commands ---

    @pytest.mark.asyncio
    async def test_run_gcode_safe_command(self, connected_adapter):
        """Test run_gcode with safe command"""
        result = await connected_adapter.run_gcode("M104 S200")

        assert result.status == CommandResultStatus.SUCCESS
        expected_topic = f"device/{self.SERIAL}/request"
        expected_payload = json.dumps(
            {
                "print": {
                    "command": "gcode_line",
                    "gcode": "M104 S200",
                }
            }
        )

        connected_adapter._mqtt_client.publish.assert_called_once_with(
            expected_topic, expected_payload, qos=1
        )

    @pytest.mark.asyncio
    async def test_run_gcode_unsafe_command(self, adapter):
        """Test run_gcode rejects unsafe commands"""
        result = await adapter.run_gcode("G1 X10 Y10")  # Movement command
        assert result.status == CommandResultStatus.FAILURE
        assert "not in safe list" in result.message

    # --- Macros ---

    @pytest.mark.asyncio
    async def test_list_macros_returns_empty(self, adapter):
        """Test list_macros returns empty list for Bambu"""
        result = await adapter.list_macros()

        assert result.status == CommandResultStatus.SUCCESS
        assert result.data == []
        assert "do not support" in result.message.lower()

    @pytest.mark.asyncio
    async def test_execute_macro_not_supported(self, adapter):
        """Test execute_macro returns failure for Bambu"""
        result = await adapter.execute_macro("TEST_MACRO")

        assert result.status == CommandResultStatus.FAILURE
        assert "do not support" in result.message.lower()

    # --- System Control ---

    @pytest.mark.asyncio
    async def test_restart_firmware_success(self, connected_adapter):
        """Test restart_firmware sends correct MQTT command"""
        result = await connected_adapter.restart_firmware()

        assert result.status == CommandResultStatus.SUCCESS
        expected_topic = f"device/{self.SERIAL}/request"
        expected_payload = json.dumps({"print": {"command": "system"}})

        connected_adapter._mqtt_client.publish.assert_called_once_with(
            expected_topic, expected_payload, qos=1
        )

    @pytest.mark.asyncio
    async def test_restart_service_not_applicable(self, adapter):
        """Test restart_service returns not applicable"""
        result = await adapter.restart_service()

        assert result.status == CommandResultStatus.FAILURE
        assert "not applicable" in result.message.lower()

    @pytest.mark.asyncio
    async def test_get_system_info_returns_cached_data(self, adapter):
        """Test get_system_info returns cached system info"""
        adapter.serial_number = self.SERIAL
        adapter._z_offset = 0.05
        adapter._z_offset_calibrated = True
        adapter._system_info = {"bed_temp": 60, "nozzle_temp": 200}

        result = await adapter.get_system_info()

        assert result.status == CommandResultStatus.SUCCESS
        assert result.data["serial_number"] == self.SERIAL
        assert result.data["z_offset"] == 0.05
        assert result.data["z_offset_calibrated"] is True
        assert result.data["bed_temp"] == 60
        assert result.data["nozzle_temp"] == 200

    @pytest.mark.asyncio
    async def test_motors_off_not_supported(self, adapter):
        """Test motors_off returns not supported"""
        result = await adapter.motors_off()

        assert result.status == CommandResultStatus.FAILURE
        assert "not supported" in result.message.lower()

    # --- MQTT Callbacks ---

    def test_on_mqtt_connect_success(self, adapter):
        """Test on_connect callback subscribes to topics on success"""
        mock_mqtt = MagicMock()
        adapter._mqtt_client = mock_mqtt

        adapter._on_mqtt_connect(mock_mqtt, None, None, 0, None)

        assert adapter._mqtt_connected
        assert mock_mqtt.subscribe.call_count == 2
        expected_status_topic = f"device/{self.SERIAL}/report"
        mock_mqtt.subscribe.assert_any_call(expected_status_topic, qos=0)

    def test_on_mqtt_connect_failure(self, adapter):
        """Test on_connect callback does not subscribe on failure"""
        mock_mqtt = MagicMock()
        adapter._mqtt_client = mock_mqtt

        adapter._on_mqtt_connect(mock_mqtt, None, None, 5, None)

        assert not adapter._mqtt_connected
        mock_mqtt.subscribe.assert_not_called()

    def test_on_mqtt_disconnect(self, adapter):
        """Test on_disconnect callback updates connection state"""
        adapter._mqtt_connected = True

        with patch("asyncio.create_task"):
            adapter._on_mqtt_disconnect(None, None, None, 0, None)

        assert not adapter._mqtt_connected

    def test_on_mqtt_message_report(self, adapter):
        """Test on_message routes report messages to handler"""
        payload = {
            "print": {
                "status": 1,
                "subtask_name": "benchy.gcode",
                "mc_percent": 42,
            }
        }
        mock_msg = MagicMock()
        mock_msg.topic = f"device/{self.SERIAL}/report"
        mock_msg.payload = json.dumps(payload).encode()

        with patch.object(adapter, "_handle_report_message") as mock_handler:
            adapter._on_mqtt_message(None, None, mock_msg)
            mock_handler.assert_called_once_with(payload)

    def test_on_mqtt_message_invalid_json(self, adapter):
        """Test on_message handles invalid JSON gracefully"""
        mock_msg = MagicMock()
        mock_msg.topic = f"device/{self.SERIAL}/report"
        mock_msg.payload = b"not json"

        adapter._on_mqtt_message(None, None, mock_msg)

    def test_on_mqtt_message_p2p(self, adapter):
        """Test on_message handles P2P messages"""
        payload = {"some": "data"}
        mock_msg = MagicMock()
        mock_msg.topic = "p2p/report"
        mock_msg.payload = json.dumps(payload).encode()

        adapter._on_mqtt_message(None, None, mock_msg)

    # --- Report Message Handling ---

    def test_handle_report_message_updates_state(self, adapter):
        """Test that report messages update last known state"""
        payload = {
            "print": {
                "status": 1,
                "subtask_name": "benchy.gcode",
                "mc_percent": 50,
            }
        }

        with patch("asyncio.create_task"):
            adapter._handle_report_message(payload)

        assert adapter._last_state == NormalizedPrinterState.PRINTING
        assert adapter._print_stats.filename == "benchy.gcode"
        assert adapter._print_stats.progress == 0.5

    def test_handle_report_message_state_change_emits_event(self, adapter):
        """Test that state change emits state_changed event"""
        adapter._last_state = NormalizedPrinterState.PRINTING

        payload = {
            "print": {
                "status": 2,
                "subtask_name": "benchy.gcode",
                "mc_percent": 50,
            }
        }

        with patch("asyncio.create_task") as mock_create_task:
            adapter._handle_report_message(payload)

        assert adapter._last_state == NormalizedPrinterState.PAUSED
        assert mock_create_task.call_count >= 1

    def test_handle_report_message_unknown_status_code(self, adapter):
        """Test that unknown status code maps to UNKNOWN"""
        payload = {
            "print": {
                "status": 999,
                "subtask_name": "",
                "mc_percent": 0,
            }
        }

        with patch("asyncio.create_task"):
            adapter._handle_report_message(payload)

        assert adapter._last_state == NormalizedPrinterState.UNKNOWN

    def test_handle_report_message_updates_z_offset(self, adapter):
        """Test that report messages update Z-offset"""
        payload = {
            "print": {
                "status": 1,
                "z_offset": 0.05,
                "bed_leveling_calibrated": True,
            }
        }

        with patch("asyncio.create_task"):
            adapter._handle_report_message(payload)

        assert adapter._z_offset == 0.05
        assert adapter._z_offset_calibrated is True

    def test_handle_report_message_updates_filament_info(self, adapter):
        """Test that report messages update filament info"""
        payload = {
            "print": {
                "status": 1,
                "ams_tray0": {
                    "tray_type": "PLA",
                    "tray_color": "FF0000",
                    "nozzle_temp_min": 190,
                    "nozzle_temp_max": 230,
                    "tray_weight": 500,
                },
            }
        }

        with patch("asyncio.create_task"):
            adapter._handle_report_message(payload)

        assert len(adapter._filament_info) == 1
        assert adapter._filament_info[0].tray_type == "PLA"
        assert adapter._filament_info[0].tray_color == "FF0000"

    def test_handle_report_message_updates_system_info(self, adapter):
        """Test that report messages update system info"""
        payload = {
            "print": {
                "status": 1,
                "bed_temper": 60,
                "nozzle_temper": 200,
                "fan_speed": 128,
                "spd_mag": 150,
            }
        }

        with patch("asyncio.create_task"):
            adapter._handle_report_message(payload)

        assert adapter._system_info["bed_temp"] == 60
        assert adapter._system_info["nozzle_temp"] == 200
        assert adapter._system_info["fan_speed"] == 128
        assert adapter._system_info["print_speed"] == 150

    # --- Telemetry Parsing ---

    def test_parse_telemetry_full_payload(self, adapter):
        """Test parsing complete telemetry payload"""
        payload = {
            "print": {
                "bed_temper": 60.0,
                "bed_target_temper": 60.0,
                "nozzle_temper": 210.0,
                "nozzle_target_temper": 215.0,
                "chamber_temper": 35.0,
                "subtask_name": "benchy.gcode",
                "mc_percent": 75,
                "mc_print_time": 3600,
                "mc_remaining_time": 1200,
                "mc_total_time": 4800,
                "layer_num": 25,
                "total_layer_num": 100,
                "z_offset": 0.05,
                "fan_speed": 128,
                "cooling_fan_speed": 128,
                "big_fan1_speed": 0,
                "big_fan2_speed": 0,
                "spd_lvl": 2,
                "spd_mag": 150,
            }
        }

        telemetry = adapter._parse_telemetry(payload)

        assert telemetry is not None
        assert telemetry.temperatures["bed"] == 60.0
        assert telemetry.temperatures["bed_target"] == 60.0
        assert telemetry.temperatures["extruder"] == 210.0
        assert telemetry.temperatures["extruder_target"] == 215.0
        assert telemetry.temperatures["chamber"] == 35.0
        assert telemetry.metadata["filename"] == "benchy.gcode"
        assert telemetry.metadata["progress"] == 0.75
        assert telemetry.metadata["print_time"] == 3600
        assert telemetry.metadata["remaining_time"] == 1200
        assert telemetry.metadata["total_time"] == 4800
        assert telemetry.metadata["current_layer"] == 25
        assert telemetry.metadata["total_layers"] == 100
        assert telemetry.metadata["z_offset"] == 0.05
        assert telemetry.metadata["fan_speed"] == 128
        assert telemetry.metadata["speed_level"] == 2
        assert telemetry.metadata["print_speed_pct"] == 150

    def test_parse_telemetry_empty_payload(self, adapter):
        """Test parsing empty payload returns None"""
        telemetry = adapter._parse_telemetry({})
        assert telemetry is None

    def test_parse_telemetry_temps_only(self, adapter):
        """Test parsing payload with only temperature data"""
        payload = {
            "temps": {
                "bed_temp": 55.0,
                "bed_temp_type": 60.0,
                "nozzle_temp": 200.0,
                "nozzle_temp_type": 210.0,
            },
        }

        telemetry = adapter._parse_telemetry(payload)

        assert telemetry is not None
        assert telemetry.temperatures["bed"] == 55.0
        assert telemetry.temperatures["extruder"] == 200.0
        assert telemetry.position is None

    def test_parse_telemetry_with_position(self, adapter):
        """Test parsing payload with position data"""
        payload = {
            "pos": {
                "x": 100.0,
                "y": 150.0,
                "z": 5.2,
            },
        }

        telemetry = adapter._parse_telemetry(payload)

        assert telemetry is not None
        assert telemetry.position["x"] == 100.0
        assert telemetry.position["y"] == 150.0
        assert telemetry.position["z"] == 5.2

    def test_parse_telemetry_with_filament(self, adapter):
        """Test parsing telemetry with filament info"""
        adapter._filament_info = [
            BambuFilamentInfo(
                tray_type="PLA",
                tray_color="00FF00",
            )
        ]

        payload = {
            "print": {
                "bed_temper": 60.0,
            }
        }

        telemetry = adapter._parse_telemetry(payload)

        assert telemetry is not None
        assert telemetry.metadata["filament_type"] == "PLA"
        assert telemetry.metadata["filament_color"] == "00FF00"

    # --- Print Progress ---

    @pytest.mark.asyncio
    async def test_get_print_progress(self, adapter):
        """Test getting print progress from last known state"""
        adapter._last_state = NormalizedPrinterState.PRINTING
        adapter._print_stats.filename = "benchy.gcode"
        adapter._print_stats.progress = 0.42
        adapter._print_stats.print_time = 1800
        adapter._print_stats.current_layer = 25
        adapter._print_stats.total_layers = 100
        adapter._system_info["nozzle_temp"] = 210
        adapter._system_info["nozzle_target"] = 215
        adapter._system_info["bed_temp"] = 60
        adapter._system_info["bed_target"] = 60

        progress = await adapter.get_print_progress()

        assert progress.state == NormalizedPrinterState.PRINTING
        assert progress.filename == "benchy.gcode"
        assert progress.progress == 0.42
        assert progress.print_time == 1800
        assert progress.current_layer == 25
        assert progress.total_layers == 100
        assert progress.nozzle_temp == 210
        assert progress.nozzle_target == 215
        assert progress.bed_temp == 60
        assert progress.bed_target == 60

    @pytest.mark.asyncio
    async def test_get_print_progress_no_state(self, adapter):
        """Test print progress returns UNKNOWN state when no data"""
        progress = await adapter.get_print_progress()

        assert progress.state == NormalizedPrinterState.UNKNOWN
        assert progress.filename == ""
        assert progress.progress == 0.0

    @pytest.mark.asyncio
    async def test_get_print_progress_zero_values_filtered(self, adapter):
        """Test that zero values for optional fields are filtered to None"""
        adapter._last_state = NormalizedPrinterState.PRINTING
        adapter._print_stats.current_layer = 0
        adapter._print_stats.total_layers = 0
        adapter._print_stats.print_time = 0

        progress = await adapter.get_print_progress()

        assert progress.current_layer is None
        assert progress.total_layers is None
        assert progress.print_time is None

    # --- Print Stats Dataclass ---

    def test_bambu_print_stats_defaults(self):
        """Test BambuPrintStats default values"""
        stats = BambuPrintStats()
        assert stats.filename == ""
        assert stats.progress == 0.0
        assert stats.print_time == 0
        assert stats.current_layer == 0
        assert stats.total_layers == 0
        assert stats.state == NormalizedPrinterState.UNKNOWN

    def test_bambu_filament_info_defaults(self):
        """Test BambuFilamentInfo default values"""
        info = BambuFilamentInfo()
        assert info.tray_type == ""
        assert info.tray_color == ""
        assert info.nozzle_temp_min == 0
        assert info.nozzle_temp_max == 0
        assert info.tray_weight == 0.0

    # --- MQTT Topic Construction ---

    def test_control_topic_uses_serial_number(self, connected_adapter):
        """Test that control topic correctly formats serial number"""
        expected = f"device/{self.SERIAL}/request"
        actual = BambuAdapter.CONTROL_TOPIC.format(serial=self.SERIAL)
        assert actual == expected

    def test_status_topic_uses_serial_number(self, connected_adapter):
        """Test that status topic correctly formats serial number"""
        expected = f"device/{self.SERIAL}/report"
        actual = BambuAdapter.STATUS_TOPIC.format(serial=self.SERIAL)
        assert actual == expected

    # --- Backoff Calculation ---

    def test_calculate_backoff_delay(self, adapter):
        """Test backoff delay calculation"""
        adapter._reconnect_attempts = 0
        delay0 = adapter._calculate_backoff_delay()
        assert delay0 >= adapter._reconnect_policy.base_delay

        adapter._reconnect_attempts = 5
        delay5 = adapter._calculate_backoff_delay()
        assert delay5 > delay0

        adapter._reconnect_attempts = 100  # Very high
        delay_max = adapter._calculate_backoff_delay()
        assert delay_max <= adapter._reconnect_policy.max_delay * 1.5  # With jitter

    # --- Validation Helpers ---

    def test_validate_temperature_valid(self, adapter):
        """Test temperature validation with valid values"""
        assert adapter._validate_temperature(200) is None
        assert adapter._validate_temperature(0) is None
        assert adapter._validate_temperature(350) is None

    def test_validate_temperature_out_of_range(self, adapter):
        """Test temperature validation with out of range values"""
        assert adapter._validate_temperature(-10, 0, 400) is not None
        assert adapter._validate_temperature(500, 0, 400) is not None

    def test_validate_temperature_invalid_type(self, adapter):
        """Test temperature validation with invalid type"""
        assert adapter._validate_temperature("invalid") is not None
        assert adapter._validate_temperature(None) is not None

    def test_validate_axis_valid(self, adapter):
        """Test axis validation with valid values"""
        assert adapter._validate_axis("") is None  # Home all
        assert adapter._validate_axis("X") is None
        assert adapter._validate_axis("XY") is None
        assert adapter._validate_axis("XYZ") is None

    def test_validate_axis_invalid(self, adapter):
        """Test axis validation with invalid values"""
        assert adapter._validate_axis("ABC") is not None
        assert adapter._validate_axis("W") is not None

    # --- Error Handling ---

    @pytest.mark.asyncio
    async def test_send_mqtt_command_not_connected(self, adapter):
        """Test _send_mqtt_command when not connected"""
        result = await adapter._send_mqtt_command({"test": "command"})
        assert result.status == CommandResultStatus.FAILURE
        assert "not connected" in result.message

    @pytest.mark.asyncio
    async def test_send_mqtt_command_publish_error(self, connected_adapter):
        """Test _send_mqtt_command handles publish errors"""
        from paho.mqtt.client import MQTT_ERR_NO_CONN

        mock_result = MagicMock()
        mock_result.rc = MQTT_ERR_NO_CONN
        connected_adapter._mqtt_client.publish.return_value = mock_result

        result = await connected_adapter._send_mqtt_command({"test": "command"})
        assert result.status == CommandResultStatus.FAILURE

    @pytest.mark.asyncio
    async def test_ensure_ftp_connected_stale_connection(self, adapter):
        """Test _ensure_ftp_connected reconnects on stale connection"""
        mock_ftp = MagicMock()
        mock_ftp.voidcmd.side_effect = Exception("Connection lost")
        adapter._ftp_client = mock_ftp
        adapter._ftp_connected = True

        with patch("ftplib.FTP_TLS") as MockFtpTls:
            new_mock = MagicMock()
            MockFtpTls.return_value = new_mock

            await adapter._ensure_ftp_connected()

            mock_ftp.voidcmd.assert_called_once_with("NOOP")
            assert adapter._ftp_client is new_mock
