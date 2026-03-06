"""
Unit tests for Bambu Lab printer adapter (MQTT + FTP/FTPS)
"""

import json
from unittest.mock import AsyncMock, MagicMock, Mock, mock_open, patch

import pytest

from sensing_voxel.adapters.bambu import BambuAdapter
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
        adapter._mqtt_client = mock_mqtt
        adapter._mqtt_connected = True
        adapter._connected = True
        return adapter

    # --- Connection Tests ---

    @pytest.mark.asyncio
    async def test_connect_creates_mqtt_client_with_tls(self, adapter):
        """Test that connect creates MQTT client with TLS and credentials"""
        with patch("paho.mqtt.client.Client") as MockClient:
            mock_mqtt = MagicMock()
            MockClient.return_value = mock_mqtt

            # Prevent the reconnect loop from running
            with patch("asyncio.create_task"):
                await adapter.connect()

            # Verify client was created with correct client_id
            MockClient.assert_called_once()
            call_args = MockClient.call_args
            assert f"bambu_adapter_{self.SERIAL}" in str(call_args)

            # Verify credentials set
            mock_mqtt.username_pw_set.assert_called_once_with("bblp", self.ACCESS_CODE)

            # Verify TLS enabled
            mock_mqtt.tls_set.assert_called_once()

            # Verify callbacks assigned
            assert mock_mqtt.on_connect is not None
            assert mock_mqtt.on_disconnect is not None
            assert mock_mqtt.on_message is not None

            assert adapter._connected

    @pytest.mark.asyncio
    async def test_connect_without_tls(self, adapter_no_tls):
        """Test that connect skips TLS when use_tls is False"""
        with patch("paho.mqtt.client.Client") as MockClient:
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

        with patch("paho.mqtt.client.Client") as MockClient:
            mock_mqtt = MagicMock()
            MockClient.return_value = mock_mqtt

            with patch("asyncio.create_task"):
                await adapter.connect()

            mock_mqtt.username_pw_set.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_idempotent(self, adapter):
        """Test that calling connect twice does not create a second client"""
        with patch("paho.mqtt.client.Client") as MockClient:
            mock_mqtt = MagicMock()
            MockClient.return_value = mock_mqtt

            with patch("asyncio.create_task"):
                await adapter.connect()
                await adapter.connect()  # second call should be no-op

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

        await connected_adapter.disconnect()

        mock_ftp.quit.assert_called_once()
        assert connected_adapter._ftp_client is None

    @pytest.mark.asyncio
    async def test_disconnect_ftp_error_handled(self, connected_adapter):
        """Test that FTP quit() error is handled gracefully"""
        mock_ftp = MagicMock()
        mock_ftp.quit.side_effect = Exception("FTP timeout")
        connected_adapter._ftp_client = mock_ftp

        # Should not raise
        await connected_adapter.disconnect()
        assert connected_adapter._ftp_client is None

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self, adapter):
        """Test that disconnect is safe when already disconnected"""
        await adapter.disconnect()  # Should be no-op, no error

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
        """Test that all Bambu status codes 0-13 are mapped"""
        expected_codes = set(range(14))
        assert set(BambuAdapter.STATE_MAP.keys()) == expected_codes

    def test_state_map_idle_codes(self, adapter):
        """Test idle status codes"""
        assert BambuAdapter.STATE_MAP[0] == NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT
        assert BambuAdapter.STATE_MAP[8] == NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT
        assert BambuAdapter.STATE_MAP[9] == NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT

    def test_state_map_printing_codes(self, adapter):
        """Test printing status codes"""
        printing_codes = [1, 3, 4, 10, 12]
        for code in printing_codes:
            assert BambuAdapter.STATE_MAP[code] == NormalizedPrinterState.PRINTING, (
                f"Code {code} should map to PRINTING"
            )

    def test_state_map_paused_codes(self, adapter):
        """Test paused status codes"""
        assert BambuAdapter.STATE_MAP[2] == NormalizedPrinterState.PAUSED
        assert BambuAdapter.STATE_MAP[11] == NormalizedPrinterState.PAUSED

    def test_state_map_terminal_codes(self, adapter):
        """Test terminal (complete/failed/finishing) status codes"""
        terminal_codes = [5, 6, 13]
        for code in terminal_codes:
            assert BambuAdapter.STATE_MAP[code] == NormalizedPrinterState.TERMINAL, (
                f"Code {code} should map to TERMINAL"
            )

    def test_state_map_error_code(self, adapter):
        """Test error status code"""
        assert BambuAdapter.STATE_MAP[7] == NormalizedPrinterState.ERROR

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
        expected_payload = json.dumps({
            "print": {
                "command": "project_file",
                "name": "plate_1.gcode.3mf",
            }
        })

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
                    result = await adapter.upload_file(
                        "/tmp/test.gcode", "test.gcode"
                    )

                assert result.status == CommandResultStatus.SUCCESS

                # Verify FTP_TLS was used (not plain FTP)
                MockFtpTls.assert_called_once()
                mock_ftp.connect.assert_called_once_with("192.168.1.100", 990)
                mock_ftp.login.assert_called_once_with(self.SERIAL, self.ACCESS_CODE)
                mock_ftp.prot_p.assert_called_once()  # Secure data connection
                mock_ftp.storbinary.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_file_plain_ftp(self, adapter_no_tls):
        """Test file upload via plain FTP (no TLS)"""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("ftplib.FTP") as MockFtp:
                mock_ftp = MagicMock()
                MockFtp.return_value = mock_ftp

                with patch("builtins.open", mock_open(read_data=b"gcode data")):
                    result = await adapter_no_tls.upload_file(
                        "/tmp/test.gcode", "test.gcode"
                    )

                assert result.status == CommandResultStatus.SUCCESS

                MockFtp.assert_called_once()
                mock_ftp.connect.assert_called_once_with("192.168.1.100", 21)
                mock_ftp.login.assert_called_once_with(self.SERIAL, self.ACCESS_CODE)
                # prot_p should NOT be called for plain FTP
                mock_ftp.prot_p.assert_not_called()

    @pytest.mark.asyncio
    async def test_upload_file_not_found(self, adapter):
        """Test upload fails when local file does not exist"""
        with patch("pathlib.Path.exists", return_value=False):
            result = await adapter.upload_file(
                "/tmp/nonexistent.gcode", "test.gcode"
            )
            assert result.status == CommandResultStatus.FAILURE
            assert "not found" in result.message.lower()

    @pytest.mark.asyncio
    async def test_upload_file_reuses_ftp_connection(self, adapter):
        """Test that upload reuses existing FTP connection"""
        mock_ftp = MagicMock()
        adapter._ftp_client = mock_ftp

        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=b"data")):
                result = await adapter.upload_file("/tmp/test.gcode", "test.gcode")

        assert result.status == CommandResultStatus.SUCCESS
        # Should not reconnect, just upload
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

    # --- Unsupported Commands ---

    @pytest.mark.asyncio
    async def test_get_z_offset_not_supported(self, adapter):
        """Test get_z_offset raises NotSupportedError"""
        with pytest.raises(NotSupportedError):
            await adapter.get_z_offset()

    @pytest.mark.asyncio
    async def test_set_z_offset_returns_failure(self, adapter):
        """Test set_z_offset returns FAILURE"""
        result = await adapter.set_z_offset(0.1)
        assert result.status == CommandResultStatus.FAILURE
        assert "not supported" in result.message.lower()

    @pytest.mark.asyncio
    async def test_unsupported_commands_return_failure(self, adapter):
        """Test that all MVP-unsupported commands return FAILURE"""
        unsupported_methods = [
            adapter.set_nozzle_temp(200),
            adapter.set_bed_temp(60),
            adapter.home_axes("XY"),
            adapter.set_fan_speed(128),
            adapter.run_gcode("G28"),
            adapter.list_files(),
            adapter.delete_file("test.gcode"),
            adapter.get_file_metadata("test.gcode"),
            adapter.move_axis("X", 10.0),
            adapter.extrude(5.0),
            adapter.set_speed_override(100),
            adapter.set_flow_override(100),
            adapter.list_macros(),
            adapter.execute_macro("TEST"),
            adapter.restart_firmware(),
            adapter.restart_service(),
            adapter.get_system_info(),
            adapter.motors_off(),
        ]

        for coro in unsupported_methods:
            result = await coro
            assert result.status == CommandResultStatus.FAILURE, (
                f"Expected FAILURE for unsupported command, got {result.status}"
            )

    # --- MQTT Callbacks ---

    def test_on_mqtt_connect_success(self, adapter):
        """Test on_connect callback subscribes to topics on success"""
        mock_mqtt = MagicMock()
        adapter._mqtt_client = mock_mqtt

        # Simulate successful connection (reason_code == 0)
        adapter._on_mqtt_connect(mock_mqtt, None, None, 0, None)

        assert adapter._mqtt_connected
        # Should subscribe to status and P2P topics
        assert mock_mqtt.subscribe.call_count == 2
        expected_status_topic = f"device/{self.SERIAL}/report"
        mock_mqtt.subscribe.assert_any_call(expected_status_topic, qos=0)

    def test_on_mqtt_connect_failure(self, adapter):
        """Test on_connect callback does not subscribe on failure"""
        mock_mqtt = MagicMock()
        adapter._mqtt_client = mock_mqtt

        # Simulate failed connection (reason_code != 0)
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

        # Should not raise
        adapter._on_mqtt_message(None, None, mock_msg)

    # --- Report Message Handling ---

    def test_handle_report_message_updates_state(self, adapter):
        """Test that report messages update last known state"""
        payload = {
            "print": {
                "status": 1,  # Printing
                "subtask_name": "benchy.gcode",
                "mc_percent": 50,
            }
        }

        with patch("asyncio.create_task"):
            adapter._handle_report_message(payload)

        assert adapter._last_state == NormalizedPrinterState.PRINTING
        assert adapter._current_filename == "benchy.gcode"
        assert adapter._current_progress == 0.5

    def test_handle_report_message_state_change_emits_event(self, adapter):
        """Test that state change emits state_changed event"""
        adapter._last_state = NormalizedPrinterState.PRINTING

        payload = {
            "print": {
                "status": 2,  # Paused
                "subtask_name": "benchy.gcode",
                "mc_percent": 50,
            }
        }

        with patch("asyncio.create_task") as mock_create_task:
            adapter._handle_report_message(payload)

        assert adapter._last_state == NormalizedPrinterState.PAUSED
        # Should have created tasks for state change and telemetry
        assert mock_create_task.call_count >= 1

    def test_handle_report_message_unknown_status_code(self, adapter):
        """Test that unknown status code maps to UNKNOWN"""
        payload = {
            "print": {
                "status": 999,  # Unknown code
                "subtask_name": "",
                "mc_percent": 0,
            }
        }

        with patch("asyncio.create_task"):
            adapter._handle_report_message(payload)

        assert adapter._last_state == NormalizedPrinterState.UNKNOWN

    # --- Telemetry Parsing ---

    def test_parse_telemetry_full_payload(self, adapter):
        """Test parsing complete telemetry payload"""
        payload = {
            "temps": {
                "bed_temp": 60.0,
                "bed_temp_type": 60.0,
                "nozzle_temp": 210.0,
                "nozzle_temp_type": 215.0,
            },
            "pos": {
                "x": 100.0,
                "y": 150.0,
                "z": 5.2,
            },
            "print": {
                "subtask_name": "benchy.gcode",
                "mc_percent": 75,
                "mc_print_time": 3600,
                "status": 1,
            },
        }

        telemetry = adapter._parse_telemetry(payload)

        assert telemetry is not None
        assert telemetry.temperatures["bed"] == 60.0
        assert telemetry.temperatures["extruder"] == 210.0
        assert telemetry.temperatures["extruder_target"] == 215.0
        assert telemetry.position["x"] == 100.0
        assert telemetry.position["y"] == 150.0
        assert telemetry.position["z"] == 5.2
        assert telemetry.metadata["filename"] == "benchy.gcode"
        assert telemetry.metadata["progress"] == 0.75
        assert telemetry.metadata["print_time"] == 3600

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

    # --- Print Progress ---

    @pytest.mark.asyncio
    async def test_get_print_progress(self, adapter):
        """Test getting print progress from last known state"""
        adapter._last_state = NormalizedPrinterState.PRINTING
        adapter._current_filename = "benchy.gcode"
        adapter._current_progress = 0.42

        progress = await adapter.get_print_progress()

        assert progress.state == NormalizedPrinterState.PRINTING
        assert progress.filename == "benchy.gcode"
        assert progress.progress == 0.42

    @pytest.mark.asyncio
    async def test_get_print_progress_no_state(self, adapter):
        """Test print progress returns UNKNOWN state when no data"""
        progress = await adapter.get_print_progress()

        assert progress.state == NormalizedPrinterState.UNKNOWN
        assert progress.filename == ""
        assert progress.progress == 0.0

    # --- MQTT Topic Construction ---

    def test_control_topic_uses_serial_number(self, connected_adapter):
        """Test that control topic correctly substitutes serial number"""
        expected = f"device/{self.SERIAL}/request"
        actual = BambuAdapter.CONTROL_TOPIC.replace("+", self.SERIAL)
        assert actual == expected

    def test_status_topic_uses_serial_number(self, connected_adapter):
        """Test that status topic correctly substitutes serial number"""
        expected = f"device/{self.SERIAL}/report"
        actual = BambuAdapter.STATUS_TOPIC.replace("+", self.SERIAL)
        assert actual == expected
