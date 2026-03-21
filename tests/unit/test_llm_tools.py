# pyright: reportMissingImports=false

import base64
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from sensing_voxel.core.models import (
    CommandResult,
    CommandResultStatus,
    NormalizedPrinterState,
    PrintProgress,
)
from sensing_voxel.llm import tools as llm_tools
from sensing_voxel.llm.tools import (
    CAMERA_TOOLS,
    PRINTER_TOOLS,
    ToolExecutor,
    validate_gcode_command,
)


def _success(message=None, data=None):
    return CommandResult(status=CommandResultStatus.SUCCESS, message=message, data=data)


def _failure(message="failed"):
    return CommandResult(status=CommandResultStatus.FAILURE, message=message)


@pytest.fixture
def mock_adapter():
    adapter = MagicMock()
    adapter.get_printer_state = AsyncMock(return_value=NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT)
    adapter.get_print_progress = AsyncMock(
        return_value=PrintProgress(
            state=NormalizedPrinterState.PRINTING,
            filename="cube.gcode",
            progress=0.42,
            print_time=600.7,
            nozzle_temp=210.34,
            nozzle_target=215.0,
            bed_temp=59.98,
            bed_target=60.0,
            current_layer=12,
            total_layers=200,
        )
    )

    for name in [
        "start_print",
        "pause_print",
        "resume_print",
        "cancel_print",
        "set_nozzle_temp",
        "set_bed_temp",
        "home_axes",
        "set_fan_speed",
        "run_gcode",
        "set_z_offset",
        "delete_file",
        "move_axis",
        "extrude",
        "motors_off",
        "set_speed_override",
        "set_flow_override",
        "execute_macro",
        "restart_firmware",
        "restart_service",
        "upload_file",
    ]:
        setattr(adapter, name, AsyncMock(return_value=_success()))

    adapter.get_z_offset = AsyncMock(return_value=0.123456)
    adapter.list_files = AsyncMock(
        return_value=_success(
            data=[
                {"path": "prints/a.gcode", "size": 1048576, "modified": 1700000000},
                {"filename": "b.gcode", "size": 2097152, "modified": 1700000001},
            ]
        )
    )
    adapter.get_file_metadata = AsyncMock(
        return_value=_success(data={"layers": 123, "height_mm": 20.5})
    )
    adapter.list_macros = AsyncMock(return_value=_success(data=["PRINT_START", "CANCEL_PRINT"]))
    adapter.get_system_info = AsyncMock(
        return_value=_success(data={"cpu_load": 0.21, "state": "ready"})
    )
    return adapter


@pytest.fixture
def mock_camera():
    camera = MagicMock()
    camera.move_step = AsyncMock(return_value={"status": "ok"})
    camera.move_to = AsyncMock(return_value={"status": "ok"})
    camera.get_presets = AsyncMock(return_value={"presets": [{"id": 1, "name": "bed"}]})
    camera.set_preset = AsyncMock(return_value={"status": "ok"})
    return camera


@pytest.fixture
def executor(mock_adapter, mock_camera):
    return ToolExecutor(adapter=mock_adapter, camera=mock_camera, rtsp_url="rtsp://cam/stream")


@pytest.fixture
def executor_no_camera(mock_adapter):
    return ToolExecutor(adapter=mock_adapter)


def _load_json(result):
    assert isinstance(result, str)
    return json.loads(result)


class TestToolCatalogAndDispatch:
    def test_catalog_includes_expected_tools(self):
        printer_names = {tool["name"] for tool in PRINTER_TOOLS}
        camera_names = {tool["name"] for tool in CAMERA_TOOLS}
        assert "run_gcode" in printer_names
        assert "upload_to_printer" in printer_names
        assert "camera_snapshot" in camera_names
        assert "camera_move_to" in camera_names

    @pytest.mark.asyncio
    async def test_dispatch_unknown_tool(self, executor):
        result = await executor.execute("nope_tool", {})
        payload = _load_json(result)
        assert payload["error"] == "Unknown tool: nope_tool"

    @pytest.mark.asyncio
    async def test_execute_catches_timeout_errors(self, executor, mock_adapter):
        mock_adapter.pause_print.side_effect = TimeoutError("adapter timeout")
        result = await executor.execute("pause_print", {})
        payload = _load_json(result)
        assert "timeout" in payload["error"]

    @pytest.mark.asyncio
    async def test_execute_catches_connection_errors(self, executor, mock_adapter):
        mock_adapter.resume_print.side_effect = RuntimeError("adapter not connected")
        result = await executor.execute("resume_print", {})
        payload = _load_json(result)
        assert payload["error"] == "adapter not connected"

    @pytest.mark.asyncio
    async def test_execute_returns_structured_content_passthrough(self, executor, monkeypatch):
        monkeypatch.setattr(llm_tools.asyncio, "to_thread", AsyncMock(return_value=b"jpeg-bytes"))
        result = await executor.execute("camera_snapshot", {})
        assert isinstance(result, dict)
        assert result["content"][0]["source"]["type"] == "base64"


class TestPrinterQueryTools:
    @pytest.mark.asyncio
    async def test_get_printer_state(self, executor):
        payload = _load_json(await executor.execute("get_printer_state", {}))
        assert payload == {"state": "IDLE_NO_ACTIVE_PRINT"}

    @pytest.mark.asyncio
    async def test_get_print_progress(self, executor):
        payload = _load_json(await executor.execute("get_print_progress", {}))
        assert payload["state"] == "PRINTING"
        assert payload["filename"] == "cube.gcode"
        assert payload["progress_percent"] == 42.0
        assert payload["print_time_seconds"] == 601
        assert payload["current_layer"] == 12
        assert payload["total_layers"] == 200

    @pytest.mark.asyncio
    async def test_get_z_offset(self, executor):
        payload = _load_json(await executor.execute("get_z_offset", {}))
        assert payload["z_offset_mm"] == 0.1235

    @pytest.mark.asyncio
    async def test_get_z_offset_error_path(self, executor, mock_adapter):
        mock_adapter.get_z_offset.side_effect = RuntimeError("sensor unavailable")
        payload = _load_json(await executor.execute("get_z_offset", {}))
        assert payload == {"error": "sensor unavailable"}

    @pytest.mark.asyncio
    async def test_list_files_transforms_data(self, executor):
        payload = _load_json(await executor.execute("list_files", {}))
        assert payload["count"] == 2
        assert payload["files"][0]["filename"] == "prints/a.gcode"
        assert payload["files"][0]["size_mb"] == 1.0
        assert payload["files"][1]["filename"] == "b.gcode"

    @pytest.mark.asyncio
    async def test_list_files_failure_passthrough(self, executor, mock_adapter):
        mock_adapter.list_files.return_value = _failure("cannot list")
        payload = _load_json(await executor.execute("list_files", {}))
        assert payload == {"status": "FAILURE", "message": "cannot list"}

    @pytest.mark.asyncio
    async def test_get_file_metadata(self, executor):
        payload = _load_json(
            await executor.execute("get_file_metadata", {"filename": "cube.gcode"})
        )
        assert payload["layers"] == 123

    @pytest.mark.asyncio
    async def test_get_system_info(self, executor):
        payload = _load_json(await executor.execute("get_system_info", {}))
        assert payload == {"cpu_load": 0.21, "state": "ready"}

    @pytest.mark.asyncio
    async def test_list_macros(self, executor):
        payload = _load_json(await executor.execute("list_macros", {}))
        assert payload == {"macros": ["PRINT_START", "CANCEL_PRINT"]}


class TestPrinterControlTools:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("tool_name", "tool_input", "adapter_method", "expected_call"),
        [
            ("start_print", {"filename": "part.gcode"}, "start_print", ("part.gcode",)),
            ("pause_print", {}, "pause_print", ()),
            ("resume_print", {}, "resume_print", ()),
            ("cancel_print", {}, "cancel_print", ()),
            ("home_axes", {"axes": "X Y"}, "home_axes", ("X Y",)),
            ("set_fan_speed", {"speed": 200}, "set_fan_speed", (200,)),
            ("set_z_offset", {"offset_mm": -0.03}, "set_z_offset", (-0.03,)),
            ("delete_file", {"filename": "old.gcode"}, "delete_file", ("old.gcode",)),
            (
                "move_axis",
                {"axis": "x", "distance": 5.5, "speed": 1234},
                "move_axis",
                ("x", 5.5, 1234),
            ),
            ("extrude", {"length": -4.2, "speed": 123}, "extrude", (-4.2, 123)),
            ("motors_off", {}, "motors_off", ()),
            ("set_speed_override", {"percent": 150}, "set_speed_override", (150,)),
            ("set_flow_override", {"percent": 120}, "set_flow_override", (120,)),
            ("execute_macro", {"name": "PRINT_START"}, "execute_macro", ("PRINT_START",)),
            ("restart_firmware", {}, "restart_firmware", ()),
            ("restart_service", {}, "restart_service", ()),
        ],
    )
    async def test_control_tool_happy_paths(
        self,
        executor,
        mock_adapter,
        tool_name,
        tool_input,
        adapter_method,
        expected_call,
    ):
        payload = _load_json(await executor.execute(tool_name, tool_input))
        assert payload["status"] == "SUCCESS"
        method = getattr(mock_adapter, adapter_method)
        method.assert_called_once_with(*expected_call)

    @pytest.mark.asyncio
    async def test_set_nozzle_temp_bounds_and_message(self, executor, mock_adapter):
        payload = _load_json(await executor.execute("set_nozzle_temp", {"temperature": 215}))
        assert payload["status"] == "SUCCESS"
        assert payload["message"] == "Nozzle temperature set to 215°C"
        mock_adapter.set_nozzle_temp.assert_called_once_with(215.0)

    @pytest.mark.asyncio
    async def test_set_bed_temp_bounds_and_message(self, executor, mock_adapter):
        payload = _load_json(await executor.execute("set_bed_temp", {"temperature": 60.2}))
        assert payload["status"] == "SUCCESS"
        assert payload["message"] == "Bed temperature set to 60°C"
        mock_adapter.set_bed_temp.assert_called_once_with(60.2)

    @pytest.mark.asyncio
    async def test_run_gcode_executes_sanitized_command(self, executor, mock_adapter):
        payload = _load_json(
            await executor.execute("run_gcode", {"command": "G1 X100 Y100 Z10 E5 F1200 ; move"})
        )
        assert payload["status"] == "SUCCESS"
        assert "warning" in payload
        mock_adapter.run_gcode.assert_called_once_with("G1 X100 Y100 Z10 E5 F1200")

    @pytest.mark.asyncio
    async def test_upload_to_printer_uses_sanitized_basename(
        self, executor, mock_adapter, monkeypatch
    ):
        monkeypatch.setattr(llm_tools.Path, "exists", lambda _self: True)
        payload = _load_json(
            await executor.execute(
                "upload_to_printer", {"filename": "../../unsafe/path/part.gcode"}
            )
        )
        assert payload["status"] == "SUCCESS"
        args = mock_adapter.upload_file.call_args[0]
        assert args[1] == "part.gcode"
        assert args[0].endswith("data/uploads/part.gcode")


class TestInputValidationAndBounds:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("tool_name", "tool_input", "error_substring", "adapter_method"),
        [
            ("start_print", {}, "filename is required", "start_print"),
            ("set_nozzle_temp", {}, "temperature is required", "set_nozzle_temp"),
            (
                "set_nozzle_temp",
                {"temperature": 301},
                "Temperature must be 0-300",
                "set_nozzle_temp",
            ),
            (
                "set_nozzle_temp",
                {"temperature": -1},
                "Temperature must be 0-300",
                "set_nozzle_temp",
            ),
            ("set_bed_temp", {}, "temperature is required", "set_bed_temp"),
            ("set_bed_temp", {"temperature": 121}, "Temperature must be 0-120", "set_bed_temp"),
            ("set_bed_temp", {"temperature": -1}, "Temperature must be 0-120", "set_bed_temp"),
            ("set_fan_speed", {}, "speed is required", "set_fan_speed"),
            ("set_fan_speed", {"speed": 300}, "Speed must be 0-255", "set_fan_speed"),
            ("set_fan_speed", {"speed": -1}, "Speed must be 0-255", "set_fan_speed"),
            ("run_gcode", {}, "command is required", "run_gcode"),
            ("set_z_offset", {}, "offset_mm is required", "set_z_offset"),
            ("delete_file", {}, "filename is required", "delete_file"),
            ("get_file_metadata", {}, "filename is required", "get_file_metadata"),
            ("move_axis", {"axis": "X"}, "axis and distance are required", "move_axis"),
            ("move_axis", {"axis": "X", "distance": 201}, "Distance too large", "move_axis"),
            ("extrude", {}, "length is required", "extrude"),
            ("extrude", {"length": 101}, "Length too large", "extrude"),
            ("set_speed_override", {}, "percent is required", "set_speed_override"),
            ("set_speed_override", {"percent": 9}, "Percent must be 10-300", "set_speed_override"),
            (
                "set_speed_override",
                {"percent": 301},
                "Percent must be 10-300",
                "set_speed_override",
            ),
            ("set_flow_override", {}, "percent is required", "set_flow_override"),
            ("set_flow_override", {"percent": 49}, "Percent must be 50-200", "set_flow_override"),
            ("set_flow_override", {"percent": 201}, "Percent must be 50-200", "set_flow_override"),
            ("execute_macro", {}, "name is required", "execute_macro"),
            ("upload_to_printer", {}, "filename is required", "upload_file"),
        ],
    )
    async def test_validation_errors_block_adapter_calls(
        self,
        executor,
        mock_adapter,
        tool_name,
        tool_input,
        error_substring,
        adapter_method,
    ):
        payload = _load_json(await executor.execute(tool_name, tool_input))
        assert error_substring in payload["error"]
        getattr(mock_adapter, adapter_method).assert_not_called()

    @pytest.mark.asyncio
    async def test_upload_to_printer_missing_file(self, executor, monkeypatch):
        monkeypatch.setattr(llm_tools.Path, "exists", lambda _self: False)
        payload = _load_json(
            await executor.execute("upload_to_printer", {"filename": "part.gcode"})
        )
        assert "not found on server" in payload["error"]


class TestGcodeValidationRules:
    @pytest.mark.parametrize("command", ["M112", "M410", "M80", "M81", "M23 test.gcode"])
    def test_gcode_blocks_dangerous_prefixes(self, command):
        error = validate_gcode_command(command)
        assert error is not None
        assert "blocked" in error.lower()

    def test_gcode_blocks_embedded_newlines(self):
        error = validate_gcode_command("G1 X10\nM104 S300")
        assert error is not None
        assert "single line" in error.lower()

    @pytest.mark.parametrize("command", ["M104 S301", "M109 S-1", "M140 S121", "M190 S130"])
    def test_gcode_enforces_temperature_bounds(self, command):
        error = validate_gcode_command(command)
        assert error is not None
        assert "must be" in error

    @pytest.mark.parametrize("command", ["M104", "M109 P10", "M140", "M190 R60"])
    def test_gcode_requires_temperature_s_parameter(self, command):
        error = validate_gcode_command(command)
        assert error is not None
        assert "requires S" in error

    @pytest.mark.parametrize("command", ["G1 X351", "G0 Y351", "G1 Z401", "G1 E101", "G1 E-11"])
    def test_gcode_enforces_axis_bounds(self, command):
        error = validate_gcode_command(command)
        assert error is not None
        assert "must be" in error

    def test_gcode_rejects_invalid_structure(self):
        error = validate_gcode_command("G1 X10 @bad")
        assert error is not None
        assert "invalid g-code structure" in error.lower()

    def test_gcode_allows_valid_line(self):
        assert validate_gcode_command("G1 X100.5 Y100 Z10 E5 F1200 ; move") is None


class TestCameraTools:
    @pytest.mark.asyncio
    async def test_camera_tools_fail_when_camera_not_configured(self, executor_no_camera):
        for tool_name, tool_input in [
            ("camera_move_step", {"direction": "left"}),
            ("camera_move_to", {"pan": 1, "tilt": 2}),
            ("camera_get_presets", {}),
            ("camera_go_to_preset", {"preset_id": 1}),
        ]:
            payload = _load_json(await executor_no_camera.execute(tool_name, tool_input))
            assert payload == {"error": f"Unknown tool: {tool_name}"}

    @pytest.mark.asyncio
    async def test_camera_move_step_passthrough_and_validation(self, executor, mock_camera):
        mock_camera.move_step.return_value = {"error": "invalid direction"}
        payload = _load_json(await executor.execute("camera_move_step", {"direction": "diagonal"}))
        assert payload == {"error": "invalid direction"}
        mock_camera.move_step.assert_called_once_with("diagonal")

    @pytest.mark.asyncio
    async def test_camera_move_to_coerces_int(self, executor, mock_camera):
        payload = _load_json(await executor.execute("camera_move_to", {"pan": "10", "tilt": "-5"}))
        assert payload == {"status": "ok"}
        mock_camera.move_to.assert_called_once_with(10, -5)

    @pytest.mark.asyncio
    async def test_camera_get_presets(self, executor):
        payload = _load_json(await executor.execute("camera_get_presets", {}))
        assert payload["presets"][0]["id"] == 1

    @pytest.mark.asyncio
    async def test_camera_go_to_preset_requires_id(self, executor):
        payload = _load_json(await executor.execute("camera_go_to_preset", {}))
        assert payload == {"error": "preset_id is required"}

    @pytest.mark.asyncio
    async def test_camera_go_to_preset_happy_path(self, executor, mock_camera):
        payload = _load_json(await executor.execute("camera_go_to_preset", {"preset_id": "3"}))
        assert payload == {"status": "ok"}
        mock_camera.set_preset.assert_called_once_with(3)

    @pytest.mark.asyncio
    async def test_camera_snapshot_base64_content(self, executor, monkeypatch):
        jpeg_bytes = b"fake-jpeg-bytes"
        monkeypatch.setattr(llm_tools.asyncio, "to_thread", AsyncMock(return_value=jpeg_bytes))
        result = await executor.execute("camera_snapshot", {})

        assert isinstance(result, dict)
        content = result["content"]
        assert content[0]["type"] == "image"
        source = content[0]["source"]
        assert source["media_type"] == "image/jpeg"
        assert base64.b64decode(source["data"]) == jpeg_bytes
        assert content[1]["type"] == "text"
        assert "Camera snapshot captured at" in content[1]["text"]

    @pytest.mark.asyncio
    async def test_camera_snapshot_capture_failure(self, executor, monkeypatch):
        monkeypatch.setattr(llm_tools.asyncio, "to_thread", AsyncMock(return_value=None))
        payload = _load_json(await executor.execute("camera_snapshot", {}))
        assert payload == {"error": "Failed to capture frame from camera"}

    @pytest.mark.asyncio
    async def test_camera_snapshot_requires_rtsp_url(self, mock_adapter, mock_camera):
        local_executor = ToolExecutor(adapter=mock_adapter, camera=mock_camera)
        payload = _load_json(await local_executor.execute("camera_snapshot", {}))
        assert payload == {"error": "Unknown tool: camera_snapshot"}
