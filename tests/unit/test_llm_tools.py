# pyright: reportMissingImports=false

import asyncio
import base64
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web

from sensing_voxel.adapters.base import LiveSafetySnapshot
from sensing_voxel.config import ShellToolConfig as ConfigShellToolConfig
from sensing_voxel.config import WebSearchToolConfig as ConfigWebSearchToolConfig
from sensing_voxel.core.models import (
    CommandResult,
    CommandResultStatus,
    NormalizedPrinterState,
    PrintProgress,
)
from sensing_voxel.llm import tools as llm_tools
from sensing_voxel.llm.runtime import ChatRequestContext, PendingApprovalStore
from sensing_voxel.llm.tools import (
    CAMERA_TOOLS,
    PRINTER_TOOLS,
    SYSTEM_TOOLS,
    ShellToolConfig,
    ToolExecutor,
    WebSearchToolConfig,
    validate_gcode_command,
)
from sensing_voxel.safety import (
    PhysicalConfirmationSnapshot,
    PrintSafetyService,
    SafetyBinding,
)


def _success(message=None, data=None):
    return CommandResult(status=CommandResultStatus.SUCCESS, message=message, data=data)


def _failure(message="failed"):
    return CommandResult(status=CommandResultStatus.FAILURE, message=message)


@pytest.fixture
def mock_adapter():
    adapter = MagicMock()
    adapter.get_printer_state = AsyncMock(return_value=NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT)
    adapter.is_connected = AsyncMock(return_value=True)
    adapter.get_live_safety_snapshot = AsyncMock(
        side_effect=lambda target: LiveSafetySnapshot(
            target_printer=target,
            probe_z_offset=-1.25,
            homing_origin_z=0.0,
        )
    )
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
    adapter.get_current_position = AsyncMock(return_value={"x": 1.0, "y": 2.0, "z": 3.0})

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
    adapter.list_config_files = AsyncMock(
        return_value=_success(
            data=[
                {"path": "printer.cfg", "size": 1234, "modified": 1700000100},
                {"path": "macros/print_start.cfg", "size": 2345, "modified": 1700000101},
            ]
        )
    )
    adapter.get_config_file = AsyncMock(
        return_value=_success(
            data={
                "filename": "printer.cfg",
                "content": "[printer]\nkinematics: cartesian\n",
                "size_bytes": 31,
                "truncated": False,
            }
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
    camera.get_position = AsyncMock(return_value={"pan": 0.1, "tilt": 0.2})
    return camera


@pytest.fixture
def executor(mock_adapter, mock_camera):
    return ToolExecutor(
        adapter=mock_adapter,
        camera=mock_camera,
        rtsp_url="rtsp://cam/stream",
        shell=ShellToolConfig(enabled=True),
    )


@pytest.fixture
def executor_no_camera(mock_adapter):
    return ToolExecutor(adapter=mock_adapter, shell=ShellToolConfig(enabled=True))


def _load_json(result):
    assert isinstance(result, str)
    return json.loads(result)


async def _approve_token(executor, token, request_context=None):
    result = await executor.execute(
        "approve_pending",
        {"approval_token": token},
        request_context=request_context,
    )
    payload = _load_json(result)
    assert payload["status"] == "approved"
    return payload


class TestPendingApprovalStore:
    def test_payload_is_snapshot_and_transitions_are_exactly_bound(self):
        store = PendingApprovalStore()
        source_payload = {"command": "dangerous", "options": {"timeout": 30}}
        record = store.create("user-a", "dashboard", "run_shell", source_payload, "high risk")
        source_payload["options"]["timeout"] = 1

        decoded = record.payload
        decoded["options"]["timeout"] = 2
        assert store.get(record.approval_token).payload == {
            "command": "dangerous",
            "options": {"timeout": 30},
        }
        assert store.consume(
            record.approval_token, "user-a", "dashboard", "run_shell"
        ) is None
        assert store.approve(
            record.approval_token, "user-b", "dashboard", "run_shell"
        ) is None
        assert store.approve(
            record.approval_token, "user-a", "telegram", "run_shell"
        ) is None
        assert store.approve(
            record.approval_token, "user-a", "dashboard", "cancel_print"
        ) is None

        approved = store.approve(
            record.approval_token, "user-a", "dashboard", "run_shell"
        )
        assert approved is not None
        assert approved.status == "approved"
        assert store.consume(
            record.approval_token, "user-b", "dashboard", "run_shell"
        ) is None
        assert store.consume(
            record.approval_token, "user-a", "telegram", "run_shell"
        ) is None
        assert store.consume(
            record.approval_token, "user-a", "dashboard", "cancel_print"
        ) is None

        consumed = store.consume(
            record.approval_token, "user-a", "dashboard", "run_shell"
        )
        assert consumed is not None
        assert consumed.status == "consumed"
        assert store.consume(
            record.approval_token, "user-a", "dashboard", "run_shell"
        ) is None

    def test_expired_approval_cannot_be_approved_or_consumed(self):
        now = [datetime(2026, 7, 18, tzinfo=UTC)]
        store = PendingApprovalStore(ttl_seconds=10, clock=lambda: now[0])
        record = store.create("user-a", "dashboard", "run_shell", {"command": "id"}, "risk")
        now[0] += timedelta(seconds=10)

        assert store.approve(
            record.approval_token, "user-a", "dashboard", "run_shell"
        ) is None
        assert store.consume(
            record.approval_token, "user-a", "dashboard", "run_shell"
        ) is None
        assert store.get(record.approval_token).status == "expired"

    def test_approved_token_is_atomically_single_use(self):
        store = PendingApprovalStore()
        record = store.create("user-a", "dashboard", "run_shell", {"command": "id"}, "risk")
        assert store.approve(
            record.approval_token, "user-a", "dashboard", "run_shell"
        ) is not None

        def consume_once(_index):
            return store.consume(
                record.approval_token, "user-a", "dashboard", "run_shell"
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(consume_once, range(16)))

        assert sum(result is not None for result in results) == 1
        assert store.get(record.approval_token).status == "consumed"


class TestToolCatalogAndDispatch:

    def test_catalog_includes_expected_tools(self):
        printer_names = {tool["name"] for tool in PRINTER_TOOLS}
        camera_names = {tool["name"] for tool in CAMERA_TOOLS}
        assert "list_printers" in printer_names
        assert "run_gcode" in printer_names
        system_names = {tool["name"] for tool in SYSTEM_TOOLS}
        assert "upload_to_printer" in printer_names
        assert "list_printer_config_files" in printer_names
        assert "get_printer_config_file" in printer_names
        assert "run_bash" not in system_names
        assert "run_shell" in system_names
        assert "web_search" in system_names
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


class TestSystemUtilityTools:
    @pytest.mark.asyncio
    async def test_run_bash_is_not_dispatched(self, executor):
        payload = _load_json(await executor.execute("run_bash", {"command": "pwd"}))

        assert payload == {"error": "Unknown tool: run_bash"}

    @pytest.mark.asyncio
    async def test_run_shell_executes_allowlisted_command(self, tmp_path):
        executor = ToolExecutor(
            shell=ShellToolConfig(enabled=True, cwd=str(tmp_path), timeout_seconds=2)
        )
        payload = _load_json(
            await executor.execute("run_shell", {"command": "pwd", "cwd": str(tmp_path)})
        )

        assert payload["status"] == "success"
        assert payload["exit_code"] == 0
        assert payload["stdout"].strip() == str(tmp_path.resolve())

    def test_executor_accepts_config_dataclasses(self, tmp_path):
        executor = ToolExecutor(
            shell=ConfigShellToolConfig(enabled=True, cwd=str(tmp_path)),
            web_search=ConfigWebSearchToolConfig(
                enabled=True, searxng_url="http://searxng:8080"
            ),
        )

        assert executor._shell_config.enabled is True
        assert executor._shell_config.cwd == str(tmp_path)
        assert executor._web_search_config.enabled is True
        assert executor._web_search_config.searxng_url == "http://searxng:8080"

    @pytest.mark.asyncio
    async def test_update_printer_config_is_disabled_by_default(self, mock_adapter):
        executor = ToolExecutor(adapter=mock_adapter)

        assert "update_printer_config" not in executor._dispatch
        payload = await executor._update_printer_config(
            {"section": "printer", "key": "max_velocity", "value": "500"}
        )

        assert payload == {"error": "Printer configuration updates are disabled by local policy"}

    @pytest.mark.asyncio
    async def test_update_printer_config_is_registered_only_with_opt_in(self, mock_adapter):
        executor = ToolExecutor(
            adapter=mock_adapter,
            shell=ShellToolConfig(allow_printer_config_updates=True),
        )

        assert "update_printer_config" in executor._dispatch
        payload = _load_json(
            await executor.execute(
                "update_printer_config",
                {"section": "printer", "key": "max_velocity", "value": "500"},
            )
        )

        assert payload["status"] == "approval_required"

    @pytest.mark.asyncio
    async def test_web_search_returns_searxng_results(self, aiohttp_client):
        async def handler(request):
            assert request.query["q"] == "sensing voxel"
            return web.json_response(
                {
                    "results": [
                        {
                            "title": "Result A",
                            "url": "https://example.com/a",
                            "content": "Snippet A",
                            "engine": "mock",
                        },
                        {
                            "title": "Result B",
                            "url": "https://example.com/b",
                            "content": "Snippet B",
                        },
                    ]
                }
            )

        app = web.Application()
        app.router.add_get("/search", handler)
        client = await aiohttp_client(app)
        executor = ToolExecutor(
            web_search=WebSearchToolConfig(
                enabled=True,
                searxng_url=str(client.make_url("/")),
                max_results=1,
            )
        )

        payload = _load_json(
            await executor.execute("web_search", {"query": "sensing voxel", "max_results": 5})
        )

        assert payload["query"] == "sensing voxel"
        assert payload["count"] == 1
        assert payload["results"][0]["title"] == "Result A"


class TestPrinterQueryTools:
    @pytest.mark.asyncio
    async def test_get_printer_state(self, executor):
        payload = _load_json(await executor.execute("get_printer_state", {}))
        assert payload == {"printer_id": "default", "state": "IDLE_NO_ACTIVE_PRINT"}

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
        assert payload == {"printer_id": "default", "error": "sensor unavailable"}

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
    async def test_list_printer_config_files_reads_remote_config_root(self, executor):
        payload = _load_json(await executor.execute("list_printer_config_files", {}))

        assert payload["printer_id"] == "default"
        assert payload["count"] == 2
        assert payload["files"][0]["filename"] == "printer.cfg"
        assert "remote Moonraker" in payload["note"]

    @pytest.mark.asyncio
    async def test_get_printer_config_file_reads_remote_content(self, executor, mock_adapter):
        payload = _load_json(
            await executor.execute("get_printer_config_file", {"filename": "printer.cfg"})
        )

        assert payload["printer_id"] == "default"
        assert payload["filename"] == "printer.cfg"
        assert "[printer]" in payload["content"]
        assert payload["truncated"] is False
        assert "not local /app" in payload["note"]
        mock_adapter.get_config_file.assert_awaited_once_with(
            filename="printer.cfg", max_bytes=50000
        )

    @pytest.mark.asyncio
    async def test_config_file_tools_report_unsupported_adapter(self):
        adapter = MagicMock()
        adapter.get_printer_state = AsyncMock(return_value=NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT)
        executor = ToolExecutor(adapter=adapter)

        payload = _load_json(await executor.execute("get_printer_config_file", {}))

        assert payload["printer_id"] == "default"
        assert "does not support remote config file reads" in payload["error"]

    @pytest.mark.asyncio
    async def test_get_system_info(self, executor):
        payload = _load_json(await executor.execute("get_system_info", {}))
        assert payload == {"cpu_load": 0.21, "state": "ready", "printer_id": "default"}

    @pytest.mark.asyncio
    async def test_list_macros(self, executor):
        payload = _load_json(await executor.execute("list_macros", {}))
        assert payload == {"printer_id": "default", "macros": ["PRINT_START", "CANCEL_PRINT"]}

    @pytest.mark.asyncio
    async def test_list_printers_and_target_non_active(self, mock_adapter):
        other = MagicMock()
        other.get_print_progress = AsyncMock(
            return_value=PrintProgress(
                state=NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT,
                filename=None,
                progress=None,
            )
        )
        executor = ToolExecutor(
            adapters={"neptune4max": mock_adapter, "rpi-printer": other},
            active_printer="neptune4max",
        )

        printers = _load_json(await executor.execute("list_printers", {}))
        assert printers["active_printer"] == "neptune4max"
        assert [p["printer_id"] for p in printers["printers"]] == ["neptune4max", "rpi-printer"]

        default_payload = _load_json(await executor.execute("get_print_progress", {}))
        assert default_payload["printer_id"] == "neptune4max"
        mock_adapter.get_print_progress.assert_called_once()

        other_payload = _load_json(
            await executor.execute("get_print_progress", {"printer_id": "rpi-printer"})
        )
        assert other_payload["printer_id"] == "rpi-printer"
        assert other_payload["state"] == "IDLE_NO_ACTIVE_PRINT"
        other.get_print_progress.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_printer_id_returns_clear_error(self, mock_adapter):
        executor = ToolExecutor(adapters={"neptune4max": mock_adapter}, active_printer="neptune4max")
        payload = _load_json(
            await executor.execute("get_print_progress", {"printer_id": "rpi-printer"})
        )
        assert "Unknown printer_id 'rpi-printer'" in payload["error"]
        assert "neptune4max" in payload["error"]


class TestPrinterControlTools:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("tool_name", "tool_input", "adapter_method", "expected_call"),
        [
            ("pause_print", {}, "pause_print", ()),
            ("resume_print", {}, "resume_print", ()),
            ("home_axes", {"axes": "X Y"}, "home_axes", ("X Y",)),
            ("set_fan_speed", {"speed": 200}, "set_fan_speed", (200,)),
            ("set_z_offset", {"offset_mm": -0.03}, "set_z_offset", (-0.03,)),
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
        first = _load_json(
            await executor.execute("run_gcode", {"command": "G1 X100 Y100 Z10 E5 F1200 ; move"})
        )
        token = first["approval_token"]
        await _approve_token(executor, token)
        payload = _load_json(
            await executor.execute(
                "run_gcode",
                {"approval_token": token, "command": "G1 X100 Y100 Z10 E5 F1200 ; move"},
            )
        )
        assert payload["status"] == "SUCCESS"
        assert "warning" in payload
        mock_adapter.run_gcode.assert_called_once_with("G1 X100 Y100 Z10 E5 F1200")

    @pytest.mark.asyncio
    async def test_start_print_requires_approval_then_executes(
        self, mock_adapter, tmp_path, monkeypatch
    ):
        uploads = tmp_path / "data" / "uploads"
        uploads.mkdir(parents=True)
        gcode = (
            b"; filament_type = PLA\n"
            b"; estimated_time = 3600\n"
            b"; filament used [mm] = 1234.5\n"
            b"G28\nM104 S200\nG90\nM83\nG1 X10 Y10 Z0.2 F1200\nG1 X20 E1\n"
        )
        (uploads / "part.gcode").write_bytes(gcode)
        monkeypatch.chdir(tmp_path)

        trusted_binding = SafetyBinding(
            user_id="runtime",
            source_trust="runtime",
            target_printer="default",
            hardware_digest="hardware-v1",
            profile_digest="profile-v1",
            config_digest="config-v1",
            macro_digest="macros-v1",
        )

        async def binding_provider():
            return trusted_binding

        async def confirmation_provider(_artifact, binding):
            return PhysicalConfirmationSnapshot("textured PEI", 0.6, binding.probe_z_offset)

        async def monitor_start(_contract):
            return None

        async def monitor_stop():
            return None

        service = PrintSafetyService(
            mock_adapter,
            binding_provider=binding_provider,
            confirmation_provider=confirmation_provider,
            monitor_start=monitor_start,
            monitor_stop=monitor_stop,
        )
        executor = ToolExecutor(
            adapter=mock_adapter,
            safety_services={"default": service},
        )
        request_context = ChatRequestContext(
            logical_user_id="user-1",
            channel="dashboard",
        )

        first = _load_json(
            await executor.execute(
                "start_print",
                {"filename": "part.gcode"},
                request_context=request_context,
            )
        )
        assert first["status"] == "approval_required"
        approval_payload = first["payload"]
        assert approval_payload["binding"]["source_trust"] == "user_gcode"
        assert approval_payload["binding"]["target_printer"] == "default"
        assert approval_payload["binding"]["probe_z_offset"] == -1.25
        assert approval_payload["binding"]["homing_origin_z"] == 0.0
        assert "live probe Z offset -1.25 mm" in first["confirmation"]
        assert "homing-origin Z 0.0 mm" in first["confirmation"]
        assert approval_payload["artifact_sha256"]
        assert approval_payload["preview_sha256"]
        assert approval_payload["preview_media_type"] == "application/json"
        assert approval_payload["preview"]["artifact_sha256"] == approval_payload[
            "artifact_sha256"
        ]
        assert approval_payload["rendered_preview_sha256"]
        assert approval_payload["rendered_preview_media_type"] == "image/svg+xml"
        assert approval_payload["rendered_preview"].startswith("<?xml ")
        assert "<svg " in approval_payload["rendered_preview"]
        assert approval_payload["validation_sha256"] == approval_payload["validation"][
            "validation_sha256"
        ]
        assert approval_payload["validation"]["artifact_sha256"] == approval_payload[
            "artifact_sha256"
        ]
        assert approval_payload["validation"]["valid"] is True
        confirmation = approval_payload["confirmation"]
        assert confirmation["material"] == "PLA"
        assert confirmation["build_plate_surface"] == "textured PEI"
        assert confirmation["hotend_nozzle_size_mm"] == 0.6
        assert confirmation["estimated_print_time_seconds"] == 3600
        assert confirmation["filament_usage_mm"] == 1234.5
        assert confirmation["z_offset_mm"] == -1.25
        assert confirmation["artifact_sha256"] == approval_payload["artifact_sha256"]
        assert confirmation["preview_sha256"] == approval_payload["preview_sha256"]
        assert confirmation["rendered_preview_sha256"] == approval_payload[
            "rendered_preview_sha256"
        ]
        assert confirmation["confirmation_sha256"] == approval_payload[
            "confirmation_sha256"
        ]

        await _approve_token(executor, first["approval_token"], request_context)
        payload = _load_json(
            await executor.execute(
                "start_print",
                {"approval_token": first["approval_token"], "filename": "part.gcode"},
                request_context=request_context,
            )
        )

        assert payload["status"] == "printing"
        assert payload["artifact_sha256"] == approval_payload["artifact_sha256"]
        mock_adapter.upload_file.assert_awaited_once()
        assert mock_adapter.upload_file.await_args.args[1] == "part.gcode"
        mock_adapter.start_print.assert_awaited_once_with("part.gcode")

    @pytest.mark.asyncio
    async def test_cancel_print_requires_approval_then_executes(self, executor, mock_adapter):
        first = _load_json(await executor.execute("cancel_print", {}))
        assert first["status"] == "approval_required"
        await _approve_token(executor, first["approval_token"])
        payload = _load_json(
            await executor.execute("cancel_print", {"approval_token": first["approval_token"]})
        )
        assert payload["status"] == "SUCCESS"
        mock_adapter.cancel_print.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_file_requires_approval_then_executes(self, executor, mock_adapter):
        first = _load_json(await executor.execute("delete_file", {"filename": "old.gcode"}))
        assert first["status"] == "approval_required"
        await _approve_token(executor, first["approval_token"])
        payload = _load_json(
            await executor.execute(
                "delete_file",
                {"approval_token": first["approval_token"], "filename": "old.gcode"},
            )
        )
        assert payload["status"] == "SUCCESS"
        mock_adapter.delete_file.assert_called_once_with("old.gcode")

    @pytest.mark.asyncio
    async def test_execute_macro_requires_approval_then_executes(self, executor, mock_adapter):
        first = _load_json(await executor.execute("execute_macro", {"name": "BED_MESH_CALIBRATE"}))
        assert first["status"] == "approval_required"
        await _approve_token(executor, first["approval_token"])
        payload = _load_json(
            await executor.execute(
                "execute_macro",
                {"approval_token": first["approval_token"], "name": "BED_MESH_CALIBRATE"},
            )
        )
        assert payload["status"] == "SUCCESS"
        mock_adapter.execute_macro.assert_called_once_with("BED_MESH_CALIBRATE")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "name",
        ["PRINT_START", "start_print BED_TEMP=60", "resume ; ignored", "my_print_start_alias"],
    )
    async def test_execute_macro_blocks_print_starts(self, executor, mock_adapter, name):
        payload = _load_json(await executor.execute("execute_macro", {"name": name}))
        assert "PrintSafetyService" in payload["error"]
        mock_adapter.execute_macro.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_macro_blocks_approved_print_start_payload(self, executor, mock_adapter):
        record = executor._approval_store.create(
            "unknown",
            "unknown",
            "execute_macro",
            {"name": "PRINT_START", "printer_id": "default"},
            "test",
        )
        await _approve_token(executor, record.approval_token)
        payload = _load_json(
            await executor.execute(
                "execute_macro",
                {"approval_token": record.approval_token, "name": "BED_MESH_CALIBRATE"},
            )
        )
        assert "PrintSafetyService" in payload["error"]
        mock_adapter.execute_macro.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_macro_rejects_multiple_commands(self, executor, mock_adapter):
        payload = _load_json(
            await executor.execute("execute_macro", {"name": "BED_MESH_CALIBRATE\nPRINT_START"})
        )
        assert payload["error"] == "Only one macro command is allowed"
        mock_adapter.execute_macro.assert_not_called()

    @pytest.mark.asyncio
    async def test_restart_firmware_requires_approval_then_executes(self, executor, mock_adapter):
        first = _load_json(await executor.execute("restart_firmware", {}))
        assert first["status"] == "approval_required"
        await _approve_token(executor, first["approval_token"])
        payload = _load_json(
            await executor.execute(
                "restart_firmware",
                {"approval_token": first["approval_token"]},
            )
        )
        assert payload["status"] == "SUCCESS"
        mock_adapter.restart_firmware.assert_called_once()

    @pytest.mark.asyncio
    async def test_restart_service_requires_approval_then_executes(self, executor, mock_adapter):
        first = _load_json(await executor.execute("restart_service", {}))
        assert first["status"] == "approval_required"
        await _approve_token(executor, first["approval_token"])
        payload = _load_json(
            await executor.execute(
                "restart_service",
                {"approval_token": first["approval_token"]},
            )
        )
        assert payload["status"] == "SUCCESS"
        mock_adapter.restart_service.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_gcode_query_command_executes_without_approval(self, executor, mock_adapter):
        mock_adapter.run_gcode.return_value = _success(
            message="X:1.000 Y:2.000 Z:3.000",
            data={"response_type": "position", "position": {"x": 1.0, "y": 2.0, "z": 3.0}},
        )
        payload = _load_json(await executor.execute("run_gcode", {"command": "M114"}))
        assert payload["status"] == "SUCCESS"
        assert payload["data"]["response_type"] == "position"
        mock_adapter.run_gcode.assert_called_once_with("M114")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "command",
        ["SDCARD_PRINT_FILE FILENAME=test.gcode", "m23 test.gcode", "M24 ; resume", "print_start"],
    )
    async def test_run_gcode_blocks_print_starts(self, executor, mock_adapter, command):
        payload = _load_json(await executor.execute("run_gcode", {"command": command}))
        assert "PrintSafetyService" in payload["error"]
        mock_adapter.run_gcode.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_gcode_allowlist_cannot_enable_print_start(self, mock_adapter):
        local_executor = ToolExecutor(
            adapter=mock_adapter,
            allowed_gcode_commands={"M24", "PRINT_START"},
        )
        payload = _load_json(await local_executor.execute("run_gcode", {"command": "m24"}))
        assert "PrintSafetyService" in payload["error"]
        mock_adapter.run_gcode.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_gcode_blocks_approved_print_start_payload(self, executor, mock_adapter):
        record = executor._approval_store.create(
            "unknown",
            "unknown",
            "run_gcode",
            {"command": "SDCARD_PRINT_FILE FILENAME=test.gcode", "printer_id": "default"},
            "test",
        )
        await _approve_token(executor, record.approval_token)
        payload = _load_json(
            await executor.execute(
                "run_gcode",
                {"approval_token": record.approval_token, "command": "M114"},
            )
        )
        assert "PrintSafetyService" in payload["error"]
        mock_adapter.run_gcode.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_shell_uses_approved_payload_once(self, executor, monkeypatch):
        process = MagicMock(returncode=0)
        process.communicate = AsyncMock(return_value=(b"approved\n", b""))
        create_process = AsyncMock(return_value=process)
        monkeypatch.setattr(llm_tools.asyncio, "create_subprocess_shell", create_process)

        approved_command = "rm -rf /tmp/nope"
        first = _load_json(
            await executor.execute("run_shell", {"command": approved_command})
        )
        assert first["status"] == "approval_required"
        token = first["approval_token"]
        await _approve_token(executor, token)

        payload = _load_json(
            await executor.execute(
                "run_shell",
                {
                    "approval_token": token,
                    "command": "touch /tmp/caller-override-must-not-run",
                },
            )
        )
        assert payload["status"] == "success"
        assert payload["command"] == approved_command
        create_process.assert_awaited_once()
        assert create_process.await_args.args[0] == approved_command

        reused = _load_json(
            await executor.execute(
                "run_shell",
                {"approval_token": token, "command": approved_command},
            )
        )
        assert reused == {"error": "Invalid or expired approval_token for run_shell"}

    @pytest.mark.asyncio
    async def test_run_shell_background_task_lifecycle(self, executor):
        async def fake_background(task_id, _request_context):
            record = executor._background_tasks.get(task_id)
            record.status = "running"
            record.status = "completed"

        executor._run_shell_background = fake_background
        payload = _load_json(
            await executor.execute(
                "run_shell",
                {
                    "command": "sleep 0.1",
                    "background": True,
                    "task_name": "short sleep",
                    "completion_message": "done",
                },
            )
        )
        assert payload["status"] == "scheduled"
        task_id = payload["task_id"]

        await asyncio.sleep(0)

        tasks_payload = _load_json(await executor.execute("get_background_tasks", {}))
        matching = [task for task in tasks_payload["tasks"] if task["task_id"] == task_id]
        assert matching
        assert matching[0]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_get_pending_approvals_lists_only_current_user(self, executor):
        current = ChatRequestContext(logical_user_id="user-1", channel="dashboard")
        other = ChatRequestContext(logical_user_id="user-2", channel="dashboard")
        own = _load_json(
            await executor.execute("cancel_print", {}, request_context=current)
        )
        await executor.execute("cancel_print", {}, request_context=other)

        payload = _load_json(
            await executor.execute("get_pending_approvals", {}, request_context=current)
        )

        assert [record["approval_token"] for record in payload["approvals"]] == [
            own["approval_token"]
        ]

    @pytest.mark.asyncio
    async def test_approve_pending_rejects_other_identity_and_channel(self, executor):
        owner = ChatRequestContext(logical_user_id="user-1", channel="dashboard")
        wrong_user = ChatRequestContext(logical_user_id="user-2", channel="dashboard")
        wrong_channel = ChatRequestContext(logical_user_id="user-1", channel="telegram")
        first = _load_json(
            await executor.execute("cancel_print", {}, request_context=owner)
        )
        token = first["approval_token"]

        for context in (wrong_user, wrong_channel):
            payload = _load_json(
                await executor.execute(
                    "approve_pending",
                    {"approval_token": token},
                    request_context=context,
                )
            )
            assert payload == {"error": f"Token {token} not found among pending approvals"}

        approved = await _approve_token(executor, token, owner)
        assert approved["tool_name"] == "cancel_print"

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
        assert payload["error"] == "invalid direction"
        assert payload["position_before"]["pan"] == 0.1
        mock_camera.move_step.assert_called_once_with("diagonal")

    @pytest.mark.asyncio
    async def test_camera_move_step_reports_limit_status_from_position_delta(
        self, executor, mock_camera
    ):
        mock_camera.get_position.side_effect = [
            {"pan": 0.1, "tilt": 0.2},
            {"pan": 0.1, "tilt": 0.2},
        ]
        payload = _load_json(await executor.execute("camera_move_step", {"direction": "left"}))
        assert payload["limit_status"] == "reached"
        assert payload["position_before"]["pan"] == 0.1
        assert payload["position_after"]["pan"] == 0.1

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
    async def test_camera_get_position(self, executor):
        payload = _load_json(await executor.execute("camera_get_position", {}))
        assert payload["status"] == "ok"
        assert payload["position"]["pan"] == 0.1

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
