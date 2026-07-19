"""
Unit tests for API routes using aiohttp test utilities
"""

import base64
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from aiohttp import web

from sensing_voxel.adapters.base import LiveSafetySnapshot
from sensing_voxel.api.server import create_app, validate_dashboard_security
from sensing_voxel.api.websocket import (
    WS_APPLICATION_PROTOCOL,
    WS_AUTH_PROTOCOL_PREFIX,
)
from sensing_voxel.config import DashboardConfig
from sensing_voxel.core.decision import DecisionEngine
from sensing_voxel.core.models import (
    AgentState,
    CommandResult,
    CommandResultStatus,
    ConsumeResult,
    JobContext,
    NormalizedPrinterState,
)
from sensing_voxel.core.state_machine import AgentStateMachine
from sensing_voxel.safety import (
    PhysicalConfirmationSnapshot,
    PrintSafetyService,
    RequestState,
    SafetyBinding,
)

VALID_GCODE = (
    b"; filament_type = PLA\n"
    b"; estimated_time = 3600\n"
    b"; filament used [mm] = 1234.5\n"
    b"G28\nM104 S200\nG90\nM83\nG1 X10 Y10 Z0.2 F1200\nG1 X20 E1\n"
)


def websocket_auth_protocol(token: str) -> str:
    encoded = base64.urlsafe_b64encode(token.encode()).decode().rstrip("=")
    return f"{WS_AUTH_PROTOCOL_PREFIX}{encoded}"


class MockWebSocketManager:
    """Mock WebSocket manager for testing"""

    def __init__(self):
        self.client_count = 0
        self.broadcast_decision_result = AsyncMock()

    async def handle_websocket(self, request):
        return web.Response(text="WebSocket")


class MockAdapter:
    """Mock printer adapter for testing"""

    def __init__(self, printer_id, printer_type="klipper"):
        self.printer_id = printer_id
        self.printer_type = printer_type
        self.upload_file = AsyncMock(
            return_value=CommandResult(CommandResultStatus.SUCCESS, "uploaded")
        )
        self.start_print = AsyncMock(
            return_value=CommandResult(CommandResultStatus.SUCCESS, "started")
        )
        self.resume_print = AsyncMock(
            return_value=CommandResult(CommandResultStatus.SUCCESS, "resumed")
        )
        self.cancel_print = AsyncMock(
            return_value=CommandResult(CommandResultStatus.SUCCESS, "cancelled")
        )
        self.is_connected = AsyncMock(return_value=True)
        self.get_printer_state = AsyncMock(
            return_value=NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT
        )
        self.get_live_safety_snapshot = AsyncMock(
            side_effect=lambda target: LiveSafetySnapshot(
                target_printer=target,
                probe_z_offset=-1.25,
                homing_origin_z=0.0,
            )
        )


def safety_binding() -> SafetyBinding:
    return SafetyBinding(
        user_id="dashboard-user",
        source_trust="user_gcode",
        target_printer="printer1",
        hardware_digest="hardware-v1",
        profile_digest="profile-v1",
        config_digest="config-v1",
        macro_digest="macros-v1",
    )


def create_safety_service(adapter: MockAdapter) -> PrintSafetyService:
    async def binding_provider() -> SafetyBinding:
        return safety_binding()

    async def confirmation_provider(
        _artifact, binding: SafetyBinding
    ) -> PhysicalConfirmationSnapshot:
        return PhysicalConfirmationSnapshot("textured PEI", 0.6, binding.probe_z_offset)

    async def monitor_start(_contract) -> None:
        return None

    async def monitor_stop() -> None:
        return None

    return PrintSafetyService(
        adapter,
        binding_provider=binding_provider,
        confirmation_provider=confirmation_provider,
        monitor_start=monitor_start,
        monitor_stop=monitor_stop,
    )


class MockDecisionEngine:
    """Mock decision engine for testing"""

    def __init__(self):
        self.consume_decision = AsyncMock(return_value=ConsumeResult.CONSUMED)


class MockChatManager:
    """Mock chat manager for testing"""

    def __init__(self):
        self.chat = AsyncMock(return_value="Test response")
        self.clear_session = MagicMock()
        self.get_history = MagicMock(return_value=[{"role": "user", "content": "Hello"}])
        self.get_context_status = MagicMock(return_value={"model": "test"})
        self.get_audit_records = MagicMock(return_value=[])
        self.get_background_tasks = MagicMock(return_value=[])
        self.get_pending_approvals = MagicMock(return_value=[])


class MockCamera:
    """Mock camera for testing"""

    def __init__(self):
        self.move_step = AsyncMock(return_value={"status": "success"})
        self.set_preset = AsyncMock(return_value={"status": "success"})


def create_test_app(
    state_machine=None,
    decision_engine=None,
    adapters=None,
    active_printer=None,
    chat_manager=None,
    camera=None,
    safety_services=None,
    api_token="test-token",
    auto_auth=True,
    **dashboard_overrides,
):
    """Create a test app with mocked dependencies."""
    config = DashboardConfig(
        enabled=True,
        host="127.0.0.1",
        port=8080,
        api_token=api_token,
        **dashboard_overrides,
    )

    if state_machine is None:
        state_machine = AgentStateMachine()

    if adapters is None:
        adapters = {}

    app, ws_manager = create_app(
        config=config,
        state_machine=state_machine,
        decision_engine=decision_engine,
        adapters=adapters,
        active_printer=active_printer,
        chat_manager=chat_manager,
        camera=camera,
        safety_services=safety_services,
    )

    if auto_auth and api_token:
        @web.middleware
        async def test_auth_middleware(request, handler):
            if "Authorization" not in request.headers:
                headers = request.headers.copy()
                headers["Authorization"] = f"Bearer {api_token}"
                request = request.clone(headers=headers)
            return await handler(request)

        app.middlewares.insert(0, test_auth_middleware)

    return app



@pytest.fixture
def mock_state_machine():
    """Fixture for mocked state machine"""
    sm = MagicMock(spec=AgentStateMachine)
    sm.current_state = AgentState.IDLE
    sm.job_context = None
    sm.external_print = False
    sm.z_offset_dirty = False
    sm.last_error_summary = None
    return sm


@pytest.fixture
def mock_decision_engine():
    """Fixture for mocked decision engine"""
    engine = MagicMock(spec=DecisionEngine)
    engine.consume_decision = AsyncMock(return_value=ConsumeResult.CONSUMED)
    return engine


@pytest.fixture
def mock_ws_manager():
    """Fixture for mocked WebSocket manager"""
    return MockWebSocketManager()


@pytest.mark.asyncio
async def test_health_check(aiohttp_client):
    """Test /health endpoint returns ok status"""
    app = create_test_app()
    client = await aiohttp_client(app)

    resp = await client.get("/health")
    assert resp.status == 200
    data = await resp.json()
    assert data["status"] == "ok"
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_get_printers_empty(aiohttp_client):
    """Test /api/v1/printers returns empty list when no adapters"""
    app = create_test_app()
    client = await aiohttp_client(app)

    resp = await client.get("/api/v1/printers")
    assert resp.status == 200
    data = await resp.json()
    assert data["printers"] == []


@pytest.mark.asyncio
async def test_get_printers_with_adapters(aiohttp_client):
    """Test /api/v1/printers returns printer list with adapters"""
    adapter = MockAdapter("printer1", "klipper")
    adapters = {"printer1": adapter}

    app = create_test_app(adapters=adapters, active_printer="printer1")
    client = await aiohttp_client(app)

    resp = await client.get("/api/v1/printers")
    assert resp.status == 200
    data = await resp.json()
    assert len(data["printers"]) == 1
    assert data["printers"][0]["id"] == "printer1"
    assert data["printers"][0]["type"] == "klipper"
    assert data["printers"][0]["active"] is True


@pytest.mark.asyncio
async def test_get_status(aiohttp_client):
    """Test /api/v1/status returns system status"""
    app = create_test_app()
    client = await aiohttp_client(app)

    resp = await client.get("/api/v1/status")
    assert resp.status == 200
    data = await resp.json()
    assert "agent_state" in data
    assert "has_job" in data
    assert "printer_connected" in data
    assert "dashboard_clients" in data


@pytest.mark.asyncio
async def test_get_job_no_active_job(aiohttp_client):
    """Test /api/v1/job returns null job when no active job"""
    app = create_test_app()
    client = await aiohttp_client(app)

    resp = await client.get("/api/v1/job")
    assert resp.status == 200
    data = await resp.json()
    assert data["job"] is None


@pytest.mark.asyncio
async def test_get_job_with_active_job(aiohttp_client):
    """Test /api/v1/job returns job context when active"""
    sm = AgentStateMachine()
    sm.job_context = JobContext(
        job_id="test-job-123",
        job_file="test.gcode",
        uploaded_at=datetime.now(),
    )

    app = create_test_app(state_machine=sm)
    client = await aiohttp_client(app)

    resp = await client.get("/api/v1/job")
    assert resp.status == 200
    data = await resp.json()
    assert data["job"] is not None
    assert data["job"]["job_id"] == "test-job-123"
    assert data["job"]["job_file"] == "test.gcode"


@pytest.mark.asyncio
async def test_get_telemetry_empty(aiohttp_client):
    """Test /api/v1/telemetry returns empty when no telemetries"""
    app = create_test_app()
    client = await aiohttp_client(app)

    resp = await client.get("/api/v1/telemetry")
    assert resp.status == 200
    data = await resp.json()
    assert data["telemetry"] == {}


@pytest.mark.asyncio
async def test_get_telemetry_by_printer(aiohttp_client):
    """Test /api/v1/telemetry with printer query parameter"""
    app = create_test_app()
    client = await aiohttp_client(app)

    resp = await client.get("/api/v1/telemetry?printer=printer1")
    assert resp.status == 200
    data = await resp.json()
    assert data["printer_id"] == "printer1"
    assert data["telemetry"] is None


@pytest.mark.asyncio
async def test_get_alerts_empty(aiohttp_client):
    """Test /api/v1/alerts returns empty list"""
    app = create_test_app()
    client = await aiohttp_client(app)

    resp = await client.get("/api/v1/alerts")
    assert resp.status == 200
    data = await resp.json()
    assert data["alerts"] == []
    assert data["count"] == 0


@pytest.mark.asyncio
async def test_post_decision_no_engine(aiohttp_client):
    """Test /api/v1/decision returns 503 when no decision engine"""
    app = create_test_app()
    client = await aiohttp_client(app)

    resp = await client.post(
        "/api/v1/decision",
        json={"action": "approve", "token": "test-token"},
    )
    assert resp.status == 503


@pytest.mark.asyncio
async def test_post_decision_invalid_json(aiohttp_client):
    """Test /api/v1/decision returns 400 for invalid JSON"""
    decision_engine = MockDecisionEngine()
    app = create_test_app(decision_engine=decision_engine)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/api/v1/decision",
        data=b"not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status == 400


@pytest.mark.asyncio
async def test_post_decision_missing_fields(aiohttp_client):
    """Test /api/v1/decision returns 400 when missing required fields"""
    decision_engine = MockDecisionEngine()
    app = create_test_app(decision_engine=decision_engine)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/api/v1/decision",
        json={"action": "approve"},
    )
    assert resp.status == 400


@pytest.mark.asyncio
async def test_post_decision_invalid_action(aiohttp_client):
    """Test /api/v1/decision returns 400 for invalid action"""
    sm = AgentStateMachine()
    sm.job_context = JobContext(
        job_id="test-job-123",
        job_file="test.gcode",
        uploaded_at=datetime.now(),
    )
    decision_engine = MockDecisionEngine()
    app = create_test_app(state_machine=sm, decision_engine=decision_engine)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/api/v1/decision",
        json={"action": "invalid_action", "token": "test-token"},
    )
    assert resp.status == 400


@pytest.mark.asyncio
async def test_post_decision_no_active_job(aiohttp_client):
    """Test /api/v1/decision returns 409 when no active job"""
    decision_engine = MockDecisionEngine()
    app = create_test_app(decision_engine=decision_engine)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/api/v1/decision",
        json={"action": "approve", "token": "test-token"},
    )
    assert resp.status == 409


@pytest.mark.asyncio
async def test_post_decision_success(aiohttp_client, tmp_path):
    """Approval uploads retained bytes and starts only through safety service."""
    sm = AgentStateMachine()
    decision_engine = MockDecisionEngine()
    decision_engine.consume_decision.return_value = ConsumeResult.CONSUMED
    adapter = MockAdapter("printer1")
    service = create_safety_service(adapter)
    uploaded_content: list[bytes] = []

    async def capture_upload(local_path: str, remote_path: str) -> CommandResult:
        uploaded_content.append(__import__("pathlib").Path(local_path).read_bytes())
        assert remote_path == "test.gcode"
        return CommandResult(CommandResultStatus.SUCCESS, "uploaded")

    adapter.upload_file.side_effect = capture_upload
    app = create_test_app(
        state_machine=sm,
        decision_engine=decision_engine,
        adapters={"printer1": adapter},
        active_printer="printer1",
        safety_services={"printer1": service},
        upload_dir=str(tmp_path / "uploads"),
    )
    client = await aiohttp_client(app)
    form = aiohttp.FormData()
    form.add_field("file", VALID_GCODE, filename="test.gcode")
    upload_response = await client.post("/api/v1/upload", data=form)
    upload = await upload_response.json()

    resp = await client.post(
        "/api/v1/decision",
        json={
            "action": "approve",
            "token": upload["decision_token"],
            "job_id": upload["job_id"],
        },
    )

    assert upload_response.status == 200
    assert resp.status == 200
    data = await resp.json()
    assert data["result"] == "CONSUMED"
    assert uploaded_content == [VALID_GCODE]
    adapter.start_print.assert_awaited_once_with("test.gcode")
    assert sm.current_state is AgentState.PRINTING


@pytest.mark.asyncio
async def test_post_decision_conflict_result(aiohttp_client, tmp_path):
    """Test /api/v1/decision returns 409 on conflict result."""
    sm = AgentStateMachine()
    decision_engine = MockDecisionEngine()
    decision_engine.consume_decision.return_value = ConsumeResult.INVALID_TOKEN
    adapter = MockAdapter("printer1")
    service = create_safety_service(adapter)
    app = create_test_app(
        state_machine=sm,
        decision_engine=decision_engine,
        adapters={"printer1": adapter},
        active_printer="printer1",
        safety_services={"printer1": service},
        upload_dir=str(tmp_path / "uploads"),
    )
    client = await aiohttp_client(app)
    form = aiohttp.FormData()
    form.add_field("file", VALID_GCODE, filename="test.gcode")
    upload_response = await client.post("/api/v1/upload", data=form)
    upload = await upload_response.json()

    resp = await client.post(
        "/api/v1/decision",
        json={
            "action": "approve",
            "token": "bad-token",
            "job_id": upload["job_id"],
        },
    )
    assert resp.status == 409
    data = await resp.json()
    assert data["result"] == "INVALID_TOKEN"
    adapter.upload_file.assert_not_awaited()
    adapter.start_print.assert_not_awaited()


@pytest.mark.asyncio
async def test_set_active_printer_success(aiohttp_client):
    """Test /api/v1/active-printer sets active printer"""
    adapter = MockAdapter("printer1")
    adapters = {"printer1": adapter}

    app = create_test_app(adapters=adapters)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/api/v1/active-printer",
        json={"printer_id": "printer1"},
    )
    assert resp.status == 200
    data = await resp.json()
    assert data["active_printer"] == "printer1"


@pytest.mark.asyncio
async def test_set_active_printer_invalid_json(aiohttp_client):
    """Test /api/v1/active-printer returns 400 for invalid JSON"""
    app = create_test_app()
    client = await aiohttp_client(app)

    resp = await client.post(
        "/api/v1/active-printer",
        data=b"not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status == 400


@pytest.mark.asyncio
async def test_set_active_printer_missing_id(aiohttp_client):
    """Test /api/v1/active-printer returns 400 when missing printer_id"""
    app = create_test_app()
    client = await aiohttp_client(app)

    resp = await client.post(
        "/api/v1/active-printer",
        json={},
    )
    assert resp.status == 400


@pytest.mark.asyncio
async def test_set_active_printer_not_found(aiohttp_client):
    """Test /api/v1/active-printer returns 404 for unknown printer"""
    app = create_test_app()
    client = await aiohttp_client(app)

    resp = await client.post(
        "/api/v1/active-printer",
        json={"printer_id": "nonexistent"},
    )
    assert resp.status == 404


@pytest.mark.asyncio
async def test_post_chat_no_manager(aiohttp_client):
    """Test /api/v1/chat returns 503 when no chat manager"""
    app = create_test_app()
    client = await aiohttp_client(app)

    resp = await client.post(
        "/api/v1/chat",
        json={"message": "Hello"},
    )
    assert resp.status == 503


@pytest.mark.asyncio
async def test_post_chat_success(aiohttp_client):
    """Test /api/v1/chat returns AI response"""
    chat_manager = MockChatManager()
    app = create_test_app(chat_manager=chat_manager)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/api/v1/chat",
        json={"message": "Hello AI"},
    )
    assert resp.status == 200
    data = await resp.json()
    assert data["response"] == "Test response"
    args = chat_manager.chat.await_args.args
    assert args[0] == "dashboard-user"
    assert args[1] == "Hello AI"
    assert args[2].channel == "dashboard"


@pytest.mark.asyncio
async def test_post_chat_empty_message(aiohttp_client):
    """Test /api/v1/chat returns 400 for empty message"""
    chat_manager = MockChatManager()
    app = create_test_app(chat_manager=chat_manager)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/api/v1/chat",
        json={"message": "   "},
    )
    assert resp.status == 400


@pytest.mark.asyncio
async def test_post_chat_invalid_json(aiohttp_client):
    """Test /api/v1/chat returns 400 for invalid JSON"""
    chat_manager = MockChatManager()
    app = create_test_app(chat_manager=chat_manager)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/api/v1/chat",
        data=b"not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status == 400


@pytest.mark.asyncio
async def test_get_chat_history_no_manager(aiohttp_client):
    """Test /api/v1/chat/history returns empty when no chat manager"""
    app = create_test_app()
    client = await aiohttp_client(app)

    resp = await client.get("/api/v1/chat/history")
    assert resp.status == 200
    data = await resp.json()
    assert data["history"] == []


@pytest.mark.asyncio
async def test_get_chat_history_with_manager(aiohttp_client):
    """Test /api/v1/chat/history returns chat history"""
    chat_manager = MockChatManager()
    app = create_test_app(chat_manager=chat_manager)
    client = await aiohttp_client(app)

    resp = await client.get("/api/v1/chat/history")
    assert resp.status == 200
    data = await resp.json()
    assert len(data["history"]) == 1
    chat_manager.get_history.assert_called_once_with("dashboard-user")


@pytest.mark.asyncio
async def test_post_chat_clear(aiohttp_client):
    """Test /api/v1/chat/clear clears chat session"""
    chat_manager = MockChatManager()
    app = create_test_app(chat_manager=chat_manager)
    client = await aiohttp_client(app)

    resp = await client.post("/api/v1/chat/clear")
    assert resp.status == 200
    data = await resp.json()
    assert data["status"] == "cleared"
    chat_manager.clear_session.assert_called_once_with("dashboard-user")


@pytest.mark.asyncio
async def test_post_camera_move_no_camera(aiohttp_client):
    """Test /api/v1/camera/move returns 503 when no camera"""
    app = create_test_app()
    client = await aiohttp_client(app)

    resp = await client.post(
        "/api/v1/camera/move",
        json={"direction": "left"},
    )
    assert resp.status == 503


@pytest.mark.asyncio
async def test_post_camera_move_success(aiohttp_client):
    """Test /api/v1/camera/move moves camera"""
    camera = MockCamera()
    app = create_test_app(camera=camera)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/api/v1/camera/move",
        json={"direction": "left"},
    )
    assert resp.status == 200
    data = await resp.json()
    assert data["status"] == "success"
    camera.move_step.assert_called_once_with("left")


@pytest.mark.asyncio
async def test_post_camera_move_invalid_direction(aiohttp_client):
    """Test /api/v1/camera/move returns 400 for invalid direction"""
    camera = MockCamera()
    app = create_test_app(camera=camera)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/api/v1/camera/move",
        json={"direction": "invalid"},
    )
    assert resp.status == 400


@pytest.mark.asyncio
async def test_post_camera_preset_no_camera(aiohttp_client):
    """Test /api/v1/camera/preset returns 503 when no camera"""
    app = create_test_app()
    client = await aiohttp_client(app)

    resp = await client.post(
        "/api/v1/camera/preset",
        json={"preset_id": 1},
    )
    assert resp.status == 503


@pytest.mark.asyncio
async def test_post_camera_preset_success(aiohttp_client):
    """Test /api/v1/camera/preset moves to preset"""
    camera = MockCamera()
    app = create_test_app(camera=camera)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/api/v1/camera/preset",
        json={"preset_id": 1},
    )
    assert resp.status == 200
    data = await resp.json()
    assert data["status"] == "success"
    camera.set_preset.assert_called_once_with(1)


@pytest.mark.asyncio
async def test_post_camera_preset_missing_id(aiohttp_client):
    """Test /api/v1/camera/preset returns 400 when missing preset_id"""
    camera = MockCamera()
    app = create_test_app(camera=camera)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/api/v1/camera/preset",
        json={},
    )
    assert resp.status == 400


@pytest.mark.asyncio
async def test_websocket_rejects_missing_auth_before_upgrade(aiohttp_client):
    app = create_test_app(auto_auth=False)
    client = await aiohttp_client(app)

    response = await client.get(
        "/api/v1/ws",
        headers={"Origin": str(client.make_url("/")).rstrip("/")},
    )

    assert response.status == 401
    assert response.headers.get("Upgrade", "").lower() != "websocket"


@pytest.mark.asyncio
async def test_websocket_rejects_disallowed_origin_before_upgrade(aiohttp_client):
    app = create_test_app(auto_auth=False)
    client = await aiohttp_client(app)

    response = await client.get(
        "/api/v1/ws",
        headers={
            "Authorization": "Bearer test-token",
            "Origin": "https://attacker.example",
        },
    )

    assert response.status == 403
    assert response.headers.get("Upgrade", "").lower() != "websocket"


@pytest.mark.asyncio
async def test_websocket_accepts_authenticated_same_origin(aiohttp_client):
    app = create_test_app(auto_auth=False)
    client = await aiohttp_client(app)
    origin = str(client.make_url("/")).rstrip("/")

    websocket = await client.ws_connect(
        "/api/v1/ws",
        headers={"Authorization": "Bearer test-token", "Origin": origin},
    )
    assert not websocket.closed
    assert websocket.protocol is None
    await websocket.close()


@pytest.mark.asyncio
async def test_websocket_accepts_browser_subprotocol_without_echoing_secret(aiohttp_client):
    app = create_test_app(auto_auth=False)
    client = await aiohttp_client(app)
    origin = str(client.make_url("/")).rstrip("/")
    auth_protocol = websocket_auth_protocol("test-token")

    websocket = await client.ws_connect(
        "/api/v1/ws",
        protocols=(WS_APPLICATION_PROTOCOL, auth_protocol),
        headers={"Origin": origin},
    )

    assert not websocket.closed
    assert websocket.protocol == WS_APPLICATION_PROTOCOL
    assert websocket.protocol != auth_protocol
    await websocket.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "protocols",
    [
        (WS_APPLICATION_PROTOCOL,),
        (WS_APPLICATION_PROTOCOL, websocket_auth_protocol("wrong-token")),
        (WS_APPLICATION_PROTOCOL, f"{WS_AUTH_PROTOCOL_PREFIX}%%%"),
        (websocket_auth_protocol("test-token"),),
    ],
)
async def test_websocket_rejects_invalid_browser_subprotocol_credentials(
    aiohttp_client, protocols
):
    app = create_test_app(auto_auth=False)
    client = await aiohttp_client(app)
    origin = str(client.make_url("/")).rstrip("/")

    response = await client.get(
        "/api/v1/ws",
        headers={
            "Origin": origin,
            "Connection": "Upgrade",
            "Upgrade": "websocket",
            "Sec-WebSocket-Version": "13",
            "Sec-WebSocket-Key": base64.b64encode(b"0123456789abcdef").decode(),
            "Sec-WebSocket-Protocol": ", ".join(protocols),
        },
    )

    assert response.status == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert response.headers.get("Sec-WebSocket-Protocol") is None


@pytest.mark.asyncio
async def test_websocket_without_configured_auth_fails_closed(aiohttp_client):
    app = create_test_app(api_token=None, auto_auth=False)
    client = await aiohttp_client(app)
    origin = str(client.make_url("/")).rstrip("/")

    response = await client.get(
        "/api/v1/ws",
        headers={
            "Authorization": "Bearer any-token",
            "Origin": origin,
        },
    )

    assert response.status == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.asyncio
async def test_get_printers_pytest(aiohttp_client):
    """Pytest-style test for printers endpoint with multiple printers"""
    adapter = MockAdapter("printer1", "klipper")
    adapters = {"printer1": adapter, "printer2": MockAdapter("printer2", "bambu")}

    app = create_test_app(adapters=adapters, active_printer="printer1")
    client = await aiohttp_client(app)

    resp = await client.get("/api/v1/printers")
    assert resp.status == 200
    data = await resp.json()
    assert len(data["printers"]) == 2

    p1 = next(p for p in data["printers"] if p["id"] == "printer1")
    assert p1["type"] == "klipper"
    assert p1["active"] is True

    p2 = next(p for p in data["printers"] if p["id"] == "printer2")
    assert p2["type"] == "bambu"
    assert p2["active"] is False


@pytest.mark.asyncio
async def test_get_telemetry_pytest(aiohttp_client):
    """Pytest-style test for telemetry endpoint"""
    app = create_test_app()
    client = await aiohttp_client(app)

    resp = await client.get("/api/v1/telemetry")
    assert resp.status == 200
    data = await resp.json()
    assert "telemetry" in data

    resp = await client.get("/api/v1/telemetry?printer=printer1")
    assert resp.status == 200
    data = await resp.json()
    assert data["printer_id"] == "printer1"
    assert data["telemetry"] is None


@pytest.mark.asyncio
async def test_post_upload(aiohttp_client, tmp_path):
    """Test file upload endpoint"""
    upload_dir = tmp_path / "uploads"
    app = create_test_app(upload_dir=str(upload_dir))
    client = await aiohttp_client(app)

    test_file = tmp_path / "test_upload.stl"
    test_file.write_text("synthetic mesh payload")

    with open(test_file, "rb") as f:
        data = {"file": f}
        resp = await client.post("/api/v1/upload", data=data)

    assert resp.status == 200
    result = await resp.json()
    assert result["status"] == "success"
    assert result["filename"] == "test_upload.stl"
    assert result["size"] == len("synthetic mesh payload")
    assert (upload_dir / result["filename"]).read_text() == "synthetic mesh payload"


@pytest.mark.asyncio
async def test_pending_approvals_preserve_nested_print_evidence(aiohttp_client):
    chat_manager = MockChatManager()
    approval = {
        "approval_token": "synthetic-approval-token",
        "tool_name": "start_print_with_monitor",
        "reason": "Review the validated print evidence.",
        "payload": {
            "print_request": {
                "artifact_sha256": "a" * 64,
                "preview_sha256": "b" * 64,
                "rendered_preview_sha256": "c" * 64,
                "validation_sha256": "d" * 64,
                "confirmation_sha256": "e" * 64,
                "rendered_preview_media_type": "image/svg+xml",
                "rendered_preview": '<svg xmlns="http://www.w3.org/2000/svg"/>',
                "confirmation": {
                    "material": "PLA",
                    "build_plate_surface": "textured PEI",
                    "hotend_nozzle_size_mm": 0.6,
                    "estimated_print_time_seconds": 3600,
                    "filament_usage_mm": 1234.5,
                    "z_offset_mm": -1.25,
                },
                "binding": {
                    "probe_z_offset": -1.25,
                    "homing_origin_z": 0.0,
                },
            }
        },
    }
    chat_manager.get_pending_approvals.return_value = [approval]
    app = create_test_app(chat_manager=chat_manager)
    client = await aiohttp_client(app)

    response = await client.get("/api/v1/pending-approvals?limit=7")

    assert response.status == 200
    assert await response.json() == {"approvals": [approval]}
    chat_manager.get_pending_approvals.assert_called_once_with("dashboard-user", limit=7)


@pytest.mark.asyncio
async def test_gcode_upload_stages_immutable_safety_request(aiohttp_client, tmp_path):
    upload_dir = tmp_path / "uploads"
    adapter = MockAdapter("printer1")
    service = create_safety_service(adapter)
    app = create_test_app(
        adapters={"printer1": adapter},
        active_printer="printer1",
        safety_services={"printer1": service},
        upload_dir=str(upload_dir),
    )
    client = await aiohttp_client(app)
    form = aiohttp.FormData()
    form.add_field("file", VALID_GCODE, filename="safe.gcode")

    response = await client.post("/api/v1/upload", data=form)
    payload = await response.json()

    assert response.status == 200
    assert payload["approval_required"] is True
    assert payload["printer_id"] == "printer1"
    assert payload["binding"] == {
        "target_printer": "printer1",
        "hardware_digest": "hardware-v1",
        "profile_digest": "profile-v1",
        "config_digest": "config-v1",
        "macro_digest": "macros-v1",
        "probe_z_offset": -1.25,
        "homing_origin_z": 0.0,
        "source_trust": "user_gcode",
        "user_id": "dashboard-user",
    }
    assert payload["confirmation"]["material"] == "PLA"
    assert payload["confirmation"]["build_plate_surface"] == "textured PEI"
    assert payload["confirmation"]["hotend_nozzle_size_mm"] == 0.6
    assert payload["confirmation"]["estimated_print_time_seconds"] == 3600
    assert payload["confirmation"]["filament_usage_mm"] == 1234.5
    assert payload["confirmation"]["z_offset_mm"] == -1.25
    assert payload["preview"]["artifact_sha256"] == payload["artifact_sha256"]
    assert payload["preview_media_type"] == "application/json"
    assert payload["rendered_preview_media_type"] == "image/svg+xml"
    assert payload["rendered_preview"].startswith('<?xml version="1.0"')
    assert "<svg" in payload["rendered_preview"]
    print_request = service.get_request(payload["request_id"])
    assert payload["artifact_sha256"] == print_request.artifact.sha256
    assert payload["preview_sha256"] == print_request.preview.sha256
    assert payload["rendered_preview_sha256"] == print_request.preview.rendered_sha256
    assert payload["validation_sha256"] == print_request.validation.result_sha256
    assert payload["confirmation_sha256"] == print_request.confirmation.confirmation_sha256
    assert payload["confirmation"]["artifact_sha256"] == payload["artifact_sha256"]
    assert payload["confirmation"]["preview_sha256"] == payload["preview_sha256"]
    assert (
        payload["confirmation"]["rendered_preview_sha256"]
        == payload["rendered_preview_sha256"]
    )
    assert payload["confirmation"]["confirmation_sha256"] == payload["confirmation_sha256"]
    assert print_request.artifact.content == VALID_GCODE
    assert print_request.state is RequestState.PENDING
    job = app["app_state"]["state_machine"].job_context
    assert job is not None
    for key in (
        "artifact_sha256",
        "preview_sha256",
        "rendered_preview_sha256",
        "validation_sha256",
        "confirmation_sha256",
        "binding",
        "confirmation",
        "preview",
        "preview_media_type",
        "rendered_preview",
        "rendered_preview_media_type",
    ):
        assert job.metadata["safety"][key] == payload[key]
    assert app["app_state"]["state_machine"].current_state is AgentState.AWAITING_APPROVAL


@pytest.mark.asyncio
async def test_gcode_upload_without_active_safety_service_fails_closed(
    aiohttp_client, tmp_path
):
    upload_dir = tmp_path / "uploads"
    adapter = MockAdapter("printer1")
    app = create_test_app(
        adapters={"printer1": adapter},
        active_printer="printer1",
        upload_dir=str(upload_dir),
    )
    client = await aiohttp_client(app)
    form = aiohttp.FormData()
    form.add_field("file", VALID_GCODE, filename="safe.gcode")

    response = await client.post("/api/v1/upload", data=form)

    assert response.status == 503
    assert list(upload_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_gcode_upload_validation_failure_deletes_staged_file(
    aiohttp_client, tmp_path
):
    upload_dir = tmp_path / "uploads"
    adapter = MockAdapter("printer1")
    service = create_safety_service(adapter)
    app = create_test_app(
        adapters={"printer1": adapter},
        active_printer="printer1",
        safety_services={"printer1": service},
        upload_dir=str(upload_dir),
    )
    client = await aiohttp_client(app)
    form = aiohttp.FormData()
    form.add_field("file", b"M104 S200\nG1 X10\n", filename="unsafe.gcode")

    response = await client.post("/api/v1/upload", data=form)

    assert response.status == 400
    assert list(upload_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_dashboard_rejection_rejects_request_and_deletes_staged_file(
    aiohttp_client, tmp_path
):
    sm = AgentStateMachine()
    decision_engine = MockDecisionEngine()
    adapter = MockAdapter("printer1")
    service = create_safety_service(adapter)
    app = create_test_app(
        state_machine=sm,
        decision_engine=decision_engine,
        adapters={"printer1": adapter},
        active_printer="printer1",
        safety_services={"printer1": service},
        upload_dir=str(tmp_path / "uploads"),
    )
    client = await aiohttp_client(app)
    form = aiohttp.FormData()
    form.add_field("file", VALID_GCODE, filename="reject.gcode")
    upload_response = await client.post("/api/v1/upload", data=form)
    upload = await upload_response.json()
    staged_path = __import__("pathlib").Path(upload["path"])

    response = await client.post(
        "/api/v1/decision",
        json={
            "action": "reject",
            "token": upload["decision_token"],
            "job_id": upload["job_id"],
        },
    )

    assert response.status == 200
    assert service.get_request(upload["request_id"]).state is RequestState.REJECTED
    assert not staged_path.exists()
    assert sm.current_state is AgentState.IDLE
    adapter.upload_file.assert_not_awaited()
    adapter.start_print.assert_not_awaited()


@pytest.mark.asyncio
async def test_upload_enforces_actual_streamed_byte_limit_and_cleans_up(aiohttp_client, tmp_path):
    upload_dir = tmp_path / "uploads"
    app = create_test_app(upload_dir=str(upload_dir), max_upload_bytes=4)
    client = await aiohttp_client(app)

    form = aiohttp.FormData()
    form.add_field(
        "file",
        b"12345",
        filename="oversize.gcode",
        content_type="application/octet-stream",
    )
    response = await client.post("/api/v1/upload", data=form)

    assert response.status == 413
    assert list(upload_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_upload_rejects_disallowed_extension_without_writing(aiohttp_client, tmp_path):
    upload_dir = tmp_path / "uploads"
    app = create_test_app(upload_dir=str(upload_dir))
    client = await aiohttp_client(app)
    form = aiohttp.FormData()
    form.add_field("file", b"payload", filename="payload.exe")

    response = await client.post("/api/v1/upload", data=form)

    assert response.status == 400
    assert not upload_dir.exists()


@pytest.mark.asyncio
async def test_upload_uses_unique_non_overwriting_names(aiohttp_client, tmp_path):
    upload_dir = tmp_path / "uploads"
    app = create_test_app(upload_dir=str(upload_dir))
    client = await aiohttp_client(app)

    first_form = aiohttp.FormData()
    first_form.add_field("file", b"first", filename="model.stl")
    first_response = await client.post("/api/v1/upload", data=first_form)
    second_form = aiohttp.FormData()
    second_form.add_field("file", b"second", filename="model.stl")
    second_response = await client.post("/api/v1/upload", data=second_form)

    first = await first_response.json()
    second = await second_response.json()
    assert first_response.status == second_response.status == 200
    assert first["filename"] == "model.stl"
    assert second["filename"] == "model-1.stl"
    assert (upload_dir / "model.stl").read_bytes() == b"first"
    assert (upload_dir / "model-1.stl").read_bytes() == b"second"


@pytest.mark.asyncio
async def test_public_dashboard_shell_and_assets_do_not_require_auth(aiohttp_client):
    """Only the shell, static assets, favicon path, and health are public."""
    app = create_test_app(api_token="secret_token", auto_auth=False)
    client = await aiohttp_client(app)

    shell = await client.get("/")
    shell_text = await shell.text()
    script = await client.get("/static/app.js")
    script_text = await script.text()
    stylesheet = await client.get("/static/style.css")
    health = await client.get("/health")
    favicon = await client.get("/favicon.ico")
    unknown = await client.get("/robots.txt")

    assert shell.status == 200
    assert shell.content_type == "text/html"
    assert 'id="auth-form"' in shell_text
    assert 'id="auth-token"' in shell_text
    assert "secret_token" not in shell_text
    assert script.status == 200
    assert script.content_type in {"application/javascript", "text/javascript"}
    assert "sessionStorage" in script_text
    assert "localStorage" not in script_text
    assert "secret_token" not in script_text
    assert stylesheet.status == 200
    assert stylesheet.content_type == "text/css"
    assert health.status == 200
    assert favicon.status == 404
    assert unknown.status == 401
    assert unknown.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.asyncio
async def test_api_with_auth_token(aiohttp_client):
    """Test API with authentication token"""
    app = create_test_app(api_token="secret_token", auto_auth=False)
    client = await aiohttp_client(app)

    resp = await client.get("/api/v1/printers")
    assert resp.status == 401

    resp = await client.get(
        "/api/v1/printers",
        headers={"Authorization": "Bearer wrong_token"},
    )
    # Invalid bearer token is an authentication failure (401) per RFC 7235,
    # not an authorization failure (403). 403 is for authenticated-but-forbidden.
    assert resp.status == 401

    resp = await client.get(
        "/api/v1/printers",
        headers={"Authorization": "Bearer secret_token"},
    )
    assert resp.status == 200


@pytest.mark.asyncio
async def test_api_without_configured_auth_fails_closed(aiohttp_client):
    """The public shell stays available but protected APIs fail closed."""
    app = create_test_app(api_token=None, auto_auth=False)
    client = await aiohttp_client(app)

    resp = await client.get("/api/v1/printers")
    assert resp.status == 401
    assert resp.headers["WWW-Authenticate"] == "Bearer"

    resp = await client.get(
        "/api/v1/status",
        headers={"Authorization": "Bearer any-token"},
    )
    assert resp.status == 401
    assert resp.headers["WWW-Authenticate"] == "Bearer"

    resp = await client.get("/")
    assert resp.status == 200

    resp = await client.get("/static/app.js")
    assert resp.status == 200

    resp = await client.get("/health")
    assert resp.status == 200


def test_dashboard_defaults_to_loopback():
    assert DashboardConfig().host == "127.0.0.1"


def test_remote_dashboard_bind_requires_authentication():
    with pytest.raises(ValueError, match="api_token is required"):
        validate_dashboard_security(DashboardConfig(host="0.0.0.0", api_token=None))


def test_remote_dashboard_bind_with_authentication_is_allowed():
    validate_dashboard_security(DashboardConfig(host="0.0.0.0", api_token="secret"))


@pytest.mark.asyncio
async def test_404_for_unknown_routes(aiohttp_client):
    """Test 404 for unknown routes"""
    app = create_test_app()
    client = await aiohttp_client(app)

    resp = await client.get("/api/v1/nonexistent")
    assert resp.status == 404


@pytest.mark.asyncio
async def test_decision_job_id_mismatch(aiohttp_client, mock_state_machine, mock_decision_engine):
    """Test decision endpoint with job ID mismatch"""
    mock_state_machine.job_context = JobContext(
        job_id="current-job-123",
        job_file="test.gcode",
        uploaded_at=datetime.now(),
    )
    mock_state_machine.current_state = AgentState.AWAITING_APPROVAL

    app = create_test_app(
        state_machine=mock_state_machine,
        decision_engine=mock_decision_engine,
    )
    client = await aiohttp_client(app)

    resp = await client.post(
        "/api/v1/decision",
        json={
            "action": "approve",
            "token": "test-token",
            "job_id": "different-job-id",
        },
    )
    assert resp.status == 409
