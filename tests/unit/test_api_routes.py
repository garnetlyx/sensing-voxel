"""
Unit tests for API routes using aiohttp test utilities
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from collections import deque

from aiohttp import web
import aiohttp

from sensing_voxel.api.server import create_app
from sensing_voxel.config import DashboardConfig
from sensing_voxel.core.state_machine import AgentStateMachine
from sensing_voxel.core.models import AgentState, JobContext, ConsumeResult, DecisionAction
from sensing_voxel.core.decision import DecisionEngine


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


class MockDecisionEngine:
    """Mock decision engine for testing"""

    def __init__(self):
        self.consume_decision = AsyncMock()


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
    api_token=None,
):
    """Create a test app with mocked dependencies"""
    config = DashboardConfig(
        enabled=True,
        host="127.0.0.1",
        port=8080,
        api_token=api_token,
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
    )

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
async def test_post_decision_success(aiohttp_client):
    """Test /api/v1/decision with valid parameters"""
    sm = AgentStateMachine()
    await sm.transition_to(AgentState.AWAITING_APPROVAL)
    sm.job_context = JobContext(
        job_id="test-job-123",
        job_file="test.gcode",
        uploaded_at=datetime.now(),
    )

    decision_engine = MockDecisionEngine()
    decision_engine.consume_decision.return_value = ConsumeResult.CONSUMED

    app = create_test_app(state_machine=sm, decision_engine=decision_engine)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/api/v1/decision",
        json={"action": "approve", "token": "test-token"},
    )
    assert resp.status == 200
    data = await resp.json()
    assert data["result"] == "CONSUMED"


@pytest.mark.asyncio
async def test_post_decision_conflict_result(aiohttp_client):
    """Test /api/v1/decision returns 409 on conflict result"""
    sm = AgentStateMachine()
    await sm.transition_to(AgentState.AWAITING_APPROVAL)
    sm.job_context = JobContext(
        job_id="test-job-123",
        job_file="test.gcode",
        uploaded_at=datetime.now(),
    )

    decision_engine = MockDecisionEngine()
    decision_engine.consume_decision.return_value = ConsumeResult.INVALID_TOKEN

    app = create_test_app(state_machine=sm, decision_engine=decision_engine)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/api/v1/decision",
        json={"action": "approve", "token": "bad-token"},
    )
    assert resp.status == 409
    data = await resp.json()
    assert data["result"] == "INVALID_TOKEN"


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
async def test_websocket_route(aiohttp_client):
    """Test /api/v1/ws endpoint exists"""
    app = create_test_app()
    client = await aiohttp_client(app)

    resp = await client.get("/api/v1/ws")
    assert resp.status in [200, 400, 426]


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
    app = create_test_app()
    client = await aiohttp_client(app)

    test_file = tmp_path / "test_upload.gcode"
    test_file.write_text("; Test G-code content")

    with open(test_file, "rb") as f:
        data = {"file": f}
        resp = await client.post("/api/v1/upload", data=data)

    assert resp.status == 200
    result = await resp.json()
    assert result["status"] == "success"
    assert result["filename"] == "test_upload.gcode"
    assert "path" in result
    assert "size" in result


@pytest.mark.asyncio
async def test_api_with_auth_token(aiohttp_client):
    """Test API with authentication token"""
    app = create_test_app(api_token="secret_token")
    client = await aiohttp_client(app)

    resp = await client.get("/api/v1/printers")
    assert resp.status == 401

    resp = await client.get(
        "/api/v1/printers",
        headers={"Authorization": "Bearer wrong_token"},
    )
    assert resp.status == 403

    resp = await client.get(
        "/api/v1/printers",
        headers={"Authorization": "Bearer secret_token"},
    )
    assert resp.status == 200


@pytest.mark.asyncio
async def test_api_no_auth_in_dev_mode(aiohttp_client):
    """Test API without auth when token is None (dev mode)"""
    app = create_test_app(api_token=None)
    client = await aiohttp_client(app)

    resp = await client.get("/api/v1/printers")
    assert resp.status == 200

    resp = await client.get("/api/v1/status")
    assert resp.status == 200

    resp = await client.get("/health")
    assert resp.status == 200


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
