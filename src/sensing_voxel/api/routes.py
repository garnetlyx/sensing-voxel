"""REST API route handlers for dashboard"""

import logging
from datetime import datetime

from aiohttp import web

from sensing_voxel.api.serializers import (
    serialize_alert,
    serialize_job_context,
    serialize_print_progress,
    serialize_state_machine,
    serialize_telemetry,
)
from sensing_voxel.core.models import (
    AgentState,
    ConsumeResult,
    DecisionAction,
)

logger = logging.getLogger(__name__)


async def health_check(request: web.Request) -> web.Response:
    """Health check endpoint (no auth)"""
    return web.json_response({"status": "ok", "timestamp": datetime.now().isoformat()})


async def get_status(request: web.Request) -> web.Response:
    """Get overall system status"""
    app_state = request.app["app_state"]
    sm = app_state["state_machine"]

    status = serialize_state_machine(sm)
    status["printer_connected"] = app_state.get("printer_connected", False)
    status["dashboard_clients"] = app_state.get("ws_manager").client_count

    return web.json_response(status)


async def get_job(request: web.Request) -> web.Response:
    """Get current job context"""
    app_state = request.app["app_state"]
    sm = app_state["state_machine"]
    job = serialize_job_context(sm.job_context)

    return web.json_response({"job": job})


async def get_telemetry(request: web.Request) -> web.Response:
    """Get latest telemetry snapshot"""
    app_state = request.app["app_state"]
    telemetry = serialize_telemetry(app_state.get("last_telemetry"))

    return web.json_response({"telemetry": telemetry})


async def get_alerts(request: web.Request) -> web.Response:
    """Get recent alerts"""
    app_state = request.app["app_state"]
    alerts = app_state.get("alerts", [])

    return web.json_response({
        "alerts": [serialize_alert(a) for a in alerts],
        "count": len(alerts),
    })


async def post_decision(request: web.Request) -> web.Response:
    """Submit a decision (approve/reject/resume/cancel)"""
    app_state = request.app["app_state"]
    sm = app_state["state_machine"]
    decision_engine = app_state.get("decision_engine")

    if not decision_engine:
        raise web.HTTPServiceUnavailable(text="Decision engine not available")

    try:
        body = await request.json()
    except Exception:
        raise web.HTTPBadRequest(text="Invalid JSON body")

    action_str = body.get("action")
    token = body.get("token")
    job_id = body.get("job_id")

    if not action_str or not token:
        raise web.HTTPBadRequest(text="Missing required fields: action, token")

    # Map action string to enum
    action_map = {
        "approve": DecisionAction.APPROVE,
        "reject": DecisionAction.REJECT,
        "resume": DecisionAction.RESUME,
        "cancel": DecisionAction.CANCEL,
    }

    action = action_map.get(action_str.lower())
    if action is None:
        raise web.HTTPBadRequest(text=f"Invalid action: {action_str}")

    # Validate job context
    if not sm.job_context:
        raise web.HTTPConflict(text="No active job")

    if job_id and sm.job_context.job_id != job_id:
        raise web.HTTPConflict(text="Job ID mismatch")

    # Determine expected state based on action
    expected_state_map = {
        DecisionAction.APPROVE: AgentState.AWAITING_APPROVAL,
        DecisionAction.REJECT: AgentState.AWAITING_APPROVAL,
        DecisionAction.RESUME: AgentState.PAUSED_PENDING_USER,
        DecisionAction.CANCEL: AgentState.PAUSED_PENDING_USER,
    }

    expected_state = expected_state_map.get(action)

    result = await decision_engine.consume_decision(
        job_context=sm.job_context,
        decision_token=token,
        action=action,
        expected_state=expected_state,
        current_state=sm.current_state,
    )

    result_messages = {
        ConsumeResult.CONSUMED: "Decision accepted",
        ConsumeResult.ALREADY_HANDLED: "Decision already handled",
        ConsumeResult.INVALID_STATE: "Invalid state for this action",
        ConsumeResult.INVALID_TOKEN: "Invalid decision token",
        ConsumeResult.INVALID_JOB: "Invalid job",
    }

    if result == ConsumeResult.CONSUMED:
        # Broadcast decision result via WebSocket
        ws_manager = app_state.get("ws_manager")
        if ws_manager:
            await ws_manager.broadcast_decision_result(
                sm.job_context.job_id, action.name
            )

        return web.json_response({
            "result": result.name,
            "message": result_messages[result],
        })
    else:
        return web.json_response(
            {"result": result.name, "message": result_messages.get(result, "Unknown error")},
            status=409,
        )


def setup_routes(app: web.Application, ws_manager):
    """Register all API routes"""
    app.router.add_get("/health", health_check)
    app.router.add_get("/api/v1/status", get_status)
    app.router.add_get("/api/v1/job", get_job)
    app.router.add_get("/api/v1/telemetry", get_telemetry)
    app.router.add_get("/api/v1/alerts", get_alerts)
    app.router.add_post("/api/v1/decision", post_decision)
    app.router.add_get("/api/v1/ws", ws_manager.handle_websocket)
