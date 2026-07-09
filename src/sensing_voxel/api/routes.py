"""REST API route handlers for dashboard"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from aiohttp import web

from sensing_voxel.api.serializers import (
    serialize_alert,
    serialize_job_context,
    serialize_print_progress,
    serialize_state_machine,
    serialize_telemetry,
)
from sensing_voxel.config import BambuConfig, KlipperConfig
from sensing_voxel.core.models import (
    AgentState,
    ConsumeResult,
    DecisionAction,
)
from sensing_voxel.storage.job_repository import JobRepository, JobStatus
from sensing_voxel.llm.runtime import ChatRequestContext

logger = logging.getLogger(__name__)


async def health_check(request: web.Request) -> web.Response:
    """Health check endpoint (no auth)"""
    return web.json_response({"status": "ok", "timestamp": datetime.now().isoformat()})


async def get_printers(request: web.Request) -> web.Response:
    """List all configured printers with connection status"""
    app_state = request.app["app_state"]
    adapters = app_state.get("adapters", {})
    connections = app_state.get("connections", {})
    active = app_state.get("active_printer")

    printers = []
    for pid, adapter in adapters.items():
        printer_type = getattr(adapter, "printer_type", None) or type(adapter).__name__.replace(
            "Adapter", ""
        ).lower()
        printers.append(
            {
                "id": pid,
                "type": printer_type,
                "connected": connections.get(pid, False),
                "active": pid == active,
            }
        )

    return web.json_response({"printers": printers})


async def get_status(request: web.Request) -> web.Response:
    """Get overall system status"""
    app_state = request.app["app_state"]
    sm = app_state["state_machine"]

    status = serialize_state_machine(sm)
    connections = app_state.get("connections", {})
    status["printer_connected"] = any(connections.values())
    status["active_printer"] = app_state.get("active_printer")
    status["dashboard_clients"] = app_state.get("ws_manager").client_count

    return web.json_response(status)


async def get_job(request: web.Request) -> web.Response:
    """Get current job context"""
    app_state = request.app["app_state"]
    sm = app_state["state_machine"]
    job = serialize_job_context(sm.job_context)

    return web.json_response({"job": job})


async def get_telemetry(request: web.Request) -> web.Response:
    """Get latest telemetry snapshot, optionally for a specific printer"""
    app_state = request.app["app_state"]
    printer_id = request.query.get("printer")
    telemetries = app_state.get("telemetries", {})

    if printer_id:
        telemetry = serialize_telemetry(telemetries.get(printer_id))
        return web.json_response({"printer_id": printer_id, "telemetry": telemetry})

    # Return all printers' telemetry
    result = {}
    for pid, data in telemetries.items():
        result[pid] = serialize_telemetry(data)
    return web.json_response({"telemetry": result})


async def get_alerts(request: web.Request) -> web.Response:
    """Get recent alerts"""
    app_state = request.app["app_state"]
    alerts = app_state.get("alerts", [])

    return web.json_response(
        {
            "alerts": [serialize_alert(a) for a in alerts],
            "count": len(alerts),
        }
    )


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
            await ws_manager.broadcast_decision_result(sm.job_context.job_id, action.name)

        return web.json_response(
            {
                "result": result.name,
                "message": result_messages[result],
            }
        )
    else:
        return web.json_response(
            {"result": result.name, "message": result_messages.get(result, "Unknown error")},
            status=409,
        )


async def post_chat(request: web.Request) -> web.Response:
    """Send a message to AI chat and return response"""
    app_state = request.app["app_state"]
    chat_manager = app_state.get("chat_manager")

    if not chat_manager:
        raise web.HTTPServiceUnavailable(text="AI chat not configured")

    try:
        body = await request.json()
    except Exception:
        raise web.HTTPBadRequest(text="Invalid JSON body")

    message = (body.get("message") or "").strip()
    if not message:
        raise web.HTTPBadRequest(text="Empty message")

    try:
        logical_user_id = app_state.get("default_logical_user_id", "dashboard-user")
        request_context = ChatRequestContext(
            logical_user_id=str(logical_user_id),
            channel="dashboard",
            channel_user_id=str(logical_user_id),
            active_printer=app_state.get("active_printer"),
        )
        response = await chat_manager.chat(
            str(logical_user_id),
            message,
            request_context,
        )
        return web.json_response({"response": response})
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return web.json_response(
            {"error": "Failed to get AI response. Please try again."},
            status=500,
        )


async def post_chat_clear(request: web.Request) -> web.Response:
    """Clear the dashboard chat session"""
    app_state = request.app["app_state"]
    chat_manager = app_state.get("chat_manager")
    logical_user_id = app_state.get("default_logical_user_id", "dashboard-user")

    if chat_manager:
        chat_manager.clear_session(str(logical_user_id))

    return web.json_response({"status": "cleared"})


async def get_chat_history(request: web.Request) -> web.Response:
    """Get the dashboard chat history"""
    app_state = request.app["app_state"]
    chat_manager = app_state.get("chat_manager")
    logical_user_id = app_state.get("default_logical_user_id", "dashboard-user")

    if not chat_manager:
        return web.json_response({"history": []})

    history = chat_manager.get_history(str(logical_user_id))
    return web.json_response({"history": history})


async def get_chat_context(request: web.Request) -> web.Response:
    """Get chat context/session diagnostics."""
    app_state = request.app["app_state"]
    chat_manager = app_state.get("chat_manager")
    if not chat_manager:
        return web.json_response({"error": "AI chat not configured"}, status=503)

    logical_user_id = app_state.get("default_logical_user_id", "dashboard-user")
    request_context = ChatRequestContext(
        logical_user_id=str(logical_user_id),
        channel="dashboard",
        channel_user_id=str(logical_user_id),
        active_printer=app_state.get("active_printer"),
    )
    return web.json_response(chat_manager.get_context_status(str(logical_user_id), request_context))


async def get_tool_audit(request: web.Request) -> web.Response:
    """Return recent tool audit records."""
    app_state = request.app["app_state"]
    chat_manager = app_state.get("chat_manager")
    if not chat_manager:
        return web.json_response({"records": []})
    limit = int(request.query.get("limit", 50))
    return web.json_response({"records": chat_manager.get_audit_records(limit=limit)})


async def get_background_tasks(request: web.Request) -> web.Response:
    """Return recent background tasks for the dashboard logical user."""
    app_state = request.app["app_state"]
    chat_manager = app_state.get("chat_manager")
    if not chat_manager:
        return web.json_response({"tasks": []})
    logical_user_id = app_state.get("default_logical_user_id", "dashboard-user")
    limit = int(request.query.get("limit", 20))
    return web.json_response(
        {"tasks": chat_manager.get_background_tasks(str(logical_user_id), limit=limit)}
    )


async def get_pending_approvals(request: web.Request) -> web.Response:
    """Return pending approvals for the dashboard logical user."""
    app_state = request.app["app_state"]
    chat_manager = app_state.get("chat_manager")
    if not chat_manager:
        return web.json_response({"approvals": []})
    logical_user_id = app_state.get("default_logical_user_id", "dashboard-user")
    limit = int(request.query.get("limit", 20))
    return web.json_response(
        {"approvals": chat_manager.get_pending_approvals(str(logical_user_id), limit=limit)}
    )


async def set_active_printer(request: web.Request) -> web.Response:
    """Set the active printer for agent control"""
    app_state = request.app["app_state"]

    try:
        body = await request.json()
    except Exception:
        raise web.HTTPBadRequest(text="Invalid JSON body")

    printer_id = body.get("printer_id")
    if not printer_id:
        raise web.HTTPBadRequest(text="Missing printer_id")

    if printer_id not in app_state.get("adapters", {}):
        raise web.HTTPNotFound(text=f"Unknown printer: {printer_id}")

    app_state["active_printer"] = printer_id
    logger.info(f"Active printer set to: {printer_id}")

    return web.json_response({"active_printer": printer_id})


async def post_camera_move(request: web.Request) -> web.Response:
    """Move camera in a direction (step movement)"""
    app_state = request.app["app_state"]
    camera = app_state.get("camera")

    if not camera:
        raise web.HTTPServiceUnavailable(text="Camera PTZ not configured")

    try:
        body = await request.json()
    except Exception:
        raise web.HTTPBadRequest(text="Invalid JSON body")

    direction = (body.get("direction") or "").strip().lower()
    if direction not in ("left", "right", "up", "down"):
        raise web.HTTPBadRequest(text="Invalid direction. Use: left, right, up, down")

    result = await camera.move_step(direction)
    if "error" in result:
        return web.json_response(result, status=500)
    return web.json_response(result)


async def post_camera_preset(request: web.Request) -> web.Response:
    """Move camera to a preset position"""
    app_state = request.app["app_state"]
    camera = app_state.get("camera")

    if not camera:
        raise web.HTTPServiceUnavailable(text="Camera PTZ not configured")

    try:
        body = await request.json()
    except Exception:
        raise web.HTTPBadRequest(text="Invalid JSON body")

    preset_id = body.get("preset_id")
    if preset_id is None:
        raise web.HTTPBadRequest(text="Missing preset_id")

    result = await camera.set_preset(int(preset_id))
    if "error" in result:
        return web.json_response(result, status=500)
    return web.json_response(result)


async def post_upload(request: web.Request) -> web.Response:
    """Handle file upload and save to data/uploads"""
    try:
        reader = await request.multipart()
    except Exception:
        raise web.HTTPBadRequest(text="Invalid multipart request")

    field = await reader.next()  # type: ignore[attr-defined]
    if not field or field.name != "file":  # type: ignore[union-attr]
        return web.json_response({"error": "No file field in multipart request"}, status=400)

    filename = field.filename  # type: ignore[union-attr]
    if not filename:
        return web.json_response({"error": "No filename provided"}, status=400)

    # Sanitize filename
    filename = os.path.basename(filename)

    # Ensure data/uploads directory exists
    # Use data directory relative to the project root
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / filename

    size = 0
    with open(file_path, "wb") as f:
        while True:
            chunk = await field.read_chunk()  # type: ignore[attr-defined]
            if not chunk:
                break
            size += len(chunk)
            f.write(chunk)

    logger.info(f"Saved uploaded file to {file_path} ({size} bytes)")

    return web.json_response(
        {"status": "success", "filename": filename, "path": str(file_path.absolute()), "size": size}
    )


async def get_job_history(request: web.Request) -> web.Response:
    """Get job history from SQLite database with pagination and filtering"""
    app_state = request.app["app_state"]
    job_repo = app_state.get("job_repository")

    if not job_repo:
        return web.json_response(
            {
                "jobs": [],
                "total": 0,
                "page": 1,
                "per_page": 20,
            }
        )

    try:
        page = int(request.query.get("page", "1"))
        per_page = min(int(request.query.get("per_page", "20")), 100)
        printer_id = request.query.get("printer_id")
        status = request.query.get("status")
        days = int(request.query.get("days", "30"))
    except ValueError:
        raise web.HTTPBadRequest(text="Invalid query parameters")

    offset = (page - 1) * per_page

    status_filter = None
    if status:
        try:
            status_filter = JobStatus[status.upper()]
        except KeyError:
            raise web.HTTPBadRequest(text=f"Invalid status: {status}")

    try:
        all_jobs = job_repo.list_jobs(
            printer_id=printer_id,
            status=status_filter,
            limit=999999,
            offset=0,
        )
        total = len(all_jobs)

        jobs = job_repo.list_jobs(
            printer_id=printer_id,
            status=status_filter,
            limit=per_page,
            offset=offset,
        )

        serialized_jobs = []
        for job in jobs:
            serialized_jobs.append(
                {
                    "id": job.id,
                    "job_id": job.job_id,
                    "printer_id": job.printer_id,
                    "file_name": job.file_name,
                    "status": job.status.name.lower(),
                    "started_at": job.started_at.isoformat() if job.started_at else None,
                    "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                    "duration_seconds": job.duration_seconds,
                    "filament_used_mm": job.filament_used_mm,
                    "notes": job.notes,
                    "metadata": job.metadata,
                }
            )

        return web.json_response(
            {
                "jobs": serialized_jobs,
                "total": total,
                "page": page,
                "per_page": per_page,
                "pages": (total + per_page - 1) // per_page,
            }
        )
    except Exception as e:
        logger.error(f"Error fetching job history: {e}", exc_info=True)
        raise web.HTTPInternalServerError(text="Failed to fetch job history")


async def get_job_stats(request: web.Request) -> web.Response:
    """Get job statistics from SQLite database"""
    app_state = request.app["app_state"]
    job_repo = app_state.get("job_repository")

    if not job_repo:
        return web.json_response(
            {
                "total_jobs": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
                "success_rate": 0.0,
                "avg_duration_seconds": 0.0,
                "total_filament_mm": 0.0,
                "period_days": 30,
            }
        )

    try:
        days = int(request.query.get("days", "30"))
        printer_id = request.query.get("printer_id")
    except ValueError:
        raise web.HTTPBadRequest(text="Invalid query parameters")

    try:
        stats = job_repo.get_job_stats(printer_id=printer_id, days=days)
        return web.json_response(stats)
    except Exception as e:
        logger.error(f"Error fetching job stats: {e}", exc_info=True)
        raise web.HTTPInternalServerError(text="Failed to fetch job stats")


async def get_queue_status(request: web.Request) -> web.Response:
    """Get current command queue status"""
    app_state = request.app["app_state"]

    # Get command queue from adapters if available
    queue_info = []
    for printer_id, adapter in app_state.get("adapters", {}).items():
        if hasattr(adapter, "_command_queue"):
            queue = adapter._command_queue
            queue_info.append(
                {
                    "printer_id": printer_id,
                    "queue_size": queue.size(),
                    "processing": queue._processing if hasattr(queue, "_processing") else False,
                }
            )

    return web.json_response(
        {
            "queues": queue_info,
            "total_pending": sum(q["queue_size"] for q in queue_info),
        }
    )


async def post_queue_clear(request: web.Request) -> web.Response:
    """Clear command queue for a printer"""
    app_state = request.app["app_state"]

    try:
        body = await request.json()
    except Exception:
        raise web.HTTPBadRequest(text="Invalid JSON body")

    printer_id = body.get("printer_id")
    if not printer_id:
        raise web.HTTPBadRequest(text="Missing printer_id")

    adapter = app_state.get("adapters", {}).get(printer_id)
    if not adapter:
        raise web.HTTPNotFound(text=f"Unknown printer: {printer_id}")

    if not hasattr(adapter, "_command_queue"):
        raise web.HTTPServiceUnavailable(text=f"Printer {printer_id} has no command queue")

    try:
        adapter._command_queue.clear()
        return web.json_response(
            {
                "status": "cleared",
                "printer_id": printer_id,
            }
        )
    except Exception as e:
        logger.error(f"Error clearing queue: {e}", exc_info=True)
        raise web.HTTPInternalServerError(text="Failed to clear queue")


def setup_routes(app: web.Application, ws_manager):
    """Register all API routes"""
    app.router.add_get("/health", health_check)
    app.router.add_get("/api/v1/printers", get_printers)
    app.router.add_get("/api/v1/status", get_status)
    app.router.add_get("/api/v1/job", get_job)
    app.router.add_get("/api/v1/telemetry", get_telemetry)
    app.router.add_get("/api/v1/alerts", get_alerts)
    app.router.add_post("/api/v1/decision", post_decision)
    app.router.add_post("/api/v1/active-printer", set_active_printer)
    app.router.add_post("/api/v1/chat", post_chat)
    app.router.add_get("/api/v1/chat/history", get_chat_history)
    app.router.add_post("/api/v1/chat/clear", post_chat_clear)
    app.router.add_get("/api/v1/chat/context", get_chat_context)
    app.router.add_get("/api/v1/tool-audit", get_tool_audit)
    app.router.add_get("/api/v1/background-tasks", get_background_tasks)
    app.router.add_get("/api/v1/pending-approvals", get_pending_approvals)
    app.router.add_post("/api/v1/upload", post_upload)
    app.router.add_post("/api/v1/camera/move", post_camera_move)
    app.router.add_post("/api/v1/camera/preset", post_camera_preset)
    # Job history endpoints
    app.router.add_get("/api/v1/jobs/history", get_job_history)
    app.router.add_get("/api/v1/jobs/stats", get_job_stats)
    # Queue management endpoints
    app.router.add_get("/api/v1/queue", get_queue_status)
    app.router.add_post("/api/v1/queue/clear", post_queue_clear)
    app.router.add_get("/api/v1/ws", ws_manager.handle_websocket)
