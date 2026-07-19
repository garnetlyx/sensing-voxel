"""REST API route handlers for dashboard"""

import json
import logging
import os
import re
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from aiohttp import web

from sensing_voxel.api.serializers import (
    serialize_alert,
    serialize_job_context,
    serialize_state_machine,
    serialize_telemetry,
)
from sensing_voxel.core.models import (
    AgentState,
    CommandResultStatus,
    ConsumeResult,
    DecisionAction,
    FileValidationStatus,
    JobContext,
    generate_decision_token,
)
from sensing_voxel.llm.runtime import ChatRequestContext
from sensing_voxel.safety.models import PrintRequest
from sensing_voxel.storage.job_repository import JobStatus


def normalize_filename(filename: str) -> str:
    """Replace spaces and special characters with dashes to avoid Moonraker path issues."""
    name, ext = os.path.splitext(filename)
    name = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-")
    return f"{name}{ext}"

logger = logging.getLogger(__name__)


def _serialize_safety_binding(print_request: PrintRequest) -> dict[str, Any]:
    """Serialize every mutable environment value bound by approval."""
    binding = print_request.binding
    return {
        "target_printer": binding.target_printer,
        "hardware_digest": binding.hardware_digest,
        "profile_digest": binding.profile_digest,
        "config_digest": binding.config_digest,
        "macro_digest": binding.macro_digest,
        "probe_z_offset": binding.probe_z_offset,
        "homing_origin_z": binding.homing_origin_z,
        "source_trust": binding.source_trust,
        "user_id": binding.user_id,
    }


def _serialize_print_confirmation(print_request: PrintRequest) -> dict[str, Any]:
    """Serialize the complete immutable evidence shown for approval."""
    confirmation = print_request.confirmation
    return {
        "material": confirmation.material,
        "build_plate_surface": confirmation.build_plate_surface,
        "hotend_nozzle_size_mm": confirmation.hotend_nozzle_size_mm,
        "estimated_print_time_seconds": confirmation.estimated_print_time_seconds,
        "filament_usage_mm": confirmation.filament_usage_mm,
        "z_offset_mm": confirmation.z_offset_mm,
        "artifact_sha256": confirmation.artifact_sha256,
        "preview_sha256": confirmation.preview_sha256,
        "rendered_preview_sha256": confirmation.rendered_preview_sha256,
        "confirmation_sha256": confirmation.confirmation_sha256,
    }


def _serialize_print_evidence(print_request: PrintRequest) -> dict[str, Any]:
    """Serialize deterministic previews and all hashes required to audit approval."""
    return {
        "artifact_sha256": print_request.artifact.sha256,
        "preview_sha256": print_request.preview.sha256,
        "rendered_preview_sha256": print_request.preview.rendered_sha256,
        "validation_sha256": print_request.validation.result_sha256,
        "confirmation_sha256": print_request.confirmation.confirmation_sha256,
        "binding": _serialize_safety_binding(print_request),
        "confirmation": _serialize_print_confirmation(print_request),
        "preview": json.loads(print_request.preview.content),
        "preview_media_type": print_request.preview.media_type,
        "rendered_preview": print_request.preview.rendered_content.decode("utf-8"),
        "rendered_preview_media_type": print_request.preview.rendered_media_type,
    }


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
        printer_type = getattr(adapter, "printer_type", None)
        if not printer_type:
            # Determine printer type from adapter class name
            printer_type = type(adapter).__name__.replace("Adapter", "").lower()
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
        raise web.HTTPBadRequest(text="Invalid JSON body") from None

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

    job = sm.job_context
    safety_context: tuple[Any, Any, Any] | None = None
    if action in {DecisionAction.APPROVE, DecisionAction.REJECT}:
        safety = job.metadata.get("safety", {})
        printer_id = safety.get("printer_id")
        request_id = safety.get("request_id")
        operation_id = safety.get("operation_id")
        service = app_state.get("safety_services", {}).get(printer_id or "")
        adapter = app_state.get("adapters", {}).get(printer_id or "")
        if (
            not printer_id
            or printer_id != app_state.get("active_printer")
            or service is None
            or adapter is None
            or not request_id
            or not operation_id
        ):
            raise web.HTTPServiceUnavailable(
                text="Print safety approval is unavailable or incomplete"
            )
        try:
            print_request: PrintRequest = service.get_request(request_id)
            expected_evidence = _serialize_print_evidence(print_request)
            if (
                print_request.operation_id != operation_id
                or any(safety.get(key) != value for key, value in expected_evidence.items())
                or print_request.binding.source_trust != "user_gcode"
            ):
                raise ValueError("stored safety context does not match the print request")
        except Exception as error:
            raise web.HTTPConflict(text=f"Invalid print safety context: {error}") from error
        safety_context = (service, adapter, print_request)

    result = await decision_engine.consume_decision(
        job_context=job,
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

    if result != ConsumeResult.CONSUMED:
        return web.json_response(
            {"result": result.name, "message": result_messages.get(result, "Unknown error")},
            status=409,
        )

    try:
        if action is DecisionAction.APPROVE:
            assert safety_context is not None
            service, adapter, print_request = safety_context
            if not await sm.transition_to(AgentState.UPLOADING):
                raise RuntimeError("could not enter upload state")
            with tempfile.NamedTemporaryFile(suffix=".gcode") as staged:
                staged.write(print_request.artifact.content)
                staged.flush()
                upload_result = await adapter.upload_file(
                    staged.name,
                    print_request.printer_file,
                )
            if upload_result.status is not CommandResultStatus.SUCCESS:
                raise RuntimeError(f"upload failed: {upload_result.message}")
            approval = service.approve(
                print_request.request_id,
                binding=print_request.binding,
            )
            receipt = await service.execute(
                approval.token,
                operation_id=print_request.operation_id,
                binding=print_request.binding,
            )
            if not receipt.started:
                raise RuntimeError(f"print start failed: {receipt.message}")
            if not await sm.transition_to(AgentState.PRINTING):
                raise RuntimeError("could not enter printing state")
        elif action is DecisionAction.REJECT:
            assert safety_context is not None
            service, _, print_request = safety_context
            service.reject(print_request.request_id, binding=print_request.binding)
            if job.gcode_path:
                Path(job.gcode_path).unlink(missing_ok=True)
            if not await sm.transition_to(AgentState.IDLE):
                raise RuntimeError("could not return to idle state")
        elif action is DecisionAction.RESUME:
            printer_id = app_state.get("active_printer")
            adapter = app_state.get("adapters", {}).get(printer_id or "")
            if adapter is None:
                raise RuntimeError("active printer adapter is unavailable")
            command = await adapter.resume_print()
            if command.status is not CommandResultStatus.SUCCESS:
                raise RuntimeError(f"resume failed: {command.message}")
            if not await sm.transition_to(AgentState.MONITORING):
                raise RuntimeError("could not return to monitoring state")
        elif action is DecisionAction.CANCEL:
            printer_id = app_state.get("active_printer")
            adapter = app_state.get("adapters", {}).get(printer_id or "")
            if adapter is None:
                raise RuntimeError("active printer adapter is unavailable")
            command = await adapter.cancel_print()
            if command.status is not CommandResultStatus.SUCCESS:
                raise RuntimeError(f"cancel failed: {command.message}")
            if not await sm.transition_to(AgentState.IDLE):
                raise RuntimeError("could not return to idle state")
    except Exception as error:
        logger.error("Dashboard decision side effect failed: %s", error, exc_info=True)
        if sm.current_state in {AgentState.UPLOADING, AgentState.PRINTING}:
            await sm.transition_to(AgentState.ERROR)
        return web.json_response(
            {
                "result": result.name,
                "message": f"Decision accepted but execution failed: {error}",
            },
            status=500,
        )

    ws_manager = app_state.get("ws_manager")
    if ws_manager:
        await ws_manager.broadcast_decision_result(job.job_id, action.name)

    return web.json_response(
        {
            "result": result.name,
            "message": result_messages[result],
        }
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
        raise web.HTTPBadRequest(text="Invalid JSON body") from None

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
        raise web.HTTPBadRequest(text="Invalid JSON body") from None

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
        raise web.HTTPBadRequest(text="Invalid JSON body") from None

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
        raise web.HTTPBadRequest(text="Invalid JSON body") from None

    preset_id = body.get("preset_id")
    if preset_id is None:
        raise web.HTTPBadRequest(text="Missing preset_id")

    result = await camera.set_preset(int(preset_id))
    if "error" in result:
        return web.json_response(result, status=500)
    return web.json_response(result)


def _reserve_unique_upload_path(upload_dir: Path, filename: str) -> Path:
    stem = Path(filename).stem or "upload"
    suffix = Path(filename).suffix
    for counter in range(10_000):
        candidate_name = filename if counter == 0 else f"{stem}-{counter}{suffix}"
        candidate = upload_dir / candidate_name
        try:
            descriptor = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            continue
        os.close(descriptor)
        return candidate
    raise web.HTTPConflict(text="Could not allocate a unique upload filename")


async def post_upload(request: web.Request) -> web.Response:
    """Store one validated upload without overwriting an existing file."""
    app_state = request.app["app_state"]
    upload_dir = Path(app_state.get("upload_dir", "data/uploads"))
    max_upload_bytes = int(app_state.get("max_upload_bytes", 100 * 1024 * 1024))
    allowed_extensions = {
        str(extension).lower()
        for extension in app_state.get(
            "allowed_upload_extensions",
            {".gcode", ".3mf", ".stl", ".step", ".stp"},
        )
    }
    if max_upload_bytes <= 0:
        raise web.HTTPInternalServerError(text="Upload size policy is invalid")

    try:
        reader = await request.multipart()
        field = await reader.next()
    except Exception as error:
        raise web.HTTPBadRequest(text="Invalid multipart request") from error

    if field is None or field.name != "file":
        raise web.HTTPBadRequest(text="No file field in multipart request")

    supplied_filename = field.filename
    if not supplied_filename:
        raise web.HTTPBadRequest(text="No filename provided")

    filename = normalize_filename(os.path.basename(supplied_filename))
    if not filename or filename in {".", ".."}:
        raise web.HTTPBadRequest(text="Invalid filename")
    extension = Path(filename).suffix.lower()
    if extension not in allowed_extensions:
        raise web.HTTPBadRequest(text="File extension is not allowed")

    upload_dir.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp_path = tempfile.mkstemp(
        dir=upload_dir,
        prefix=".upload-",
        suffix=".tmp",
    )
    temp_path = Path(raw_temp_path)
    final_path: Path | None = None
    size = 0
    try:
        with os.fdopen(descriptor, "wb") as handle:
            while True:
                chunk = await field.read_chunk()
                if not chunk:
                    break
                size += len(chunk)
                if size > max_upload_bytes:
                    raise web.HTTPRequestEntityTooLarge(
                        max_size=max_upload_bytes,
                        actual_size=size,
                    )
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())

        final_path = _reserve_unique_upload_path(upload_dir, filename)
        os.replace(temp_path, final_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        if final_path is not None:
            final_path.unlink(missing_ok=True)
        raise

    logger.info("Saved uploaded file to %s (%d bytes)", final_path, size)

    response: dict[str, Any] = {
        "status": "success",
        "filename": final_path.name,
        "path": str(final_path.absolute()),
        "size": size,
    }
    if extension != ".gcode":
        return web.json_response(response)

    printer_id = app_state.get("active_printer")
    service = app_state.get("safety_services", {}).get(printer_id or "")
    if not printer_id or service is None:
        final_path.unlink(missing_ok=True)
        raise web.HTTPServiceUnavailable(
            text="Print safety service is unavailable for the active printer"
        )

    logical_user_id = str(app_state.get("default_logical_user_id", "dashboard-user"))
    operation_id = uuid.uuid4().hex
    try:
        binding = await service.create_binding(
            user_id=logical_user_id,
            source_trust="user_gcode",
        )
        print_request = await service.request_from_path(
            final_path,
            operation_id=operation_id,
            binding=binding,
            printer_file=final_path.name,
        )
    except Exception as error:
        final_path.unlink(missing_ok=True)
        logger.warning("Dashboard G-code safety validation failed: %s", error)
        raise web.HTTPBadRequest(text=f"G-code safety validation failed: {error}") from error

    job_id = uuid.uuid4().hex
    decision_token = generate_decision_token()
    safety_evidence = _serialize_print_evidence(print_request)
    safety_metadata = {
        "printer_id": printer_id,
        "request_id": print_request.request_id,
        "operation_id": print_request.operation_id,
        **safety_evidence,
    }
    job_context = JobContext(
        job_id=job_id,
        job_file=str(final_path),
        uploaded_at=datetime.now(),
        decision_token=decision_token,
        gcode_path=str(final_path),
        gcode_hash=print_request.artifact.sha256,
        metadata={"safety": safety_metadata},
        file_validation_status=FileValidationStatus.VALID,
    )
    state_machine = app_state["state_machine"]
    if state_machine.current_state is not AgentState.IDLE:
        service.reject(print_request.request_id, binding=print_request.binding)
        final_path.unlink(missing_ok=True)
        raise web.HTTPConflict(text="Another job is already active")
    state_machine.job_context = job_context
    if not await state_machine.transition_to(AgentState.AWAITING_APPROVAL):
        service.reject(print_request.request_id, binding=print_request.binding)
        state_machine.job_context = None
        final_path.unlink(missing_ok=True)
        raise web.HTTPConflict(text="Could not stage print approval")

    ws_manager = app_state.get("ws_manager")
    if ws_manager:
        await ws_manager.broadcast_decision_request(
            job_id,
            final_path.name,
            decision_token,
        )

    response.update(
        {
            "job_id": job_id,
            "decision_token": decision_token,
            "approval_required": True,
            "printer_id": printer_id,
            "request_id": print_request.request_id,
            "operation_id": print_request.operation_id,
            **safety_evidence,
        }
    )
    return web.json_response(response)


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
    except ValueError:
        raise web.HTTPBadRequest(text="Invalid query parameters") from None

    offset = (page - 1) * per_page

    status_filter = None
    if status:
        try:
            status_filter = JobStatus[status.upper()]
        except KeyError:
            raise web.HTTPBadRequest(text=f"Invalid status: {status}") from None

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
        raise web.HTTPInternalServerError(text="Failed to fetch job history") from e


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
        raise web.HTTPBadRequest(text="Invalid query parameters") from None

    try:
        stats = job_repo.get_job_stats(printer_id=printer_id, days=days)
        return web.json_response(stats)
    except Exception as e:
        logger.error(f"Error fetching job stats: {e}", exc_info=True)
        raise web.HTTPInternalServerError(text="Failed to fetch job stats") from e


async def post_approve(request: web.Request) -> web.Response:
    """Approve or reject a pending SVA tool approval."""
    app_state = request.app.get("app_state", {})
    chat_manager = app_state.get("chat_manager")
    if not chat_manager or not chat_manager._tool_executor:
        return web.json_response({"error": "SVA not configured"}, status=503)

    try:
        body = await request.json()
    except Exception:
        raise web.HTTPBadRequest(text="Invalid JSON body") from None

    token = body.get("approval_token", "").strip()
    action = body.get("action", "approve")
    if not token:
        return web.json_response({"error": "approval_token is required"}, status=400)

    store = chat_manager._tool_executor._approval_store
    record = store.get(token)
    if record is None or record.status != "pending":
        return web.json_response({"error": "Token not found among pending approvals"}, status=404)

    logical_user_id = str(app_state.get("default_logical_user_id", "dashboard-user"))
    if record.logical_user_id != logical_user_id or record.channel != "dashboard":
        return web.json_response({"error": "Token does not belong to this dashboard user"}, status=403)

    if action == "reject":
        rejected = store.reject(
            token,
            logical_user_id,
            "dashboard",
            record.tool_name,
        )
        if rejected is None:
            return web.json_response({"error": "Rejection failed"}, status=400)
        return web.json_response({"status": "rejected", "token": token})
    if action != "approve":
        return web.json_response({"error": "action must be approve or reject"}, status=400)

    approved = store.approve(
        token,
        logical_user_id,
        "dashboard",
        record.tool_name,
    )
    if approved is None:
        return web.json_response({"error": "Approval failed"}, status=400)
    return web.json_response(
        {
            "status": "approved",
            "token": token,
            "tool_name": approved.tool_name,
            "payload": approved.payload,
        }
    )


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
    app.router.add_post("/api/v1/approve", post_approve)
    app.router.add_get("/api/v1/ws", ws_manager.handle_websocket)
