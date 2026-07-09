"""
Anthropic tool definitions and executor for printer control
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from sensing_voxel.adapters.base import PrinterAdapter
from sensing_voxel.core.models import CommandResultStatus
from sensing_voxel.infra.gcode import _strip_gcode_comment, validate_gcode_command
from sensing_voxel.llm.runtime import (
    BackgroundTaskManager,
    ChatRequestContext,
    PendingApprovalStore,
    ToolAuditLogger,
    ToolAuditRecord,
    classify_gcode_command,
    classify_shell_command,
)

logger = logging.getLogger(__name__)

# Safety limits (kept for tool schema descriptions)
NOZZLE_TEMP_MAX = 300
BED_TEMP_MAX = 120


PRINTER_TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_printer_state",
        "description": "Get the current printer state (idle, printing, paused, error, etc.)",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_print_progress",
        "description": (
            "Get detailed print progress including filename, percentage, "
            "temperatures, layer info, and print time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_current_position",
        "description": "Get the current nozzle/toolhead position (X, Y, Z) if available.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "start_print",
        "description": "Start printing a G-code file that is already uploaded to the printer.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "G-code filename on the printer",
                },
                "approval_token": {
                    "type": "string",
                    "description": "Approval token returned by a previous approval_required response.",
                },
            },
            "required": ["filename"],
        },
    },
    {
        "name": "pause_print",
        "description": "Pause the current print job.",
        "input_schema": {
            "type": "object",
            "properties": {
                "approval_token": {
                    "type": "string",
                    "description": "Approval token returned by a previous approval_required response.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "resume_print",
        "description": "Resume a paused print job.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "cancel_print",
        "description": (
            "Cancel the current print job. This is destructive and cannot be undone. "
            "Always confirm with the user before cancelling."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "set_nozzle_temp",
        "description": f"Set the nozzle/hotend temperature in Celsius (0-{NOZZLE_TEMP_MAX}).",
        "input_schema": {
            "type": "object",
            "properties": {
                "temperature": {
                    "type": "number",
                    "description": "Target temperature in Celsius",
                },
                "approval_token": {
                    "type": "string",
                    "description": "Approval token returned by a previous approval_required response.",
                },
            },
            "required": ["temperature"],
        },
    },
    {
        "name": "set_bed_temp",
        "description": f"Set the heated bed temperature in Celsius (0-{BED_TEMP_MAX}).",
        "input_schema": {
            "type": "object",
            "properties": {
                "temperature": {
                    "type": "number",
                    "description": "Target temperature in Celsius",
                },
                "approval_token": {
                    "type": "string",
                    "description": "Approval token returned by a previous approval_required response.",
                },
            },
            "required": ["temperature"],
        },
    },
    {
        "name": "home_axes",
        "description": (
            "Home printer axes. Specify axes like 'X', 'Y', 'Z', 'X Y', or leave empty to home all."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "axes": {
                    "type": "string",
                    "description": "Axes to home (e.g. 'X Y Z' or '' for all)",
                    "default": "",
                },
            },
            "required": [],
        },
    },
    {
        "name": "set_fan_speed",
        "description": "Set part cooling fan speed (0=off, 255=max).",
        "input_schema": {
            "type": "object",
            "properties": {
                "speed": {
                    "type": "integer",
                    "description": "Fan speed 0-255",
                },
            },
            "required": ["speed"],
        },
    },
    {
        "name": "run_gcode",
        "description": (
            "Run an arbitrary G-code command on the printer. "
            "Use with caution. M112 (emergency stop) is blocked — use emergency_cancel instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "G-code command to execute",
                },
                "approval_token": {
                    "type": "string",
                    "description": "Approval token returned by a previous approval_required response.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_z_offset",
        "description": "Get the current Z-offset value in millimeters.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "set_z_offset",
        "description": "Adjust Z-offset by the given amount in mm (-10 to +10).",
        "input_schema": {
            "type": "object",
            "properties": {
                "offset_mm": {
                    "type": "number",
                    "description": "Z-offset adjustment in mm",
                },
            },
            "required": ["offset_mm"],
        },
    },
    # --- File management ---
    {
        "name": "list_files",
        "description": "List all G-code files on the printer.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "delete_file",
        "description": "Delete a G-code file from the printer. Confirm with user first.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Filename to delete"},
                "approval_token": {
                    "type": "string",
                    "description": "Approval token returned by a previous approval_required response.",
                },
            },
            "required": ["filename"],
        },
    },
    {
        "name": "get_file_metadata",
        "description": (
            "Get G-code file metadata: total layers, layer height, object height, "
            "filament type, slicer info. Parses the G-code header if needed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "G-code filename"},
            },
            "required": ["filename"],
        },
    },
    # --- Manual motion ---
    {
        "name": "move_axis",
        "description": (
            "Move an axis by a relative distance in mm. "
            "Printer must be homed first. Positive = away from endstop."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "axis": {"type": "string", "description": "Axis: X, Y, or Z"},
                "distance": {
                    "type": "number",
                    "description": "Distance in mm (negative = reverse)",
                },
                "speed": {
                    "type": "integer",
                    "description": "Feed rate in mm/min (default 3000)",
                    "default": 3000,
                },
            },
            "required": ["axis", "distance"],
        },
    },
    {
        "name": "extrude",
        "description": (
            "Extrude or retract filament. Positive = extrude, negative = retract. "
            "Nozzle must be hot enough."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "length": {"type": "number", "description": "Length in mm (negative = retract)"},
                "speed": {
                    "type": "integer",
                    "description": "Feed rate in mm/min (default 300)",
                    "default": 300,
                },
            },
            "required": ["length"],
        },
    },
    {
        "name": "motors_off",
        "description": "Turn off all stepper motors (M84).",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    # --- Speed/flow override ---
    {
        "name": "set_speed_override",
        "description": "Set print speed override percentage (100 = normal, 50 = half speed, 200 = double).",
        "input_schema": {
            "type": "object",
            "properties": {
                "percent": {"type": "integer", "description": "Speed percentage (10-300)"},
            },
            "required": ["percent"],
        },
    },
    {
        "name": "set_flow_override",
        "description": "Set flow/extrusion multiplier percentage (100 = normal).",
        "input_schema": {
            "type": "object",
            "properties": {
                "percent": {"type": "integer", "description": "Flow percentage (50-200)"},
            },
            "required": ["percent"],
        },
    },
    # --- Macros ---
    {
        "name": "list_macros",
        "description": "List all available Klipper G-code macros.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "execute_macro",
        "description": "Execute a Klipper macro by name (e.g. PRINT_START, RESUME, CANCEL_PRINT).",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Macro name"},
                "approval_token": {
                    "type": "string",
                    "description": "Approval token returned by a previous approval_required response.",
                },
            },
            "required": ["name"],
        },
    },
    # --- System ---
    {
        "name": "restart_firmware",
        "description": "Restart Klipper firmware (FIRMWARE_RESTART). Use after MCU errors.",
        "input_schema": {
            "type": "object",
            "properties": {
                "approval_token": {
                    "type": "string",
                    "description": "Approval token returned by a previous approval_required response.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "restart_service",
        "description": "Restart Klipper host service. Use when firmware restart isn't enough.",
        "input_schema": {
            "type": "object",
            "properties": {
                "approval_token": {
                    "type": "string",
                    "description": "Approval token returned by a previous approval_required response.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_system_info",
        "description": "Get system info: CPU load, memory, Klipper state and messages.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "upload_to_printer",
        "description": (
            "Upload a file from the server's local storage to the printer. "
            "Use this when a user has uploaded a G-code file to the dashboard."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "The name of the file to upload to the printer.",
                },
            },
            "required": ["filename"],
        },
    },
    {
        "name": "run_shell",
        "description": (
            "Run a shell command inside the agent container. "
            "Low-risk commands like sleep, curl, and read-only inspection may run directly. "
            "High-risk commands require explicit user approval."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                },
                "approval_token": {
                    "type": "string",
                    "description": "Approval token returned by a previous approval_required response.",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 30)",
                    "default": 30,
                },
                "cwd": {
                    "type": "string",
                    "description": "Optional working directory",
                },
                "background": {
                    "type": "boolean",
                    "description": "Run the shell command in the background without blocking the chat turn.",
                    "default": False,
                },
                "task_name": {
                    "type": "string",
                    "description": "Optional human-readable label for the background task.",
                },
                "completion_message": {
                    "type": "string",
                    "description": "Optional message to send when the background task completes.",
                },
                "progress_interval_seconds": {
                    "type": "integer",
                    "description": "Optional cadence for progress callbacks while the task is running.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_background_tasks",
        "description": "List recent background tasks started by the current user.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of tasks to return (default 10).",
                    "default": 10,
                },
            },
            "required": [],
        },
    },
    {
        "name": "cancel_background_task",
        "description": "Cancel a running background task by task_id.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "Task identifier returned by run_shell(background=true).",
                },
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "get_pending_approvals",
        "description": "List recent approval-gated actions waiting for explicit confirmation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of approvals to return (default 10).",
                    "default": 10,
                },
            },
            "required": [],
        },
    },
]

for tool in PRINTER_TOOLS:
    if tool["name"] in {
        "run_shell",
        "get_background_tasks",
        "cancel_background_task",
        "get_pending_approvals",
    }:
        continue
    properties = tool["input_schema"].setdefault("properties", {})
    properties["printer"] = {
        "type": "string",
        "description": "Optional target printer ID. Defaults to the active printer in request context.",
    }


CAMERA_TOOLS: list[dict[str, Any]] = [
    {
        "name": "camera_snapshot",
        "description": (
            "Capture a snapshot from the monitoring camera and analyze it visually. "
            "Use this when the user asks you to look at the printer, check print quality, "
            "or see what's happening."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "camera_move_step",
        "description": (
            "Move the monitoring camera one step in a direction. "
            "Use this to adjust the camera view of the printer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["left", "right", "up", "down"],
                    "description": "Direction to move the camera",
                },
            },
            "required": ["direction"],
        },
    },
    {
        "name": "camera_move_to",
        "description": (
            "Move the monitoring camera to an absolute pan/tilt position. "
            "Pan is horizontal, tilt is vertical."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pan": {
                    "type": "integer",
                    "description": "Horizontal pan position",
                },
                "tilt": {
                    "type": "integer",
                    "description": "Vertical tilt position",
                },
            },
            "required": ["pan", "tilt"],
        },
    },
    {
        "name": "camera_get_presets",
        "description": "Get the list of saved camera position presets.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "camera_go_to_preset",
        "description": "Move the camera to a saved preset position by its ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "preset_id": {
                    "type": "integer",
                    "description": "ID of the preset to move to",
                },
            },
            "required": ["preset_id"],
        },
    },
    {
        "name": "camera_get_position",
        "description": "Get the current camera pan/tilt position if supported by the PTZ backend.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


class ToolExecutor:
    """Dispatches tool calls to the printer adapter and optional camera."""

    def __init__(
        self,
        adapters: dict[str, PrinterAdapter] | None = None,
        adapter: PrinterAdapter | None = None,
        camera=None,
        rtsp_url: str | None = None,
        audit_logger: ToolAuditLogger | None = None,
        photo_sender=None,
        text_notifier=None,
        background_tasks: BackgroundTaskManager | None = None,
        approval_store: PendingApprovalStore | None = None,
        allowed_gcode_commands: set[str] | None = None,
    ):
        if adapters is None:
            adapters = {"default": adapter} if adapter is not None else {}
        self._adapters = adapters
        self._camera: Any | None = camera
        self._rtsp_url: str | None = rtsp_url
        self._audit_logger = audit_logger
        self._photo_sender = photo_sender
        self._text_notifier = text_notifier
        self._background_tasks = background_tasks or BackgroundTaskManager()
        self._approval_store = approval_store or PendingApprovalStore()
        self._allowed_gcode_commands = {cmd.upper() for cmd in (allowed_gcode_commands or set())}
        self._dispatch: dict[str, Any] = {
            "get_printer_state": self._get_printer_state,
            "get_print_progress": self._get_print_progress,
            "get_current_position": self._get_current_position,
            "start_print": self._start_print,
            "pause_print": self._pause_print,
            "resume_print": self._resume_print,
            "cancel_print": self._cancel_print,
            "set_nozzle_temp": self._set_nozzle_temp,
            "set_bed_temp": self._set_bed_temp,
            "home_axes": self._home_axes,
            "set_fan_speed": self._set_fan_speed,
            "run_gcode": self._run_gcode,
            "get_z_offset": self._get_z_offset,
            "set_z_offset": self._set_z_offset,
            "list_files": self._list_files,
            "delete_file": self._delete_file,
            "get_file_metadata": self._get_file_metadata,
            "move_axis": self._move_axis,
            "extrude": self._extrude,
            "motors_off": self._motors_off,
            "set_speed_override": self._set_speed_override,
            "set_flow_override": self._set_flow_override,
            "list_macros": self._list_macros,
            "execute_macro": self._execute_macro,
            "restart_firmware": self._restart_firmware,
            "restart_service": self._restart_service,
            "get_system_info": self._get_system_info,
            "upload_to_printer": self._upload_to_printer,
            "run_shell": self._run_shell,
            "get_background_tasks": self._get_background_tasks,
            "cancel_background_task": self._cancel_background_task,
            "get_pending_approvals": self._get_pending_approvals,
        }
        if camera:
            self._dispatch.update(
                {
                    "camera_move_step": self._camera_move_step,
                    "camera_move_to": self._camera_move_to,
                    "camera_get_presets": self._camera_get_presets,
                    "camera_go_to_preset": self._camera_go_to_preset,
                    "camera_get_position": self._camera_get_position,
                }
            )
        if rtsp_url:
            self._dispatch["camera_snapshot"] = self._camera_snapshot

    async def execute(
        self,
        tool_name: str,
        tool_input: dict,
        request_context: ChatRequestContext | None = None,
    ) -> str | dict:
        """Execute a tool call and return JSON string or structured content blocks."""
        request_context = request_context or ChatRequestContext(
            logical_user_id="unknown",
            channel="unknown",
            active_printer=None,
        )
        handler = self._dispatch.get(tool_name)
        if not handler:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        started_at = datetime.now().isoformat()
        try:
            result = await handler(tool_input, request_context)
            self._write_audit_record(
                request_context,
                tool_name,
                tool_input,
                result,
                started_at,
                status="success",
            )
            # Structured content (e.g. image blocks) — pass through as-is
            if isinstance(result, dict) and "content" in result:
                return result
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Tool execution error ({tool_name}): {e}", exc_info=True)
            error_result = {"error": str(e)}
            self._write_audit_record(
                request_context,
                tool_name,
                tool_input,
                error_result,
                started_at,
                status="error",
            )
            return json.dumps({"error": str(e)})

    # --- Read-only tools ---

    def _resolve_adapter(
        self,
        tool_input: dict,
        request_context: ChatRequestContext,
    ) -> tuple[str | None, PrinterAdapter | None]:
        printer_id = tool_input.get("printer") or request_context.active_printer
        if not printer_id and self._adapters:
            printer_id = next(iter(self._adapters))
        return printer_id, self._adapters.get(printer_id) if printer_id else None

    def _should_include_printer(
        self,
        tool_input: dict,
        request_context: ChatRequestContext,
        printer_id: str | None,
    ) -> bool:
        return (
            len(self._adapters) > 1
            or "printer" in tool_input
            or request_context.active_printer is not None
            or (printer_id not in (None, "default"))
        )

    async def _notify(
        self,
        request_context: ChatRequestContext,
        message: str,
        level: str = "info",
        data: dict[str, Any] | None = None,
    ) -> None:
        if self._text_notifier is None:
            return
        try:
            await self._text_notifier(request_context, message, level=level, data=data or {})
        except Exception:
            logger.warning("Failed to send background notification", exc_info=True)

    def _create_approval(
        self,
        request_context: ChatRequestContext,
        tool_name: str,
        payload: dict[str, Any],
        reason: str,
    ) -> dict[str, Any]:
        record = self._approval_store.create(
            logical_user_id=request_context.logical_user_id,
            channel=request_context.channel,
            tool_name=tool_name,
            payload=payload,
            reason=reason,
        )
        return {
            "status": "approval_required",
            "approval_token": record.approval_token,
            "tool": tool_name,
            "reason": reason,
            "payload": payload,
        }

    def _consume_approval(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        request_context: ChatRequestContext,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        token = (tool_input.get("approval_token") or "").strip()
        if not token:
            return None, None
        record = self._approval_store.approve(token, request_context.logical_user_id, tool_name)
        if record is None:
            return None, {"error": f"Invalid or expired approval_token for {tool_name}"}
        merged = dict(record.payload)
        merged.update({k: v for k, v in tool_input.items() if k != "approval_token"})
        return merged, None

    def _write_audit_record(
        self,
        request_context: ChatRequestContext,
        tool_name: str,
        tool_input: dict,
        result: Any,
        started_at: str,
        status: str,
    ) -> None:
        if not self._audit_logger:
            return

        contains_image = False
        if isinstance(result, dict) and "content" in result:
            contains_image = any(block.get("type") == "image" for block in result["content"])

        preview = json.dumps(result, ensure_ascii=False)[:400]
        self._audit_logger.append(
            ToolAuditRecord(
                request_id=request_context.request_id,
                logical_user_id=request_context.logical_user_id,
                channel=request_context.channel,
                tool_name=tool_name,
                tool_input=tool_input,
                started_at=started_at,
                finished_at=datetime.now().isoformat(),
                status=status,
                result_preview=preview,
                contains_image=contains_image,
            )
        )

    async def _get_printer_state(self, tool_input: dict, request_context: ChatRequestContext) -> dict:
        printer_id, adapter = self._resolve_adapter(tool_input, request_context)
        if adapter is None:
            return {"error": f"Unknown printer: {printer_id}"}
        state = await adapter.get_printer_state()
        result = {"state": state.name}
        if self._should_include_printer(tool_input, request_context, printer_id):
            result["printer"] = printer_id
        return result

    async def _get_current_position(
        self,
        tool_input: dict,
        request_context: ChatRequestContext,
    ) -> dict:
        printer_id, adapter = self._resolve_adapter(tool_input, request_context)
        if adapter is None:
            return {"error": f"Unknown printer: {printer_id}"}
        position = await adapter.get_current_position()
        if position is None:
            return {"printer": printer_id, "position": None}
        return {"printer": printer_id, "position": position}

    async def _get_print_progress(self, tool_input: dict, request_context: ChatRequestContext) -> dict:
        printer_id, adapter = self._resolve_adapter(tool_input, request_context)
        if adapter is None:
            return {"error": f"Unknown printer: {printer_id}"}
        progress = await adapter.get_print_progress()
        result: dict[str, Any] = {"state": progress.state.name}
        if self._should_include_printer(tool_input, request_context, printer_id):
            result["printer"] = printer_id
        if progress.filename:
            result["filename"] = progress.filename
        if progress.progress is not None:
            result["progress_percent"] = round(progress.progress * 100, 1)
        if progress.print_time is not None:
            result["print_time_seconds"] = round(progress.print_time, 0)
        if progress.nozzle_temp is not None:
            result["nozzle_temp"] = round(progress.nozzle_temp, 1)
        if progress.nozzle_target is not None:
            result["nozzle_target"] = round(progress.nozzle_target, 1)
        if progress.bed_temp is not None:
            result["bed_temp"] = round(progress.bed_temp, 1)
        if progress.bed_target is not None:
            result["bed_target"] = round(progress.bed_target, 1)
        if progress.current_layer is not None:
            result["current_layer"] = progress.current_layer
        if progress.total_layers is not None:
            result["total_layers"] = progress.total_layers
        return result

    # --- Print lifecycle ---

    async def _start_print(self, tool_input: dict, request_context: ChatRequestContext) -> dict:
        approved_input, approval_error = self._consume_approval(
            "start_print", tool_input, request_context
        )
        if approval_error:
            return approval_error
        effective_input = approved_input or tool_input
        filename = effective_input.get("filename", "")
        if not filename:
            return {"error": "filename is required"}
        if approved_input is None:
            return self._create_approval(
                request_context,
                "start_print",
                {"filename": filename, "printer": effective_input.get("printer")},
                "Starting a print job changes printer state and requires explicit approval.",
            )
        printer_id, adapter = self._resolve_adapter(effective_input, request_context)
        if adapter is None:
            return {"error": f"Unknown printer: {printer_id}"}
        cmd = await adapter.start_print(filename)
        return self._cmd_result(cmd)

    async def _pause_print(self, tool_input: dict, request_context: ChatRequestContext) -> dict:
        printer_id, adapter = self._resolve_adapter(tool_input, request_context)
        if adapter is None:
            return {"error": f"Unknown printer: {printer_id}"}
        cmd = await adapter.pause_print()
        return self._cmd_result(cmd)

    async def _resume_print(self, tool_input: dict, request_context: ChatRequestContext) -> dict:
        printer_id, adapter = self._resolve_adapter(tool_input, request_context)
        if adapter is None:
            return {"error": f"Unknown printer: {printer_id}"}
        cmd = await adapter.resume_print()
        return self._cmd_result(cmd)

    async def _cancel_print(self, tool_input: dict, request_context: ChatRequestContext) -> dict:
        approved_input, approval_error = self._consume_approval(
            "cancel_print", tool_input, request_context
        )
        if approval_error:
            return approval_error
        effective_input = approved_input or tool_input
        if approved_input is None:
            return self._create_approval(
                request_context,
                "cancel_print",
                {"printer": effective_input.get("printer")},
                "Cancelling an active print job is destructive and requires approval.",
            )
        printer_id, adapter = self._resolve_adapter(effective_input, request_context)
        if adapter is None:
            return {"error": f"Unknown printer: {printer_id}"}
        cmd = await adapter.cancel_print()
        result = self._cmd_result(cmd)
        result["warning"] = "Print has been cancelled. This cannot be undone."
        return result

    # --- Temperature & control ---

    async def _set_nozzle_temp(self, tool_input: dict, request_context: ChatRequestContext) -> dict:
        approved_input, approval_error = self._consume_approval(
            "set_nozzle_temp", tool_input, request_context
        )
        if approval_error:
            return approval_error
        effective_input = approved_input or tool_input
        temp = effective_input.get("temperature")
        if temp is None:
            return {"error": "temperature is required"}
        temp = float(temp)
        if temp < 0 or temp > NOZZLE_TEMP_MAX:
            return {"error": f"Temperature must be 0-{NOZZLE_TEMP_MAX}°C, got {temp}"}
        if temp >= 260 and approved_input is None:
            return self._create_approval(
                request_context,
                "set_nozzle_temp",
                {"temperature": temp, "printer": effective_input.get("printer")},
                "High nozzle temperatures require explicit approval.",
            )
        printer_id, adapter = self._resolve_adapter(effective_input, request_context)
        if adapter is None:
            return {"error": f"Unknown printer: {printer_id}"}
        cmd = await adapter.set_nozzle_temp(temp)
        result = self._cmd_result(cmd)
        if cmd.status == CommandResultStatus.SUCCESS:
            result["message"] = f"Nozzle temperature set to {int(temp)}°C"
        return result

    async def _set_bed_temp(self, tool_input: dict, request_context: ChatRequestContext) -> dict:
        approved_input, approval_error = self._consume_approval(
            "set_bed_temp", tool_input, request_context
        )
        if approval_error:
            return approval_error
        effective_input = approved_input or tool_input
        temp = effective_input.get("temperature")
        if temp is None:
            return {"error": "temperature is required"}
        temp = float(temp)
        if temp < 0 or temp > BED_TEMP_MAX:
            return {"error": f"Temperature must be 0-{BED_TEMP_MAX}°C, got {temp}"}
        if temp >= 100 and approved_input is None:
            return self._create_approval(
                request_context,
                "set_bed_temp",
                {"temperature": temp, "printer": effective_input.get("printer")},
                "High bed temperatures require explicit approval.",
            )
        printer_id, adapter = self._resolve_adapter(effective_input, request_context)
        if adapter is None:
            return {"error": f"Unknown printer: {printer_id}"}
        cmd = await adapter.set_bed_temp(temp)
        result = self._cmd_result(cmd)
        if cmd.status == CommandResultStatus.SUCCESS:
            result["message"] = f"Bed temperature set to {int(temp)}°C"
        return result

    async def _home_axes(self, tool_input: dict, request_context: ChatRequestContext) -> dict:
        axes = tool_input.get("axes", "")
        printer_id, adapter = self._resolve_adapter(tool_input, request_context)
        if adapter is None:
            return {"error": f"Unknown printer: {printer_id}"}
        cmd = await adapter.home_axes(axes)
        result = self._cmd_result(cmd)
        if cmd.status == CommandResultStatus.SUCCESS:
            label = axes.upper() if axes.strip() else "all"
            result["message"] = f"Homing {label} axes"
        return result

    async def _set_fan_speed(self, tool_input: dict, request_context: ChatRequestContext) -> dict:
        speed = tool_input.get("speed")
        if speed is None:
            return {"error": "speed is required"}
        speed = int(speed)
        if speed < 0 or speed > 255:
            return {"error": f"Speed must be 0-255, got {speed}"}
        printer_id, adapter = self._resolve_adapter(tool_input, request_context)
        if adapter is None:
            return {"error": f"Unknown printer: {printer_id}"}
        cmd = await adapter.set_fan_speed(speed)
        result = self._cmd_result(cmd)
        if cmd.status == CommandResultStatus.SUCCESS:
            pct = round(speed / 255 * 100)
            result["message"] = f"Fan speed set to {speed}/255 ({pct}%)"
        return result

    async def _run_gcode(self, tool_input: dict, request_context: ChatRequestContext) -> dict:
        approved_input, approval_error = self._consume_approval("run_gcode", tool_input, request_context)
        if approval_error:
            return approval_error
        effective_input = approved_input or tool_input

        command = effective_input.get("command", "").strip()
        if not command:
            return {"error": "command is required"}
        validation_error = validate_gcode_command(command)
        if validation_error:
            return {"error": validation_error}
        safe_command = _strip_gcode_comment(command)
        first_token = safe_command.split()[0].upper()
        if first_token not in self._allowed_gcode_commands:
            risk_status, reason = classify_gcode_command(safe_command)
            if risk_status == "approval_required" and approved_input is None:
                return self._create_approval(
                    request_context,
                    "run_gcode",
                    {"command": safe_command, "printer": effective_input.get("printer")},
                    (
                        "Raw G-code is not in the configured allowlist. "
                        f"{reason}. Explicit approval is required."
                    ),
                )
        printer_id, adapter = self._resolve_adapter(tool_input, request_context)
        if adapter is None:
            return {"error": f"Unknown printer: {printer_id}"}
        cmd = await adapter.run_gcode(safe_command)
        result = self._cmd_result(cmd)
        if cmd.data:
            result["data"] = cmd.data
        result["warning"] = f"Executed raw G-code: {safe_command}"
        return result

    # --- Z-offset ---

    async def _get_z_offset(self, tool_input: dict, request_context: ChatRequestContext) -> dict:
        try:
            printer_id, adapter = self._resolve_adapter(tool_input, request_context)
            if adapter is None:
                return {"error": f"Unknown printer: {printer_id}"}
            offset = await adapter.get_z_offset()
            result = {"z_offset_mm": round(offset, 4)}
            if self._should_include_printer(tool_input, request_context, printer_id):
                result["printer"] = printer_id
            return result
        except Exception as e:
            return {"error": str(e)}

    async def _set_z_offset(self, tool_input: dict, request_context: ChatRequestContext) -> dict:
        offset = tool_input.get("offset_mm")
        if offset is None:
            return {"error": "offset_mm is required"}
        printer_id, adapter = self._resolve_adapter(tool_input, request_context)
        if adapter is None:
            return {"error": f"Unknown printer: {printer_id}"}
        cmd = await adapter.set_z_offset(float(offset))
        return self._cmd_result(cmd)

    # --- File management ---

    async def _list_files(self, tool_input: dict, request_context: ChatRequestContext) -> dict:
        printer_id, adapter = self._resolve_adapter(tool_input, request_context)
        if adapter is None:
            return {"error": f"Unknown printer: {printer_id}"}
        cmd = await adapter.list_files()
        if cmd.status != CommandResultStatus.SUCCESS:
            return self._cmd_result(cmd)
        files = cmd.data or []
        result = {
            "files": [
                {
                    "filename": f.get("path", f.get("filename", "")),
                    "size_mb": round(f.get("size", 0) / 1024 / 1024, 1),
                    "modified": f.get("modified"),
                }
                for f in files
            ],
            "count": len(files),
        }
        if self._should_include_printer(tool_input, request_context, printer_id):
            result["printer"] = printer_id
        return result

    async def _delete_file(self, tool_input: dict, request_context: ChatRequestContext) -> dict:
        approved_input, approval_error = self._consume_approval(
            "delete_file", tool_input, request_context
        )
        if approval_error:
            return approval_error
        effective_input = approved_input or tool_input
        filename = effective_input.get("filename", "")
        if not filename:
            return {"error": "filename is required"}
        if approved_input is None:
            return self._create_approval(
                request_context,
                "delete_file",
                {"filename": filename, "printer": effective_input.get("printer")},
                "Deleting printer files is destructive and requires explicit approval.",
            )
        printer_id, adapter = self._resolve_adapter(tool_input, request_context)
        if adapter is None:
            return {"error": f"Unknown printer: {printer_id}"}
        cmd = await adapter.delete_file(filename)
        result = self._cmd_result(cmd)
        result["warning"] = f"Deleted file: {filename}"
        return result

    async def _get_file_metadata(self, tool_input: dict, request_context: ChatRequestContext) -> dict:
        filename = tool_input.get("filename", "")
        if not filename:
            return {"error": "filename is required"}
        printer_id, adapter = self._resolve_adapter(tool_input, request_context)
        if adapter is None:
            return {"error": f"Unknown printer: {printer_id}"}
        cmd = await adapter.get_file_metadata(filename)
        if cmd.status != CommandResultStatus.SUCCESS:
            return self._cmd_result(cmd)
        result = cmd.data or {}
        if self._should_include_printer(tool_input, request_context, printer_id):
            result["printer"] = printer_id
        return result

    async def _upload_to_printer(self, tool_input: dict, request_context: ChatRequestContext) -> dict:
        filename = tool_input.get("filename", "")
        if not filename:
            return {"error": "filename is required"}

        # Sanitize filename
        filename = os.path.basename(filename)
        local_path = Path("data/uploads") / filename

        if not local_path.exists():
            return {"error": f"File '{filename}' not found on server. Ask user to upload it first."}

        printer_id, adapter = self._resolve_adapter(tool_input, request_context)
        if adapter is None:
            return {"error": f"Unknown printer: {printer_id}"}
        cmd = await adapter.upload_file(str(local_path.absolute()), filename)
        result = self._cmd_result(cmd)
        if cmd.status == CommandResultStatus.SUCCESS:
            result["message"] = f"Successfully uploaded '{filename}' to the printer."
        return result

    # --- Manual motion ---

    async def _move_axis(self, tool_input: dict, request_context: ChatRequestContext) -> dict:
        axis = tool_input.get("axis", "")
        distance = tool_input.get("distance")
        speed = tool_input.get("speed", 3000)
        if not axis or distance is None:
            return {"error": "axis and distance are required"}
        if abs(distance) > 200:
            return {"error": f"Distance too large: {distance}mm. Max 200mm per move."}
        printer_id, adapter = self._resolve_adapter(tool_input, request_context)
        if adapter is None:
            return {"error": f"Unknown printer: {printer_id}"}
        cmd = await adapter.move_axis(axis, float(distance), int(speed))
        result = self._cmd_result(cmd)
        if cmd.status == CommandResultStatus.SUCCESS:
            result["message"] = f"Moved {axis.upper()} {distance:+.1f}mm"
        return result

    async def _extrude(self, tool_input: dict, request_context: ChatRequestContext) -> dict:
        length = tool_input.get("length")
        speed = tool_input.get("speed", 300)
        if length is None:
            return {"error": "length is required"}
        if abs(length) > 100:
            return {"error": f"Length too large: {length}mm. Max 100mm per extrude."}
        printer_id, adapter = self._resolve_adapter(tool_input, request_context)
        if adapter is None:
            return {"error": f"Unknown printer: {printer_id}"}
        cmd = await adapter.extrude(float(length), int(speed))
        result = self._cmd_result(cmd)
        if cmd.status == CommandResultStatus.SUCCESS:
            action = "Extruded" if length > 0 else "Retracted"
            result["message"] = f"{action} {abs(length):.1f}mm"
        return result

    async def _motors_off(self, tool_input: dict, request_context: ChatRequestContext) -> dict:
        printer_id, adapter = self._resolve_adapter(tool_input, request_context)
        if adapter is None:
            return {"error": f"Unknown printer: {printer_id}"}
        cmd = await adapter.motors_off()
        return self._cmd_result(cmd)

    # --- Speed/flow override ---

    async def _set_speed_override(self, tool_input: dict, request_context: ChatRequestContext) -> dict:
        percent = tool_input.get("percent")
        if percent is None:
            return {"error": "percent is required"}
        percent = int(percent)
        if percent < 10 or percent > 300:
            return {"error": f"Percent must be 10-300, got {percent}"}
        printer_id, adapter = self._resolve_adapter(tool_input, request_context)
        if adapter is None:
            return {"error": f"Unknown printer: {printer_id}"}
        cmd = await adapter.set_speed_override(percent)
        result = self._cmd_result(cmd)
        if cmd.status == CommandResultStatus.SUCCESS:
            result["message"] = f"Speed override set to {percent}%"
        return result

    async def _set_flow_override(self, tool_input: dict, request_context: ChatRequestContext) -> dict:
        percent = tool_input.get("percent")
        if percent is None:
            return {"error": "percent is required"}
        percent = int(percent)
        if percent < 50 or percent > 200:
            return {"error": f"Percent must be 50-200, got {percent}"}
        printer_id, adapter = self._resolve_adapter(tool_input, request_context)
        if adapter is None:
            return {"error": f"Unknown printer: {printer_id}"}
        cmd = await adapter.set_flow_override(percent)
        result = self._cmd_result(cmd)
        if cmd.status == CommandResultStatus.SUCCESS:
            result["message"] = f"Flow override set to {percent}%"
        return result

    # --- Macros ---

    async def _list_macros(self, tool_input: dict, request_context: ChatRequestContext) -> dict:
        printer_id, adapter = self._resolve_adapter(tool_input, request_context)
        if adapter is None:
            return {"error": f"Unknown printer: {printer_id}"}
        cmd = await adapter.list_macros()
        if cmd.status != CommandResultStatus.SUCCESS:
            return self._cmd_result(cmd)
        result = {"macros": cmd.data or []}
        if self._should_include_printer(tool_input, request_context, printer_id):
            result["printer"] = printer_id
        return result

    async def _execute_macro(self, tool_input: dict, request_context: ChatRequestContext) -> dict:
        approved_input, approval_error = self._consume_approval(
            "execute_macro", tool_input, request_context
        )
        if approval_error:
            return approval_error
        effective_input = approved_input or tool_input
        name = effective_input.get("name", "").strip()
        if not name:
            return {"error": "name is required"}
        if approved_input is None:
            return self._create_approval(
                request_context,
                "execute_macro",
                {"name": name, "printer": effective_input.get("printer")},
                "Executing arbitrary printer macros requires explicit approval.",
            )
        printer_id, adapter = self._resolve_adapter(effective_input, request_context)
        if adapter is None:
            return {"error": f"Unknown printer: {printer_id}"}
        cmd = await adapter.execute_macro(name)
        result = self._cmd_result(cmd)
        if cmd.status == CommandResultStatus.SUCCESS:
            result["message"] = f"Executed macro: {name}"
        return result

    # --- System ---

    async def _restart_firmware(self, tool_input: dict, request_context: ChatRequestContext) -> dict:
        approved_input, approval_error = self._consume_approval(
            "restart_firmware", tool_input, request_context
        )
        if approval_error:
            return approval_error
        effective_input = approved_input or tool_input
        if approved_input is None:
            return self._create_approval(
                request_context,
                "restart_firmware",
                {"printer": effective_input.get("printer")},
                "Firmware restart interrupts the printer and requires approval.",
            )
        printer_id, adapter = self._resolve_adapter(effective_input, request_context)
        if adapter is None:
            return {"error": f"Unknown printer: {printer_id}"}
        cmd = await adapter.restart_firmware()
        result = self._cmd_result(cmd)
        result["warning"] = "Firmware restart initiated"
        return result

    async def _restart_service(self, tool_input: dict, request_context: ChatRequestContext) -> dict:
        approved_input, approval_error = self._consume_approval(
            "restart_service", tool_input, request_context
        )
        if approval_error:
            return approval_error
        effective_input = approved_input or tool_input
        if approved_input is None:
            return self._create_approval(
                request_context,
                "restart_service",
                {"printer": effective_input.get("printer")},
                "Service restart interrupts printer control and requires approval.",
            )
        printer_id, adapter = self._resolve_adapter(effective_input, request_context)
        if adapter is None:
            return {"error": f"Unknown printer: {printer_id}"}
        cmd = await adapter.restart_service()
        result = self._cmd_result(cmd)
        result["warning"] = "Klipper service restart initiated"
        return result

    async def _get_system_info(self, tool_input: dict, request_context: ChatRequestContext) -> dict:
        printer_id, adapter = self._resolve_adapter(tool_input, request_context)
        if adapter is None:
            return {"error": f"Unknown printer: {printer_id}"}
        cmd = await adapter.get_system_info()
        if cmd.status != CommandResultStatus.SUCCESS:
            return self._cmd_result(cmd)
        result = cmd.data or {}
        if self._should_include_printer(tool_input, request_context, printer_id):
            result["printer"] = printer_id
        return result

    async def _run_shell(self, tool_input: dict, request_context: ChatRequestContext) -> dict:
        approved_input, approval_error = self._consume_approval(
            "run_shell", tool_input, request_context
        )
        if approval_error:
            return approval_error
        effective_input = approved_input or tool_input

        command = (effective_input.get("command") or "").strip()
        if not command:
            return {"error": "command is required"}

        risk_status, reason = classify_shell_command(command)
        if risk_status == "error":
            return {"error": reason}
        if risk_status == "approval_required" and approved_input is None:
            return self._create_approval(
                request_context,
                "run_shell",
                {
                    "command": command,
                    "timeout_seconds": effective_input.get("timeout_seconds", 30),
                    "cwd": effective_input.get("cwd"),
                    "background": bool(effective_input.get("background", False)),
                    "task_name": effective_input.get("task_name"),
                    "completion_message": effective_input.get("completion_message"),
                    "progress_interval_seconds": effective_input.get("progress_interval_seconds"),
                },
                reason,
            )

        timeout_seconds = int(effective_input.get("timeout_seconds", 30) or 30)
        cwd = effective_input.get("cwd") or None
        background = bool(effective_input.get("background", False))
        task_name = (effective_input.get("task_name") or command[:80]).strip() or "shell task"
        completion_message = effective_input.get("completion_message")
        progress_interval_seconds = effective_input.get("progress_interval_seconds")
        progress_interval_seconds = (
            int(progress_interval_seconds) if progress_interval_seconds else None
        )

        if background:
            record = self._background_tasks.create_record(
                logical_user_id=request_context.logical_user_id,
                channel=request_context.channel,
                task_name=task_name,
                command=command,
                timeout_seconds=timeout_seconds,
                progress_interval_seconds=progress_interval_seconds,
                completion_message=completion_message,
                cwd=cwd,
            )
            task = asyncio.create_task(
                self._run_shell_background(
                    record.task_id,
                    request_context,
                )
            )
            self._background_tasks.attach_task(record.task_id, task)
            return {
                "status": "scheduled",
                "task_id": record.task_id,
                "task_name": task_name,
                "command": command,
                "background": True,
            }

        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return {
                "status": "timeout",
                "command": command,
                "timeout_seconds": timeout_seconds,
            }

        return {
            "status": "success" if proc.returncode == 0 else "failure",
            "command": command,
            "exit_code": proc.returncode,
            "stdout": stdout.decode("utf-8", errors="replace")[:2000],
            "stderr": stderr.decode("utf-8", errors="replace")[:2000],
        }

    async def _run_shell_background(
        self,
        task_id: str,
        request_context: ChatRequestContext,
    ) -> None:
        record = self._background_tasks.get(task_id)
        if record is None:
            return

        record.status = "running"
        record.started_at = datetime.now().isoformat()
        proc = await asyncio.create_subprocess_shell(
            record.command,
            cwd=record.cwd or None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        communicate_task = asyncio.create_task(proc.communicate())
        started_monotonic = asyncio.get_running_loop().time()

        try:
            while True:
                remaining_timeout = None
                if record.timeout_seconds:
                    elapsed = asyncio.get_running_loop().time() - started_monotonic
                    remaining_timeout = max(0.1, record.timeout_seconds - elapsed)
                wait_timeout = remaining_timeout
                if record.progress_interval_seconds:
                    wait_timeout = min(
                        remaining_timeout if remaining_timeout is not None else record.progress_interval_seconds,
                        record.progress_interval_seconds,
                    )
                try:
                    stdout, stderr = await asyncio.wait_for(
                        asyncio.shield(communicate_task),
                        timeout=wait_timeout,
                    )
                    break
                except TimeoutError:
                    elapsed = int(asyncio.get_running_loop().time() - started_monotonic)
                    if record.timeout_seconds and elapsed >= record.timeout_seconds:
                        proc.kill()
                        stdout, stderr = await communicate_task
                        record.status = "timeout"
                        record.exit_code = proc.returncode
                        record.stdout = stdout.decode("utf-8", errors="replace")[:2000]
                        record.stderr = stderr.decode("utf-8", errors="replace")[:2000]
                        record.finished_at = datetime.now().isoformat()
                        await self._notify(
                            request_context,
                            f"Background task '{record.task_name}' timed out after {record.timeout_seconds}s.",
                            level="warning",
                            data={"task_id": record.task_id},
                        )
                        return
                    if record.progress_interval_seconds:
                        await self._notify(
                            request_context,
                            f"Background task '{record.task_name}' is still running ({elapsed}s elapsed).",
                            level="info",
                            data={"task_id": record.task_id},
                        )

            record.exit_code = proc.returncode
            record.stdout = stdout.decode("utf-8", errors="replace")[:2000]
            record.stderr = stderr.decode("utf-8", errors="replace")[:2000]
            record.finished_at = datetime.now().isoformat()
            record.status = "completed" if proc.returncode == 0 else "failed"
            if record.completion_message:
                message = record.completion_message
            elif proc.returncode == 0:
                message = f"Background task '{record.task_name}' completed successfully."
            else:
                message = (
                    f"Background task '{record.task_name}' failed with exit code {proc.returncode}."
                )
            await self._notify(
                request_context,
                message,
                level="success" if proc.returncode == 0 else "warning",
                data={"task_id": record.task_id, "exit_code": proc.returncode},
            )
        except asyncio.CancelledError:
            proc.kill()
            await proc.wait()
            record.status = "cancelled"
            record.finished_at = datetime.now().isoformat()
            await self._notify(
                request_context,
                f"Background task '{record.task_name}' was cancelled.",
                level="warning",
                data={"task_id": record.task_id},
            )
            raise

    async def _get_background_tasks(
        self,
        tool_input: dict,
        request_context: ChatRequestContext,
    ) -> dict:
        limit = int(tool_input.get("limit", 10) or 10)
        return {
            "tasks": self._background_tasks.tail(
                logical_user_id=request_context.logical_user_id,
                limit=max(1, min(limit, 50)),
            )
        }

    async def _cancel_background_task(
        self,
        tool_input: dict,
        request_context: ChatRequestContext,
    ) -> dict:
        task_id = (tool_input.get("task_id") or "").strip()
        if not task_id:
            return {"error": "task_id is required"}
        record = self._background_tasks.get(task_id)
        if record is None or record.logical_user_id != request_context.logical_user_id:
            return {"error": f"Unknown background task: {task_id}"}
        cancelled = self._background_tasks.cancel(task_id)
        return {"task_id": task_id, "cancelled": cancelled}

    async def _get_pending_approvals(
        self,
        tool_input: dict,
        request_context: ChatRequestContext,
    ) -> dict:
        limit = int(tool_input.get("limit", 10) or 10)
        return {
            "approvals": self._approval_store.tail(
                logical_user_id=request_context.logical_user_id,
                limit=max(1, min(limit, 50)),
                status="pending",
            )
        }

    # --- Camera tools ---

    async def _camera_move_step(self, tool_input: dict, _request_context: ChatRequestContext) -> dict:
        if self._camera is None:
            return {"error": "Camera is not configured"}
        direction = tool_input.get("direction", "")
        before = None
        after = None
        if hasattr(self._camera, "get_position"):
            before = await self._maybe_await(self._camera.get_position())
        result = await self._camera.move_step(direction)
        if "error" not in result and hasattr(self._camera, "get_position"):
            await asyncio.sleep(0.2)
            after = await self._maybe_await(self._camera.get_position())
        if before is not None:
            result["position_before"] = before
        if after is not None:
            result["position_after"] = after
        if "error" not in result:
            limit_status = "unknown"
            if before is not None and after is not None:
                axis = "pan" if direction in {"left", "right"} else "tilt"
                delta = abs(float(after.get(axis, 0.0)) - float(before.get(axis, 0.0)))
                limit_status = "reached" if delta < 1e-4 else "moved"
            result.setdefault("limit_status", limit_status)
        return result

    async def _camera_move_to(self, tool_input: dict, _request_context: ChatRequestContext) -> dict:
        if self._camera is None:
            return {"error": "Camera is not configured"}
        pan = tool_input.get("pan", 0)
        tilt = tool_input.get("tilt", 0)
        return await self._camera.move_to(int(pan), int(tilt))

    async def _camera_get_presets(self, _input: dict, _request_context: ChatRequestContext) -> dict:
        if self._camera is None:
            return {"error": "Camera is not configured"}
        return await self._camera.get_presets()

    async def _camera_go_to_preset(self, tool_input: dict, _request_context: ChatRequestContext) -> dict:
        if self._camera is None:
            return {"error": "Camera is not configured"}
        preset_id = tool_input.get("preset_id")
        if preset_id is None:
            return {"error": "preset_id is required"}
        return await self._camera.set_preset(int(preset_id))

    async def _camera_get_position(self, _input: dict, _request_context: ChatRequestContext) -> dict:
        if self._camera is None:
            return {"error": "Camera is not configured"}
        if not hasattr(self._camera, "get_position"):
            return {"status": "unsupported"}
        position = await self._maybe_await(self._camera.get_position())
        if position is None:
            return {"status": "unsupported"}
        return {"status": "ok", "position": position}

    async def _camera_snapshot(self, _input: dict, request_context: ChatRequestContext) -> dict:
        """Capture RTSP frame and return as image content blocks."""
        if not self._rtsp_url:
            return {"error": "Camera RTSP URL not configured"}
        rtsp_url = self._rtsp_url

        def _capture():
            import cv2

            cap = cv2.VideoCapture(rtsp_url)
            try:
                ret, frame = cap.read()
                if not ret:
                    return None
                frame = cv2.resize(frame, (640, 360))
                ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                return jpeg.tobytes() if ret else None
            finally:
                cap.release()

        jpeg_bytes = await asyncio.to_thread(_capture)
        if not jpeg_bytes:
            return {"error": "Failed to capture frame from camera"}

        b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
        snapshot_dir = Path("data/camera_snapshots")
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_dir / f"snapshot-{request_context.request_id}.jpg"
        snapshot_path.write_bytes(jpeg_bytes)

        if (
            request_context.channel == "telegram"
            and request_context.chat_id is not None
            and self._photo_sender is not None
        ):
            await self._photo_sender(request_context.chat_id, str(snapshot_path))

        return {
            "snapshot_path": str(snapshot_path),
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64,
                    },
                },
                {
                    "type": "text",
                    "text": f"Camera snapshot captured at {datetime.now().strftime('%H:%M:%S')}",
                },
            ]
        }

    # --- Helpers ---

    @staticmethod
    def _cmd_result(cmd) -> dict:
        result: dict[str, Any] = {"status": cmd.status.name}
        if cmd.message:
            result["message"] = cmd.message
        return result

    @staticmethod
    async def _maybe_await(value):
        if asyncio.iscoroutine(value) or hasattr(value, "__await__"):
            return await value
        return value
