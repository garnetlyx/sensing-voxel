"""
Anthropic tool definitions and executor for printer control
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from datetime import datetime
from typing import Any

from sensing_voxel.adapters.base import PrinterAdapter
from sensing_voxel.core.models import CommandResultStatus

logger = logging.getLogger(__name__)

# Safety limits
NOZZLE_TEMP_MAX = 300
BED_TEMP_MAX = 120
BLOCKED_GCODE_PREFIXES = ("M112",)  # Emergency stop goes through dedicated path

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
        "name": "start_print",
        "description": "Start printing a G-code file that is already uploaded to the printer.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "G-code filename on the printer",
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
            "properties": {},
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
            },
            "required": ["command"],
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
                "distance": {"type": "number", "description": "Distance in mm (negative = reverse)"},
                "speed": {"type": "integer", "description": "Feed rate in mm/min (default 3000)", "default": 3000},
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
                "speed": {"type": "integer", "description": "Feed rate in mm/min (default 300)", "default": 300},
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
            },
            "required": ["name"],
        },
    },
    # --- System ---
    {
        "name": "restart_firmware",
        "description": "Restart Klipper firmware (FIRMWARE_RESTART). Use after MCU errors.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "restart_service",
        "description": "Restart Klipper host service. Use when firmware restart isn't enough.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_system_info",
        "description": "Get system info: CPU load, memory, Klipper state and messages.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]


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
]


class ToolExecutor:
    """Dispatches tool calls to the printer adapter and optional camera."""

    def __init__(self, adapter: PrinterAdapter, camera=None, rtsp_url: str | None = None):
        self._adapter = adapter
        self._camera = camera
        self._rtsp_url = rtsp_url
        self._dispatch: dict[str, Any] = {
            "get_printer_state": self._get_printer_state,
            "get_print_progress": self._get_print_progress,
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
        }
        if camera:
            self._dispatch.update({
                "camera_move_step": self._camera_move_step,
                "camera_move_to": self._camera_move_to,
                "camera_get_presets": self._camera_get_presets,
                "camera_go_to_preset": self._camera_go_to_preset,
            })
        if rtsp_url:
            self._dispatch["camera_snapshot"] = self._camera_snapshot

    async def execute(self, tool_name: str, tool_input: dict) -> str | dict:
        """Execute a tool call and return JSON string or structured content blocks."""
        handler = self._dispatch.get(tool_name)
        if not handler:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        try:
            result = await handler(tool_input)
            # Structured content (e.g. image blocks) — pass through as-is
            if isinstance(result, dict) and "content" in result:
                return result
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Tool execution error ({tool_name}): {e}", exc_info=True)
            return json.dumps({"error": str(e)})

    # --- Read-only tools ---

    async def _get_printer_state(self, _input: dict) -> dict:
        state = await self._adapter.get_printer_state()
        return {"state": state.name}

    async def _get_print_progress(self, _input: dict) -> dict:
        progress = await self._adapter.get_print_progress()
        result: dict[str, Any] = {"state": progress.state.name}
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

    async def _start_print(self, tool_input: dict) -> dict:
        filename = tool_input.get("filename", "")
        if not filename:
            return {"error": "filename is required"}
        cmd = await self._adapter.start_print(filename)
        return self._cmd_result(cmd)

    async def _pause_print(self, _input: dict) -> dict:
        cmd = await self._adapter.pause_print()
        return self._cmd_result(cmd)

    async def _resume_print(self, _input: dict) -> dict:
        cmd = await self._adapter.resume_print()
        return self._cmd_result(cmd)

    async def _cancel_print(self, _input: dict) -> dict:
        cmd = await self._adapter.cancel_print()
        result = self._cmd_result(cmd)
        result["warning"] = "Print has been cancelled. This cannot be undone."
        return result

    # --- Temperature & control ---

    async def _set_nozzle_temp(self, tool_input: dict) -> dict:
        temp = tool_input.get("temperature")
        if temp is None:
            return {"error": "temperature is required"}
        temp = float(temp)
        if temp < 0 or temp > NOZZLE_TEMP_MAX:
            return {"error": f"Temperature must be 0-{NOZZLE_TEMP_MAX}°C, got {temp}"}
        cmd = await self._adapter.set_nozzle_temp(temp)
        result = self._cmd_result(cmd)
        if cmd.status == CommandResultStatus.SUCCESS:
            result["message"] = f"Nozzle temperature set to {int(temp)}°C"
        return result

    async def _set_bed_temp(self, tool_input: dict) -> dict:
        temp = tool_input.get("temperature")
        if temp is None:
            return {"error": "temperature is required"}
        temp = float(temp)
        if temp < 0 or temp > BED_TEMP_MAX:
            return {"error": f"Temperature must be 0-{BED_TEMP_MAX}°C, got {temp}"}
        cmd = await self._adapter.set_bed_temp(temp)
        result = self._cmd_result(cmd)
        if cmd.status == CommandResultStatus.SUCCESS:
            result["message"] = f"Bed temperature set to {int(temp)}°C"
        return result

    async def _home_axes(self, tool_input: dict) -> dict:
        axes = tool_input.get("axes", "")
        cmd = await self._adapter.home_axes(axes)
        result = self._cmd_result(cmd)
        if cmd.status == CommandResultStatus.SUCCESS:
            label = axes.upper() if axes.strip() else "all"
            result["message"] = f"Homing {label} axes"
        return result

    async def _set_fan_speed(self, tool_input: dict) -> dict:
        speed = tool_input.get("speed")
        if speed is None:
            return {"error": "speed is required"}
        speed = int(speed)
        if speed < 0 or speed > 255:
            return {"error": f"Speed must be 0-255, got {speed}"}
        cmd = await self._adapter.set_fan_speed(speed)
        result = self._cmd_result(cmd)
        if cmd.status == CommandResultStatus.SUCCESS:
            pct = round(speed / 255 * 100)
            result["message"] = f"Fan speed set to {speed}/255 ({pct}%)"
        return result

    async def _run_gcode(self, tool_input: dict) -> dict:
        command = tool_input.get("command", "").strip()
        if not command:
            return {"error": "command is required"}
        # Block dangerous commands
        upper = command.upper()
        for prefix in BLOCKED_GCODE_PREFIXES:
            if upper.startswith(prefix):
                return {"error": f"G-code '{prefix}' is blocked. Use emergency_cancel instead."}
        cmd = await self._adapter.run_gcode(command)
        result = self._cmd_result(cmd)
        result["warning"] = f"Executed raw G-code: {command}"
        return result

    # --- Z-offset ---

    async def _get_z_offset(self, _input: dict) -> dict:
        try:
            offset = await self._adapter.get_z_offset()
            return {"z_offset_mm": round(offset, 4)}
        except Exception as e:
            return {"error": str(e)}

    async def _set_z_offset(self, tool_input: dict) -> dict:
        offset = tool_input.get("offset_mm")
        if offset is None:
            return {"error": "offset_mm is required"}
        cmd = await self._adapter.set_z_offset(float(offset))
        return self._cmd_result(cmd)

    # --- File management ---

    async def _list_files(self, _input: dict) -> dict:
        cmd = await self._adapter.list_files()
        if cmd.status != CommandResultStatus.SUCCESS:
            return self._cmd_result(cmd)
        files = cmd.data or []
        return {
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

    async def _delete_file(self, tool_input: dict) -> dict:
        filename = tool_input.get("filename", "")
        if not filename:
            return {"error": "filename is required"}
        cmd = await self._adapter.delete_file(filename)
        result = self._cmd_result(cmd)
        result["warning"] = f"Deleted file: {filename}"
        return result

    async def _get_file_metadata(self, tool_input: dict) -> dict:
        filename = tool_input.get("filename", "")
        if not filename:
            return {"error": "filename is required"}
        cmd = await self._adapter.get_file_metadata(filename)
        if cmd.status != CommandResultStatus.SUCCESS:
            return self._cmd_result(cmd)
        return cmd.data or {}

    # --- Manual motion ---

    async def _move_axis(self, tool_input: dict) -> dict:
        axis = tool_input.get("axis", "")
        distance = tool_input.get("distance")
        speed = tool_input.get("speed", 3000)
        if not axis or distance is None:
            return {"error": "axis and distance are required"}
        if abs(distance) > 200:
            return {"error": f"Distance too large: {distance}mm. Max 200mm per move."}
        cmd = await self._adapter.move_axis(axis, float(distance), int(speed))
        result = self._cmd_result(cmd)
        if cmd.status == CommandResultStatus.SUCCESS:
            result["message"] = f"Moved {axis.upper()} {distance:+.1f}mm"
        return result

    async def _extrude(self, tool_input: dict) -> dict:
        length = tool_input.get("length")
        speed = tool_input.get("speed", 300)
        if length is None:
            return {"error": "length is required"}
        if abs(length) > 100:
            return {"error": f"Length too large: {length}mm. Max 100mm per extrude."}
        cmd = await self._adapter.extrude(float(length), int(speed))
        result = self._cmd_result(cmd)
        if cmd.status == CommandResultStatus.SUCCESS:
            action = "Extruded" if length > 0 else "Retracted"
            result["message"] = f"{action} {abs(length):.1f}mm"
        return result

    async def _motors_off(self, _input: dict) -> dict:
        cmd = await self._adapter.motors_off()
        return self._cmd_result(cmd)

    # --- Speed/flow override ---

    async def _set_speed_override(self, tool_input: dict) -> dict:
        percent = tool_input.get("percent")
        if percent is None:
            return {"error": "percent is required"}
        percent = int(percent)
        if percent < 10 or percent > 300:
            return {"error": f"Percent must be 10-300, got {percent}"}
        cmd = await self._adapter.set_speed_override(percent)
        result = self._cmd_result(cmd)
        if cmd.status == CommandResultStatus.SUCCESS:
            result["message"] = f"Speed override set to {percent}%"
        return result

    async def _set_flow_override(self, tool_input: dict) -> dict:
        percent = tool_input.get("percent")
        if percent is None:
            return {"error": "percent is required"}
        percent = int(percent)
        if percent < 50 or percent > 200:
            return {"error": f"Percent must be 50-200, got {percent}"}
        cmd = await self._adapter.set_flow_override(percent)
        result = self._cmd_result(cmd)
        if cmd.status == CommandResultStatus.SUCCESS:
            result["message"] = f"Flow override set to {percent}%"
        return result

    # --- Macros ---

    async def _list_macros(self, _input: dict) -> dict:
        cmd = await self._adapter.list_macros()
        if cmd.status != CommandResultStatus.SUCCESS:
            return self._cmd_result(cmd)
        return {"macros": cmd.data or []}

    async def _execute_macro(self, tool_input: dict) -> dict:
        name = tool_input.get("name", "").strip()
        if not name:
            return {"error": "name is required"}
        cmd = await self._adapter.execute_macro(name)
        result = self._cmd_result(cmd)
        if cmd.status == CommandResultStatus.SUCCESS:
            result["message"] = f"Executed macro: {name}"
        return result

    # --- System ---

    async def _restart_firmware(self, _input: dict) -> dict:
        cmd = await self._adapter.restart_firmware()
        result = self._cmd_result(cmd)
        result["warning"] = "Firmware restart initiated"
        return result

    async def _restart_service(self, _input: dict) -> dict:
        cmd = await self._adapter.restart_service()
        result = self._cmd_result(cmd)
        result["warning"] = "Klipper service restart initiated"
        return result

    async def _get_system_info(self, _input: dict) -> dict:
        cmd = await self._adapter.get_system_info()
        if cmd.status != CommandResultStatus.SUCCESS:
            return self._cmd_result(cmd)
        return cmd.data or {}

    # --- Camera tools ---

    async def _camera_move_step(self, tool_input: dict) -> dict:
        direction = tool_input.get("direction", "")
        return await self._camera.move_step(direction)

    async def _camera_move_to(self, tool_input: dict) -> dict:
        pan = tool_input.get("pan", 0)
        tilt = tool_input.get("tilt", 0)
        return await self._camera.move_to(int(pan), int(tilt))

    async def _camera_get_presets(self, _input: dict) -> dict:
        return await self._camera.get_presets()

    async def _camera_go_to_preset(self, tool_input: dict) -> dict:
        preset_id = tool_input.get("preset_id")
        if preset_id is None:
            return {"error": "preset_id is required"}
        return await self._camera.set_preset(int(preset_id))

    async def _camera_snapshot(self, _input: dict) -> dict:
        """Capture RTSP frame and return as image content blocks."""
        if not self._rtsp_url:
            return {"error": "Camera RTSP URL not configured"}

        def _capture():
            import cv2
            cap = cv2.VideoCapture(self._rtsp_url)
            try:
                ret, frame = cap.read()
                if not ret:
                    return None
                frame = cv2.resize(frame, (320, 180))
                ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                return jpeg.tobytes() if ret else None
            finally:
                cap.release()

        jpeg_bytes = await asyncio.to_thread(_capture)
        if not jpeg_bytes:
            return {"error": "Failed to capture frame from camera"}

        b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
        return {
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
