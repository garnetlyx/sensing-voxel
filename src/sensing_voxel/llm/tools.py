"""
Anthropic tool definitions and executor for printer control
"""

from __future__ import annotations

import json
import logging
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
]


class ToolExecutor:
    """Dispatches tool calls to the printer adapter."""

    def __init__(self, adapter: PrinterAdapter):
        self._adapter = adapter
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
        }

    async def execute(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool call and return JSON result string."""
        handler = self._dispatch.get(tool_name)
        if not handler:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        try:
            result = await handler(tool_input)
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

    # --- Helpers ---

    @staticmethod
    def _cmd_result(cmd) -> dict:
        result: dict[str, Any] = {"status": cmd.status.name}
        if cmd.message:
            result["message"] = cmd.message
        return result
