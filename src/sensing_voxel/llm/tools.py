"""
Anthropic tool definitions and executor for printer control
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
import os
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

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


@dataclass
class ShellToolConfig:
    """Configuration for the run_bash LLM tool."""

    enabled: bool = False
    cwd: str = "/app"
    timeout_seconds: float = 10.0
    max_output_chars: int = 8000


@dataclass
class WebSearchToolConfig:
    """Configuration for the web_search LLM tool."""

    enabled: bool = False
    searxng_url: str | None = None
    timeout_seconds: float = 10.0
    max_results: int = 5


SYSTEM_TOOLS: list[dict[str, Any]] = [
    {
        "name": "run_bash",
        "description": (
            "Run a short bash command inside the Sensing Voxel container for diagnostics, "
            "file inspection, and harmless local automation. Do not use for destructive actions "
            "unless the user explicitly confirms."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Bash command to run",
                },
                "cwd": {
                    "type": "string",
                    "description": "Optional working directory under the configured shell root",
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": "Optional timeout in seconds",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "web_search",
        "description": "Search the web through a configured SearXNG JSON API instance.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                },
                "language": {
                    "type": "string",
                    "description": "Optional search language, e.g. en or zh",
                },
            },
            "required": ["query"],
        },
    },
]


PRINTER_TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_printers",
        "description": (
            "List all configured printers, their IDs, connection status, and which one is the default. "
            "Use this before comparing printers or when the user does not specify which printer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
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
        "name": "list_printer_config_files",
        "description": (
            "List Klipper/Moonraker configuration files from the selected printer's remote "
            "config root. Use this instead of run_bash when looking for printer.cfg or "
            "other printer configuration files."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_printer_config_file",
        "description": (
            "Read a Klipper/Moonraker configuration file such as printer.cfg from the "
            "selected printer's remote config root. This does not read files from the "
            "Sensing Voxel container."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Config filename under Moonraker's config root, default printer.cfg",
                    "default": "printer.cfg",
                },
                "max_bytes": {
                    "type": "integer",
                    "description": "Maximum bytes of file content to return, default 50000",
                    "default": 50000,
                },
            },
            "required": [],
        },
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
        "name": "get_current_position",
        "description": "Get the current nozzle/toolhead position (X, Y, Z) if available.",
        "input_schema": {
            "type": "object",
            "properties": {},
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
                    "description": "Task identifier returned by a background task tool.",
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
    {
        "name": "update_printer_config",
        "description": (
            "Update a single key in a Klipper config section (e.g. nozzle_diameter in [extruder]). "
            "Automatically creates a timestamped backup before writing. "
            "Requires approval — destructive operation that modifies printer.cfg."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "printer_id": {
                    "type": "string",
                    "description": "Printer ID (e.g. 'neptune4max'). Defaults to active printer.",
                },
                "section": {
                    "type": "string",
                    "description": "Config section name without brackets (e.g. 'extruder', 'heater_bed')",
                },
                "key": {
                    "type": "string",
                    "description": "Config key to update (e.g. 'nozzle_diameter')",
                },
                "value": {
                    "type": "string",
                    "description": "New value (e.g. '0.600')",
                },
            },
            "required": ["section", "key", "value"],
        },
    },
]


def _add_optional_printer_id_to_tools() -> None:
    """Add a shared optional printer_id argument to all printer-specific tools."""
    printer_id_schema = {
        "type": "string",
        "description": (
            "Optional printer ID. Use list_printers to discover IDs. "
            "If omitted, the current default/active printer is used."
        ),
    }
    for tool in PRINTER_TOOLS:
        if tool.get("name") in {
            "list_printers",
            "get_background_tasks",
            "cancel_background_task",
            "get_pending_approvals",
        }:
            continue
        schema = tool.setdefault("input_schema", {"type": "object"})
        props = schema.setdefault("properties", {})
        props.setdefault("printer_id", printer_id_schema)
        tool["description"] = (
            f'{tool.get("description", "")} '
            "Accepts optional printer_id to target a specific printer."
        ).strip()


_add_optional_printer_id_to_tools()


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


MEMO_TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_printer_memo",
        "description": (
            "Read SVA's stored notes about a printer's physical attributes that "
            "cannot be queried from Klipper (e.g. nozzle diameter, bed surface type, "
            "filament brand). Pass a specific key to read one attribute, or omit key "
            "to read all stored notes for the printer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "printer_id": {
                    "type": "string",
                    "description": "Printer ID (e.g. 'neptune4max'). Defaults to active printer.",
                },
                "key": {
                    "type": "string",
                    "description": "Specific attribute key (e.g. 'nozzle_diameter'). Omit to read all.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "set_printer_memo",
        "description": (
            "Store or update a note about a printer's physical attribute that SVA "
            "should remember across sessions (e.g. nozzle_diameter=0.6, "
            "bed_surface=PEI_textured, filament_brand=Polymaker). "
            "Only store information that cannot be read from Klipper."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "printer_id": {
                    "type": "string",
                    "description": "Printer ID (e.g. 'neptune4max'). Defaults to active printer.",
                },
                "key": {
                    "type": "string",
                    "description": "Attribute key (e.g. 'nozzle_diameter')",
                },
                "value": {
                    "type": "string",
                    "description": "Attribute value (e.g. '0.6')",
                },
            },
            "required": ["key", "value"],
        },
    },
]


class ToolExecutor:
    """Dispatches tool calls to one or more printer adapters and optional camera."""

    # Tools that move physical hardware — rate-limited to prevent congestion.
    _PHYSICAL_TOOLS: set[str] = frozenset({
        "camera_move_step",
        "camera_move_to",
        "camera_go_to_preset",
        "camera_snapshot",
        "run_gcode",
        "move_axis",
        "extrude",
        "home_axes",
        "set_nozzle_temp",
        "set_bed_temp",
        "set_fan_speed",
        "set_z_offset",
        "motors_off",
        "set_speed_override",
        "set_flow_override",
        "execute_macro",
        "start_print",
        "pause_print",
        "resume_print",
        "cancel_print",
        "restart_firmware",
        "restart_service",
        "upload_to_printer",
        "delete_file",
        "run_shell",
    })

    def __init__(
        self,
        adapter: PrinterAdapter | None = None,
        camera=None,
        rtsp_url: str | None = None,
        adapters: dict[str, PrinterAdapter] | None = None,
        active_printer: str | None = None,
        shell: ShellToolConfig | dict[str, Any] | None = None,
        web_search: WebSearchToolConfig | dict[str, Any] | None = None,
        audit_logger: ToolAuditLogger | None = None,
        photo_sender=None,
        text_notifier=None,
        background_tasks: BackgroundTaskManager | None = None,
        approval_store: PendingApprovalStore | None = None,
        allowed_gcode_commands: set[str] | None = None,
        tool_cooldown_seconds: float = 0.0,
        memo_repository=None,
    ):
        if adapters is None:
            adapters = {}
        if adapter is not None and not adapters:
            adapter_id = active_printer or "default"
            adapters = {adapter_id: adapter}
            active_printer = adapter_id

        self._adapters = adapters
        self._active_printer = active_printer or next(iter(adapters), None)
        self._adapter = adapters.get(self._active_printer) if self._active_printer else adapter
        self._camera: Any | None = camera
        self._rtsp_url: str | None = rtsp_url
        self._shell_config = self._coerce_shell_config(shell)
        self._web_search_config = self._coerce_web_search_config(web_search)
        self._audit_logger = audit_logger
        self._photo_sender = photo_sender
        self._text_notifier = text_notifier
        self._background_tasks = background_tasks or BackgroundTaskManager()
        self._approval_store = approval_store or PendingApprovalStore()
        self._allowed_gcode_commands = {cmd.upper() for cmd in (allowed_gcode_commands or set())}
        self._tool_cooldown_seconds = tool_cooldown_seconds
        self._last_tool_call: dict[str, float] = {}
        self._memo_repository = memo_repository
        self._dispatch: dict[str, Any] = {
            "list_printers": self._list_printers_tool,
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
            "list_printer_config_files": self._list_printer_config_files,
            "get_printer_config_file": self._get_printer_config_file,
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
            "get_background_tasks": self._get_background_tasks,
            "cancel_background_task": self._cancel_background_task,
            "get_pending_approvals": self._get_pending_approvals,
            "run_shell": self._run_shell,
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
        if self._shell_config.enabled:
            self._dispatch["run_bash"] = self._run_bash
        if self._web_search_config.enabled:
            self._dispatch["web_search"] = self._web_search
        if self._memo_repository:
            self._dispatch["get_printer_memo"] = self._get_printer_memo
            self._dispatch["set_printer_memo"] = self._set_printer_memo
        self._dispatch["update_printer_config"] = self._update_printer_config

    @staticmethod
    def _coerce_shell_config(config: ShellToolConfig | dict[str, Any] | None) -> ShellToolConfig:
        if config is None:
            return ShellToolConfig()
        if isinstance(config, ShellToolConfig):
            return config
        if is_dataclass(config):
            return ShellToolConfig(**asdict(config))
        return ShellToolConfig(**config)

    @staticmethod
    def _coerce_web_search_config(
        config: WebSearchToolConfig | dict[str, Any] | None,
    ) -> WebSearchToolConfig:
        if config is None:
            return WebSearchToolConfig()
        if isinstance(config, WebSearchToolConfig):
            return config
        if is_dataclass(config):
            return WebSearchToolConfig(**asdict(config))
        return WebSearchToolConfig(**config)

    async def execute(
        self,
        tool_name: str,
        tool_input: dict | None,
        request_context: ChatRequestContext | None = None,
    ) -> str | dict:
        """Execute a tool call and return JSON string or structured content blocks."""
        if tool_input is None:
            tool_input = {}
        request_context = request_context or ChatRequestContext(
            logical_user_id="unknown",
            channel="unknown",
            active_printer=None,
        )

        if (
            self._tool_cooldown_seconds > 0
            and tool_name in self._PHYSICAL_TOOLS
        ):
            last = self._last_tool_call.get(tool_name)
            now = asyncio.get_event_loop().time()
            if last is not None:
                elapsed = now - last
                if elapsed < self._tool_cooldown_seconds:
                    wait = self._tool_cooldown_seconds - elapsed
                    logger.info(
                        f"Tool cooldown: {tool_name} waiting {wait:.1f}s "
                        f"(last called {elapsed:.1f}s ago)"
                    )
                    await asyncio.sleep(wait)

        handler = self._dispatch.get(tool_name)
        if not handler:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        if self._tool_cooldown_seconds > 0 and tool_name in self._PHYSICAL_TOOLS:
            self._last_tool_call[tool_name] = asyncio.get_event_loop().time()

        needs_context = "request_context" in inspect.signature(handler).parameters
        started_at = datetime.now().isoformat()
        try:
            if needs_context:
                result = await handler(tool_input, request_context)
            else:
                result = await handler(tool_input)
            self._write_audit_record(
                request_context, tool_name, tool_input, result, started_at, status="success"
            )
            # Structured content (e.g. image blocks) — pass through as-is.
            # Plain JSON tools may also have a string field named "content"
            # (for example remote config file reads), so only pass through
            # actual content-block lists.
            if (
                isinstance(result, dict)
                and "content" in result
                and isinstance(result["content"], list)
            ):
                return result
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Tool execution error ({tool_name}): {e}", exc_info=True)
            self._write_audit_record(
                request_context,
                tool_name,
                tool_input,
                {"error": str(e)},
                started_at,
                status="error",
            )
            return json.dumps({"error": str(e)})

    def _select_adapter(self, tool_input: dict | None = None) -> tuple[str | None, PrinterAdapter | None]:
        """Resolve target adapter from optional tool_input['printer_id']."""
        if tool_input is None:
            tool_input = {}
        printer_id = tool_input.get("printer_id") or self._active_printer
        if not printer_id:
            return None, None
        return printer_id, self._adapters.get(printer_id)

    async def _list_printers_tool(self, _input: dict) -> dict:
        return {
            "active_printer": self._active_printer,
            "printers": [
                {
                    "printer_id": pid,
                    "active": pid == self._active_printer,
                    "type": type(adapter).__name__.replace("Adapter", "").lower(),
                }
                for pid, adapter in self._adapters.items()
            ],
        }

    def _require_adapter(self, tool_input: dict | None = None) -> tuple[str, PrinterAdapter] | dict:
        printer_id, adapter = self._select_adapter(tool_input)
        if adapter is None:
            return {
                "error": (
                    f"Unknown printer_id '{printer_id}'. "
                    f"Available printers: {', '.join(self._adapters) or 'none'}"
                )
            }
        return printer_id or "default", adapter

    # --- Read-only tools ---

    async def _get_printer_state(self, tool_input: dict) -> dict:
        selected = self._require_adapter(tool_input)
        if isinstance(selected, dict):
            return selected
        printer_id, adapter = selected
        state = await adapter.get_printer_state()
        return {"printer_id": printer_id, "state": state.name}

    async def _get_print_progress(self, tool_input: dict) -> dict:
        selected = self._require_adapter(tool_input)
        if isinstance(selected, dict):
            return selected
        printer_id, adapter = selected
        progress = await adapter.get_print_progress()
        result: dict[str, Any] = {"printer_id": printer_id, "state": progress.state.name}
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

    async def _start_print(
        self,
        tool_input: dict,
        request_context: ChatRequestContext | None = None,
    ) -> dict:
        request_context = request_context or ChatRequestContext(
            logical_user_id="unknown", channel="unknown", active_printer=None
        )
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
                {"filename": filename, "printer_id": effective_input.get("printer_id")},
                "Starting a print job changes printer state and requires explicit approval.",
            )
        selected = self._require_adapter(effective_input)
        if isinstance(selected, dict):
            return selected
        _printer_id, adapter = selected
        cmd = await adapter.start_print(filename)
        return self._cmd_result(cmd)

    async def _pause_print(self, tool_input: dict) -> dict:
        selected = self._require_adapter(tool_input)
        if isinstance(selected, dict):
            return selected
        _printer_id, adapter = selected
        cmd = await adapter.pause_print()
        return self._cmd_result(cmd)

    async def _resume_print(self, tool_input: dict) -> dict:
        selected = self._require_adapter(tool_input)
        if isinstance(selected, dict):
            return selected
        _printer_id, adapter = selected
        cmd = await adapter.resume_print()
        return self._cmd_result(cmd)

    async def _cancel_print(
        self,
        tool_input: dict,
        request_context: ChatRequestContext | None = None,
    ) -> dict:
        request_context = request_context or ChatRequestContext(
            logical_user_id="unknown", channel="unknown", active_printer=None
        )
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
                {"printer_id": effective_input.get("printer_id")},
                "Cancelling an active print job is destructive and requires approval.",
            )
        selected = self._require_adapter(effective_input)
        if isinstance(selected, dict):
            return selected
        _printer_id, adapter = selected
        cmd = await adapter.cancel_print()
        result = self._cmd_result(cmd)
        result["warning"] = "Print has been cancelled. This cannot be undone."
        return result

    # --- Temperature & control ---

    async def _set_nozzle_temp(self, tool_input: dict) -> dict:
        selected = self._require_adapter(tool_input)
        if isinstance(selected, dict):
            return selected
        _printer_id, adapter = selected
        temp = tool_input.get("temperature")
        if temp is None:
            return {"error": "temperature is required"}
        temp = float(temp)
        if temp < 0 or temp > NOZZLE_TEMP_MAX:
            return {"error": f"Temperature must be 0-{NOZZLE_TEMP_MAX}°C, got {temp}"}
        cmd = await adapter.set_nozzle_temp(temp)
        result = self._cmd_result(cmd)
        if cmd.status == CommandResultStatus.SUCCESS:
            result["message"] = f"Nozzle temperature set to {int(temp)}°C"
        return result

    async def _set_bed_temp(self, tool_input: dict) -> dict:
        selected = self._require_adapter(tool_input)
        if isinstance(selected, dict):
            return selected
        _printer_id, adapter = selected
        temp = tool_input.get("temperature")
        if temp is None:
            return {"error": "temperature is required"}
        temp = float(temp)
        if temp < 0 or temp > BED_TEMP_MAX:
            return {"error": f"Temperature must be 0-{BED_TEMP_MAX}°C, got {temp}"}
        cmd = await adapter.set_bed_temp(temp)
        result = self._cmd_result(cmd)
        if cmd.status == CommandResultStatus.SUCCESS:
            result["message"] = f"Bed temperature set to {int(temp)}°C"
        return result

    async def _home_axes(self, tool_input: dict) -> dict:
        selected = self._require_adapter(tool_input)
        if isinstance(selected, dict):
            return selected
        _printer_id, adapter = selected
        axes = tool_input.get("axes", "")
        cmd = await adapter.home_axes(axes)
        result = self._cmd_result(cmd)
        if cmd.status == CommandResultStatus.SUCCESS:
            label = axes.upper() if axes.strip() else "all"
            result["message"] = f"Homing {label} axes"
        return result

    async def _set_fan_speed(self, tool_input: dict) -> dict:
        selected = self._require_adapter(tool_input)
        if isinstance(selected, dict):
            return selected
        _printer_id, adapter = selected
        speed = tool_input.get("speed")
        if speed is None:
            return {"error": "speed is required"}
        speed = int(speed)
        if speed < 0 or speed > 255:
            return {"error": f"Speed must be 0-255, got {speed}"}
        cmd = await adapter.set_fan_speed(speed)
        result = self._cmd_result(cmd)
        if cmd.status == CommandResultStatus.SUCCESS:
            pct = round(speed / 255 * 100)
            result["message"] = f"Fan speed set to {speed}/255 ({pct}%)"
        return result

    async def _run_gcode(
        self,
        tool_input: dict,
        request_context: ChatRequestContext | None = None,
    ) -> dict:
        request_context = request_context or ChatRequestContext(
            logical_user_id="unknown", channel="unknown", active_printer=None
        )
        approved_input, approval_error = self._consume_approval(
            "run_gcode", tool_input, request_context
        )
        if approval_error:
            return approval_error
        effective_input = approved_input or tool_input

        selected = self._require_adapter(effective_input)
        if isinstance(selected, dict):
            return selected
        printer_id, adapter = selected
        command = effective_input.get("command", "").strip()
        if not command:
            return {"error": "command is required"}
        validation_error = validate_gcode_command(command)
        if validation_error:
            return {"error": validation_error}
        safe_command = _strip_gcode_comment(command)
        first_token = safe_command.split()[0].upper() if safe_command.split() else ""
        if first_token not in self._allowed_gcode_commands:
            risk_status, reason = classify_gcode_command(safe_command)
            if risk_status == "approval_required" and approved_input is None:
                return self._create_approval(
                    request_context,
                    "run_gcode",
                    {"command": safe_command, "printer_id": printer_id},
                    (
                        "Raw G-code is not in the configured allowlist. "
                        f"{reason}. Explicit approval is required."
                    ),
                )
        cmd = await adapter.run_gcode(safe_command)
        result = self._cmd_result(cmd)
        if cmd.data:
            result["data"] = cmd.data
        result["warning"] = f"Executed raw G-code: {safe_command}"
        return result

    # --- Z-offset ---

    async def _get_z_offset(self, tool_input: dict) -> dict:
        selected = self._require_adapter(tool_input)
        if isinstance(selected, dict):
            return selected
        printer_id, adapter = selected
        try:
            offset = await adapter.get_z_offset()
            return {"printer_id": printer_id, "z_offset_mm": round(offset, 4)}
        except Exception as e:
            return {"printer_id": printer_id, "error": str(e)}

    async def _set_z_offset(self, tool_input: dict) -> dict:
        selected = self._require_adapter(tool_input)
        if isinstance(selected, dict):
            return selected
        _printer_id, adapter = selected
        offset = tool_input.get("offset_mm")
        if offset is None:
            return {"error": "offset_mm is required"}
        cmd = await adapter.set_z_offset(float(offset))
        return self._cmd_result(cmd)

    # --- File management ---

    async def _list_files(self, tool_input: dict) -> dict:
        selected = self._require_adapter(tool_input)
        if isinstance(selected, dict):
            return selected
        printer_id, adapter = selected
        cmd = await adapter.list_files()
        if cmd.status != CommandResultStatus.SUCCESS:
            return self._cmd_result(cmd)
        files = cmd.data or []
        return {
            "printer_id": printer_id,
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

    async def _list_printer_config_files(self, tool_input: dict) -> dict:
        selected = self._require_adapter(tool_input)
        if isinstance(selected, dict):
            return selected
        printer_id, adapter = selected
        list_config_files = getattr(adapter, "list_config_files", None)
        if list_config_files is None or not inspect.iscoroutinefunction(list_config_files):
            return {
                "printer_id": printer_id,
                "error": "This printer adapter does not support remote config file listing",
            }
        cmd = await list_config_files()
        if cmd.status != CommandResultStatus.SUCCESS:
            return self._cmd_result(cmd)
        files = cmd.data or []
        return {
            "printer_id": printer_id,
            "files": [
                {
                    "filename": f.get("path", f.get("filename", f.get("name", ""))),
                    "size_bytes": f.get("size"),
                    "modified": f.get("modified"),
                }
                for f in files
            ],
            "count": len(files),
            "note": "These are remote Moonraker/Klipper config files, not local /app files.",
        }

    async def _get_printer_config_file(self, tool_input: dict) -> dict:
        selected = self._require_adapter(tool_input)
        if isinstance(selected, dict):
            return selected
        printer_id, adapter = selected
        get_config_file = getattr(adapter, "get_config_file", None)
        if get_config_file is None or not inspect.iscoroutinefunction(get_config_file):
            return {
                "printer_id": printer_id,
                "error": "This printer adapter does not support remote config file reads",
            }
        filename = tool_input.get("filename") or "printer.cfg"
        max_bytes = int(tool_input.get("max_bytes") or 50000)
        cmd = await get_config_file(filename=filename, max_bytes=max_bytes)
        if cmd.status != CommandResultStatus.SUCCESS:
            return self._cmd_result(cmd)
        data = cmd.data or {}
        return {
            "printer_id": printer_id,
            "filename": data.get("filename", filename),
            "content": data.get("content", ""),
            "size_bytes": data.get("size_bytes"),
            "truncated": data.get("truncated", False),
            "note": "Read from remote Moonraker/Klipper config root, not local /app.",
        }

    async def _delete_file(
        self,
        tool_input: dict,
        request_context: ChatRequestContext | None = None,
    ) -> dict:
        request_context = request_context or ChatRequestContext(
            logical_user_id="unknown", channel="unknown", active_printer=None
        )
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
                {"filename": filename, "printer_id": effective_input.get("printer_id")},
                "Deleting printer files is destructive and requires explicit approval.",
            )
        selected = self._require_adapter(tool_input)
        if isinstance(selected, dict):
            return selected
        _printer_id, adapter = selected
        cmd = await adapter.delete_file(filename)
        result = self._cmd_result(cmd)
        result["warning"] = f"Deleted file: {filename}"
        return result

    async def _get_file_metadata(self, tool_input: dict) -> dict:
        selected = self._require_adapter(tool_input)
        if isinstance(selected, dict):
            return selected
        printer_id, adapter = selected
        filename = tool_input.get("filename", "")
        if not filename:
            return {"error": "filename is required"}
        cmd = await adapter.get_file_metadata(filename)
        if cmd.status != CommandResultStatus.SUCCESS:
            return self._cmd_result(cmd)
        result = cmd.data or {}
        if isinstance(result, dict):
            result.setdefault("printer_id", printer_id)
        return result

    async def _upload_to_printer(self, tool_input: dict) -> dict:
        selected = self._require_adapter(tool_input)
        if isinstance(selected, dict):
            return selected
        _printer_id, adapter = selected
        filename = tool_input.get("filename", "")
        if not filename:
            return {"error": "filename is required"}

        # Sanitize filename
        filename = os.path.basename(filename)
        local_path = Path("data/uploads") / filename

        if not local_path.exists():
            return {"error": f"File '{filename}' not found on server. Ask user to upload it first."}

        cmd = await adapter.upload_file(str(local_path.absolute()), filename)
        result = self._cmd_result(cmd)
        if cmd.status == CommandResultStatus.SUCCESS:
            result["message"] = f"Successfully uploaded '{filename}' to the printer."
        return result

    # --- Manual motion ---

    async def _move_axis(self, tool_input: dict) -> dict:
        selected = self._require_adapter(tool_input)
        if isinstance(selected, dict):
            return selected
        _printer_id, adapter = selected
        axis = tool_input.get("axis", "")
        distance = tool_input.get("distance")
        speed = tool_input.get("speed", 3000)
        if not axis or distance is None:
            return {"error": "axis and distance are required"}
        if abs(distance) > 200:
            return {"error": f"Distance too large: {distance}mm. Max 200mm per move."}
        cmd = await adapter.move_axis(axis, float(distance), int(speed))
        result = self._cmd_result(cmd)
        if cmd.status == CommandResultStatus.SUCCESS:
            result["message"] = f"Moved {axis.upper()} {distance:+.1f}mm"
        return result

    async def _extrude(self, tool_input: dict) -> dict:
        selected = self._require_adapter(tool_input)
        if isinstance(selected, dict):
            return selected
        _printer_id, adapter = selected
        length = tool_input.get("length")
        speed = tool_input.get("speed", 300)
        if length is None:
            return {"error": "length is required"}
        if abs(length) > 100:
            return {"error": f"Length too large: {length}mm. Max 100mm per extrude."}
        cmd = await adapter.extrude(float(length), int(speed))
        result = self._cmd_result(cmd)
        if cmd.status == CommandResultStatus.SUCCESS:
            action = "Extruded" if length > 0 else "Retracted"
            result["message"] = f"{action} {abs(length):.1f}mm"
        return result

    async def _motors_off(self, tool_input: dict) -> dict:
        selected = self._require_adapter(tool_input)
        if isinstance(selected, dict):
            return selected
        _printer_id, adapter = selected
        cmd = await adapter.motors_off()
        return self._cmd_result(cmd)

    # --- Speed/flow override ---

    async def _set_speed_override(self, tool_input: dict) -> dict:
        selected = self._require_adapter(tool_input)
        if isinstance(selected, dict):
            return selected
        _printer_id, adapter = selected
        percent = tool_input.get("percent")
        if percent is None:
            return {"error": "percent is required"}
        percent = int(percent)
        if percent < 10 or percent > 300:
            return {"error": f"Percent must be 10-300, got {percent}"}
        cmd = await adapter.set_speed_override(percent)
        result = self._cmd_result(cmd)
        if cmd.status == CommandResultStatus.SUCCESS:
            result["message"] = f"Speed override set to {percent}%"
        return result

    async def _set_flow_override(self, tool_input: dict) -> dict:
        selected = self._require_adapter(tool_input)
        if isinstance(selected, dict):
            return selected
        _printer_id, adapter = selected
        percent = tool_input.get("percent")
        if percent is None:
            return {"error": "percent is required"}
        percent = int(percent)
        if percent < 50 or percent > 200:
            return {"error": f"Percent must be 50-200, got {percent}"}
        cmd = await adapter.set_flow_override(percent)
        result = self._cmd_result(cmd)
        if cmd.status == CommandResultStatus.SUCCESS:
            result["message"] = f"Flow override set to {percent}%"
        return result

    # --- Macros ---

    async def _list_macros(self, tool_input: dict) -> dict:
        selected = self._require_adapter(tool_input)
        if isinstance(selected, dict):
            return selected
        printer_id, adapter = selected
        cmd = await adapter.list_macros()
        if cmd.status != CommandResultStatus.SUCCESS:
            return self._cmd_result(cmd)
        return {"printer_id": printer_id, "macros": cmd.data or []}

    async def _execute_macro(
        self,
        tool_input: dict,
        request_context: ChatRequestContext | None = None,
    ) -> dict:
        request_context = request_context or ChatRequestContext(
            logical_user_id="unknown", channel="unknown", active_printer=None
        )
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
                {"name": name, "printer_id": effective_input.get("printer_id")},
                "Executing arbitrary printer macros requires explicit approval.",
            )
        selected = self._require_adapter(effective_input)
        if isinstance(selected, dict):
            return selected
        _printer_id, adapter = selected
        cmd = await adapter.execute_macro(name)
        result = self._cmd_result(cmd)
        if cmd.status == CommandResultStatus.SUCCESS:
            result["message"] = f"Executed macro: {name}"
        return result

    # --- System ---

    async def _restart_firmware(
        self,
        tool_input: dict,
        request_context: ChatRequestContext | None = None,
    ) -> dict:
        request_context = request_context or ChatRequestContext(
            logical_user_id="unknown", channel="unknown", active_printer=None
        )
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
                {"printer_id": effective_input.get("printer_id")},
                "Firmware restart interrupts the printer and requires approval.",
            )
        selected = self._require_adapter(effective_input)
        if isinstance(selected, dict):
            return selected
        _printer_id, adapter = selected
        cmd = await adapter.restart_firmware()
        result = self._cmd_result(cmd)
        result["warning"] = "Firmware restart initiated"
        return result

    async def _restart_service(
        self,
        tool_input: dict,
        request_context: ChatRequestContext | None = None,
    ) -> dict:
        request_context = request_context or ChatRequestContext(
            logical_user_id="unknown", channel="unknown", active_printer=None
        )
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
                {"printer_id": effective_input.get("printer_id")},
                "Service restart interrupts printer control and requires approval.",
            )
        selected = self._require_adapter(effective_input)
        if isinstance(selected, dict):
            return selected
        _printer_id, adapter = selected
        cmd = await adapter.restart_service()
        result = self._cmd_result(cmd)
        result["warning"] = "Klipper service restart initiated"
        return result

    async def _get_system_info(self, tool_input: dict) -> dict:
        selected = self._require_adapter(tool_input)
        if isinstance(selected, dict):
            return selected
        printer_id, adapter = selected
        cmd = await adapter.get_system_info()
        if cmd.status != CommandResultStatus.SUCCESS:
            return self._cmd_result(cmd)
        result = cmd.data or {}
        if isinstance(result, dict):
            result.setdefault("printer_id", printer_id)
        return result


    # --- System utility tools ---

    async def _run_bash(self, tool_input: dict) -> dict:
        command = str(tool_input.get("command", "")).strip()
        if not command:
            return {"error": "command is required"}

        timeout = min(
            float(tool_input.get("timeout_seconds") or self._shell_config.timeout_seconds),
            self._shell_config.timeout_seconds,
        )
        root = Path(self._shell_config.cwd).resolve()
        requested_cwd = tool_input.get("cwd") or "."
        cwd = (root / str(requested_cwd)).resolve()
        if cwd != root and root not in cwd.parents:
            return {"error": f"cwd must stay under {root}"}
        if not cwd.exists() or not cwd.is_dir():
            return {"error": f"cwd does not exist: {cwd}"}

        proc = await asyncio.create_subprocess_exec(
            "/bin/bash",
            "-lc",
            command,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        timed_out = False
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            timed_out = True
            proc.kill()
            stdout_b, stderr_b = await proc.communicate()

        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")
        truncated = False
        max_chars = self._shell_config.max_output_chars
        if len(stdout) > max_chars:
            stdout = stdout[:max_chars] + "\n... [stdout truncated]"
            truncated = True
        if len(stderr) > max_chars:
            stderr = stderr[:max_chars] + "\n... [stderr truncated]"
            truncated = True

        return {
            "command": command,
            "cwd": str(cwd),
            "exit_code": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": timed_out,
            "truncated": truncated,
        }

    async def _web_search(self, tool_input: dict) -> dict:
        query = str(tool_input.get("query", "")).strip()
        if not query:
            return {"error": "query is required"}
        if not self._web_search_config.searxng_url:
            return {"error": "web_search is enabled but searxng_url is not configured"}

        max_results = int(tool_input.get("max_results") or self._web_search_config.max_results)
        max_results = max(1, min(max_results, self._web_search_config.max_results))
        params = {"q": query, "format": "json"}
        if tool_input.get("language"):
            params["language"] = str(tool_input["language"])

        base_url = self._web_search_config.searxng_url.rstrip("/") + "/"
        url = urljoin(base_url, "search")
        timeout = self._web_search_config.timeout_seconds

        import aiohttp

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.get(url, params=params) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    return {"error": f"SearXNG request failed: HTTP {resp.status}", "details": text[:500]}
                payload = await resp.json()

        results = []
        for item in payload.get("results", [])[:max_results]:
            results.append(
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "content": item.get("content") or item.get("snippet"),
                    "engine": item.get("engine"),
                }
            )
        return {"query": query, "results": results, "count": len(results)}

    # --- Background shell execution ---

    async def _run_shell(
        self,
        tool_input: dict,
        request_context: ChatRequestContext | None = None,
    ) -> dict:
        request_context = request_context or ChatRequestContext(
            logical_user_id="unknown", channel="unknown", active_printer=None
        )
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
            task = asyncio.create_task(self._run_shell_background(record.task_id, request_context))
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
            return {"status": "timeout", "command": command, "timeout_seconds": timeout_seconds}

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
                        remaining_timeout
                        if remaining_timeout is not None
                        else record.progress_interval_seconds,
                        record.progress_interval_seconds,
                    )
                try:
                    stdout, stderr = await asyncio.wait_for(
                        asyncio.shield(communicate_task), timeout=wait_timeout
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

    # --- Position / background tasks / approvals ---

    async def _get_current_position(self, tool_input: dict) -> dict:
        selected = self._require_adapter(tool_input)
        if isinstance(selected, dict):
            return selected
        printer_id, adapter = selected
        if not hasattr(adapter, "get_current_position"):
            return {"printer_id": printer_id, "error": "unsupported"}
        position = await adapter.get_current_position()
        if position is None:
            return {"printer_id": printer_id, "position": None}
        return {"printer_id": printer_id, "position": position}

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

    async def _camera_move_step(self, tool_input: dict) -> dict:
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

    async def _camera_move_to(self, tool_input: dict) -> dict:
        if self._camera is None:
            return {"error": "Camera is not configured"}
        pan = tool_input.get("pan", 0)
        tilt = tool_input.get("tilt", 0)
        return await self._camera.move_to(int(pan), int(tilt))

    async def _camera_get_presets(self, _input: dict) -> dict:
        if self._camera is None:
            return {"error": "Camera is not configured"}
        return await self._camera.get_presets()

    async def _camera_go_to_preset(self, tool_input: dict) -> dict:
        if self._camera is None:
            return {"error": "Camera is not configured"}
        preset_id = tool_input.get("preset_id")
        if preset_id is None:
            return {"error": "preset_id is required"}
        return await self._camera.set_preset(int(preset_id))

    async def _camera_get_position(self, _input: dict) -> dict:
        if self._camera is None:
            return {"error": "Camera is not configured"}
        if not hasattr(self._camera, "get_position"):
            return {"status": "unsupported"}
        position = await self._maybe_await(self._camera.get_position())
        if position is None:
            return {"status": "unsupported"}
        return {"status": "ok", "position": position}

    async def _camera_snapshot(
        self,
        _input: dict,
        request_context: ChatRequestContext | None = None,
    ) -> dict:
        """Capture RTSP frame and return as image content blocks."""
        request_context = request_context or ChatRequestContext(
            logical_user_id="unknown", channel="unknown", active_printer=None
        )
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
            try:
                await self._photo_sender(request_context.chat_id, str(snapshot_path))
            except Exception:
                logger.warning("Failed to send camera snapshot photo", exc_info=True)

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

    # --- Audit / approval / notification ---

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
        if isinstance(result, dict) and isinstance(result.get("content"), list):
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
        record = self._approval_store.approve(
            token, request_context.logical_user_id, tool_name
        )
        if record is None:
            return None, {"error": f"Invalid or expired approval_token for {tool_name}"}
        merged = dict(record.payload)
        merged.update({k: v for k, v in tool_input.items() if k != "approval_token"})
        return merged, None

    # --- Printer memo tools ---

    async def _get_printer_memo(self, tool_input: dict) -> dict:
        if self._memo_repository is None:
            return {"error": "Memo storage is not configured"}
        selected = self._require_adapter(tool_input)
        if isinstance(selected, dict):
            return selected
        printer_id, _adapter = selected
        key = tool_input.get("key")
        if key:
            record = self._memo_repository.get(printer_id, key)
            if record is None:
                return {"printer_id": printer_id, "key": key, "found": False}
            return {
                "printer_id": record.printer_id,
                "key": record.key,
                "value": record.value,
                "updated_at": str(record.updated_at) if record.updated_at else None,
                "found": True,
            }
        else:
            all_memos = self._memo_repository.get_as_dict(printer_id)
            return {"printer_id": printer_id, "memos": all_memos, "count": len(all_memos)}

    async def _set_printer_memo(self, tool_input: dict) -> dict:
        if self._memo_repository is None:
            return {"error": "Memo storage is not configured"}
        selected = self._require_adapter(tool_input)
        if isinstance(selected, dict):
            return selected
        printer_id, _adapter = selected
        key = tool_input.get("key", "").strip()
        value = str(tool_input.get("value", "")).strip()
        if not key or not value:
            return {"error": "Both 'key' and 'value' are required"}
        self._memo_repository.upsert(printer_id, key, value)
        return {
            "printer_id": printer_id,
            "key": key,
            "value": value,
            "status": "saved",
        }

    async def _update_printer_config(
        self,
        tool_input: dict,
        request_context: ChatRequestContext | None = None,
    ) -> dict:
        request_context = request_context or ChatRequestContext(
            logical_user_id="unknown", channel="unknown", active_printer=None,
        )
        approved_input, approval_error = self._consume_approval(
            "update_printer_config", tool_input, request_context
        )
        if approval_error:
            return approval_error

        section = tool_input.get("section", "").strip()
        key = tool_input.get("key", "").strip()
        value = str(tool_input.get("value", "")).strip()
        if not section or not key:
            return {"error": "Both 'section' and 'key' are required"}

        selected = self._require_adapter(tool_input)
        if isinstance(selected, dict):
            return selected
        printer_id, adapter = selected

        if approved_input is None:
            return self._create_approval(
                request_context,
                "update_printer_config",
                {"section": section, "key": key, "value": value, "printer_id": printer_id},
                f"Modifying printer.cfg [{section}] {key}={value} — this changes printer configuration and requires explicit approval.",
            )

        read_fn = getattr(adapter, "get_config_file", None)
        write_fn = getattr(adapter, "write_config_file", None)
        if not read_fn or not write_fn:
            return {"error": "This printer adapter does not support remote config writes"}

        current = await read_fn(filename="printer.cfg", max_bytes=200000)
        if current.status != CommandResultStatus.SUCCESS:
            return {"error": f"Failed to read current printer.cfg: {current.message}"}

        content = current.data.get("content", "")
        import re
        pattern = re.compile(
            rf"^(\[{re.escape(section)}\][^\[]*?{re.escape(key)}:\s*)(.*)$",
            re.MULTILINE,
        )
        match = pattern.search(content)
        if not match:
            return {"error": f"Could not find [{section}] {key}: in printer.cfg"}

        old_value = match.group(2).strip()
        new_content = pattern.sub(rf"\g<1>{value}", content, count=1)
        write_result = await write_fn("printer.cfg", new_content)
        if write_result.status != CommandResultStatus.SUCCESS:
            return {"error": f"Failed to write printer.cfg: {write_result.message}"}

        return {
            "printer_id": printer_id,
            "section": section,
            "key": key,
            "old_value": old_value,
            "new_value": value,
            "status": "updated",
            "note": "Backup created automatically. Run FIRMWARE_RESTART to apply (not done automatically).",
        }
