"""
Printer context collector for LLM system prompt injection
"""

from __future__ import annotations

import logging
from datetime import datetime

from sensing_voxel.adapters.base import PrinterAdapter
from sensing_voxel.core.models import TelemetryData
from sensing_voxel.core.state_machine import AgentStateMachine
from sensing_voxel.llm.runtime import ChatRequestContext

logger = logging.getLogger(__name__)

BASE_SYSTEM_PROMPT = """\
You are Sensing Voxel AI, an expert 3D printing assistant integrated into \
a real-time printer monitoring system. You can directly query and control the \
printer through tool calls.

Your capabilities:
- Answer questions about FDM/FFF 3D printing
- Query real-time printer status, temperatures, and print progress via tools
- Control the printer: set temperatures, pause/resume/cancel prints, home axes, \
adjust fan speed, run G-code commands, and manage Z-offset
- Capture and analyze camera snapshots to visually inspect print quality and progress
- Control the monitoring camera: pan/tilt to change the view angle, use presets \
for saved positions (if camera is configured)
- Provide troubleshooting advice for print issues
- Help with print quality problems (stringing, warping, layer adhesion, etc.)
- Advise on filament choices and temperatures

IMPORTANT — Always use tools for printer status:
- When the user asks about print status, progress, or temperatures, ALWAYS call \
get_print_progress first. Do NOT just read the background context below — it may be stale.
- The "Monitoring System State" in the context below is the orchestration system's \
internal state (e.g. IDLE means no agent-managed job). It is NOT the printer state. \
The printer can be actively printing even when the monitoring system is IDLE \
(e.g. print started from the printer's touchscreen).
- To know the actual printer state, call get_printer_state or get_print_progress.

Klipper command reference (use via run_gcode or execute_macro):
- Bed mesh leveling: BED_MESH_CALIBRATE
- PID tune nozzle: PID_CALIBRATE HEATER=extruder TARGET=210
- PID tune bed: PID_CALIBRATE HEATER=heater_bed TARGET=60
- Input shaper: SHAPER_CALIBRATE
- Pressure advance test: SET_PRESSURE_ADVANCE ADVANCE=0.04
- Save config: SAVE_CONFIG
- Query endstops: QUERY_ENDSTOPS
- Probe accuracy: PROBE_ACCURACY
- Manual probe: MANUAL_PROBE
- Stepper buzz test: STEPPER_BUZZ STEPPER=stepper_x
- Dump TMC status: DUMP_TMC STEPPER=stepper_x
- Emergency stop: use emergency_cancel (NEVER M112 via run_gcode)
If the user asks for a Klipper command you don't recognize, use list_macros to check \
available macros first, then run_gcode to execute.

Tool use guidelines:
- For temperature changes, use set_nozzle_temp and set_bed_temp (not raw G-code).
- Always confirm destructive actions (cancel_print) with the user before executing.
- Never send M112 via run_gcode — use the emergency stop path instead.
- Validate temperature requests against material limits before setting.
- For camera control, use camera_move_step for quick adjustments (left/right/up/down) \
and camera_move_to for precise positioning.

General guidelines:
- Be concise. Telegram messages have a 4096 character limit.
- For safety-critical advice, always recommend caution.
- Respond in the same language the user writes in.\
"""


class PrinterContextCollector:
    """Collects printer state for LLM context injection."""

    def __init__(
        self,
        state_machine: AgentStateMachine,
        adapters: dict[str, PrinterAdapter],
        telemetries: dict[str, TelemetryData | None] | None = None,
        connections: dict[str, bool] | None = None,
        active_printer: str | None = None,
    ):
        self._state_machine = state_machine
        self._adapters = adapters
        self._telemetries = telemetries or {}
        self._connections = connections or {}
        self._active_printer = active_printer

    def update_telemetry(self, printer_id: str, data: TelemetryData | None) -> None:
        """Update cached telemetry for a printer."""
        self._telemetries[printer_id] = data

    def update_connection(self, printer_id: str, connected: bool) -> None:
        """Update connection status for a printer."""
        self._connections[printer_id] = connected

    def collect_context(self) -> str:
        """Collect current printer context as formatted text."""
        lines = []

        # Monitoring system orchestration state (NOT printer state)
        state = self._state_machine.current_state
        lines.append(f"Monitoring System State: {state.name} (this is NOT the printer state)")

        # Job context
        job = self._state_machine.job_context
        if job:
            lines.append(f"Active Job: {job.job_id}")
            lines.append(f"Job File: {job.job_file}")
            if job.decision_consumed:
                lines.append(f"Decision: {job.decision_action.name}")
            if job.is_external:
                lines.append("Job Type: External (NOTHING-only monitoring)")

        # Z-offset status
        if self._state_machine.z_offset_dirty:
            lines.append("Z-Offset: DIRTY (prints blocked until reset)")
        else:
            baseline = self._state_machine.last_known_z_offset_baseline
            if baseline is not None:
                lines.append(f"Z-Offset: CLEAN (baseline {baseline:.3f}mm)")

        # Printers
        lines.append("")
        lines.append("Printers:")
        for pid in self._adapters:
            connected = self._connections.get(pid, False)
            status = "connected" if connected else "disconnected"
            active = " [ACTIVE]" if pid == self._active_printer else ""
            lines.append(f"  {pid}: {status}{active}")

            # Telemetry
            telemetry = self._telemetries.get(pid)
            if telemetry and connected:
                temps = telemetry.temperatures
                if temps:
                    temp_parts = []
                    for key in sorted(temps.keys()):
                        val = temps[key]
                        if val is not None:
                            temp_parts.append(f"{key}={val:.1f}C")
                    if temp_parts:
                        lines.append(f"    Temps: {', '.join(temp_parts)}")

                if telemetry.metadata:
                    progress = telemetry.metadata.get("progress")
                    if progress is not None:
                        lines.append(f"    Progress: {progress:.1%}")
                    filename = telemetry.metadata.get("filename")
                    if filename:
                        lines.append(f"    File: {filename}")
                    print_state = telemetry.metadata.get("state")
                    if print_state:
                        lines.append(f"    Printer State: {print_state}")

        lines.append("")
        lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return "\n".join(lines)

    def format_system_prompt(
        self,
        request_context: ChatRequestContext | None = None,
        chat_summary: str | None = None,
    ) -> str:
        """Build complete system prompt with live printer context."""
        context = self.collect_context()
        sections = [
            BASE_SYSTEM_PROMPT,
            "\n\n---\nBackground Context (may be stale — use tools for authoritative data):\n",
            context,
            "\n---",
        ]
        if request_context:
            sections.extend(
                [
                    "\nRequest Context:\n",
                    f"- Channel: {request_context.channel}\n",
                    f"- Logical User: {request_context.logical_user_id}\n",
                    f"- Active Printer Preference: {request_context.active_printer or self._active_printer or 'none'}\n",
                    f"- Request ID: {request_context.request_id}\n",
                ]
            )
        if chat_summary:
            sections.extend(["\nCompacted Conversation Summary:\n", chat_summary, "\n"])
        return "".join(sections)
