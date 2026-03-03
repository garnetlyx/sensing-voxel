"""
Printer context collector for LLM system prompt injection
"""

from __future__ import annotations

import logging
from datetime import datetime

from sensing_voxel.adapters.base import PrinterAdapter
from sensing_voxel.core.models import TelemetryData
from sensing_voxel.core.state_machine import AgentStateMachine

logger = logging.getLogger(__name__)

BASE_SYSTEM_PROMPT = """\
You are Sensing Voxel AI, an expert 3D printing assistant integrated into \
a real-time printer monitoring system. You have live access to printer telemetry.

Your capabilities:
- Answer questions about FDM/FFF 3D printing
- Analyze printer status and telemetry data
- Provide troubleshooting advice for print issues
- Explain print settings and their effects
- Help with print quality problems (stringing, warping, layer adhesion, etc.)
- Advise on filament choices and temperatures

Guidelines:
- Be concise. Telegram messages have a 4096 character limit.
- When reporting temps or progress, reference the live data below.
- For safety-critical advice, always recommend caution.
- You can suggest printer actions but cannot execute them directly.
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

        # Agent state
        state = self._state_machine.current_state
        lines.append(f"Agent State: {state.name}")

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

    def format_system_prompt(self) -> str:
        """Build complete system prompt with live printer context."""
        context = self.collect_context()
        return f"{BASE_SYSTEM_PROMPT}\n\n---\nLive Printer Context:\n{context}\n---"
