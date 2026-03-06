"""
Integration tests for external print detection and monitoring.
"""

import pytest
import asyncio
from datetime import datetime

from sensing_voxel.core.models import (
    AgentState,
    NormalizedPrinterState,
    JobContext,
    generate_job_id,
)
from sensing_voxel.core.persistence import StatePersistence
from sensing_voxel.core.state_machine import AgentStateMachine
from sensing_voxel.core.models import CommandResult, CommandResultStatus
from sensing_voxel.monitoring.monitor import MonitoringOrchestrator
from sensing_voxel.adapters.base import PrinterAdapter


class MockPrinterAdapter(PrinterAdapter):
    """Mock adapter for integration tests."""

    def __init__(self):
        super().__init__()
        self._state = NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT

    async def start_print(self, gcode_path): pass
    async def pause_print(self): pass
    async def resume_print(self): pass
    async def cancel_print(self): pass
    async def emergency_cancel(self): pass
    async def upload_file(self, local_path, remote_name): pass
    async def get_z_offset(self): return 0.0
    async def set_z_offset(self, offset_mm): pass
    async def set_nozzle_temp(self, temperature): pass
    async def set_bed_temp(self, temperature): pass
    async def home_axes(self, axes=""): pass
    async def set_fan_speed(self, speed): pass
    async def run_gcode(self, command): pass
    async def list_files(self): pass
    async def delete_file(self, filename): pass
    async def get_file_metadata(self, filename): pass
    async def move_axis(self, axis, distance, speed=3000): pass
    async def extrude(self, length, speed=300): pass
    async def set_speed_override(self, percent): pass
    async def set_flow_override(self, percent): pass
    async def list_macros(self): pass
    async def execute_macro(self, name): pass
    async def restart_firmware(self): pass
    async def restart_service(self): pass
    async def get_system_info(self): pass
    async def motors_off(self): pass
    async def get_printer_state(self): return self._state
    async def get_print_progress(self): pass
    async def is_connected(self): return True
    async def connect(self): pass
    async def disconnect(self): pass


class TestExternalPrintDetection:
    """Test external print detection and NOTHING-only monitoring."""

    @pytest.fixture
    def setup(self, tmp_path):
        sm = AgentStateMachine()
        adapter = MockPrinterAdapter()
        persistence = StatePersistence(tmp_path / "state.json")
        orchestrator = MonitoringOrchestrator(
            state_machine=sm,
            adapter=adapter,
            persistence=persistence,
            debounce_interval_seconds=1,
            debounce_required_updates=2,
        )
        return sm, adapter, persistence, orchestrator

    @pytest.mark.asyncio
    async def test_external_print_detection(self, setup):
        """Printer starts printing while agent is IDLE → external print detected."""
        sm, adapter, persistence, orchestrator = setup
        await orchestrator.start()

        assert sm.current_state == AgentState.IDLE

        # Simulate external print start (two state changes within debounce window)
        await orchestrator._handle_state_change(
            NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT,
            NormalizedPrinterState.PRINTING,
        )
        await orchestrator._handle_state_change(
            NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT,
            NormalizedPrinterState.PRINTING,
        )

        assert sm.current_state == AgentState.MONITORING
        assert sm.job_context is not None
        assert sm.job_context.is_external
        assert sm.job_context.external_monitoring

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_external_print_external_monitoring_policy(self, setup):
        """External prints use NOTHING-only monitoring (no first-layer correction)."""
        sm, adapter, persistence, orchestrator = setup
        await orchestrator.start()

        # Trigger external detection
        await orchestrator._handle_state_change(
            NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT,
            NormalizedPrinterState.PRINTING,
        )
        await orchestrator._handle_state_change(
            NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT,
            NormalizedPrinterState.PRINTING,
        )

        job = sm.job_context
        assert job is not None
        assert job.external_monitoring is True
        assert job.is_external is True

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_external_print_completion(self, setup):
        """External print completes → MONITORING → COMPLETE."""
        sm, adapter, persistence, orchestrator = setup
        await orchestrator.start()

        # Trigger external detection
        await orchestrator._handle_state_change(
            NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT,
            NormalizedPrinterState.PRINTING,
        )
        await orchestrator._handle_state_change(
            NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT,
            NormalizedPrinterState.PRINTING,
        )

        assert sm.current_state == AgentState.MONITORING

        # Simulate print completion
        await orchestrator._handle_state_change(
            NormalizedPrinterState.PRINTING,
            NormalizedPrinterState.TERMINAL,
        )

        assert sm.current_state == AgentState.COMPLETE

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_no_detection_when_agent_busy(self, setup):
        """No external print detection when agent is not IDLE."""
        sm, adapter, persistence, orchestrator = setup
        await orchestrator.start()

        # Put agent in MONITORING state (already watching a print)
        sm.current_state = AgentState.MONITORING

        # Simulate printer state change — should be ignored
        await orchestrator._handle_state_change(
            NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT,
            NormalizedPrinterState.PRINTING,
        )
        await orchestrator._handle_state_change(
            NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT,
            NormalizedPrinterState.PRINTING,
        )

        # State should NOT have changed (already MONITORING)
        assert sm.current_state == AgentState.MONITORING

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_single_update_no_detection(self, setup):
        """Single state change below debounce threshold doesn't trigger detection."""
        sm, adapter, persistence, orchestrator = setup
        await orchestrator.start()

        await orchestrator._handle_state_change(
            NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT,
            NormalizedPrinterState.PRINTING,
        )

        # Should still be IDLE (debounce not met)
        assert sm.current_state == AgentState.IDLE

        await orchestrator.stop()
