"""
Unit tests for MonitoringOrchestrator
"""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from datetime import datetime, timedelta
import asyncio

from sensing_voxel.monitoring.monitor import MonitoringOrchestrator
from sensing_voxel.core.state_machine import AgentStateMachine
from sensing_voxel.core.models import (
    AgentState,
    NormalizedPrinterState,
    JobContext,
    generate_job_id,
)
from sensing_voxel.core.persistence import StatePersistence
from sensing_voxel.adapters.base import PrinterAdapter


class MockAdapter(PrinterAdapter):
    """Mock adapter for testing"""

    async def start_print(self, gcode_path: str):
        pass

    async def pause_print(self):
        pass

    async def resume_print(self):
        pass

    async def cancel_print(self):
        pass

    async def emergency_cancel(self):
        pass

    async def upload_file(self, local_path: str, remote_name: str):
        pass

    async def get_z_offset(self) -> float:
        return 0.0

    async def set_z_offset(self, offset_mm: float):
        pass

    async def set_nozzle_temp(self, temperature: float):
        pass

    async def set_bed_temp(self, temperature: float):
        pass

    async def home_axes(self, axes: str = ""):
        pass

    async def set_fan_speed(self, speed: int):
        pass

    async def run_gcode(self, command: str):
        pass

    async def list_files(self):
        pass

    async def delete_file(self, filename: str):
        pass

    async def get_file_metadata(self, filename: str):
        pass

    async def move_axis(self, axis: str, distance: float, speed: int = 3000):
        pass

    async def extrude(self, length: float, speed: int = 300):
        pass

    async def set_speed_override(self, percent: int):
        pass

    async def set_flow_override(self, percent: int):
        pass

    async def list_macros(self):
        pass

    async def execute_macro(self, name: str):
        pass

    async def restart_firmware(self):
        pass

    async def restart_service(self):
        pass

    async def get_system_info(self):
        pass

    async def motors_off(self):
        pass

    async def get_current_position(self) -> dict[str, float] | None:
        return {"x": 0.0, "y": 0.0, "z": 0.0}

    async def get_printer_state(self) -> NormalizedPrinterState:
        return NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT

    async def get_print_progress(self):
        pass

    async def is_connected(self) -> bool:
        return True

    async def connect(self):
        pass

    async def disconnect(self):
        pass


class TestMonitoringOrchestrator:
    """Test MonitoringOrchestrator"""

    @pytest.fixture
    def mock_persistence(self, tmp_path):
        """Create mock persistence"""
        persistence_path = tmp_path / "test_state.json"
        return StatePersistence(persistence_path)

    @pytest.fixture
    def orchestrator(self, mock_persistence):
        """Create orchestrator instance"""
        state_machine = AgentStateMachine()
        adapter = MockAdapter()
        orchestrator = MonitoringOrchestrator(
            state_machine=state_machine,
            adapter=adapter,
            persistence=mock_persistence,
            debounce_interval_seconds=3,
            debounce_required_updates=2,
        )
        return orchestrator

    @pytest.mark.asyncio
    async def test_initialization(self, orchestrator):
        """Test orchestrator initialization"""
        assert orchestrator.debounce_interval == 3
        assert orchestrator.debounce_required_updates == 2
        assert orchestrator._debounce_counter == 0
        assert orchestrator._debounce_window_start is None
        assert orchestrator._is_monitoring is False

    @pytest.mark.asyncio
    async def test_start_registers_handler(self, orchestrator):
        """Test that start registers state change handler"""
        await orchestrator.start()

        assert orchestrator._state_change_handler is not None
        assert len(orchestrator.adapter._state_changed_callbacks) > 0

    @pytest.mark.asyncio
    async def test_stop_removes_handler(self, orchestrator):
        """Test that stop removes state change handler"""
        await orchestrator.start()
        handler_count = len(orchestrator.adapter._state_changed_callbacks)

        await orchestrator.stop()

        assert len(orchestrator.adapter._state_changed_callbacks) < handler_count

    @pytest.mark.asyncio
    async def test_ignores_non_printing_transitions(self, orchestrator):
        """Test that non-PRINTING transitions are ignored"""
        await orchestrator.start()

        # Simulate state change to PAUSED
        await orchestrator._handle_state_change(
            NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT,
            NormalizedPrinterState.PAUSED,
        )

        # Should not trigger debounce
        assert orchestrator._debounce_counter == 0

    @pytest.mark.asyncio
    async def test_ignores_printing_when_not_idle(self, orchestrator):
        """Test that PRINTING transitions are ignored when agent is not IDLE"""
        await orchestrator.start()

        # Set agent to RECEIVING state
        await orchestrator.state_machine.transition_to(AgentState.RECEIVING)

        # Simulate state change to PRINTING
        await orchestrator._handle_state_change(
            NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT,
            NormalizedPrinterState.PRINTING,
        )

        # Should not trigger debounce
        assert orchestrator._debounce_counter == 0

    @pytest.mark.asyncio
    async def test_debounce_single_update_not_triggered(self, orchestrator):
        """Test that single update doesn't trigger detection"""
        await orchestrator.start()

        # First update
        await orchestrator._handle_state_change(
            NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT,
            NormalizedPrinterState.PRINTING,
        )

        assert orchestrator._debounce_counter == 1
        assert orchestrator.state_machine.current_state == AgentState.IDLE

    @pytest.mark.asyncio
    async def test_debounce_threshold_reached_triggers_detection(
        self, orchestrator, mock_persistence
    ):
        """Test that reaching threshold triggers external print detection"""
        await orchestrator.start()

        # First update
        await orchestrator._handle_state_change(
            NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT,
            NormalizedPrinterState.PRINTING,
        )
        assert orchestrator._debounce_counter == 1

        # Second update within window
        await orchestrator._handle_state_change(
            NormalizedPrinterState.PRINTING,
            NormalizedPrinterState.PRINTING,
        )

        # Counter should have been incremented to 2 before reset
        # State should transition to MONITORING
        assert orchestrator.state_machine.current_state == AgentState.MONITORING
        assert orchestrator.state_machine.job_context is not None
        assert orchestrator._is_monitoring is True
        assert orchestrator.state_machine.job_context.is_external is True
        assert orchestrator.state_machine.job_context.external_monitoring is True

    @pytest.mark.asyncio
    async def test_debounce_window_resets_after_timeout(self, orchestrator):
        """Test that debounce window resets after timeout"""
        await orchestrator.start()

        # First update
        await orchestrator._handle_state_change(
            NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT,
            NormalizedPrinterState.PRINTING,
        )
        assert orchestrator._debounce_counter == 1

        # Wait for window to expire
        await asyncio.sleep(3.1)  # Slightly more than debounce interval

        # Check that state was reset
        assert orchestrator._debounce_counter == 0
        assert orchestrator._debounce_window_start is None

    @pytest.mark.asyncio
    async def test_external_print_job_context(self, orchestrator, mock_persistence):
        """Test that external print job context has correct fields"""
        await orchestrator.start()

        # Trigger detection
        await orchestrator._handle_state_change(
            NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT,
            NormalizedPrinterState.PRINTING,
        )
        await orchestrator._handle_state_change(
            NormalizedPrinterState.PRINTING,
            NormalizedPrinterState.PRINTING,
        )

        job_context = orchestrator.state_machine.job_context
        assert job_context is not None
        assert job_context.is_external is True
        assert job_context.external_monitoring is True
        assert job_context.job_file == "external_print"
        assert job_context.decision_consumed is False

    @pytest.mark.asyncio
    async def test_agent_print_monitoring(self, orchestrator, mock_persistence):
        """Test starting monitoring for agent-started print"""
        await orchestrator.start()

        # Set state to PRINTING through valid path: IDLE -> RECEIVING -> SLICING -> AWAITING_APPROVAL -> UPLOADING -> PRINTING
        await orchestrator.state_machine.transition_to(AgentState.RECEIVING)
        await orchestrator.state_machine.transition_to(AgentState.SLICING)
        await orchestrator.state_machine.transition_to(AgentState.AWAITING_APPROVAL)
        await orchestrator.state_machine.transition_to(AgentState.UPLOADING)
        await orchestrator.state_machine.transition_to(AgentState.PRINTING)

        # Create job context
        job_context = JobContext(
            job_id=generate_job_id(),
            job_file="test.stl",
            uploaded_at=datetime.now(),
            is_external=False,
            external_monitoring=False,
        )

        # Start monitoring
        await orchestrator.start_monitoring_for_agent_print(job_context)

        assert orchestrator.state_machine.current_state == AgentState.MONITORING
        assert orchestrator.state_machine.job_context.job_id == job_context.job_id
        assert orchestrator._is_monitoring is True

    @pytest.mark.asyncio
    async def test_stop_monitoring(self, orchestrator, mock_persistence):
        """Test stopping monitoring"""
        await orchestrator.start()

        # Trigger external print detection
        await orchestrator._handle_state_change(
            NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT,
            NormalizedPrinterState.PRINTING,
        )
        await orchestrator._handle_state_change(
            NormalizedPrinterState.PRINTING,
            NormalizedPrinterState.PRINTING,
        )

        assert orchestrator._is_monitoring is True

        # Stop monitoring
        await orchestrator.stop_monitoring()

        assert orchestrator._is_monitoring is False
        assert orchestrator._monitoring_job_id is None

    @pytest.mark.asyncio
    async def test_handle_print_pause(self, orchestrator, mock_persistence):
        """Test handling print pause"""
        await orchestrator.start()

        # Trigger external print detection
        await orchestrator._handle_state_change(
            NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT,
            NormalizedPrinterState.PRINTING,
        )
        await orchestrator._handle_state_change(
            NormalizedPrinterState.PRINTING,
            NormalizedPrinterState.PRINTING,
        )

        assert orchestrator.state_machine.current_state == AgentState.MONITORING

        # Handle pause
        await orchestrator.handle_print_pause()

        assert orchestrator.state_machine.current_state == AgentState.PAUSED_PENDING_USER

    @pytest.mark.asyncio
    async def test_handle_NOTHING_pause(self, orchestrator, mock_persistence):
        """Test handling NOTHING detection pause"""
        await orchestrator.start()

        # Trigger external print detection
        await orchestrator._handle_state_change(
            NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT,
            NormalizedPrinterState.PRINTING,
        )
        await orchestrator._handle_state_change(
            NormalizedPrinterState.PRINTING,
            NormalizedPrinterState.PRINTING,
        )

        # Handle NOTHING pause
        await orchestrator.NOTHING(confidence=0.7)

        assert orchestrator.state_machine.current_state == AgentState.PAUSED_PENDING_USER

    @pytest.mark.asyncio
    async def test_handle_NOTHING_emergency_cancel(self, orchestrator, mock_persistence):
        """Test handling NOTHING detection emergency cancel"""
        await orchestrator.start()

        # Trigger external print detection
        await orchestrator._handle_state_change(
            NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT,
            NormalizedPrinterState.PRINTING,
        )
        await orchestrator._handle_state_change(
            NormalizedPrinterState.PRINTING,
            NormalizedPrinterState.PRINTING,
        )

        # Handle emergency cancel
        await orchestrator.NOTHING(confidence=0.9)

        assert orchestrator.state_machine.current_state == AgentState.EMERGENCY_FAILED

    @pytest.mark.asyncio
    async def test_debounce_counter_reset_after_detection(
        self, orchestrator, mock_persistence
    ):
        """Test that debounce state is reset after detection"""
        await orchestrator.start()

        # Trigger detection
        await orchestrator._handle_state_change(
            NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT,
            NormalizedPrinterState.PRINTING,
        )
        await orchestrator._handle_state_change(
            NormalizedPrinterState.PRINTING,
            NormalizedPrinterState.PRINTING,
        )

        # State should be reset
        assert orchestrator._debounce_counter == 0
        assert orchestrator._debounce_window_start is None
        assert orchestrator._debounce_timer is None
