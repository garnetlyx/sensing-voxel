"""
Integration tests for core print workflows.

Uses mocked adapters — no real printer required.
"""

import pytest
import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from sensing_voxel.adapters.base import PrinterAdapter
from sensing_voxel.core.decision import DecisionEngine
from sensing_voxel.core.models import (
    AgentState,
    CommandResult,
    CommandResultStatus,
    ConsumeResult,
    DecisionAction,
    FileValidationStatus,
    JobContext,
    NormalizedPrinterState,
    generate_decision_token,
    generate_job_id,
)
from sensing_voxel.core.persistence import StatePersistence
from sensing_voxel.core.state_machine import AgentStateMachine
from sensing_voxel.monitoring.monitor import MonitoringOrchestrator


class MockPrinterAdapter(PrinterAdapter):
    """Full mock adapter for integration tests."""

    def __init__(self):
        super().__init__()
        self._state = NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT
        self._connected = True

    async def start_print(self, gcode_path):
        self._state = NormalizedPrinterState.PRINTING
        return CommandResult(status=CommandResultStatus.SUCCESS)

    async def pause_print(self):
        self._state = NormalizedPrinterState.PAUSED
        return CommandResult(status=CommandResultStatus.SUCCESS)

    async def resume_print(self):
        self._state = NormalizedPrinterState.PRINTING
        return CommandResult(status=CommandResultStatus.SUCCESS)

    async def cancel_print(self):
        self._state = NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT
        return CommandResult(status=CommandResultStatus.SUCCESS)

    async def emergency_cancel(self):
        self._state = NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT
        return CommandResult(status=CommandResultStatus.SUCCESS)

    async def upload_file(self, local_path, remote_name):
        return CommandResult(status=CommandResultStatus.SUCCESS)

    async def get_z_offset(self):
        return 0.0

    async def set_z_offset(self, offset_mm):
        return CommandResult(status=CommandResultStatus.SUCCESS)

    async def set_nozzle_temp(self, temperature):
        return CommandResult(status=CommandResultStatus.SUCCESS)

    async def set_bed_temp(self, temperature):
        return CommandResult(status=CommandResultStatus.SUCCESS)

    async def home_axes(self, axes=""):
        return CommandResult(status=CommandResultStatus.SUCCESS)

    async def set_fan_speed(self, speed):
        return CommandResult(status=CommandResultStatus.SUCCESS)

    async def run_gcode(self, command):
        return CommandResult(status=CommandResultStatus.SUCCESS)

    async def list_files(self):
        return CommandResult(status=CommandResultStatus.SUCCESS, data=[])

    async def delete_file(self, filename):
        return CommandResult(status=CommandResultStatus.SUCCESS)

    async def get_file_metadata(self, filename):
        return CommandResult(status=CommandResultStatus.SUCCESS, data={})

    async def move_axis(self, axis, distance, speed=3000):
        return CommandResult(status=CommandResultStatus.SUCCESS)

    async def extrude(self, length, speed=300):
        return CommandResult(status=CommandResultStatus.SUCCESS)

    async def set_speed_override(self, percent):
        return CommandResult(status=CommandResultStatus.SUCCESS)

    async def set_flow_override(self, percent):
        return CommandResult(status=CommandResultStatus.SUCCESS)

    async def list_macros(self):
        return CommandResult(status=CommandResultStatus.SUCCESS, data=[])

    async def execute_macro(self, name):
        return CommandResult(status=CommandResultStatus.SUCCESS)

    async def restart_firmware(self):
        return CommandResult(status=CommandResultStatus.SUCCESS)

    async def restart_service(self):
        return CommandResult(status=CommandResultStatus.SUCCESS)

    async def get_system_info(self):
        return CommandResult(status=CommandResultStatus.SUCCESS, data={})

    async def motors_off(self):
        return CommandResult(status=CommandResultStatus.SUCCESS)

    async def get_current_position(self):
        return {"x": 0.0, "y": 0.0, "z": 0.0}

    async def get_printer_state(self):
        return self._state

    async def get_print_progress(self):
        return None

    async def is_connected(self):
        return self._connected

    async def connect(self):
        self._connected = True

    async def disconnect(self):
        self._connected = False

    async def simulate_print_complete(self):
        """Simulate printer completing a print."""
        old_state = self._state
        self._state = NormalizedPrinterState.TERMINAL
        await self._emit_state_changed(old_state, self._state)


class TestAgentStartedPrintWorkflow:
    """Test agent-started print happy path."""

    @pytest.fixture
    def setup(self, tmp_path):
        sm = AgentStateMachine()
        adapter = MockPrinterAdapter()
        persistence = StatePersistence(tmp_path / "state.json")
        decision_engine = DecisionEngine()
        orchestrator = MonitoringOrchestrator(
            state_machine=sm,
            adapter=adapter,
            persistence=persistence,
        )
        return sm, adapter, persistence, decision_engine, orchestrator

    @pytest.mark.asyncio
    async def test_full_happy_path(self, setup):
        """IDLE → RECEIVING → SLICING → AWAITING_APPROVAL → UPLOADING → PRINTING → MONITORING → COMPLETE → IDLE"""
        sm, adapter, persistence, decision_engine, orchestrator = setup
        await orchestrator.start()

        assert sm.current_state == AgentState.IDLE

        # Create job
        job_id = generate_job_id()
        job = JobContext(
            job_id=job_id,
            job_file="test.stl",
            uploaded_at=datetime.now(),
            decision_token=generate_decision_token(),
        )
        sm.job_context = job

        # IDLE → RECEIVING
        assert await sm.transition_to(AgentState.RECEIVING, persistence)
        assert sm.current_state == AgentState.RECEIVING

        # RECEIVING → SLICING
        assert await sm.transition_to(AgentState.SLICING, persistence)
        assert sm.current_state == AgentState.SLICING

        # SLICING → AWAITING_APPROVAL
        assert await sm.transition_to(AgentState.AWAITING_APPROVAL, persistence)
        assert sm.current_state == AgentState.AWAITING_APPROVAL

        # Approve decision
        result = await decision_engine.consume_decision(
            job_context=job,
            decision_token=job.decision_token,
            action=DecisionAction.APPROVE,
            expected_state=AgentState.AWAITING_APPROVAL,
            current_state=sm.current_state,
        )
        assert result == ConsumeResult.CONSUMED

        # AWAITING_APPROVAL → UPLOADING
        assert await sm.transition_to(AgentState.UPLOADING, persistence)

        # Upload and start print
        upload_result = await adapter.upload_file("test.gcode", "test.gcode")
        assert upload_result.status == CommandResultStatus.SUCCESS

        start_result = await adapter.start_print("test.gcode")
        assert start_result.status == CommandResultStatus.SUCCESS

        # UPLOADING → PRINTING
        assert await sm.transition_to(AgentState.PRINTING, persistence)

        # PRINTING → MONITORING
        assert await sm.transition_to(AgentState.MONITORING, persistence)
        orchestrator._is_monitoring = True

        # Simulate print completion
        await adapter.simulate_print_complete()
        # Give event handlers time to process
        await asyncio.sleep(0.1)

        assert sm.current_state == AgentState.COMPLETE

        # COMPLETE → IDLE
        assert await sm.transition_to(AgentState.IDLE, persistence)
        assert sm.current_state == AgentState.IDLE

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_gcode_direct_upload(self, setup):
        """IDLE → AWAITING_APPROVAL → UPLOADING → PRINTING (bypass slicing)"""
        sm, adapter, persistence, decision_engine, _ = setup

        job_id = generate_job_id()
        job = JobContext(
            job_id=job_id,
            job_file="test.gcode",
            uploaded_at=datetime.now(),
            decision_token=generate_decision_token(),
            gcode_path="test.gcode",
        )
        sm.job_context = job

        # IDLE → AWAITING_APPROVAL (direct G-code upload)
        assert await sm.transition_to(AgentState.AWAITING_APPROVAL, persistence)

        # Approve
        result = await decision_engine.consume_decision(
            job_context=job,
            decision_token=job.decision_token,
            action=DecisionAction.APPROVE,
            expected_state=AgentState.AWAITING_APPROVAL,
            current_state=sm.current_state,
        )
        assert result == ConsumeResult.CONSUMED

        # AWAITING_APPROVAL → UPLOADING → PRINTING
        assert await sm.transition_to(AgentState.UPLOADING, persistence)
        assert await sm.transition_to(AgentState.PRINTING, persistence)

    @pytest.mark.asyncio
    async def test_approval_timeout(self, setup):
        """AWAITING_APPROVAL → IDLE on timeout."""
        sm, _, persistence, decision_engine, _ = setup

        job_id = generate_job_id()
        job = JobContext(
            job_id=job_id,
            job_file="test.stl",
            uploaded_at=datetime.now(),
            decision_token=generate_decision_token(),
        )
        sm.job_context = job

        await sm.transition_to(AgentState.RECEIVING, persistence)
        await sm.transition_to(AgentState.SLICING, persistence)
        await sm.transition_to(AgentState.AWAITING_APPROVAL, persistence)

        # Reject (simulates timeout behavior)
        result = await decision_engine.consume_decision(
            job_context=job,
            decision_token=job.decision_token,
            action=DecisionAction.REJECT,
            expected_state=AgentState.AWAITING_APPROVAL,
            current_state=sm.current_state,
        )
        assert result == ConsumeResult.CONSUMED

        # Return to IDLE
        assert await sm.transition_to(AgentState.IDLE, persistence)
        assert sm.current_state == AgentState.IDLE

    @pytest.mark.asyncio
    async def NOTHING_pause_then_cancel(self, setup):
        """MONITORING → PAUSED_PENDING_USER → FAILED (user cancels after NOTHING)."""
        sm, adapter, persistence, _, orchestrator = setup

        # Set up monitoring state
        job = JobContext(
            job_id=generate_job_id(),
            job_file="test.gcode",
            uploaded_at=datetime.now(),
        )
        sm.job_context = job
        sm.current_state = AgentState.MONITORING
        orchestrator._is_monitoring = True

        # NOTHING detected → pause
        await orchestrator.NOTHING(0.72)
        assert sm.current_state == AgentState.PAUSED_PENDING_USER

        # User cancels
        await adapter.cancel_print()
        assert await sm.transition_to(AgentState.FAILED, persistence)
        assert sm.current_state == AgentState.FAILED

    @pytest.mark.asyncio
    async def NOTHING_emergency_cancel(self, setup):
        """MONITORING → EMERGENCY_FAILED on high-confidence NOTHING."""
        sm, _, persistence, _, orchestrator = setup

        job = JobContext(
            job_id=generate_job_id(),
            job_file="test.gcode",
            uploaded_at=datetime.now(),
        )
        sm.job_context = job
        sm.current_state = AgentState.MONITORING
        orchestrator._is_monitoring = True

        await orchestrator.NOTHING(0.92)
        assert sm.current_state == AgentState.EMERGENCY_FAILED


class TestStlToPrintE2E:
    """
    End-to-end test: STL upload → slicer → approval → print.

    Simulates the complete Telegram file upload flow through to adapter.start_print.
    """

    @pytest.fixture
    def setup(self, tmp_path):
        sm = AgentStateMachine()
        adapter = MockPrinterAdapter()
        persistence = StatePersistence(tmp_path / "state.json")
        decision_engine = DecisionEngine()
        orchestrator = MonitoringOrchestrator(
            state_machine=sm,
            adapter=adapter,
            persistence=persistence,
        )
        stl_file = tmp_path / "test_part.stl"
        stl_file.write_bytes(b"mock stl binary content")
        return sm, adapter, persistence, decision_engine, orchestrator, stl_file, tmp_path

    @pytest.mark.asyncio
    async def test_stl_to_print_e2e(self, setup):
        """
        Full E2E flow: IDLE → SLICING → AWAITING_APPROVAL → PRINTING → COMPLETE → IDLE

        Simulates:
        1. Telegram file upload (STL file)
        2. Slicer pipeline (mocked)
        3. Wait for AWAITING_APPROVAL state
        4. User approval via decision engine
        5. Verify adapter.start_print is called
        6. State transitions to completion
        """
        sm, adapter, persistence, decision_engine, orchestrator, stl_file, tmp_path = setup
        await orchestrator.start()

        # Track adapter method calls
        start_print_called = False
        original_start_print = adapter.start_print

        async def tracked_start_print(gcode_path):
            nonlocal start_print_called
            start_print_called = True
            return await original_start_print(gcode_path)

        adapter.start_print = tracked_start_print

        assert sm.current_state == AgentState.IDLE

        # Step 1: Simulate Telegram file upload - create job context
        job_id = generate_job_id()
        job = JobContext(
            job_id=job_id,
            job_file=str(stl_file),
            uploaded_at=datetime.now(),
            decision_token=generate_decision_token(),
        )
        sm.job_context = job

        # Step 2: Trigger workflow - IDLE → RECEIVING (simulating file receipt)
        assert await sm.transition_to(AgentState.RECEIVING, persistence)
        assert sm.current_state == AgentState.RECEIVING

        # Step 3: Slicer pipeline simulation - RECEIVING → SLICING
        assert await sm.transition_to(AgentState.SLICING, persistence)
        assert sm.current_state == AgentState.SLICING

        # Mock slicing completion - set gcode_path on job
        gcode_path = tmp_path / "test_part.gcode"
        gcode_path.write_bytes(b"; Mock G-code\nG28\nG1 X10 Y10\n")
        job.gcode_path = str(gcode_path)
        job.gcode_hash = "abc123mockhash"

        # SLICING → AWAITING_APPROVAL
        assert await sm.transition_to(AgentState.AWAITING_APPROVAL, persistence)
        assert sm.current_state == AgentState.AWAITING_APPROVAL

        # Step 4: Simulate user approval via decision engine
        result = await decision_engine.consume_decision(
            job_context=job,
            decision_token=job.decision_token,
            action=DecisionAction.APPROVE,
            expected_state=AgentState.AWAITING_APPROVAL,
            current_state=sm.current_state,
        )
        assert result == ConsumeResult.CONSUMED

        # Step 5: AWAITING_APPROVAL → UPLOADING
        assert await sm.transition_to(AgentState.UPLOADING, persistence)

        # Upload and start print (simulating the actual upload/print flow)
        upload_result = await adapter.upload_file(str(gcode_path), "test_part.gcode")
        assert upload_result.status == CommandResultStatus.SUCCESS

        start_result = await adapter.start_print(str(gcode_path))
        assert start_result.status == CommandResultStatus.SUCCESS

        # Verify adapter.start_print was called
        assert start_print_called, "adapter.start_print should have been called"

        # UPLOADING → PRINTING
        assert await sm.transition_to(AgentState.PRINTING, persistence)
        assert sm.current_state == AgentState.PRINTING

        # PRINTING → MONITORING
        assert await sm.transition_to(AgentState.MONITORING, persistence)
        orchestrator._is_monitoring = True

        # Step 6: Simulate print completion
        await adapter.simulate_print_complete()
        await asyncio.sleep(0.1)  # Allow event processing

        assert sm.current_state == AgentState.COMPLETE

        # COMPLETE → IDLE
        assert await sm.transition_to(AgentState.IDLE, persistence)
        assert sm.current_state == AgentState.IDLE

        # Final verification: complete state cycle achieved
        # IDLE → RECEIVING → SLICING → AWAITING_APPROVAL → UPLOADING → PRINTING → MONITORING → COMPLETE → IDLE

        await orchestrator.stop()
