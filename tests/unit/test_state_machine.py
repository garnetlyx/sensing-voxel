"""
Unit tests for AgentStateMachine
"""

import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime
from sensing_voxel.core.state_machine import AgentStateMachine
from sensing_voxel.core.models import AgentState, JobContext
from sensing_voxel.core.persistence import StatePersistence
from pathlib import Path
import tempfile


class TestAgentStateMachine:
    """Test AgentStateMachine transitions and persistence"""

    def test_initial_state(self):
        """Test that state machine starts in IDLE state"""
        sm = AgentStateMachine()
        assert sm.current_state == AgentState.IDLE
        assert sm.job_context is None

    def test_valid_transitions(self):
        """Test that valid transitions are accepted"""
        sm = AgentStateMachine()

        # IDLE -> RECEIVING
        assert sm.can_transition_to(AgentState.RECEIVING)

        # IDLE -> MONITORING (external print)
        assert sm.can_transition_to(AgentState.MONITORING)

        # IDLE -> PAUSED_PENDING_USER
        assert sm.can_transition_to(AgentState.PAUSED_PENDING_USER)

    def test_invalid_transitions(self):
        """Test that invalid transitions are rejected"""
        sm = AgentStateMachine()

        # IDLE -> SLICING is invalid (must go through RECEIVING)
        assert not sm.can_transition_to(AgentState.SLICING)

        # IDLE -> PRINTING is invalid
        assert not sm.can_transition_to(AgentState.PRINTING)

    @pytest.mark.asyncio
    async def test_transition_success(self):
        """Test successful state transition"""
        sm = AgentStateMachine()

        success = await sm.transition_to(AgentState.RECEIVING)
        assert success
        assert sm.current_state == AgentState.RECEIVING

    @pytest.mark.asyncio
    async def test_transition_failure(self):
        """Test that invalid transition fails"""
        sm = AgentStateMachine()

        success = await sm.transition_to(AgentState.SLICING)
        assert not success
        assert sm.current_state == AgentState.IDLE

    @pytest.mark.asyncio
    async def test_transition_hooks(self):
        """Test that transition hooks are called"""
        sm = AgentStateMachine()

        enter_mock = AsyncMock()
        exit_mock = AsyncMock()

        await sm.on_enter(AgentState.RECEIVING, enter_mock)
        await sm.on_exit(AgentState.IDLE, exit_mock)

        success = await sm.transition_to(AgentState.RECEIVING)

        assert success
        exit_mock.assert_called_once()
        enter_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_state_entry(self):
        """Test entering ERROR state"""
        sm = AgentStateMachine()

        await sm.enter_error_state("Test error", None)

        assert sm.current_state == AgentState.ERROR
        assert sm.last_error_summary == "Test error"

    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test recovering from ERROR state"""
        sm = AgentStateMachine()

        await sm.enter_error_state("Test error", None)
        assert sm.current_state == AgentState.ERROR

        success = await sm.recover_from_error(None)

        assert success
        assert sm.current_state == AgentState.IDLE
        assert sm.last_error_summary is None

    @pytest.mark.asyncio
    async def test_persistence_integration(self):
        """Test state persistence integration"""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "test_state.json"
            persistence = StatePersistence(state_file)

            sm = AgentStateMachine()
            sm.job_context = JobContext(
                job_id="test-123",
                job_file="test.stl",
                uploaded_at=datetime.now(),
            )

            # Persist
            await sm.persist(persistence)

            # Load
            loaded = persistence.load()

            assert loaded is not None
            assert loaded.current_state == AgentState.IDLE
            assert loaded.job_context.job_id == "test-123"
