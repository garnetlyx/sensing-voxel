"""
BUG TESTS: State Machine Issues

These tests document critical bugs in the state machine implementation.
Each test demonstrates a specific failure mode.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from sensing_voxel.core.state_machine import AgentStateMachine
from sensing_voxel.core.models import AgentState, JobContext
from datetime import datetime


class TestStateMachineBugs:
    """Tests exposing state machine bugs"""

    @pytest.mark.asyncio
    async def test_enter_hook_exception_leaves_state_changed(self):
        """
        BUG: If enter hook raises exception, state has already changed
        but hook failed. This leaves system in inconsistent state.

        Expected: State transition should be atomic (state + hooks succeed/fail together).
        Actual: State changes before hook runs, so hook failure = inconsistent state.
        """
        sm = AgentStateMachine()

        failing_hook = AsyncMock(side_effect=RuntimeError("Hook failed"))
        await sm.on_enter(AgentState.RECEIVING, failing_hook)

        # This should NOT change state if hook fails
        # But currently it WILL change state before calling hook
        with pytest.raises(RuntimeError):
            await sm.transition_to(AgentState.RECEIVING)

        # BUG: State changed even though hook failed!
        # This assertion documents the bug - it SHOULD be IDLE but is RECEIVING
        assert sm.current_state == AgentState.RECEIVING, \
            "BUG: State changed despite hook failure - inconsistent state!"

    @pytest.mark.asyncio
    async def test_exit_hook_exception_leaves_state_unchanged_but_hook_failed(self):
        """
        BUG: If exit hook raises exception, we never transition but
        the exit hook's side effects may have partially executed.

        This is less severe than enter hook failure but still inconsistent.
        """
        sm = AgentStateMachine()
        sm.current_state = AgentState.IDLE

        failing_hook = AsyncMock(side_effect=RuntimeError("Exit hook failed"))
        await sm.on_exit(AgentState.IDLE, failing_hook)

        # This should transition but exit hook fails
        with pytest.raises(RuntimeError):
            await sm.transition_to(AgentState.RECEIVING)

        # State should still be IDLE (transition didn't complete)
        assert sm.current_state == AgentState.IDLE

    @pytest.mark.asyncio
    async def test_persistence_failure_not_rolled_back(self):
        """
        BUG: If persistence.save() fails during transition, state has already
        changed in memory but wasn't persisted. On restart, system will be
        in different state than expected.

        Expected: Either persist before state change, or roll back on failure.
        Actual: State changes, persistence fails, no rollback.
        """
        sm = AgentStateMachine()

        failing_persistence = Mock()
        failing_persistence.save = Mock(side_effect=IOError("Disk full"))

        # Transition with failing persistence
        success = await sm.transition_to(AgentState.RECEIVING, failing_persistence)

        # BUG: transition_to returns True (transition succeeded)
        # but persistence failed! State is now different in memory vs on disk.
        assert success is True, "Transition reported success"
        assert sm.current_state == AgentState.RECEIVING, "State changed in memory"

        # The persistence failure was logged but not propagated
        # Next restart will load OLD state from disk (IDLE)
        # but actual state is RECEIVING - INCONSISTENCY!

    @pytest.mark.asyncio
    async def test_hook_registration_not_validated(self):
        """
        BUG: Hooks can be registered for invalid states without error.

        This allows typos or mistakes to go unnoticed until runtime.
        """
        sm = AgentStateMachine()

        hook = AsyncMock()

        # This should fail or warn - INVALID_STATE is not a real state
        # But currently it silently accepts any state value
        await sm.on_enter("INVALID_STATE", hook)  # type: ignore
        await sm.on_exit(123, hook)  # type: ignore

        # No error was raised - hooks registered for invalid keys
        # This will cause silent failures at runtime
        assert "INVALID_STATE" in sm._on_enter_hooks
        assert 123 in sm._on_exit_hooks

    @pytest.mark.asyncio
    async def test_missing_transition_in_error_recovery(self):
        """
        BUG: Error recovery only works from ERROR state.
        If somehow called from other states, it warns but doesn't prevent
        transition attempt (which will fail silently).

        The warning is logged but the method returns False - easy to miss.
        """
        sm = AgentStateMachine()
        sm.current_state = AgentState.PRINTING

        # Try to recover from non-error state
        result = await sm.recover_from_error(None)

        # Returns False but no exception - easy to ignore
        assert result is False
        assert sm.current_state == AgentState.PRINTING  # Still in PRINTING

        # If caller doesn't check return value, they think recovery happened!
