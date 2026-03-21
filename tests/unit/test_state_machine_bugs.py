from importlib import import_module
from unittest.mock import AsyncMock, Mock

pytest = import_module("pytest")
AgentStateMachine = import_module("sensing_voxel.core.state_machine").AgentStateMachine
AgentState = import_module("sensing_voxel.core.models").AgentState


class TestStateMachineBugs:
    @pytest.mark.asyncio
    async def test_enter_hook_exception_rolls_back_state(self):
        sm = AgentStateMachine()
        old_state = sm.current_state

        failing_hook = AsyncMock(side_effect=RuntimeError("Hook failed"))
        await sm.on_enter(AgentState.RECEIVING, failing_hook)

        success = await sm.transition_to(AgentState.RECEIVING)

        assert success is False
        assert sm.current_state == old_state

    @pytest.mark.asyncio
    async def test_exit_hook_exception_does_not_block_transition(self):
        sm = AgentStateMachine()
        sm.current_state = AgentState.IDLE

        failing_hook = AsyncMock(side_effect=RuntimeError("Exit hook failed"))
        await sm.on_exit(AgentState.IDLE, failing_hook)

        success = await sm.transition_to(AgentState.RECEIVING)

        assert success is True
        assert sm.current_state == AgentState.RECEIVING

    @pytest.mark.asyncio
    async def test_persistence_failure_returns_failure_without_state_change(self):
        sm = AgentStateMachine()
        old_state = sm.current_state

        failing_persistence = Mock()
        failing_persistence.save = Mock(side_effect=IOError("Disk full"))

        success = await sm.transition_to(AgentState.RECEIVING, failing_persistence)

        assert success is False
        assert sm.current_state == old_state

    @pytest.mark.asyncio
    async def test_hook_registration_validates_state_type(self):
        sm = AgentStateMachine()

        hook = AsyncMock()

        with pytest.raises(TypeError):
            await sm.on_enter("INVALID_STATE", hook)  # type: ignore[arg-type]

        with pytest.raises(TypeError):
            await sm.on_exit(123, hook)  # type: ignore[arg-type]

        assert not sm._on_enter_hooks
        assert not sm._on_exit_hooks

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
