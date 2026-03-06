"""
Agent state machine with transition table enforcement
"""

import asyncio
import copy
import logging
from collections.abc import Awaitable, Callable

from sensing_voxel.core.models import (
    AgentState,
    JobContext,
)

logger = logging.getLogger(__name__)


class AgentStateMachine:
    """Agent state machine with enforced transitions"""

    # Transition table: valid transitions from each state
    TRANSITIONS: dict[AgentState, set[AgentState]] = {
        AgentState.IDLE: {
            AgentState.RECEIVING,
            AgentState.AWAITING_APPROVAL,  # Direct G-code upload (bypass slicing)
            AgentState.MONITORING,
            AgentState.PAUSED_PENDING_USER,
            AgentState.ERROR,  # Allow direct entry for error recovery
        },
        AgentState.RECEIVING: {
            AgentState.SLICING,
            AgentState.ERROR,
        },
        AgentState.SLICING: {
            AgentState.AWAITING_APPROVAL,
            AgentState.ERROR,
        },
        AgentState.AWAITING_APPROVAL: {
            AgentState.UPLOADING,
            AgentState.IDLE,
        },
        AgentState.UPLOADING: {
            AgentState.PRINTING,
            AgentState.ERROR,
        },
        AgentState.PRINTING: {
            AgentState.MONITORING,
            AgentState.FAILED,
            AgentState.ERROR,
        },
        AgentState.MONITORING: {
            AgentState.PAUSED_PENDING_USER,
            AgentState.EMERGENCY_FAILED,
            AgentState.COMPLETE,
            AgentState.FAILED,
        },
        AgentState.PAUSED_PENDING_USER: {
            AgentState.MONITORING,
            AgentState.FAILED,
            AgentState.EMERGENCY_FAILED,
        },
        AgentState.COMPLETE: {
            AgentState.IDLE,
        },
        AgentState.FAILED: {
            AgentState.IDLE,
        },
        AgentState.EMERGENCY_FAILED: {
            AgentState.IDLE,
        },
        AgentState.ERROR: {
            AgentState.IDLE,
        },
    }

    def __init__(self):
        self.current_state: AgentState = AgentState.IDLE
        self.job_context: JobContext | None = None
        self.last_error_summary: str | None = None
        self.z_offset_dirty: bool = False
        self.last_known_z_offset_baseline: float | None = None
        self.moonraker_compat_mode: bool = False
        self.external_print: bool = False  # True if print detected externally

        # Transition hooks
        self._on_enter_hooks: dict[AgentState, Callable[[], Awaitable[None]] | None] = {}
        self._on_exit_hooks: dict[AgentState, Callable[[], Awaitable[None]] | None] = {}

        # Global lock to prevent concurrent state transitions (instance-level)
        self._global_lock = asyncio.Lock()

    @classmethod
    async def load_or_create(cls, persistence) -> "AgentStateMachine":
        """Load state from persistence or create new instance"""
        state_machine = persistence.load()
        if state_machine:
            logger.info(f"Loaded state from persistence: {state_machine.current_state.name}")
            return state_machine
        logger.info("No persisted state found, creating new state machine")
        return cls()

    def can_transition_to(self, new_state: AgentState) -> bool:
        """Check if transition to new_state is valid"""
        return new_state in self.TRANSITIONS.get(self.current_state, set())

    async def transition_to(self, new_state: AgentState, persistence=None) -> bool:
        """Transition to new state if valid"""

        # Acquire global lock to prevent concurrent transitions
        async with self._global_lock:
            if not self.can_transition_to(new_state):
                logger.warning(f"Invalid transition: {self.current_state.name} -> {new_state.name}")
                return False

            old_state = self.current_state

            # Try to execute exit hook
            exit_hook = self._on_exit_hooks.get(old_state)
            if exit_hook:
                try:
                    await exit_hook()
                except Exception as e:
                    logger.error(f"Exit hook failed for state {old_state.name}: {e}")
                    # Exit hook failure doesn't prevent transition
                    # but caller should be aware - this is a known behavior

            try:
                # Persist state BEFORE changing it (atomic commit pattern)
                # This ensures crash recovery will restore the new state
                # Use deepcopy to prevent shallow copy mutation issues
                if persistence:
                    # Create deep copy with the new state for persistence
                    state_to_persist = copy.deepcopy(self)
                    state_to_persist.current_state = new_state
                    persistence.save(state_to_persist)
                    logger.debug(f"State persisted: {new_state.name}")

                # Change state in memory BEFORE executing enter hook
                # This allows hook to assume it's in the new state
                old_state_for_rollback = self.current_state
                self.current_state = new_state
                logger.info(f"State transition: {old_state.name} -> {new_state.name}")

                # Try to execute enter hook
                enter_hook = self._on_enter_hooks.get(new_state)
                if enter_hook:
                    try:
                        await enter_hook()
                    except Exception as e:
                        # Enter hook failed - ROLLBACK state change
                        logger.error(f"Enter hook failed for state {new_state.name}: {e}")
                        self.current_state = old_state_for_rollback
                        logger.warning(
                            f"Rolled back state to {old_state_for_rollback.name} due to hook failure"
                        )
                        raise

                return True

            except Exception as e:
                # If persistence or anything else fails, state may have been changed
                logger.error(f"Error during transition to {new_state.name}: {e}")
                return False

    async def on_enter(self, state: AgentState, hook: Callable[[], Awaitable[None]]):
        """Register hook for entering state"""
        if not isinstance(state, AgentState):
            raise TypeError(f"state must be AgentState enum, got {type(state)}")
        if not callable(hook):
            raise TypeError("hook must be callable")
        # Validate hook is async
        import inspect

        if not inspect.iscoroutinefunction(hook):
            raise TypeError("hook must be an async function (async def)")
        self._on_enter_hooks[state] = hook
        logger.debug(f"Registered on_enter hook for {state.name}")

    async def on_exit(self, state: AgentState, hook: Callable[[], Awaitable[None]]):
        """Register hook for exiting state"""
        if not isinstance(state, AgentState):
            raise TypeError(f"state must be AgentState enum, got {type(state)}")
        if not callable(hook):
            raise TypeError("hook must be callable")
        # Validate hook is async
        import inspect

        if not inspect.iscoroutinefunction(hook):
            raise TypeError("hook must be an async function (async def)")
        self._on_exit_hooks[state] = hook
        logger.debug(f"Registered on_exit hook for {state.name}")

    async def persist(self, persistence):
        """Persist current state"""
        try:
            persistence.save(self)
            logger.debug(f"State persisted: {self.current_state.name}")
        except Exception as e:
            logger.error(f"Failed to persist state: {e}")
            raise

    async def enter_error_state(self, error_summary: str, persistence=None):
        """Enter ERROR state with error summary"""
        self.last_error_summary = error_summary
        await self.transition_to(AgentState.ERROR, persistence)
        logger.error(f"Entered ERROR state: {error_summary}")

    async def recover_from_error(self, persistence=None):
        """Recover from ERROR state to IDLE"""
        if self.current_state != AgentState.ERROR:
            logger.warning("Cannot recover from non-ERROR state")
            return False

        self.last_error_summary = None
        return await self.transition_to(AgentState.IDLE, persistence)
