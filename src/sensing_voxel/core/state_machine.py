"""
Agent state machine with transition table enforcement
"""

from typing import Dict, Set, Optional, Callable, Awaitable
from datetime import datetime
import asyncio
import logging

from sensing_voxel.core.models import (
    AgentState,
    NormalizedPrinterState,
    JobContext,
    DecisionAction,
    ConsumeResult,
)

logger = logging.getLogger(__name__)


class AgentStateMachine:
    """Agent state machine with enforced transitions"""

    # Transition table: valid transitions from each state
    TRANSITIONS: Dict[AgentState, Set[AgentState]] = {
        AgentState.IDLE: {
            AgentState.RECEIVING,
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
        self.job_context: Optional[JobContext] = None
        self.last_error_summary: Optional[str] = None
        self.z_offset_dirty: bool = False
        self.last_known_z_offset_baseline: Optional[float] = None
        self.moonraker_compat_mode: bool = False
        self.external_print: bool = False  # True if print detected externally

        # Transition hooks
        self._on_enter_hooks: Dict[AgentState, Optional[Callable[[], Awaitable[None]]]] = {}
        self._on_exit_hooks: Dict[AgentState, Optional[Callable[[], Awaitable[None]]]] = {}

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
        if not self.can_transition_to(new_state):
            logger.warning(
                f"Invalid transition: {self.current_state.name} -> {new_state.name}"
            )
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
            # Change state first
            self.current_state = new_state
            logger.info(f"State transition: {old_state.name} -> {new_state.name}")

            # Try to execute enter hook
            enter_hook = self._on_enter_hooks.get(new_state)
            if enter_hook:
                await enter_hook()

            # Persist state after transition
            if persistence:
                try:
                    persistence.save(self)
                except Exception as e:
                    logger.error(f"Failed to persist state after transition: {e}")
                    # Persistence failure is logged but doesn't rollback state
                    # This is a known trade-off: in-memory state may differ from persisted state
                    # until next successful persist

        except Exception as e:
            # If anything fails after state change, log error but don't revert
            # Reverting would create more inconsistency
            logger.error(f"Error during transition to {new_state.name}: {e}")
            raise

        return True

    async def on_enter(self, state: AgentState, hook: Callable[[], Awaitable[None]]):
        """Register hook for entering state"""
        if not isinstance(state, AgentState):
            raise TypeError(f"state must be AgentState enum, got {type(state)}")
        if not callable(hook):
            raise TypeError("hook must be callable")
        self._on_enter_hooks[state] = hook
        logger.debug(f"Registered on_enter hook for {state.name}")

    async def on_exit(self, state: AgentState, hook: Callable[[], Awaitable[None]]):
        """Register hook for exiting state"""
        if not isinstance(state, AgentState):
            raise TypeError(f"state must be AgentState enum, got {type(state)}")
        if not callable(hook):
            raise TypeError("hook must be callable")
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
