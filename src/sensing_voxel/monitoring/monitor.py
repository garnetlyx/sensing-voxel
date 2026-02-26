"""
Monitoring orchestrator for print supervision

Coordinates external print detection and active print monitoring.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime

from sensing_voxel.adapters.base import PrinterAdapter
from sensing_voxel.core.models import (
    AgentState,
    JobContext,
    NormalizedPrinterState,
    generate_job_id,
)
from sensing_voxel.core.persistence import StatePersistence
from sensing_voxel.core.state_machine import AgentStateMachine

logger = logging.getLogger(__name__)


class MonitoringOrchestrator:
    """
    Orchestrates print monitoring lifecycle

    Handles:
    - External print detection (debounced)
    - Monitoring lifecycle for agent-started prints
    - State machine transitions based on printer events
    """

    def __init__(
        self,
        state_machine: AgentStateMachine,
        adapter: PrinterAdapter,
        persistence: StatePersistence,
        debounce_interval_seconds: int = 3,
        debounce_required_updates: int = 2,
    ):
        """
        Initialize monitoring orchestrator

        Args:
            state_machine: Agent state machine
            adapter: Printer adapter
            persistence: State persistence
            debounce_interval_seconds: Time window for debounce (default: 3s)
            debounce_required_updates: Number of updates required within window (default: 2)
        """
        self.state_machine = state_machine
        self.adapter = adapter
        self.persistence = persistence

        self.debounce_interval = debounce_interval_seconds
        self.debounce_required_updates = debounce_required_updates

        # Debounce state
        self._debounce_counter = 0
        self._debounce_window_start: datetime | None = None
        self._debounce_timer: asyncio.Task | None = None

        # Monitoring state
        self._is_monitoring = False
        self._monitoring_job_id: str | None = None

        # Event handlers
        self._state_change_handler: Callable | None = None

        logger.info(
            f"Monitoring orchestrator initialized with debounce: "
            f"{debounce_required_updates} updates / {debounce_interval_seconds}s"
        )

    async def start(self):
        """Start monitoring orchestrator"""

        # Register state change handler
        async def on_state_changed(
            old_state: NormalizedPrinterState, new_state: NormalizedPrinterState
        ):
            await self._handle_state_change(old_state, new_state)

        self._state_change_handler = on_state_changed
        self.adapter.on_state_changed(on_state_changed)

        logger.info("Monitoring orchestrator started")

    async def stop(self):
        """Stop monitoring orchestrator"""
        if self._state_change_handler:
            self.adapter.remove_state_changed_handler(self._state_change_handler)

        # Cancel debounce timer if active
        if self._debounce_timer and not self._debounce_timer.done():
            self._debounce_timer.cancel()
            try:
                await self._debounce_timer
            except asyncio.CancelledError:
                pass

        logger.info("Monitoring orchestrator stopped")

    async def _handle_state_change(
        self,
        old_state: NormalizedPrinterState,
        new_state: NormalizedPrinterState,
    ):
        """
        Handle printer state change

        Primary entry point for external print detection.
        """
        logger.debug(f"Printer state changed: {old_state.name} -> {new_state.name}")

        # Only handle transitions to PRINTING state
        if new_state != NormalizedPrinterState.PRINTING:
            return

        # Check if this is an external print (not agent-started)
        if self.state_machine.current_state != AgentState.IDLE:
            # Agent is already in a workflow - this is an agent-started print
            logger.debug("State change ignored - agent is not IDLE (agent-started print)")
            return

        # External print detected - apply debounce
        await self._apply_debounce_detection()

    async def _apply_debounce_detection(self):
        """
        Apply debounce logic for external print detection

        Debouncing prevents false positives from transient state changes.
        Requires N updates within time window to trigger external print detection.
        """
        now = datetime.now()

        # Check if we're within the debounce window
        if self._debounce_window_start is None:
            # First update - start window
            self._debounce_window_start = now
            self._debounce_counter = 1
            logger.debug(f"Debounce window started (1/{self.debounce_required_updates} updates)")

            # Schedule window reset after interval
            if self._debounce_timer:
                self._debounce_timer.cancel()

            self._debounce_timer = asyncio.create_task(self._reset_debounce_window_after_delay())
            return

        # Calculate time since window start
        elapsed = (now - self._debounce_window_start).total_seconds()

        if elapsed <= self.debounce_interval:
            # Within window - increment counter
            self._debounce_counter += 1
            logger.debug(
                f"Debounce update {self._debounce_counter}/{self.debounce_required_updates} "
                f"within {elapsed:.1f}s window"
            )

            # Check if we've reached threshold
            if self._debounce_counter >= self.debounce_required_updates:
                logger.info(
                    f"External print detected: {self._debounce_counter} updates in "
                    f"{elapsed:.1f}s (threshold: {self.debounce_required_updates})"
                )
                await self._trigger_external_print_detection()
        else:
            # Window expired - reset
            logger.debug(f"Debounce window expired after {elapsed:.1f}s, resetting")
            await self._reset_debounce_state()

    async def _reset_debounce_window_after_delay(self):
        """Reset debounce window after delay"""
        try:
            await asyncio.sleep(self.debounce_interval)
            await self._reset_debounce_state()
        except asyncio.CancelledError:
            # Timer cancelled - expected during state transitions
            pass

    async def _reset_debounce_state(self):
        """Reset debounce state"""
        self._debounce_counter = 0
        self._debounce_window_start = None
        self._debounce_timer = None
        logger.debug("Debounce state reset")

    async def _trigger_external_print_detection(self):
        """Trigger external print detection workflow"""
        # Capture counter value before reset for logging
        detection_counter = self._debounce_counter

        # Generate job ID for external print
        job_id = generate_job_id()

        # Create job context for external print
        # Note: external prints are NOTHING-only monitoring
        # First-layer correction is disabled for safety
        job_context = JobContext(
            job_id=job_id,
            job_file="external_print",  # Placeholder - no actual file
            uploaded_at=datetime.now(),
            is_external=True,
            external_monitoring=True,  # Critical: external prints are NOTHING-only
        )

        # Set job context in state machine
        self.state_machine.job_context = job_context

        # Transition to MONITORING state
        # External prints skip RECEIVING/SLICING/APPROVAL/UPLOADING
        success = await self.state_machine.transition_to(
            AgentState.MONITORING,
            self.persistence,
        )

        # Reset debounce state AFTER transition
        await self._reset_debounce_state()

        if success:
            logger.info(
                f"External print monitoring started (job_id={job_id}, "
                f"external_monitoring={job_context.external_monitoring}, "
                f"debounce_count={detection_counter})"
            )
            self._is_monitoring = True
            self._monitoring_job_id = job_id

            # Start NOTHING detection (TODO: integrate with vision module when ready)
            # await self.NOTHING(job_context)
        else:
            logger.error("Failed to transition to MONITORING for external print")
            self.state_machine.job_context = None

    async def start_monitoring_for_agent_print(self, job_context: JobContext):
        """
        Start monitoring for agent-started print

        Called after agent successfully uploads and starts print.

        Args:
            job_context: Job context from agent workflow
        """
        if self.state_machine.current_state != AgentState.PRINTING:
            logger.error(
                f"Cannot start monitoring - state is {self.state_machine.current_state.name}, "
                "expected PRINTING"
            )
            return

        # Set job context
        self.state_machine.job_context = job_context

        # Transition to MONITORING
        success = await self.state_machine.transition_to(
            AgentState.MONITORING,
            self.persistence,
        )

        if success:
            logger.info(
                f"Agent-started print monitoring started (job_id={job_context.job_id}, "
                f"external_monitoring={job_context.external_monitoring})"
            )
            self._is_monitoring = True
            self._monitoring_job_id = job_context.job_id

            # Start full monitoring (NOTHING + first-layer if enabled)
            # await self._start_full_monitoring(job_context)
        else:
            logger.error("Failed to transition to MONITORING for agent print")

    async def stop_monitoring(self):
        """Stop active monitoring"""
        if not self._is_monitoring:
            logger.debug("No active monitoring to stop")
            return

        logger.info(f"Stopping monitoring for job {self._monitoring_job_id}")

        self._is_monitoring = False
        self._monitoring_job_id = None

        # Cleanup debounce state
        await self._reset_debounce_state()

    async def handle_print_pause(self):
        """Handle print pause event"""
        if not self._is_monitoring:
            return

        # Transition to PAUSED_PENDING_USER
        # This allows user to resume or cancel
        await self.state_machine.transition_to(
            AgentState.PAUSED_PENDING_USER,
            self.persistence,
        )

    async def NOTHING(self, confidence: float):
        """
        Handle NOTHING detection pause request

        Called when moderate confidence NOTHING is detected.

        Args:
            confidence: Detection confidence (0.0-1.0)
        """
        if not self._is_monitoring:
            return

        logger.info(f"NOTHING detected (confidence={confidence:.2f}), pausing for user review")

        await self.handle_print_pause()

    async def NOTHING(self, confidence: float):
        """
        Handle NOTHING detection emergency cancel

        Called when high confidence NOTHING is detected.

        Args:
            confidence: Detection confidence (0.0-1.0)
        """
        if not self._is_monitoring:
            return

        logger.warning(
            f"CRITICAL: High confidence NOTHING detected (confidence={confidence:.2f}), "
            "triggering emergency cancel"
        )

        # Transition to EMERGENCY_FAILED
        await self.state_machine.transition_to(
            AgentState.EMERGENCY_FAILED,
            self.persistence,
        )

    # TODO: Integrate with vision modules
    # async def NOTHING(self, job_context: JobContext):
    #     """Start NOTHING-only monitoring"""
    #     pass
    #
    # async def _start_full_monitoring(self, job_context: JobContext):
    #     """Start full monitoring (NOTHING + first-layer)"""
    #     pass
