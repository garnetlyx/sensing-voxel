"""
Decision idempotency engine with lock-guarded CAS
"""

import asyncio
import logging
from datetime import datetime

from sensing_voxel.core.models import (
    AgentState,
    ConsumeResult,
    DecisionAction,
    JobContext,
)

logger = logging.getLogger(__name__)


class DecisionEngine:
    """Decision idempotency engine"""

    def __init__(self):
        self._decision_locks: dict[str, asyncio.Lock] = {}
        self._timeout_tasks: dict[str, asyncio.Task] = {}
        self._approval_timeout: int = 600  # 10 minutes
        self.NOTHING: int = 300  # 5 minutes

    def _get_lock(self, job_id: str) -> asyncio.Lock:
        """Get or create lock for job_id"""
        # Always cache locks for all jobs - prevents race condition where
        # concurrent consume_decision() calls get different lock instances
        # Memory leak prevention is handled by explicit cleanup after consumption
        if job_id not in self._decision_locks:
            self._decision_locks[job_id] = asyncio.Lock()
        return self._decision_locks[job_id]

    def _cleanup_job_resources(self, job_id: str):
        """Clean up resources for completed job"""
        if job_id in self._decision_locks:
            del self._decision_locks[job_id]
        if job_id in self._timeout_tasks:
            task = self._timeout_tasks[job_id]
            if not task.done():
                task.cancel()
            del self._timeout_tasks[job_id]
        logger.debug(f"Cleaned up resources for job {job_id}")

    def cleanup_all_resources(self):
        """Clean up all resources (useful for shutdown or periodic cleanup)"""
        job_ids = list(self._decision_locks.keys()) + list(self._timeout_tasks.keys())
        for job_id in set(job_ids):
            self._cleanup_job_resources(job_id)
        logger.info("Cleaned up all job resources")

    def cleanup_stale_locks(self, max_age_seconds: int = 3600):
        """
        Clean up locks that haven't been used recently.
        This is a safety mechanism to prevent memory leaks from abandoned jobs.
        """
        # Since we don't track lock usage time, we just clean all locks
        # In a real implementation, we would track last access time
        count = len(self._decision_locks)
        if count > 0:
            logger.warning(f"Cleaning up {count} stale locks")
            for job_id in list(self._decision_locks.keys()):
                if job_id not in self._timeout_tasks:
                    self._cleanup_job_resources(job_id)

    def _get_timeout_for_action(self, action: DecisionAction) -> int:
        """Get timeout duration for action type"""
        if action in [
            DecisionAction.TIMEOUT_APPROVAL,
            DecisionAction.APPROVE,
            DecisionAction.REJECT,
        ]:
            return self._approval_timeout
        elif action in [
            DecisionAction.NOTHING,
            DecisionAction.RESUME,
            DecisionAction.CANCEL,
        ]:
            return self.NOTHING
        return self._approval_timeout

    async def _timeout_handler(self, job_id: str, job_context: JobContext, action: DecisionAction):
        """Handle decision timeout"""
        timeout = self._get_timeout_for_action(action)

        try:
            await asyncio.sleep(timeout)

            # Try to consume timeout decision
            result = await self.consume_decision(
                job_context=job_context,
                decision_token=job_context.decision_token,
                action=action,
                expected_state=None,  # No state check for timeout
                current_state=None,  # Will be fetched fresh
            )

            if result == ConsumeResult.CONSUMED:
                logger.info(f"Timeout decision {action.name} consumed for job {job_id}")
        except asyncio.CancelledError:
            logger.debug(f"Timeout task for job {job_id} was cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in timeout handler for job {job_id}: {e}")
            # Don't propagate - task should complete gracefully

    async def consume_decision(
        self,
        job_context: JobContext,
        decision_token: str,
        action: DecisionAction,
        expected_state: AgentState | None,
        current_state: AgentState | None,
    ) -> ConsumeResult:
        """
        Consume a decision with lock-guarded CAS

        Args:
            job_context: Job context
            decision_token: Decision token to verify
            action: Decision action
            expected_state: Expected agent state (None for timeout)
            current_state: Current agent state (None for timeout)

        Returns:
            ConsumeResult indicating outcome
        """
        job_id = job_context.job_id

        # Acquire per-job lock
        lock = self._get_lock(job_id)

        try:
            async with lock:
                # Cancel any pending timeout for this job
                if job_id in self._timeout_tasks:
                    task = self._timeout_tasks[job_id]
                    task.cancel()
                    try:
                        # Wait for the task to be cancelled (handles CancelledError)
                        await asyncio.wait_for(task, timeout=1.0)
                    except (TimeoutError, asyncio.CancelledError):
                        # Task was cancelled or timed out - both are expected
                        pass
                    except Exception as e:
                        # Log unexpected errors but don't propagate
                        logger.warning(f"Error while cancelling timeout task for {job_id}: {e}")
                    finally:
                        # Always clean up the task reference
                        if job_id in self._timeout_tasks:
                            del self._timeout_tasks[job_id]

                # Re-check guards (skip for timeout actions)
                if action not in [
                    DecisionAction.TIMEOUT_APPROVAL,
                    DecisionAction.NOTHING,
                ]:
                    if current_state is None or expected_state is None:
                        logger.error(
                            f"State validation required for non-timeout action {action.name}"
                        )
                        # Clean up lock before returning
                        self._cleanup_job_resources(job_id)
                        return ConsumeResult.INVALID_STATE

                    if current_state != expected_state:
                        logger.warning(
                            f"State mismatch for job {job_id}: "
                            f"expected {expected_state.name}, got {current_state.name}"
                        )
                        # Clean up lock before returning
                        self._cleanup_job_resources(job_id)
                        return ConsumeResult.INVALID_STATE

                if job_context.decision_consumed:
                    logger.info(f"Decision already consumed for job {job_id}")
                    # Clean up resources for already-consumed job
                    self._cleanup_job_resources(job_id)
                    return ConsumeResult.ALREADY_HANDLED

                if job_context.decision_token != decision_token:
                    logger.warning(f"Invalid decision token for job {job_id}")
                    # Clean up lock before returning
                    self._cleanup_job_resources(job_id)
                    return ConsumeResult.INVALID_TOKEN

                # All guards passed - consume decision
                job_context.decision_consumed = True
                job_context.decision_action = action
                job_context.decision_consumed_at = datetime.now()

                logger.info(
                    f"Consumed decision {action.name} for job {job_id} "
                    f"in state {current_state.name if current_state else 'N/A'}"
                )

                # Clean up lock immediately after consuming decision
                # This prevents memory leak - lock is no longer needed
                self._cleanup_job_resources(job_id)

                return ConsumeResult.CONSUMED
        except Exception as e:
            logger.error(f"Exception in consume_decision for job {job_id}: {e}", exc_info=True)
            # Ensure cleanup on any exception
            self._cleanup_job_resources(job_id)
            raise

    def schedule_timeout(
        self,
        job_id: str,
        job_context: JobContext,
        timeout_action: DecisionAction,
    ):
        """Schedule timeout for decision"""
        # Check if task exists and clean up if completed
        if job_id in self._timeout_tasks:
            task = self._timeout_tasks[job_id]
            if task.done():
                # Task is completed, clean it up
                del self._timeout_tasks[job_id]
                logger.debug(f"Cleaned up completed timeout task for job {job_id}")
            else:
                logger.debug(f"Timeout already running for job {job_id}")
                return

        task = asyncio.create_task(self._timeout_handler(job_id, job_context, timeout_action))
        self._timeout_tasks[job_id] = task

        logger.debug(f"Scheduled {timeout_action.name} timeout for job {job_id}")

    def cancel_timeout(self, job_id: str):
        """Cancel scheduled timeout"""
        if job_id in self._timeout_tasks:
            task = self._timeout_tasks[job_id]
            task.cancel()
            try:
                # Create a background task to wait for cancellation
                # We don't block here since this is a sync method
                # The task will clean itself up when done
                pass
            except (asyncio.CancelledError, RuntimeError):
                # Task was cancelled or already finished - expected
                pass
            finally:
                # Always clean up the reference
                if job_id in self._timeout_tasks:
                    del self._timeout_tasks[job_id]
            logger.debug(f"Cancelled timeout for job {job_id}")

    def set_timeouts(
        self,
        approval_timeout_seconds: int,
        NOTHING: int,
    ):
        """Configure timeout durations"""
        self._approval_timeout = approval_timeout_seconds
        self.NOTHING = NOTHING
        logger.info(
            f"Timeouts configured: approval={approval_timeout_seconds}s, "
            f"NOTHING_pause={NOTHING}s"
        )

    def get_timeouts(self) -> tuple[int, int]:
        """Get current timeout durations (for testing)"""
        return self._approval_timeout, self.NOTHING
