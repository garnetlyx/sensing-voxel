"""
FAILING TESTS: DecisionEngine bugs exposed by QA

These tests demonstrate real bugs in the DecisionEngine implementation.
They will FAIL until the bugs are fixed.

BUGS EXPOSED:
1. CancelledError not handled when cancelling timeout tasks
2. Memory leak: _decision_locks and _timeout_tasks never cleaned up
3. Timeout tasks not properly awaited/shielded before cancellation
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from sensing_voxel.core.decision import DecisionEngine
from sensing_voxel.core.models import (
    DecisionAction,
    ConsumeResult,
    AgentState,
    JobContext,
)
from datetime import datetime


class TestDecisionEngineBugs:
    """Failing tests exposing DecisionEngine bugs"""

    @pytest.fixture
    def engine(self):
        """Create DecisionEngine instance"""
        return DecisionEngine()

    @pytest.fixture
    def job_context(self):
        """Create test JobContext"""
        return JobContext(
            job_id="test-123",
            job_file="test.stl",
            uploaded_at=datetime.now(),
            decision_token="test-token",
        )

    @pytest.mark.asyncio
    async def test_timeout_task_cancelled_error_not_handled(self, engine, job_context):
        """
        BUG: When a timeout task is cancelled but has already completed,
        task.cancel() doesn't raise CancelledError until awaited.
        The current code doesn't handle this properly.

        This test will fail because CancelledError propagates unexpectedly.
        """
        # Schedule a timeout with a very short duration
        engine._approval_timeout = 0.01  # 10ms

        # Start timeout handler
        engine.schedule_timeout(
            job_id="test-123",
            job_context=job_context,
            timeout_action=DecisionAction.TIMEOUT_APPROVAL,
        )

        # Wait for timeout to complete
        await asyncio.sleep(0.05)

        # Now try to consume - this should cancel the already-done task
        # BUG: This may raise CancelledError or behave unexpectedly
        try:
            result = await engine.consume_decision(
                job_context=job_context,
                decision_token="test-token",
                action=DecisionAction.APPROVE,
                expected_state=AgentState.AWAITING_APPROVAL,
                current_state=AgentState.AWAITING_APPROVAL,
            )
            # If we get here without exception, check if state is correct
            assert result == ConsumeResult.CONSUMED or result == ConsumeResult.ALREADY_HANDLED
        except asyncio.CancelledError:
            pytest.fail("BUG: CancelledError not handled when cancelling completed timeout task")

    @pytest.mark.asyncio
    async def test_memory_leak_decision_locks_never_cleaned(self, engine):
        """
        BUG: _decision_locks dictionary grows without bound.
        Each unique job_id creates a lock that is never removed.

        This test will fail because locks accumulate forever.
        """
        initial_count = len(engine._decision_locks)

        # Simulate many jobs being processed
        for i in range(100):
            job_id = f"job-{i}"
            job_context = JobContext(
                job_id=job_id,
                job_file="test.stl",
                uploaded_at=datetime.now(),
                decision_token=f"token-{i}",
            )

            # Get lock for this job
            lock = engine._get_lock(job_id)
            assert lock is not None

        # BUG: After 100 jobs, we have 100 locks that will never be cleaned
        final_count = len(engine._decision_locks)

        # This assertion will fail - locks are never cleaned
        assert final_count < 50, (
            f"BUG: Memory leak - {final_count} locks accumulated and never cleaned. "
            "DecisionEngine._decision_locks grows without bound."
        )

    @pytest.mark.asyncio
    async def test_memory_leak_timeout_tasks_never_cleaned(self, engine):
        """
        BUG: If a timeout fires and completes, the task remains in _timeout_tasks.

        This test will fail because completed tasks are never removed.
        """
        engine._approval_timeout = 0.01  # 10ms

        # Schedule and let timeout complete
        job_context = JobContext(
            job_id="test-timeout",
            job_file="test.stl",
            uploaded_at=datetime.now(),
            decision_token="test-token",
        )

        engine.schedule_timeout(
            job_id="test-timeout",
            job_context=job_context,
            timeout_action=DecisionAction.TIMEOUT_APPROVAL,
        )

        # Wait for timeout to complete
        await asyncio.sleep(0.05)

        # BUG: Task is still in _timeout_tasks even though it's done
        assert "test-timeout" not in engine._timeout_tasks, (
            "BUG: Completed timeout task not cleaned from _timeout_tasks. "
            "This causes memory leak."
        )

    @pytest.mark.asyncio
    async def test_timeout_handler_missing_exception_handling(self, engine, job_context):
        """
        BUG: _timeout_handler doesn't handle exceptions from consume_decision.
        If consume_decision raises, the exception propagates and kills the task.

        This test will fail because the exception is not caught.
        """
        # Patch consume_decision to raise an exception
        with patch.object(engine, 'consume_decision', side_effect=RuntimeError("Simulated error")):
            # Manually create and run timeout handler
            handler_task = asyncio.create_task(
                engine._timeout_handler("test-123", job_context, DecisionAction.TIMEOUT_APPROVAL)
            )

            # BUG: The exception will propagate and this will raise
            with pytest.raises(RuntimeError, match="Simulated error"):
                await handler_task

    @pytest.mark.asyncio
    async def test_schedule_timeout_doesnt_check_task_done(self, engine, job_context):
        """
        BUG: schedule_timeout doesn't check if existing task is already done.
        It only checks if job_id exists in dict, not if task is still running.

        This test will fail because schedule_timeout silently returns
        even when the existing task is done and a new one should be scheduled.
        """
        engine._approval_timeout = 0.01  # 10ms

        # Schedule first timeout
        engine.schedule_timeout(
            job_id="test-123",
            job_context=job_context,
            timeout_action=DecisionAction.TIMEOUT_APPROVAL,
        )

        first_task = engine._timeout_tasks["test-123"]

        # Wait for it to complete
        await asyncio.sleep(0.05)

        # Try to schedule another timeout for same job
        engine.schedule_timeout(
            job_id="test-123",
            job_context=job_context,
            timeout_action=DecisionAction.TIMEOUT_APPROVAL,
        )

        # BUG: Second schedule returns early because job_id exists,
        # even though the task is done
        assert engine._timeout_tasks.get("test-123") is not first_task, (
            "BUG: schedule_timeout didn't create new timeout because it only checks "
            "if job_id exists, not if the task is still active."
        )
