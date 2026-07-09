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
        FIXED: DecisionEngine now cleans up per-job entries in _decision_locks
        (and _timeout_tasks) after consume_decision completes, invalid-token returns,
        or error paths. Regression guard: processing many jobs must not grow the map
        without bound.
        """
        initial_count = len(engine._decision_locks)

        for i in range(100):
            job_id = f"job-{i}"
            job_context = JobContext(
                job_id=job_id,
                job_file="test.stl",
                uploaded_at=datetime.now(),
                decision_token=f"token-{i}",
            )
            lock = engine._get_lock(job_id)
            assert lock is not None

        # Locks were created for all 100 jobs via _get_lock
        assert len(engine._decision_locks) == initial_count + 100

        # Now consume decisions; locks must be cleaned up afterward
        for i in range(100):
            job_id = f"job-{i}"
            job_context = JobContext(
                job_id=job_id,
                job_file="test.stl",
                uploaded_at=datetime.now(),
                decision_token=f"token-{i}",
            )
            await engine.consume_decision(
                job_context=job_context,
                decision_token=f"token-{i}",
                action=DecisionAction.APPROVE,
                expected_state=AgentState.AWAITING_APPROVAL,
                current_state=AgentState.AWAITING_APPROVAL,
            )

        final_count = len(engine._decision_locks)
        assert final_count == initial_count, (
            f"Memory leak regression: {final_count} locks remain after consuming all jobs; "
            "locks must be cleaned up when decisions are fully consumed."
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
        _timeout_handler must swallow exceptions from consume_decision so that
        a failing decision does not kill the background timeout task.
        """
        engine._approval_timeout = 0.01

        with patch.object(engine, 'consume_decision', side_effect=RuntimeError("Simulated error")):
            handler_task = asyncio.create_task(
                engine._timeout_handler("test-123", job_context, DecisionAction.TIMEOUT_APPROVAL)
            )

            await asyncio.wait_for(handler_task, timeout=5)

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
