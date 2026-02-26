"""
BUG TESTS: Concurrency and Race Conditions

These tests expose concurrency issues in the codebase.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from sensing_voxel.core.decision import DecisionEngine
from sensing_voxel.core.state_machine import AgentStateMachine
from sensing_voxel.core.models import (
    DecisionAction,
    ConsumeResult,
    AgentState,
    JobContext,
)
from sensing_voxel.adapters.command_queue import CommandQueue, CommandPriority
from datetime import datetime


class TestConcurrencyBugs:
    """Tests exposing race conditions and concurrency issues"""

    @pytest.mark.asyncio
    async def test_concurrent_consume_decision_race(self):
        """
        BUG: Multiple concurrent consume_decision calls can race.

        The lock is acquired after checking if decision is consumed,
        but the consumed flag is checked inside the lock. However,
        if two calls arrive simultaneously, both could pass the initial
        checks and one could consume while the other is waiting.

        This is LOW severity - the lock should prevent this, but testing
        the actual behavior is important.
        """
        engine = DecisionEngine()

        job = JobContext(
            job_id="race-test",
            job_file="test.stl",
            uploaded_at=datetime.now(),
            decision_token="test-token",
            decision_consumed=False,
        )

        results = []

        async def consume():
            result = await engine.consume_decision(
                job_context=job,
                decision_token="test-token",
                action=DecisionAction.APPROVE,
                expected_state=AgentState.AWAITING_APPROVAL,
                current_state=AgentState.AWAITING_APPROVAL,
            )
            results.append(result)

        # Start multiple concurrent consumes
        tasks = [asyncio.create_task(consume()) for _ in range(5)]
        await asyncio.gather(*tasks)

        # Exactly one should be CONSUMED, rest should be ALREADY_HANDLED
        consumed_count = sum(1 for r in results if r == ConsumeResult.CONSUMED)
        already_handled_count = sum(1 for r in results if r == ConsumeResult.ALREADY_HANDLED)

        assert consumed_count == 1, (
            f"BUG: Race condition in consume_decision. "
            f"Expected 1 CONSUMED, got {consumed_count}. "
            f"Results: {results}"
        )
        assert already_handled_count == 4, (
            f"BUG: Expected 4 ALREADY_HANDLED, got {already_handled_count}"
        )

    @pytest.mark.asyncio
    async def test_state_machine_transition_not_atomic(self):
        """
        BUG: State machine transition is not atomic - state changes before
        persistence and hooks complete.

        If a crash occurs between state change and persistence, the system
        will be in an inconsistent state.

        This is MEDIUM severity - can lead to state inconsistency.
        """
        sm = AgentStateMachine()

        # Mock persistence to fail
        failing_persistence = Mock()
        failing_persistence.save = Mock(side_effect=IOError("Disk full"))

        original_state = sm.current_state

        # Transition with failing persistence
        success = await sm.transition_to(AgentState.RECEIVING, failing_persistence)

        # State changed in memory
        assert sm.current_state == AgentState.RECEIVING

        # But persistence failed - inconsistency!
        # On restart, system will think it's IDLE but it's actually RECEIVING
        assert success is True, (
            "BUG: Transition reported success but persistence failed. "
            "State is now inconsistent between memory and disk."
        )

    @pytest.mark.asyncio
    async def test_decision_engine_lock_leak_on_exception(self):
        """
        BUG: If an exception occurs during consume_decision after the lock
        is acquired, the lock is released but the _decision_locks entry
        remains forever.

        This is MEDIUM severity - memory leak under error conditions.
        """
        engine = DecisionEngine()

        job = JobContext(
            job_id="leak-test",
            job_file="test.stl",
            uploaded_at=datetime.now(),
            decision_token="test-token",
        )

        # Make job_context raise on attribute access after lock is acquired
        original_decision_consumed = job.decision_consumed

        def raise_on_second_access(*args, **kwargs):
            nonlocal access_count
            access_count += 1
            if access_count > 1:
                raise RuntimeError("Simulated error")
            return original_decision_consumed

        access_count = 0

        with patch.object(type(job), 'decision_consumed', property(raise_on_second_access)):
            try:
                await engine.consume_decision(
                    job_context=job,
                    decision_token="test-token",
                    action=DecisionAction.APPROVE,
                    expected_state=AgentState.AWAITING_APPROVAL,
                    current_state=AgentState.AWAITING_APPROVAL,
                )
            except RuntimeError:
                pass  # Expected

        # After exception, lock should be released but entry remains
        # BUG: The lock entry is never cleaned up
        assert "leak-test" in engine._decision_locks, (
            "BUG: Lock entry not cleaned up after exception. "
            "Memory leak under error conditions."
        )

    @pytest.mark.asyncio
    async def test_command_queue_processor_restart_race(self):
        """
        BUG: If stop() is called and immediately followed by reset() and
        start_processor(), the old processor might still be running.

        This is LOW severity - edge case in lifecycle management.
        """
        queue = CommandQueue()

        processor1 = AsyncMock()

        # Start first processor
        task1 = asyncio.create_task(queue.start_processor(processor1))
        await asyncio.sleep(0.05)

        # Stop and immediately reset
        queue.stop()
        queue.reset()

        # Start second processor before first fully stops
        processor2 = AsyncMock()
        task2 = asyncio.create_task(queue.start_processor(processor2))

        await asyncio.sleep(0.1)

        # Both might be running - this is problematic
        queue.stop()

        try:
            await asyncio.wait_for(task1, timeout=0.5)
        except asyncio.TimeoutError:
            task1.cancel()
        except asyncio.CancelledError:
            pass

        try:
            await asyncio.wait_for(task2, timeout=0.5)
        except asyncio.TimeoutError:
            task2.cancel()
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_websocket_reconnect_counter_overflow(self):
        """
        BUG: KlipperAdapter._reconnect_attempts can theoretically overflow
        if the printer is down for a very long time.

        This is LOW severity - theoretical issue.
        """
        from sensing_voxel.adapters.klipper import KlipperAdapter

        adapter = KlipperAdapter(host="localhost", port=7125)

        # Simulate many reconnection attempts
        adapter._reconnect_attempts = 2**31 - 1  # Near int32 max

        # Compute delay - this could theoretically overflow
        delay = min(
            adapter._reconnect_policy.base_delay
            * (adapter._reconnect_policy.factor ** adapter._reconnect_attempts),
            adapter._reconnect_policy.max_delay,
        )

        # Should be capped at max_delay, but the exponentiation might overflow
        # before the min() is applied
        assert delay <= adapter._reconnect_policy.max_delay

    @pytest.mark.asyncio
    async def test_timeout_task_cancellation_race(self):
        """
        BUG: In consume_decision(), if timeout task completes between
        the check and the cancel(), we try to cancel an already-done task.

        The current code handles this, but let's verify.
        """
        engine = DecisionEngine()
        engine._approval_timeout = 0.01  # 10ms

        job = JobContext(
            job_id="cancel-race",
            job_file="test.stl",
            uploaded_at=datetime.now(),
            decision_token="test-token",
        )

        # Schedule timeout
        engine.schedule_timeout("cancel-race", job, DecisionAction.TIMEOUT_APPROVAL)

        # Wait for timeout to complete
        await asyncio.sleep(0.05)

        # Now try to consume - this should handle already-done task gracefully
        result = await engine.consume_decision(
            job_context=job,
            decision_token="test-token",
            action=DecisionAction.APPROVE,
            expected_state=AgentState.AWAITING_APPROVAL,
            current_state=AgentState.AWAITING_APPROVAL,
        )

        # Should succeed without raising
        assert result in [ConsumeResult.CONSUMED, ConsumeResult.ALREADY_HANDLED]
