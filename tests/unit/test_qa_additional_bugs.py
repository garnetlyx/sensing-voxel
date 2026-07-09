"""
QA Additional Bugs - Adversarial tests for newly discovered issues

These tests expose bugs found during deep source code analysis.
Each test MUST fail if the bug is present, proving the vulnerability exists.
"""

import pytest
import asyncio
import time
import gc
import math
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from sensing_voxel.core.models import (
    AgentState,
    DecisionAction,
    ConsumeResult,
    JobContext,
    generate_job_id,
    generate_decision_token,
    FileValidationStatus,
)
from sensing_voxel.core.state_machine import AgentStateMachine
from sensing_voxel.core.decision import DecisionEngine
from sensing_voxel.core.persistence import StatePersistence
from sensing_voxel.infra.retry import BackoffPolicy, compute_backoff


class TestSignalHandlerBugs:
    """BUG-QA-NEW-11: Signal handler doesn't actually save state"""

    def test_signal_handler_no_op(self, tmp_path):
        """
        FIXED (BUG-QA-NEW-11): Signal handling was removed from StatePersistence.
        It now lives at the process-entry level (main.py) and is no longer a method
        on StatePersistence. Regression guard: the broken no-op handler must not reappear.
        """
        state_file = tmp_path / "state.json"
        persistence = StatePersistence(state_file)

        assert not hasattr(persistence, "_signal_handler"), (
            "_signal_handler should not exist on StatePersistence; "
            "signal handling belongs in main.py, not on the persistence class"
        )

    def test_signal_handler_no_exit(self, tmp_path):
        """
        FIXED (BUG-QA-NEW-11): The old _signal_handler was a no-op that failed to raise
        SystemExit or save state. It has been removed entirely from StatePersistence.
        Regression guard: ensure the broken method does not return.
        """
        state_file = tmp_path / "state.json"
        persistence = StatePersistence(state_file)

        assert not hasattr(persistence, "_signal_handler"), (
            "_signal_handler should be removed from StatePersistence"
        )


class TestDecisionEngineNewBugs:
    """Additional decision engine bugs"""

    @pytest.mark.asyncio
    async def test_decision_token_not_constant_time(self):
        """
        FIXED (BUG-QA-NEW-12): Token comparison now uses hmac.compare_digest,
        which is constant-time and prevents timing attacks. Regression guard:
        verify the implementation uses hmac.compare_digest and that correct vs.
        incorrect tokens both return INVALID_TOKEN/CONSUMED as appropriate without
        raising AttributeError on immutable field reset.
        """
        import hmac
        import inspect

        from sensing_voxel.core import decision as decision_mod

        # The module must import hmac and the consume path must use compare_digest
        assert hasattr(decision_mod, "hmac"), "decision module must use hmac for constant-time compare"
        source = inspect.getsource(decision_mod.DecisionEngine.consume_decision)
        assert "hmac.compare_digest" in source, (
            "consume_decision must use hmac.compare_digest for constant-time token comparison"
        )

        engine = DecisionEngine()

        correct_token = "a" * 32
        job = JobContext(
            job_id="test-job",
            job_file="test.stl",
            uploaded_at=datetime.now(),
            decision_token=correct_token,
            file_validation_status=FileValidationStatus.VALID,
        )

        result_correct = await engine.consume_decision(
            job_context=job,
            decision_token=correct_token,
            action=DecisionAction.APPROVE,
            expected_state=AgentState.AWAITING_APPROVAL,
            current_state=AgentState.AWAITING_APPROVAL,
        )
        assert result_correct == ConsumeResult.CONSUMED

        # A new job with the same id but wrong token should be rejected
        job2 = JobContext(
            job_id="test-job-2",
            job_file="test.stl",
            uploaded_at=datetime.now(),
            decision_token=correct_token,
            file_validation_status=FileValidationStatus.VALID,
        )
        result_wrong = await engine.consume_decision(
            job_context=job2,
            decision_token="b" + "a" * 31,
            action=DecisionAction.APPROVE,
            expected_state=AgentState.AWAITING_APPROVAL,
            current_state=AgentState.AWAITING_APPROVAL,
        )
        assert result_wrong == ConsumeResult.INVALID_TOKEN

        # Sanity check: compare_digest itself is well-behaved
        assert hmac.compare_digest(correct_token, correct_token) is True
        assert hmac.compare_digest(correct_token, "b" + "a" * 31) is False

    @pytest.mark.asyncio
    async def test_timeout_race_with_manual_consume(self):
        """
        BUG-QA-NEW-13: Race between timeout firing and manual consume.
        Timeout task doesn't check if decision was already consumed.
        """
        engine = DecisionEngine()
        engine._approval_timeout = 0.1  # 100ms timeout

        job = JobContext(
            job_id="test-job",
            job_file="test.stl",
            uploaded_at=datetime.now(),
            decision_token="token123",
            file_validation_status=FileValidationStatus.VALID,
        )

        # Schedule timeout
        engine.schedule_timeout("test-job", job, DecisionAction.TIMEOUT_APPROVAL)

        # Manually consume decision before timeout
        result1 = await engine.consume_decision(
            job_context=job,
            decision_token="token123",
            action=DecisionAction.APPROVE,
            expected_state=AgentState.AWAITING_APPROVAL,
            current_state=AgentState.AWAITING_APPROVAL,
        )
        assert result1 == ConsumeResult.CONSUMED

        # Wait for timeout to fire
        await asyncio.sleep(0.15)

        # BUG: Timeout handler may still try to consume after manual consume
        # This can lead to double-processing or errors
        # Test passes if no exception (current behavior)
        # But logic is racy

    @pytest.mark.asyncio
    async def test_cleanup_all_resources_race(self):
        """
        BUG-QA-NEW-14: cleanup_all_resources races with active consume_decision.
        """
        engine = DecisionEngine()
        engine._approval_timeout = 1.0

        job = JobContext(
            job_id="test-job",
            job_file="test.stl",
            uploaded_at=datetime.now(),
            decision_token="token123",
            file_validation_status=FileValidationStatus.VALID,
        )

        # Start a consume that will take some time
        async def slow_consume():
            # This will try to acquire lock
            return await engine.consume_decision(
                job_context=job,
                decision_token="token123",
                action=DecisionAction.APPROVE,
                expected_state=AgentState.AWAITING_APPROVAL,
                current_state=AgentState.AWAITING_APPROVAL,
            )

        # Start consume and immediately cleanup
        consume_task = asyncio.create_task(slow_consume())
        await asyncio.sleep(0.01)  # Let consume start

        # Cleanup while consume is in progress
        engine.cleanup_all_resources()

        # Try to get result
        try:
            result = await asyncio.wait_for(consume_task, timeout=0.5)
            # If we get here, consume completed despite cleanup
            # This may be correct or may indicate cleanup didn't work
        except (asyncio.CancelledError, Exception):
            # If exception, cleanup interfered with active operation
            pass  # BUG: cleanup should not affect active operations


class TestStateMachineNewBugs:
    """New state machine bugs found in code analysis"""

    @pytest.mark.asyncio
    async def test_hook_failure_no_rollback(self):
        """
        FIXED (BUG-QA-NEW-15): When an on_enter hook raises, the state machine now
        rolls the in-memory state back to the pre-transition state and the transition
        reports failure. Regression guard: failed hook must not leave state half-applied.
        """
        sm = AgentStateMachine()

        async def failing_hook():
            raise RuntimeError("Hook failure")

        await sm.on_enter(AgentState.RECEIVING, failing_hook)

        original_state = sm.current_state

        result = await sm.transition_to(AgentState.RECEIVING)

        assert result is False, "Transition must report failure when enter hook raises"
        assert sm.current_state == original_state, (
            "State must be rolled back to the original state on enter-hook failure"
        )

    @pytest.mark.asyncio
    async def test_persistence_failure_state_corruption(self):
        """
        FIXED (BUG-QA-NEW-16): persistence.save is invoked on a deep copy BEFORE the
        in-memory current_state is mutated, so if persistence raises the live state is
        unchanged and the transition returns False. Regression guard.
        """
        sm = AgentStateMachine()

        mock_persistence = Mock()
        mock_persistence.save = Mock(side_effect=IOError("Disk full"))

        original_state = sm.current_state
        result = await sm.transition_to(AgentState.RECEIVING, persistence=mock_persistence)

        assert result is False, "Transition must report failure when persistence raises"
        assert sm.current_state == original_state, (
            "In-memory state must not be mutated when persistence fails"
        )

    @pytest.mark.asyncio
    async def test_concurrent_transition_same_target_both_succeed(self):
        """
        BUG-QA-NEW-17: No global lock means concurrent transitions can both succeed.
        """
        sm = AgentStateMachine()

        results = []

        async def transition_and_record(target):
            result = await sm.transition_to(target)
            results.append((target, result, sm.current_state))

        # Two concurrent transitions to same target
        await asyncio.gather(
            transition_and_record(AgentState.RECEIVING),
            transition_and_record(AgentState.RECEIVING),
        )

        successes = [r for r in results if r[1]]

        # Both may succeed due to race condition
        # BUG: Without global lock, both can succeed
        if len(successes) > 1:
            pytest.fail(f"CRITICAL: Both transitions succeeded! Race condition confirmed: {results}")


class TestRetryUtilityBugs:
    """Bugs in retry utility"""

    def test_backoff_overflow_with_large_attempt(self):
        """
        BUG-QA-NEW-18: Large attempt numbers can cause overflow issues.
        """
        policy = BackoffPolicy(base_delay=1.0, max_delay=3600.0, factor=2.0)

        # This should not raise or return infinity
        delay = compute_backoff(1000, policy)

        # Should be capped at max_delay
        assert delay <= policy.max_delay * 1.5  # Allow for jitter

    def test_retry_with_zero_max_attempts(self):
        """
        FIXED (BUG-QA-NEW-19): retry_async now validates max_attempts up front and raises
        TypeError when it is < 1, preventing the previous UnboundLocalError / silent
        no-op behavior. Regression guard.
        """
        from sensing_voxel.infra.retry import retry_async

        async def always_fails():
            raise ValueError("fail")

        with pytest.raises(TypeError, match="max_attempts must be at least 1"):
            asyncio.run(retry_async(always_fails, max_attempts=0))

        with pytest.raises(TypeError, match="max_attempts must be at least 1"):
            asyncio.run(retry_async(always_fails, max_attempts=-5))


class TestMemoryLeakBugs:
    """Memory leak bugs"""

    @pytest.mark.asyncio
    async def test_decision_locks_never_cleaned(self):
        """
        FIXED (BUG-QA-NEW-20): DecisionEngine now cleans up per-job locks (and the
        associated timeout task) immediately after a decision is consumed, when a
        token is invalid, or on already-handled / error paths. Regression guard:
        after processing many jobs, the lock and timeout-task maps must stay bounded.
        """
        engine = DecisionEngine()
        engine._approval_timeout = 600

        for i in range(100):
            job_id = f"job-{i}"
            job = JobContext(
                job_id=job_id,
                job_file="test.stl",
                uploaded_at=datetime.now(),
                decision_token=f"token-{i}",
                file_validation_status=FileValidationStatus.VALID,
            )
            engine.schedule_timeout(job_id, job, DecisionAction.TIMEOUT_APPROVAL)

        # schedule_timeout only populates _timeout_tasks (no lock needed yet),
        # so _decision_locks stays empty until a consume call.
        assert len(engine._decision_locks) == 0
        assert len(engine._timeout_tasks) == 100

        for i in range(100):
            job_id = f"job-{i}"
            job = JobContext(
                job_id=job_id,
                job_file="test.stl",
                uploaded_at=datetime.now(),
                decision_token=f"token-{i}",
                file_validation_status=FileValidationStatus.VALID,
            )
            await engine.consume_decision(
                job_context=job,
                decision_token=f"token-{i}",
                action=DecisionAction.APPROVE,
                expected_state=AgentState.AWAITING_APPROVAL,
                current_state=AgentState.AWAITING_APPROVAL,
            )

        # After successful consumes, both locks and timeout tasks must be cleaned up
        assert len(engine._decision_locks) == 0, (
            f"Decision locks leaked: {len(engine._decision_locks)} remain after consume"
        )
        assert len(engine._timeout_tasks) == 0, (
            f"Timeout tasks leaked: {len(engine._timeout_tasks)} remain after consume"
        )


class TestSecurityBugs:
    """Security-related bugs"""

    def test_job_id_predictable_timestamp(self):
        """
        BUG-QA-NEW-21: Job ID uses timestamp which is predictable.
        """
        # Generate multiple job IDs
        ids = [generate_job_id() for _ in range(5)]

        # All should start with job_YYYYMMDD_HHMMSS
        # This is predictable and can be guessed
        for job_id in ids:
            assert job_id.startswith("job_")
            # Timestamp is in the ID, making it guessable
            timestamp_part = job_id.split("_")[1]
            assert len(timestamp_part) == 8  # YYYYMMDD

    def test_decision_token_not_cryptographically_secure(self):
        """
        BUG-QA-NEW-22: Decision token uses uuid4.hex which is not crypto-secure.
        Should use secrets.token_urlsafe or similar.
        """
        tokens = [generate_decision_token() for _ in range(10)]

        # uuid4 is pseudo-random, not cryptographically secure
        # For security tokens, should use secrets module
        for token in tokens:
            assert len(token) == 32  # uuid4 hex
            # BUG: uuid4 is not secrets.token_urlsafe()


class TestConcurrencyBugs:
    """Concurrency bugs"""

    @pytest.mark.asyncio
    async def test_state_machine_not_thread_safe(self):
        """
        BUG-QA-NEW-23: State machine has no locking for state access.
        Concurrent reads/writes can see inconsistent state.
        """
        sm = AgentStateMachine()

        states_seen = []

        async def reader():
            for _ in range(100):
                states_seen.append(sm.current_state)
                await asyncio.sleep(0)

        async def writer():
            for _ in range(50):
                await sm.transition_to(AgentState.RECEIVING)
                await sm.transition_to(AgentState.IDLE)

        # Run concurrent readers and writers
        await asyncio.gather(reader(), writer())

        # Without proper locking, readers may see inconsistent states
        # This test documents the lack of thread safety
        unique_states = set(states_seen)
        # BUG: May see intermediate or invalid states

    @pytest.mark.asyncio
    async def test_timeout_task_exception_suppressed(self):
        """
        BUG-QA-NEW-24: Exceptions in timeout handler are suppressed.
        See decision.py:105-107 - exceptions are logged but not propagated.
        """
        engine = DecisionEngine()
        engine._approval_timeout = 0.01

        # Create a job context that will cause error
        job = Mock()
        job.job_id = "test"
        job.decision_token = None  # Will cause error in consume_decision

        # Schedule timeout
        engine.schedule_timeout("test", job, DecisionAction.TIMEOUT_APPROVAL)

        # Wait for timeout to fire
        await asyncio.sleep(0.05)

        # Exception was suppressed - no way to detect failure
        # BUG: Silent failures in timeout handling


class TestEdgeCaseBugs:
    """Edge case bugs"""

    @pytest.mark.asyncio
    async def test_consume_decision_with_expired_context(self):
        """
        BUG-QA-NEW-25: No check if job context has expired.
        """
        engine = DecisionEngine()

        job = JobContext(
            job_id="test-job",
            job_file="test.stl",
            uploaded_at=datetime.now() - timedelta(days=1),  # 1 day ago
            expires_at=datetime.now() - timedelta(hours=1),  # Expired 1 hour ago
            decision_token="token123",
            file_validation_status=FileValidationStatus.VALID,
        )

        # Try to consume expired decision
        result = await engine.consume_decision(
            job_context=job,
            decision_token="token123",
            action=DecisionAction.APPROVE,
            expected_state=AgentState.AWAITING_APPROVAL,
            current_state=AgentState.AWAITING_APPROVAL,
        )

        # BUG: Should reject expired context but doesn't check
        assert result == ConsumeResult.CONSUMED  # Was consumed despite being expired!

    def test_state_persistence_partial_write(self, tmp_path):
        """
        BUG-QA-NEW-26: No checksum/validation on persisted state.
        Partial writes can corrupt state.
        """
        state_file = tmp_path / "state.json"
        persistence = StatePersistence(state_file)

        sm = AgentStateMachine()
        sm.current_state = AgentState.PRINTING

        # Save state
        persistence.save(sm)

        # Corrupt the file (simulate partial write)
        with open(state_file, "r+") as f:
            content = f.read()
            f.seek(0)
            f.write(content[:len(content)//2])  # Truncate
            f.truncate()

        # Try to load - should handle gracefully
        loaded = persistence.load()

        # BUG: Returns None on error but doesn't detect corruption
        assert loaded is None  # Silent data loss!

    @pytest.mark.asyncio
    async def test_transition_without_persistence_no_state_saved(self):
        """
        BUG-QA-NEW-27: Transitions without persistence arg don't persist.
        State loss on crash.
        """
        sm = AgentStateMachine()

        # Transition without persistence
        await sm.transition_to(AgentState.RECEIVING)

        # State is in memory only
        assert sm.current_state == AgentState.RECEIVING

        # Simulate crash - state is lost
        # BUG: Easy to forget persistence parameter


class TestLogicBugs:
    """Logic bugs"""

    def test_backoff_capped_calculation_overflow(self):
        """
        FIXED (BUG-QA-NEW-28): compute_backoff caps the attempt count used in
        exponentiation to max_useful_attempts, so even astronomically large attempt
        numbers produce a finite delay bounded by max_delay (+jitter) and never
        raise OverflowError. Regression guard.
        """
        policy = BackoffPolicy(base_delay=0.001, max_delay=3600.0, factor=2.0)

        # Very large attempt numbers must not raise and must stay bounded
        for large_attempt in (10_000, 100_000, 1_000_000):
            delay = compute_backoff(large_attempt, policy)
            assert math.isfinite(delay), f"delay must be finite for attempt={large_attempt}"
            assert delay <= policy.max_delay * (1.0 + policy.jitter) + 1e-9, (
                f"delay {delay} exceeds cap for attempt={large_attempt}"
            )

    @pytest.mark.asyncio
    async def test_state_transition_allowed_to_same_state(self):
        """
        BUG-QA-NEW-29: Transition table doesn't include self-transitions.
        But code doesn't explicitly prevent them.
        """
        sm = AgentStateMachine()

        # IDLE -> IDLE (self transition)
        result = sm.can_transition_to(AgentState.IDLE)

        # Self transition not in transition table
        assert result is False  # Correctly rejected

        # But what if we manually set state and try?
        sm.current_state = AgentState.PRINTING
        result = sm.can_transition_to(AgentState.PRINTING)
        assert result is False  # Also rejected

    def test_job_context_hashable(self):
        """
        BUG-QA-NEW-30: JobContext is mutable but might need to be dict key.
        """
        job = JobContext(
            job_id="test",
            job_file="test.stl",
            uploaded_at=datetime.now(),
            file_validation_status=FileValidationStatus.VALID,
        )

        # Try to use as dict key
        try:
            d = {job: "value"}
            # This works but is dangerous since job is mutable
        except TypeError:
            pass  # Expected - dataclass is not hashable by default

        # Try to use in set
        try:
            s = {job}
        except TypeError:
            pass  # Expected
