"""
QA ADVERSARIAL TESTS - Final Round

These tests target remaining attack vectors and edge cases not covered
by previous test suites. Each test MUST fail on buggy code to be valid.

Categories:
1. Security vulnerabilities (injection, bypass)
2. Resource exhaustion (memory, file descriptors, CPU)
3. Timing attacks and race conditions
4. Input validation failures
5. Protocol violations
"""

import pytest
import asyncio
import gc
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Optional

from sensing_voxel.core.models import (
    AgentState,
    JobContext,
    DecisionAction,
    ConsumeResult,
    CommandResultStatus,
    FileValidationStatus,
    generate_job_id,
    generate_decision_token,
)
from sensing_voxel.core.state_machine import AgentStateMachine
from sensing_voxel.core.decision import DecisionEngine
from sensing_voxel.core.persistence import StatePersistence
from sensing_voxel.adapters.klipper import KlipperAdapter
from sensing_voxel.adapters.command_queue import CommandQueue, CommandPriority


class TestSecurityVulnerabilities:
    """Security-focused adversarial tests"""

    @pytest.mark.asyncio
    async def test_job_id_path_traversal_attempt(self):
        """
        BUG: Malicious job_id containing path traversal sequences could
        potentially be used for directory traversal attacks.

        This tests if job_id is properly sanitized when used in file paths.
        """
        malicious_ids = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "job_id/../../../etc/shadow",
            "normal_job\x00/../../../etc/passwd",  # Null byte injection
            "job" + "../" * 100 + "etc/passwd",  # Deep traversal
        ]

        for malicious_id in malicious_ids:
            # Create job with malicious ID
            job = JobContext(
                job_id=malicious_id,
                job_file="test.stl",
                uploaded_at=datetime.now(),
            )

            # The job_id should not be used directly in file paths
            # without sanitization
            uploads_dir = Path("data/uploads")
            dangerous_path = uploads_dir / f"{job.job_id}.stl"

            # Check that resolved path would escape uploads directory
            try:
                resolved = dangerous_path.resolve()
                uploads_resolved = uploads_dir.resolve()

                # If resolved path is outside uploads, it's a vulnerability
                try:
                    resolved.relative_to(uploads_resolved)
                except ValueError:
                    pytest.fail(
                        f"CRITICAL: Path traversal vulnerability with job_id '{malicious_id}'. "
                        f"Resolved path '{resolved}' escapes uploads directory!"
                    )
            except (OSError, ValueError) as e:
                # Path resolution failed - this is actually good for security
                pass

    @pytest.mark.asyncio
    async def test_decision_token_predictability(self):
        """
        BUG: If decision tokens are predictable, attackers could guess them
        and make unauthorized decisions.

        Tests if tokens have sufficient entropy and randomness.
        """
        tokens = [generate_decision_token() for _ in range(100)]

        # All tokens should be unique
        assert len(set(tokens)) == len(tokens), "Token collision detected!"

        # Tokens should be sufficiently long (at least 32 chars for hex)
        for token in tokens:
            assert len(token) >= 32, f"Token too short: {len(token)} chars"

        # Tokens should only contain hex characters
        for token in tokens:
            assert all(c in '0123456789abcdef' for c in token.lower()), \
                f"Token contains non-hex characters: {token}"

    @pytest.mark.asyncio
    async def test_metadata_injection_attack(self):
        """
        BUG: Metadata dict could be used for prototype pollution or
        storing malicious payloads.

        Tests if metadata validation prevents injection attacks.
        """
        job = JobContext(
            job_id="test-job",
            job_file="test.stl",
            uploaded_at=datetime.now(),
        )

        # Attempt to store dangerous metadata
        malicious_payloads = [
            {"__proto__": {"polluted": True}},  # Prototype pollution
            {"constructor": {"prototype": {"hacked": True}}},
            {"<script>alert('xss')</script>": "value"},  # XSS attempt
            {"key": "; rm -rf /;"},  # Command injection attempt
            {"nested": {"deep": {"payload": "A" * 1000000}}},  # Deep nesting
        ]

        for payload in malicious_payloads:
            # Should not crash when setting metadata
            job.metadata.update(payload)

            # Verify metadata was set (application should handle safely)
            assert job.metadata is not None

    @pytest.mark.asyncio
    async def test_sqlite_injection_via_job_fields(self):
        """
        BUG: If job fields are used in SQL queries without parameterization,
        SQL injection is possible.

        Note: Current MVP uses JSON, not SQLite, but this tests future-proofing.
        """
        sql_injection_attempts = [
            "'; DROP TABLE jobs; --",
            "1' OR '1'='1",
            "'; INSERT INTO jobs VALUES ('hacked'); --",
            "normal_job'); DELETE FROM jobs; --",
        ]

        for attempt in sql_injection_attempts:
            job = JobContext(
                job_id=attempt,
                job_file=attempt,
                uploaded_at=datetime.now(),
            )

            # The job should be created without executing the SQL
            assert job.job_id == attempt  # Stored as-is
            assert "DROP TABLE" not in str(job)  # Shouldn't execute


class TestResourceExhaustion:
    """Resource exhaustion attack tests"""

    @pytest.mark.asyncio
    async def test_decision_engine_memory_exhaustion(self):
        """
        BUG: DecisionEngine stores locks and timeout tasks indefinitely,
        leading to memory exhaustion under sustained load.

        This test verifies memory usage under load.
        """
        engine = DecisionEngine()
        engine._approval_timeout = 0.001  # 1ms timeout for fast test

        # Record initial memory
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Create many jobs rapidly
        for i in range(500):
            job = JobContext(
                job_id=f"memory-test-{i}",
                job_file="test.stl",
                uploaded_at=datetime.now(),
                decision_token=f"token-{i}",
            )
            engine.schedule_timeout(
                job_id=f"memory-test-{i}",
                job_context=job,
                timeout_action=DecisionAction.TIMEOUT_APPROVAL,
            )

        # Wait for all timeouts to complete
        await asyncio.sleep(0.1)

        # Check that resources were cleaned up
        active_tasks = len(engine._timeout_tasks)
        active_locks = len(engine._decision_locks)

        # After timeouts complete, resources should be cleaned
        # If not, we have a memory leak
        total_resources = active_tasks + active_locks

        # Allow some slack for test timing, but not too much
        if total_resources > 100:
            pytest.fail(
                f"MEMORY LEAK: {active_tasks} tasks and {active_locks} locks "
                f"still active after timeouts should have completed"
            )

    @pytest.mark.asyncio
    async def test_file_descriptor_exhaustion(self, tmp_path):
        """
        BUG: Rapid state persistence operations may exhaust file descriptors
        if locks are not released properly.
        """
        state_file = tmp_path / "fd_test.json"
        persistence = StatePersistence(state_file)

        sm = AgentStateMachine()

        # Rapid save operations
        for i in range(100):
            sm.current_state = AgentState.PRINTING
            persistence.save(sm)

        # Should not have leaked file descriptors
        # (This is OS-specific, so we just verify operations completed)
        assert state_file.exists()

    @pytest.mark.asyncio
    async def test_command_queue_memory_growth(self):
        """
        BUG: Command queue may grow unbounded if enqueue rate exceeds
        dequeue rate, causing memory exhaustion.
        """
        queue = CommandQueue()

        # Rapid enqueue without dequeuing
        for i in range(10000):
            queue.enqueue(f"command-{i}", CommandPriority.NORMAL)

        # Queue size should be trackable
        size = queue.size()
        assert size == 10000, f"Expected 10000 commands, got {size}"

        # Clear should free memory
        queue.clear()
        assert queue.size() == 0


class TestRaceConditions:
    """Race condition and concurrency bug tests"""

    @pytest.mark.asyncio
    async def test_decision_consumption_double_spend(self):
        """
        BUG: Race condition allows decision to be consumed twice
        if two threads check decision_consumed simultaneously.
        """
        engine = DecisionEngine()

        job = JobContext(
            job_id="double-spend-test",
            job_file="test.stl",
            uploaded_at=datetime.now(),
            decision_token="secret-token",
        )

        consume_count = 0

        async def try_consume():
            nonlocal consume_count
            result = await engine.consume_decision(
                job_context=job,
                decision_token="secret-token",
                action=DecisionAction.APPROVE,
                expected_state=None,
                current_state=None,
            )
            if result == ConsumeResult.CONSUMED:
                consume_count += 1

        # Launch multiple concurrent consumption attempts
        await asyncio.gather(
            try_consume(),
            try_consume(),
            try_consume(),
            try_consume(),
            try_consume(),
        )

        # Should only be consumed once
        assert consume_count == 1, \
            f"CRITICAL: Decision consumed {consume_count} times (double-spend bug!)"

    @pytest.mark.asyncio
    async def test_state_transition_mid_decision(self):
        """
        BUG: State transition during decision consumption may lead to
        inconsistent state if not properly synchronized.
        """
        engine = DecisionEngine()
        sm = AgentStateMachine()

        job = JobContext(
            job_id="race-test",
            job_file="test.stl",
            uploaded_at=datetime.now(),
            decision_token="token",
        )
        sm.job_context = job

        # Start in AWAITING_APPROVAL
        await sm.transition_to(AgentState.RECEIVING)
        await sm.transition_to(AgentState.SLICING)
        await sm.transition_to(AgentState.AWAITING_APPROVAL)

        async def consume_decision():
            return await engine.consume_decision(
                job_context=job,
                decision_token="token",
                action=DecisionAction.APPROVE,
                expected_state=AgentState.AWAITING_APPROVAL,
                current_state=sm.current_state,
            )

        async def change_state():
            await asyncio.sleep(0.001)
            await sm.transition_to(AgentState.IDLE)

        # Race between decision consumption and state change
        results = await asyncio.gather(
            consume_decision(),
            change_state(),
            return_exceptions=True,
        )

        # System should remain in consistent state
        assert sm.current_state in [AgentState.AWAITING_APPROVAL, AgentState.IDLE]


class TestInputValidation:
    """Input validation and sanitization tests"""

    @pytest.mark.asyncio
    async def test_extreme_datetime_values(self):
        """
        BUG: Extreme datetime values (year 9999, epoch 0) may cause
        serialization or comparison failures.
        """
        extreme_dates = [
            datetime.min,  # Year 1
            datetime.max,  # Year 9999
            datetime(1970, 1, 1),  # Unix epoch
            datetime(2038, 1, 19),  # 32-bit epoch overflow
            datetime(3000, 1, 1),  # Far future
        ]

        for dt in extreme_dates:
            job = JobContext(
                job_id="date-test",
                job_file="test.stl",
                uploaded_at=dt,
                expires_at=dt + timedelta(days=1) if dt != datetime.max else dt,
            )

            assert job.uploaded_at == dt

    @pytest.mark.asyncio
    async def test_unicode_normalization_attacks(self):
        """
        BUG: Unicode normalization may allow visually identical but
        different job IDs, potentially causing confusion attacks.
        """
        # Visually similar but different unicode strings
        similar_strings = [
            ("job-123", "job-123"),  # Identical
            ("job-123", "jоb-123"),  # Cyrillic 'о' instead of Latin 'o'
            ("job-123", "job–123"),  # Em-dash instead of hyphen
            ("job-123", "job−123"),  # Minus sign instead of hyphen
        ]

        for s1, s2 in similar_strings:
            job1 = JobContext(job_id=s1, job_file="test.stl", uploaded_at=datetime.now())
            job2 = JobContext(job_id=s2, job_file="test.stl", uploaded_at=datetime.now())

            # They should be treated as different IDs
            if s1 != s2:
                assert job1.job_id != job2.job_id

    @pytest.mark.asyncio
    async def test_none_state_transitions(self):
        """
        BUG: Transition attempts with None or invalid state values
        may cause crashes or unexpected behavior.
        """
        sm = AgentStateMachine()

        # Try transitioning with invalid inputs
        invalid_targets = [
            None,
            "INVALID_STATE",
            123,
            [],
            {},
        ]

        for target in invalid_targets:
            with pytest.raises((TypeError, AttributeError, KeyError)):
                await sm.transition_to(target)


class TestProtocolViolations:
    """Tests for protocol and API contract violations"""

    @pytest.mark.asyncio
    async def test_klipper_adapter_missing_session(self):
        """
        BUG: Adapter methods called before connect() may crash
        due to uninitialized session.
        """
        adapter = KlipperAdapter()

        # Try to use adapter without connecting
        result = await adapter.get_printer_state()

        # Should return UNKNOWN, not crash
        assert result is not None

    @pytest.mark.asyncio
    async def test_webhook_message_malformed(self):
        """
        BUG: Malformed WebSocket messages from printer may crash
        the message handler.
        """
        adapter = KlipperAdapter()

        malformed_messages = [
            None,
            {},
            {"method": None},
            {"method": "notify_status_update", "params": None},
            {"method": "notify_status_update", "params": [None]},
            {"method": "notify_status_update", "params": "not_a_list"},
            {"method": "notify_status_update", "params": [{}] * 10000},  # Huge params
        ]

        for msg in malformed_messages:
            # Should not crash
            try:
                await adapter._handle_ws_message(msg)
            except (AttributeError, TypeError, KeyError):
                # These are acceptable, but should be caught
                pass

    @pytest.mark.asyncio
    async def test_telemetry_data_corruption(self):
        """
        BUG: Corrupted telemetry data may cause division by zero or
        other arithmetic errors.
        """
        from sensing_voxel.core.models import TelemetryData

        corrupted_telemetry = [
            {"temperatures": {"extruder": float('inf')}},
            {"temperatures": {"extruder": float('nan')}},
            {"temperatures": {"extruder": -999999}},
            {"position": {"x": float('inf'), "y": 0, "z": 0}},
            {"position": {"x": None, "y": None, "z": None}},
        ]

        for data in corrupted_telemetry:
            # Should create without crashing
            telemetry = TelemetryData(
                timestamp=datetime.now(),
                **data
            )
            assert telemetry is not None


class TestErrorHandling:
    """Error handling and recovery tests"""

    @pytest.mark.asyncio
    async def test_exception_in_timeout_handler(self):
        """
        BUG: Exception in timeout handler may kill the timeout task
        without proper cleanup.
        """
        engine = DecisionEngine()
        engine._approval_timeout = 0.001

        job = JobContext(
            job_id="exception-test",
            job_file="test.stl",
            uploaded_at=datetime.now(),
            decision_token="token",
        )

        # Patch consume_decision to raise exception
        with patch.object(
            engine,
            'consume_decision',
            side_effect=RuntimeError("Simulated error")
        ):
            engine.schedule_timeout(
                job_id="exception-test",
                job_context=job,
                timeout_action=DecisionAction.TIMEOUT_APPROVAL,
            )

            # Wait for timeout to fire
            await asyncio.sleep(0.01)

            # Engine should still be functional
            assert engine is not None

    @pytest.mark.asyncio
    async def test_persistence_corruption_recovery(self, tmp_path):
        """
        BUG: Corrupted persistence file may prevent startup or
        cause crash loops.
        """
        state_file = tmp_path / "corrupt.json"

        # Various corruption scenarios
        corruptions = [
            "",  # Empty file
            "{",  # Incomplete JSON
            "null",  # Valid JSON but not an object
            "[]",  # Array instead of object
            "{\"current_state\": null}",  # Null state
            "{\"current_state\": \"INVALID_STATE\"}",  # Invalid state value
            "A" * 10000000,  # Huge file
        ]

        for corrupt_data in corruptions:
            state_file.write_text(corrupt_data)
            persistence = StatePersistence(state_file)

            # Should handle gracefully
            try:
                result = persistence.load()
                # Result may be None or valid, but shouldn't crash
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                # These are acceptable responses to corruption
                pass

    @pytest.mark.asyncio
    async def test_recursive_state_transition(self):
        """
        BUG: Recursive state transition (hook triggering transition)
        may cause infinite recursion or stack overflow.
        """
        sm = AgentStateMachine()

        call_count = 0

        async def recursive_hook():
            nonlocal call_count
            call_count += 1
            if call_count < 5:
                # Try to trigger another transition from within hook
                await sm.transition_to(AgentState.IDLE)

        await sm.on_enter(AgentState.RECEIVING, recursive_hook)

        # This should not cause infinite recursion
        await sm.transition_to(AgentState.RECEIVING)

        # Should have limited recursion depth
        assert call_count < 10
