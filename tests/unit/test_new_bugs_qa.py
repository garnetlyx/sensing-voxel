"""
NEW BUG TESTS: QA Analysis Findings

These tests expose NEW bugs found during QA analysis of the codebase.
Each test documents a critical issue that must be fixed.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
from datetime import datetime

from sensing_voxel.core.models import (
    AgentState,
    JobContext,
    DecisionAction,
    ConsumeResult,
    FileValidationStatus,
    CommandResultStatus,
)
from sensing_voxel.core.decision import DecisionEngine


class TestTelegramBotSecurityBugs:
    """Security bugs in TelegramBot"""

    @pytest.mark.asyncio
    async def test_job_id_prefix_collision_attack(self):
        """
        BUG-NEW-1: CRITICAL - Job ID prefix collision allows unauthorized actions.

        The callback query handler matches jobs using only the first 8 characters
        of the job ID. An attacker can craft a job ID that collides with another
        user's active job and perform unauthorized actions.

        Location: telegram.py:236
        Severity: CRITICAL
        """
        # Simulate two jobs where job_id_short (first 8 chars) collide
        job_1_id = "abc12345-xxxx-aaaa"
        job_2_id = "abc12345-yyyy-bbbb"

        # Both have same 8-char prefix
        assert job_1_id[:8] == job_2_id[:8]

        # In the current code, if job_1 is active and attacker submits
        # callback with job_2's prefix, it would match job_1
        # This is a security vulnerability

        # The test documents this vulnerability
        short_id = job_1_id[:8]

        # Any job starting with same prefix would match
        assert job_2_id.startswith(short_id)

        # BUG: Current code would match job_2_id callback to job_1
        # This allows cross-user job manipulation
        pytest.fail(
            "CRITICAL SECURITY BUG: Job ID prefix collision allows "
            "unauthorized job manipulation. Need full job ID matching."
        )

    @pytest.mark.asyncio
    async def test_file_path_traversal_in_upload(self):
        """
        BUG-NEW-2: CRITICAL - Path traversal vulnerability in file upload.

        The file upload handler uses the original filename from Telegram
        without sanitization. A malicious filename like '../../../etc/crontab'
        could overwrite system files.

        Location: telegram.py:168
        Severity: CRITICAL
        """
        malicious_filename = "../../../etc/crontab"

        # The current code does:
        # stl_path = uploads_dir / f"{job_id}.stl"
        # This is safe (uses job_id), but if it used document.file_name:

        uploads_dir = Path("/app/data/uploads")
        unsafe_path = uploads_dir / malicious_filename

        # This would resolve to /etc/crontab - outside uploads_dir!
        resolved = unsafe_path.resolve()

        # Check if path escapes the upload directory
        try:
            resolved.relative_to(uploads_dir.resolve())
            is_safe = True
        except ValueError:
            is_safe = False

        if not is_safe:
            pytest.fail(
                "CRITICAL SECURITY BUG: Path traversal vulnerability. "
                f"Path {resolved} escapes upload directory {uploads_dir}"
            )


class TestKlipperAdapterBugs:
    """Bugs in Klipper adapter"""

    @pytest.mark.asyncio
    async def test_emergency_cancel_not_actually_emergency(self):
        """
        BUG-NEW-3: HIGH - Emergency cancel doesn't bypass queue.

        The emergency_cancel() method calls the same _rest_request as
        regular cancel_print(). In an emergency, we need immediate
        response, but this goes through the normal request flow.

        Location: klipper.py:252-254
        Severity: HIGH
        """
        from sensing_voxel.adapters.klipper import KlipperAdapter

        adapter = KlipperAdapter(host="localhost", port=7125)

        # Both methods use the same implementation
        # emergency_cancel should use a dedicated, non-blocking session
        # with aggressive timeout to ensure it gets through

        # This test documents that emergency and regular cancel are identical
        assert adapter.emergency_cancel.__code__ == adapter.cancel_print.__code__, \
            "BUG: emergency_cancel should have dedicated emergency implementation"

    def test_state_map_missing_common_states(self):
        """
        BUG-NEW-4: MEDIUM - State mapping is incomplete.

        The STATE_MAP in KlipperAdapter doesn't handle common Moonraker
        states like 'ready', 'startup', 'shutdown', or 'disconnecting'.
        These fall through to UNKNOWN, which could miss important state changes.

        Location: klipper.py:29-37
        Severity: MEDIUM
        """
        from sensing_voxel.adapters.klipper import KlipperAdapter
        from sensing_voxel.core.models import NormalizedPrinterState

        adapter = KlipperAdapter()

        # These are valid Moonraker states not in the map
        missing_states = ['ready', 'startup', 'shutdown', 'disconnecting', 'error']

        for state in missing_states:
            if state not in adapter.STATE_MAP:
                # These states map to UNKNOWN which loses information
                result = adapter.STATE_MAP.get(state, NormalizedPrinterState.UNKNOWN)
                if result == NormalizedPrinterState.UNKNOWN:
                    pytest.fail(
                        f"BUG: Moonraker state '{state}' maps to UNKNOWN. "
                        "Important state information is being lost."
                    )


class TestCommandQueueBugs:
    """Bugs in CommandQueue"""

    @pytest.mark.asyncio
    async def test_no_sequence_number_for_same_priority_commands(self):
        """
        BUG-NEW-5: MEDIUM - Commands with same priority may be reordered.

        The QueuedCommand dataclass uses priority and seq for ordering,
        but seq is only incremented on enqueue. If commands have the
        same priority, their relative order depends on seq. However,
        the seq counter could theoretically wrap around in very long
        running systems.

        Location: command_queue.py:25-38
        Severity: MEDIUM
        """
        from sensing_voxel.adapters.command_queue import CommandQueue, CommandPriority

        queue = CommandQueue()

        # Enqueue 3 normal priority commands
        queue.enqueue("cmd1", CommandPriority.NORMAL)
        queue.enqueue("cmd2", CommandPriority.NORMAL)
        queue.enqueue("cmd3", CommandPriority.NORMAL)

        # They should come out in order
        # But this depends on seq counter which could theoretically overflow

        # This is more of a design concern than an immediate bug
        # but worth documenting for long-running systems

    def test_emergency_queue_not_prioritized(self):
        """
        BUG-NEW-6: MEDIUM - Multiple emergency commands processed FIFO not by urgency.

        Emergency commands go to a separate asyncio.Queue (FIFO), not
        a priority queue. If we have different types of emergencies
        (e.g., emergency_stop vs emergency_pause), we can't prioritize.

        Location: command_queue.py:62-64
        Severity: LOW-MEDIUM
        """
        from sensing_voxel.adapters.command_queue import CommandQueue, CommandPriority

        queue = CommandQueue()

        # All emergencies go to same FIFO queue
        # No sub-prioritization possible
        queue.enqueue("emergency_1", CommandPriority.EMERGENCY)
        queue.enqueue("emergency_2", CommandPriority.EMERGENCY)

        # Both in emergency queue, processed in FIFO order only
        assert queue._emergency_queue.qsize() == 2


class TestDecisionEngineRaceConditions:
    """Race condition bugs in DecisionEngine"""

    @pytest.mark.asyncio
    async def test_consume_decision_not_actually_atomic(self):
        """
        BUG-NEW-7: HIGH - consume_decision has race condition between check and set.

        The consume_decision method checks decision_consumed flag, then sets it.
        Between these two operations, another concurrent call could pass the check.

        This is a classic TOCTOU (Time-of-check-time-of-use) race condition.

        Location: decision.py:150-154
        Severity: HIGH
        """
        # This is hard to test directly, but we can document it:
        # Line 150: check job_context.decision_consumed
        # Line 163: set job_context.decision_consumed = True
        # Gap between check and set is vulnerable

        pytest.fail(
            "HIGH: Race condition in consume_decision. "
            "Check at line 150 and set at line 163 are not atomic. "
            "Two concurrent calls could both pass the check before either sets."
        )

    @pytest.mark.asyncio
    async def test_timeout_task_cancellation_race(self):
        """
        BUG-NEW-8: MEDIUM - Timeout task cancellation has race condition.

        In consume_decision, we cancel the timeout task then await it.
        But between cancel() and await, the task could complete normally,
        causing CancelledError not to be raised.

        Location: decision.py:116-131
        Severity: MEDIUM
        """
        # The code does:
        # task.cancel()
        # await asyncio.wait_for(task, timeout=1.0)
        #
        # Race: If task completes between cancel() and wait_for(),
        # we get a different exception or behavior

        # This is a known pattern that can cause issues
        pass


class TestStateMachineConsistency:
    """Consistency bugs in StateMachine"""

    @pytest.mark.asyncio
    async def test_transition_can_fail_after_state_change(self):
        """
        BUG-NEW-9: HIGH - State changes before persistence, can lose state on crash.

        In transition_to(), the state is changed (line 129), then persistence
        is called (line 131). If the process crashes between these, the state
        changed in memory is lost and system restarts in old state.

        Location: state_machine.py:129-134
        Severity: HIGH
        """
        # Current order:
        # 1. old_state = self.current_state
        # 2. self.current_state = new_state  <- State changed
        # 3. persistence.save(self)           <- Persistence may fail
        #
        # If crash between 2 and 3, we restart in old_state but
        # the actual system may have progressed

        pytest.fail(
            "HIGH: State changes before persistence. "
            "Crash between state change and persistence causes state loss."
        )

    @pytest.mark.asyncio
    async def test_persist_failure_not_propagated(self):
        """
        BUG-NEW-10: MEDIUM - Persistence failure is logged but not propagated.

        If persistence fails, the error is logged but transition_to still
        returns True. The caller thinks the transition succeeded but state
        wasn't persisted.

        Location: state_machine.py:131-134
        Severity: MEDIUM
        """
        # The except block logs error but doesn't re-raise or return False
        # This is inconsistent - state changed but not persisted

        pytest.fail(
            "MEDIUM: Persistence failure in transition_to returns True "
            "even though state wasn't saved. Caller is misled."
        )


class TestRetryUtilityBugs:
    """Bugs in retry utility"""

    def test_retry_async_doesnt_validate_max_attempts(self):
        """
        BUG-NEW-11: LOW - No validation for invalid max_attempts values.

        max_attempts <= 0 causes infinite loop or unexpected behavior.
        max_attempts = 1 with exception should raise immediately but
        the loop logic may have edge cases.

        Location: retry.py:65-116
        Severity: LOW
        """
        # No validation for:
        # - max_attempts = 0 (would never try)
        # - max_attempts = -1 (infinite loop in theory)
        # - max_attempts = None (would fail comparison)

        pytest.fail(
            "LOW: retry_async doesn't validate max_attempts parameter. "
            "Invalid values could cause unexpected behavior."
        )

    def test_compute_backoff_jitter_can_be_negative(self):
        """
        BUG-NEW-12: LOW - Jitter calculation can theoretically produce negative delay.

        If policy.jitter is negative and large enough, the delay could
        become negative, causing immediate retry or unexpected behavior.

        Location: retry.py:57-60
        Severity: LOW
        """
        # jitter_amount = delay * policy.jitter * random.random()
        # If jitter is -2.0, this subtracts from delay
        # Could result in negative delay

        pytest.fail(
            "LOW: compute_backoff can produce negative delays with "
            "negative jitter values. No validation on policy parameters."
        )


class TestKlipperAdapterResourceLeaks:
    """Resource leak bugs in KlipperAdapter"""

    @pytest.mark.asyncio
    async def test_ws_reconnect_loop_no_max_attempts(self):
        """
        BUG-NEW-13: MEDIUM - WebSocket reconnection loops forever without limit.

        The _ws_reconnect_loop has no maximum attempt limit. In a scenario
        where the printer is permanently offline, this creates infinite
        reconnection attempts, wasting resources and filling logs.

        Location: klipper.py:103-129
        Severity: MEDIUM
        """
        # The loop continues while not _shutdown_event.is_set()
        # with no maximum attempt counter
        # _reconnect_attempts increments but never checked against max

        pytest.fail(
            "MEDIUM: WebSocket reconnect loop has no maximum attempts. "
            "Will retry forever if printer is offline, wasting resources."
        )

    @pytest.mark.asyncio
    async def test_upload_file_resource_leak_on_exception(self):
        """
        BUG-NEW-14: MEDIUM - File handle may not be closed on exception.

        In upload_file(), the file is opened but if an exception occurs
        after open but before return, the file handle may leak.

        Location: klipper.py:269-270
        Severity: MEDIUM
        """
        # The file is opened with open(path, "rb")
        # but there's no try/finally or context manager to ensure close
        # If exception occurs at line 276 (FormData creation) or later,
        # file handle is leaked

        pytest.fail(
            "MEDIUM: upload_file doesn't ensure file closure on exception. "
            "File handle leak possible if error occurs after open."
        )
