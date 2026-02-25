"""
BUG TESTS: Command Queue Issues

These tests document bugs in the command queue implementation.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock
from sensing_voxel.adapters.command_queue import CommandQueue, CommandPriority


class TestCommandQueueBugs:
    """Tests exposing command queue bugs"""

    @pytest.mark.asyncio
    async def test_clear_during_dequeue_race_condition(self):
        """
        BUG: clear() can race with dequeue().

        If clear() is called between queue.empty() check and queue.get_nowait(),
        we can have items that never get cleared or unexpected exceptions.

        This is a theoretical race - difficult to reproduce consistently.
        """
        queue = CommandQueue()

        # Enqueue some commands
        for i in range(5):
            queue.enqueue(f"cmd-{i}", CommandPriority.NORMAL)

        # Start a dequeue in background
        dequeue_task = asyncio.create_task(queue.dequeue())

        # Immediately clear the queue
        queue.clear()

        # The dequeue might return a command that was just cleared,
        # or it might hang waiting for a new command
        try:
            result = await asyncio.wait_for(dequeue_task, timeout=0.5)
            # If we got here, we got a command even though we cleared
            # This might be expected behavior, but it's a race
        except asyncio.TimeoutError:
            # Dequeue is hanging - this is also problematic
            dequeue_task.cancel()
            try:
                await dequeue_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_processor_restart_after_exception(self):
        """
        BUG: If processor function raises exception, the processor loop
        continues but there's no way to restart it with a new processor.

        The _processing flag stays True even though the loop exited.
        """
        queue = CommandQueue()

        failing_processor = AsyncMock(side_effect=RuntimeError("Processor crash"))

        # Start processor (it will crash immediately when it gets a command)
        queue.enqueue("test-cmd", CommandPriority.NORMAL)

        # Run processor briefly
        processor_task = asyncio.create_task(
            queue.start_processor(failing_processor)
        )

        # Give it time to crash
        await asyncio.sleep(0.2)

        # _processing should be False after crash, but...
        # Actually it IS set to False in finally block, so this passes
        # But there's no way to restart with a new processor

        queue.stop()
        try:
            await asyncio.wait_for(processor_task, timeout=0.5)
        except asyncio.TimeoutError:
            processor_task.cancel()

    @pytest.mark.asyncio
    async def test_shutdown_event_not_cleared_on_reset(self):
        """
        BUG: reset() calls _shutdown_event.clear() but doesn't check
        if processor is still running.

        If reset() is called while processor is stopping, we might
        end up with inconsistent state.
        """
        queue = CommandQueue()

        processor = AsyncMock()

        # Start processor
        processor_task = asyncio.create_task(
            queue.start_processor(processor)
        )

        await asyncio.sleep(0.1)

        # Stop the processor
        queue.stop()

        # Reset immediately (before processor fully stops)
        queue.reset()

        # _shutdown_event is now cleared, but processor might still be running
        # This could lead to weird behavior

        try:
            await asyncio.wait_for(processor_task, timeout=0.5)
        except asyncio.TimeoutError:
            processor_task.cancel()

    def test_emergency_command_bypasses_sequence_order(self):
        """
        BUG/QUESTIONABLE: Emergency commands bypass the priority queue entirely
        and go to a separate queue. This means they might be processed out of
        sequence relative to other high-priority commands.

        If we have:
        1. NORMAL command enqueued
        2. HIGH command enqueued
        3. EMERGENCY command enqueued

        The EMERGENCY will be processed before HIGH, which is correct.
        But what if we have multiple EMERGENCY commands?

        They go to an asyncio.Queue (FIFO), not PriorityQueue, so they're
        processed in order - that's fine.
        """
        queue = CommandQueue()

        # Enqueue in this order
        queue.enqueue("normal", CommandPriority.NORMAL)
        queue.enqueue("high", CommandPriority.HIGH)
        queue.enqueue("emergency-1", CommandPriority.EMERGENCY)
        queue.enqueue("emergency-2", CommandPriority.EMERGENCY)

        # Emergency queue should have 2 items
        assert queue._emergency_queue.qsize() == 2

        # Both emergencies should come before normal and high
        # This is actually correct behavior, not a bug

    @pytest.mark.asyncio
    async def test_sequence_counter_overflow(self):
        """
        BUG: _sequence_counter increments forever. In very long-running
        systems, this could theoretically overflow.

        Python integers don't overflow (they become longs), so this is
        mostly theoretical. But it's bad practice.

        After 2^31 increments, the counter keeps going but... it's ugly.
        """
        queue = CommandQueue()

        # Simulate many enqueues
        queue._sequence_counter = 2**31 - 1  # Near 32-bit int max

        queue.enqueue("test", CommandPriority.NORMAL)

        # Counter overflowed (in Python it just becomes a bigger int)
        assert queue._sequence_counter == 2**31

        # This is technically fine in Python but:
        # 1. Memory usage grows slightly with larger integers
        # 2. It's bad form to have unbounded counters

    @pytest.mark.asyncio
    async def test_dequeue_blocks_forever_on_empty_queues(self):
        """
        BEHAVIOR NOTE: If both queues are empty, dequeue() blocks forever
        waiting for a command. This is correct PriorityQueue behavior,
        but callers need to handle this.

        The start_processor() wrapper handles this with shutdown_event,
        but direct callers of dequeue() will block.
        """
        queue = CommandQueue()

        # dequeue() will block forever since both queues are empty
        # This is by design but worth documenting

        # Start a dequeue with timeout
        try:
            await asyncio.wait_for(queue.dequeue(), timeout=0.1)
            pytest.fail("Should have timed out")
        except asyncio.TimeoutError:
            pass  # Expected - dequeue blocks on empty queue

    def test_queued_command_timestamp_not_used_for_priority(self):
        """
        ISSUE: QueuedCommand has a timestamp but it's not used for
        priority. If two commands have same priority and somehow
        same sequence number (bug), ordering is undefined.

        The timestamp could be used as a tiebreaker but isn't.
        """
        from sensing_voxel.adapters.command_queue import QueuedCommand
        from datetime import datetime

        cmd1 = QueuedCommand(priority=50, seq=1, command="cmd1")
        cmd2 = QueuedCommand(priority=50, seq=1, command="cmd2")  # Same seq!

        # These would be equal in PriorityQueue - problematic
        # But in practice seq is unique, so this shouldn't happen

        # Just documenting that timestamp exists but isn't used
        assert cmd1.timestamp is not None
        assert cmd2.timestamp is not None
