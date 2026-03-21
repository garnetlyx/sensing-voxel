import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from sensing_voxel.adapters.command_queue import CommandPriority, CommandQueue


class TestCommandQueue:
    def make_queue(self) -> CommandQueue:
        return CommandQueue()

    def test_initialization(self):
        queue = self.make_queue()
        assert queue.size() == 0
        assert queue._sequence_counter == 0
        assert queue._processing is False
        assert queue._shutdown_event.is_set() is False

    @pytest.mark.asyncio
    async def test_enqueue_with_priorities_and_size(self):
        queue = self.make_queue()
        queue.enqueue("normal", CommandPriority.NORMAL)
        queue.enqueue("high", CommandPriority.HIGH)
        queue.enqueue("low", CommandPriority.LOW)
        queue.enqueue("emergency", CommandPriority.EMERGENCY)

        assert queue.size() == 4
        assert queue._sequence_counter == 4
        assert queue._queue.qsize() == 3
        assert queue._emergency_queue.qsize() == 1

        assert await queue.dequeue() == "emergency"
        assert await queue.dequeue() == "high"
        assert await queue.dequeue() == "normal"
        assert await queue.dequeue() == "low"

    @pytest.mark.asyncio
    async def test_fifo_within_same_priority(self):
        queue = self.make_queue()
        queue.enqueue("first", CommandPriority.NORMAL)
        queue.enqueue("second", CommandPriority.NORMAL)
        queue.enqueue("third", CommandPriority.NORMAL)

        assert await queue.dequeue() == "first"
        assert await queue.dequeue() == "second"
        assert await queue.dequeue() == "third"

    @pytest.mark.asyncio
    async def test_emergency_bypass_mechanism(self):
        queue = self.make_queue()
        queue.enqueue("normal-1", CommandPriority.NORMAL)
        queue.enqueue("normal-2", CommandPriority.NORMAL)
        queue.enqueue("emergency", CommandPriority.EMERGENCY)

        assert await queue.dequeue() == "emergency"
        assert await queue.dequeue() == "normal-1"
        assert await queue.dequeue() == "normal-2"

    @pytest.mark.asyncio
    async def test_dequeue_empty_blocks(self):
        queue = self.make_queue()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(queue.dequeue(), timeout=0.05)

    def test_clear_and_reset(self):
        queue = self.make_queue()
        queue.enqueue("normal", CommandPriority.NORMAL)
        queue.enqueue("emergency", CommandPriority.EMERGENCY)
        assert queue.size() == 2

        queue.clear()
        assert queue.size() == 0
        assert queue._sequence_counter == 2

        queue.stop()
        assert queue._shutdown_event.is_set() is True

        queue.reset()
        assert queue.size() == 0
        assert queue._sequence_counter == 0
        assert queue._shutdown_event.is_set() is False

    def test_clear_handles_queueempty_race(self):
        queue = self.make_queue()
        queue.enqueue("normal", CommandPriority.NORMAL)
        queue.enqueue("emergency", CommandPriority.EMERGENCY)

        queue._queue.empty = lambda: False  # type: ignore[method-assign]
        queue._emergency_queue.empty = lambda: False  # type: ignore[method-assign]
        queue._queue.get_nowait = Mock(side_effect=asyncio.QueueEmpty)  # type: ignore[method-assign]
        queue._emergency_queue.get_nowait = Mock(side_effect=asyncio.QueueEmpty)  # type: ignore[method-assign]

        queue.clear()

    @pytest.mark.asyncio
    async def test_start_stop_graceful_shutdown(self):
        queue = self.make_queue()
        processed: list[str] = []

        async def processor(command: str):
            processed.append(command)
            queue.stop()

        queue.enqueue("cmd", CommandPriority.NORMAL)
        task = asyncio.create_task(queue.start_processor(processor))
        await asyncio.wait_for(task, timeout=1.0)

        assert processed == ["cmd"]
        assert queue._processing is False

    @pytest.mark.asyncio
    async def test_start_processor_ignores_second_start(self, caplog):
        queue = self.make_queue()
        first_processor = AsyncMock()
        second_processor = AsyncMock()

        async def delayed_stop_processor(command: str):
            await first_processor(command)
            await asyncio.sleep(0.05)
            queue.stop()

        queue.enqueue("cmd", CommandPriority.NORMAL)
        first_task = asyncio.create_task(queue.start_processor(delayed_stop_processor))

        await asyncio.sleep(0)
        with caplog.at_level("WARNING"):
            await queue.start_processor(second_processor)

        await asyncio.wait_for(first_task, timeout=1.0)

        second_processor.assert_not_called()
        assert "already running" in caplog.text

    @pytest.mark.asyncio
    async def test_processor_continues_after_processor_exception(self):
        queue = self.make_queue()
        seen: list[str] = []
        fail_once = {"value": True}

        async def processor(command: str):
            seen.append(command)
            if fail_once["value"]:
                fail_once["value"] = False
                raise RuntimeError("boom")
            queue.stop()

        queue.enqueue("first", CommandPriority.NORMAL)
        queue.enqueue("second", CommandPriority.NORMAL)
        task = asyncio.create_task(queue.start_processor(processor))
        await asyncio.wait_for(task, timeout=1.0)

        assert seen == ["first", "second"]

    @pytest.mark.asyncio
    async def test_processor_handles_dequeue_timeout_when_empty(self):
        queue = self.make_queue()
        processor = AsyncMock()

        original_wait_for = asyncio.wait_for
        timed_out_once = {"value": False}

        async def fake_wait_for(awaitable, timeout):
            if not timed_out_once["value"]:
                timed_out_once["value"] = True
                queue.stop()
                awaitable.close()
                raise TimeoutError
            return await original_wait_for(awaitable, timeout)

        with patch(
            "sensing_voxel.adapters.command_queue.asyncio.wait_for", side_effect=fake_wait_for
        ):
            processor_task = asyncio.create_task(queue.start_processor(processor))
            await original_wait_for(processor_task, timeout=1.0)

        assert timed_out_once["value"] is True
        processor.assert_not_called()

    @pytest.mark.asyncio
    async def test_outer_processor_loop_recovers_from_dequeue_error(self):
        queue = self.make_queue()

        async def broken_dequeue():
            raise RuntimeError("dequeue failed")

        queue.dequeue = broken_dequeue  # type: ignore[method-assign]
        processor = AsyncMock()

        async def fake_sleep(_: float):
            queue.stop()

        with patch(
            "sensing_voxel.adapters.command_queue.asyncio.sleep", side_effect=fake_sleep
        ) as sleep_mock:
            task = asyncio.create_task(queue.start_processor(processor))
            await asyncio.wait_for(task, timeout=1.0)

        assert sleep_mock.called
        processor.assert_not_called()

    @pytest.mark.asyncio
    async def test_concurrent_enqueue_and_processing(self):
        queue = self.make_queue()
        processed: list[int] = []
        lock = asyncio.Lock()

        async def processor(command: int):
            async with lock:
                processed.append(command)
                if len(processed) == 20:
                    queue.stop()

        async def producer(start: int):
            for i in range(start, start + 10):
                queue.enqueue(i, CommandPriority.NORMAL)
                await asyncio.sleep(0)

        producer_a = asyncio.create_task(producer(0))
        producer_b = asyncio.create_task(producer(10))
        consumer = asyncio.create_task(queue.start_processor(processor))

        await asyncio.wait_for(asyncio.gather(producer_a, producer_b, consumer), timeout=2.0)

        assert len(processed) == 20
        assert sorted(processed) == list(range(20))
