"""
Priority command queue for printer adapters

Ensures commands are executed in priority order with emergency bypass.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any

logger = logging.getLogger(__name__)


class CommandPriority(IntEnum):
    """Command priority levels"""

    EMERGENCY = 0  # Emergency cancel - bypasses queue
    HIGH = 10  # Critical operations (pause, resume)
    NORMAL = 50  # Standard operations (start print, upload)
    LOW = 100  # Background operations


@dataclass(order=True)
class QueuedCommand:
    """A command in the queue"""

    priority: int
    seq: int  # Sequence number for FIFO within same priority
    command: Any = field(compare=False)  # Command payload
    timestamp: datetime = field(default_factory=datetime.now, compare=False)


class CommandQueue:
    """Priority command queue with emergency bypass"""

    def __init__(self):
        self._queue: asyncio.PriorityQueue[QueuedCommand] = asyncio.PriorityQueue()
        self._emergency_queue: asyncio.Queue[Any] = asyncio.Queue()
        self._sequence_counter = 0
        self._processing = False
        self._shutdown_event = asyncio.Event()

    def enqueue(self, command: Any, priority: CommandPriority = CommandPriority.NORMAL):
        """Enqueue a command with given priority"""
        seq = self._sequence_counter
        self._sequence_counter += 1

        queued_cmd = QueuedCommand(priority=int(priority), seq=seq, command=command)

        # Emergency commands go to separate queue for immediate processing
        if priority == CommandPriority.EMERGENCY:
            self._emergency_queue.put_nowait(command)
            logger.debug("Enqueued emergency command")
        else:
            self._queue.put_nowait(queued_cmd)
            logger.debug(f"Enqueued command (priority={priority.name}, seq={seq})")

    async def dequeue(self) -> Any:
        """
        Dequeue next command.
        Emergency commands take precedence over normal queue.
        """
        # Check emergency queue first
        try:
            return self._emergency_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass

        # Otherwise get from priority queue
        queued_cmd = await self._queue.get()
        logger.debug(f"Dequeued command (priority={queued_cmd.priority}, seq={queued_cmd.seq})")
        return queued_cmd.command

    def size(self) -> int:
        """Get total number of commands in queue"""
        return self._queue.qsize() + self._emergency_queue.qsize()

    def clear(self):
        """Clear all commands from queue"""
        # Clear priority queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Clear emergency queue
        while not self._emergency_queue.empty():
            try:
                self._emergency_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        logger.info("Command queue cleared")

    async def start_processor(self, processor: callable):
        """
        Start background processor that continuously dequeues and processes commands.

        Args:
            processor: Async function that takes a command and processes it
        """
        if self._processing:
            logger.warning("Command queue processor already running")
            return

        self._processing = True
        logger.info("Command queue processor started")

        try:
            while not self._shutdown_event.is_set():
                try:
                    # Use wait_for with a short timeout to periodically check shutdown
                    try:
                        command = await asyncio.wait_for(self.dequeue(), timeout=0.5)
                    except TimeoutError:
                        # Timeout occurred - check shutdown again and loop
                        continue

                    # Process command
                    try:
                        await processor(command)
                    except Exception as e:
                        logger.error(f"Error processing command: {e}", exc_info=True)

                except Exception as e:
                    logger.error(f"Error in command queue processor: {e}", exc_info=True)
                    # Small delay to prevent tight error loop
                    await asyncio.sleep(0.1)

        finally:
            self._processing = False
            logger.info("Command queue processor stopped")

    def stop(self):
        """Stop the command queue processor"""
        self._shutdown_event.set()
        logger.info("Command queue shutdown initiated")

    def reset(self):
        """Reset queue state"""
        self._shutdown_event.clear()
        self.clear()
        self._sequence_counter = 0
        logger.info("Command queue reset")
