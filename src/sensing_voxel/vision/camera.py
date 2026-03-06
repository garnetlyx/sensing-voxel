"""
Camera stream capture via RTSP/OpenCV with reconnection.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import deque
from datetime import datetime

import cv2

from sensing_voxel.config import CameraConfig
from sensing_voxel.core.models import VisionFrame

logger = logging.getLogger(__name__)


class CameraStream:
    """Async RTSP camera stream with background capture and reconnection."""

    def __init__(
        self,
        config: CameraConfig,
        max_queue_size: int = 10,
        reconnect_base_delay: float = 1.0,
        reconnect_max_delay: float = 30.0,
        reconnect_factor: float = 1.8,
    ):
        self.config = config
        self._cap: cv2.VideoCapture | None = None
        self._running = False
        self._capture_task: asyncio.Task | None = None
        self._frame_queue: deque[VisionFrame] = deque(maxlen=max_queue_size)
        self._frame_counter = 0
        self._new_frame_event = asyncio.Event()

        # Reconnection params
        self._reconnect_base_delay = reconnect_base_delay
        self._reconnect_max_delay = reconnect_max_delay
        self._reconnect_factor = reconnect_factor

    async def start(self):
        """Start background capture loop."""
        if self._running:
            return
        self._running = True
        self._capture_task = asyncio.create_task(self._capture_loop())
        logger.info(f"Camera stream started: {self.config.rtsp_url}")

    async def stop(self):
        """Stop capture and release resources."""
        self._running = False
        if self._capture_task:
            self._capture_task.cancel()
            try:
                await self._capture_task
            except asyncio.CancelledError:
                pass
            self._capture_task = None
        self._release()
        logger.info("Camera stream stopped")

    async def get_frame(self, timeout: float = 5.0) -> VisionFrame | None:
        """Get the most recent frame, waiting up to timeout seconds."""
        if self._frame_queue:
            return self._frame_queue[-1]

        self._new_frame_event.clear()
        try:
            await asyncio.wait_for(self._new_frame_event.wait(), timeout=timeout)
            return self._frame_queue[-1] if self._frame_queue else None
        except asyncio.TimeoutError:
            return None

    @property
    def is_connected(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def _connect(self) -> bool:
        """Open RTSP stream."""
        try:
            self._cap = cv2.VideoCapture(self.config.rtsp_url)
            if self._cap.isOpened():
                logger.info("Camera connected")
                return True
            logger.warning("Camera failed to open")
            return False
        except Exception as e:
            logger.error(f"Camera connection error: {e}")
            return False

    def _release(self):
        """Release OpenCV capture."""
        if self._cap:
            self._cap.release()
            self._cap = None

    async def _capture_loop(self):
        """Background loop: grab frames, reconnect on failure."""
        delay = self._reconnect_base_delay
        target_interval = 1.0 / max(1, self.config.fps)

        while self._running:
            if not self.is_connected:
                if not self._connect():
                    jitter = random.random() * delay * 0.25
                    await asyncio.sleep(delay + jitter)
                    delay = min(delay * self._reconnect_factor, self._reconnect_max_delay)
                    continue
                delay = self._reconnect_base_delay

            try:
                loop = asyncio.get_event_loop()
                ret, raw_frame = await loop.run_in_executor(None, self._read_frame)

                if not ret or raw_frame is None:
                    logger.warning("Frame read failed, reconnecting...")
                    self._release()
                    continue

                self._frame_counter += 1
                h, w = raw_frame.shape[:2]
                frame = VisionFrame(
                    frame_data=raw_frame,
                    timestamp=datetime.now(),
                    frame_id=self._frame_counter,
                    width=w,
                    height=h,
                )
                self._frame_queue.append(frame)
                self._new_frame_event.set()

                await asyncio.sleep(target_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Capture error: {e}", exc_info=True)
                self._release()
                await asyncio.sleep(1.0)

    def _read_frame(self) -> tuple[bool, any]:
        """Read a single frame (called in executor)."""
        if self._cap is None:
            return False, None
        return self._cap.read()
