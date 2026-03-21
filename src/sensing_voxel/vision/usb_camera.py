"""USB camera capture for filament spool monitoring."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime

import cv2

from sensing_voxel.config import FilamentMonitorConfig
from sensing_voxel.core.models import VisionFrame

logger = logging.getLogger(__name__)


class USBCameraStream:
    """USB camera stream for filament spool monitoring."""

    def __init__(
        self,
        config: FilamentMonitorConfig,
        max_queue_size: int = 5,
    ):
        self.config = config
        self._cap: cv2.VideoCapture | None = None
        self._running = False
        self._capture_task: asyncio.Task | None = None
        self._frame_queue: deque[VisionFrame] = deque(maxlen=max_queue_size)
        self._frame_counter = 0
        self._new_frame_event = asyncio.Event()

    async def start(self) -> bool:
        """Start USB camera capture. Returns True if successful."""
        if self._running:
            return True

        if not self._connect():
            return False

        self._running = True
        self._capture_task = asyncio.create_task(self._capture_loop())
        logger.info(f"USB camera stream started (index={self.config.usb_camera_index})")
        return True

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
        logger.info("USB camera stream stopped")

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
        """Open USB camera."""
        try:
            self._cap = cv2.VideoCapture(self.config.usb_camera_index)
            if not self._cap.isOpened():
                logger.warning(f"Failed to open USB camera (index={self.config.usb_camera_index})")
                return False

            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
            self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)

            actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"USB camera connected: {actual_width}x{actual_height}")
            return True
        except Exception as e:
            logger.error(f"USB camera connection error: {e}")
            return False

    def _release(self):
        """Release OpenCV capture."""
        if self._cap:
            self._cap.release()
            self._cap = None

    async def _capture_loop(self):
        """Background capture loop."""
        target_interval = 1.0 / max(1, self.config.fps)

        while self._running:
            if not self.is_connected:
                await asyncio.sleep(1.0)
                self._connect()
                continue

            try:
                loop = asyncio.get_event_loop()
                ret, raw_frame = await loop.run_in_executor(None, self._read_frame)

                if not ret or raw_frame is None:
                    await asyncio.sleep(0.1)
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
                logger.error(f"USB capture error: {e}")
                await asyncio.sleep(1.0)

    def _read_frame(self) -> tuple[bool, any]:
        """Read a single frame (called in executor)."""
        if self._cap is None:
            return False, None
        return self._cap.read()
