"""Tapo camera PTZ controller"""

from __future__ import annotations

import asyncio
import logging

from sensing_voxel.config import TapoCameraConfig

logger = logging.getLogger(__name__)

VALID_DIRECTIONS = ("left", "right", "up", "down")


class TapoCamera:
    """Async wrapper around pytapo for PTZ control."""

    def __init__(self, config: TapoCameraConfig):
        self._config = config
        self._tapo = None

    async def connect(self) -> None:
        """Initialize pytapo connection."""
        from pytapo import Tapo

        self._tapo = await asyncio.to_thread(
            Tapo, self._config.host, self._config.user, self._config.password
        )
        logger.info(f"Tapo camera connected: {self._config.host}")

    async def disconnect(self) -> None:
        """Cleanup."""
        self._tapo = None

    async def move_step(self, direction: str) -> dict:
        """Move camera one step in a direction (left/right/up/down)."""
        if direction not in VALID_DIRECTIONS:
            return {"error": f"Invalid direction: {direction}. Use: {', '.join(VALID_DIRECTIONS)}"}
        if not self._tapo:
            return {"error": "Camera not connected"}
        await asyncio.to_thread(self._tapo.moveMotorStep, direction)
        return {"status": "ok", "direction": direction}

    async def move_to(self, pan: int, tilt: int) -> dict:
        """Move camera to absolute pan/tilt position."""
        if not self._tapo:
            return {"error": "Camera not connected"}
        await asyncio.to_thread(self._tapo.moveMotor, pan, tilt)
        return {"status": "ok", "pan": pan, "tilt": tilt}

    async def get_presets(self) -> dict:
        """Get saved camera presets."""
        if not self._tapo:
            return {"error": "Camera not connected"}
        presets = await asyncio.to_thread(self._tapo.getPresets)
        return {"presets": presets}

    async def set_preset(self, preset_id: int) -> dict:
        """Move camera to a saved preset position."""
        if not self._tapo:
            return {"error": "Camera not connected"}
        await asyncio.to_thread(self._tapo.setPreset, preset_id)
        return {"status": "ok", "preset_id": preset_id}
