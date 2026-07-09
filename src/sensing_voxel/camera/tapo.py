"""Tapo camera PTZ controller via ONVIF"""

from __future__ import annotations

import asyncio
import logging

from sensing_voxel.config import TapoCameraConfig

logger = logging.getLogger(__name__)

VALID_DIRECTIONS = ("left", "right", "up", "down")

# Velocity for step movements (0.0 to 1.0)
STEP_VELOCITY = 0.5
STEP_DURATION = 0.3  # seconds


class TapoCamera:
    """Async ONVIF-based PTZ control for Tapo cameras."""

    def __init__(self, config: TapoCameraConfig):
        self._config = config
        self._cam = None
        self._ptz = None
        self._profile_token = None

    async def connect(self) -> None:
        """Initialize ONVIF connection."""
        from onvif import ONVIFCamera

        port = self._config.onvif_port or 2020
        self._cam = await asyncio.to_thread(
            ONVIFCamera,
            self._config.host,
            port,
            self._config.user,
            self._config.password,
        )
        media = await asyncio.to_thread(self._cam.create_media_service)
        self._ptz = await asyncio.to_thread(self._cam.create_ptz_service)
        profiles = await asyncio.to_thread(media.GetProfiles)
        self._profile_token = profiles[0].token
        logger.info(f"Tapo camera connected via ONVIF: {self._config.host}:{port}")

    async def disconnect(self) -> None:
        """Cleanup."""
        self._cam = None
        self._ptz = None
        self._profile_token = None

    async def move_step(self, direction: str) -> dict:
        """Move camera one step in a direction (left/right/up/down)."""
        if direction not in VALID_DIRECTIONS:
            return {"error": f"Invalid direction: {direction}. Use: {', '.join(VALID_DIRECTIONS)}"}
        if not self._ptz:
            return {"error": "Camera not connected"}

        velocity = {"PanTilt": {"x": 0.0, "y": 0.0}}
        if direction == "left":
            velocity["PanTilt"]["x"] = -STEP_VELOCITY
        elif direction == "right":
            velocity["PanTilt"]["x"] = STEP_VELOCITY
        elif direction == "up":
            velocity["PanTilt"]["y"] = STEP_VELOCITY
        elif direction == "down":
            velocity["PanTilt"]["y"] = -STEP_VELOCITY

        request = {
            "ProfileToken": self._profile_token,
            "Velocity": velocity,
        }
        await asyncio.to_thread(self._ptz.ContinuousMove, request)
        await asyncio.sleep(STEP_DURATION)
        await asyncio.to_thread(self._ptz.Stop, {"ProfileToken": self._profile_token})
        return {"status": "ok", "direction": direction}

    async def move_to(self, pan: float, tilt: float) -> dict:
        """Move camera to absolute pan/tilt position (0.0-1.0 range)."""
        if not self._ptz:
            return {"error": "Camera not connected"}

        request = {
            "ProfileToken": self._profile_token,
            "Position": {"PanTilt": {"x": pan, "y": tilt}},
            "Speed": {"PanTilt": {"x": 1.0, "y": 1.0}},
        }
        await asyncio.to_thread(self._ptz.AbsoluteMove, request)
        return {"status": "ok", "pan": pan, "tilt": tilt}

    async def get_presets(self) -> dict:
        """Get saved camera presets."""
        if not self._ptz:
            return {"error": "Camera not connected"}
        presets = await asyncio.to_thread(
            self._ptz.GetPresets, {"ProfileToken": self._profile_token}
        )
        result = [{"token": p.token, "name": getattr(p, "Name", None)} for p in presets]
        return {"presets": result}

    async def set_preset(self, preset_id: int) -> dict:
        """Move camera to a saved preset position."""
        if not self._ptz:
            return {"error": "Camera not connected"}
        await asyncio.to_thread(
            self._ptz.GotoPreset,
            {"ProfileToken": self._profile_token, "PresetToken": str(preset_id)},
        )
        return {"status": "ok", "preset_id": preset_id}

    async def get_position(self) -> dict | None:
        """Get current PTZ position when supported by the ONVIF backend."""
        if not self._ptz:
            return None
        try:
            status = await asyncio.to_thread(
                self._ptz.GetStatus,
                {"ProfileToken": self._profile_token},
            )
        except Exception:
            return None

        position = getattr(status, "Position", None)
        pan_tilt = getattr(position, "PanTilt", None)
        if pan_tilt is None:
            return None
        return {
            "pan": getattr(pan_tilt, "x", None),
            "tilt": getattr(pan_tilt, "y", None),
        }
