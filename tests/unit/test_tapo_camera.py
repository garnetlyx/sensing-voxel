"""Tests for Tapo camera PTZ controller"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sensing_voxel.camera.tapo import TapoCamera, VALID_DIRECTIONS
from sensing_voxel.config import TapoCameraConfig


@pytest.fixture
def config():
    return TapoCameraConfig(host="192.168.1.100", user="admin", password="pass")


@pytest.fixture
def camera(config):
    cam = TapoCamera(config)
    # Mock the internal pytapo instance
    cam._tapo = MagicMock()
    return cam


class TestTapoCameraConnect:
    async def test_connect_creates_tapo_instance(self, config):
        cam = TapoCamera(config)
        mock_tapo = MagicMock()
        with patch("sensing_voxel.camera.tapo.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = AsyncMock(return_value=mock_tapo)
            await cam.connect()
            assert cam._tapo is mock_tapo

    async def test_disconnect_clears_tapo(self, camera):
        assert camera._tapo is not None
        await camera.disconnect()
        assert camera._tapo is None


class TestTapoCameraMoveStep:
    async def test_valid_directions(self, camera):
        for direction in VALID_DIRECTIONS:
            result = await camera.move_step(direction)
            assert result["status"] == "ok"
            assert result["direction"] == direction

    async def test_invalid_direction(self, camera):
        result = await camera.move_step("diagonal")
        assert "error" in result

    async def test_not_connected(self, config):
        cam = TapoCamera(config)
        result = await cam.move_step("left")
        assert "error" in result
        assert "not connected" in result["error"]


class TestTapoCameraMoveTo:
    async def test_move_to(self, camera):
        result = await camera.move_to(90, -45)
        assert result["status"] == "ok"
        assert result["pan"] == 90
        assert result["tilt"] == -45

    async def test_not_connected(self, config):
        cam = TapoCamera(config)
        result = await cam.move_to(0, 0)
        assert "error" in result


class TestTapoCameraPresets:
    async def test_get_presets(self, camera):
        camera._tapo.getPresets = MagicMock(return_value={"1": "Home", "2": "Bed"})
        result = await camera.get_presets()
        assert "presets" in result

    async def test_set_preset(self, camera):
        result = await camera.set_preset(1)
        assert result["status"] == "ok"
        assert result["preset_id"] == 1

    async def test_get_presets_not_connected(self, config):
        cam = TapoCamera(config)
        result = await cam.get_presets()
        assert "error" in result

    async def test_set_preset_not_connected(self, config):
        cam = TapoCamera(config)
        result = await cam.set_preset(1)
        assert "error" in result
