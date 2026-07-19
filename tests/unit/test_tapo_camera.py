"""Tests for Tapo camera PTZ controller"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sensing_voxel.camera.tapo import TapoCamera, VALID_DIRECTIONS
from sensing_voxel.config import TapoCameraConfig


@pytest.fixture
def config():
    return TapoCameraConfig(
        host="192.0.2.20", user="camera-user", password="test-camera-password"
    )


@pytest.fixture
def camera(config):
    cam = TapoCamera(config)
    # Mock the ONVIF PTZ service and profile token
    cam._ptz = MagicMock()
    cam._profile_token = "mock_token"
    return cam


class TestTapoCameraConnect:
    async def test_connect_creates_tapo_instance(self, config):
        cam = TapoCamera(config)
        mock_cam = MagicMock()
        mock_ptz = MagicMock()
        mock_media = MagicMock()
        mock_profiles = [MagicMock(token="test_token")]
        mock_onvif = MagicMock()
        mock_onvif.ONVIFCamera = MagicMock()
        with (
            patch.dict("sys.modules", {"onvif": mock_onvif}),
            patch("sensing_voxel.camera.tapo.asyncio") as mock_asyncio,
        ):
            mock_asyncio.to_thread = AsyncMock(
                side_effect=[mock_cam, mock_media, mock_ptz, mock_profiles]
            )
            mock_asyncio.sleep = AsyncMock()
            await cam.connect()
            assert cam._cam is not None
            assert cam._ptz is not None
            assert cam._profile_token == "test_token"

    async def test_disconnect_clears_ptz(self, camera):
        assert camera._ptz is not None
        await camera.disconnect()
        assert camera._ptz is None


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
        camera._ptz.GetPresets = MagicMock(return_value=[])
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
