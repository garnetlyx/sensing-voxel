"""
Pytest configuration for Sensing Voxel
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock


# Add src to path for imports
pytest_plugins = []


@pytest.fixture
def mock_config():
    """Mock configuration fixture"""
    from sensing_voxel.config import (
        Config,
        TelegramConfig,
        KlipperConfig,
        StorageConfig,
    )

    return Config(
        telegram=TelegramConfig(
            bot_token="test_token",
            allowed_users=[100000001],
        ),
        klipper=KlipperConfig(
            host="localhost",
            port=7125,
        ),
        storage=StorageConfig(
            data_dir="data/tests",
        ),
    )


@pytest.fixture
def temp_data_dir(tmp_path):
    """Temporary data directory for tests"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_image_path():
    """Path to sample test images"""
    return Path(__file__).parent / "data" / "images"


def pytest_collection_modifyitems(items):
    """Add asyncio marker to all async tests; tag live-system tests."""
    for item in items:
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)
    # Tests under tests/e2e/ and tests/real_world/ require live hardware,
    # a running dashboard, or network access to printers. Tag them as
    # "live" so they can be deselected in CI and default local runs.
    live_paths = ("tests/e2e/", "tests/real_world/")
    for item in items:
        rel = str(item.fspath)
        if rel.startswith(live_paths):
            item.add_marker(pytest.mark.live)


def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers",
        "asyncio: mark test as async (automatically applied)",
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test",
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow",
    )
    config.addinivalue_line(
        "markers",
        "live: requires live hardware, running dashboard, or network access",
    )
