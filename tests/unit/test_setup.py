"""
Test that pytest and async fixtures work
"""

import pytest


@pytest.mark.asyncio
async def test_async_fixture():
    """Test that async fixtures work"""
    assert True


def test_sync():
    """Test that sync tests work"""
    assert True
