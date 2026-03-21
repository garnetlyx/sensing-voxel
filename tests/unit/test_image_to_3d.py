"""
Unit tests for image-to-3D modeling module.

Tests API clients, caching, and cost tracking.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from PIL import Image

from sensing_voxel.modeling.cache import FileCacheBackend, MemoryCacheBackend, ModelCache
from sensing_voxel.modeling.cost_tracker import CostTracker, GenerationCost
from sensing_voxel.modeling.image_to_3d import (
    GenerationResult,
    ImageTo3DProvider,
    ImageTo3DService,
    InputType,
    MeshyClient,
    MultiViewInput,
    Tripo3DClient,
    TripoSRClient,
)


@pytest.fixture
def mock_image():
    """Create a mock PIL Image."""
    img = Image.new("RGB", (512, 512), color="red")
    return img


@pytest.fixture
def mock_multiview(mock_image):
    """Create a mock MultiViewInput with 4 views."""
    return MultiViewInput(
        images=[mock_image, mock_image, mock_image, mock_image],
        view_angles=[0, 90, 180, 270],
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# === Test TripoSR Client ===


class TestTripoSRClient:
    """Tests for TripoSR API client."""

    def test_init(self):
        """Test client initialization."""
        client = TripoSRClient(api_key="test_key")
        assert client.api_key == "test_key"
        assert client.base_url == TripoSRClient.DEFAULT_BASE_URL
        assert client.provider == ImageTo3DProvider.TRIPOSR

    def test_get_headers(self):
        """Test header generation."""
        client = TripoSRClient(api_key="test_key")
        headers = client._get_headers()
        assert headers["Authorization"] == "Bearer test_key"
        assert headers["Content-Type"] == "application/json"

    def test_image_to_base64(self, mock_image):
        """Test image conversion to base64."""
        client = TripoSRClient(api_key="test_key")
        b64_string = client._image_to_base64(mock_image)
        assert isinstance(b64_string, str)
        assert len(b64_string) > 0

    def test_compute_image_hash(self, mock_image):
        """Test image hash computation."""
        client = TripoSRClient(api_key="test_key")
        hash1 = client._compute_image_hash(mock_image)
        hash2 = client._compute_image_hash(mock_image)
        assert isinstance(hash1, str)
        assert len(hash1) == 64
        assert hash1 == hash2

    def test_compute_multiview_hash(self, mock_image):
        """Test multiview hash computation."""
        client = TripoSRClient(api_key="test_key")
        multiview = MultiViewInput(
            images=[mock_image, mock_image],
            view_angles=[0, 180],
        )
        hash_val = client._compute_multiview_hash(multiview)
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64


class TestMeshyClient:
    """Tests for Meshy API client."""

    def test_init(self):
        """Test client initialization."""
        client = MeshyClient(api_key="test_key")
        assert client.api_key == "test_key"
        assert client.base_url == MeshyClient.DEFAULT_BASE_URL
        assert client.provider == ImageTo3DProvider.MESHY

    def test_get_headers(self):
        """Test header generation."""
        client = MeshyClient(api_key="test_key")
        headers = client._get_headers()
        assert headers["Authorization"] == "Bearer test_key"
        assert headers["Content-Type"] == "application/json"

    def test_image_to_base64(self, mock_image):
        """Test image conversion to base64."""
        client = MeshyClient(api_key="test_key")
        b64_string = client._image_to_base64(mock_image)
        assert isinstance(b64_string, str)
        assert len(b64_string) > 0


class TestTripo3DClient:
    """Tests for Tripo3D API client."""

    def test_init(self):
        """Test client initialization."""
        client = Tripo3DClient(api_key="test_key")
        assert client.api_key == "test_key"
        assert client.base_url == Tripo3DClient.DEFAULT_BASE_URL
        assert client.provider == ImageTo3DProvider.TRIPO3D

    def test_get_headers(self):
        """Test header generation."""
        client = Tripo3DClient(api_key="test_key")
        headers = client._get_headers()
        assert headers["Authorization"] == "Bearer test_key"
        assert headers["Content-Type"] == "application/json"

    def test_image_to_base64(self, mock_image):
        """Test image conversion to base64."""
        client = Tripo3DClient(api_key="test_key")
        b64_string = client._image_to_base64(mock_image)
        assert isinstance(b64_string, str)
        assert len(b64_string) > 0


# === Test Cache ===


class TestMemoryCacheBackend:
    """Tests for in-memory cache backend."""

    @pytest.mark.asyncio
    async def test_get_set(self):
        """Test get and set operations."""
        cache = MemoryCacheBackend()
        data = {"model_url": "https://example.com/model.obj"}

        await cache.set("key1", data)
        result = await cache.get("key1")

        assert result == data

    @pytest.mark.asyncio
    async def test_get_missing(self):
        """Test getting non-existent key."""
        cache = MemoryCacheBackend()
        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self):
        """Test delete operation."""
        cache = MemoryCacheBackend()
        data = {"model_url": "https://example.com/model.obj"}

        await cache.set("key1", data)
        await cache.delete("key1")
        result = await cache.get("key1")

        assert result is None

    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clear operation."""
        cache = MemoryCacheBackend()
        data = {"model_url": "https://example.com/model.obj"}

        await cache.set("key1", data)
        await cache.set("key2", data)
        await cache.clear()

        assert await cache.get("key1") is None
        assert await cache.get("key2") is None


class TestFileCacheBackend:
    """Tests for file-based cache backend."""

    @pytest.mark.asyncio
    async def test_get_set(self, temp_dir):
        """Test get and set operations."""
        cache = FileCacheBackend(temp_dir)
        data = {"model_url": "https://example.com/model.obj"}

        await cache.set("key1", data)
        result = await cache.get("key1")

        assert result == data

    @pytest.mark.asyncio
    async def test_persistence(self, temp_dir):
        """Test that data persists across cache instances."""
        cache1 = FileCacheBackend(temp_dir)
        data = {"model_url": "https://example.com/model.obj"}

        await cache1.set("key1", data)

        cache2 = FileCacheBackend(temp_dir)
        result = await cache2.get("key1")

        assert result == data

    @pytest.mark.asyncio
    async def test_clear(self, temp_dir):
        """Test clear operation."""
        cache = FileCacheBackend(temp_dir)
        data = {"model_url": "https://example.com/model.obj"}

        await cache.set("key1", data)
        await cache.clear()

        assert await cache.get("key1") is None


class TestModelCache:
    """Tests for ModelCache wrapper."""

    @pytest.mark.asyncio
    async def test_with_memory_backend(self):
        """Test with memory backend."""
        backend = MemoryCacheBackend()
        cache = ModelCache(backend)
        data = {"model_url": "https://example.com/model.obj"}

        await cache.set("key1", data)
        result = await cache.get("key1")

        assert result == data


# === Test Cost Tracker ===


class TestCostTracker:
    """Tests for cost tracking."""

    def test_record_generation(self, temp_dir):
        """Test recording a generation."""
        storage_path = temp_dir / "costs.json"
        tracker = CostTracker(storage_path=storage_path)

        cost = asyncio_run(
            tracker.record_generation(
                user_id="user_123",
                job_id="job_456",
                provider=ImageTo3DProvider.TRIPOSR,
                success=True,
            )
        )

        assert cost.user_id == "user_123"
        assert cost.job_id == "job_456"
        assert cost.provider == ImageTo3DProvider.TRIPOSR
        assert cost.success is True
        assert cost.estimated_cost_usd == 0.02

    def test_multiview_cost_calculation(self, temp_dir):
        """Test multiview cost is 1.5x base cost."""
        storage_path = temp_dir / "costs.json"
        tracker = CostTracker(storage_path=storage_path)

        cost = asyncio_run(
            tracker.record_generation(
                user_id="user_123",
                job_id="job_456",
                provider=ImageTo3DProvider.TRIPOSR,
                success=True,
                metadata={"multiview": True},
            )
        )

        assert cost.estimated_cost_usd == 0.03

    def test_get_user_costs(self, temp_dir):
        """Test getting costs for a user."""
        storage_path = temp_dir / "costs.json"
        tracker = CostTracker(storage_path=storage_path)

        asyncio_run(
            tracker.record_generation(
                user_id="user_123",
                job_id="job_1",
                provider=ImageTo3DProvider.TRIPOSR,
                success=True,
            )
        )
        asyncio_run(
            tracker.record_generation(
                user_id="user_123",
                job_id="job_2",
                provider=ImageTo3DProvider.MESHY,
                success=False,
            )
        )
        asyncio_run(
            tracker.record_generation(
                user_id="user_456",
                job_id="job_3",
                provider=ImageTo3DProvider.TRIPOSR,
                success=True,
            )
        )

        user_costs = tracker.get_user_costs("user_123")

        assert len(user_costs) == 2
        assert all(c.user_id == "user_123" for c in user_costs)

    def test_get_job_cost(self, temp_dir):
        """Test getting cost for a specific job."""
        storage_path = temp_dir / "costs.json"
        tracker = CostTracker(storage_path=storage_path)

        asyncio_run(
            tracker.record_generation(
                user_id="user_123",
                job_id="job_789",
                provider=ImageTo3DProvider.TRIPOSR,
                success=True,
            )
        )

        cost = tracker.get_job_cost("job_789")

        assert cost is not None
        assert cost.job_id == "job_789"

    def test_get_total_cost(self, temp_dir):
        """Test getting total cost."""
        storage_path = temp_dir / "costs.json"
        tracker = CostTracker(storage_path=storage_path)

        asyncio_run(
            tracker.record_generation(
                user_id="user_123",
                job_id="job_1",
                provider=ImageTo3DProvider.TRIPOSR,
                success=True,
            )
        )
        asyncio_run(
            tracker.record_generation(
                user_id="user_123",
                job_id="job_2",
                provider=ImageTo3DProvider.MESHY,
                success=True,
            )
        )

        total = tracker.get_total_cost()
        assert total == 0.03

        user_total = tracker.get_total_cost("user_123")
        assert user_total == 0.03

    def test_get_user_summary(self, temp_dir):
        """Test getting user cost summary."""
        storage_path = temp_dir / "costs.json"
        tracker = CostTracker(storage_path=storage_path)

        asyncio_run(
            tracker.record_generation(
                user_id="user_123",
                job_id="job_1",
                provider=ImageTo3DProvider.TRIPOSR,
                success=True,
            )
        )
        asyncio_run(
            tracker.record_generation(
                user_id="user_123",
                job_id="job_2",
                provider=ImageTo3DProvider.MESHY,
                success=False,
            )
        )

        summary = tracker.get_user_summary("user_123")

        assert summary["user_id"] == "user_123"
        assert summary["total_generations"] == 2
        assert summary["successful_generations"] == 1
        assert summary["failed_generations"] == 1
        assert summary["total_cost"] == 0.03

    def test_get_provider_usage(self, temp_dir):
        """Test getting provider usage stats."""
        storage_path = temp_dir / "costs.json"
        tracker = CostTracker(storage_path=storage_path)

        asyncio_run(
            tracker.record_generation(
                user_id="user_123",
                job_id="job_1",
                provider=ImageTo3DProvider.TRIPOSR,
                success=True,
            )
        )
        asyncio_run(
            tracker.record_generation(
                user_id="user_123",
                job_id="job_2",
                provider=ImageTo3DProvider.TRIPOSR,
                success=True,
            )
        )
        asyncio_run(
            tracker.record_generation(
                user_id="user_123",
                job_id="job_3",
                provider=ImageTo3DProvider.TRIPOSR,
                success=False,
            )
        )

        usage = tracker.get_provider_usage(ImageTo3DProvider.TRIPOSR)

        assert usage["total_calls"] == 3
        assert usage["successful_calls"] == 2
        assert usage["failed_calls"] == 1
        assert usage["success_rate"] == 2 / 3
        assert usage["total_cost"] == 0.06

    def test_persistence(self, temp_dir):
        """Test that costs persist across tracker instances."""
        storage_path = temp_dir / "costs.json"

        tracker1 = CostTracker(storage_path=storage_path)
        asyncio_run(
            tracker1.record_generation(
                user_id="user_123",
                job_id="job_1",
                provider=ImageTo3DProvider.TRIPOSR,
                success=True,
            )
        )

        tracker2 = CostTracker(storage_path=storage_path)
        costs = tracker2.get_user_costs("user_123")

        assert len(costs) == 1

    def test_clear_history_user(self, temp_dir):
        """Test clearing history for a specific user."""
        storage_path = temp_dir / "costs.json"
        tracker = CostTracker(storage_path=storage_path)

        asyncio_run(
            tracker.record_generation(
                user_id="user_123",
                job_id="job_1",
                provider=ImageTo3DProvider.TRIPOSR,
                success=True,
            )
        )
        asyncio_run(
            tracker.record_generation(
                user_id="user_456",
                job_id="job_2",
                provider=ImageTo3DProvider.MESHY,
                success=True,
            )
        )

        cleared = tracker.clear_history("user_123")

        assert cleared == 1
        assert len(tracker.get_user_costs("user_123")) == 0
        assert len(tracker.get_user_costs("user_456")) == 1

    def test_clear_history_all(self, temp_dir):
        """Test clearing all history."""
        storage_path = temp_dir / "costs.json"
        tracker = CostTracker(storage_path=storage_path)

        asyncio_run(
            tracker.record_generation(
                user_id="user_123",
                job_id="job_1",
                provider=ImageTo3DProvider.TRIPOSR,
                success=True,
            )
        )

        cleared = tracker.clear_history()

        assert cleared == 1
        assert len(tracker._costs) == 0


# === Test GenerationCost ===


class TestGenerationCost:
    """Tests for GenerationCost dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        cost = GenerationCost(
            user_id="user_123",
            job_id="job_456",
            provider=ImageTo3DProvider.TRIPOSR,
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            success=True,
            estimated_cost_usd=0.02,
            api_calls=1,
            metadata={"multiview": False},
        )

        data = cost.to_dict()

        assert data["user_id"] == "user_123"
        assert data["job_id"] == "job_456"
        assert data["provider"] == "TRIPOSR"
        assert data["success"] is True
        assert data["estimated_cost_usd"] == 0.02

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "user_id": "user_123",
            "job_id": "job_456",
            "provider": "TRIPOSR",
            "timestamp": "2024-01-01T12:00:00",
            "success": True,
            "estimated_cost_usd": 0.02,
            "api_calls": 1,
            "metadata": {},
        }

        cost = GenerationCost.from_dict(data)

        assert cost.user_id == "user_123"
        assert cost.job_id == "job_456"
        assert cost.provider == ImageTo3DProvider.TRIPOSR
        assert cost.success is True


# === Test ImageTo3DService ===


class TestImageTo3DService:
    """Tests for ImageTo3DService."""

    @pytest.mark.asyncio
    async def test_register_client(self):
        """Test registering a client."""
        service = ImageTo3DService()
        mock_client = Mock(spec=TripoSRClient)

        service.register_client(ImageTo3DProvider.TRIPOSR, mock_client)

        assert ImageTo3DProvider.TRIPOSR in service._clients

    @pytest.mark.asyncio
    async def test_generate_no_client(self, mock_image):
        """Test generation without registered client."""
        service = ImageTo3DService()

        result = await service.generate(
            image=mock_image,
            user_id="user_123",
            job_id="job_456",
            provider=ImageTo3DProvider.TRIPOSR,
        )

        assert result.success is False
        assert "No client registered" in result.error_message

    @pytest.mark.asyncio
    async def test_generate_with_cache_hit(self, mock_image):
        """Test generation with cache hit."""
        cache = ModelCache(MemoryCacheBackend())
        service = ImageTo3DService(cache=cache)

        cached_data = {
            "success": True,
            "model_url": "https://example.com/cached.obj",
            "generation_id": "cached_123",
            "error_message": None,
            "provider": "TRIPOSR",
            "metadata": {},
        }
        await cache.set("TRIPOSR:abc123:PHOTO", cached_data)

        mock_client = Mock(spec=TripoSRClient)
        mock_client._compute_image_hash.return_value = "abc123"
        service.register_client(ImageTo3DProvider.TRIPOSR, mock_client)

        result = await service.generate(
            image=mock_image,
            user_id="user_123",
            job_id="job_456",
            provider=ImageTo3DProvider.TRIPOSR,
        )

        assert result.success is True
        assert result.metadata.get("from_cache") is True


# === Test MultiViewInput ===


class TestMultiViewInput:
    """Tests for MultiViewInput dataclass."""

    def test_default_view_angles(self, mock_image):
        """Test default view angles are generated."""
        multiview = MultiViewInput(images=[mock_image, mock_image, mock_image, mock_image])

        assert multiview.view_angles == [0, 90, 180, 270]

    def test_custom_view_angles(self, mock_image):
        """Test custom view angles."""
        multiview = MultiViewInput(
            images=[mock_image, mock_image],
            view_angles=[45, 225],
        )

        assert multiview.view_angles == [45, 225]


# === Helper ===


def asyncio_run(coro):
    """Helper to run async functions in sync tests."""
    import asyncio

    return asyncio.run(coro)


# === Test InputType ===


class TestInputType:
    """Tests for InputType enum."""

    def test_input_types(self):
        """Test input types exist."""
        assert InputType.PHOTO
        assert InputType.SKETCH
