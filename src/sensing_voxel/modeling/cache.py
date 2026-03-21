"""
Result caching for image-to-3D generations.

Provides caching by image hash to avoid duplicate API calls.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    async def get(self, key: str) -> dict[str, Any] | None:
        """Get value from cache by key."""
        pass

    @abstractmethod
    async def set(self, key: str, value: dict[str, Any], ttl: int | None = None) -> None:
        """Set value in cache with optional TTL in seconds."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cached values."""
        pass


class FileCacheBackend(CacheBackend):
    """File-based cache backend using JSON files."""

    def __init__(self, cache_dir: Path | str, extension: str = ".json"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.extension = extension

    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key."""
        safe_key = "".join(c for c in key if c.isalnum() or c in "-_").rstrip()
        return self.cache_dir / f"{safe_key}{self.extension}"

    async def get(self, key: str) -> dict[str, Any] | None:
        """Get cached value from file."""
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read cache file {cache_path}: {e}")
            return None

    async def set(self, key: str, value: dict[str, Any], ttl: int | None = None) -> None:
        """Write value to cache file."""
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(value, f, indent=2)
        except OSError as e:
            logger.warning(f"Failed to write cache file {cache_path}: {e}")

    async def delete(self, key: str) -> None:
        """Delete cache file."""
        cache_path = self._get_cache_path(key)
        try:
            cache_path.unlink(missing_ok=True)
        except OSError as e:
            logger.warning(f"Failed to delete cache file {cache_path}: {e}")

    async def clear(self) -> None:
        """Clear all cache files."""
        for cache_file in self.cache_dir.glob(f"*{self.extension}"):
            try:
                cache_file.unlink()
            except OSError as e:
                logger.warning(f"Failed to delete cache file {cache_file}: {e}")


class MemoryCacheBackend(CacheBackend):
    """In-memory cache backend."""

    def __init__(self):
        self._cache: dict[str, dict[str, Any]] = {}

    async def get(self, key: str) -> dict[str, Any] | None:
        """Get value from memory cache."""
        return self._cache.get(key)

    async def set(self, key: str, value: dict[str, Any], ttl: int | None = None) -> None:
        """Set value in memory cache."""
        self._cache[key] = value

    async def delete(self, key: str) -> None:
        """Delete from memory cache."""
        self._cache.pop(key, None)

    async def clear(self) -> None:
        """Clear all memory cache."""
        self._cache.clear()


class ModelCache:
    """Cache manager for 3D model generations."""

    def __init__(self, backend: CacheBackend | None = None):
        self.backend = backend or MemoryCacheBackend()

    async def get(self, key: str) -> dict[str, Any] | None:
        """Get cached generation result."""
        return await self.backend.get(key)

    async def set(self, key: str, value: dict[str, Any], ttl: int | None = None) -> None:
        """Cache generation result."""
        await self.backend.set(key, value, ttl)

    async def delete(self, key: str) -> None:
        """Delete cached result."""
        await self.backend.delete(key)

    async def clear(self) -> None:
        """Clear all cached results."""
        await self.backend.clear()
