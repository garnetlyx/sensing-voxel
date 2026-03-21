"""
Image-to-3D API clients for multiple providers.

Supports TripoSR, Meshy, and Tripo3D APIs with a common interface.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

import aiohttp
from PIL import Image

logger = logging.getLogger(__name__)


class ImageTo3DProvider(Enum):
    """Supported image-to-3D providers."""

    TRIPOSR = auto()
    MESHY = auto()
    TRIPO3D = auto()


class InputType(Enum):
    """Input image types supported by the APIs."""

    PHOTO = auto()
    SKETCH = auto()


@dataclass
class GenerationResult:
    """Result of a 3D model generation."""

    success: bool
    model_url: str | None = None
    model_data: bytes | None = None
    error_message: str | None = None
    provider: ImageTo3DProvider | None = None
    generation_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiViewInput:
    """Multi-view image input for 3D generation."""

    images: list[Image.Image]
    view_angles: list[float] | None = None

    def __post_init__(self):
        if self.view_angles is None:
            self.view_angles = [i * (360 / len(self.images)) for i in range(len(self.images))]


class ImageTo3DClient(ABC):
    """Abstract base class for image-to-3D API clients."""

    def __init__(self, api_key: str, base_url: str | None = None, timeout: int = 300):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> ImageTo3DClient:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers=self._get_headers(),
        )
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    @abstractmethod
    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for API requests."""
        pass

    @abstractmethod
    async def generate_from_image(
        self,
        image: Image.Image,
        input_type: InputType = InputType.PHOTO,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate 3D model from a single image."""
        pass

    @abstractmethod
    async def generate_from_multiview(
        self,
        multiview: MultiViewInput,
        input_type: InputType = InputType.PHOTO,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate 3D model from multiple views."""
        pass

    @abstractmethod
    async def get_generation_status(self, generation_id: str) -> GenerationResult:
        """Check the status of a generation."""
        pass

    def _image_to_base64(self, image: Image.Image, format: str = "PNG") -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _compute_image_hash(self, image: Image.Image) -> str:
        """Compute a hash for the image for caching purposes."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return hashlib.sha256(buffer.getvalue()).hexdigest()

    def _compute_multiview_hash(self, multiview: MultiViewInput) -> str:
        """Compute a combined hash for multi-view images."""
        hashes = []
        for img in multiview.images:
            hashes.append(self._compute_image_hash(img))
        combined = "".join(sorted(hashes))
        return hashlib.sha256(combined.encode()).hexdigest()


class TripoSRClient(ImageTo3DClient):
    """Client for TripoSR API (Stability AI)."""

    DEFAULT_BASE_URL = "https://api.stability.ai/v2beta/3d"

    def __init__(self, api_key: str, base_url: str | None = None, timeout: int = 300):
        super().__init__(api_key, base_url or self.DEFAULT_BASE_URL, timeout)
        self.provider = ImageTo3DProvider.TRIPOSR

    def _get_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def generate_from_image(
        self,
        image: Image.Image,
        input_type: InputType = InputType.PHOTO,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate 3D model from image using TripoSR."""
        if not self._session:
            raise RuntimeError("Client not initialized. Use async context manager.")

        try:
            image_b64 = self._image_to_base64(image)

            payload = {
                "image": f"data:image/png;base64,{image_b64}",
                "format": kwargs.get("format", "obj"),
            }

            if input_type == InputType.SKETCH:
                payload["sketch_mode"] = True

            async with self._session.post(
                f"{self.base_url}/stable-fast-3d",
                json=payload,
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return GenerationResult(
                        success=True,
                        model_url=result.get("model_url"),
                        generation_id=result.get("id"),
                        provider=self.provider,
                        metadata=result,
                    )
                else:
                    error_text = await response.text()
                    logger.error(f"TripoSR API error: {response.status} - {error_text}")
                    return GenerationResult(
                        success=False,
                        error_message=f"API error {response.status}: {error_text}",
                        provider=self.provider,
                    )
        except Exception as e:
            logger.exception("TripoSR generation failed")
            return GenerationResult(
                success=False,
                error_message=str(e),
                provider=self.provider,
            )

    async def generate_from_multiview(
        self,
        multiview: MultiViewInput,
        input_type: InputType = InputType.PHOTO,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate 3D model from multiple views using TripoSR."""
        if not self._session:
            raise RuntimeError("Client not initialized. Use async context manager.")

        try:
            images_b64 = [self._image_to_base64(img) for img in multiview.images]

            payload = {
                "images": [f"data:image/png;base64,{img}" for img in images_b64],
                "format": kwargs.get("format", "obj"),
            }

            if input_type == InputType.SKETCH:
                payload["sketch_mode"] = True

            async with self._session.post(
                f"{self.base_url}/multiview-to-3d",
                json=payload,
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return GenerationResult(
                        success=True,
                        model_url=result.get("model_url"),
                        generation_id=result.get("id"),
                        provider=self.provider,
                        metadata=result,
                    )
                else:
                    error_text = await response.text()
                    logger.error(f"TripoSR API error: {response.status} - {error_text}")
                    return GenerationResult(
                        success=False,
                        error_message=f"API error {response.status}: {error_text}",
                        provider=self.provider,
                    )
        except Exception as e:
            logger.exception("TripoSR multiview generation failed")
            return GenerationResult(
                success=False,
                error_message=str(e),
                provider=self.provider,
            )

    async def get_generation_status(self, generation_id: str) -> GenerationResult:
        """Check generation status for TripoSR."""
        if not self._session:
            raise RuntimeError("Client not initialized. Use async context manager.")

        try:
            async with self._session.get(
                f"{self.base_url}/results/{generation_id}",
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return GenerationResult(
                        success=result.get("status") == "completed",
                        model_url=result.get("model_url"),
                        generation_id=generation_id,
                        provider=self.provider,
                        metadata=result,
                    )
                else:
                    error_text = await response.text()
                    return GenerationResult(
                        success=False,
                        error_message=f"API error {response.status}: {error_text}",
                        provider=self.provider,
                    )
        except Exception as e:
            logger.exception("TripoSR status check failed")
            return GenerationResult(
                success=False,
                error_message=str(e),
                provider=self.provider,
            )


class MeshyClient(ImageTo3DClient):
    """Client for Meshy API."""

    DEFAULT_BASE_URL = "https://api.meshy.ai/openapi/v2"

    def __init__(self, api_key: str, base_url: str | None = None, timeout: int = 300):
        super().__init__(api_key, base_url or self.DEFAULT_BASE_URL, timeout)
        self.provider = ImageTo3DProvider.MESHY

    def _get_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def generate_from_image(
        self,
        image: Image.Image,
        input_type: InputType = InputType.PHOTO,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate 3D model from image using Meshy."""
        if not self._session:
            raise RuntimeError("Client not initialized. Use async context manager.")

        try:
            image_b64 = self._image_to_base64(image)

            payload: dict[str, Any] = {
                "image_url": f"data:image/png;base64,{image_b64}",
                "mode": "preview" if kwargs.get("preview", True) else "refine",
            }

            if input_type == InputType.SKETCH:
                payload["art_style"] = "sketch"

            async with self._session.post(
                f"{self.base_url}/text-to-3d",
                json=payload,
            ) as response:
                if response.status == 202:
                    result = await response.json()
                    return GenerationResult(
                        success=True,
                        generation_id=result.get("result"),
                        provider=self.provider,
                        metadata=result,
                    )
                else:
                    error_text = await response.text()
                    logger.error(f"Meshy API error: {response.status} - {error_text}")
                    return GenerationResult(
                        success=False,
                        error_message=f"API error {response.status}: {error_text}",
                        provider=self.provider,
                    )
        except Exception as e:
            logger.exception("Meshy generation failed")
            return GenerationResult(
                success=False,
                error_message=str(e),
                provider=self.provider,
            )

    async def generate_from_multiview(
        self,
        multiview: MultiViewInput,
        input_type: InputType = InputType.PHOTO,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate 3D model from multiple views using Meshy."""
        if not self._session:
            raise RuntimeError("Client not initialized. Use async context manager.")

        try:
            images_b64 = [self._image_to_base64(img) for img in multiview.images]

            payload: dict[str, Any] = {
                "images": [f"data:image/png;base64,{img}" for img in images_b64],
                "mode": "preview" if kwargs.get("preview", True) else "refine",
            }

            if input_type == InputType.SKETCH:
                payload["art_style"] = "sketch"

            async with self._session.post(
                f"{self.base_url}/multiview-to-3d",
                json=payload,
            ) as response:
                if response.status == 202:
                    result = await response.json()
                    return GenerationResult(
                        success=True,
                        generation_id=result.get("result"),
                        provider=self.provider,
                        metadata=result,
                    )
                else:
                    error_text = await response.text()
                    logger.error(f"Meshy API error: {response.status} - {error_text}")
                    return GenerationResult(
                        success=False,
                        error_message=f"API error {response.status}: {error_text}",
                        provider=self.provider,
                    )
        except Exception as e:
            logger.exception("Meshy multiview generation failed")
            return GenerationResult(
                success=False,
                error_message=str(e),
                provider=self.provider,
            )

    async def get_generation_status(self, generation_id: str) -> GenerationResult:
        """Check generation status for Meshy."""
        if not self._session:
            raise RuntimeError("Client not initialized. Use async context manager.")

        try:
            async with self._session.get(
                f"{self.base_url}/text-to-3d/{generation_id}",
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    status = result.get("status", "")
                    return GenerationResult(
                        success=status == "SUCCEEDED",
                        model_url=result.get("model_urls", {}).get("obj"),
                        generation_id=generation_id,
                        provider=self.provider,
                        metadata=result,
                    )
                else:
                    error_text = await response.text()
                    return GenerationResult(
                        success=False,
                        error_message=f"API error {response.status}: {error_text}",
                        provider=self.provider,
                    )
        except Exception as e:
            logger.exception("Meshy status check failed")
            return GenerationResult(
                success=False,
                error_message=str(e),
                provider=self.provider,
            )


class Tripo3DClient(ImageTo3DClient):
    """Client for Tripo3D API (VAST AI)."""

    DEFAULT_BASE_URL = "https://api.tripo3d.ai/v2/openapi"

    def __init__(self, api_key: str, base_url: str | None = None, timeout: int = 300):
        super().__init__(api_key, base_url or self.DEFAULT_BASE_URL, timeout)
        self.provider = ImageTo3DProvider.TRIPO3D

    def _get_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def generate_from_image(
        self,
        image: Image.Image,
        input_type: InputType = InputType.PHOTO,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate 3D model from image using Tripo3D."""
        if not self._session:
            raise RuntimeError("Client not initialized. Use async context manager.")

        try:
            image_b64 = self._image_to_base64(image)

            payload: dict[str, Any] = {
                "image": f"data:image/png;base64,{image_b64}",
                "type": "image_to_model",
            }

            if input_type == InputType.SKETCH:
                payload["style"] = "sketch"

            async with self._session.post(
                f"{self.base_url}/task",
                json=payload,
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return GenerationResult(
                        success=True,
                        generation_id=result.get("data", {}).get("task_id"),
                        provider=self.provider,
                        metadata=result,
                    )
                else:
                    error_text = await response.text()
                    logger.error(f"Tripo3D API error: {response.status} - {error_text}")
                    return GenerationResult(
                        success=False,
                        error_message=f"API error {response.status}: {error_text}",
                        provider=self.provider,
                    )
        except Exception as e:
            logger.exception("Tripo3D generation failed")
            return GenerationResult(
                success=False,
                error_message=str(e),
                provider=self.provider,
            )

    async def generate_from_multiview(
        self,
        multiview: MultiViewInput,
        input_type: InputType = InputType.PHOTO,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate 3D model from multiple views using Tripo3D."""
        if not self._session:
            raise RuntimeError("Client not initialized. Use async context manager.")

        try:
            images_b64 = [self._image_to_base64(img) for img in multiview.images]

            payload: dict[str, Any] = {
                "images": [f"data:image/png;base64,{img}" for img in images_b64],
                "type": "multiview_to_model",
            }

            if input_type == InputType.SKETCH:
                payload["style"] = "sketch"

            async with self._session.post(
                f"{self.base_url}/task",
                json=payload,
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return GenerationResult(
                        success=True,
                        generation_id=result.get("data", {}).get("task_id"),
                        provider=self.provider,
                        metadata=result,
                    )
                else:
                    error_text = await response.text()
                    logger.error(f"Tripo3D API error: {response.status} - {error_text}")
                    return GenerationResult(
                        success=False,
                        error_message=f"API error {response.status}: {error_text}",
                        provider=self.provider,
                    )
        except Exception as e:
            logger.exception("Tripo3D multiview generation failed")
            return GenerationResult(
                success=False,
                error_message=str(e),
                provider=self.provider,
            )

    async def get_generation_status(self, generation_id: str) -> GenerationResult:
        """Check generation status for Tripo3D."""
        if not self._session:
            raise RuntimeError("Client not initialized. Use async context manager.")

        try:
            async with self._session.get(
                f"{self.base_url}/task/{generation_id}",
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    data = result.get("data", {})
                    status = data.get("status", "")
                    return GenerationResult(
                        success=status == "success",
                        model_url=data.get("output", {}).get("model"),
                        generation_id=generation_id,
                        provider=self.provider,
                        metadata=result,
                    )
                else:
                    error_text = await response.text()
                    return GenerationResult(
                        success=False,
                        error_message=f"API error {response.status}: {error_text}",
                        provider=self.provider,
                    )
        except Exception as e:
            logger.exception("Tripo3D status check failed")
            return GenerationResult(
                success=False,
                error_message=str(e),
                provider=self.provider,
            )


class ImageTo3DService:
    """Service layer for image-to-3D generation with caching and cost tracking."""

    def __init__(
        self,
        cache: Any | None = None,
        cost_tracker: Any | None = None,
        default_provider: ImageTo3DProvider = ImageTo3DProvider.TRIPOSR,
    ):
        self.cache = cache
        self.cost_tracker = cost_tracker
        self.default_provider = default_provider
        self._clients: dict[ImageTo3DProvider, ImageTo3DClient] = {}

    def register_client(
        self,
        provider: ImageTo3DProvider,
        client: ImageTo3DClient,
    ) -> None:
        """Register an API client for a provider."""
        self._clients[provider] = client

    async def generate(
        self,
        image: Image.Image,
        user_id: int | str,
        job_id: str,
        provider: ImageTo3DProvider | None = None,
        input_type: InputType = InputType.PHOTO,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate 3D model with caching and cost tracking."""
        provider = provider or self.default_provider

        if provider not in self._clients:
            return GenerationResult(
                success=False,
                error_message=f"No client registered for provider {provider.name}",
                provider=provider,
            )

        client = self._clients[provider]

        image_hash = client._compute_image_hash(image)
        cache_key = f"{provider.name}:{image_hash}:{input_type.name}"

        if use_cache and self.cache:
            cached = await self.cache.get(cache_key)
            if cached:
                logger.info(f"Cache hit for {cache_key}")
                cached_result = GenerationResult(**cached)
                cached_result.metadata["from_cache"] = True
                return cached_result

        async with client:
            result = await client.generate_from_image(image, input_type, **kwargs)

        if result.success and self.cache:
            await self.cache.set(cache_key, result.__dict__)

        if self.cost_tracker:
            await self.cost_tracker.record_generation(
                user_id=user_id,
                job_id=job_id,
                provider=provider,
                success=result.success,
                metadata=kwargs,
            )

        return result

    async def generate_multiview(
        self,
        multiview: MultiViewInput,
        user_id: int | str,
        job_id: str,
        provider: ImageTo3DProvider | None = None,
        input_type: InputType = InputType.PHOTO,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate 3D model from multiple views with caching and cost tracking."""
        provider = provider or self.default_provider

        if provider not in self._clients:
            return GenerationResult(
                success=False,
                error_message=f"No client registered for provider {provider.name}",
                provider=provider,
            )

        client = self._clients[provider]

        multiview_hash = client._compute_multiview_hash(multiview)
        cache_key = f"{provider.name}:multiview:{multiview_hash}:{input_type.name}"

        if use_cache and self.cache:
            cached = await self.cache.get(cache_key)
            if cached:
                logger.info(f"Cache hit for {cache_key}")
                cached_result = GenerationResult(**cached)
                cached_result.metadata["from_cache"] = True
                return cached_result

        async with client:
            result = await client.generate_from_multiview(multiview, input_type, **kwargs)

        if result.success and self.cache:
            await self.cache.set(cache_key, result.__dict__)

        if self.cost_tracker:
            await self.cost_tracker.record_generation(
                user_id=user_id,
                job_id=job_id,
                provider=provider,
                success=result.success,
                metadata={"multiview": True, **kwargs},
            )

        return result

    async def check_status(
        self,
        generation_id: str,
        provider: ImageTo3DProvider,
    ) -> GenerationResult:
        """Check generation status."""
        if provider not in self._clients:
            return GenerationResult(
                success=False,
                error_message=f"No client registered for provider {provider.name}",
                provider=provider,
            )

        client = self._clients[provider]
        async with client:
            return await client.get_generation_status(generation_id)
