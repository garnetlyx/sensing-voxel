"""
Async LLM client wrapper using Anthropic SDK
"""

from __future__ import annotations

import logging

import anthropic

from sensing_voxel.config import LlmConfig
from sensing_voxel.infra.retry import BackoffPolicy, retry_async

logger = logging.getLogger(__name__)

# Errors that are safe to retry
RETRYABLE_ERRORS = (
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)


class LlmClient:
    """Async LLM client wrapper using Anthropic SDK"""

    def __init__(self, config: LlmConfig):
        self._config = config
        self._client: anthropic.AsyncAnthropic | None = None

    async def initialize(self) -> None:
        """Initialize the Anthropic async client."""
        if not self._config.api_key:
            raise ValueError("LLM api_key is required")

        self._client = anthropic.AsyncAnthropic(
            api_key=self._config.api_key,
            base_url=self._config.base_url,
            timeout=self._config.timeout_seconds,
        )
        logger.info(
            f"LLM client initialized: model={self._config.model}, "
            f"base_url={self._config.base_url}"
        )

    async def send_message(
        self,
        messages: list[dict],
        system_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Send messages and return assistant response text.

        Args:
            messages: Conversation messages [{"role": "user"/"assistant", "content": "..."}]
            system_prompt: System prompt with context
            max_tokens: Override max tokens (default from config)
            temperature: Override temperature (default from config)

        Returns:
            Assistant response text
        """
        if not self._client:
            raise RuntimeError("LLM client not initialized. Call initialize() first.")

        async def _call():
            response = await self._client.messages.create(
                model=self._config.model,
                max_tokens=max_tokens or self._config.max_tokens,
                temperature=temperature if temperature is not None else self._config.temperature,
                system=system_prompt,
                messages=messages,
            )
            return response.content[0].text

        return await retry_async(
            _call,
            max_attempts=3,
            policy=BackoffPolicy(base_delay=1.0, max_delay=10.0, factor=2.0, jitter=0.25),
            retryable_exceptions=RETRYABLE_ERRORS,
            on_retry=lambda attempt, e: logger.warning(
                f"LLM API retry {attempt + 1}/3: {type(e).__name__}: {e}"
            ),
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("LLM client closed")

    @property
    def is_available(self) -> bool:
        """Check if client is initialized and ready."""
        return self._client is not None
