"""
Async LLM client wrapper supporting MiniMax (Anthropic SDK) and Ollama (OpenAI-compatible)
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

import anthropic
import httpx

from sensing_voxel.config import LlmConfig, LlmProvider
from sensing_voxel.infra.retry import BackoffPolicy, retry_async

logger = logging.getLogger(__name__)

# Errors that are safe to retry for Anthropic
ANTHROPIC_RETRYABLE_ERRORS = (
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)

# Errors that are safe to retry for Ollama
OLLAMA_RETRYABLE_ERRORS = (
    httpx.ConnectError,
    httpx.TimeoutException,
    httpx.NetworkError,
)


class BaseLlmClient(ABC):
    """Abstract base class for LLM clients"""

    def __init__(self, config: LlmConfig):
        self._config = config

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the client."""
        pass

    @abstractmethod
    async def send_message(
        self,
        messages: list[dict],
        system_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Send messages and return assistant response text."""
        pass

    @abstractmethod
    async def send_message_with_tools(
        self,
        messages: list[dict],
        system_prompt: str,
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Any:
        """Send messages with tool definitions and return response object."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the underlying HTTP client."""
        pass


class AnthropicClient(BaseLlmClient):
    """Async LLM client wrapper using Anthropic SDK (MiniMax)"""

    def __init__(self, config: LlmConfig):
        super().__init__(config)
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
            f"Anthropic client initialized: model={self._config.model}, "
            f"base_url={self._config.base_url}"
        )

    async def send_message(
        self,
        messages: list[dict],
        system_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Send messages and return assistant response text."""
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
            # Find the text block (some models return ThinkingBlock before TextBlock)
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return response.content[0].text if response.content else ""

        return await retry_async(
            _call,
            max_attempts=3,
            policy=BackoffPolicy(base_delay=1.0, max_delay=10.0, factor=2.0, jitter=0.25),
            retryable_exceptions=ANTHROPIC_RETRYABLE_ERRORS,
            on_retry=lambda attempt, e: logger.warning(
                f"LLM API retry {attempt + 1}/3: {type(e).__name__}: {e}"
            ),
        )

    async def send_message_with_tools(
        self,
        messages: list[dict],
        system_prompt: str,
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> anthropic.types.Message:
        """Send messages with tool definitions and return full Message object."""
        if not self._client:
            raise RuntimeError("LLM client not initialized. Call initialize() first.")

        async def _call():
            kwargs: dict = {
                "model": self._config.model,
                "max_tokens": max_tokens or self._config.max_tokens,
                "temperature": (
                    temperature if temperature is not None else self._config.temperature
                ),
                "system": system_prompt,
                "messages": messages,
            }
            if tools:
                kwargs["tools"] = tools
            return await self._client.messages.create(**kwargs)

        return await retry_async(
            _call,
            max_attempts=3,
            policy=BackoffPolicy(base_delay=1.0, max_delay=10.0, factor=2.0, jitter=0.25),
            retryable_exceptions=ANTHROPIC_RETRYABLE_ERRORS,
            on_retry=lambda attempt, e: logger.warning(
                f"LLM API retry {attempt + 1}/3: {type(e).__name__}: {e}"
            ),
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("Anthropic client closed")

    @property
    def is_available(self) -> bool:
        """Check if client is initialized and ready."""
        return self._client is not None


class OllamaClient(BaseLlmClient):
    """Async LLM client wrapper using Ollama's OpenAI-compatible API"""

    def __init__(self, config: LlmConfig):
        super().__init__(config)
        self._client: httpx.AsyncClient | None = None
        self._base_url = config.base_url or "http://localhost:11434/v1"
        self._headers = {}
        if config.api_key:
            self._headers["Authorization"] = f"Bearer {config.api_key}"

    async def initialize(self) -> None:
        """Initialize the Ollama HTTP client."""
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._config.timeout_seconds,
            headers=self._headers,
        )
        logger.info(
            f"Ollama client initialized: model={self._config.model}, base_url={self._base_url}"
        )

    def _convert_messages_to_openai(self, messages: list[dict], system_prompt: str) -> list[dict]:
        """Convert internal message format to OpenAI format."""
        openai_messages = []
        if system_prompt:
            openai_messages.append({"role": "system", "content": system_prompt})
        for msg in messages:
            openai_msg = {"role": msg["role"], "content": msg["content"]}
            openai_messages.append(openai_msg)
        return openai_messages

    def _convert_tools_to_openai(self, tools: list[dict] | None) -> list[dict] | None:
        """Convert Anthropic tool format to OpenAI tool format."""
        if not tools:
            return None
        openai_tools = []
        for tool in tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            }
            openai_tools.append(openai_tool)
        return openai_tools

    async def send_message(
        self,
        messages: list[dict],
        system_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Send messages and return assistant response text."""
        if not self._client:
            raise RuntimeError("LLM client not initialized. Call initialize() first.")

        async def _call():
            payload = {
                "model": self._config.model,
                "messages": self._convert_messages_to_openai(messages, system_prompt),
                "max_tokens": max_tokens or self._config.max_tokens,
                "temperature": temperature if temperature is not None else self._config.temperature,
            }
            response = await self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

        return await retry_async(
            _call,
            max_attempts=3,
            policy=BackoffPolicy(base_delay=1.0, max_delay=10.0, factor=2.0, jitter=0.25),
            retryable_exceptions=OLLAMA_RETRYABLE_ERRORS,
            on_retry=lambda attempt, e: logger.warning(
                f"Ollama API retry {attempt + 1}/3: {type(e).__name__}: {e}"
            ),
        )

    async def send_message_with_tools(
        self,
        messages: list[dict],
        system_prompt: str,
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict:
        """Send messages with tool definitions and return response dict.

        Returns a dict compatible with Anthropic Message-like interface.
        """
        if not self._client:
            raise RuntimeError("LLM client not initialized. Call initialize() first.")

        async def _call():
            payload = {
                "model": self._config.model,
                "messages": self._convert_messages_to_openai(messages, system_prompt),
                "max_tokens": max_tokens or self._config.max_tokens,
                "temperature": temperature if temperature is not None else self._config.temperature,
            }
            if tools:
                openai_tools = self._convert_tools_to_openai(tools)
                if openai_tools:
                    payload["tools"] = openai_tools

            response = await self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            message = data["choices"][0]["message"]

            # Convert OpenAI response to Anthropic-like structure
            content = []
            if message.get("content"):
                content.append({"type": "text", "text": message["content"]})

            # Handle tool calls
            if message.get("tool_calls"):
                for tool_call in message["tool_calls"]:
                    if tool_call.get("type") == "function":
                        func = tool_call["function"]
                        content.append(
                            {
                                "type": "tool_use",
                                "name": func["name"],
                                "input": json.loads(func["arguments"]),
                            }
                        )

            return {
                "content": content,
                "role": message.get("role", "assistant"),
                "model": data.get("model", self._config.model),
            }

        return await retry_async(
            _call,
            max_attempts=3,
            policy=BackoffPolicy(base_delay=1.0, max_delay=10.0, factor=2.0, jitter=0.25),
            retryable_exceptions=OLLAMA_RETRYABLE_ERRORS,
            on_retry=lambda attempt, e: logger.warning(
                f"Ollama API retry {attempt + 1}/3: {type(e).__name__}: {e}"
            ),
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("Ollama client closed")

    @property
    def is_available(self) -> bool:
        """Check if client is initialized and ready."""
        return self._client is not None


class LlmClient:
    """Unified LLM client that auto-selects backend based on provider"""

    def __init__(self, config: LlmConfig):
        self._config = config
        self._impl: BaseLlmClient | None = None

    async def initialize(self) -> None:
        """Initialize the appropriate LLM client based on provider."""
        if self._config.provider == LlmProvider.OLLAMA:
            self._impl = OllamaClient(self._config)
        else:
            self._impl = AnthropicClient(self._config)
        await self._impl.initialize()

    async def send_message(
        self,
        messages: list[dict],
        system_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Send messages and return assistant response text."""
        if not self._impl:
            raise RuntimeError("LLM client not initialized. Call initialize() first.")
        return await self._impl.send_message(messages, system_prompt, max_tokens, temperature)

    async def send_message_with_tools(
        self,
        messages: list[dict],
        system_prompt: str,
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Any:
        """Send messages with tool definitions and return response."""
        if not self._impl:
            raise RuntimeError("LLM client not initialized. Call initialize() first.")
        return await self._impl.send_message_with_tools(
            messages, system_prompt, tools, max_tokens, temperature
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._impl:
            await self._impl.close()
            self._impl = None

    @property
    def is_available(self) -> bool:
        """Check if client is initialized and ready."""
        return self._impl is not None and self._impl.is_available
