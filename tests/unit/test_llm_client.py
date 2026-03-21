"""Tests for LLM client wrapper"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from sensing_voxel.config import LlmConfig, LlmProvider
from sensing_voxel.llm.client import LlmClient, AnthropicClient, OllamaClient


@pytest.fixture
def minimax_config():
    return LlmConfig(
        provider=LlmProvider.MINIMAX,
        api_key="test-api-key",
        base_url="https://api.test.com",
        model="test-model",
        max_tokens=1024,
        temperature=0.5,
        timeout_seconds=10,
        enabled=True,
    )


@pytest.fixture
def ollama_config():
    return LlmConfig(
        provider=LlmProvider.OLLAMA,
        api_key=None,
        base_url="http://localhost:11434/v1",
        model="llama3.2",
        max_tokens=1024,
        temperature=0.5,
        timeout_seconds=10,
        enabled=True,
    )


@pytest.fixture
def llm_client(minimax_config):
    return LlmClient(minimax_config)


class TestLlmClient:
    async def test_is_available_before_init(self, llm_client):
        assert llm_client.is_available is False

    async def test_initialize_selects_anthropic_for_minimax(self, minimax_config):
        with patch("sensing_voxel.llm.client.anthropic.AsyncAnthropic"):
            client = LlmClient(minimax_config)
            await client.initialize()
            assert client.is_available is True
            assert isinstance(client._impl, AnthropicClient)

    async def test_initialize_selects_ollama_for_ollama_provider(self, ollama_config):
        client = LlmClient(ollama_config)
        await client.initialize()
        assert client.is_available is True
        assert isinstance(client._impl, OllamaClient)
        await client.close()

    async def test_initialize_without_api_key_raises_for_minimax(self):
        config = LlmConfig(provider=LlmProvider.MINIMAX, api_key=None)
        client = LlmClient(config)
        with pytest.raises(ValueError, match="api_key is required"):
            await client.initialize()

    async def test_ollama_no_api_key_required(self, ollama_config):
        client = LlmClient(ollama_config)
        # Should not raise
        await client.initialize()
        assert client.is_available is True
        await client.close()

    async def test_send_message_returns_text(self, llm_client):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hello from AI")]

        mock_anthropic = AsyncMock()
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)

        with patch(
            "sensing_voxel.llm.client.anthropic.AsyncAnthropic", return_value=mock_anthropic
        ):
            await llm_client.initialize()

        llm_client._impl._client = mock_anthropic
        result = await llm_client.send_message(
            messages=[{"role": "user", "content": "hi"}],
            system_prompt="You are a test bot",
        )

        assert result == "Hello from AI"
        mock_anthropic.messages.create.assert_called_once()
        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["max_tokens"] == 1024
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["system"] == "You are a test bot"

    async def test_send_message_without_init_raises(self, llm_client):
        with pytest.raises(RuntimeError, match="not initialized"):
            await llm_client.send_message(
                messages=[{"role": "user", "content": "hi"}],
                system_prompt="test",
            )

    async def test_send_message_override_params(self, llm_client):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="response")]

        mock_anthropic = AsyncMock()
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)
        llm_client._impl = AnthropicClient(llm_client._config)
        llm_client._impl._client = mock_anthropic

        await llm_client.send_message(
            messages=[{"role": "user", "content": "hi"}],
            system_prompt="test",
            max_tokens=500,
            temperature=0.9,
        )

        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 500
        assert call_kwargs["temperature"] == 0.9

    async def test_close_cleans_up(self, llm_client):
        mock_anthropic = AsyncMock()
        llm_client._impl = AnthropicClient(llm_client._config)
        llm_client._impl._client = mock_anthropic

        await llm_client.close()
        mock_anthropic.close.assert_called_once()
        assert llm_client.is_available is False

    async def test_close_when_not_initialized(self, llm_client):
        # Should not raise
        await llm_client.close()
        assert llm_client.is_available is False

    async def test_send_message_with_tools_returns_full_response(self, llm_client):
        """send_message_with_tools returns the raw Message object, not just text."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(type="text", text="Let me check that."),
            MagicMock(
                type="tool_use",
                id="toolu_01",
                name="get_printer_state",
                input={},
            ),
        ]
        mock_response.stop_reason = "tool_use"

        mock_anthropic = AsyncMock()
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)
        llm_client._impl = AnthropicClient(llm_client._config)
        llm_client._impl._client = mock_anthropic

        result = await llm_client.send_message_with_tools(
            messages=[{"role": "user", "content": "What is the printer state?"}],
            system_prompt="You are a printer assistant",
            tools=[
                {
                    "name": "get_printer_state",
                    "description": "Get state",
                    "input_schema": {"type": "object"},
                }
            ],
        )

        assert result is mock_response
        assert result.stop_reason == "tool_use"
        mock_anthropic.messages.create.assert_called_once()
        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["tools"] == [
            {
                "name": "get_printer_state",
                "description": "Get state",
                "input_schema": {"type": "object"},
            }
        ]

    async def test_send_message_with_tools_without_init_raises(self, llm_client):
        """send_message_with_tools raises RuntimeError when client is not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            await llm_client.send_message_with_tools(
                messages=[{"role": "user", "content": "hi"}],
                system_prompt="test",
            )

    async def test_send_message_with_tools_no_tools_omits_key(self, llm_client):
        """When tools=None, the 'tools' key should not be passed to the API."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="OK")]
        mock_response.stop_reason = "end_turn"

        mock_anthropic = AsyncMock()
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)
        llm_client._impl = AnthropicClient(llm_client._config)
        llm_client._impl._client = mock_anthropic

        await llm_client.send_message_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            system_prompt="test",
            tools=None,
        )

        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert "tools" not in call_kwargs

    async def test_send_message_with_tools_overrides_params(self, llm_client):
        """max_tokens and temperature overrides are forwarded to the API."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="OK")]

        mock_anthropic = AsyncMock()
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)
        llm_client._impl = AnthropicClient(llm_client._config)
        llm_client._impl._client = mock_anthropic

        await llm_client.send_message_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            system_prompt="test",
            max_tokens=4096,
            temperature=0.0,
        )

        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 4096
        assert call_kwargs["temperature"] == 0.0

    async def test_send_message_extracts_text_skipping_thinking_block(self, llm_client):
        """send_message finds the TextBlock even when ThinkingBlock comes first."""
        thinking_block = MagicMock()
        thinking_block.type = "thinking"
        del thinking_block.text  # ThinkingBlock has no .text attribute

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "The actual answer"

        mock_response = MagicMock()
        mock_response.content = [thinking_block, text_block]

        mock_anthropic = AsyncMock()
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)
        llm_client._impl = AnthropicClient(llm_client._config)
        llm_client._impl._client = mock_anthropic

        result = await llm_client.send_message(
            messages=[{"role": "user", "content": "think hard"}],
            system_prompt="test",
        )

        assert result == "The actual answer"

    async def test_send_message_empty_content_returns_empty_string(self, llm_client):
        """send_message returns empty string when response content is empty."""
        mock_response = MagicMock()
        mock_response.content = []

        mock_anthropic = AsyncMock()
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)
        llm_client._impl = AnthropicClient(llm_client._config)
        llm_client._impl._client = mock_anthropic

        result = await llm_client.send_message(
            messages=[{"role": "user", "content": "hi"}],
            system_prompt="test",
        )

        assert result == ""


class TestOllamaClient:
    """Tests for Ollama client"""

    async def test_ollama_initialization(self, ollama_config):
        client = OllamaClient(ollama_config)
        await client.initialize()
        assert client.is_available is True
        assert client._base_url == "http://localhost:11434/v1"
        await client.close()

    async def test_ollama_convert_messages(self, ollama_config):
        client = OllamaClient(ollama_config)
        openai_messages = client._convert_messages_to_openai(
            [{"role": "user", "content": "Hello"}], "You are a helpful assistant"
        )
        assert len(openai_messages) == 2
        assert openai_messages[0]["role"] == "system"
        assert openai_messages[0]["content"] == "You are a helpful assistant"
        assert openai_messages[1]["role"] == "user"

    async def test_ollama_convert_tools(self, ollama_config):
        client = OllamaClient(ollama_config)
        anthropic_tools = [
            {
                "name": "get_printer_state",
                "description": "Get printer state",
                "input_schema": {"type": "object"},
            }
        ]
        openai_tools = client._convert_tools_to_openai(anthropic_tools)
        assert len(openai_tools) == 1
        assert openai_tools[0]["type"] == "function"
        assert openai_tools[0]["function"]["name"] == "get_printer_state"
