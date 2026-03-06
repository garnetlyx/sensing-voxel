"""Tests for LLM chat session and manager"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from sensing_voxel.llm.chat import ChatManager, ChatSession, MAX_TELEGRAM_MESSAGE_LENGTH


class TestChatSession:
    def test_add_user_message(self):
        session = ChatSession(user_id=123, max_turns=10)
        session.add_user_message("hello")
        assert len(session.messages) == 1
        assert session.messages[0] == {"role": "user", "content": "hello"}

    def test_add_assistant_message(self):
        session = ChatSession(user_id=123, max_turns=10)
        session.add_assistant_message("hi there")
        assert len(session.messages) == 1
        assert session.messages[0] == {"role": "assistant", "content": "hi there"}

    def test_get_messages_returns_copy(self):
        session = ChatSession(user_id=123, max_turns=10)
        session.add_user_message("hello")
        msgs = session.get_messages()
        msgs.append({"role": "user", "content": "extra"})
        assert len(session.messages) == 1  # Original unchanged

    def test_history_trimming(self):
        session = ChatSession(user_id=123, max_turns=2)
        # Add 3 turns (6 messages) — should trim to 2 turns (4 messages)
        for i in range(3):
            session.add_user_message(f"user msg {i}")
            session.add_assistant_message(f"assistant msg {i}")

        assert len(session.messages) == 4
        assert session.messages[0]["content"] == "user msg 1"
        assert session.messages[-1]["content"] == "assistant msg 2"

    def test_clear(self):
        session = ChatSession(user_id=123, max_turns=10)
        session.add_user_message("hello")
        session.add_assistant_message("hi")
        session.clear()
        assert len(session.messages) == 0


class TestChatManager:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.send_message = AsyncMock(return_value="AI response")
        client.is_available = True
        return client

    @pytest.fixture
    def mock_context_collector(self):
        collector = MagicMock()
        collector.format_system_prompt.return_value = "System prompt with context"
        return collector

    @pytest.fixture
    def chat_manager(self, mock_client, mock_context_collector):
        return ChatManager(
            client=mock_client,
            context_collector=mock_context_collector,
            max_turns=20,
        )

    async def test_chat_creates_session(self, chat_manager):
        await chat_manager.chat(user_id=123, message="hello")
        assert chat_manager.get_session_count() == 1

    async def test_chat_returns_response(self, chat_manager):
        result = await chat_manager.chat(user_id=123, message="hello")
        assert result == "AI response"

    async def test_chat_calls_llm_with_context(self, chat_manager, mock_client, mock_context_collector):
        await chat_manager.chat(user_id=123, message="hello")

        mock_context_collector.format_system_prompt.assert_called_once()
        mock_client.send_message.assert_called_once()
        call_kwargs = mock_client.send_message.call_args.kwargs
        assert call_kwargs["system_prompt"] == "System prompt with context"
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["content"] == "hello"

    async def test_chat_preserves_history(self, chat_manager, mock_client):
        mock_client.send_message = AsyncMock(side_effect=["first", "second"])

        await chat_manager.chat(user_id=123, message="msg1")
        await chat_manager.chat(user_id=123, message="msg2")

        # Second call should include full history
        call_kwargs = mock_client.send_message.call_args.kwargs
        assert len(call_kwargs["messages"]) == 3  # user, assistant, user

    async def test_chat_reuses_session(self, chat_manager):
        await chat_manager.chat(user_id=123, message="first")
        await chat_manager.chat(user_id=123, message="second")
        assert chat_manager.get_session_count() == 1

    async def test_chat_separate_sessions_per_user(self, chat_manager):
        await chat_manager.chat(user_id=123, message="hello")
        await chat_manager.chat(user_id=456, message="hi")
        assert chat_manager.get_session_count() == 2

    async def test_chat_truncates_long_response(self, chat_manager, mock_client):
        long_response = "x" * 5000
        mock_client.send_message = AsyncMock(return_value=long_response)

        result = await chat_manager.chat(user_id=123, message="hello")
        assert len(result) <= MAX_TELEGRAM_MESSAGE_LENGTH
        assert result.endswith("...")

    def test_clear_session(self, chat_manager):
        chat_manager._sessions[123] = ChatSession(user_id=123)
        chat_manager._sessions[123].add_user_message("hello")
        chat_manager.clear_session(123)
        assert len(chat_manager._sessions[123].messages) == 0

    def test_clear_nonexistent_session(self, chat_manager):
        # Should not raise
        chat_manager.clear_session(999)

    def test_session_count(self, chat_manager):
        assert chat_manager.get_session_count() == 0
        chat_manager._sessions[1] = ChatSession(user_id=1)
        chat_manager._sessions[2] = ChatSession(user_id=2)
        assert chat_manager.get_session_count() == 2


class TestChatManagerToolUse:
    """Tests for the agentic tool-use loop in ChatManager."""

    @pytest.fixture
    def mock_tool_client(self):
        """Client mock that returns full Message objects (for send_message_with_tools)."""
        client = MagicMock()
        client.is_available = True
        client.send_message = AsyncMock(return_value="simple response")
        client.send_message_with_tools = AsyncMock()
        return client

    @pytest.fixture
    def mock_tool_executor(self):
        executor = MagicMock()
        executor.execute = AsyncMock(return_value="tool output")
        return executor

    @pytest.fixture
    def mock_context_collector(self):
        collector = MagicMock()
        collector.format_system_prompt.return_value = "System prompt"
        return collector

    @pytest.fixture
    def chat_manager_with_tools(self, mock_tool_client, mock_context_collector, mock_tool_executor):
        manager = ChatManager(
            client=mock_tool_client,
            context_collector=mock_context_collector,
            max_turns=20,
            max_tool_iterations=3,
        )
        # Inject tool executor and tools directly
        manager._tool_executor = mock_tool_executor
        manager._tools = [{"name": "get_printer_state", "description": "test"}]
        return manager

    async def test_tool_use_single_round(self, chat_manager_with_tools, mock_tool_client, mock_tool_executor):
        """LLM calls a tool, gets the result, then produces a final text response."""
        # First response: tool_use
        tool_response = MagicMock()
        tool_use_block = MagicMock()
        tool_use_block.type = "tool_use"
        tool_use_block.id = "toolu_01"
        tool_use_block.name = "get_printer_state"
        tool_use_block.input = {}
        tool_response.content = [tool_use_block]
        tool_response.stop_reason = "tool_use"

        # Second response: final text
        final_response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "The printer is idle."
        final_response.content = [text_block]
        final_response.stop_reason = "end_turn"

        mock_tool_client.send_message_with_tools = AsyncMock(
            side_effect=[tool_response, final_response]
        )

        result = await chat_manager_with_tools.chat(user_id=1, message="What is the printer state?")

        assert result == "The printer is idle."
        assert mock_tool_executor.execute.call_count == 1
        mock_tool_executor.execute.assert_called_with("get_printer_state", {})

    async def test_tool_use_multiple_rounds(self, chat_manager_with_tools, mock_tool_client, mock_tool_executor):
        """LLM calls tools across multiple iterations before producing final text."""
        tool_block_1 = MagicMock(type="tool_use", id="toolu_01", name="get_printer_state", input={})
        tool_response_1 = MagicMock(content=[tool_block_1], stop_reason="tool_use")

        tool_block_2 = MagicMock(type="tool_use", id="toolu_02", name="get_printer_state", input={})
        tool_response_2 = MagicMock(content=[tool_block_2], stop_reason="tool_use")

        text_block = MagicMock(type="text", text="All done.")
        final_response = MagicMock(content=[text_block], stop_reason="end_turn")

        mock_tool_client.send_message_with_tools = AsyncMock(
            side_effect=[tool_response_1, tool_response_2, final_response]
        )

        result = await chat_manager_with_tools.chat(user_id=1, message="Do two checks")

        assert result == "All done."
        assert mock_tool_executor.execute.call_count == 2

    async def test_max_iterations_exhaustion(self, chat_manager_with_tools, mock_tool_client, mock_tool_executor):
        """When the LLM keeps calling tools past max_tool_iterations, return exhaustion message."""
        # All responses are tool_use — will never finish
        def make_tool_response():
            block = MagicMock(type="tool_use", id="toolu_xx", name="get_printer_state", input={})
            resp = MagicMock(content=[block], stop_reason="tool_use")
            return resp

        mock_tool_client.send_message_with_tools = AsyncMock(
            side_effect=[make_tool_response() for _ in range(5)]
        )

        result = await chat_manager_with_tools.chat(user_id=1, message="infinite loop")

        assert "maximum number of tool calls" in result
        # max_tool_iterations is 3, so exactly 3 iterations should run
        assert mock_tool_client.send_message_with_tools.call_count == 3
        assert mock_tool_executor.execute.call_count == 3

    async def test_tool_use_with_image_content_result(self, chat_manager_with_tools, mock_tool_client, mock_tool_executor):
        """Tool executor returning dict with 'content' key passes content blocks to session."""
        tool_block = MagicMock(type="tool_use", id="toolu_01", name="take_snapshot", input={})
        tool_response = MagicMock(content=[tool_block], stop_reason="tool_use")

        text_block = MagicMock(type="text", text="I can see the bed.")
        final_response = MagicMock(content=[text_block], stop_reason="end_turn")

        mock_tool_client.send_message_with_tools = AsyncMock(
            side_effect=[tool_response, final_response]
        )

        # Return dict with "content" key (image + text content blocks)
        mock_tool_executor.execute = AsyncMock(return_value={
            "content": [
                {"type": "image", "source": {"data": "base64data"}},
                {"type": "text", "text": "Camera snapshot"},
            ]
        })

        result = await chat_manager_with_tools.chat(user_id=1, message="Take a photo")

        assert result == "I can see the bed."
        # Verify tool result was added to session with content blocks
        session = chat_manager_with_tools._sessions[1]
        # Find the tool_result message
        tool_result_msgs = [
            m for m in session.messages
            if m["role"] == "user" and isinstance(m["content"], list)
            and any(b.get("type") == "tool_result" for b in m["content"])
        ]
        assert len(tool_result_msgs) == 1
        tool_result_content = tool_result_msgs[0]["content"][0]["content"]
        assert isinstance(tool_result_content, list)
        assert tool_result_content[0]["type"] == "image"


class TestChatManagerSerializeContent:
    """Tests for _serialize_content and _extract_text static methods."""

    def test_serialize_text_block(self):
        """Text blocks are serialized to dict with type and text."""
        block = MagicMock(type="text", text="Hello world")
        result = ChatManager._serialize_content([block])
        assert result == [{"type": "text", "text": "Hello world"}]

    def test_serialize_tool_use_block(self):
        """Tool use blocks are serialized with id, name, and input."""
        block = MagicMock(type="tool_use", id="toolu_01", input={"key": "value"})
        # MagicMock reserves 'name' kwarg — must set it as an attribute
        block.name = "get_printer_state"
        result = ChatManager._serialize_content([block])
        assert result == [{
            "type": "tool_use",
            "id": "toolu_01",
            "name": "get_printer_state",
            "input": {"key": "value"},
        }]

    def test_serialize_thinking_block(self):
        """Thinking blocks are serialized with the thinking text."""
        block = MagicMock(type="thinking", thinking="Let me think about this...")
        result = ChatManager._serialize_content([block])
        assert result == [{"type": "thinking", "thinking": "Let me think about this..."}]

    def test_serialize_mixed_content(self):
        """Multiple block types in a single response are all serialized."""
        thinking = MagicMock(type="thinking", thinking="Hmm...")
        text = MagicMock(type="text", text="Here is the answer")
        tool = MagicMock(type="tool_use", id="t1", name="foo", input={})

        result = ChatManager._serialize_content([thinking, text, tool])
        assert len(result) == 3
        assert result[0]["type"] == "thinking"
        assert result[1]["type"] == "text"
        assert result[2]["type"] == "tool_use"

    def test_serialize_unknown_block_type_ignored(self):
        """Unknown block types are silently ignored."""
        block = MagicMock(type="unknown_new_type")
        result = ChatManager._serialize_content([block])
        assert result == []

    def test_extract_text_from_content(self):
        """_extract_text returns the first text block."""
        text_block = MagicMock()
        text_block.text = "The answer is 42"
        tool_block = MagicMock(spec=[])  # No .text attribute

        result = ChatManager._extract_text([tool_block, text_block])
        assert result == "The answer is 42"

    def test_extract_text_empty_content(self):
        """_extract_text returns empty string when no text blocks present."""
        result = ChatManager._extract_text([])
        assert result == ""

    def test_extract_text_no_text_blocks(self):
        """_extract_text returns empty string when content has no text blocks."""
        block = MagicMock(spec=[])  # No .text attribute
        result = ChatManager._extract_text([block])
        assert result == ""


class TestChatSessionToolResult:
    """Tests for add_tool_result and trimming with tool messages."""

    def test_add_tool_result(self):
        session = ChatSession(user_id=1, max_turns=10)
        session.add_tool_result("toolu_01", "Tool output text")

        assert len(session.messages) == 1
        msg = session.messages[0]
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        assert msg["content"][0]["type"] == "tool_result"
        assert msg["content"][0]["tool_use_id"] == "toolu_01"
        assert msg["content"][0]["content"] == "Tool output text"

    def test_add_tool_result_with_list_content(self):
        """Tool result can contain a list of content blocks (e.g. image + text)."""
        session = ChatSession(user_id=1, max_turns=10)
        content_blocks = [
            {"type": "image", "source": {"data": "base64"}},
            {"type": "text", "text": "snapshot"},
        ]
        session.add_tool_result("toolu_02", content_blocks)

        assert session.messages[0]["content"][0]["content"] == content_blocks

    def test_trim_preserves_tool_result_pairs(self):
        """Trimming should not orphan tool_result messages — it finds a safe user text start."""
        session = ChatSession(user_id=1, max_turns=2)

        # Turn 1: user (text) + assistant
        session.add_user_message("start")
        session.add_assistant_message("ok")

        # Turn 2: assistant (tool_use blocks) + tool_result + assistant (final)
        session.add_assistant_message([{"type": "tool_use", "id": "t1", "name": "foo", "input": {}}])
        session.add_tool_result("t1", "result")
        session.add_user_message("next question")
        session.add_assistant_message("final answer")

        # After trimming, messages should start with a plain user text message
        first_msg = session.messages[0]
        assert first_msg["role"] == "user"
        assert isinstance(first_msg["content"], str)
