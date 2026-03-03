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
