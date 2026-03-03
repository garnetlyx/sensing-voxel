"""
Chat session manager for AI conversations
"""

from __future__ import annotations

import logging

from sensing_voxel.llm.client import LlmClient
from sensing_voxel.llm.context import PrinterContextCollector

logger = logging.getLogger(__name__)

# Telegram message limit
MAX_TELEGRAM_MESSAGE_LENGTH = 4000


class ChatSession:
    """Per-user chat session with conversation history."""

    def __init__(self, user_id: int, max_turns: int = 20):
        self.user_id = user_id
        self.max_turns = max_turns
        self.messages: list[dict] = []

    def add_user_message(self, content: str) -> None:
        """Add user message and trim history if needed."""
        self.messages.append({"role": "user", "content": content})
        self._trim()

    def add_assistant_message(self, content: str) -> None:
        """Add assistant response."""
        self.messages.append({"role": "assistant", "content": content})
        self._trim()

    def get_messages(self) -> list[dict]:
        """Get message history for API call."""
        return list(self.messages)

    def clear(self) -> None:
        """Clear conversation history."""
        self.messages.clear()

    def _trim(self) -> None:
        """Trim oldest messages to stay within max_turns (keeping pairs)."""
        # Each turn = 1 user + 1 assistant message = 2 entries
        max_entries = self.max_turns * 2
        if len(self.messages) > max_entries:
            self.messages = self.messages[-max_entries:]


class ChatManager:
    """Manages chat sessions across users."""

    def __init__(
        self,
        client: LlmClient,
        context_collector: PrinterContextCollector,
        max_turns: int = 20,
    ):
        self._client = client
        self._context_collector = context_collector
        self._max_turns = max_turns
        self._sessions: dict[int, ChatSession] = {}

    async def chat(self, user_id: int, message: str) -> str:
        """Process user message and return AI response.

        Args:
            user_id: Telegram user ID
            message: User's message text

        Returns:
            AI response text (truncated to Telegram limit)
        """
        session = self._get_or_create_session(user_id)
        session.add_user_message(message)

        # Build system prompt with live context
        system_prompt = self._context_collector.format_system_prompt()

        # Call LLM
        response = await self._client.send_message(
            messages=session.get_messages(),
            system_prompt=system_prompt,
        )

        session.add_assistant_message(response)

        # Truncate for Telegram
        if len(response) > MAX_TELEGRAM_MESSAGE_LENGTH:
            response = response[: MAX_TELEGRAM_MESSAGE_LENGTH - 3] + "..."

        return response

    def clear_session(self, user_id: int) -> None:
        """Clear user's conversation history."""
        if user_id in self._sessions:
            self._sessions[user_id].clear()

    def get_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self._sessions)

    def _get_or_create_session(self, user_id: int) -> ChatSession:
        """Get existing session or create new one."""
        if user_id not in self._sessions:
            self._sessions[user_id] = ChatSession(user_id, self._max_turns)
        return self._sessions[user_id]
