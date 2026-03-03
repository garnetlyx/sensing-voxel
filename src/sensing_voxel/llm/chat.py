"""
Chat session manager for AI conversations with tool use
"""

from __future__ import annotations

import logging
from typing import Any

from sensing_voxel.adapters.base import PrinterAdapter
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

    def add_assistant_message(self, content: Any) -> None:
        """Add assistant response (str or list of content blocks)."""
        if isinstance(content, str):
            self.messages.append({"role": "assistant", "content": content})
        else:
            # Content blocks from tool-use response
            self.messages.append({"role": "assistant", "content": content})
        self._trim()

    def add_tool_result(self, tool_use_id: str, result: str) -> None:
        """Add tool result message."""
        self.messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result,
                }
            ],
        })

    def get_messages(self) -> list[dict]:
        """Get message history for API call."""
        return list(self.messages)

    def clear(self) -> None:
        """Clear conversation history."""
        self.messages.clear()

    def _trim(self) -> None:
        """Trim oldest messages to stay within max_turns.

        Ensures we never orphan tool_result messages (they must follow
        an assistant message containing the corresponding tool_use block).
        """
        max_entries = self.max_turns * 2
        if len(self.messages) <= max_entries:
            return

        # Find a safe trim point — don't start with a tool_result or assistant
        start = len(self.messages) - max_entries
        while start < len(self.messages):
            msg = self.messages[start]
            # Safe to start with a plain user message (string content)
            if msg["role"] == "user" and isinstance(msg["content"], str):
                break
            start += 1

        if start >= len(self.messages):
            # Fallback: keep everything
            return

        self.messages = self.messages[start:]


class ChatManager:
    """Manages chat sessions across users with optional tool use."""

    def __init__(
        self,
        client: LlmClient,
        context_collector: PrinterContextCollector,
        adapter: PrinterAdapter | None = None,
        max_turns: int = 20,
        max_tool_iterations: int = 5,
    ):
        self._client = client
        self._context_collector = context_collector
        self._adapter = adapter
        self._max_turns = max_turns
        self._max_tool_iterations = max_tool_iterations
        self._sessions: dict[int, ChatSession] = {}

        # Create tool executor if adapter is available
        self._tool_executor = None
        self._tools = None
        if adapter:
            from sensing_voxel.llm.tools import PRINTER_TOOLS, ToolExecutor

            self._tool_executor = ToolExecutor(adapter)
            self._tools = PRINTER_TOOLS

    async def chat(self, user_id: int, message: str) -> str:
        """Process user message and return AI response.

        If tools are available, runs an agentic loop: the LLM can call
        tools, observe results, and continue until it produces a final
        text response or hits the iteration limit.
        """
        session = self._get_or_create_session(user_id)
        session.add_user_message(message)

        system_prompt = self._context_collector.format_system_prompt()

        # No tools — simple path
        if not self._tool_executor or not self._tools:
            response = await self._client.send_message(
                messages=session.get_messages(),
                system_prompt=system_prompt,
            )
            session.add_assistant_message(response)
            return self._truncate(response)

        # Agentic tool-use loop
        for _iteration in range(self._max_tool_iterations):
            response = await self._client.send_message_with_tools(
                messages=session.get_messages(),
                system_prompt=system_prompt,
                tools=self._tools,
            )

            # Serialize content blocks for session history
            content_blocks = self._serialize_content(response.content)
            session.add_assistant_message(content_blocks)

            # If no tool use, we're done
            if response.stop_reason != "tool_use":
                text = self._extract_text(response.content)
                return self._truncate(text)

            # Execute each tool call and append results
            for block in response.content:
                if block.type == "tool_use":
                    logger.info(f"Tool call: {block.name}({block.input})")
                    result = await self._tool_executor.execute(block.name, block.input)
                    logger.info(f"Tool result: {result[:200]}")
                    session.add_tool_result(block.id, result)

        # Exhausted iterations
        return self._truncate(
            "I've reached the maximum number of tool calls for this turn. "
            "Please try again or simplify your request."
        )

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

    @staticmethod
    def _serialize_content(content) -> list[dict]:
        """Convert SDK content blocks to serializable dicts for message history."""
        blocks = []
        for block in content:
            if block.type == "text":
                blocks.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                blocks.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
            elif block.type == "thinking":
                blocks.append({"type": "thinking", "thinking": block.thinking})
        return blocks

    @staticmethod
    def _extract_text(content) -> str:
        """Extract text from response content blocks."""
        for block in content:
            if hasattr(block, "text"):
                return block.text
        return ""

    @staticmethod
    def _truncate(text: str) -> str:
        """Truncate for Telegram message limit."""
        if len(text) > MAX_TELEGRAM_MESSAGE_LENGTH:
            return text[: MAX_TELEGRAM_MESSAGE_LENGTH - 3] + "..."
        return text
