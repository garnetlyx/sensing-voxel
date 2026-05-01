"""
Chat session manager for AI conversations with tool use
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
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

    def to_dict(self) -> dict:
        """Convert session to serializable dict."""
        return {
            "user_id": self.user_id,
            "max_turns": self.max_turns,
            "messages": self.messages,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ChatSession:
        """Create session from dict."""
        session = cls(data["user_id"], data.get("max_turns", 20))
        session.messages = data.get("messages", [])
        return session

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

    def add_tool_result(self, tool_use_id: str, result: str | list | dict) -> None:
        """Add tool result message (str or content blocks list)."""
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": result,
                    }
                ],
            }
        )

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
        adapters: dict[str, PrinterAdapter] | None = None,
        active_printer: str | None = None,
        camera=None,
        rtsp_url: str | None = None,
        shell=None,
        web_search=None,
        max_turns: int = 20,
        max_tool_iterations: int = 5,
        history_path: Path | None = None,
    ):
        self._client = client
        self._context_collector = context_collector
        self._adapter = adapter
        self._adapters = adapters or ({active_printer or "default": adapter} if adapter else {})
        self._active_printer = active_printer or next(iter(self._adapters), None)
        self._camera = camera
        self._max_turns = max_turns
        self._max_tool_iterations = max_tool_iterations
        self._history_path = history_path or Path("data/chat_history.json")
        self._sessions: dict[int, ChatSession] = {}

        # Create tool executor when printer, camera, or system tools are available
        self._tool_executor = None
        self._tools = None
        if self._adapters or camera or rtsp_url or shell or web_search:
            from sensing_voxel.llm.tools import CAMERA_TOOLS, PRINTER_TOOLS, SYSTEM_TOOLS, ToolExecutor

            self._tool_executor = ToolExecutor(
                adapters=self._adapters,
                active_printer=self._active_printer,
                camera=camera,
                rtsp_url=rtsp_url,
                shell=shell,
                web_search=web_search,
            )
            self._tools = list(PRINTER_TOOLS) if self._adapters else []
            if getattr(self._tool_executor._shell_config, "enabled", False):
                self._tools.extend([t for t in SYSTEM_TOOLS if t["name"] == "run_bash"])
            if getattr(self._tool_executor._web_search_config, "enabled", False):
                self._tools.extend([t for t in SYSTEM_TOOLS if t["name"] == "web_search"])
            if camera or rtsp_url:
                self._tools.extend(CAMERA_TOOLS)

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
            self.save_history()
            return self._truncate(response)

        # Agentic tool-use loop
        for _iteration in range(self._max_tool_iterations):
            response = await self._client.send_message_with_tools(
                messages=session.get_messages(),
                system_prompt=system_prompt,
                tools=self._tools,
            )

            response_content = (
                response.content if hasattr(response, "content") else response.get("content", [])
            )

            # Serialize content blocks for session history
            content_blocks = self._serialize_content(response_content)
            session.add_assistant_message(content_blocks)

            # If no tool use, we're done
            stop_reason = getattr(response, "stop_reason", None)
            if stop_reason is None and isinstance(response, dict):
                stop_reason = response.get("stop_reason", "end_turn")
            if stop_reason != "tool_use":
                self.save_history()
                text = self._extract_text(response_content)
                return self._truncate(text)

            # Execute each tool call and append results
            for block in response_content:
                block_type = block.type if hasattr(block, "type") else block.get("type")
                if block_type == "tool_use":
                    block_name = block.name if hasattr(block, "name") else block.get("name")
                    block_input = block.input if hasattr(block, "input") else block.get("input")
                    block_id = block.id if hasattr(block, "id") else block.get("id")
                    logger.info(f"Tool call: {block_name}({block_input})")
                    result = await self._tool_executor.execute(block_name, block_input)
                    if isinstance(result, dict) and "content" in result:
                        logger.info(f"Tool result: [image + text content blocks]")
                        session.add_tool_result(block_id, result["content"])
                    else:
                        logger.info(f"Tool result: {result[:200]}")
                        session.add_tool_result(block_id, result)

        # Exhausted iterations
        return self._truncate(
            "I've reached the maximum number of tool calls for this turn. "
            "Please try again or simplify your request."
        )

    def clear_session(self, user_id: int) -> None:
        """Clear user's conversation history."""
        if user_id in self._sessions:
            self._sessions[user_id].clear()
            self.save_history()

    def get_history(self, user_id: int) -> list[dict]:
        """Get message history for a user."""
        session = self._get_or_create_session(user_id)
        return session.get_messages()

    def save_history(self, path: Path | str | None = None) -> None:
        """Save all sessions to disk."""
        history_path = Path(path) if path is not None else self._history_path
        try:
            data = {str(uid): s.to_dict() for uid, s in self._sessions.items()}
            history_path.parent.mkdir(parents=True, exist_ok=True)
            with history_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save chat history: {e}")

    def load_history(self, path: Path | str | None = None) -> None:
        """Load sessions from disk."""
        history_path = Path(path) if path is not None else self._history_path
        if not history_path.exists():
            return
        try:
            with history_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for uid_str, s_data in data.items():
                self._sessions[int(uid_str)] = ChatSession.from_dict(s_data)
            logger.info(f"Loaded {len(self._sessions)} chat sessions from {history_path}")
        except Exception as e:
            logger.error(f"Failed to load chat history: {e}")

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
        # Handle both Anthropic SDK objects and Ollama dict responses
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    # Ollama/OpenAI format
                    blocks.append(block)
                else:
                    # Anthropic SDK format
                    if block.type == "text":
                        blocks.append({"type": "text", "text": block.text})
                    elif block.type == "tool_use":
                        blocks.append(
                            {
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                                "input": block.input,
                            }
                        )
                    elif block.type == "thinking":
                        blocks.append({"type": "thinking", "thinking": block.thinking})
                    elif block.type == "image":
                        blocks.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": block.source.media_type,
                                    "data": block.source.data,
                                },
                            }
                        )
        return blocks

    @staticmethod
    def _extract_text(content) -> str:
        """Extract text from response content blocks."""
        for block in content:
            if hasattr(block, "text"):
                return block.text
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text", "")
        return ""

    @staticmethod
    def _truncate(text: str) -> str:
        """Truncate for Telegram message limit."""
        if len(text) > MAX_TELEGRAM_MESSAGE_LENGTH:
            return text[: MAX_TELEGRAM_MESSAGE_LENGTH - 3] + "..."
        return text
