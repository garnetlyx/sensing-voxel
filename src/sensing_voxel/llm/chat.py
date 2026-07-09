"""
Chat session manager for AI conversations with tool use
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from sensing_voxel.adapters.base import PrinterAdapter
from sensing_voxel.llm.client import LlmClient
from sensing_voxel.llm.context import PrinterContextCollector
from sensing_voxel.llm.runtime import (
    BackgroundTaskManager,
    ChatRequestContext,
    PendingApprovalStore,
    ToolAuditLogger,
    estimate_token_count,
    summarize_messages,
)

logger = logging.getLogger(__name__)

# Telegram message limit
MAX_TELEGRAM_MESSAGE_LENGTH = 4000
MAX_TOOL_PROGRESS_MESSAGE_LENGTH = 900
SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "auth",
    "bot_token",
    "credential",
    "password",
    "secret",
    "token",
)
_URL_CREDENTIALS_RE = re.compile(r"([a-zA-Z][a-zA-Z0-9+.-]*://)([^/\s:@]+):([^@\s/]+)@")

ToolProgressCallback = Callable[[str], Awaitable[None]]


class ChatSession:
    """Per-user chat session with conversation history."""

    def __init__(self, user_id: str, max_turns: int = 20):
        self.user_id = user_id
        self.max_turns = max_turns
        self.messages: list[dict] = []
        self.summary: str | None = None
        self.last_compacted_at: str | None = None

    def to_dict(self) -> dict:
        """Convert session to serializable dict."""
        return {
            "user_id": self.user_id,
            "max_turns": self.max_turns,
            "messages": self.messages,
            "summary": self.summary,
            "last_compacted_at": self.last_compacted_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ChatSession:
        """Create session from dict."""
        session = cls(data["user_id"], data.get("max_turns", 20))
        session.messages = data.get("messages", [])
        session.summary = data.get("summary")
        session.last_compacted_at = data.get("last_compacted_at")
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
        self.summary = None
        self.last_compacted_at = None

    def compact_if_needed(self, context_window_tokens: int) -> bool:
        """Compact older turns into a summary when the estimate crosses threshold."""
        if estimate_token_count(self.messages, self.summary) <= int(context_window_tokens * 0.7):
            return False

        keep_messages = self.messages[-12:]
        old_messages = self.messages[:-12]
        if not old_messages:
            return False

        new_summary = summarize_messages(old_messages)
        if self.summary:
            self.summary = f"{self.summary}\n{new_summary}"[:2400]
        else:
            self.summary = new_summary
        self.messages = keep_messages
        self.last_compacted_at = datetime.now().isoformat()
        return True

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
        adapters: dict[str, PrinterAdapter] | None = None,
        adapter: PrinterAdapter | None = None,
        active_printer: str | None = None,
        camera=None,
        rtsp_url: str | None = None,
        shell=None,
        web_search=None,
        max_turns: int = 20,
        max_tool_iterations: int = 5,
        context_window_tokens: int = 32000,
        history_path: Path | None = None,
        audit_log_path: Path | None = None,
        allowed_gcode_commands: set[str] | None = None,
        verbose_tool_use: bool = True,
    ):
        self._client = client
        self._context_collector = context_collector
        self._adapter = adapter
        self._adapters = adapters or ({active_printer or "default": adapter} if adapter else {})
        self._active_printer = active_printer or next(iter(self._adapters), None)
        self._camera = camera
        self._max_turns = max_turns
        self._max_tool_iterations = max_tool_iterations
        self._context_window_tokens = context_window_tokens
        self._verbose_tool_use = verbose_tool_use
        self._history_path = history_path or Path("data/chat_history.json")
        self._audit_logger = ToolAuditLogger(audit_log_path or Path("data/tool_audit.jsonl"))
        self._background_tasks = BackgroundTaskManager()
        self._approval_store = PendingApprovalStore()
        self._sessions: dict[str, ChatSession] = {}

        # Create tool executor when printer, camera, or system tools are available
        self._tool_executor = None
        self._tools = None
        if self._adapters or camera or rtsp_url or shell or web_search:
            from sensing_voxel.llm.tools import (
                CAMERA_TOOLS,
                PRINTER_TOOLS,
                SYSTEM_TOOLS,
                ToolExecutor,
            )

            self._tool_executor = ToolExecutor(
                adapters=self._adapters,
                active_printer=self._active_printer,
                camera=camera,
                rtsp_url=rtsp_url,
                shell=shell,
                web_search=web_search,
                audit_logger=self._audit_logger,
                background_tasks=self._background_tasks,
                approval_store=self._approval_store,
                allowed_gcode_commands=allowed_gcode_commands,
            )
            self._tools = list(PRINTER_TOOLS) if self._adapters else []
            if getattr(self._tool_executor._shell_config, "enabled", False):
                self._tools.extend([t for t in SYSTEM_TOOLS if t["name"] == "run_bash"])
            if getattr(self._tool_executor._web_search_config, "enabled", False):
                self._tools.extend([t for t in SYSTEM_TOOLS if t["name"] == "web_search"])
            if camera or rtsp_url:
                self._tools.extend(CAMERA_TOOLS)

    def set_photo_sender(self, sender) -> None:
        """Attach a channel-specific photo sender after construction."""
        if self._tool_executor is not None:
            self._tool_executor._photo_sender = sender

    def set_text_notifier(self, sender) -> None:
        """Attach a channel-aware text notifier for background task callbacks."""
        if self._tool_executor is not None:
            self._tool_executor._text_notifier = sender

    async def chat(
        self,
        logical_user_id: str | int | None = None,
        message: str = "",
        request_context: ChatRequestContext | None = None,
        user_id: str | int | None = None,
        tool_progress_callback: ToolProgressCallback | None = None,
    ) -> str:
        """Process user message and return AI response.

        If tools are available, runs an agentic loop: the LLM can call
        tools, observe results, and continue until it produces a final
        text response or hits the iteration limit.
        """
        identity = logical_user_id if logical_user_id is not None else user_id
        if identity is None:
            raise ValueError("logical_user_id or user_id is required")
        user_key = str(identity)
        request_context = request_context or ChatRequestContext(
            logical_user_id=user_key,
            channel="unknown",
        )
        session = self._get_or_create_session(identity)
        session.add_user_message(message)
        session.compact_if_needed(self._context_window_tokens)

        system_prompt = self._context_collector.format_system_prompt(
            request_context=request_context,
            chat_summary=session.summary,
        )

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
                    await self._emit_tool_progress(
                        tool_progress_callback,
                        self._format_tool_start(block_name, block_input),
                    )
                    result = await self._tool_executor.execute(
                        block_name,
                        block_input,
                        request_context,
                    )
                    await self._emit_tool_progress(
                        tool_progress_callback,
                        self._format_tool_result(block_name, result),
                    )
                    if isinstance(result, dict) and "content" in result:
                        logger.info(f"Tool result: [image + text content blocks]")
                        session.add_tool_result(block_id, result["content"])
                    else:
                        logger.info(f"Tool result: {result[:200]}")
                        session.add_tool_result(block_id, result)

        # Exhausted iterations
        await self._emit_tool_progress(
            tool_progress_callback,
            "⚠️ Tool loop stopped: reached maximum tool-call iterations.",
        )
        return self._truncate(
            "I've reached the maximum number of tool calls for this turn. "
            "Please try again or simplify your request."
        )

    async def _emit_tool_progress(
        self, callback: ToolProgressCallback | None, message: str
    ) -> None:
        """Emit user-visible tool progress without breaking the main chat."""
        if not self._verbose_tool_use or callback is None:
            return
        try:
            await callback(self._truncate(message))
        except Exception as e:
            logger.warning(f"Failed to emit tool progress: {e}")

    @classmethod
    def _format_tool_start(cls, name: str, tool_input: Any) -> str:
        safe_input = cls._safe_preview(tool_input, max_chars=500)
        return f"🔧 Tool use: {name}\ninput: {safe_input}"

    @classmethod
    def _format_tool_result(cls, name: str, result: Any) -> str:
        if isinstance(result, dict) and isinstance(result.get("content"), list):
            return f"✅ Tool result: {name} returned content blocks (image/text)."

        parsed = None
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
            except Exception:
                parsed = None

        if isinstance(parsed, dict) and parsed.get("error"):
            preview = cls._safe_preview(parsed["error"], max_chars=500)
            return f"❌ Tool result: {name} error\n{preview}"
        if isinstance(parsed, dict):
            preview = cls._safe_preview(parsed, max_chars=650)
            return f"✅ Tool result: {name}\n{preview}"

        preview = cls._safe_preview(result, max_chars=650)
        return f"✅ Tool result: {name}\n{preview}"

    @classmethod
    def _safe_preview(cls, value: Any, max_chars: int = MAX_TOOL_PROGRESS_MESSAGE_LENGTH) -> str:
        safe_value = cls._redact(value)
        try:
            if isinstance(safe_value, str):
                text = safe_value
            else:
                text = json.dumps(safe_value, ensure_ascii=False, sort_keys=True)
        except Exception:
            text = str(safe_value)
        text = cls._redact_text(text).replace("\x00", "")
        if len(text) > max_chars:
            return text[: max_chars - 3] + "..."
        return text

    @classmethod
    def _redact(cls, value: Any) -> Any:
        if isinstance(value, dict):
            redacted = {}
            for key, item in value.items():
                key_text = str(key).lower()
                if any(part in key_text for part in SENSITIVE_KEY_PARTS):
                    redacted[key] = "<redacted>"
                else:
                    redacted[key] = cls._redact(item)
            return redacted
        if isinstance(value, list):
            return [cls._redact(item) for item in value[:20]]
        if isinstance(value, tuple):
            return tuple(cls._redact(item) for item in value[:20])
        if isinstance(value, str):
            return cls._redact_text(value)
        return value

    @staticmethod
    def _redact_text(text: str) -> str:
        text = _URL_CREDENTIALS_RE.sub(r"\1<redacted>:<redacted>@", text)
        text = re.sub(
            r'(?i)("(?:api[_-]?key|bot[_-]?token|password|secret|token|auth)"\s*:\s*")[^"]+("?)',
            r"\1<redacted>\2",
            text,
        )
        return text

    def clear_session(self, user_id: str | int) -> None:
        """Clear user's conversation history."""
        user_key = str(user_id)
        if user_id in self._sessions:
            self._sessions[user_id].clear()
            self.save_history()
            return
        if user_key in self._sessions:
            self._sessions[user_key].clear()
            self.save_history()

    def get_history(self, user_id: str | int) -> list[dict]:
        """Get message history for a user."""
        session = self._get_or_create_session(user_id)
        return session.get_messages()

    def get_context_status(
        self,
        user_id: str | int,
        request_context: ChatRequestContext | None = None,
    ) -> dict[str, Any]:
        """Expose session/context diagnostics for the UI and bot."""
        user_key = str(user_id)
        session = self._get_or_create_session(user_key)
        return {
            "logical_user_id": user_key,
            "channel": request_context.channel if request_context else None,
            "active_printer": (
                request_context.active_printer
                if request_context and request_context.active_printer
                else getattr(self._context_collector, "_active_printer", None)
            ),
            "model": getattr(getattr(self._client, "_config", None), "model", None),
            "context_window_tokens": self._context_window_tokens,
            "estimated_tokens": estimate_token_count(session.messages, session.summary),
            "message_count": len(session.messages),
            "last_compacted_at": session.last_compacted_at,
            "summary": session.summary,
        }

    def get_audit_records(self, limit: int = 50) -> list[dict[str, Any]]:
        """Read recent tool audit records."""
        return self._audit_logger.tail(limit=limit)

    def get_background_tasks(
        self,
        logical_user_id: str | int,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Expose recent background tasks for a logical user."""
        return self._background_tasks.tail(str(logical_user_id), limit=limit)

    def get_pending_approvals(
        self,
        logical_user_id: str | int,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Expose pending approvals for a logical user."""
        return self._approval_store.tail(str(logical_user_id), limit=limit, status="pending")

    def save_history(self, path: Path | str | None = None) -> None:
        """Save all sessions to disk."""
        history_path = Path(path) if path is not None else self._history_path
        try:
            data = {str(uid): s.to_dict() for uid, s in self._sessions.items()}
            history_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = history_path.with_suffix(f"{history_path.suffix}.tmp")
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            tmp_path.replace(history_path)
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
                self._sessions[str(uid_str)] = ChatSession.from_dict(s_data)
            logger.info(f"Loaded {len(self._sessions)} chat sessions from {history_path}")
        except Exception as e:
            logger.error(f"Failed to load chat history: {e}")
            try:
                corrupt_path = history_path.with_suffix(
                    f"{history_path.suffix}.corrupt-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                )
                shutil.move(str(history_path), str(corrupt_path))
                logger.warning(f"Moved corrupt chat history to {corrupt_path}")
            except Exception as move_error:
                logger.error(f"Failed to quarantine corrupt chat history: {move_error}")

    def get_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self._sessions)

    def _get_or_create_session(self, user_id: str | int) -> ChatSession:
        """Get existing session or create new one."""
        user_key = str(user_id)
        if user_id in self._sessions:
            return self._sessions[user_id]
        if isinstance(user_id, int) and user_key not in self._sessions:
            self._sessions[user_id] = ChatSession(user_id, self._max_turns)
            return self._sessions[user_id]
        if user_key not in self._sessions:
            self._sessions[user_key] = ChatSession(user_key, self._max_turns)
        return self._sessions[user_key]

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
