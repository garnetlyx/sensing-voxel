"""Runtime helpers for chat requests, session state, approvals, and background tasks."""

from __future__ import annotations

import json
import re
import shlex
import asyncio
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


def generate_request_id() -> str:
    """Generate a stable request identifier for logs and audit records."""
    return uuid4().hex


@dataclass(slots=True)
class ChatRequestContext:
    """Request metadata that should come from the transport, not the model."""

    logical_user_id: str
    channel: str
    channel_user_id: str | None = None
    chat_id: int | None = None
    active_printer: str | None = None
    request_id: str = field(default_factory=generate_request_id)


@dataclass(slots=True)
class ToolAuditRecord:
    """Structured audit record for a tool invocation."""

    request_id: str
    logical_user_id: str
    channel: str
    tool_name: str
    tool_input: dict[str, Any]
    started_at: str
    finished_at: str
    status: str
    result_preview: str
    contains_image: bool = False


class ToolAuditLogger:
    """Append-only JSONL logger for tool calls."""

    def __init__(self, path: Path | str):
        self._path = Path(path)

    def append(self, record: ToolAuditRecord) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def tail(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        lines = self._path.read_text(encoding="utf-8").splitlines()
        recent = lines[-limit:]
        return [json.loads(line) for line in recent]


@dataclass(slots=True)
class PendingApproval:
    """A high-risk action waiting for explicit approval."""

    approval_token: str
    logical_user_id: str
    channel: str
    tool_name: str
    payload: dict[str, Any]
    reason: str
    created_at: str
    status: str = "pending"
    approved_at: str | None = None


class PendingApprovalStore:
    """In-memory registry for approval-gated actions."""

    def __init__(self) -> None:
        self._records: dict[str, PendingApproval] = {}

    def create(
        self,
        logical_user_id: str,
        channel: str,
        tool_name: str,
        payload: dict[str, Any],
        reason: str,
    ) -> PendingApproval:
        token = uuid4().hex[:12]
        record = PendingApproval(
            approval_token=token,
            logical_user_id=logical_user_id,
            channel=channel,
            tool_name=tool_name,
            payload=payload,
            reason=reason,
            created_at=datetime.now().isoformat(),
        )
        self._records[token] = record
        return record

    def approve(
        self,
        token: str,
        logical_user_id: str,
        tool_name: str,
    ) -> PendingApproval | None:
        record = self._records.get(token)
        if record is None:
            return None
        if record.logical_user_id != logical_user_id or record.tool_name != tool_name:
            return None
        if record.status != "pending":
            return None
        record.status = "approved"
        record.approved_at = datetime.now().isoformat()
        return record

    def tail(
        self,
        logical_user_id: str | None = None,
        limit: int = 20,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        records = list(self._records.values())
        if logical_user_id is not None:
            records = [record for record in records if record.logical_user_id == logical_user_id]
        if status is not None:
            records = [record for record in records if record.status == status]
        records.sort(key=lambda record: record.created_at)
        return [asdict(record) for record in records[-limit:]]


@dataclass(slots=True)
class BackgroundTaskRecord:
    """Tracked background task state."""

    task_id: str
    logical_user_id: str
    channel: str
    task_name: str
    command: str
    created_at: str
    status: str = "queued"
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    timeout_seconds: int | None = None
    progress_interval_seconds: int | None = None
    completion_message: str | None = None
    cwd: str | None = None


class BackgroundTaskManager:
    """Runs long-lived shell tasks without blocking the chat loop."""

    def __init__(self) -> None:
        self._records: dict[str, BackgroundTaskRecord] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    def create_record(
        self,
        logical_user_id: str,
        channel: str,
        task_name: str,
        command: str,
        timeout_seconds: int | None = None,
        progress_interval_seconds: int | None = None,
        completion_message: str | None = None,
        cwd: str | None = None,
    ) -> BackgroundTaskRecord:
        task_id = uuid4().hex[:12]
        record = BackgroundTaskRecord(
            task_id=task_id,
            logical_user_id=logical_user_id,
            channel=channel,
            task_name=task_name,
            command=command,
            created_at=datetime.now().isoformat(),
            timeout_seconds=timeout_seconds,
            progress_interval_seconds=progress_interval_seconds,
            completion_message=completion_message,
            cwd=cwd,
        )
        self._records[task_id] = record
        return record

    def attach_task(self, task_id: str, task: asyncio.Task) -> None:
        self._tasks[task_id] = task

    def get(self, task_id: str) -> BackgroundTaskRecord | None:
        return self._records.get(task_id)

    def tail(
        self,
        logical_user_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        records = list(self._records.values())
        if logical_user_id is not None:
            records = [record for record in records if record.logical_user_id == logical_user_id]
        records.sort(key=lambda record: record.created_at)
        return [asdict(record) for record in records[-limit:]]

    def cancel(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        record = self._records.get(task_id)
        if task is None or task.done():
            return False
        task.cancel()
        if record is not None and record.status not in {"completed", "failed", "cancelled"}:
            record.status = "cancelled"
            record.finished_at = datetime.now().isoformat()
        return True


def estimate_token_count(messages: list[dict[str, Any]], summary: str | None = None) -> int:
    """Rough token estimate for message history and summary."""
    parts: list[str] = []
    if summary:
        parts.append(summary)
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        else:
            parts.append(json.dumps(content, ensure_ascii=False))
    total_chars = sum(len(part) for part in parts)
    return max(1, total_chars // 4)


def summarize_messages(messages: list[dict[str, Any]], max_chars: int = 1200) -> str:
    """Create a compact deterministic summary of older turns."""
    snippets: list[str] = []
    for message in messages:
        role = message.get("role", "unknown")
        content = message.get("content")
        if isinstance(content, str):
            text = content.strip().replace("\n", " ")
            if text:
                snippets.append(f"{role}: {text[:180]}")
        elif isinstance(content, list):
            text_parts = []
            for block in content:
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    text_parts.append(f"tool_use:{block.get('name', '')}")
                elif block.get("type") == "tool_result":
                    text_parts.append("tool_result")
            if text_parts:
                snippets.append(f"{role}: {' | '.join(text_parts)[:180]}")
        if sum(len(s) for s in snippets) >= max_chars:
            break
    return "\n".join(snippets)[:max_chars]


SAFE_SHELL_PREFIXES = {
    "sleep",
    "curl",
    "wget",
    "cat",
    "ls",
    "pwd",
    "env",
    "printenv",
    "rg",
    "grep",
    "sed",
    "awk",
    "tail",
    "head",
    "wc",
    "ps",
    "top",
    "df",
    "du",
    "uname",
    "date",
    "which",
    "ssh",
}

HIGH_RISK_SHELL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(^|\s)rm(\s|$)",
        r"(^|\s)mv(\s|$)",
        r"(^|\s)chmod(\s|$)",
        r"(^|\s)chown(\s|$)",
        r"(^|\s)sudo(\s|$)",
        r"(^|\s)apt(\s|$)",
        r"(^|\s)brew(\s|$)",
        r"(^|\s)pip(\s|$)",
        r"(^|\s)tee(\s|$)",
        r"(^|\s)scp(\s|$)",
        r"printer\.cfg",
        r"SAVE_CONFIG",
        r"FIRMWARE_RESTART",
        r"M112",
        r"G0\s+.*Z-",
        r"G1\s+.*Z-",
        r"M104\s+S2\d{2,}",
        r"M109\s+S2\d{2,}",
        r"M140\s+S1\d{2,}",
        r"M190\s+S1\d{2,}",
    )
]

LOW_RISK_GCODE_QUERY_PREFIXES = {
    "M114",
    "GET_POSITION",
    "STATUS",
    "QUERY_ENDSTOPS",
    "GCODE_LIST",
    "HELP",
}

HIGH_RISK_GCODE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(^|\s)SAVE_CONFIG(\s|$)",
        r"(^|\s)FIRMWARE_RESTART(\s|$)",
        r"(^|\s)RESTART(\s|$)",
        r"(^|\s)SET_KINEMATIC_POSITION(\s|$)",
        r"(^|\s)FORCE_MOVE(\s|$)",
        r"(^|\s)G28(\s|$)",
        r"(^|\s)G29(\s|$)",
        r"(^|\s)PROBE(\s|$)",
        r"(^|\s)(G0|G1)\s",
        r"(^|\s)M109\s+S(2[6-9]\d|[3-9]\d{2,})",
        r"(^|\s)M190\s+S(10\d|1[1-9]\d|[2-9]\d{2,})",
    )
]


def classify_shell_command(command: str) -> tuple[str, str]:
    """Classify shell command risk for permission gating."""
    stripped = command.strip()
    if not stripped:
        return "error", "Command is empty"

    for pattern in HIGH_RISK_SHELL_PATTERNS:
        if pattern.search(stripped):
            return "approval_required", "Command matches a high-risk pattern"

    try:
        first_token = shlex.split(stripped, posix=True)[0]
    except ValueError:
        return "approval_required", "Command could not be parsed safely"

    if first_token in SAFE_SHELL_PREFIXES:
        return "ok", "Command is low risk"

    return "approval_required", "Command is outside the low-risk allowlist"


def classify_gcode_command(command: str) -> tuple[str, str]:
    """Classify G-code risk for permission gating."""
    stripped = command.strip()
    if not stripped:
        return "error", "Command is empty"

    first_token = stripped.split()[0].upper()
    if first_token in LOW_RISK_GCODE_QUERY_PREFIXES:
        return "ok", "Query-style G-code is low risk"

    for pattern in HIGH_RISK_GCODE_PATTERNS:
        if pattern.search(stripped):
            return "approval_required", "G-code matches a high-risk pattern"

    return "approval_required", "Raw printer control G-code requires approval"
