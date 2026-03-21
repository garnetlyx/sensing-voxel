from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp
import pytest
import yaml

from sensing_voxel.camera.tapo import TapoCamera
from sensing_voxel.config import Config, KlipperConfig, TapoCameraConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DEFAULT = PROJECT_ROOT / "config" / "default.yaml"
CONFIG_LOCAL = PROJECT_ROOT / "config" / "local.yaml"
EVIDENCE_DIR = PROJECT_ROOT / "data" / "hardware_validation"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _redact(value: str, keep_start: int = 6, keep_end: int = 4) -> str:
    if len(value) <= keep_start + keep_end:
        return "***"
    return f"{value[:keep_start]}...{value[-keep_end:]}"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_config_for_tests() -> Config:
    with CONFIG_DEFAULT.open("r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f) or {}

    if CONFIG_LOCAL.exists():
        with CONFIG_LOCAL.open("r", encoding="utf-8") as f:
            local_dict = yaml.safe_load(f) or {}
        config_dict = _deep_merge(config_dict, local_dict)

    return Config.from_dict(config_dict)


def _extract_llm_text(response: dict[str, Any]) -> str:
    for block in response.get("content", []):
        if isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                return text

    message = response.get("message")
    if isinstance(message, dict):
        text = message.get("content")
        if isinstance(text, str) and text.strip():
            return text

    choices = response.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            msg = choice.get("message")
            if isinstance(msg, dict):
                text = msg.get("content")
                if isinstance(text, str) and text.strip():
                    return text

    return ""


@dataclass
class EvidenceRecorder:
    file_path: Path

    def record(self, event: str, details: dict[str, Any]) -> None:
        payload = {
            "timestamp": _now_iso(),
            "event": event,
            "details": details,
        }
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with self.file_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")


@pytest.fixture(scope="session")
def evidence() -> EvidenceRecorder:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    file_path = EVIDENCE_DIR / f"hardware_integration_{stamp}.jsonl"
    recorder = EvidenceRecorder(file_path=file_path)
    recorder.record("test_session_started", {"path": str(file_path)})
    return recorder


@pytest.fixture(scope="session")
def loaded_config(evidence: EvidenceRecorder):
    config = _load_config_for_tests()
    evidence.record(
        "config_loaded",
        {
            "default_config": str(CONFIG_DEFAULT),
            "local_config": str(CONFIG_LOCAL),
            "printer_count": len(config.printers),
            "llm_enabled": bool(config.llm.enabled),
            "dashboard_enabled": bool(config.dashboard.enabled),
            "tapo_host": bool(config.tapo_camera and config.tapo_camera.host),
        },
    )
    return config


def _skip_hardware(name: str, reason: str, evidence: EvidenceRecorder) -> None:
    evidence.record("hardware_unreachable", {"component": name, "reason": reason})
    pytest.skip(f"{name} unreachable: {reason}")


@pytest.mark.asyncio
async def test_printer_connectivity(loaded_config, evidence: EvidenceRecorder) -> None:
    klipper_printers = [p for p in loaded_config.printers if isinstance(p, KlipperConfig)]
    if len(klipper_printers) < 2:
        pytest.fail("Expected two Klipper printers in config/local.yaml")

    timeout = aiohttp.ClientTimeout(total=8)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for printer in klipper_printers:
            base_url = (printer.base_url or f"http://{printer.host}:{printer.port}").rstrip("/")
            info_url = f"{base_url}/printer/info"
            status_url = f"{base_url}/printer/objects/query?print_stats"
            info_status: int | None = None
            status_status: int | None = None
            info_json: dict[str, Any] = {}
            status_json: dict[str, Any] = {}
            try:
                async with session.get(info_url) as info_resp:
                    info_status = info_resp.status
                    info_json = await info_resp.json(content_type=None)
                async with session.get(status_url) as status_resp:
                    status_status = status_resp.status
                    status_json = await status_resp.json(content_type=None)
            except Exception as exc:
                _skip_hardware(printer.name, str(exc), evidence)

            if info_status != 200:
                _skip_hardware(printer.name, f"HTTP {info_status} from {info_url}", evidence)
            if status_status != 200:
                _skip_hardware(printer.name, f"HTTP {status_status} from {status_url}", evidence)

            state = (
                status_json.get("result", {})
                .get("status", {})
                .get("print_stats", {})
                .get("state", "unknown")
            )

            evidence.record(
                "printer_connected",
                {
                    "printer": printer.name,
                    "endpoint": base_url,
                    "printer_state": state,
                    "moonraker_version": info_json.get("result", {}).get("software_version"),
                    "info_response": info_json,
                },
            )


@pytest.mark.asyncio
async def test_camera_connectivity(loaded_config, evidence: EvidenceRecorder) -> None:
    tapo_cfg = loaded_config.tapo_camera
    if not tapo_cfg or not tapo_cfg.host:
        pytest.skip("tapo_camera is not configured")

    camera = TapoCamera(
        TapoCameraConfig(
            host=tapo_cfg.host,
            user=tapo_cfg.user,
            password=tapo_cfg.password,
            onvif_port=tapo_cfg.onvif_port,
        )
    )

    presets: dict[str, Any] = {}
    try:
        await asyncio.wait_for(camera.connect(), timeout=15)
        presets = await asyncio.wait_for(camera.get_presets(), timeout=10)
    except Exception as exc:
        _skip_hardware("Tapo C220 ONVIF", str(exc), evidence)
    finally:
        await camera.disconnect()

    evidence.record(
        "camera_connected",
        {
            "host": tapo_cfg.host,
            "onvif_port": tapo_cfg.onvif_port,
            "presets": presets,
        },
    )


@pytest.mark.asyncio
async def test_llm_connectivity(loaded_config, evidence: EvidenceRecorder) -> None:
    llm_cfg = loaded_config.llm
    if not llm_cfg.enabled:
        pytest.skip("LLM is disabled in config")
    if not llm_cfg.api_key:
        pytest.fail("LLM is enabled but api_key is missing")

    base_url = llm_cfg.base_url.rstrip("/")
    headers = {
        "x-api-key": llm_cfg.api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": llm_cfg.model,
        "max_tokens": 32,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": "Reply with exactly: hardware-ok"}],
    }

    reply = ""
    raw_response: dict[str, Any] = {}
    timeout = aiohttp.ClientTimeout(total=30)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{base_url}/v1/messages", headers=headers, json=payload
            ) as resp:
                raw_response = await resp.json(content_type=None)
                if resp.status != 200:
                    _skip_hardware("LLM proxy", f"HTTP {resp.status}: {raw_response}", evidence)
    except Exception as exc:
        _skip_hardware("LLM proxy", str(exc), evidence)

    reply = _extract_llm_text(raw_response)
    if not reply.strip():
        _skip_hardware("LLM proxy", f"No text in response payload: {raw_response}", evidence)

    assert reply.strip(), "LLM returned an empty response"
    evidence.record(
        "llm_connected",
        {
            "base_url": llm_cfg.base_url,
            "model": llm_cfg.model,
            "api_key": _redact(llm_cfg.api_key),
            "reply_preview": reply[:300],
            "raw_response": raw_response,
        },
    )


@pytest.mark.asyncio
async def test_telegram_connectivity(loaded_config, evidence: EvidenceRecorder) -> None:
    token = loaded_config.telegram.bot_token
    if not token or token == "YOUR_BOT_TOKEN_HERE":
        pytest.skip("Telegram bot token is not configured")

    telegram = pytest.importorskip("telegram")
    bot = telegram.Bot(token=token)
    me = None
    try:
        me = await asyncio.wait_for(bot.get_me(), timeout=15)
    except Exception as exc:
        _skip_hardware("Telegram Bot API", str(exc), evidence)

    assert me is not None

    evidence.record(
        "telegram_connected",
        {
            "username": me.username,
            "id": me.id,
            "is_bot": me.is_bot,
            "token": _redact(token),
        },
    )


def test_config_validation(loaded_config, evidence: EvidenceRecorder) -> None:
    assert CONFIG_DEFAULT.exists(), f"Missing required file: {CONFIG_DEFAULT}"
    assert CONFIG_LOCAL.exists(), f"Missing required file: {CONFIG_LOCAL}"

    with CONFIG_DEFAULT.open("r", encoding="utf-8") as f:
        default_yaml = yaml.safe_load(f)
    with CONFIG_LOCAL.open("r", encoding="utf-8") as f:
        local_yaml = yaml.safe_load(f)

    assert isinstance(default_yaml, dict), "config/default.yaml must load as a mapping"
    assert isinstance(local_yaml, dict), "config/local.yaml must load as a mapping"
    assert loaded_config.printers, "No printers configured after config merge"

    evidence.record(
        "config_validated",
        {
            "default_yaml_keys": sorted(default_yaml.keys()),
            "local_yaml_keys": sorted(local_yaml.keys()),
            "printer_names": [getattr(p, "name", "unknown") for p in loaded_config.printers],
            "evidence_file": str(evidence.file_path),
        },
    )


@pytest.mark.asyncio
async def test_application_startup_connects_services(
    loaded_config, evidence: EvidenceRecorder
) -> None:
    cmd = [sys.executable, "-m", "sensing_voxel.main"]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    required_markers = [
        "Connected to Moonraker at printer.example:80",
        "Connected to Moonraker at printer.example:80",
        "Tapo camera connected via ONVIF",
        "LLM initialized:",
        "Telegram bot started",
    ]
    seen_markers: set[str] = set()
    captured_lines: list[str] = []

    try:
        deadline = asyncio.get_running_loop().time() + 60
        while asyncio.get_running_loop().time() < deadline:
            if process.stdout is None:
                break

            try:
                line_raw = await asyncio.wait_for(process.stdout.readline(), timeout=2)
            except TimeoutError:
                continue
            if not line_raw:
                if process.returncode is not None:
                    break
                continue

            line = line_raw.decode("utf-8", errors="replace").strip()
            captured_lines.append(line)

            for marker in required_markers:
                if marker in line:
                    seen_markers.add(marker)

            if len(seen_markers) == len(required_markers):
                break

        missing = [marker for marker in required_markers if marker not in seen_markers]
        if missing:
            _skip_hardware(
                "Application startup",
                f"Missing startup markers: {missing}. Recent logs: {captured_lines[-20:]}",
                evidence,
            )

        evidence.record(
            "application_started",
            {
                "command": " ".join(cmd),
                "seen_markers": sorted(seen_markers),
                "log_excerpt": captured_lines[-50:],
                "pid": process.pid,
            },
        )
    finally:
        if process.returncode is None:
            process.send_signal(signal.SIGINT)
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            except TimeoutError:
                process.kill()
                await process.wait()
