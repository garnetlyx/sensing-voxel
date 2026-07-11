"""
Background print monitoring — periodic camera check + LLM anomaly detection.

When SVA starts a print, this module runs a background asyncio task that:
1. Every N minutes, captures a camera snapshot and analyzes it via LLM
2. If camera isn't on the printer, searches by panning/tilting
3. If anomaly detected (spaghetti, fire, air printing), notifies user
4. If no user response within timeout, auto-pauses the print
5. On print completion, notifies user and stops monitoring

SVA's permissions in this module: start print + pause print only.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine

import cv2

from sensing_voxel.adapters.base import PrinterAdapter
from sensing_voxel.config import PrintMonitorConfig

logger = logging.getLogger(__name__)


class MonitorSeverity(str, Enum):
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class AnalysisResult:
    severity: MonitorSeverity
    anomaly_type: str | None
    confidence: float
    printer_visible: bool
    summary: str
    recommended_action: str | None = None


@dataclass
class MonitorState:
    printer_id: str
    filename: str | None = None
    started_at: datetime = field(default_factory=datetime.now)
    check_count: int = 0
    last_analysis: AnalysisResult | None = None
    anomaly_pending: bool = False


SYSTEM_PROMPT = """You are Sensing Voxel's autonomous 3D print monitor.
Analyze the camera snapshot for print-failure anomalies.
Detect: spaghetti/detached print, air printing (no filament), fire/smoke,
empty bed when print should be active, print detached from bed.
Be conservative — if uncertain, use warning not critical.
Respond as strict JSON only:
{"severity": "ok|warning|critical|unknown", "anomaly_type": "string_or_null", "confidence": 0.0-1.0, "printer_visible": true/false, "summary": "one sentence", "recommended_action": "string_or_null"}"""

SEARCH_PROMPT = """Is a 3D printer visible in this camera frame? If yes, is a print in progress on the bed?
Respond as JSON: {"printer_visible": true/false, "print_active": true/false, "summary": "brief"}"""


class PrintMonitor:
    def __init__(
        self,
        printer_id: str,
        adapter: PrinterAdapter,
        rtsp_url: str,
        llm_send_fn: Callable[..., Coroutine[Any, Any, str]],
        notifier_fn: Callable[[str, str, str], Coroutine[Any, Any, None]],
        config: PrintMonitorConfig,
    ):
        self.printer_id = printer_id
        self._adapter = adapter
        self._rtsp_url = rtsp_url
        self._llm_send = llm_send_fn
        self._notify = notifier_fn
        self._config = config
        self._task: asyncio.Task | None = None
        self._pause_task: asyncio.Task | None = None
        self._state: MonitorState | None = None
        self._stop_event = asyncio.Event()
        self._ack_event = asyncio.Event()
        self._auto_enabled = False

    def enable_auto_monitoring(self) -> None:
        """Register adapter event listeners for automatic monitor activation."""
        self._auto_enabled = True
        if hasattr(self._adapter, "on_state_changed"):
            self._adapter.on_state_changed(self._on_adapter_state_changed)

    async def _on_adapter_state_changed(self, old_state: Any, new_state: Any) -> None:
        """Auto-activate monitoring when printer enters PRINTING state."""
        if not self._auto_enabled:
            return

        new_name = new_state.name if hasattr(new_state, "name") else str(new_state)
        old_name = old_state.name if hasattr(old_state, "name") else str(old_state)

        if new_name == "PRINTING" and old_name != "PRINTING":
            if not self.is_monitoring:
                logger.info(f"Auto-activating monitor: {self.printer_id} entered PRINTING")
                await self.start_monitoring()

        if old_name == "PRINTING" and new_name in ("STANDBY", "COMPLETE", "IDLE_NO_ACTIVE_PRINT", "CANCELLED", "ERROR"):
            if self.is_monitoring:
                if self._config.notify_on_completion:
                    await self._notify(
                        self.printer_id, "info",
                        f"Print ended on {self.printer_id} (state: {new_name})",
                    )
                await self.stop_monitoring(reason=f"state_changed_to_{new_name}")

    async def start_monitoring(self, filename: str | None = None) -> None:
        if self._task and not self._task.done():
            await self.stop_monitoring(reason="replaced")

        self._state = MonitorState(printer_id=self.printer_id, filename=filename)
        self._stop_event.clear()
        self._ack_event.clear()
        self._task = asyncio.create_task(
            self._loop(), name=f"print-monitor:{self.printer_id}"
        )
        logger.info(f"Print monitoring started for {self.printer_id}: {filename}")

    async def stop_monitoring(self, reason: str = "manual") -> None:
        self._stop_event.set()
        for t in (self._task, self._pause_task):
            if t and not t.done():
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
        self._task = None
        self._pause_task = None
        self._state = None
        logger.info(f"Print monitoring stopped for {self.printer_id}: {reason}")

    async def acknowledge(self) -> None:
        self._ack_event.set()
        if self._state:
            self._state.anomaly_pending = False

    @property
    def is_monitoring(self) -> bool:
        return self._task is not None and not self._task.done()

    async def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self._check_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Monitor check failed")

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._config.check_interval_seconds,
                )
            except asyncio.TimeoutError:
                pass

    async def _check_once(self) -> None:
        if not self._state:
            return
        self._state.check_count += 1

        progress = await self._adapter.get_print_progress()
        progress_data = progress.data if hasattr(progress, "data") else {}
        state_str = str(progress_data.get("state", "unknown"))

        if state_str in ("standby", "complete", "cancelled", "error"):
            if self._config.notify_on_completion:
                await self._notify(
                    self.printer_id,
                    "success",
                    f"Print completed on {self.printer_id}"
                    + (f": {self._state.filename}" if self._state.filename else ""),
                )
            await self.stop_monitoring(reason="print_complete")
            return

        snapshot_b64 = await self._capture_rtsp()
        if snapshot_b64 is None:
            if self._config.notify_on_camera_failure:
                await self._notify(
                    self.printer_id, "warning",
                    f"Camera unreachable — cannot monitor {self.printer_id}. Check camera connection.",
                )
            return

        context = {
            "printer_id": self.printer_id,
            "filename": self._state.filename,
            "state": state_str,
            "temps": {
                "nozzle": progress_data.get("extruder_temp"),
                "bed": progress_data.get("bed_temp"),
            },
        }

        result = await self._analyze(snapshot_b64, context)
        self._state.last_analysis = result

        if not result.printer_visible and self._config.camera_search_enabled:
            logger.info(f"Printer not visible, searching...")
            result = await self._search_and_reanalyze(context)

        if self._is_anomaly(result):
            await self._handle_anomaly(result)
        elif result.severity == MonitorSeverity.OK:
            logger.info(
                f"Check #{self._state.check_count} OK: {result.summary} "
                f"(visible={result.printer_visible})"
            )

    async def _capture_rtsp(self) -> str | None:
        def _grab() -> str | None:
            cap = cv2.VideoCapture(self._rtsp_url)
            if not cap.isOpened():
                return None
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return None
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return base64.b64encode(buf).decode()

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_grab), timeout=15
            )
        except Exception as e:
            logger.warning(f"RTSP capture failed: {e}")
            return None

    async def _analyze(
        self, jpeg_b64: str, context: dict
    ) -> AnalysisResult:
        prompt = (
            f"Analyze this 3D printer snapshot.\n"
            f"Context: {json.dumps(context, default=str)}\n\n"
            f"Check for: spaghetti, air printing, fire/smoke, print failure, "
            f"empty bed, detached print.\n"
            f"Respond as strict JSON."
        )
        try:
            response = await self._llm_send(
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": jpeg_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
                system_prompt=SYSTEM_PROMPT,
                max_tokens=self._config.llm_max_tokens,
            )
            return self._parse_analysis(response)
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return AnalysisResult(
                severity=MonitorSeverity.UNKNOWN,
                anomaly_type="llm_error",
                confidence=0.0,
                printer_visible=False,
                summary=f"Analysis failed: {e}",
            )

    def _parse_analysis(self, response: str) -> AnalysisResult:
        text = response
        if hasattr(response, "content"):
            for block in response.content:
                if hasattr(block, "text"):
                    text = block.text
                    break
        elif isinstance(response, dict) and "content" in response:
            for block in response["content"]:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    break

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            for line in text.split("\n"):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        data = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue
            else:
                return AnalysisResult(
                    severity=MonitorSeverity.UNKNOWN,
                    anomaly_type="parse_error",
                    confidence=0.0,
                    printer_visible=True,
                    summary=text[:200],
                )

        return AnalysisResult(
            severity=MonitorSeverity(data.get("severity", "unknown")),
            anomaly_type=data.get("anomaly_type"),
            confidence=float(data.get("confidence", 0.0)),
            printer_visible=bool(data.get("printer_visible", True)),
            summary=str(data.get("summary", "")),
            recommended_action=data.get("recommended_action"),
        )

    async def _search_and_reanalyze(self, context: dict) -> AnalysisResult:
        moves = ["left", "left", "right", "right", "up", "down"]
        last = self._state.last_analysis if self._state else None

        for i, direction in enumerate(moves[: self._config.camera_search_max_moves]):
            if self._stop_event.is_set():
                break

            camera = self._adapter._camera if hasattr(self._adapter, "_camera") else None
            if camera and hasattr(camera, "move_step"):
                try:
                    await camera.move_step(direction)
                except Exception:
                    pass
            await asyncio.sleep(self._config.camera_search_settle_seconds)

            snapshot_b64 = await self._capture_rtsp()
            if snapshot_b64 is None:
                continue

            result = await self._analyze(snapshot_b64, context)
            if result.printer_visible:
                logger.info(f"Found printer after {i+1} moves")
                return result
            last = result

        return last or AnalysisResult(
            severity=MonitorSeverity.WARNING,
            anomaly_type="printer_not_visible",
            confidence=0.5,
            printer_visible=False,
            summary="Could not locate printer after searching",
        )

    def _is_anomaly(self, result: AnalysisResult) -> bool:
        return (
            result.severity in {MonitorSeverity.WARNING, MonitorSeverity.CRITICAL}
            and result.confidence >= 0.6
            and result.anomaly_type not in (None, "printer_not_visible", "llm_error")
        )

    async def _handle_anomaly(self, result: AnalysisResult) -> None:
        if self._state and self._state.anomaly_pending:
            return
        if self._state:
            self._state.anomaly_pending = True
        self._ack_event.clear()

        level = "critical" if result.severity == MonitorSeverity.CRITICAL else "warning"
        msg = (
            f"⚠️ Print anomaly on {self.printer_id}: {result.summary}\n"
            f"Type: {result.anomaly_type} | Confidence: {result.confidence:.0%}\n"
        )
        if result.recommended_action:
            msg += f"Suggested: {result.recommended_action}\n"
        msg += (
            f"Auto-pause in {self._config.anomaly_response_timeout_seconds:.0f}s "
            "if not acknowledged."
        )

        await self._notify(self.printer_id, level, msg)

        self._pause_task = asyncio.create_task(
            self._wait_or_pause(result), name=f"anomaly-wait:{self.printer_id}"
        )

    async def _wait_or_pause(self, result: AnalysisResult) -> None:
        try:
            await asyncio.wait_for(
                self._ack_event.wait(),
                timeout=self._config.anomaly_response_timeout_seconds,
            )
            logger.info(f"Anomaly acknowledged by user for {self.printer_id}")
            return
        except asyncio.TimeoutError:
            pass

        if not self._config.auto_pause_enabled:
            return

        logger.warning(f"Auto-pausing {self.printer_id} — unacknowledged anomaly")
        await self._adapter.pause_print()
        await self._notify(
            self.printer_id,
            "critical",
            f"🛑 Auto-paused {self.printer_id}: unacknowledged anomaly — {result.summary}. "
            "Print is paused. Investigate and resume when ready.",
        )
