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
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

import cv2

from sensing_voxel.adapters.base import PrinterAdapter
from sensing_voxel.config import PrintMonitorConfig
from sensing_voxel.safety.models import MonitorStartContract

logger = logging.getLogger(__name__)

FIRST_LAYER_CHECK_INTERVAL_SECONDS = 60.0
WaitFn = Callable[[asyncio.Event, float], Coroutine[Any, Any, bool]]


class MonitorSeverity(StrEnum):
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
    camera_failures: int = 0
    camera_notified: bool = False
    first_layer_complete: bool = False
    first_layer_z: float | None = None
    required: bool = False


SYSTEM_PROMPT = """You are Sensing Voxel's autonomous 3D print monitor.
Analyze the camera snapshot for print-failure anomalies.

IMPORTANT — distinguish real failures from normal prints:
- Textured/patterned walls, vase-mode prints, lattice structures are NORMAL, not spaghetti
- Spaghetti = loose tangled filament strands piled randomly with no structure
- A tall print with visible layer lines or texture is NOT a failure
- Only flag as anomaly if you see: actual tangled mess, fire/smoke, empty bed (when print should be active), detached print moving freely

Be conservative — if uncertain, use warning not critical. Never report an analysis failure or unknown state as ok.
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
        wait_fn: WaitFn | None = None,
    ):
        self.printer_id = printer_id
        self._adapter = adapter
        self._rtsp_url = rtsp_url
        self._llm_send = llm_send_fn
        self._notify = notifier_fn
        self._config = config
        self._wait_fn = wait_fn or self._wait_for_stop
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
            await asyncio.sleep(5)
            progress = await self._adapter.get_print_progress()
            current_state = getattr(progress, "state", None)
            current_name = current_state.name if hasattr(current_state, "name") else str(current_state)
            if current_name != "PRINTING":
                logger.info(f"Skipping monitor activation: state changed to {current_name} within 5s debounce")
                return
            if not self.is_monitoring:
                logger.info(f"Auto-activating monitor: {self.printer_id} entered PRINTING")
                await self.start_monitoring()

        if old_name == "PRINTING" and new_name in ("STANDBY", "COMPLETE", "IDLE_NO_ACTIVE_PRINT", "CANCELLED", "ERROR"):
            await asyncio.sleep(5)
            progress = await self._adapter.get_print_progress()
            current_state = getattr(progress, "state", None)
            current_name = current_state.name if hasattr(current_state, "name") else str(current_state)
            if current_name == "PRINTING":
                logger.info("Skipping 'Print ended' notification: state returned to PRINTING within 5s debounce")
                return
            if self.is_monitoring:
                if self._config.notify_on_completion:
                    await self._notify(
                        self.printer_id, "info",
                        f"Print ended on {self.printer_id} (state: {new_name})",
                    )
                await self.stop_monitoring(reason=f"state_changed_to_{new_name}")

    async def start_monitoring(
        self,
        filename: str | None = None,
        *,
        contract: MonitorStartContract | None = None,
    ) -> None:
        if contract is not None:
            if filename is not None and filename != contract.filename:
                raise ValueError("monitor filename conflicts with start contract")
            filename = contract.filename
        if contract is not None and contract.required:
            if not filename:
                raise RuntimeError("required monitor filename is unavailable")
            if not contract.source_trust:
                raise RuntimeError("required monitor source identity is unavailable")
            if contract.first_layer_z is None:
                raise RuntimeError("required monitor first-layer preview data is unavailable")

        if self._task and not self._task.done():
            if contract is not None and self._state and self._state.required is True:
                if self._state.filename == contract.filename:
                    return
                raise RuntimeError("required print monitor is already active for another print")
            await self.stop_monitoring(reason="replaced")

        if contract is not None and contract.required:
            await self._validate_required_readiness(contract)

        self._state = MonitorState(
            printer_id=self.printer_id,
            filename=filename,
            first_layer_z=contract.first_layer_z if contract else None,
            required=bool(contract and contract.required),
        )
        self._stop_event.clear()
        self._ack_event.clear()
        self._task = asyncio.create_task(
            self._loop(), name=f"print-monitor:{self.printer_id}"
        )
        logger.info(f"Print monitoring started for {self.printer_id}: {filename}")

    async def _validate_required_readiness(self, contract: MonitorStartContract) -> None:
        try:
            progress = await self._adapter.get_print_progress()
        except Exception as exc:
            raise RuntimeError("required monitor progress is unavailable") from exc
        state = getattr(progress, "state", None)
        state_name = state.name if hasattr(state, "name") else str(state)
        if state_name in ("UNKNOWN", "None", ""):
            raise RuntimeError("required monitor progress state is unknown")

        try:
            snapshot = await self._capture_rtsp()
        except Exception as exc:
            raise RuntimeError("required monitor camera is unavailable") from exc
        if snapshot is None:
            raise RuntimeError("required monitor camera is unavailable")

        context = {
            "printer_id": self.printer_id,
            "filename": contract.filename,
            "state": state_name,
            "first_layer_z": contract.first_layer_z,
        }
        try:
            result = await asyncio.wait_for(self._analyze(snapshot, context), timeout=60)
        except Exception as exc:
            raise RuntimeError("required monitor LLM analysis is unavailable") from exc
        if (
            result.severity == MonitorSeverity.UNKNOWN
            or result.anomaly_type in {"llm_timeout", "llm_error", "parse_error", "analysis_unknown"}
        ):
            raise RuntimeError("required monitor LLM analysis is unavailable")
        if not result.printer_visible:
            raise RuntimeError("required monitor camera view is unknown")

    async def stop_monitoring(self, reason: str = "manual") -> None:
        self._stop_event.set()
        current_task = asyncio.current_task()
        for task in (self._task, self._pause_task):
            if task is current_task or task is None or task.done():
                continue
            task.cancel()
            try:
                await task
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
                logger.info(f"Monitor loop: starting check for {self.printer_id}")
                await self._check_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception(f"Monitor check failed: {exc}")
                if self._state and self._state.required is True:
                    await self._fail_closed("monitor_check_unavailable", exc)
                    return

            if self._stop_event.is_set():
                return
            interval = self._current_check_interval()
            if await self._wait_fn(self._stop_event, interval):
                return

    @staticmethod
    async def _wait_for_stop(stop_event: asyncio.Event, timeout: float) -> bool:
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=timeout)
        except TimeoutError:
            return False
        return True

    def _current_check_interval(self) -> float:
        if self._state and not self._state.first_layer_complete:
            return FIRST_LAYER_CHECK_INTERVAL_SECONDS
        return self._config.check_interval_seconds

    def _update_first_layer_state(self, progress: Any) -> None:
        if not self._state or self._state.first_layer_complete:
            return
        current_layer = getattr(progress, "current_layer", None)
        if isinstance(current_layer, int) and not isinstance(current_layer, bool):
            if current_layer > 1:
                self._state.first_layer_complete = True
            return
        if current_layer is not None:
            logger.warning("Ignoring unreliable current_layer value: %r", current_layer)

    async def _fail_closed(self, reason: str, error: Exception | None = None) -> None:
        detail = f": {error}" if error else ""
        logger.error("Required monitoring failed closed for %s: %s%s", self.printer_id, reason, detail)
        try:
            await self._adapter.pause_print()
        except Exception:
            logger.exception("Failed to pause %s after required monitor failure", self.printer_id)
        try:
            await self._notify(
                self.printer_id,
                "critical",
                f"Required print monitoring is unavailable on {self.printer_id} ({reason}). Print paused.",
            )
        except Exception:
            logger.exception("Failed to notify required monitor failure for %s", self.printer_id)
        self._stop_event.set()

    async def _check_once(self) -> None:
        if not self._state:
            return
        self._state.check_count += 1
        logger.info(f"Check #{self._state.check_count} starting for {self.printer_id}")

        progress = await self._adapter.get_print_progress()
        logger.info(f"Check #{self._state.check_count}: got progress")
        self._update_first_layer_state(progress)

        progress_state = getattr(progress, "state", None)
        progress_state_name = progress_state.name if hasattr(progress_state, "name") else str(progress_state)

        if progress_state_name in ("UNKNOWN", "None", ""):
            if self._state.required is True:
                await self._fail_closed("printer_progress_unknown")
            return
        if progress_state_name not in ("PRINTING", "PAUSED"):
            if self._state.check_count == 1 and self._state.required is True:
                logger.info("Required monitor armed before printer start; waiting for printing state")
                return
            if self._config.notify_on_completion:
                await self._notify(
                    self.printer_id,
                    "success",
                    f"Print completed on {self.printer_id}"
                    + (f": {self._state.filename}" if self._state.filename else ""),
                )
            await self.stop_monitoring(reason="print_complete")
            return

        if progress_state_name == "PAUSED":
            logger.info(f"Check #{self._state.check_count}: printer paused, skipping check")
            return

        current_layer = getattr(progress, "current_layer", None)
        if self._state.required is True and (
            not isinstance(current_layer, int)
            or isinstance(current_layer, bool)
            or current_layer < 1
        ):
            await self._fail_closed("current_layer_unknown")
            return

        logger.info(f"Check #{self._state.check_count}: capturing RTSP...")
        snapshot_b64 = await self._capture_rtsp()
        logger.info(f"Check #{self._state.check_count}: RTSP result: {'OK' if snapshot_b64 else 'FAILED'}")
        if snapshot_b64 is None:
            self._state.camera_failures += 1
            if self._state.required is True:
                await self._fail_closed("camera_unavailable")
                return
            if self._config.notify_on_camera_failure and not self._state.camera_notified:
                self._state.camera_notified = True
                await self._notify(
                    self.printer_id, "warning",
                    f"Camera unreachable — cannot visually monitor {self.printer_id}. "
                    f"Print continues but SVA is blind. Check camera connection.",
                )
            elif self._state.camera_failures == 5:
                await self._notify(
                    self.printer_id, "critical",
                    f"Camera has been unreachable for {self._state.camera_failures} consecutive checks "
                    f"on {self.printer_id}. Print is still running without visual monitoring. "
                    f"Consider manually checking the printer.",
                )
            return

        if self._state.camera_failures > 0 and self._state.camera_notified:
            self._state.camera_notified = False
            await self._notify(
                self.printer_id, "info",
                f"Camera reconnected for {self.printer_id} — visual monitoring resumed.",
            )
        self._state.camera_failures = 0

        context = {
            "printer_id": self.printer_id,
            "filename": self._state.filename,
            "state": progress_state_name,
            "temps": {
                "nozzle": getattr(progress, "nozzle_temp", None),
                "bed": getattr(progress, "bed_temp", None),
            },
        }

        logger.info(f"Check #{self._state.check_count}: analyzing snapshot...")
        try:
            result = await asyncio.wait_for(
                self._analyze(snapshot_b64, context),
                timeout=60,
            )
        except TimeoutError:
            logger.warning(f"Check #{self._state.check_count}: LLM analysis timed out after 60s")
            result = AnalysisResult(
                severity=MonitorSeverity.WARNING,
                anomaly_type="llm_timeout",
                confidence=0.0,
                printer_visible=True,
                summary="LLM analysis timed out",
            )
        logger.info(f"Check #{self._state.check_count}: analysis done: severity={result.severity.value} confidence={result.confidence} anomaly_type={result.anomaly_type} summary={result.summary[:100]}")
        self._state.last_analysis = result

        if self._state.required is True and (
            result.severity == MonitorSeverity.UNKNOWN
            or result.anomaly_type in {"llm_timeout", "llm_error", "parse_error", "analysis_unknown"}
        ):
            await self._fail_closed("llm_analysis_unavailable")
            return

        if not result.printer_visible and self._config.camera_search_enabled:
            logger.info("Printer not visible, searching...")
            result = await self._search_and_reanalyze(context)
            self._state.last_analysis = result

        if self._state.required is True and not result.printer_visible:
            await self._fail_closed("camera_view_unknown")
            return

        is_anom = self._is_anomaly(result)
        logger.info(f"Check #{self._state.check_count}: _is_anomaly={is_anom}")
        if is_anom:
            logger.warning(f"ANOMALY DETECTED on {self.printer_id}: {result.anomaly_type} — {result.summary[:100]}")
            await self._handle_anomaly(result)
        elif result.severity == MonitorSeverity.OK:
            logger.info(
                f"Check #{self._state.check_count} OK: {result.summary} "
                f"(visible={result.printer_visible})"
            )

    async def _capture_rtsp(self) -> str | None:
        def _grab() -> str | None:
            cap = cv2.VideoCapture(self._rtsp_url, cv2.CAP_FFMPEG)
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
                severity=MonitorSeverity.WARNING,
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
            found = False
            for line in text.split("\n"):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        data = json.loads(line)
                        found = True
                        break
                    except json.JSONDecodeError:
                        continue
            if not found:
                text_lower = text.lower()
                if any(w in text_lower for w in ["fire", "smoke", "spaghetti", "critical", "failed", "detached"]):
                    return AnalysisResult(
                        severity=MonitorSeverity.WARNING,
                        anomaly_type="natural_language_warning",
                        confidence=0.7,
                        printer_visible=True,
                        summary=text[:200],
                    )
                return AnalysisResult(
                    severity=MonitorSeverity.WARNING,
                    anomaly_type="parse_error",
                    confidence=1.0,
                    printer_visible=False,
                    summary=f"Could not parse monitor analysis: {text[:200]}",
                )

        try:
            severity = MonitorSeverity(data.get("severity", "unknown"))
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            return AnalysisResult(
                severity=MonitorSeverity.WARNING,
                anomaly_type="parse_error",
                confidence=1.0,
                printer_visible=False,
                summary="Monitor analysis contained invalid severity or confidence",
            )
        anomaly_type = data.get("anomaly_type")
        if severity == MonitorSeverity.UNKNOWN:
            severity = MonitorSeverity.WARNING
            anomaly_type = anomaly_type or "analysis_unknown"
            confidence = max(confidence, 1.0)

        return AnalysisResult(
            severity=severity,
            anomaly_type=anomaly_type,
            confidence=confidence,

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
        if result.severity in (MonitorSeverity.WARNING, MonitorSeverity.CRITICAL, MonitorSeverity.UNKNOWN):
            return True
        if result.anomaly_type is not None:
            return True
        if not result.printer_visible:
            return True
        return False


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
        except TimeoutError:
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
