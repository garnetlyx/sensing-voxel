"""
Monitoring orchestrator for print supervision

Coordinates external print detection and active print monitoring.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

import numpy as np

from sensing_voxel.adapters.base import PrinterAdapter
from sensing_voxel.core.models import (
    AgentState,
    JobContext,
    NormalizedPrinterState,
    NOTHING,
    generate_job_id,
)
from sensing_voxel.core.persistence import StatePersistence
from sensing_voxel.core.state_machine import AgentStateMachine
from sensing_voxel.monitoring.z_offset import ZOffsetController

logger = logging.getLogger(__name__)

try:
    from sensing_voxel.vision.camera import CameraStream
    from sensing_voxel.vision.calibration import CameraCalibrator
    from sensing_voxel.vision.first_layer import FirstLayerClassifier
    from sensing_voxel.vision.NOTHING import NOTHING, NOTHING
    from sensing_voxel.vision.filament import FilamentMonitor

    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    logger.warning("Vision modules not available, some features will be disabled")
    FilamentMonitor = None  # type: ignore[misc,assignment]

try:
    from sensing_voxel.audio.detector import AudioAnomalyDetector, DetectionResult
    from sensing_voxel.audio.baseline import BaselineManager

    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    logger.warning("Audio modules not available, audio anomaly detection disabled")
    AudioAnomalyDetector = None  # type: ignore[misc,assignment]
    BaselineManager = None  # type: ignore[misc,assignment]


class MonitoringOrchestrator:
    """
    Orchestrates print monitoring lifecycle

    Handles:
    - External print detection (debounced)
    - Monitoring lifecycle for agent-started prints
    - State machine transitions based on printer events
    - Audio anomaly detection for mechanical failures
    """

    def __init__(
        self,
        state_machine: AgentStateMachine,
        adapter: PrinterAdapter,
        persistence: StatePersistence,
        camera: CameraStream | None = None,
        NOTHING_detector: NOTHING | None = None,
        first_layer_classifier: FirstLayerClassifier | None = None,
        z_offset_controller: ZOffsetController | None = None,
        calibrator: CameraCalibrator | None = None,
        audio_detector: AudioAnomalyDetector | None = None,
        baseline_manager: BaselineManager | None = None,
        filament_monitor: FilamentMonitor | None = None,
        printer_id: str = "default",
        debounce_interval_seconds: int = 3,
        debounce_required_updates: int = 2,
        enable_audio_monitoring: bool = True,
        enable_filament_monitoring: bool = True,
    ):
        self.state_machine = state_machine
        self.adapter = adapter
        self.persistence = persistence
        self.printer_id = printer_id
        self.enable_audio_monitoring = enable_audio_monitoring and AUDIO_AVAILABLE
        self.enable_filament_monitoring = enable_filament_monitoring and VISION_AVAILABLE

        # Vision components (optional — gracefully degrade if missing)
        self.camera = camera
        self.NOTHING_detector = NOTHING_detector
        self.first_layer_classifier = first_layer_classifier
        self.z_offset_controller = z_offset_controller
        self.calibrator = calibrator

        # Audio components (optional)
        self.audio_detector = audio_detector
        self.baseline_manager = baseline_manager

        # Filament monitor (optional)
        self.filament_monitor = filament_monitor

        self.debounce_interval = debounce_interval_seconds
        self.debounce_required_updates = debounce_required_updates

        # Debounce state
        self._debounce_counter = 0
        self._debounce_window_start: datetime | None = None
        self._debounce_timer: asyncio.Task | None = None

        # Monitoring state
        self._is_monitoring = False
        self._monitoring_job_id: str | None = None
        self._NOTHING_task: asyncio.Task | None = None
        self._first_layer_task: asyncio.Task | None = None
        self._audio_task: asyncio.Task | None = None
        self._filament_task: asyncio.Task | None = None

        # Layer tracking
        self._current_layer: int | None = None
        self._total_layers: int | None = None

        # Event handlers
        self._state_change_handler: Callable | None = None
        self._connection_lost_handler: Callable | None = None
        self._connection_restored_handler: Callable | None = None

        logger.info(
            f"Monitoring orchestrator initialized with debounce: "
            f"{debounce_required_updates} updates / {debounce_interval_seconds}s, "
            f"audio={'enabled' if self.enable_audio_monitoring else 'disabled'}"
        )

    async def start(self):
        """Start monitoring orchestrator"""

        # Register state change handler
        async def on_state_changed(
            old_state: NormalizedPrinterState, new_state: NormalizedPrinterState
        ):
            await self._handle_state_change(old_state, new_state)

        self._state_change_handler = on_state_changed
        self.adapter.on_state_changed(on_state_changed)

        # Register connection handlers for disconnection handling
        async def on_connection_lost():
            await self._handle_connection_lost()

        async def on_connection_restored():
            await self._handle_connection_restored()

        self._connection_lost_handler = on_connection_lost
        self._connection_restored_handler = on_connection_restored
        self.adapter.on_connection_lost(on_connection_lost)
        self.adapter.on_connection_restored(on_connection_restored)

        logger.info("Monitoring orchestrator started")

    async def stop(self):
        """Stop monitoring orchestrator"""
        if self._state_change_handler:
            self.adapter.remove_state_changed_handler(self._state_change_handler)
        if self._connection_lost_handler:
            self.adapter.remove_connection_lost_handler(self._connection_lost_handler)
        if self._connection_restored_handler:
            self.adapter.remove_connection_restored_handler(self._connection_restored_handler)

        # Cancel vision tasks
        await self._cancel_vision_tasks()

        # Cancel audio monitoring
        await self._cancel_audio_monitoring()

        # Cancel debounce timer if active
        if self._debounce_timer and not self._debounce_timer.done():
            self._debounce_timer.cancel()
            try:
                await self._debounce_timer
            except asyncio.CancelledError:
                pass

        logger.info("Monitoring orchestrator stopped")

    async def _handle_state_change(
        self,
        old_state: NormalizedPrinterState,
        new_state: NormalizedPrinterState,
    ):
        """
        Handle printer state change

        Primary entry point for external print detection and print completion.
        """
        logger.debug(f"Printer state changed: {old_state.name} -> {new_state.name}")

        # Handle print completion (including during disconnection)
        if old_state == NormalizedPrinterState.PRINTING and new_state in (
            NormalizedPrinterState.TERMINAL,
            NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT,
        ):
            if self._is_monitoring:
                logger.info("Print completed, transitioning to COMPLETE")
                await self.state_machine.transition_to(
                    AgentState.COMPLETE,
                    self.persistence,
                )
                await self.stop_monitoring()
            return

        # Handle pause during monitoring
        if new_state == NormalizedPrinterState.PAUSED and self._is_monitoring:
            logger.info("Print paused by printer")
            await self.handle_print_pause()
            return

        # External print detection: only for transitions to PRINTING when agent is IDLE
        if new_state != NormalizedPrinterState.PRINTING:
            return

        if self.state_machine.current_state != AgentState.IDLE:
            logger.debug("State change ignored - agent is not IDLE (agent-started print)")
            return

        # External print detected - apply debounce
        await self._apply_debounce_detection()

    async def _apply_debounce_detection(self):
        """
        Apply debounce logic for external print detection

        Debouncing prevents false positives from transient state changes.
        Requires N updates within time window to trigger external print detection.
        """
        now = datetime.now()

        # Check if we're within the debounce window
        if self._debounce_window_start is None:
            # First update - start window
            self._debounce_window_start = now
            self._debounce_counter = 1
            logger.debug(f"Debounce window started (1/{self.debounce_required_updates} updates)")

            # Schedule window reset after interval
            if self._debounce_timer:
                self._debounce_timer.cancel()

            self._debounce_timer = asyncio.create_task(self._reset_debounce_window_after_delay())
            return

        # Calculate time since window start
        elapsed = (now - self._debounce_window_start).total_seconds()

        if elapsed <= self.debounce_interval:
            # Within window - increment counter
            self._debounce_counter += 1
            logger.debug(
                f"Debounce update {self._debounce_counter}/{self.debounce_required_updates} "
                f"within {elapsed:.1f}s window"
            )

            # Check if we've reached threshold
            if self._debounce_counter >= self.debounce_required_updates:
                logger.info(
                    f"External print detected: {self._debounce_counter} updates in "
                    f"{elapsed:.1f}s (threshold: {self.debounce_required_updates})"
                )
                await self._trigger_external_print_detection()
        else:
            # Window expired - reset
            logger.debug(f"Debounce window expired after {elapsed:.1f}s, resetting")
            await self._reset_debounce_state()

    async def _reset_debounce_window_after_delay(self):
        """Reset debounce window after delay"""
        try:
            await asyncio.sleep(self.debounce_interval)
            await self._reset_debounce_state()
        except asyncio.CancelledError:
            # Timer cancelled - expected during state transitions
            pass

    async def _reset_debounce_state(self):
        """Reset debounce state"""
        self._debounce_counter = 0
        self._debounce_window_start = None
        self._debounce_timer = None
        logger.debug("Debounce state reset")

    async def _trigger_external_print_detection(self):
        """Trigger external print detection workflow"""
        # Capture counter value before reset for logging
        detection_counter = self._debounce_counter

        # Generate job ID for external print
        job_id = generate_job_id()

        # Create job context for external print
        # Note: external prints are NOTHING-only monitoring
        # First-layer correction is disabled for safety
        job_context = JobContext(
            job_id=job_id,
            job_file="external_print",  # Placeholder - no actual file
            uploaded_at=datetime.now(),
            is_external=True,
            external_monitoring=True,  # Critical: external prints are NOTHING-only
        )

        # Set job context in state machine
        self.state_machine.job_context = job_context

        # Transition to MONITORING state
        # External prints skip RECEIVING/SLICING/APPROVAL/UPLOADING
        success = await self.state_machine.transition_to(
            AgentState.MONITORING,
            self.persistence,
        )

        # Reset debounce state AFTER transition
        await self._reset_debounce_state()

        if success:
            logger.info(
                f"External print monitoring started (job_id={job_id}, "
                f"external_monitoring={job_context.external_monitoring}, "
                f"debounce_count={detection_counter})"
            )
            self._is_monitoring = True
            self._monitoring_job_id = job_id

            await self.NOTHING(job_context)
        else:
            logger.error("Failed to transition to MONITORING for external print")
            self.state_machine.job_context = None

    async def start_monitoring_for_agent_print(self, job_context: JobContext):
        """
        Start monitoring for agent-started print

        Called after agent successfully uploads and starts print.

        Args:
            job_context: Job context from agent workflow
        """
        if self.state_machine.current_state != AgentState.PRINTING:
            logger.error(
                f"Cannot start monitoring - state is {self.state_machine.current_state.name}, "
                "expected PRINTING"
            )
            return

        # Set job context
        self.state_machine.job_context = job_context

        # Transition to MONITORING
        success = await self.state_machine.transition_to(
            AgentState.MONITORING,
            self.persistence,
        )

        if success:
            logger.info(
                f"Agent-started print monitoring started (job_id={job_context.job_id}, "
                f"external_monitoring={job_context.external_monitoring})"
            )
            self._is_monitoring = True
            self._monitoring_job_id = job_context.job_id

            await self._start_full_monitoring(job_context)
        else:
            logger.error("Failed to transition to MONITORING for agent print")

    @property
    def layer_info(self) -> str | None:
        """Get current layer info for notification context"""
        if self._current_layer is not None and self._total_layers is not None:
            return f"Layer {self._current_layer}/{self._total_layers}"
        return None

    def update_layer_progress(self, progress: float | None, total_layers: int | None):
        """Update layer tracking from telemetry data"""
        if total_layers is not None:
            self._total_layers = total_layers
        if progress is not None and self._total_layers:
            self._current_layer = max(1, int(progress * self._total_layers))

    async def stop_monitoring(self):
        """Stop active monitoring"""
        if not self._is_monitoring:
            logger.debug("No active monitoring to stop")
            return

        logger.info(f"Stopping monitoring for job {self._monitoring_job_id}")

        # Cancel vision tasks
        await self._cancel_vision_tasks()

        # Restore Z-offset baseline if controller was active
        if self.z_offset_controller and self.z_offset_controller._running:
            await self.z_offset_controller.stop(self.adapter)

        self._is_monitoring = False
        self._monitoring_job_id = None
        self._current_layer = None
        self._total_layers = None

        # Cleanup debounce state
        await self._reset_debounce_state()

    async def handle_print_pause(self):
        """Handle print pause event"""
        if not self._is_monitoring:
            return

        # Transition to PAUSED_PENDING_USER
        # This allows user to resume or cancel
        await self.state_machine.transition_to(
            AgentState.PAUSED_PENDING_USER,
            self.persistence,
        )

    async def NOTHING(self, confidence: float):
        """
        Handle NOTHING detection pause request

        Called when moderate confidence NOTHING is detected.

        Args:
            confidence: Detection confidence (0.0-1.0)
        """
        if not self._is_monitoring:
            return

        logger.info(f"NOTHING detected (confidence={confidence:.2f}), pausing for user review")

        await self.handle_print_pause()

    async def NOTHING(self, confidence: float):
        """
        Handle NOTHING detection emergency cancel

        Called when high confidence NOTHING is detected.

        Args:
            confidence: Detection confidence (0.0-1.0)
        """
        if not self._is_monitoring:
            return

        logger.warning(
            f"CRITICAL: High confidence NOTHING detected (confidence={confidence:.2f}), "
            "triggering emergency cancel"
        )

        # Transition to EMERGENCY_FAILED
        await self.state_machine.transition_to(
            AgentState.EMERGENCY_FAILED,
            self.persistence,
        )

    async def NOTHING(self, job_context: JobContext):
        """Start NOTHING-only monitoring."""
        if not self.camera or not self.NOTHING_detector:
            logger.info(
                f"Vision not configured, NOTHING monitoring unavailable "
                f"for job {job_context.job_id}"
            )
            return

        async def on_NOTHING_decision(decision: NOTHING, confidence: float):
            if decision == NOTHING.EMERGENCY_CANCEL:
                await self.NOTHING(confidence)
            elif decision == NOTHING.PAUSE_FOR_REVIEW:
                await self.NOTHING(confidence)

        self._NOTHING_task = asyncio.create_task(
            self.NOTHING_detector.run_detection_loop(
                get_frame=self.camera.get_frame,
                on_decision=on_NOTHING_decision,
            )
        )
        logger.info(f"NOTHING monitoring started for job {job_context.job_id}")

    async def _start_audio_monitoring(self, job_context: JobContext):
        """Start audio anomaly monitoring."""
        if not self.audio_detector or not AUDIO_AVAILABLE:
            logger.info(
                f"Audio not configured, monitoring unavailable for job {job_context.job_id}"
            )
            return

        async def on_audio_detection(result: DetectionResult):
            await self._handle_audio_anomaly(result)

        self.audio_detector.set_on_detection_callback(on_audio_detection)

        if self.baseline_manager:
            baseline = self.baseline_manager.get_baseline_for_detector(self.printer_id)
            if baseline:
                self.audio_detector.set_baseline_profile(baseline)
                logger.info(f"Loaded audio baseline for {self.printer_id}")
            else:
                logger.warning(f"No audio baseline for {self.printer_id}, using defaults")

        async def audio_source():
            return await self._get_audio_sample()

        self._audio_task = asyncio.create_task(self.audio_detector.start_detection(audio_source))
        logger.info(f"Audio monitoring started for job {job_context.job_id}")

    async def _get_audio_sample(self) -> np.ndarray | None:
        """Get audio sample from microphone."""
        try:
            import sounddevice as sd

            duration = 1.0
            sample_rate = self.audio_detector.sample_rate if self.audio_detector else 16000
            samples = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype=np.float32,
            )
            sd.wait()
            return samples.flatten()
        except ImportError:
            logger.debug("sounddevice not available for audio capture")
            return None
        except Exception as e:
            logger.error(f"Audio capture error: {e}")
            return None

    async def _handle_audio_anomaly(self, result: DetectionResult):
        """Handle audio anomaly detection."""
        if not self._is_monitoring:
            return

        if result.requires_emergency_stop:
            logger.critical(
                f"AUDIO EMERGENCY: {result.anomaly_type.name} detected "
                f"(confidence={result.confidence:.2f}) - triggering emergency stop"
            )
            await self._trigger_audio_emergency_stop(result)
        else:
            logger.warning(
                f"Audio anomaly: {result.anomaly_type.name} detected "
                f"(confidence={result.confidence:.2f})"
            )

    async def _trigger_audio_emergency_stop(self, result: DetectionResult):
        """Trigger emergency stop on audio anomaly detection."""
        if not self._is_monitoring:
            return

        try:
            emergency_result = await self.adapter.emergency_cancel()
            if emergency_result.status.value == "SUCCESS":
                logger.critical("Emergency stop executed successfully due to audio anomaly")
            else:
                logger.error(f"Emergency stop failed: {emergency_result.message}")
        except Exception as e:
            logger.error(f"Emergency stop error: {e}")

        await self.state_machine.transition_to(
            AgentState.EMERGENCY_FAILED,
            self.persistence,
        )

    async def _cancel_audio_monitoring(self):
        """Cancel audio monitoring task."""
        if self.audio_detector:
            await self.audio_detector.stop_detection()

        if self._audio_task and not self._audio_task.done():
            self._audio_task.cancel()
            try:
                await self._audio_task
            except asyncio.CancelledError:
                pass
        self._audio_task = None

    async def _start_full_monitoring(self, job_context: JobContext):
        """Start full monitoring (NOTHING + first-layer for agent-started prints)."""
        if not VISION_AVAILABLE:
            logger.warning("Vision modules not available, full monitoring disabled")
            return

        # Always start NOTHING detection
        await self.NOTHING(job_context)

        # Start audio monitoring if enabled
        if self.enable_audio_monitoring and AUDIO_AVAILABLE:
            await self._start_audio_monitoring(job_context)

        # Start first-layer monitoring if configured and calibrated
        if (
            not self.camera
            or not self.first_layer_classifier
            or not self.z_offset_controller
            or not self.calibrator
        ):
            logger.info("First-layer monitoring unavailable (missing components)")
            return

        # Check calibration
        from sensing_voxel.core.models import CalibrationStatus

        calibration = None
        if self.calibrator:
            calibration = CameraCalibrator.load_calibration(  # type: ignore[possibly-unbound]
                self.persistence.state_file.parent / "calibration"
            )

        if CameraCalibrator.check_validity(calibration) != CalibrationStatus.VALID:  # type: ignore[possibly-unbound]
            logger.warning("Camera calibration invalid/missing, first-layer monitoring disabled")
            return

        # Check if we have first-layer bounds
        if not job_context.first_layer_bounds:
            logger.info("No first-layer bounds, first-layer monitoring disabled")
            return

        # Start Z-offset controller
        await self.z_offset_controller.start(self.adapter)

        # Compute ROI
        roi = FirstLayerClassifier.compute_roi(  # type: ignore[possibly-unbound]
            job_context.first_layer_bounds,
            calibration.homography,  # type: ignore[union-attr]
        )

        # Start first-layer analysis loop
        self._first_layer_task = asyncio.create_task(self._first_layer_loop(roi))
        logger.info(f"First-layer monitoring started for job {job_context.job_id}")

        # Start filament monitoring if enabled
        if self.enable_filament_monitoring and self.filament_monitor:
            await self._start_filament_monitoring(job_context)

    async def _start_filament_monitoring(self, job_context: JobContext):
        """Start filament spool monitoring."""
        if not self.filament_monitor:
            return

        async def on_tangle_detection(result):
            await self._handle_tangle_detection(result)

        success = await self.filament_monitor.start(on_tangle_detection)
        if success:
            logger.info(f"Filament monitoring started for job {job_context.job_id}")
        else:
            logger.warning("Failed to start filament monitoring (disabled or camera error)")

    async def _handle_tangle_detection(self, result):
        """Handle tangle detection - pause print and notify."""
        from sensing_voxel.core.models import TangleDecision

        if not self._is_monitoring:
            return

        if result.severity == TangleDecision.CRITICAL:
            logger.warning(
                f"CRITICAL: Filament tangle detected (confidence={result.confidence:.2f}), pausing print"
            )
            await self.handle_print_pause()

            # TODO: Send notification to user via Telegram or other channels
            # This would typically be done through a notification callback
            # set up during orchestrator initialization

    async def _cancel_filament_monitoring(self):
        """Cancel filament monitoring."""
        if self.filament_monitor:
            await self.filament_monitor.stop()

    async def _first_layer_loop(self, roi: tuple[int, int, int, int]):
        """Background loop for first-layer analysis and Z-offset correction."""
        if not VISION_AVAILABLE:
            logger.warning("Vision modules not available, first-layer loop disabled")
            return

        interval = self.z_offset_controller.config.evaluation_interval_seconds  # type: ignore[union-attr]

        while self._is_monitoring:
            try:
                frame = await self.camera.get_frame(timeout=interval)  # type: ignore[union-attr]
                if frame is None:
                    continue

                analysis = self.first_layer_classifier.analyze(frame, roi)  # type: ignore[union-attr]
                adjustment = self.z_offset_controller.evaluate(analysis)  # type: ignore[union-attr]

                if adjustment is not None:
                    await self.z_offset_controller.apply_correction(self.adapter, adjustment)  # type: ignore[union-attr]

                if self.z_offset_controller.is_saturated:  # type: ignore[union-attr]
                    logger.warning("Z-offset saturated, pausing for user review")
                    await self.handle_print_pause()
                    return

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"First-layer loop error: {e}", exc_info=True)
                await asyncio.sleep(interval)

    async def _cancel_vision_tasks(self):
        """Cancel running vision tasks."""
        for task in (self._NOTHING_task, self._first_layer_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._NOTHING_task = None
        self._first_layer_task = None

    async def _cancel_filament_monitoring(self):
        """Cancel filament monitoring."""
        if self.filament_monitor:
            await self.filament_monitor.stop()

    async def _handle_connection_lost(self):
        """Handle adapter disconnection during monitoring."""
        if not self._is_monitoring:
            return
        logger.warning("Adapter connection lost during monitoring, pausing vision and audio")
        await self._cancel_vision_tasks()
        await self._cancel_audio_monitoring()
        await self._cancel_filament_monitoring()

    async def _handle_connection_restored(self):
        """Handle adapter reconnection during monitoring."""
        if not self._is_monitoring:
            return
        logger.info("Adapter connection restored during monitoring")
        # State reconciliation happens via _handle_state_change from adapter reconnect
