"""
Audio anomaly detector using YAMNet for mechanical failure detection.
"""

from __future__ import annotations

import asyncio
import io
import logging
import wave
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Coroutine

import numpy as np

logger = logging.getLogger(__name__)

YAMNET_SAMPLE_RATE = 16000
YAMNET_FRAME_LENGTH = 0.975  # seconds
YAMNET_STFT_LENGTH = 0.025  # seconds

ANOMALY_SOUND_CLASSES = {
    "grinding": ["Grinding", "Scraping", "Sawing"],
    "crashing": ["Crash", "Banging", "Thud", "Thump"],
    "motor_fault": ["Buzz", "Hum", "Whir", "Mechanisms"],
    "vibration": ["Rattle", "Vibration", "Shatter"],
}

NORMAL_SOUND_CLASSES = {
    "normal_operation": ["Printing", "Machine", "Fan"],
    "quiet": ["Silence", "Background noise"],
}


class AnomalyType(Enum):
    """Types of audio anomalies detected."""

    GRINDING = auto()
    CRASHING = auto()
    MOTOR_FAULT = auto()
    VIBRATION = auto()
    UNKNOWN = auto()


@dataclass
class DetectionResult:
    """Result of audio anomaly detection."""

    is_anomaly: bool
    confidence: float  # 0.0 to 1.0
    anomaly_type: AnomalyType
    detected_classes: dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    requires_emergency_stop: bool = False


@dataclass
class AudioFrame:
    """Audio frame with metadata."""

    samples: np.ndarray
    timestamp: datetime
    sample_rate: int


class AudioAnomalyDetector:
    """
    YAMNet-based audio anomaly detector for 3D printer monitoring.

    Detects mechanical failures like grinding, crashing, and motor faults
    by analyzing audio from a microphone. Uses temporal confirmation
    to reduce false positives.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 975,
        confidence_threshold: float = 0.7,
        emergency_threshold: float = 0.85,
        temporal_confirmation_count: int = 2,
        temporal_window_seconds: float = 5.0,
        model_path: str | None = None,
    ):
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.confidence_threshold = confidence_threshold
        self.emergency_threshold = emergency_threshold
        self.temporal_confirmation_count = temporal_confirmation_count
        self.temporal_window_seconds = temporal_window_seconds
        self.model_path = model_path

        self._yamnet_model: Any | None = None
        self._class_names: list[str] | None = None
        self._is_running = False
        self._detection_task: asyncio.Task | None = None
        self._audio_buffer: deque[AudioFrame] = deque(maxlen=100)
        self._recent_detections: deque[tuple[datetime, DetectionResult]] = deque(
            maxlen=temporal_confirmation_count * 2
        )

        self._on_detection_callback: (
            Callable[[DetectionResult], Coroutine[Any, Any, None]] | None
        ) = None
        self._baseline_profile: dict[str, Any] | None = None

    async def load_model(self) -> bool:
        """
        Load the YAMNet model.

        Returns True if model loaded successfully, False otherwise.
        Falls back to mock detection if tensorflow-hub is not available.
        """
        try:
            import tensorflow as tf
            import tensorflow_hub as hub

            if self.model_path:
                self._yamnet_model = hub.load(self.model_path)
            else:
                self._yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

            self._class_names = self._load_class_names()
            logger.info("YAMNet model loaded successfully")
            return True

        except ImportError as e:
            logger.warning(f"TensorFlow/Hub not available: {e}. Using mock detection.")
            self._yamnet_model = None
            self._class_names = self._load_class_names()
            return True

        except Exception as e:
            logger.error(f"Failed to load YAMNet model: {e}")
            return False

    def _load_class_names(self) -> list[str]:
        """Load YAMNet class names."""
        try:
            import tensorflow as tf

            class_map_path = tf.keras.utils.get_file(
                "yamnet_class_map.csv",
                "https://raw.githubusercontent.com/tensorflow/models/master/research/"
                "audioset/yamnet/yamnet_class_map.csv",
            )
            import csv

            with open(class_map_path) as f:
                reader = csv.DictReader(f)
                return [row["display_name"] for row in reader]
        except Exception as e:
            logger.warning(f"Could not load class names: {e}")
            return []

    def set_baseline_profile(self, profile: dict[str, Any]) -> None:
        """Set baseline profile for false-positive reduction."""
        self._baseline_profile = profile
        logger.info("Baseline profile loaded")

    def set_on_detection_callback(
        self, callback: Callable[[DetectionResult], Coroutine[Any, Any, None]]
    ) -> None:
        """Set callback for anomaly detection events."""
        self._on_detection_callback = callback

    async def start_detection(
        self,
        audio_source: Callable[[], Coroutine[Any, Any, np.ndarray | None]],
    ) -> None:
        """
        Start continuous audio anomaly detection.

        Args:
            audio_source: Async callable that returns audio samples or None
        """
        if self._is_running:
            logger.warning("Detection already running")
            return

        if self._yamnet_model is None:
            await self.load_model()

        self._is_running = True
        self._detection_task = asyncio.create_task(self._detection_loop(audio_source))
        logger.info("Audio anomaly detection started")

    async def stop_detection(self) -> None:
        """Stop audio anomaly detection."""
        self._is_running = False

        if self._detection_task and not self._detection_task.done():
            self._detection_task.cancel()
            try:
                await self._detection_task
            except asyncio.CancelledError:
                pass

        self._detection_task = None
        self._audio_buffer.clear()
        self._recent_detections.clear()
        logger.info("Audio anomaly detection stopped")

    async def _detection_loop(
        self, audio_source: Callable[[], Coroutine[Any, Any, np.ndarray | None]]
    ) -> None:
        """Main detection loop."""
        chunk_samples = int(self.sample_rate * self.chunk_duration_ms / 1000)

        while self._is_running:
            try:
                samples = await audio_source()
                if samples is None:
                    await asyncio.sleep(0.1)
                    continue

                samples = self._resample_if_needed(samples)
                samples = self._normalize_audio(samples)

                if len(samples) >= chunk_samples:
                    chunk = samples[:chunk_samples]
                    frame = AudioFrame(
                        samples=chunk,
                        timestamp=datetime.now(),
                        sample_rate=self.sample_rate,
                    )
                    self._audio_buffer.append(frame)

                    result = self._detect_anomaly(chunk)

                    if result.is_anomaly:
                        confirmed = self._apply_temporal_confirmation(result)
                        if confirmed:
                            await self._handle_detection(confirmed)

                await asyncio.sleep(self.chunk_duration_ms / 1000 / 2)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Detection loop error: {e}", exc_info=True)
                await asyncio.sleep(1.0)

    def _resample_if_needed(self, samples: np.ndarray) -> np.ndarray:
        """Resample audio to target sample rate if needed."""
        if self._baseline_profile and "sample_rate" in self._baseline_profile:
            source_rate = self._baseline_profile["sample_rate"]
            if source_rate != self.sample_rate:
                from scipy import signal

                samples = signal.resample(
                    samples,
                    int(len(samples) * self.sample_rate / source_rate),
                )
        return samples

    def _normalize_audio(self, samples: np.ndarray) -> np.ndarray:
        """Normalize audio samples to [-1, 1] range."""
        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32768.0
        elif samples.dtype == np.int32:
            samples = samples.astype(np.float32) / 2147483648.0
        else:
            samples = samples.astype(np.float32)

        max_val = np.max(np.abs(samples))
        if max_val > 0:
            samples = samples / max_val

        return samples

    def _detect_anomaly(self, samples: np.ndarray) -> DetectionResult:
        """
        Detect anomalies in audio samples.

        Returns DetectionResult with anomaly classification and confidence.
        """
        if self._yamnet_model is None:
            return self._mock_detect(samples)

        try:
            import tensorflow as tf

            waveform = tf.constant(samples, dtype=tf.float32)
            scores, embeddings, spectrogram = self._yamnet_model(waveform)
            scores = scores.numpy()

            class_scores = {}
            for i, score in enumerate(scores[0]):
                if self._class_names and i < len(self._class_names):
                    class_scores[self._class_names[i]] = float(score)

            return self._classify_anomaly(class_scores)

        except Exception as e:
            logger.error(f"Detection error: {e}")
            return DetectionResult(
                is_anomaly=False,
                confidence=0.0,
                anomaly_type=AnomalyType.UNKNOWN,
            )

    def _mock_detect(self, samples: np.ndarray) -> DetectionResult:
        """Mock detection for testing without TensorFlow."""
        energy = np.mean(samples**2)

        if energy > 0.3:
            return DetectionResult(
                is_anomaly=True,
                confidence=0.8,
                anomaly_type=AnomalyType.VIBRATION,
                detected_classes={"Rattle": 0.8, "Vibration": 0.7},
            )

        return DetectionResult(
            is_anomaly=False,
            confidence=0.0,
            anomaly_type=AnomalyType.UNKNOWN,
        )

    def _classify_anomaly(self, class_scores: dict[str, float]) -> DetectionResult:
        """Classify audio based on YAMNet class scores."""
        anomaly_scores: dict[AnomalyType, float] = {}

        for anomaly_type, keywords in ANOMALY_SOUND_CLASSES.items():
            max_score = 0.0
            for keyword in keywords:
                for class_name, score in class_scores.items():
                    if keyword.lower() in class_name.lower():
                        max_score = max(max_score, score)
            if max_score > 0:
                anomaly_scores[AnomalyType[anomaly_type.upper()]] = max_score

        if not anomaly_scores:
            return DetectionResult(
                is_anomaly=False,
                confidence=0.0,
                anomaly_type=AnomalyType.UNKNOWN,
            )

        best_anomaly = max(anomaly_scores.items(), key=lambda x: x[1])
        anomaly_type, confidence = best_anomaly

        confidence = self._apply_baseline_adjustment(anomaly_type, confidence)

        requires_emergency = confidence >= self.emergency_threshold

        if anomaly_type == AnomalyType.GRINDING and confidence >= self.confidence_threshold:
            requires_emergency = True
        if anomaly_type == AnomalyType.CRASHING and confidence >= self.confidence_threshold:
            requires_emergency = True

        return DetectionResult(
            is_anomaly=confidence >= self.confidence_threshold,
            confidence=confidence,
            anomaly_type=anomaly_type,
            detected_classes={k: v for k, v in class_scores.items() if v > 0.3},
            requires_emergency_stop=requires_emergency,
        )

    def _apply_baseline_adjustment(self, anomaly_type: AnomalyType, confidence: float) -> float:
        """Adjust confidence based on baseline profile to reduce false positives."""
        if not self._baseline_profile:
            return confidence

        baseline_adjustments = self._baseline_profile.get("anomaly_thresholds", {})
        type_key = anomaly_type.name.lower()

        if type_key in baseline_adjustments:
            baseline_threshold = baseline_adjustments[type_key]
            if confidence < baseline_threshold * 1.5:
                confidence *= 0.5

        return confidence

    def _apply_temporal_confirmation(self, result: DetectionResult) -> DetectionResult | None:
        """Apply temporal confirmation to reduce false positives."""
        now = datetime.now()
        window_start = now - timedelta(seconds=self.temporal_window_seconds)

        self._recent_detections.append((now, result))

        recent_same_type = [
            (t, r)
            for t, r in self._recent_detections
            if t >= window_start and r.anomaly_type == result.anomaly_type
        ]

        if len(recent_same_type) >= self.temporal_confirmation_count:
            avg_confidence = sum(r.confidence for _, r in recent_same_type) / len(recent_same_type)
            confirmed_result = DetectionResult(
                is_anomaly=True,
                confidence=avg_confidence,
                anomaly_type=result.anomaly_type,
                detected_classes=result.detected_classes,
                timestamp=now,
                requires_emergency_stop=result.requires_emergency_stop
                or avg_confidence >= self.emergency_threshold,
            )
            return confirmed_result

        return None

    async def _handle_detection(self, result: DetectionResult) -> None:
        """Handle confirmed anomaly detection."""
        if result.requires_emergency_stop:
            logger.critical(
                f"AUDIO EMERGENCY: {result.anomaly_type.name} detected "
                f"(confidence={result.confidence:.2f})"
            )
        else:
            logger.warning(
                f"Audio anomaly: {result.anomaly_type.name} detected "
                f"(confidence={result.confidence:.2f})"
            )

        if self._on_detection_callback:
            try:
                await self._on_detection_callback(result)
            except Exception as e:
                logger.error(f"Detection callback error: {e}")

    def analyze_sample(
        self, audio_data: bytes | np.ndarray, sample_rate: int = 16000
    ) -> DetectionResult:
        """
        Analyze a single audio sample.

        Args:
            audio_data: Audio data as bytes (WAV) or numpy array
            sample_rate: Sample rate of the audio

        Returns:
            DetectionResult with anomaly classification
        """
        if isinstance(audio_data, bytes):
            samples = self._load_wav_bytes(audio_data)
        else:
            samples = audio_data

        samples = self._resample_if_needed(samples)
        samples = self._normalize_audio(samples)

        return self._detect_anomaly(samples)

    def _load_wav_bytes(self, wav_bytes: bytes) -> np.ndarray:
        """Load audio from WAV bytes."""
        with io.BytesIO(wav_bytes) as wav_io:
            with wave.open(wav_io, "rb") as wav_file:
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                n_frames = wav_file.getnframes()
                raw_data = wav_file.readframes(n_frames)

                if sample_width == 2:
                    dtype = np.int16
                elif sample_width == 4:
                    dtype = np.int32
                else:
                    dtype = np.uint8

                samples = np.frombuffer(raw_data, dtype=dtype)

                if n_channels > 1:
                    samples = samples.reshape(-1, n_channels).mean(axis=1)

                return samples.astype(np.float32)

    @property
    def is_running(self) -> bool:
        """Check if detection is running."""
        return self._is_running

    @property
    def recent_detection_count(self) -> int:
        """Get number of recent detections."""
        return len(self._recent_detections)
