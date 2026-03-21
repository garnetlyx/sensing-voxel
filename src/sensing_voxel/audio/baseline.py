"""
Baseline profile management for audio anomaly detection.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BaselineProfile:
    """
    Baseline audio profile for a specific printer.

    Captures normal operating sounds to reduce false positives
    by learning per-printer acoustic signatures.
    """

    printer_id: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    sample_count: int = 0
    sample_rate: int = 16000
    duration_seconds: float = 0.0

    # Statistical features of normal operation
    mean_spectrum: list[float] = field(default_factory=list)
    std_spectrum: list[float] = field(default_factory=list)
    energy_mean: float = 0.0
    energy_std: float = 0.0
    zcr_mean: float = 0.0  # Zero crossing rate
    zcr_std: float = 0.0

    # Per-anomaly-type thresholds learned from baseline
    anomaly_thresholds: dict[str, float] = field(default_factory=dict)

    # YAMNet class scores during normal operation
    normal_class_scores: dict[str, float] = field(default_factory=dict)

    metadata: dict[str, Any] = field(default_factory=dict)

    def is_stale(self, max_age_days: int = 30) -> bool:
        """Check if baseline profile is older than max_age_days."""
        age = datetime.now() - self.updated_at
        return age > timedelta(days=max_age_days)

    def to_dict(self) -> dict[str, Any]:
        """Convert profile to dictionary."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaselineProfile:
        """Create profile from dictionary."""
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)


class BaselineManager:
    """
    Manages baseline audio profiles for multiple printers.

    Handles capture, storage, and retrieval of per-printer
    baseline profiles to reduce false-positive anomaly detections.
    """

    def __init__(self, storage_dir: str | Path = "data/audio_baselines"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._profiles: dict[str, BaselineProfile] = {}
        self._is_capturing = False

    def get_profile(self, printer_id: str) -> BaselineProfile | None:
        """
        Get baseline profile for a printer.

        Loads from disk if not in memory cache.
        """
        if printer_id in self._profiles:
            return self._profiles[printer_id]

        profile_path = self._get_profile_path(printer_id)
        if profile_path.exists():
            try:
                with open(profile_path, "r") as f:
                    data = json.load(f)
                profile = BaselineProfile.from_dict(data)
                self._profiles[printer_id] = profile
                logger.info(f"Loaded baseline profile for {printer_id}")
                return profile
            except Exception as e:
                logger.error(f"Failed to load profile for {printer_id}: {e}")
                return None

        return None

    def save_profile(self, profile: BaselineProfile) -> bool:
        """Save baseline profile to disk."""
        try:
            profile_path = self._get_profile_path(profile.printer_id)
            with open(profile_path, "w") as f:
                json.dump(profile.to_dict(), f, indent=2)
            self._profiles[profile.printer_id] = profile
            logger.info(f"Saved baseline profile for {profile.printer_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save profile: {e}")
            return False

    def delete_profile(self, printer_id: str) -> bool:
        """Delete baseline profile."""
        profile_path = self._get_profile_path(printer_id)
        if profile_path.exists():
            try:
                profile_path.unlink()
                self._profiles.pop(printer_id, None)
                logger.info(f"Deleted baseline profile for {printer_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete profile: {e}")
                return False
        return False

    def list_profiles(self) -> list[str]:
        """List all printer IDs with saved profiles."""
        profiles = []
        for path in self.storage_dir.glob("*.json"):
            profiles.append(path.stem)
        return profiles

    def _get_profile_path(self, printer_id: str) -> Path:
        """Get file path for printer profile."""
        safe_id = "".join(c for c in printer_id if c.isalnum() or c in "-_")
        return self.storage_dir / f"{safe_id}.json"

    async def capture_baseline(
        self,
        printer_id: str,
        audio_source: Any,
        duration_seconds: float = 60.0,
        callback: Any = None,
    ) -> BaselineProfile | None:
        """
        Capture baseline audio profile for a printer.

        Records audio samples during normal operation and computes
        statistical features to distinguish from anomalies.

        Args:
            printer_id: Unique identifier for the printer
            audio_source: Async generator yielding audio chunks
            duration_seconds: How long to record for baseline
            callback: Optional progress callback (current, total)

        Returns:
            BaselineProfile if successful, None otherwise
        """
        if self._is_capturing:
            logger.warning("Baseline capture already in progress")
            return None

        self._is_capturing = True
        logger.info(f"Starting baseline capture for {printer_id} ({duration_seconds}s)")

        try:
            samples_list: list[np.ndarray] = []
            start_time = datetime.now()
            elapsed = 0.0

            async for chunk in audio_source:
                if elapsed >= duration_seconds:
                    break

                if isinstance(chunk, np.ndarray):
                    samples_list.append(chunk)
                elapsed = (datetime.now() - start_time).total_seconds()

                if callback:
                    await callback(elapsed, duration_seconds)

            if not samples_list:
                logger.error("No audio samples captured")
                return None

            all_samples = np.concatenate(samples_list)
            profile = self._compute_profile(printer_id, all_samples)

            self.save_profile(profile)
            logger.info(f"Baseline capture complete for {printer_id}")
            return profile

        except Exception as e:
            logger.error(f"Baseline capture failed: {e}", exc_info=True)
            return None
        finally:
            self._is_capturing = False

    def _compute_profile(
        self, printer_id: str, samples: np.ndarray, sample_rate: int = 16000
    ) -> BaselineProfile:
        """Compute baseline profile from audio samples."""
        samples = self._normalize_audio(samples)

        # Compute spectral features
        n_fft = 512
        hop_length = n_fft // 4

        stft = np.array(
            [
                np.abs(np.fft.rfft(samples[i : i + n_fft]))
                for i in range(0, len(samples) - n_fft, hop_length)
                if len(samples[i : i + n_fft]) == n_fft
            ]
        )

        if len(stft) > 0:
            mean_spectrum = stft.mean(axis=0).tolist()
            std_spectrum = stft.std(axis=0).tolist()
        else:
            mean_spectrum = []
            std_spectrum = []

        # Compute energy statistics
        frame_size = sample_rate // 10  # 100ms frames
        energies = [
            np.sum(samples[i : i + frame_size] ** 2)
            for i in range(0, len(samples) - frame_size, frame_size)
        ]
        energy_mean = float(np.mean(energies)) if energies else 0.0
        energy_std = float(np.std(energies)) if energies else 0.0

        # Compute zero crossing rate
        zcr = np.sum(np.abs(np.diff(np.sign(samples)))) / (2 * len(samples))
        zcr_mean = float(zcr)
        zcr_std = 0.1  # Default small std

        # Set anomaly thresholds based on normal operation
        anomaly_thresholds = {
            "grinding": 0.6,
            "crashing": 0.7,
            "motor_fault": 0.65,
            "vibration": energy_mean + 2 * energy_std if energy_std > 0 else 0.5,
        }

        return BaselineProfile(
            printer_id=printer_id,
            sample_count=len(samples),
            sample_rate=sample_rate,
            duration_seconds=len(samples) / sample_rate,
            mean_spectrum=mean_spectrum,
            std_spectrum=std_spectrum,
            energy_mean=energy_mean,
            energy_std=energy_std,
            zcr_mean=zcr_mean,
            zcr_std=zcr_std,
            anomaly_thresholds=anomaly_thresholds,
            normal_class_scores={},
            metadata={"capture_method": "continuous_recording"},
        )

    def _normalize_audio(self, samples: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range."""
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

    def get_baseline_for_detector(self, printer_id: str) -> dict[str, Any] | None:
        """
        Get baseline data formatted for AudioAnomalyDetector.

        Returns dict with anomaly thresholds and normal operation stats.
        """
        profile = self.get_profile(printer_id)
        if not profile:
            return None

        return {
            "anomaly_thresholds": profile.anomaly_thresholds,
            "energy_mean": profile.energy_mean,
            "energy_std": profile.energy_std,
            "zcr_mean": profile.zcr_mean,
            "zcr_std": profile.zcr_std,
            "sample_rate": profile.sample_rate,
        }

    def is_profile_valid(self, printer_id: str, max_age_days: int = 30) -> bool:
        """Check if profile exists and is not stale."""
        profile = self.get_profile(printer_id)
        if not profile:
            return False
        return not profile.is_stale(max_age_days)

    def update_anomaly_threshold(
        self, printer_id: str, anomaly_type: str, threshold: float
    ) -> bool:
        """Update anomaly threshold for a specific anomaly type."""
        profile = self.get_profile(printer_id)
        if not profile:
            return False

        profile.anomaly_thresholds[anomaly_type] = threshold
        profile.updated_at = datetime.now()
        return self.save_profile(profile)

    @property
    def is_capturing(self) -> bool:
        """Check if baseline capture is in progress."""
        return self._is_capturing
