"""
Audio anomaly detection module for Sensing Voxel

Provides YAMNet-based audio classification for detecting mechanical failures
in 3D printers, with per-printer baseline calibration for false-positive reduction.
"""

from sensing_voxel.audio.detector import AudioAnomalyDetector, DetectionResult, AnomalyType
from sensing_voxel.audio.baseline import BaselineManager, BaselineProfile

__all__ = [
    "AudioAnomalyDetector",
    "DetectionResult",
    "AnomalyType",
    "BaselineManager",
    "BaselineProfile",
]
