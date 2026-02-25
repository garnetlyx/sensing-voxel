"""
Configuration management for Sensing Voxel
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class KlipperConfig:
    """Klipper/Moonraker adapter configuration"""
    host: str = "localhost"
    port: int = 7125
    api_key: Optional[str] = None
    ws_reconnect_max_attempts: int = 10
    ws_reconnect_base_delay: float = 1.0
    ws_reconnect_max_delay: float = 30.0
    ws_reconnect_factor: float = 1.8
    ws_reconnect_jitter: float = 0.25
    compatibility_mode: bool = False  # For Elegoo/Neptune modified payloads


@dataclass
class BambuConfig:
    """Bambu adapter configuration"""
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    mqtt_username: Optional[str] = None
    mqtt_password: Optional[str] = None
    ftp_host: str = "localhost"
    ftp_port: int = 21
    ftp_username: Optional[str] = None
    ftp_password: Optional[str] = None
    mqtt_reconnect_max_attempts: int = 10
    mqtt_reconnect_base_delay: float = 1.0
    mqtt_reconnect_max_delay: float = 30.0


@dataclass
class TelegramConfig:
    """Telegram bot configuration"""
    bot_token: str
    allowed_users: List[int] = field(default_factory=list)
    approval_timeout_seconds: int = 600  # 10 minutes
    NOTHING: int = 300  # 5 minutes


@dataclass
class CameraConfig:
    """Camera configuration"""
    rtsp_url: str
    calibration_expiry_days: int = 30
    frame_width: int = 1920
    frame_height: int = 1080
    fps: int = 10


@dataclass
class SlicerConfig:
    """Slicer configuration"""
    prusaslicer_path: str = "prusa-slicer"
    profile_path: Optional[str] = None
    output_dir: str = "data/sliced"


@dataclass
class StorageConfig:
    """Storage configuration"""
    data_dir: str = "data"
    state_file: str = "agent_state.json"
    calibration_dir: str = "calibration"
    models_dir: str = "models"
    uploads_dir: str = "uploads"


@dataclass
class Config:
    """Main configuration"""
    klipper: Optional[KlipperConfig] = None
    bambu: Optional[BambuConfig] = None
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    camera: Optional[CameraConfig] = None
    slicer: SlicerConfig = field(default_factory=SlicerConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    debug: bool = False

    @property
    def state_file_path(self) -> Path:
        """Get full path to state file"""
        return Path(self.storage.data_dir) / self.storage.state_file

    @property
    def calibration_path(self) -> Path:
        """Get full path to calibration directory"""
        return Path(self.storage.data_dir) / self.storage.calibration_dir

    @property
    def models_path(self) -> Path:
        """Get full path to models directory"""
        return Path(self.storage.data_dir) / self.storage.models_dir

    @property
    def uploads_path(self) -> Path:
        """Get full path to uploads directory"""
        return Path(self.storage.data_dir) / self.storage.uploads_dir
