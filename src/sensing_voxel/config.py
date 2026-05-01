"""
Configuration management for Sensing Voxel
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any


class LlmProvider(Enum):
    """LLM backend provider"""

    MINIMAX = "minimax"
    OLLAMA = "ollama"


class ImageTo3DProvider(Enum):
    """Image-to-3D API provider"""

    TRIPOSR = "triposr"
    MESHY = "meshy"
    TRIPO3D = "tripo3d"


class SlicerBackend(Enum):
    AUTO = "auto"
    PRUSASLICER = "prusaslicer"
    ORCASLICER = "orcaslicer"


@dataclass
class KlipperConfig:
    """Klipper/Moonraker adapter configuration"""

    name: str = "klipper"
    host: str = "localhost"
    port: int = 7125
    api_key: str | None = None
    base_url: str | None = None  # Override host:port (e.g. "http://printer.example")
    ws_reconnect_max_attempts: int = 10
    ws_reconnect_base_delay: float = 1.0
    ws_reconnect_max_delay: float = 30.0
    ws_reconnect_factor: float = 1.8
    ws_reconnect_jitter: float = 0.25
    compatibility_mode: bool = False  # For Elegoo/Neptune modified payloads


@dataclass
class BambuConfig:
    """Bambu adapter configuration"""

    name: str = "bambu"
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    mqtt_username: str | None = None
    mqtt_password: str | None = None
    ftp_host: str = "localhost"
    ftp_port: int = 21
    ftp_username: str | None = None
    ftp_password: str | None = None
    mqtt_reconnect_max_attempts: int = 10
    mqtt_reconnect_base_delay: float = 1.0
    mqtt_reconnect_max_delay: float = 30.0


@dataclass
class TelegramConfig:
    """Telegram bot configuration"""

    bot_token: str
    allowed_users: list[int] = field(default_factory=list)
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
class TapoCameraConfig:
    """Tapo PTZ camera configuration"""

    host: str = ""
    user: str = ""
    password: str = ""
    onvif_port: int = 2020


@dataclass
class FilamentMonitorConfig:
    """Filament spool monitor configuration"""

    enabled: bool = False
    usb_camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    fps: int = 5
    check_interval_seconds: float = 30.0
    spool_full_weight_g: float = 1000.0
    spool_empty_weight_g: float = 250.0
    filament_diameter_mm: float = 1.75
    filament_density_g_cm3: float = 1.24
    tangle_detection_threshold: float = 0.7
    tangle_confirmation_frames: int = 3
    min_filament_safety_margin_percent: float = 10.0


@dataclass
class SlicerConfig:
    backend: SlicerBackend = SlicerBackend.AUTO
    prusaslicer_path: str = "prusa-slicer"
    orcaslicer_path: str = "orca-slicer"
    profile_path: str | None = None
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
class DashboardConfig:
    """Web dashboard configuration"""

    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8080
    api_token: str | None = None  # None = no auth (dev mode)
    camera_rtsp_url: str | None = None
    camera_mjpeg_fps: int = 5
    camera_mjpeg_quality: int = 80


@dataclass
class ShellToolConfig:
    """LLM bash execution tool configuration."""

    enabled: bool = False
    cwd: str = "/app"
    timeout_seconds: float = 10.0
    max_output_chars: int = 8000


@dataclass
class WebSearchToolConfig:
    """LLM web search tool configuration."""

    enabled: bool = False
    searxng_url: str | None = None
    timeout_seconds: float = 10.0
    max_results: int = 5


@dataclass
class LlmConfig:
    """LLM API configuration (Anthropic-compatible)"""

    provider: LlmProvider = LlmProvider.MINIMAX
    api_key: str | None = None
    base_url: str = "https://api.anthropic.com"
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 2048
    temperature: float = 0.7
    timeout_seconds: int = 30
    max_conversation_turns: int = 20
    enabled: bool = False
    shell: ShellToolConfig = field(default_factory=ShellToolConfig)
    web_search: WebSearchToolConfig = field(default_factory=WebSearchToolConfig)


@dataclass
class ImageTo3DConfig:
    """Image-to-3D generation configuration"""

    enabled: bool = False
    default_provider: ImageTo3DProvider = ImageTo3DProvider.TRIPOSR
    triposr_api_key: str | None = None
    triposr_base_url: str = "https://api.stability.ai/v2beta/3d"
    meshy_api_key: str | None = None
    meshy_base_url: str = "https://api.meshy.ai/openapi/v2"
    tripo3d_api_key: str | None = None
    tripo3d_base_url: str = "https://api.tripo3d.ai/v2/openapi"
    timeout_seconds: int = 300
    enable_caching: bool = True
    cache_ttl_seconds: int = 86400  # 1 day
    enable_cost_tracking: bool = True
    max_generation_cost: float = 1.0  # Max USD per generation
    default_output_format: str = "obj"


@dataclass
class QueueConfig:
    """Multi-printer job queue configuration"""

    enabled: bool = False
    max_queue_size: int = 100
    default_priority: str = "NORMAL"
    auto_assignment: bool = True
    assignment_strategy: str = "OPTIMAL"
    enable_load_balancing: bool = True
    enable_priority_boost: bool = True
    wait_time_boost_threshold_seconds: int = 1800  # 30 minutes
    wait_time_boost_amount: int = 1
    printer_timeout_seconds: float = 300.0  # Mark printer offline after 5 minutes
    job_timeout_seconds: float = 86400.0  # 24 hours max wait time
    max_retries: int = 3
    cooldown_between_assignments_seconds: int = 5


@dataclass
class ModelingConfig:
    """3D modeling and photo-to-print pipeline configuration"""

    enabled: bool = False
    image_to_3d: ImageTo3DConfig = field(default_factory=ImageTo3DConfig)
    output_dir: str = "data/modeling"
    auto_approve: bool = False  # For testing only
    require_manual_approval: bool = True
    max_file_size_mb: int = 50
    supported_input_formats: list[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".webp"]
    )
    temp_dir: str = "data/tmp"


PrinterConfig = KlipperConfig | BambuConfig

PRINTER_TYPE_MAP = {
    "klipper": KlipperConfig,
    "bambu": BambuConfig,
}


@dataclass
class Config:
    """Main configuration"""

    printers: list[PrinterConfig] = field(default_factory=list)
    klipper: KlipperConfig | None = None
    bambu: BambuConfig | None = None
    telegram: TelegramConfig = field(default_factory=lambda: TelegramConfig(bot_token=""))
    camera: CameraConfig | None = None
    tapo_camera: TapoCameraConfig | None = None
    filament_monitor: FilamentMonitorConfig = field(default_factory=FilamentMonitorConfig)
    slicer: SlicerConfig = field(default_factory=SlicerConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    modeling: ModelingConfig = field(default_factory=ModelingConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    debug: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> Config:
        """Deserialize from a nested dict (e.g. YAML output)."""
        section_map = {
            "telegram": TelegramConfig,
            "camera": CameraConfig,
            "tapo_camera": TapoCameraConfig,
            "filament_monitor": FilamentMonitorConfig,
            "slicer": SlicerConfig,
            "storage": StorageConfig,
            "dashboard": DashboardConfig,
            "llm": LlmConfig,
        }
        parsed: dict = {}
        for key, value in data.items():
            if key == "printers" and isinstance(value, list):
                parsed["printers"] = _parse_printers_list(value)
            elif key in ("klipper", "bambu") and isinstance(value, dict):
                parsed[key] = PRINTER_TYPE_MAP[key](**value)
            elif key == "modeling" and isinstance(value, dict):
                # Handle nested image_to_3d config
                modeling_data = value.copy()
                if "image_to_3d" in modeling_data and isinstance(
                    modeling_data["image_to_3d"], dict
                ):
                    image_to_3d_data = modeling_data["image_to_3d"].copy()
                    if isinstance(image_to_3d_data.get("default_provider"), str):
                        image_to_3d_data["default_provider"] = ImageTo3DProvider(
                            image_to_3d_data["default_provider"]
                        )
                    modeling_data["image_to_3d"] = ImageTo3DConfig(**image_to_3d_data)
                parsed["modeling"] = ModelingConfig(**modeling_data)
            elif key == "queue" and isinstance(value, dict):
                parsed["queue"] = QueueConfig(**value)
            elif key == "slicer" and isinstance(value, dict):
                slicer_data = value.copy()
                if isinstance(slicer_data.get("backend"), str):
                    slicer_data["backend"] = SlicerBackend(slicer_data["backend"])
                parsed["slicer"] = SlicerConfig(**slicer_data)
            elif key == "llm" and isinstance(value, dict):
                llm_data = value.copy()
                if isinstance(llm_data.get("provider"), str):
                    llm_data["provider"] = LlmProvider(llm_data["provider"])
                if isinstance(llm_data.get("shell"), dict):
                    llm_data["shell"] = ShellToolConfig(**llm_data["shell"])
                if isinstance(llm_data.get("web_search"), dict):
                    llm_data["web_search"] = WebSearchToolConfig(**llm_data["web_search"])
                parsed["llm"] = LlmConfig(**llm_data)
            elif key in section_map and isinstance(value, dict):
                parsed[key] = section_map[key](**value)
            else:
                parsed[key] = value

        config = cls(**parsed)

        # Auto-migrate legacy single-printer fields into printers list
        if not config.printers:
            if config.klipper:
                config.printers.append(config.klipper)
            if config.bambu:
                config.printers.append(config.bambu)

        return config

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


def _parse_printers_list(items: list[dict]) -> list[PrinterConfig]:
    """Parse printers list from YAML."""
    printers: list[PrinterConfig] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        item = dict(item)  # Don't mutate input
        printer_type = item.pop("type", "klipper")
        config_cls = PRINTER_TYPE_MAP.get(printer_type)
        if config_cls:
            printers.append(config_cls(**item))
    return printers
