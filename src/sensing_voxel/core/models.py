"""
Core data models for Sensing Voxel

Enums, data classes, and event types used across components.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any


class AgentState(Enum):
    """Agent state machine states"""

    IDLE = auto()
    RECEIVING = auto()
    SLICING = auto()
    AWAITING_APPROVAL = auto()
    UPLOADING = auto()
    PRINTING = auto()
    MONITORING = auto()
    PAUSED_PENDING_USER = auto()
    COMPLETE = auto()
    FAILED = auto()
    EMERGENCY_FAILED = auto()
    ERROR = auto()


class NormalizedPrinterState(Enum):
    """Normalized printer states across adapters"""

    PRINTING = auto()
    PAUSED = auto()
    IDLE_NO_ACTIVE_PRINT = auto()
    TERMINAL = auto()  # Complete or cancelled
    ERROR = auto()
    UNKNOWN = auto()


class DecisionAction(Enum):
    """Decision actions from user or timeout"""

    APPROVE = auto()
    REJECT = auto()
    RESUME = auto()
    CANCEL = auto()
    TIMEOUT_APPROVAL = auto()
    NOTHING = auto()


class ConsumeResult(Enum):
    """Result of consume_decision()"""

    CONSUMED = auto()
    ALREADY_HANDLED = auto()
    INVALID_STATE = auto()
    INVALID_TOKEN = auto()
    INVALID_JOB = auto()


class FileValidationStatus(Enum):
    """File validation status"""

    VALID = auto()
    INVALID_TYPE = auto()
    TOO_LARGE = auto()
    CORRUPT = auto()
    CONVERSION_FAILED = auto()


class ModelFileType(Enum):
    """Supported 3D model file types"""

    STL = auto()
    STEP = auto()
    STP = auto()
    THREE_MF = auto()  # 3MF (3D Manufacturing Format)
    GCODE = auto()

    @classmethod
    def from_extension(cls, ext: str) -> "ModelFileType | None":
        """Get file type from extension (without dot)"""
        ext_lower = ext.lower().lstrip(".")
        mapping = {
            "stl": cls.STL,
            "step": cls.STEP,
            "stp": cls.STP,
            "3mf": cls.THREE_MF,
            "gcode": cls.GCODE,
            "g": cls.GCODE,
            "gco": cls.GCODE,
        }
        return mapping.get(ext_lower)

    @property
    def is_slicable(self) -> bool:
        """Whether this file type can be sliced (3D model vs G-code)"""
        return self in {self.STL, self.STEP, self.STP, self.THREE_MF}


class SlicingStatus(Enum):
    """Slicing pipeline status"""

    PENDING = auto()
    IN_PROGRESS = auto()
    SUCCESS = auto()
    FAILURE = auto()
    TIMEOUT = auto()


class VisionConfidenceLevel(Enum):
    """Vision confidence levels"""

    HIGH = auto()  # >0.85
    MODERATE = auto()  # 0.6-0.85
    LOW = auto()  # <0.6


class FirstLayerQuality(Enum):
    """First layer quality classification"""

    EXCELLENT = auto()
    GOOD = auto()
    ACCEPTABLE = auto()
    POOR = auto()


class NOTHING(Enum):
    """NOTHING detection decision"""

    CONTINUE = auto()  # No action
    PAUSE_FOR_REVIEW = auto()  # Moderate confidence
    EMERGENCY_CANCEL = auto()  # High confidence


class CalibrationStatus(Enum):
    """Camera calibration status"""

    VALID = auto()
    EXPIRED = auto()
    MISSING = auto()
    INVALID = auto()


class ZOffsetStatus(Enum):
    """Z-offset status"""

    CLEAN = auto()
    DIRTY = auto()
    UNKNOWN = auto()


class FilamentSpoolStatus(Enum):
    """Filament spool status"""

    SUFFICIENT = auto()  # Enough filament for estimated job
    LOW = auto()  # Running low, warn user
    INSUFFICIENT = auto()  # Not enough for job
    UNKNOWN = auto()  # Cannot determine


class TangleDecision(Enum):
    """Tangle detection decision"""

    NONE = auto()  # No tangle detected
    WARNING = auto()  # Possible tangle, monitor
    CRITICAL = auto()  # Clear tangle detected, pause print


class CommandResultStatus(Enum):
    """Command execution status"""

    SUCCESS = auto()
    FAILURE = auto()
    NOT_SUPPORTED = auto()


class LlmProvider(Enum):
    """LLM backend provider"""

    MINIMAX = "minimax"
    OLLAMA = "ollama"


@dataclass
class JobContext:
    """Context for a print job"""

    job_id: str
    job_file: str  # Path to STL or G-code
    uploaded_at: datetime
    expires_at: datetime | None = None
    decision_token: str | None = None
    decision_consumed: bool = False
    decision_action: DecisionAction | None = None
    decision_consumed_at: datetime | None = None
    gcode_path: str | None = None
    gcode_hash: str | None = None
    first_layer_params: FirstLayerParams | None = None
    first_layer_bounds: FirstLayerBounds | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    slicing_status: SlicingStatus = SlicingStatus.PENDING
    file_validation_status: FileValidationStatus = FileValidationStatus.VALID
    is_external: bool = False  # True for externally-started prints
    external_monitoring: bool = False  # Policy flag (True for external prints)

    # Immutable fields that should not be modified after creation
    _immutable_fields = {"job_id", "job_file", "uploaded_at", "decision_token"}

    def __post_init__(self):
        """Validate job_id is safe for file path usage"""
        if not self.job_id:
            raise ValueError("job_id cannot be empty")
        # Ensure job_id doesn't contain path traversal sequences
        if "/" in self.job_id or "\\" in self.job_id or ".." in self.job_id:
            raise ValueError(
                f"job_id contains invalid characters for safe file path usage: {self.job_id}"
            )

    def __setattr__(self, key, value):
        # Block modification of immutable fields after initialization
        if key in self._immutable_fields and key in self.__dict__:
            raise AttributeError(f"Cannot modify immutable field '{key}'")
        super().__setattr__(key, value)


@dataclass
class PrintProgress:
    """Printer progress information"""

    state: NormalizedPrinterState
    filename: str | None = None
    progress: float | None = None  # 0.0 to 1.0
    print_time: float | None = None  # seconds
    current_layer: int | None = None
    total_layers: int | None = None
    nozzle_temp: float | None = None
    nozzle_target: float | None = None
    bed_temp: float | None = None
    bed_target: float | None = None


@dataclass
class TelemetryData:
    """Real-time telemetry from printer"""

    timestamp: datetime
    temperatures: dict[str, float] = field(default_factory=dict)
    position: dict[str, float] | None = None
    velocity: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FirstLayerParams:
    """First layer parameters from G-code"""

    layer_height: float
    nozzle_temp: float
    bed_temp: float
    speed: float
    fan_speed: float | None = None


@dataclass
class FirstLayerBounds:
    """Bounding box of first layer moves"""

    min_x: float
    max_x: float
    min_y: float
    max_y: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class CommandResult:
    """Result of a printer command"""

    status: CommandResultStatus
    message: str | None = None
    data: Any | None = None


class AdapterCommandError(Exception):
    """Printer adapter command error"""

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class AdapterConnectionError(Exception):
    """Printer adapter connection error"""

    pass


# --- Vision & Calibration Models ---


@dataclass
class HomographyMatrix:
    """Camera homography matrix for bed calibration"""

    matrix: list[list[float]]  # 3x3 matrix
    mm_per_pixel: float
    camera_id: str | None = None
    calibration_timestamp: datetime = field(default_factory=datetime.now)
    valid_until: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_valid(self, now: datetime | None = None) -> bool:
        """Check if calibration is still valid"""
        now = now or datetime.now()
        if self.valid_until is None:
            return True
        return now < self.valid_until


@dataclass
class CameraCalibration:
    """Complete camera calibration artifact"""

    homography: HomographyMatrix
    roi_points: list[tuple[float, float]] | None = None  # Checkerboard corners
    error_mm: float | None = None  # Reprojection error
    notes: str | None = None


@dataclass
class NOTHING:
    """Result of NOTHING detection inference"""

    confidence: float  # 0.0 to 1.0
    bounding_box: tuple[float, float, float, float] | None = None  # x, y, w, h
    timestamp: datetime = field(default_factory=datetime.now)
    frame_id: int | None = None

    @property
    def confidence_level(self) -> VisionConfidenceLevel:
        """Get confidence level from raw confidence"""
        if self.confidence > 0.85:
            return VisionConfidenceLevel.HIGH
        elif self.confidence > 0.6:
            return VisionConfidenceLevel.MODERATE
        else:
            return VisionConfidenceLevel.LOW


@dataclass
class FirstLayerAnalysisResult:
    """Result of first-layer quality analysis"""

    quality: FirstLayerQuality
    confidence: float  # 0.0 to 1.0
    coverage_ratio: float | None = None  # ROI coverage (0.0 to 1.0)
    defects: list[str] = field(default_factory=list)  # e.g., ["under_extrusion", "gaps"]
    timestamp: datetime = field(default_factory=datetime.now)
    roi: tuple[float, float, float, float] | None = None  # x, y, w, h


@dataclass
class VisionFrame:
    """Single frame from camera stream"""

    frame_data: Any  # OpenCV Mat or bytes
    timestamp: datetime = field(default_factory=datetime.now)
    frame_id: int = 0
    width: int = 0
    height: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# --- Monitoring & Z-Offset Models ---


@dataclass
class ZOffsetState:
    """Z-offset state tracking"""

    baseline_offset: float | None = None  # Original baseline at print start
    current_offset: float = 0.0  # Cumulative applied offset
    last_known_baseline: float | None = None  # Persistent across restarts
    status: ZOffsetStatus = ZOffsetStatus.CLEAN
    dirty_since: datetime | None = None

    @property
    def is_dirty(self) -> bool:
        """Check if Z-offset is dirty"""
        return self.status == ZOffsetStatus.DIRTY

    @property
    def magnitude(self) -> float:
        """Get absolute magnitude of current offset"""
        return abs(self.current_offset)


@dataclass
class MonitoringConfig:
    """Monitoring configuration for active print"""

    NOTHING: bool = True
    enable_first_layer: bool = True  # Only for agent-started jobs
    evaluation_interval_seconds: int = 5
    cooldown_seconds: int = 10
    min_consecutive_evaluations: int = 2  # Hysteresis threshold
    NOTHING: float = 0.85
    NOTHING: float = 0.6
    z_offset_step_mm: float = 0.01
    z_offset_max_correction_mm: float = 0.05
    z_offset_saturation_count: int = 3  # Consecutive poor before pause


@dataclass
class MonitoringStats:
    """Monitoring statistics during active print"""

    frames_processed: int = 0
    NOTHING: int = 0
    first_layer_evaluations: int = 0
    z_offset_corrections: int = 0
    NOTHING: datetime | None = None
    last_first_layer_timestamp: datetime | None = None
    first_layer_active_until_layer: int | None = None  # Stop after this layer


# --- Slicer & Geometry Models ---


@dataclass
class SlicingResult:
    gcode_path: Path
    gcode_hash: str
    first_layer_params: FirstLayerParams | None = None
    first_layer_bounds: FirstLayerBounds | None = None
    estimated_print_time: timedelta | None = None
    estimated_filament_mm: float | None = None
    slicer_backend: str | None = None
    slicer_version: str | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class GeometryAnalysis:
    """Geometry analysis of STL model"""

    has_overhangs: bool
    has_bridges: bool
    has_thin_walls: bool
    max_overhang_angle: float | None = None  # degrees
    wall_thickness_min_mm: float | None = None
    bounding_box: FirstLayerBounds | None = None
    warnings: list[str] = field(default_factory=list)


# --- Telegram Bot Models ---


@dataclass
class TelegramUser:
    """Telegram user information"""

    user_id: int
    username: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    is_authorized: bool = False


@dataclass
class BotMessage:
    """Telegram bot message"""

    chat_id: int
    text: str
    message_id: int | None = None
    reply_to_message_id: int | None = None
    parse_mode: str = "HTML"  # "HTML" or "Markdown"
    disable_web_page_preview: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionTimeout:
    """Decision timeout configuration"""

    decision_type: DecisionAction
    timeout_seconds: int
    job_id: str
    scheduled_at: datetime = field(default_factory=datetime.now)


# --- Filament Monitoring Models ---


@dataclass
class FilamentSpoolMeasurement:
    """Filament spool measurement from vision analysis"""

    estimated_diameter_mm: float
    estimated_weight_g: float
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    frame_id: int = 0


@dataclass
class TangleDetectionResult:
    """Tangle detection result"""

    detected: bool
    confidence: float
    severity: TangleDecision
    bounding_boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    frame_id: int = 0


@dataclass
class FilamentCheckResult:
    """Result of pre-print filament check"""

    status: FilamentSpoolStatus
    estimated_remaining_g: float
    estimated_required_g: float
    safety_margin_percent: float = 10.0
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


# --- Helper Functions ---


def generate_job_id() -> str:
    """Generate unique job ID"""
    return f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def generate_decision_token() -> str:
    """Generate secure decision token"""
    return uuid.uuid4().hex
