"""
Core data models for Sensing Voxel

Enums, data classes, and event types used across components.
"""

from __future__ import annotations

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime, timedelta
from pathlib import Path
import uuid


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


class CommandResultStatus(Enum):
    """Command execution status"""
    SUCCESS = auto()
    FAILURE = auto()
    NOT_SUPPORTED = auto()


@dataclass
class JobContext:
    """Context for a print job"""
    job_id: str
    job_file: str  # Path to STL or G-code
    uploaded_at: datetime
    expires_at: Optional[datetime] = None
    decision_token: Optional[str] = None
    decision_consumed: bool = False
    decision_action: Optional[DecisionAction] = None
    decision_consumed_at: Optional[datetime] = None
    gcode_path: Optional[str] = None
    gcode_hash: Optional[str] = None
    first_layer_params: Optional["FirstLayerParams"] = None
    first_layer_bounds: Optional["FirstLayerBounds"] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    slicing_status: SlicingStatus = SlicingStatus.PENDING
    file_validation_status: FileValidationStatus = FileValidationStatus.VALID
    is_external: bool = False  # True for externally-started prints
    external_monitoring: bool = False  # Policy flag (True for external prints)


@dataclass
class PrintProgress:
    """Printer progress information"""
    state: NormalizedPrinterState
    filename: Optional[str] = None
    progress: Optional[float] = None  # 0.0 to 1.0
    print_time: Optional[float] = None  # seconds
    current_layer: Optional[int] = None
    total_layers: Optional[int] = None
    nozzle_temp: Optional[float] = None
    nozzle_target: Optional[float] = None
    bed_temp: Optional[float] = None
    bed_target: Optional[float] = None


@dataclass
class TelemetryData:
    """Real-time telemetry from printer"""
    timestamp: datetime
    temperatures: Dict[str, float] = field(default_factory=dict)
    position: Optional[Dict[str, float]] = None
    velocity: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FirstLayerParams:
    """First layer parameters from G-code"""
    layer_height: float
    nozzle_temp: float
    bed_temp: float
    speed: float
    fan_speed: Optional[float] = None


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
    message: Optional[str] = None
    data: Optional[Any] = None


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
    matrix: List[List[float]]  # 3x3 matrix
    mm_per_pixel: float
    camera_id: Optional[str] = None
    calibration_timestamp: datetime = field(default_factory=datetime.now)
    valid_until: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self, now: Optional[datetime] = None) -> bool:
        """Check if calibration is still valid"""
        now = now or datetime.now()
        if self.valid_until is None:
            return True
        return now < self.valid_until


@dataclass
class CameraCalibration:
    """Complete camera calibration artifact"""
    homography: HomographyMatrix
    roi_points: Optional[List[Tuple[float, float]]] = None  # Checkerboard corners
    error_mm: Optional[float] = None  # Reprojection error
    notes: Optional[str] = None


@dataclass
class NOTHING:
    """Result of NOTHING detection inference"""
    confidence: float  # 0.0 to 1.0
    bounding_box: Optional[Tuple[float, float, float, float]] = None  # x, y, w, h
    timestamp: datetime = field(default_factory=datetime.now)
    frame_id: Optional[int] = None

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
    coverage_ratio: Optional[float] = None  # ROI coverage (0.0 to 1.0)
    defects: List[str] = field(default_factory=list)  # e.g., ["under_extrusion", "gaps"]
    timestamp: datetime = field(default_factory=datetime.now)
    roi: Optional[Tuple[float, float, float, float]] = None  # x, y, w, h


@dataclass
class VisionFrame:
    """Single frame from camera stream"""
    frame_data: Any  # OpenCV Mat or bytes
    timestamp: datetime = field(default_factory=datetime.now)
    frame_id: int = 0
    width: int = 0
    height: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# --- Monitoring & Z-Offset Models ---


@dataclass
class ZOffsetState:
    """Z-offset state tracking"""
    baseline_offset: Optional[float] = None  # Original baseline at print start
    current_offset: float = 0.0  # Cumulative applied offset
    last_known_baseline: Optional[float] = None  # Persistent across restarts
    status: ZOffsetStatus = ZOffsetStatus.CLEAN
    dirty_since: Optional[datetime] = None

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
    NOTHING: Optional[datetime] = None
    last_first_layer_timestamp: Optional[datetime] = None
    first_layer_active_until_layer: Optional[int] = None  # Stop after this layer


# --- Slicer & Geometry Models ---


@dataclass
class SlicingResult:
    """Result of slicing operation"""
    gcode_path: Path
    gcode_hash: str
    first_layer_params: Optional[FirstLayerParams] = None
    first_layer_bounds: Optional[FirstLayerBounds] = None
    estimated_print_time: Optional[timedelta] = None
    estimated_filament_mm: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class GeometryAnalysis:
    """Geometry analysis of STL model"""
    has_overhangs: bool
    has_bridges: bool
    has_thin_walls: bool
    max_overhang_angle: Optional[float] = None  # degrees
    wall_thickness_min_mm: Optional[float] = None
    bounding_box: Optional[FirstLayerBounds] = None
    warnings: List[str] = field(default_factory=list)


# --- Telegram Bot Models ---


@dataclass
class TelegramUser:
    """Telegram user information"""
    user_id: int
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_authorized: bool = False


@dataclass
class BotMessage:
    """Telegram bot message"""
    chat_id: int
    text: str
    message_id: Optional[int] = None
    reply_to_message_id: Optional[int] = None
    parse_mode: str = "HTML"  # "HTML" or "Markdown"
    disable_web_page_preview: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionTimeout:
    """Decision timeout configuration"""
    decision_type: DecisionAction
    timeout_seconds: int
    job_id: str
    scheduled_at: datetime = field(default_factory=datetime.now)


# --- Helper Functions ---


def generate_job_id() -> str:
    """Generate unique job ID"""
    return f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def generate_decision_token() -> str:
    """Generate secure decision token"""
    return uuid.uuid4().hex
