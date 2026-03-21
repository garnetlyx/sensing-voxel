"""
Queue data models for multi-printer scheduling

Defines job queue, printer capabilities, and scheduling-related data structures.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any

from sensing_voxel.core.models import (
    FirstLayerBounds,
    FirstLayerParams,
    JobContext,
    SlicingResult,
)


class QueueJobStatus(Enum):
    """Job status in the queue"""

    PENDING = auto()
    SCHEDULED = auto()
    ASSIGNING = auto()
    WAITING_FOR_PRINTER = auto()
    STARTING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class QueuePriority(Enum):
    """Job priority levels"""

    CRITICAL = 0  # Emergency jobs
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class PrinterStatus(Enum):
    """Printer availability status"""

    AVAILABLE = auto()
    BUSY = auto()
    PRINTING = auto()
    PAUSED = auto()
    OFFLINE = auto()
    ERROR = auto()
    MAINTENANCE = auto()


class AssignmentStrategy(Enum):
    """Job assignment strategy"""

    FIRST_AVAILABLE = auto()
    ROUND_ROBIN = auto()
    LEAST_LOADED = auto()
    FILAMENT_MATCH = auto()
    BED_SIZE_MATCH = auto()
    OPTIMAL = auto()  # Combines multiple factors


@dataclass
class PrinterCapabilities:
    """Printer capabilities for job matching"""

    printer_id: str
    bed_width_mm: float = 220.0
    bed_depth_mm: float = 220.0
    bed_height_mm: float = 250.0
    supported_filaments: set[str] = field(default_factory=lambda: {"PLA", "PETG", "ABS"})
    nozzle_diameter_mm: float = 0.4
    has_heated_bed: bool = True
    has_enclosure: bool = False
    max_nozzle_temp: float = 300.0
    max_bed_temp: float = 110.0
    is_multicolor: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def bed_area_mm2(self) -> float:
        """Calculate bed area in square mm"""
        return self.bed_width_mm * self.bed_depth_mm

    def can_fit_bounds(self, bounds: FirstLayerBounds) -> bool:
        """Check if printer can fit given bounds"""
        return bounds.width <= self.bed_width_mm and bounds.height <= self.bed_depth_mm

    def supports_filament(self, filament_type: str) -> bool:
        """Check if printer supports given filament type"""
        return filament_type.upper() in {f.upper() for f in self.supported_filaments}


@dataclass
class PrinterLoad:
    """Current load state of a printer"""

    printer_id: str
    status: PrinterStatus = PrinterStatus.OFFLINE
    current_job_id: str | None = None
    queue_position: int = 0  # Position in queue (0 = at front)
    estimated_completion: datetime | None = None
    utilization_percent: float = 0.0  # 0-100
    jobs_completed_today: int = 0
    total_print_time_today: timedelta = field(default_factory=lambda: timedelta(0))
    last_activity: datetime | None = None
    error_count: int = 0
    maintenance_due: bool = False


@dataclass
class QueuedJob:
    """Job waiting in queue"""

    queue_job_id: str
    job_context: JobContext
    status: QueueJobStatus = QueueJobStatus.PENDING
    priority: QueuePriority = QueuePriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    assigned_printer_id: str | None = None
    deadline: datetime | None = None  # For deadline-based priority
    filament_type: str | None = None
    estimated_duration: timedelta | None = None
    estimated_filament_m: float | None = None
    required_bed_temp: float | None = None
    required_nozzle_temp: float | None = None
    slicing_result: SlicingResult | None = None
    user_id: str | None = None  # Who submitted the job
    tags: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3

    @property
    def wait_time(self) -> timedelta:
        """Calculate time spent waiting in queue"""
        now = datetime.now()
        start = self.scheduled_at or self.created_at
        return now - start

    @property
    def is_overdue(self) -> bool:
        """Check if job has passed its deadline"""
        if self.deadline is None:
            return False
        return datetime.now() > self.deadline

    @property
    def priority_score(self) -> float:
        """
        Calculate dynamic priority score (lower = higher priority)
        Combines static priority with wait time bonus
        """
        base_score = self.priority.value * 100

        # Wait time bonus: increase priority every 5 minutes of waiting
        wait_minutes = self.wait_time.total_seconds() / 60
        wait_bonus = wait_minutes / 5  # Decrease score (higher priority) over time

        # Deadline penalty: increase score (lower priority) if far from deadline
        deadline_penalty = 0.0
        if self.deadline:
            time_to_deadline = (self.deadline - datetime.now()).total_seconds()
            if time_to_deadline < 0:
                deadline_penalty = -500  # Overdue jobs get priority boost
            elif time_to_deadline < 3600:  # Less than 1 hour
                deadline_penalty = -200
            elif time_to_deadline < 86400:  # Less than 24 hours
                deadline_penalty = -50

        return base_score - wait_bonus + deadline_penalty


@dataclass
class AssignmentResult:
    """Result of job assignment attempt"""

    success: bool
    job_id: str
    printer_id: str | None = None
    reason: str | None = None
    assigned_at: datetime = field(default_factory=datetime.now)
    strategy_used: AssignmentStrategy = AssignmentStrategy.FIRST_AVAILABLE


@dataclass
class QueueStats:
    """Queue statistics"""

    total_jobs: int = 0
    pending_jobs: int = 0
    running_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    avg_wait_time: timedelta = field(default_factory=lambda: timedelta(0))
    avg_processing_time: timedelta = field(default_factory=lambda: timedelta(0))
    printer_utilization: dict[str, float] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ScheduleWindow:
    """Time window for scheduling"""

    start: datetime
    end: datetime
    priority_boost: int = 0  # Additional priority during window

    def contains(self, dt: datetime) -> bool:
        """Check if datetime is within window"""
        return self.start <= dt <= self.end


@dataclass
class QueueConfiguration:
    """Queue configuration parameters"""

    max_queue_size: int = 100
    default_priority: QueuePriority = QueuePriority.NORMAL
    auto_assignment: bool = True
    assignment_strategy: AssignmentStrategy = AssignmentStrategy.OPTIMAL
    enable_load_balancing: bool = True
    enable_priority_boost: bool = True  # Boost priority based on wait time
    wait_time_boost_threshold: timedelta = field(default_factory=lambda: timedelta(minutes=30))
    wait_time_boost_amount: int = 1  # Reduce priority level by this much
    maintenance_window: ScheduleWindow | None = None
    printer_timeout_seconds: float = 300.0  # Mark printer offline after timeout
    job_timeout_seconds: float = 86400.0  # Max time job can wait in queue
    max_retries: int = 3
    cooldown_between_assignments: timedelta = field(default_factory=lambda: timedelta(seconds=5))


# Type aliases for callbacks
JobStatusCallback = Callable[[QueuedJob, QueueJobStatus, QueueJobStatus], Any]
PrinterAssignmentCallback = Callable[[str, str], Any]  # job_id, printer_id


def generate_queue_job_id() -> str:
    """Generate unique queue job ID"""
    return f"qj_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
