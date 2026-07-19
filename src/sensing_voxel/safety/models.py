"""Immutable models and canonical hashing for print authorization."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any


class SafetyError(Exception):
    """Base class for safety gate failures."""


class ValidationError(SafetyError):
    """Raised when G-code cannot be proven safe."""


class ApprovalError(SafetyError):
    """Raised when an approval is missing, stale, consumed, or mismatched."""


class PreflightError(SafetyError):
    """Raised when execution preflight cannot prove the target is safe."""


class RequestState(StrEnum):
    """Lifecycle state of a print authorization request."""

    PENDING = "pending"
    APPROVED = "approved"
    EXECUTED = "executed"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class Artifact:
    """Immutable, content-addressed bytes captured for validation and preview."""

    content: bytes
    sha256: str
    size_bytes: int
    source_path: Path | None = None

    @classmethod
    def from_bytes(cls, content: bytes, *, source_path: Path | None = None) -> Artifact:
        immutable_content = bytes(content)
        return cls(
            content=immutable_content,
            sha256=hashlib.sha256(immutable_content).hexdigest(),
            size_bytes=len(immutable_content),
            source_path=source_path,
        )

    @classmethod
    def from_path(cls, path: str | Path) -> Artifact:
        source_path = Path(path).resolve()
        return cls.from_bytes(source_path.read_bytes(), source_path=source_path)

    def verify(self) -> bool:
        """Verify that the stored bytes still match the content address."""
        return (
            len(self.content) == self.size_bytes
            and hashlib.sha256(self.content).hexdigest() == self.sha256
        )


@dataclass(frozen=True, slots=True)
class SafetyLimits:
    """Machine/profile limits used by the deterministic validator."""

    min_x: float = 0.0
    max_x: float = 350.0
    min_y: float = 0.0
    max_y: float = 350.0
    min_z: float = 0.0
    max_z: float = 400.0
    max_hotend_c: float = 300.0
    max_bed_c: float = 120.0
    min_extrude_c: float = 150.0
    max_feedrate_mm_min: float = 36_000.0
    max_commands: int = 50_000_000
    firmware: str = "klipper"
    allowed_macros: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not (self.min_x < self.max_x and self.min_y < self.max_y):
            raise ValueError("X/Y safety bounds must have a positive range")
        if not self.min_z < self.max_z:
            raise ValueError("Z safety bounds must have a positive range")
        if self.max_hotend_c <= 0 or self.max_bed_c <= 0:
            raise ValueError("temperature limits must be positive")
        if self.max_feedrate_mm_min <= 0 or self.max_commands <= 0:
            raise ValueError("feedrate and command limits must be positive")
        if self.firmware.lower() not in {"klipper", "marlin"}:
            raise ValueError("firmware must be 'klipper' or 'marlin'")


@dataclass(frozen=True, slots=True)
class Envelope:
    """Cartesian toolpath bounds."""

    min_x: float | None = None
    max_x: float | None = None
    min_y: float | None = None
    max_y: float | None = None
    min_z: float | None = None
    max_z: float | None = None


@dataclass(frozen=True, slots=True)
class TimelineEntry:
    """Normalized command entry used by validation and preview."""

    line: int
    command: str
    parameters: tuple[tuple[str, float | str], ...] = ()


@dataclass(frozen=True, slots=True)
class MotionRecord:
    """A validated toolpath motion and its conservative envelope."""

    line: int
    command: str
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    envelope: Envelope
    extrusion_delta: float


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """A deterministic validation failure."""

    line: int | None
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Whole-file validator output."""

    artifact_sha256: str
    valid: bool
    command_count: int
    line_count: int
    toolpath_envelope: Envelope
    first_layer_envelope: Envelope
    first_layer_z: float | None
    timeline: tuple[TimelineEntry, ...]
    motions: tuple[MotionRecord, ...]
    issues: tuple[ValidationIssue, ...]
    result_sha256: str


@dataclass(frozen=True, slots=True)
class PreviewArtifact:
    """Deterministic summary and rendered preview of exact G-code bytes."""

    content: bytes
    sha256: str
    rendered_content: bytes
    rendered_sha256: str
    artifact_sha256: str
    validation_sha256: str
    media_type: str = "application/json"
    rendered_media_type: str = "image/svg+xml"

    def verify(self) -> bool:
        """Verify both deterministic preview representations."""
        return (
            hashlib.sha256(self.content).hexdigest() == self.sha256
            and hashlib.sha256(self.rendered_content).hexdigest()
            == self.rendered_sha256
        )


@dataclass(frozen=True, slots=True)
class PhysicalConfirmationSnapshot:
    """Trusted physical printer values required before confirmation."""

    build_plate_surface: str
    hotend_nozzle_size_mm: float
    z_offset_mm: float

    def __post_init__(self) -> None:
        if not isinstance(self.build_plate_surface, str) or not self.build_plate_surface.strip():
            raise ValueError("build_plate_surface must be a non-empty string")
        for name in ("hotend_nozzle_size_mm", "z_offset_mm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be finite")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.hotend_nozzle_size_mm <= 0:
            raise ValueError("hotend_nozzle_size_mm must be positive")


@dataclass(frozen=True, slots=True)
class PrintConfirmation:
    """Complete immutable physical and slicer metadata shown for approval."""

    material: str
    build_plate_surface: str
    hotend_nozzle_size_mm: float
    estimated_print_time_seconds: int
    filament_usage_mm: float
    z_offset_mm: float
    artifact_sha256: str
    preview_sha256: str
    rendered_preview_sha256: str
    confirmation_sha256: str = ""

    def __post_init__(self) -> None:
        for name in ("material", "build_plate_surface"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("artifact_sha256", "preview_sha256", "rendered_preview_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 hex digest")
            try:
                int(value, 16)
            except ValueError as exc:
                raise ValueError(f"{name} must be a SHA-256 hex digest") from exc
        for name in ("hotend_nozzle_size_mm", "filament_usage_mm", "z_offset_mm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be finite")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.hotend_nozzle_size_mm <= 0:
            raise ValueError("hotend_nozzle_size_mm must be positive")
        if self.filament_usage_mm < 0:
            raise ValueError("filament_usage_mm must be non-negative")
        if (
            isinstance(self.estimated_print_time_seconds, bool)
            or not isinstance(self.estimated_print_time_seconds, int)
            or self.estimated_print_time_seconds <= 0
        ):
            raise ValueError("estimated_print_time_seconds must be a positive integer")
        if self.confirmation_sha256:
            if self.confirmation_sha256 != canonical_sha256(self):
                raise ValueError("confirmation_sha256 does not match confirmation content")
        else:
            object.__setattr__(self, "confirmation_sha256", canonical_sha256(self))


@dataclass(frozen=True, slots=True)
class SafetyBinding:
    """All identities and mutable environment values bound by approval."""

    user_id: str
    source_trust: str
    target_printer: str
    hardware_digest: str
    profile_digest: str
    config_digest: str
    macro_digest: str
    probe_z_offset: float | None = None
    homing_origin_z: float | None = None

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if field.name in {"probe_z_offset", "homing_origin_z"}:
                continue
            if not value or not isinstance(value, str):
                raise ValueError(f"{field.name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class PrintRequest:
    """Validated print request awaiting explicit approval."""

    request_id: str
    operation_id: str
    artifact: Artifact
    preview: PreviewArtifact
    validation: ValidationResult
    confirmation: PrintConfirmation
    binding: SafetyBinding
    printer_file: str
    requested_at: datetime
    state: RequestState = RequestState.PENDING


@dataclass(frozen=True, slots=True)
class Approval:
    """Opaque, expiring, single-use authorization capability."""

    token: str
    nonce: str
    request_id: str
    operation_id: str
    artifact_sha256: str
    preview_sha256: str
    rendered_preview_sha256: str
    validation_sha256: str
    confirmation_sha256: str
    binding: SafetyBinding
    issued_at: datetime
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class MonitorStartContract:
    """Validated context required to arm monitoring before a print command."""

    filename: str
    source_trust: str
    first_layer_z: float | None
    required: bool = True


@dataclass(frozen=True, slots=True)
class PreflightSnapshot:
    """Observed printer state immediately before print start."""

    connected: bool
    printer_state: str
    binding: SafetyBinding
    probe_z_offset: float
    homing_origin_z: float


@dataclass(frozen=True, slots=True)
class ExecutionReceipt:
    """Idempotent result returned by the final safety gate."""

    operation_id: str
    request_id: str
    started: bool
    adapter_status: str
    message: str | None
    artifact_sha256: str
    executed_at: datetime


def _canonical_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        excluded_fields: set[str] = set()
        if isinstance(value, Artifact):
            excluded_fields.add("content")
        elif isinstance(value, PreviewArtifact):
            excluded_fields.update({"content", "rendered_content"})
        elif isinstance(value, ValidationResult):
            excluded_fields.add("result_sha256")
        elif isinstance(value, PrintConfirmation):
            excluded_fields.add("confirmation_sha256")
        return {
            field.name: _canonical_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if field.name not in excluded_fields
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat(timespec="microseconds")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, dict):
        return {str(key): _canonical_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize supported values to stable UTF-8 JSON bytes."""
    return json.dumps(
        _canonical_value(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Hash the canonical representation of a value."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
