"""Unified final authorization gate for starting validated print jobs."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import secrets
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sensing_voxel.adapters.base import LiveSafetySnapshot, PrinterAdapter
from sensing_voxel.core.models import CommandResultStatus, NormalizedPrinterState
from sensing_voxel.safety.models import (
    Approval,
    ApprovalError,
    Artifact,
    ExecutionReceipt,
    MonitorStartContract,
    PhysicalConfirmationSnapshot,
    PreflightError,
    PreflightSnapshot,
    PreviewArtifact,
    PrintConfirmation,
    PrintRequest,
    RequestState,
    SafetyBinding,
    SafetyLimits,
    ValidationError,
    canonical_sha256,
)
from sensing_voxel.safety.preview import generate_preview
from sensing_voxel.safety.validator import GcodeValidator
from sensing_voxel.slicer.metadata import GcodeMetadataExtractor

MANDATORY_MONITOR_SOURCES = frozenset({"user_gcode", "sva_generated"})

logger = logging.getLogger(__name__)


class ArtifactStore:
    """In-memory content-addressed storage retaining immutable source bytes."""

    def __init__(self) -> None:
        self._content: dict[str, bytes] = {}

    def capture_bytes(
        self, content: bytes, *, source_path: Path | None = None
    ) -> Artifact:
        artifact = Artifact.from_bytes(content, source_path=source_path)
        stored = self._content.setdefault(artifact.sha256, artifact.content)
        if stored != artifact.content:
            raise ValidationError("content-address collision")
        return artifact

    def capture_path(self, path: str | Path) -> Artifact:
        source_path = Path(path).resolve()
        return self.capture_bytes(source_path.read_bytes(), source_path=source_path)

    def get(self, sha256: str) -> Artifact:
        try:
            content = self._content[sha256]
        except KeyError as exc:
            raise ValidationError("artifact is not present in immutable storage") from exc
        return Artifact.from_bytes(content)


class PrintSafetyService:
    """Sole intended implementation of print validation and authorization."""

    def __init__(
        self,
        adapter: PrinterAdapter,
        *,
        limits: SafetyLimits | None = None,
        approval_ttl: timedelta = timedelta(minutes=10),
        clock: Callable[[], datetime] | None = None,
        artifact_store: ArtifactStore | None = None,
        binding_provider: Callable[[], Awaitable[SafetyBinding]] | None = None,
        confirmation_provider: Callable[
            [Artifact, SafetyBinding], Awaitable[PhysicalConfirmationSnapshot]
        ]
        | None = None,
        monitor_start: Callable[[MonitorStartContract], Awaitable[None]] | None = None,
        monitor_stop: Callable[[], Awaitable[None]] | None = None,
        monitor_required_sources: frozenset[str] = MANDATORY_MONITOR_SOURCES,
    ) -> None:
        if approval_ttl <= timedelta(0):
            raise ValueError("approval_ttl must be positive")
        if not MANDATORY_MONITOR_SOURCES.issubset(monitor_required_sources):
            missing = ", ".join(sorted(MANDATORY_MONITOR_SOURCES - monitor_required_sources))
            raise ValueError(f"mandatory monitored source types are missing: {missing}")
        self._adapter = adapter
        self._validator = GcodeValidator(limits)
        self._approval_ttl = approval_ttl
        self._clock = clock or (lambda: datetime.now(UTC))
        self._artifacts = artifact_store or ArtifactStore()
        self._binding_provider = binding_provider
        self._confirmation_provider = confirmation_provider
        self._monitor_start = monitor_start
        self._monitor_stop = monitor_stop
        self._monitor_required_sources = monitor_required_sources
        self._requests: dict[str, PrintRequest] = {}
        self._approvals: dict[str, Approval] = {}
        self._consumed_tokens: set[str] = set()
        self._receipts: dict[str, ExecutionReceipt] = {}
        self._lock = asyncio.Lock()

    async def create_binding(self, *, user_id: str, source_trust: str) -> SafetyBinding:
        """Build caller identity around fresh trusted environment and live safety data."""
        if not user_id or not source_trust:
            raise ValueError("user_id and source_trust must be non-empty")
        if self._binding_provider is None:
            raise PreflightError("trusted printer binding provider is unavailable")
        try:
            trusted = await self._binding_provider()
            snapshot = await self._adapter.get_live_safety_snapshot(trusted.target_printer)
        except Exception as exc:
            raise PreflightError("trusted printer binding query failed") from exc
        probe_z_offset, homing_origin_z = self._validate_live_snapshot(
            snapshot, trusted.target_printer
        )
        return replace(
            trusted,
            user_id=user_id,
            source_trust=source_trust,
            probe_z_offset=probe_z_offset,
            homing_origin_z=homing_origin_z,
        )

    async def request_from_bytes(
        self,
        content: bytes,
        *,
        operation_id: str,
        binding: SafetyBinding,
        printer_file: str,
    ) -> PrintRequest:
        """Create a request from immutable caller-supplied bytes."""
        return await self.request_print(
            content,
            operation_id=operation_id,
            binding=binding,
            printer_file=printer_file,
        )

    async def request_from_path(
        self,
        path: str | Path,
        *,
        operation_id: str,
        binding: SafetyBinding,
        printer_file: str,
    ) -> PrintRequest:
        """Create a request while retaining the source path for mutation checks."""
        return await self.request_print(
            path,
            operation_id=operation_id,
            binding=binding,
            printer_file=printer_file,
        )

    async def request_print(
        self,
        source: bytes | str | Path | Artifact,
        *,
        operation_id: str,
        binding: SafetyBinding,
        printer_file: str,
    ) -> PrintRequest:
        """Capture exact bytes, validate the whole file, and create a pending request."""
        if not operation_id or not printer_file:
            raise ValueError("operation_id and printer_file must be non-empty")
        if any(request.operation_id == operation_id for request in self._requests.values()):
            raise ApprovalError("operation_id already belongs to a print request")

        artifact = self._capture(source)
        validation = self._validator.validate(artifact)
        if not validation.valid:
            details = "; ".join(
                f"{issue.code}: {issue.message}" for issue in validation.issues
            )
            raise ValidationError(details or "G-code validation failed")
        preview = generate_preview(artifact, validation)
        confirmation = await self._build_confirmation(artifact, preview, binding)
        request = PrintRequest(
            request_id=uuid.uuid4().hex,
            operation_id=operation_id,
            artifact=artifact,
            preview=preview,
            validation=validation,
            confirmation=confirmation,
            binding=binding,
            printer_file=printer_file,
            requested_at=self._now(),
        )
        self._requests[request.request_id] = request
        return request

    def get_request(self, request_id: str) -> PrintRequest:
        """Return an immutable request for transport delivery and confirmation."""
        return self._request(request_id)

    def reject(self, request_id: str, *, binding: SafetyBinding) -> PrintRequest:
        """Permanently reject a pending request after explicit user refusal."""
        request = self._request(request_id)
        if request.state is not RequestState.PENDING:
            raise ApprovalError("request is not awaiting approval")
        if binding != request.binding:
            raise ApprovalError("rejection binding does not match the request")
        rejected = replace(request, state=RequestState.REJECTED)
        self._requests[request_id] = rejected
        return rejected

    def approve(self, request_id: str, *, binding: SafetyBinding) -> Approval:
        """Issue an opaque, expiring capability after explicit binding confirmation."""
        request = self._request(request_id)
        if request.state is not RequestState.PENDING:
            raise ApprovalError("request is not awaiting approval")
        if binding != request.binding:
            raise ApprovalError("approval binding does not match the request")
        self._verify_request_artifacts(request)

        now = self._now()
        nonce = secrets.token_urlsafe(24)
        token = secrets.token_urlsafe(32)
        approval = Approval(
            token=token,
            nonce=nonce,
            request_id=request.request_id,
            operation_id=request.operation_id,
            artifact_sha256=request.artifact.sha256,
            preview_sha256=request.preview.sha256,
            rendered_preview_sha256=request.preview.rendered_sha256,
            validation_sha256=request.validation.result_sha256,
            confirmation_sha256=request.confirmation.confirmation_sha256,
            binding=request.binding,
            issued_at=now,
            expires_at=now + self._approval_ttl,
        )
        self._approvals[token] = approval
        self._requests[request_id] = replace(request, state=RequestState.APPROVED)
        return approval

    async def execute(
        self,
        token: str,
        *,
        operation_id: str,
        binding: SafetyBinding,
    ) -> ExecutionReceipt:
        """Consume approval and start at most one print for an operation ID."""
        async with self._lock:
            approval = self._approval(token)
            request = self._request(approval.request_id)
            self._verify_execution_binding(approval, request, operation_id, binding)

            existing = self._receipts.get(operation_id)
            if existing is not None:
                return existing
            if token in self._consumed_tokens:
                raise ApprovalError("approval token has already been consumed")
            if self._now() >= approval.expires_at:
                raise ApprovalError("approval token has expired")

            self._consumed_tokens.add(token)
            self._verify_request_artifacts(request)
            trusted_binding = await self._resolve_trusted_binding(binding)
            await self._preflight(trusted_binding)
            monitor_started = await self._start_required_monitor(request)
            await self._preflight(trusted_binding)

            try:
                result = await self._adapter.start_print(request.printer_file)
            except Exception as exc:
                self._requests[request.request_id] = replace(
                    request, state=RequestState.REJECTED
                )
                if monitor_started:
                    await self._rollback_required_monitor()
                raise PreflightError("printer start command failed") from exc

            started = result.status is CommandResultStatus.SUCCESS
            self._requests[request.request_id] = replace(
                request,
                state=RequestState.EXECUTED if started else RequestState.REJECTED,
            )
            if monitor_started and not started:
                await self._rollback_required_monitor()
            receipt = ExecutionReceipt(
                operation_id=operation_id,
                request_id=request.request_id,
                started=started,
                adapter_status=result.status.name,
                message=result.message,
                artifact_sha256=request.artifact.sha256,
                executed_at=self._now(),
            )
            self._receipts[operation_id] = receipt
            return receipt

    async def _build_confirmation(
        self,
        artifact: Artifact,
        preview: PreviewArtifact,
        binding: SafetyBinding,
    ) -> PrintConfirmation:
        if self._confirmation_provider is None:
            raise ValidationError("trusted print confirmation provider is unavailable")
        metadata = GcodeMetadataExtractor.extract_metadata_bytes(artifact.content)
        material = metadata.get("filament_type")
        estimated_time = metadata.get("estimated_time")
        filament_usage = metadata.get("filament_used_mm")
        if not isinstance(material, str) or not material.strip():
            raise ValidationError("G-code material metadata is unavailable")
        if (
            isinstance(estimated_time, bool)
            or not isinstance(estimated_time, int)
            or estimated_time <= 0
        ):
            raise ValidationError("G-code estimated print time metadata is unavailable")
        if (
            isinstance(filament_usage, bool)
            or not isinstance(filament_usage, (int, float))
            or float(filament_usage) < 0
        ):
            raise ValidationError("G-code filament usage metadata is unavailable")
        try:
            physical = await self._confirmation_provider(artifact, binding)
        except Exception as exc:
            raise ValidationError("trusted physical confirmation data is unavailable") from exc
        if not isinstance(physical, PhysicalConfirmationSnapshot):
            raise ValidationError("trusted physical confirmation data is invalid")
        return PrintConfirmation(
            material=material.strip(),
            build_plate_surface=physical.build_plate_surface,
            hotend_nozzle_size_mm=physical.hotend_nozzle_size_mm,
            estimated_print_time_seconds=estimated_time,
            filament_usage_mm=float(filament_usage),
            z_offset_mm=physical.z_offset_mm,
            artifact_sha256=artifact.sha256,
            preview_sha256=preview.sha256,
            rendered_preview_sha256=preview.rendered_sha256,
        )

    def _capture(self, source: bytes | str | Path | Artifact) -> Artifact:
        if isinstance(source, Artifact):
            if not source.verify():
                raise ValidationError("provided artifact failed integrity verification")
            return self._artifacts.capture_bytes(
                source.content, source_path=source.source_path
            )
        if isinstance(source, bytes):
            return self._artifacts.capture_bytes(source)
        return self._artifacts.capture_path(source)

    def _request(self, request_id: str) -> PrintRequest:
        try:
            return self._requests[request_id]
        except KeyError as exc:
            raise ApprovalError("unknown print request") from exc

    def _approval(self, token: str) -> Approval:
        try:
            return self._approvals[token]
        except KeyError as exc:
            raise ApprovalError("unknown approval token") from exc

    def _verify_execution_binding(
        self,
        approval: Approval,
        request: PrintRequest,
        operation_id: str,
        binding: SafetyBinding,
    ) -> None:
        if operation_id != approval.operation_id or operation_id != request.operation_id:
            raise ApprovalError("operation ID does not match approval")
        if binding != approval.binding or binding != request.binding:
            raise ApprovalError("current printer/config binding does not match approval")
        if approval.request_id != request.request_id:
            raise ApprovalError("approval request binding mismatch")
        expected = (
            request.artifact.sha256,
            request.preview.sha256,
            request.preview.rendered_sha256,
            request.validation.result_sha256,
            request.confirmation.confirmation_sha256,
        )
        actual = (
            approval.artifact_sha256,
            approval.preview_sha256,
            approval.rendered_preview_sha256,
            approval.validation_sha256,
            approval.confirmation_sha256,
        )
        if actual != expected:
            raise ApprovalError("approval content binding mismatch")

    def _verify_request_artifacts(self, request: PrintRequest) -> None:
        artifact = request.artifact
        if not artifact.verify():
            raise ApprovalError("artifact integrity verification failed")
        stored = self._artifacts.get(artifact.sha256)
        if stored.content != artifact.content:
            raise ApprovalError("stored artifact bytes differ from approved bytes")
        if artifact.source_path is not None:
            try:
                current = artifact.source_path.read_bytes()
            except OSError as exc:
                raise ApprovalError("approved source file is unavailable") from exc
            if hashlib.sha256(current).hexdigest() != artifact.sha256:
                raise ApprovalError("approved source file changed after validation")
        if request.validation.artifact_sha256 != artifact.sha256:
            raise ApprovalError("validation artifact binding mismatch")
        if request.validation.result_sha256 != canonical_sha256(request.validation):
            raise ApprovalError("validation result integrity verification failed")
        if not request.validation.valid:
            raise ApprovalError("approved validation result is not valid")
        if request.preview.artifact_sha256 != artifact.sha256:
            raise ApprovalError("preview artifact binding mismatch")
        if request.preview.validation_sha256 != request.validation.result_sha256:
            raise ApprovalError("preview validation binding mismatch")
        if not request.preview.verify():
            raise ApprovalError("preview integrity verification failed")
        confirmation = request.confirmation
        expected_confirmation_bindings = (
            artifact.sha256,
            request.preview.sha256,
            request.preview.rendered_sha256,
        )
        actual_confirmation_bindings = (
            confirmation.artifact_sha256,
            confirmation.preview_sha256,
            confirmation.rendered_preview_sha256,
        )
        if actual_confirmation_bindings != expected_confirmation_bindings:
            raise ApprovalError("confirmation content binding mismatch")
        if confirmation.confirmation_sha256 != canonical_sha256(confirmation):
            raise ApprovalError("confirmation integrity verification failed")

    async def _resolve_trusted_binding(
        self, approved_binding: SafetyBinding
    ) -> SafetyBinding:
        if self._binding_provider is None:
            return approved_binding
        try:
            current_binding = await self._binding_provider()
        except Exception as exc:
            raise PreflightError("trusted printer binding query failed") from exc
        current_binding = replace(
            current_binding,
            user_id=approved_binding.user_id,
            source_trust=approved_binding.source_trust,
            probe_z_offset=approved_binding.probe_z_offset,
            homing_origin_z=approved_binding.homing_origin_z,
        )
        if current_binding != approved_binding:
            raise ApprovalError("trusted printer/config binding changed after approval")
        return current_binding

    async def _start_required_monitor(self, request: PrintRequest) -> bool:
        if request.binding.source_trust not in self._monitor_required_sources:
            return False
        if request.validation.first_layer_z is None:
            raise PreflightError("validated first-layer monitor data is unavailable")
        if self._monitor_start is None or self._monitor_stop is None:
            raise PreflightError("required print monitor is unavailable")
        contract = MonitorStartContract(
            filename=request.printer_file,
            source_trust=request.binding.source_trust,
            first_layer_z=request.validation.first_layer_z,
        )
        try:
            await self._monitor_start(contract)
        except Exception as exc:
            raise PreflightError("required print monitor failed to start") from exc
        return True

    async def _rollback_required_monitor(self) -> None:
        if self._monitor_stop is None:
            return
        try:
            await self._monitor_stop()
        except Exception:
            logger.exception("required print monitor failed to stop after printer start failure")

    async def _preflight(self, binding: SafetyBinding) -> PreflightSnapshot:
        try:
            connected = await self._adapter.is_connected()
            state = await self._adapter.get_printer_state()
            snapshot = await self._adapter.get_live_safety_snapshot(binding.target_printer)
        except Exception as exc:
            raise PreflightError("printer preflight query failed") from exc
        if not connected:
            raise PreflightError("printer is not connected")
        if state is not NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT:
            raise PreflightError(f"printer is not idle: {state.name}")
        probe_z_offset, homing_origin_z = self._validate_live_snapshot(
            snapshot, binding.target_printer
        )
        if (
            probe_z_offset != binding.probe_z_offset
            or homing_origin_z != binding.homing_origin_z
        ):
            raise ApprovalError("live Z/probe safety snapshot changed after approval")
        return PreflightSnapshot(
            True,
            state.name,
            binding,
            probe_z_offset,
            homing_origin_z,
        )

    @staticmethod
    def _validate_live_snapshot(
        snapshot: LiveSafetySnapshot, target_printer: str
    ) -> tuple[float, float]:
        snapshot_target = snapshot.target_printer
        probe_z_offset = snapshot.probe_z_offset
        homing_origin_z = snapshot.homing_origin_z
        if snapshot_target != target_printer:
            raise PreflightError("live safety snapshot target does not match selected printer")
        if (
            isinstance(probe_z_offset, bool)
            or isinstance(homing_origin_z, bool)
            or not isinstance(probe_z_offset, (int, float))
            or not isinstance(homing_origin_z, (int, float))
            or not math.isfinite(probe_z_offset)
            or not math.isfinite(homing_origin_z)
        ):
            raise PreflightError("live Z/probe safety snapshot is missing required values")
        if probe_z_offset == 0.0:
            raise PreflightError("probe z_offset is zero or uninitialized")
        if homing_origin_z > 0.0:
            raise PreflightError("homing_origin Z is positive")
        return float(probe_z_offset), float(homing_origin_z)

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return now
