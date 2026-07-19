from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from sensing_voxel.adapters.base import LiveSafetySnapshot
from sensing_voxel.core.models import (
    CommandResult,
    CommandResultStatus,
    NormalizedPrinterState,
)
from sensing_voxel.safety import (
    ApprovalError,
    PhysicalConfirmationSnapshot,
    PreflightError,
    PrintSafetyService,
    RequestState,
    SafetyBinding,
    ValidationError,
)

VALID_GCODE = (
    b"; filament_type = PLA\n"
    b"; estimated_time = 3600\n"
    b"; filament used [mm] = 1234.5\n"
    b"G28\nM104 S200\nG90\nM83\nG1 X10 Y10 Z0.2 F1200\nG1 X20 E1\n"
)


async def confirmation_provider(
    _artifact, binding: SafetyBinding
) -> PhysicalConfirmationSnapshot:
    return PhysicalConfirmationSnapshot("textured PEI", 0.6, binding.probe_z_offset)


def safety_service(adapter, **kwargs) -> PrintSafetyService:
    return PrintSafetyService(
        adapter,
        confirmation_provider=confirmation_provider,
        **kwargs,
    )


def binding(**changes: str) -> SafetyBinding:
    values = {
        "user_id": "user-1",
        "source_trust": "user-upload",
        "target_printer": "printer-1",
        "hardware_digest": "hardware-v1",
        "profile_digest": "profile-v1",
        "config_digest": "config-v1",
        "macro_digest": "macros-v1",
        "probe_z_offset": -1.25,
        "homing_origin_z": 0.0,
    }
    values.update(changes)
    return SafetyBinding(**values)


def adapter_mock():
    adapter = AsyncMock()
    adapter.is_connected.return_value = True
    adapter.get_printer_state.return_value = NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT
    adapter.get_live_safety_snapshot.return_value = LiveSafetySnapshot(
        target_printer="printer-1",
        probe_z_offset=-1.25,
        homing_origin_z=0.0,
    )
    adapter.start_print.return_value = CommandResult(
        CommandResultStatus.SUCCESS, "started"
    )
    return adapter


async def approved_service(*, now: list[datetime] | None = None):
    adapter = adapter_mock()
    clock = (lambda: now[0]) if now is not None else None

    async def monitor_start(_contract) -> None:
        return None

    service = safety_service(adapter, clock=clock, monitor_start=monitor_start)
    current_binding = binding()
    request = await service.request_print(VALID_GCODE,
    operation_id="operation-1",
    binding=current_binding,
    printer_file="validated.gcode",)
    approval = service.approve(request.request_id, binding=current_binding)
    return service, adapter, current_binding, request, approval


@pytest.mark.asyncio
async def test_request_rejects_invalid_gcode_before_approval() -> None:
    service = safety_service(adapter_mock())

    with pytest.raises(ValidationError, match="missing_homing"):
        await service.request_print(b"M104 S200\nG1 X10\n",
        operation_id="unsafe-operation",
        binding=binding(),
        printer_file="unsafe.gcode",)


@pytest.mark.asyncio
async def test_preview_is_deterministic_and_bound_to_artifact() -> None:
    service = safety_service(adapter_mock())
    first = await service.request_print(VALID_GCODE,
    operation_id="first",
    binding=binding(),
    printer_file="first.gcode",)
    second = await service.request_print(VALID_GCODE,
    operation_id="second",
    binding=binding(),
    printer_file="second.gcode",)

    assert first.artifact.sha256 == second.artifact.sha256
    assert first.preview == second.preview
    assert first.preview.artifact_sha256 == first.artifact.sha256
    assert first.preview.validation_sha256 == first.validation.result_sha256


@pytest.mark.asyncio
async def test_duplicate_operation_id_is_rejected_at_request_time() -> None:
    service = safety_service(adapter_mock())
    current_binding = binding()
    await service.request_print(VALID_GCODE,
    operation_id="duplicate",
    binding=current_binding,
    printer_file="first.gcode",)

    with pytest.raises(ApprovalError, match="operation_id"):
        await service.request_print(VALID_GCODE,
        operation_id="duplicate",
        binding=current_binding,
        printer_file="second.gcode",)


@pytest.mark.asyncio
async def test_approval_rejects_binding_mismatch() -> None:
    service = safety_service(adapter_mock())
    request = await service.request_print(VALID_GCODE,
    operation_id="operation",
    binding=binding(),
    printer_file="validated.gcode",)

    with pytest.raises(ApprovalError, match="binding"):
        service.approve(
            request.request_id,
            binding=binding(config_digest="config-v2"),
        )


@pytest.mark.asyncio
async def test_reject_transitions_pending_request_and_prevents_approval() -> None:
    service = safety_service(adapter_mock())
    current_binding = binding()
    request = await service.request_print(VALID_GCODE,
    operation_id="operation",
    binding=current_binding,
    printer_file="validated.gcode",)

    rejected = service.reject(request.request_id, binding=current_binding)

    assert rejected.state is RequestState.REJECTED
    assert service.get_request(request.request_id).state is RequestState.REJECTED
    with pytest.raises(ApprovalError, match="not awaiting approval"):
        service.approve(request.request_id, binding=current_binding)


@pytest.mark.asyncio
async def test_reject_requires_exact_binding_and_leaves_request_pending() -> None:
    service = safety_service(adapter_mock())
    current_binding = binding()
    request = await service.request_print(VALID_GCODE,
    operation_id="operation",
    binding=current_binding,
    printer_file="validated.gcode",)

    with pytest.raises(ApprovalError, match="rejection binding"):
        service.reject(
            request.request_id,
            binding=binding(config_digest="config-v2"),
        )

    assert service.get_request(request.request_id).state is RequestState.PENDING


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changed_binding",
    [
        binding(target_printer="printer-2"),
        binding(hardware_digest="hardware-v2"),
        binding(profile_digest="profile-v2"),
        binding(config_digest="config-v2"),
        binding(macro_digest="macros-v2"),
        binding(user_id="user-2"),
        binding(source_trust="untrusted"),
    ],
)
async def test_execute_rejects_every_binding_mismatch(
    changed_binding: SafetyBinding,
) -> None:
    service, adapter, _, _, approval = await approved_service()

    with pytest.raises(ApprovalError, match="binding"):
        await service.execute(
            approval.token,
            operation_id="operation-1",
            binding=changed_binding,
        )

    adapter.start_print.assert_not_awaited()


@pytest.mark.asyncio
async def test_source_mutation_after_approval_fails(tmp_path: Path) -> None:
    source = tmp_path / "validated.gcode"
    source.write_bytes(VALID_GCODE)
    adapter = adapter_mock()
    service = safety_service(adapter)
    current_binding = binding()
    request = await service.request_print(source,
    operation_id="operation-1",
    binding=current_binding,
    printer_file="validated.gcode",)
    approval = service.approve(request.request_id, binding=current_binding)
    source.write_bytes(VALID_GCODE + b"M112\n")

    with pytest.raises(ApprovalError, match="changed"):
        await service.execute(
            approval.token,
            operation_id="operation-1",
            binding=current_binding,
        )

    adapter.start_print.assert_not_awaited()


@pytest.mark.asyncio
async def test_source_deletion_after_approval_fails(tmp_path: Path) -> None:
    source = tmp_path / "validated.gcode"
    source.write_bytes(VALID_GCODE)
    adapter = adapter_mock()
    service = safety_service(adapter)
    current_binding = binding()
    request = await service.request_print(
        source,
        operation_id="operation-1",
        binding=current_binding,
        printer_file="validated.gcode",
    )
    approval = service.approve(request.request_id, binding=current_binding)
    source.unlink()

    with pytest.raises(ApprovalError, match="unavailable"):
        await service.execute(
            approval.token,
            operation_id="operation-1",
            binding=current_binding,
        )

    adapter.start_print.assert_not_awaited()


@pytest.mark.asyncio
async def test_expired_token_is_rejected() -> None:
    now = [datetime(2026, 7, 18, 12, tzinfo=UTC)]
    adapter = adapter_mock()
    service = safety_service(
        adapter,
        approval_ttl=timedelta(seconds=30),
        clock=lambda: now[0],
    )
    current_binding = binding()
    request = await service.request_print(VALID_GCODE,
    operation_id="operation-1",
    binding=current_binding,
    printer_file="validated.gcode",)
    approval = service.approve(request.request_id, binding=current_binding)
    now[0] += timedelta(seconds=30)

    with pytest.raises(ApprovalError, match="expired"):
        await service.execute(
            approval.token,
            operation_id="operation-1",
            binding=current_binding,
        )

    adapter.start_print.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_preflight_consumes_single_use_token() -> None:
    service, adapter, current_binding, _, approval = await approved_service()
    adapter.is_connected.side_effect = RuntimeError("query failed")

    with pytest.raises(PreflightError, match="query failed"):
        await service.execute(
            approval.token,
            operation_id="operation-1",
            binding=current_binding,
        )

    adapter.is_connected.side_effect = None
    adapter.is_connected.return_value = True
    with pytest.raises(ApprovalError, match="consumed"):
        await service.execute(
            approval.token,
            operation_id="operation-1",
            binding=current_binding,
        )
    adapter.start_print.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "connected,state",
    [
        (False, NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT),
        (True, NormalizedPrinterState.PRINTING),
        (True, NormalizedPrinterState.UNKNOWN),
    ],
)
async def test_preflight_fails_closed_when_printer_is_not_proven_idle(
    connected: bool,
    state: NormalizedPrinterState,
) -> None:
    service, adapter, current_binding, _, approval = await approved_service()
    adapter.is_connected.return_value = connected
    adapter.get_printer_state.return_value = state

    with pytest.raises(PreflightError):
        await service.execute(
            approval.token,
            operation_id="operation-1",
            binding=current_binding,
        )

    adapter.start_print.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("snapshot", "message"),
    [
        (LiveSafetySnapshot("printer-1", 0.0, 0.0), "zero or uninitialized"),
        (LiveSafetySnapshot("printer-1", -1.25, 0.01), "positive"),
        (LiveSafetySnapshot("printer-1", None, 0.0), "missing required"),
        (LiveSafetySnapshot("printer-1", -1.25, None), "missing required"),
        (LiveSafetySnapshot("printer-2", -1.25, 0.0), "target does not match"),
    ],
)
async def test_preflight_rejects_unsafe_or_mismatched_live_snapshot(
    snapshot: LiveSafetySnapshot,
    message: str,
) -> None:
    service, adapter, current_binding, _, approval = await approved_service()
    adapter.get_live_safety_snapshot.return_value = snapshot

    with pytest.raises(PreflightError, match=message):
        await service.execute(
            approval.token,
            operation_id="operation-1",
            binding=current_binding,
        )

    adapter.start_print.assert_not_awaited()


@pytest.mark.asyncio
async def test_preflight_fails_closed_when_live_snapshot_query_errors() -> None:
    service, adapter, current_binding, _, approval = await approved_service()
    adapter.get_live_safety_snapshot.side_effect = RuntimeError("query failed")

    with pytest.raises(PreflightError, match="query failed"):
        await service.execute(
            approval.token,
            operation_id="operation-1",
            binding=current_binding,
        )

    adapter.start_print.assert_not_awaited()


@pytest.mark.asyncio
async def test_live_snapshot_change_after_confirmation_invalidates_approval() -> None:
    service, adapter, current_binding, _, approval = await approved_service()
    adapter.get_live_safety_snapshot.return_value = LiveSafetySnapshot(
        target_printer="printer-1",
        probe_z_offset=-1.20,
        homing_origin_z=0.0,
    )

    with pytest.raises(ApprovalError, match="changed after approval"):
        await service.execute(
            approval.token,
            operation_id="operation-1",
            binding=current_binding,
        )

    adapter.start_print.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_binding_captures_live_z_values_for_confirmation() -> None:
    adapter = adapter_mock()
    trusted = binding(user_id="runtime", source_trust="runtime")
    service = safety_service(adapter, binding_provider=AsyncMock(return_value=trusted))

    current = await service.create_binding(user_id="operator", source_trust="user-upload")

    assert current.probe_z_offset == -1.25
    assert current.homing_origin_z == 0.0
    assert current.target_printer == "printer-1"
    adapter.get_live_safety_snapshot.assert_awaited_once_with("printer-1")


@pytest.mark.asyncio
async def test_unsupported_adapter_live_snapshot_fails_closed() -> None:
    class UnsupportedAdapter:
        pass

    service = safety_service(
        UnsupportedAdapter(),
        binding_provider=AsyncMock(
            return_value=binding(user_id="runtime", source_trust="runtime")
        ),
    )

    with pytest.raises(PreflightError, match="binding query failed"):
        await service.create_binding(user_id="operator", source_trust="user-upload")


@pytest.mark.asyncio
async def test_successful_retry_returns_receipt_without_double_start() -> None:
    service, adapter, current_binding, _, approval = await approved_service()

    first, second = await __import__("asyncio").gather(
        service.execute(
            approval.token,
            operation_id="operation-1",
            binding=current_binding,
        ),
        service.execute(
            approval.token,
            operation_id="operation-1",
            binding=current_binding,
        ),
    )

    assert first == second
    assert first.started
    adapter.start_print.assert_awaited_once_with("validated.gcode")


@pytest.mark.asyncio
async def test_tampered_approval_binding_is_rejected() -> None:
    service, adapter, current_binding, request, approval = await approved_service()
    service._approvals[approval.token] = replace(  # intentional internal corruption test
        approval,
        artifact_sha256="0" * 64,
    )

    with pytest.raises(ApprovalError, match="content binding"):
        await service.execute(
            approval.token,
            operation_id=request.operation_id,
            binding=current_binding,
        )

    adapter.start_print.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_rejects_fresh_trusted_binding_drift() -> None:
    adapter = adapter_mock()
    approved_binding = binding()
    binding_provider = AsyncMock(
        return_value=binding(config_digest="config-v2")
    )
    service = safety_service(adapter, binding_provider=binding_provider)
    request = await service.request_print(VALID_GCODE,
    operation_id="operation-1",
    binding=approved_binding,
    printer_file="validated.gcode",)
    approval = service.approve(request.request_id, binding=approved_binding)

    with pytest.raises(ApprovalError, match="changed after approval"):
        await service.execute(
            approval.token,
            operation_id=request.operation_id,
            binding=approved_binding,
        )

    binding_provider.assert_awaited_once_with()
    adapter.start_print.assert_not_awaited()


async def approved_required_service(
    *,
    adapter=None,
    monitor_start=None,
    monitor_stop=None,
):
    adapter = adapter or adapter_mock()
    monitor_start = monitor_start or AsyncMock()
    monitor_stop = monitor_stop or AsyncMock()
    current_binding = binding(source_trust="sva_generated")
    service = safety_service(
        adapter,
        monitor_start=monitor_start,
        monitor_stop=monitor_stop,
    )
    request = await service.request_print(
        VALID_GCODE,
        operation_id="operation-1",
        binding=current_binding,
        printer_file="validated.gcode",
    )
    approval = service.approve(request.request_id, binding=current_binding)
    return service, adapter, monitor_start, monitor_stop, current_binding, request, approval


@pytest.mark.asyncio
async def test_required_monitor_starts_before_printer_and_remains_active() -> None:
    events: list[str] = []
    adapter = adapter_mock()

    async def monitor_start(contract) -> None:
        events.append(f"monitor:{contract.filename}")

    monitor_stop = AsyncMock()

    async def live_snapshot(target_printer: str) -> LiveSafetySnapshot:
        events.append("preflight")
        return LiveSafetySnapshot(target_printer, -1.25, 0.0)

    async def start_print(filename: str) -> CommandResult:
        events.append(f"printer:{filename}")
        return CommandResult(CommandResultStatus.SUCCESS, "started")

    adapter.get_live_safety_snapshot.side_effect = live_snapshot
    adapter.start_print.side_effect = start_print
    (
        service,
        _,
        _,
        _,
        current_binding,
        request,
        approval,
    ) = await approved_required_service(
        adapter=adapter,
        monitor_start=monitor_start,
        monitor_stop=monitor_stop,
    )

    receipt = await service.execute(
        approval.token,
        operation_id=request.operation_id,
        binding=current_binding,
    )

    assert receipt.started
    assert service.get_request(request.request_id).state is RequestState.EXECUTED
    assert events == [
        "preflight",
        "monitor:validated.gcode",
        "preflight",
        "printer:validated.gcode",
    ]
    monitor_stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_printer_start_exception_stops_monitor_and_rejects_request() -> None:
    adapter = adapter_mock()
    start_error = RuntimeError("Moonraker unavailable")
    adapter.start_print.side_effect = start_error
    (
        service,
        _,
        monitor_start,
        monitor_stop,
        current_binding,
        request,
        approval,
    ) = await approved_required_service(adapter=adapter)

    with pytest.raises(PreflightError, match="printer start command failed") as raised:
        await service.execute(
            approval.token,
            operation_id=request.operation_id,
            binding=current_binding,
        )

    assert raised.value.__cause__ is start_error
    monitor_start.assert_awaited_once()
    monitor_stop.assert_awaited_once_with()
    assert service.get_request(request.request_id).state is RequestState.REJECTED
    assert approval.token in service._consumed_tokens


@pytest.mark.asyncio
async def test_printer_start_failure_stops_monitor_and_returns_failure_receipt() -> None:
    adapter = adapter_mock()
    adapter.start_print.return_value = CommandResult(
        CommandResultStatus.FAILURE,
        "printer rejected start",
    )
    (
        service,
        _,
        monitor_start,
        monitor_stop,
        current_binding,
        request,
        approval,
    ) = await approved_required_service(adapter=adapter)

    receipt = await service.execute(
        approval.token,
        operation_id=request.operation_id,
        binding=current_binding,
    )

    assert receipt.started is False
    assert receipt.adapter_status == CommandResultStatus.FAILURE.name
    assert receipt.message == "printer rejected start"
    monitor_start.assert_awaited_once()
    monitor_stop.assert_awaited_once_with()
    assert service.get_request(request.request_id).state is RequestState.REJECTED
    assert approval.token in service._consumed_tokens


@pytest.mark.asyncio
async def test_monitor_rollback_exception_preserves_printer_start_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    adapter = adapter_mock()
    start_error = RuntimeError("printer transport failed")
    adapter.start_print.side_effect = start_error
    monitor_stop = AsyncMock(side_effect=RuntimeError("monitor teardown failed"))
    (
        service,
        _,
        _,
        _,
        current_binding,
        request,
        approval,
    ) = await approved_required_service(
        adapter=adapter,
        monitor_stop=monitor_stop,
    )

    with pytest.raises(PreflightError, match="printer start command failed") as raised:
        await service.execute(
            approval.token,
            operation_id=request.operation_id,
            binding=current_binding,
        )

    assert raised.value.__cause__ is start_error
    monitor_stop.assert_awaited_once_with()
    assert service.get_request(request.request_id).state is RequestState.REJECTED
    assert "required print monitor failed to stop" in caplog.text


@pytest.mark.asyncio
async def test_required_monitor_failure_blocks_print() -> None:
    adapter = adapter_mock()
    monitor_start = AsyncMock(side_effect=RuntimeError("camera unavailable"))
    monitor_stop = AsyncMock()
    (
        service,
        _,
        _,
        _,
        current_binding,
        request,
        approval,
    ) = await approved_required_service(
        adapter=adapter,
        monitor_start=monitor_start,
        monitor_stop=monitor_stop,
    )

    with pytest.raises(PreflightError, match="monitor failed"):
        await service.execute(
            approval.token,
            operation_id=request.operation_id,
            binding=current_binding,
        )

    adapter.start_print.assert_not_awaited()
    monitor_stop.assert_not_awaited()
