from __future__ import annotations

from dataclasses import asdict
from unittest.mock import AsyncMock, MagicMock

import pytest

import sensing_voxel.bot.telegram as telegram_module
import sensing_voxel.main as main_module
from sensing_voxel.adapters.base import LiveSafetySnapshot
from sensing_voxel.config import (
    Config,
    KlipperConfig,
    PrintMonitorConfig,
    SlicerConfig,
    StorageConfig,
    TelegramConfig,
)
from sensing_voxel.core.models import (
    CommandResult,
    CommandResultStatus,
    NormalizedPrinterState,
)
from sensing_voxel.main import create_safety_services
from sensing_voxel.safety import PreflightError
from sensing_voxel.safety.models import canonical_sha256
from sensing_voxel.storage import PrinterMemoRepository, create_database

VALID_GCODE = (
    b"; filament_type = PLA\n"
    b"; estimated_time = 3600\n"
    b"; filament used [mm] = 1234.5\n"
    b"G28\nM104 S200\nG90\nM83\nG1 X10 Y10 Z0.2 F1200\nG1 X20 E1\n"
)


def successful_result(data: dict) -> CommandResult:
    return CommandResult(CommandResultStatus.SUCCESS, "ok", data)


def trusted_adapter(identity: str, nozzle_diameter: float = 0.6) -> AsyncMock:
    adapter = AsyncMock()
    adapter.get_config_file.return_value = successful_result(
        {
            "content": (
                f"# printer {identity}\n"
                "[extruder]\n"
                f"nozzle_diameter: {nozzle_diameter}\n"
            ),
            "truncated": False,
        }
    )
    adapter.get_hardware_info.return_value = successful_result(
        {"serial": identity, "model": "synthetic-printer"}
    )
    adapter.list_macros.return_value = successful_result(
        {"macros": [f"START_{identity.upper()}"]}
    )
    adapter.is_connected.return_value = True
    adapter.get_printer_state.return_value = NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT
    adapter.get_live_safety_snapshot.side_effect = lambda target: LiveSafetySnapshot(
        target_printer=target,
        probe_z_offset=-1.25,
        homing_origin_z=0.0,
    )
    adapter.start_print.return_value = successful_result({})
    return adapter


def trusted_memos(tmp_path, surfaces: dict[str, str]) -> PrinterMemoRepository:
    repository = PrinterMemoRepository(create_database(tmp_path))
    for printer_id, surface in surfaces.items():
        repository.upsert(printer_id, "bed_surface", surface)
    return repository


@pytest.mark.asyncio
async def test_create_safety_services_isolates_trusted_binding_per_printer(
    tmp_path,
) -> None:
    slicer = SlicerConfig(machine_profile="runtime-machine", process_profile="runtime-process")
    config = Config(
        printers=[
            KlipperConfig(
                name="alpha",
                build_plate_surface="textured PEI",
                hotend_nozzle_size_mm=0.6,
            ),
            KlipperConfig(
                name="beta",
                build_plate_surface="smooth PEI",
                hotend_nozzle_size_mm=0.4,
            ),
        ],
        slicer=slicer,
    )
    adapters = {
        "alpha": trusted_adapter("alpha-hardware", 0.6),
        "beta": trusted_adapter("beta-hardware", 0.4),
    }
    memos = trusted_memos(
        tmp_path,
        {"alpha": "textured PEI", "beta": "smooth PEI"},
    )

    services = create_safety_services(config, adapters, {}, memos)
    alpha = await services["alpha"].create_binding(
        user_id="operator-a", source_trust="user_gcode"
    )
    beta = await services["beta"].create_binding(
        user_id="operator-b", source_trust="sva_generated"
    )

    assert set(services) == {"alpha", "beta"}
    assert services["alpha"] is not services["beta"]
    assert alpha.target_printer == "alpha"
    assert beta.target_printer == "beta"
    assert alpha.user_id == "operator-a"
    assert beta.source_trust == "sva_generated"
    assert alpha.profile_digest == canonical_sha256(asdict(slicer))
    assert beta.profile_digest == alpha.profile_digest
    assert alpha.hardware_digest != beta.hardware_digest
    assert alpha.probe_z_offset == -1.25
    assert alpha.homing_origin_z == 0.0
    assert beta.probe_z_offset == -1.25
    assert beta.homing_origin_z == 0.0
    adapters["alpha"].get_config_file.assert_awaited_once_with(
        filename="printer.cfg", max_bytes=200000
    )
    adapters["beta"].get_config_file.assert_awaited_once_with(
        filename="printer.cfg", max_bytes=200000
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("source_trust", ["user_gcode", "sva_generated"])
async def test_required_monitor_fails_closed_until_runtime_monitor_exists(
    source_trust: str,
    tmp_path,
) -> None:
    config = Config(
        printers=[
            KlipperConfig(
                name="alpha",
                build_plate_surface="textured PEI",
                hotend_nozzle_size_mm=0.6,
            )
        ],
        print_monitor=PrintMonitorConfig(required=True),
    )
    adapter = trusted_adapter("alpha-hardware", 0.6)
    monitors: dict[str, AsyncMock] = {}
    memos = trusted_memos(tmp_path, {"alpha": "textured PEI"})
    service = create_safety_services(config, {"alpha": adapter}, monitors, memos)[
        "alpha"
    ]
    binding = await service.create_binding(user_id="operator", source_trust=source_trust)
    request = await service.request_from_bytes(
        VALID_GCODE,
        operation_id="autonomous-print",
        binding=binding,
        printer_file="validated.gcode",
    )
    approval = service.approve(request.request_id, binding=binding)

    with pytest.raises(PreflightError, match="required print monitor failed to start"):
        await service.execute(
            approval.token,
            operation_id=request.operation_id,
            binding=binding,
        )

    adapter.start_print.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("source_trust", ["user_gcode", "sva_generated"])
async def test_required_monitor_starts_before_printer_command(
    source_trust: str,
    tmp_path,
) -> None:
    config = Config(
        printers=[
            KlipperConfig(
                name="alpha",
                build_plate_surface="textured PEI",
                hotend_nozzle_size_mm=0.6,
            )
        ],
        print_monitor=PrintMonitorConfig(required=True),
    )
    events: list[str] = []
    adapter = trusted_adapter("alpha-hardware", 0.6)
    adapter.start_print.side_effect = lambda _filename: (
        events.append("printer"),
        successful_result({}),
    )[1]
    monitor = AsyncMock()
    monitor.is_monitoring = True
    monitor.start_monitoring.side_effect = lambda **_kwargs: events.append("monitor")
    memos = trusted_memos(tmp_path, {"alpha": "textured PEI"})
    service = create_safety_services(
        config, {"alpha": adapter}, {"alpha": monitor}, memos
    )["alpha"]
    binding = await service.create_binding(user_id="operator", source_trust=source_trust)
    request = await service.request_from_bytes(
        VALID_GCODE,
        operation_id="monitored-print",
        binding=binding,
        printer_file="validated.gcode",
    )
    approval = service.approve(request.request_id, binding=binding)

    receipt = await service.execute(
        approval.token,
        operation_id=request.operation_id,
        binding=binding,
    )

    assert receipt.started is True
    assert events == ["monitor", "printer"]
    contract = monitor.start_monitoring.await_args.kwargs["contract"]
    assert contract.filename == "validated.gcode"
    assert contract.source_trust == source_trust
    assert contract.first_layer_z == 0.2
    assert contract.required is True
    adapter.start_print.assert_awaited_once_with("validated.gcode")


@pytest.mark.asyncio
async def test_printer_start_failure_stops_only_the_matching_monitor(tmp_path) -> None:
    config = Config(
        printers=[
            KlipperConfig(
                name="alpha",
                build_plate_surface="textured PEI",
                hotend_nozzle_size_mm=0.6,
            ),
            KlipperConfig(
                name="beta",
                build_plate_surface="smooth PEI",
                hotend_nozzle_size_mm=0.4,
            ),
        ],
        print_monitor=PrintMonitorConfig(required=True),
    )
    adapters = {
        "alpha": trusted_adapter("alpha-hardware", 0.6),
        "beta": trusted_adapter("beta-hardware", 0.4),
    }
    adapters["alpha"].start_print.return_value = CommandResult(
        CommandResultStatus.FAILURE, "start rejected"
    )
    monitors = {"alpha": AsyncMock(), "beta": AsyncMock()}
    monitors["alpha"].is_monitoring = True
    monitors["beta"].is_monitoring = True
    memos = trusted_memos(
        tmp_path,
        {"alpha": "textured PEI", "beta": "smooth PEI"},
    )
    services = create_safety_services(config, adapters, monitors, memos)
    service = services["alpha"]
    binding = await service.create_binding(
        user_id="operator", source_trust="user_gcode"
    )
    request = await service.request_from_bytes(
        VALID_GCODE,
        operation_id="alpha-print",
        binding=binding,
        printer_file="validated.gcode",
    )
    approval = service.approve(request.request_id, binding=binding)

    receipt = await service.execute(
        approval.token,
        operation_id=request.operation_id,
        binding=binding,
    )

    assert receipt.started is False
    monitors["alpha"].stop_monitoring.assert_awaited_once_with(
        reason="printer start failed"
    )
    monitors["beta"].stop_monitoring.assert_not_awaited()


@pytest.mark.asyncio
async def test_main_supplies_telegram_shared_safety_services_and_active_printer(
    monkeypatch, tmp_path
) -> None:
    config = Config(
        printers=[KlipperConfig(name="alpha"), KlipperConfig(name="beta")],
        telegram=TelegramConfig(bot_token="test-token", allowed_users=[12345]),
        storage=StorageConfig(data_dir=str(tmp_path)),
    )
    adapters = {"alpha": AsyncMock(), "beta": AsyncMock()}
    safety_services = {"alpha": MagicMock(), "beta": MagicMock()}
    captured: dict = {}

    monkeypatch.setattr(main_module, "load_config", lambda: config)
    monkeypatch.setattr(main_module, "create_adapters", lambda _config: adapters)
    monkeypatch.setattr(
        main_module,
        "create_safety_services",
        lambda _config, actual_adapters, _monitors, _memos: (
            safety_services
            if actual_adapters is adapters
            else pytest.fail("main did not share the production adapter mapping")
        ),
    )

    class FakeTelegramBot:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def start(self):
            raise RuntimeError("stop before network startup")

        async def stop(self):
            return None

    monkeypatch.setattr(telegram_module, "TelegramBot", FakeTelegramBot)

    with pytest.raises(RuntimeError, match="stop before network startup"):
        await main_module.main()

    assert captured["adapter"] is adapters["alpha"]
    assert captured["safety_services"] is safety_services
    assert captured["active_printer"] == "alpha"
