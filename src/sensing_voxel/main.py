#!/usr/bin/env python3
"""
Sensing Voxel - Main entry point

AI-powered 3D printer monitoring and control system.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

import yaml

from sensing_voxel.adapters.klipper import KlipperAdapter
from sensing_voxel.bot.telegram import TelegramBot
from sensing_voxel.config import Config
from sensing_voxel.core.decision import DecisionEngine
from sensing_voxel.core.persistence import StatePersistence
from sensing_voxel.core.state_machine import AgentStateMachine

logger = logging.getLogger(__name__)


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base dict (override wins)."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config() -> Config:
    """Load config from default.yaml, overridden by local.yaml if present."""
    config_path = Path("config/default.yaml")
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    local_path = Path("config/local.yaml")
    if local_path.exists():
        with open(local_path) as f:
            local_dict = yaml.safe_load(f)
        if local_dict:
            config_dict = deep_merge(config_dict, local_dict)
            logger.info("Loaded config/local.yaml overrides")

    return Config.from_dict(config_dict)


async def shutdown(
    signal: signal.Signals,
    loop: asyncio.AbstractEventLoop,
    state_machine: AgentStateMachine,
    persistence: StatePersistence,
    dashboard_runner=None,
) -> None:
    """Graceful shutdown handler."""
    print(f"Received exit signal {signal.name}...")

    # Stop dashboard server
    if dashboard_runner:
        await dashboard_runner.cleanup()

    # Cancel all tasks
    tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()

    # Persist final state
    try:
        await state_machine.persist(persistence)
    except Exception as e:
        print(f"Error during final persistence: {e}")

    print(f"Cancelling {len(tasks)} outstanding tasks")
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()


async def main() -> None:
    """Main async entry point."""
    # Load configuration
    config = load_config()

    # Initialize persistence
    persistence = StatePersistence(config.state_file_path)

    # Load or create initial state
    state_machine = await AgentStateMachine.load_or_create(persistence)

    # Initialize decision engine
    decision_engine = DecisionEngine()

    # Initialize adapter
    adapter: KlipperAdapter | None = None
    if config.klipper:
        adapter = KlipperAdapter(
            host=config.klipper.host,
            port=config.klipper.port,
            api_key=config.klipper.api_key,
        )

    # Initialize bot (optional — skip if no real token configured)
    bot = None
    token = config.telegram.bot_token
    if token and token != "YOUR_BOT_TOKEN_HERE":
        bot = TelegramBot(
            token=token,
            allowed_users=config.telegram.allowed_users,
            state_machine=state_machine,
            adapter=adapter,
            config=config,
            persistence=persistence,
        )
    else:
        logger.info("Telegram bot disabled (no token configured)")

    # Start dashboard if enabled
    dashboard_runner = None
    ws_manager = None
    if config.dashboard.enabled:
        from sensing_voxel.api.server import start_server

        dashboard_runner, ws_manager = await start_server(
            config=config.dashboard,
            state_machine=state_machine,
            decision_engine=decision_engine,
            adapter=adapter,
        )

        # Wire adapter events to WebSocket broadcast
        if adapter and ws_manager:
            from sensing_voxel.api.serializers import serialize_telemetry
            from sensing_voxel.core.models import NormalizedPrinterState, TelemetryData

            app_state = dashboard_runner.app["app_state"] if dashboard_runner.app else None

            async def on_state_changed(old: NormalizedPrinterState, new: NormalizedPrinterState):
                await ws_manager.broadcast_state_change(state_machine.current_state.name, new.name)

            async def on_telemetry(data: TelemetryData):
                if app_state:
                    app_state["last_telemetry"] = data
                tel = serialize_telemetry(data)
                if tel:
                    await ws_manager.broadcast_telemetry(tel)

            async def on_connection_lost():
                if app_state:
                    app_state["printer_connected"] = False
                await ws_manager.broadcast_alert("warning", "Printer connection lost")

            async def on_connection_restored():
                if app_state:
                    app_state["printer_connected"] = True
                await ws_manager.broadcast_alert("success", "Printer reconnected")

            adapter.on_state_changed(on_state_changed)
            adapter.on_telemetry_update(on_telemetry)
            adapter.on_connection_lost(on_connection_lost)
            adapter.on_connection_restored(on_connection_restored)

    # Set up signal handlers
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(
            sig,
            lambda sig=sig: asyncio.create_task(
                shutdown(sig, loop, state_machine, persistence, dashboard_runner)
            ),
        )

    # Start bot and adapter
    try:
        if adapter:
            await adapter.connect()

        if bot:
            await bot.start()

        # Keep running until shutdown
        await asyncio.Event().wait()

    finally:
        # Cleanup
        if adapter:
            await adapter.disconnect()
        if bot:
            await bot.stop()
        if dashboard_runner:
            await dashboard_runner.cleanup()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
