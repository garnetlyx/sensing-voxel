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

from sensing_voxel.adapters.base import PrinterAdapter
from sensing_voxel.adapters.klipper import KlipperAdapter
from sensing_voxel.bot.telegram import TelegramBot
from sensing_voxel.config import BambuConfig, Config, KlipperConfig
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


def create_adapters(config: Config) -> dict[str, PrinterAdapter]:
    """Create printer adapters from config."""
    adapters: dict[str, PrinterAdapter] = {}
    for printer_cfg in config.printers:
        if isinstance(printer_cfg, KlipperConfig):
            adapters[printer_cfg.name] = KlipperAdapter(
                host=printer_cfg.host,
                port=printer_cfg.port,
                api_key=printer_cfg.api_key,
                base_url=printer_cfg.base_url,
            )
        elif isinstance(printer_cfg, BambuConfig):
            from sensing_voxel.adapters.bambu import BambuAdapter

            adapters[printer_cfg.name] = BambuAdapter(
                host=printer_cfg.mqtt_host,
                mqtt_port=printer_cfg.mqtt_port,
            )
        logger.info(f"Created adapter: {printer_cfg.name} ({type(printer_cfg).__name__})")
    return adapters


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

    # Persist final state (use async lock to avoid race with in-flight saves)
    try:
        await persistence.save_async(state_machine)
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

    # Initialize adapters (multi-printer)
    adapters = create_adapters(config)
    active_printer = next(iter(adapters), None)

    # Initialize Tapo camera (optional)
    tapo_camera = None
    if config.tapo_camera and config.tapo_camera.host:
        from sensing_voxel.camera.tapo import TapoCamera

        tapo_camera = TapoCamera(config.tapo_camera)
        try:
            await tapo_camera.connect()
        except Exception as e:
            logger.error(f"Failed to connect Tapo camera: {e}")
            tapo_camera = None

    # Initialize LLM (optional)
    llm_client = None
    chat_manager = None
    if config.llm.enabled and config.llm.api_key:
        from sensing_voxel.llm.chat import ChatManager
        from sensing_voxel.llm.client import LlmClient
        from sensing_voxel.llm.context import PrinterContextCollector

        llm_client = LlmClient(config.llm)
        await llm_client.initialize()

        context_collector = PrinterContextCollector(
            state_machine=state_machine,
            adapters=adapters,
            active_printer=active_printer,
        )

        chat_manager = ChatManager(
            client=llm_client,
            context_collector=context_collector,
            adapters=adapters,
            active_printer=active_printer,
            camera=tapo_camera,
            rtsp_url=config.dashboard.camera_rtsp_url,
            shell=config.llm.shell,
            web_search=config.llm.web_search,
            max_turns=config.llm.max_conversation_turns,
            history_path=Path(config.storage.data_dir) / "chat_history.json",
            verbose_tool_use=config.llm.verbose_tool_use,
        )
        chat_manager.load_history()
        logger.info(f"LLM initialized: model={config.llm.model}")
    else:
        logger.info("LLM disabled (no api_key or enabled=false)")

    # Initialize bot (optional — skip if no real token configured)
    bot = None
    token = config.telegram.bot_token
    if token and token != "YOUR_BOT_TOKEN_HERE":
        # Bot uses the active (first) printer adapter
        active_adapter = adapters.get(active_printer) if active_printer else None
        bot = TelegramBot(
            token=token,
            allowed_users=config.telegram.allowed_users,
            state_machine=state_machine,
            adapter=active_adapter,
            config=config,
            persistence=persistence,
            chat_manager=chat_manager,
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
            adapters=adapters,
            active_printer=active_printer,
            chat_manager=chat_manager,
            camera=tapo_camera,
        )

        # Wire per-printer adapter events to WebSocket broadcast
        if adapters and ws_manager:
            from sensing_voxel.api.serializers import serialize_telemetry
            from sensing_voxel.core.models import NormalizedPrinterState, TelemetryData

            app_state = dashboard_runner.app["app_state"] if dashboard_runner.app else None

            for printer_id, adapter in adapters.items():

                async def on_state_changed(
                    old: NormalizedPrinterState,
                    new: NormalizedPrinterState,
                    pid: str = printer_id,
                ):
                    await ws_manager.broadcast_state_change(
                        state_machine.current_state.name, new.name, printer_id=pid
                    )

                async def on_telemetry(data: TelemetryData, pid: str = printer_id):
                    if app_state:
                        app_state["telemetries"][pid] = data
                    if chat_manager:
                        chat_manager._context_collector.update_telemetry(pid, data)
                    tel = serialize_telemetry(data)
                    if tel:
                        await ws_manager.broadcast_telemetry(tel, printer_id=pid)

                async def on_connection_lost(pid: str = printer_id):
                    if app_state:
                        app_state["connections"][pid] = False
                    if chat_manager:
                        chat_manager._context_collector.update_connection(pid, False)
                    await ws_manager.broadcast_alert(
                        "warning", f"Printer '{pid}' connection lost", printer_id=pid
                    )

                async def on_connection_restored(pid: str = printer_id):
                    if app_state:
                        app_state["connections"][pid] = True
                    if chat_manager:
                        chat_manager._context_collector.update_connection(pid, True)
                    await ws_manager.broadcast_alert(
                        "success", f"Printer '{pid}' reconnected", printer_id=pid
                    )

                adapter.on_state_changed(on_state_changed)
                adapter.on_telemetry_update(on_telemetry)
                adapter.on_connection_lost(on_connection_lost)
                adapter.on_connection_restored(on_connection_restored)

    # Wire LLM context updates when dashboard is NOT enabled
    if chat_manager and not ws_manager and adapters:
        from sensing_voxel.core.models import NormalizedPrinterState, TelemetryData

        for printer_id, adapter in adapters.items():

            async def on_telemetry_llm(data: TelemetryData, pid: str = printer_id):
                chat_manager._context_collector.update_telemetry(pid, data)

            async def on_connection_lost_llm(pid: str = printer_id):
                chat_manager._context_collector.update_connection(pid, False)

            async def on_connection_restored_llm(pid: str = printer_id):
                chat_manager._context_collector.update_connection(pid, True)

            adapter.on_telemetry_update(on_telemetry_llm)
            adapter.on_connection_lost(on_connection_lost_llm)
            adapter.on_connection_restored(on_connection_restored_llm)

    # Set up signal handlers
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(
            sig,
            lambda sig=sig: asyncio.create_task(
                shutdown(sig, loop, state_machine, persistence, dashboard_runner)
            ),
        )

    # Start bot and adapters
    try:
        for printer_id, adapter in adapters.items():
            logger.info(f"Connecting adapter: {printer_id}")
            try:
                await adapter.connect()
            except Exception as e:
                logger.error(f"Failed to connect adapter '{printer_id}': {e}")

        if bot:
            await bot.start()

        # Keep running until shutdown
        await asyncio.Event().wait()

    finally:
        # Cleanup
        if tapo_camera:
            await tapo_camera.disconnect()
        if llm_client:
            await llm_client.close()
        for adapter in adapters.values():
            try:
                await adapter.disconnect()
            except Exception:
                pass
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
