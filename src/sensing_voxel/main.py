#!/usr/bin/env python3
"""
Sensing Voxel - Main entry point

AI-powered 3D printer monitoring and control system.
"""

import asyncio
import signal
import sys
from pathlib import Path

import yaml

from sensing_voxel.adapters.klipper import KlipperAdapter
from sensing_voxel.bot.telegram import TelegramBot
from sensing_voxel.config import Config
from sensing_voxel.core.persistence import StatePersistence
from sensing_voxel.core.state_machine import AgentStateMachine


async def shutdown(
    signal: signal.Signals,
    loop: asyncio.AbstractEventLoop,
    state_machine: AgentStateMachine,
    persistence: StatePersistence,
) -> None:
    """Graceful shutdown handler."""
    print(f"Received exit signal {signal.name}...")

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
    config_path = Path("config/default.yaml")
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    config = Config(**config_dict)

    # Initialize persistence
    persistence = StatePersistence(config.state_file_path)

    # Load or create initial state
    state_machine = await AgentStateMachine.load_or_create(persistence)

    # Initialize adapter
    adapter: KlipperAdapter | None = None
    if config.klipper:
        adapter = KlipperAdapter(
            host=config.klipper.host,
            port=config.klipper.port,
            api_key=config.klipper.api_key,
        )

    # Initialize bot
    bot = TelegramBot(
        token=config.telegram.bot_token,
        allowed_users=config.telegram.allowed_users,
        state_machine=state_machine,
        adapter=adapter,
        config=config,
        persistence=persistence,
    )

    # Set up signal handlers
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(
            sig,
            lambda sig=sig: asyncio.create_task(shutdown(sig, loop, state_machine, persistence)),
        )

    # Start bot and adapter
    try:
        if adapter:
            await adapter.connect()

        await bot.start()

        # Keep running until shutdown
        await asyncio.Event().wait()

    finally:
        # Cleanup
        if adapter:
            await adapter.disconnect()
        await bot.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
