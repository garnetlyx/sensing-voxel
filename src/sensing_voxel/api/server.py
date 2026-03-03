"""Dashboard web server factory using aiohttp"""

import logging
from collections import deque
from pathlib import Path

from aiohttp import web

from sensing_voxel.api.auth import create_auth_middleware
from sensing_voxel.api.camera_proxy import camera_stream
from sensing_voxel.api.routes import setup_routes
from sensing_voxel.api.websocket import WebSocketManager
from sensing_voxel.config import DashboardConfig
from sensing_voxel.core.decision import DecisionEngine
from sensing_voxel.core.state_machine import AgentStateMachine

logger = logging.getLogger(__name__)

# Max alerts kept in ring buffer
MAX_ALERTS = 50


def create_app(
    config: DashboardConfig,
    state_machine: AgentStateMachine,
    decision_engine: DecisionEngine | None = None,
    adapters: dict | None = None,
    active_printer: str | None = None,
    chat_manager=None,
) -> tuple[web.Application, WebSocketManager]:
    """Create and configure the dashboard web application.

    Returns (app, ws_manager) so the caller can wire up event callbacks.
    """
    if adapters is None:
        adapters = {}

    ws_manager = WebSocketManager()

    # Middleware stack
    middlewares = [create_auth_middleware(config.api_token)]

    app = web.Application(middlewares=middlewares)

    # Shared application state (multi-printer)
    app["app_state"] = {
        "state_machine": state_machine,
        "decision_engine": decision_engine,
        "adapters": adapters,
        "connections": {pid: False for pid in adapters},
        "telemetries": {pid: None for pid in adapters},
        "active_printer": active_printer,
        "ws_manager": ws_manager,
        "alerts": deque(maxlen=MAX_ALERTS),
        "camera_rtsp_url": config.camera_rtsp_url,
        "camera_mjpeg_fps": config.camera_mjpeg_fps,
        "camera_mjpeg_quality": config.camera_mjpeg_quality,
        "chat_manager": chat_manager,
    }

    # Register REST + WebSocket routes
    setup_routes(app, ws_manager)

    # Camera stream route
    if config.camera_rtsp_url:
        app.router.add_get("/api/v1/camera/stream", camera_stream)

    # Static files (dashboard frontend)
    static_dir = Path(__file__).parent.parent / "dashboard" / "static"
    if static_dir.exists():
        app.router.add_static("/static/", static_dir, show_index=False)

        # Serve index.html at /
        async def index_handler(request: web.Request) -> web.FileResponse:
            return web.FileResponse(static_dir / "index.html")

        app.router.add_get("/", index_handler)

    return app, ws_manager


async def start_server(
    config: DashboardConfig,
    state_machine: AgentStateMachine,
    decision_engine: DecisionEngine | None = None,
    adapters: dict | None = None,
    active_printer: str | None = None,
    chat_manager=None,
) -> tuple[web.AppRunner, WebSocketManager]:
    """Start the dashboard server (non-blocking, coexists with asyncio loop).

    Returns (runner, ws_manager). Caller must call runner.cleanup() on shutdown.
    """
    app, ws_manager = create_app(
        config, state_machine, decision_engine, adapters, active_printer, chat_manager
    )

    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, config.host, config.port)
    await site.start()

    logger.info(f"Dashboard server started at http://{config.host}:{config.port}")

    return runner, ws_manager
