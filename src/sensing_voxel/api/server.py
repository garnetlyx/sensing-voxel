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
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def validate_dashboard_security(config: DashboardConfig) -> None:
    """Reject unauthenticated dashboard exposure beyond the local machine."""
    if config.host.strip().lower() not in _LOOPBACK_HOSTS and not config.api_token:
        raise ValueError("Dashboard api_token is required when binding to a non-loopback host")


def create_app(
    config: DashboardConfig,
    state_machine: AgentStateMachine,
    decision_engine: DecisionEngine | None = None,
    adapters: dict | None = None,
    active_printer: str | None = None,
    chat_manager=None,
    camera=None,
    safety_services: dict | None = None,
) -> tuple[web.Application, WebSocketManager]:
    """Create and configure the dashboard web application.

    Returns (app, ws_manager) so the caller can wire up event callbacks.
    """
    if adapters is None:
        adapters = {}

    validate_dashboard_security(config)
    ws_manager = WebSocketManager(
        api_token=config.api_token,
        allowed_origins=set(config.allowed_origins),
    )

    # Middleware stack
    middlewares = [create_auth_middleware(config.api_token)]

    app = web.Application(middlewares=middlewares)

    # Shared application state (multi-printer)
    app["app_state"] = {
        "state_machine": state_machine,
        "decision_engine": decision_engine,
        "adapters": adapters,
        "connections": dict.fromkeys(adapters, False),
        "telemetries": dict.fromkeys(adapters),
        "active_printer": active_printer,
        "ws_manager": ws_manager,
        "alerts": deque(maxlen=MAX_ALERTS),
        "camera_rtsp_url": config.camera_rtsp_url,
        "camera_mjpeg_fps": config.camera_mjpeg_fps,
        "camera_mjpeg_quality": config.camera_mjpeg_quality,
        "chat_manager": chat_manager,
        "camera": camera,
        "safety_services": safety_services or {},
        "upload_dir": Path(config.upload_dir),
        "max_upload_bytes": config.max_upload_bytes,
        "allowed_upload_extensions": {
            extension.lower() for extension in config.allowed_upload_extensions
        },
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
    camera=None,
    safety_services: dict | None = None,
) -> tuple[web.AppRunner, WebSocketManager]:
    """Start the dashboard server (non-blocking, coexists with asyncio loop).

    Returns (runner, ws_manager). Caller must call runner.cleanup() on shutdown.
    """
    app, ws_manager = create_app(
        config,
        state_machine,
        decision_engine,
        adapters,
        active_printer,
        chat_manager,
        camera,
        safety_services,
    )

    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, config.host, config.port)
    await site.start()

    logger.info(f"Dashboard server started at http://{config.host}:{config.port}")

    return runner, ws_manager
