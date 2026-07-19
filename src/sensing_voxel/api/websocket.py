"""WebSocket handler with fan-out broadcasting"""

import asyncio
import base64
import binascii
import hmac
import json
import logging
import weakref
from datetime import datetime
from urllib.parse import urlsplit

from aiohttp import WSMsgType, web

logger = logging.getLogger(__name__)

WS_APPLICATION_PROTOCOL = "sensing-voxel.v1"
WS_AUTH_PROTOCOL_PREFIX = "sensing-voxel.auth."


class WebSocketManager:
    """Manages authenticated WebSocket connections and broadcasts events."""

    def __init__(
        self,
        api_token: str | None = None,
        allowed_origins: set[str] | None = None,
    ):
        self._api_token = api_token
        self._allowed_origins = frozenset(allowed_origins or ())
        self._clients: weakref.WeakSet[web.WebSocketResponse] = weakref.WeakSet()
        self._lock = asyncio.Lock()

    @staticmethod
    def _normalize_origin(origin: str) -> str:
        parsed = urlsplit(origin.strip())
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Origin must be an absolute HTTP(S) URL")
        return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"

    def _origin_allowed(self, request: web.Request, origin: str) -> bool:
        try:
            normalized = self._normalize_origin(origin)
        except ValueError:
            return False

        if self._allowed_origins:
            try:
                allowed = {self._normalize_origin(value) for value in self._allowed_origins}
            except ValueError:
                logger.error("Invalid configured WebSocket origin")
                return False
            return normalized in allowed

        return normalized == f"{request.scheme.lower()}://{request.host.lower()}"

    @staticmethod
    def _offered_protocols(request: web.Request) -> tuple[str, ...]:
        header = request.headers.get("Sec-WebSocket-Protocol", "")
        return tuple(protocol.strip() for protocol in header.split(",") if protocol.strip())

    def _protocol_token(self, request: web.Request) -> bytes | None:
        protocols = self._offered_protocols(request)
        if WS_APPLICATION_PROTOCOL not in protocols:
            return None

        encoded_tokens = [
            protocol.removeprefix(WS_AUTH_PROTOCOL_PREFIX)
            for protocol in protocols
            if protocol.startswith(WS_AUTH_PROTOCOL_PREFIX)
        ]
        if len(encoded_tokens) != 1 or not encoded_tokens[0]:
            return None

        encoded = encoded_tokens[0]
        try:
            padding = "=" * (-len(encoded) % 4)
            return base64.b64decode(encoded + padding, altchars=b"-_", validate=True)
        except (binascii.Error, ValueError):
            return None

    def _authorize_upgrade(self, request: web.Request) -> None:
        if not self._api_token:
            raise web.HTTPUnauthorized(
                text="WebSocket authentication is not configured",
                headers={"WWW-Authenticate": "Bearer"},
            )

        expected = self._api_token.encode()
        auth_header = request.headers.get("Authorization", "")
        scheme, separator, bearer_token = auth_header.partition(" ")
        bearer_valid = (
            separator == " "
            and scheme.lower() == "bearer"
            and bool(bearer_token)
            and hmac.compare_digest(bearer_token.encode(), expected)
        )
        protocol_token = self._protocol_token(request)
        protocol_valid = protocol_token is not None and hmac.compare_digest(
            protocol_token, expected
        )
        if not bearer_valid and not protocol_valid:
            raise web.HTTPUnauthorized(
                text="Missing or invalid WebSocket credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        origin = request.headers.get("Origin")
        if not origin or not self._origin_allowed(request, origin):
            raise web.HTTPForbidden(text="WebSocket origin is not allowed")

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Validate and handle a new WebSocket connection."""
        self._authorize_upgrade(request)
        offered_protocols = self._offered_protocols(request)
        protocols = (
            (WS_APPLICATION_PROTOCOL,) if WS_APPLICATION_PROTOCOL in offered_protocols else ()
        )
        ws = web.WebSocketResponse(heartbeat=30.0, protocols=protocols)
        await ws.prepare(request)

        async with self._lock:
            self._clients.add(ws)

        logger.info(f"WebSocket client connected ({self.client_count} total)")

        # Send initial snapshot
        app_state = request.app.get("app_state")
        if app_state:
            await self._send_snapshot(ws, app_state)

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    # Clients can send ping/pong or requests
                    try:
                        data = json.loads(msg.data)
                        if data.get("type") == "ping":
                            await ws.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})
                    except (json.JSONDecodeError, TypeError):
                        pass
                elif msg.type == WSMsgType.ERROR:
                    logger.warning(f"WebSocket error: {ws.exception()}")
        finally:
            async with self._lock:
                self._clients.discard(ws)
            logger.info(f"WebSocket client disconnected ({self.client_count} total)")

        return ws

    async def _send_snapshot(self, ws: web.WebSocketResponse, app_state: dict):
        """Send full state snapshot to a single client (multi-printer aware)"""
        from sensing_voxel.api.serializers import (
            serialize_job_context,
            serialize_state_machine,
            serialize_telemetry,
        )

        sm = app_state.get("state_machine")
        if not sm:
            return

        # Build per-printer data
        connections = app_state.get("connections", {})
        telemetries = app_state.get("telemetries", {})
        printers = []
        for pid in app_state.get("adapters", {}):
            printers.append({
                "id": pid,
                "connected": connections.get(pid, False),
                "telemetry": serialize_telemetry(telemetries.get(pid)),
            })

        snapshot = {
            "type": "snapshot",
            "timestamp": datetime.now().isoformat(),
            "state": serialize_state_machine(sm),
            "job": serialize_job_context(sm.job_context),
            "printers": printers,
            "active_printer": app_state.get("active_printer"),
        }

        try:
            await ws.send_json(snapshot)
        except (ConnectionResetError, RuntimeError):
            pass

    async def broadcast(self, event: dict):
        """Broadcast event to all connected clients"""
        if not self._clients:
            return

        event.setdefault("timestamp", datetime.now().isoformat())
        payload = json.dumps(event)

        # Copy set to avoid mutation during iteration
        async with self._lock:
            clients = list(self._clients)

        closed = []
        for ws in clients:
            try:
                await ws.send_str(payload)
            except (ConnectionResetError, RuntimeError):
                closed.append(ws)

        if closed:
            async with self._lock:
                for ws in closed:
                    self._clients.discard(ws)

    async def broadcast_state_change(
        self, agent_state: str, printer_state: str | None = None, printer_id: str | None = None
    ):
        """Broadcast state change event"""
        await self.broadcast({
            "type": "state_change",
            "printer_id": printer_id,
            "agent_state": agent_state,
            "printer_state": printer_state,
        })

    async def broadcast_telemetry(self, telemetry: dict, printer_id: str | None = None):
        """Broadcast telemetry update"""
        await self.broadcast({
            "type": "telemetry",
            "printer_id": printer_id,
            **telemetry,
        })

    async def broadcast_progress(self, progress: dict, printer_id: str | None = None):
        """Broadcast print progress update"""
        await self.broadcast({
            "type": "progress",
            "printer_id": printer_id,
            **progress,
        })

    async def broadcast_alert(
        self, alert_type: str, message: str, data: dict | None = None, printer_id: str | None = None
    ):
        """Broadcast alert to all clients"""
        await self.broadcast({
            "type": "alert",
            "printer_id": printer_id,
            "alert_type": alert_type,
            "message": message,
            "data": data,
        })

    async def broadcast_decision_request(self, job_id: str, job_file: str, token: str):
        """Broadcast decision request (approval needed)"""
        await self.broadcast({
            "type": "decision_request",
            "job_id": job_id,
            "job_file": job_file,
            "token": token,
        })

    async def broadcast_decision_result(self, job_id: str, action: str):
        """Broadcast decision result"""
        await self.broadcast({
            "type": "decision_result",
            "job_id": job_id,
            "action": action,
        })
