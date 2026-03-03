"""WebSocket handler with fan-out broadcasting"""

import asyncio
import json
import logging
import weakref
from datetime import datetime

from aiohttp import web, WSMsgType

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections and broadcasts events"""

    def __init__(self):
        self._clients: weakref.WeakSet[web.WebSocketResponse] = weakref.WeakSet()
        self._lock = asyncio.Lock()

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Handle new WebSocket connection"""
        ws = web.WebSocketResponse(heartbeat=30.0)
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
