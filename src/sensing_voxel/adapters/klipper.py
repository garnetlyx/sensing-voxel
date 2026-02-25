"""
Klipper/Moonraker adapter

Adapted from OpenClaw's skills/lucakaufmann/snapmaker/scripts/snapmaker.py
"""

import aiohttp
from typing import Optional
import logging
import asyncio

from sensing_voxel.adapters.base import PrinterAdapter, NotSupportedError
from sensing_voxel.core.models import (
    NormalizedPrinterState,
    PrintProgress,
    TelemetryData,
    CommandResult,
    CommandResultStatus,
    AdapterConnectionError,
)
from sensing_voxel.infra.retry import retry_async, BackoffPolicy

logger = logging.getLogger(__name__)


class KlipperAdapter(PrinterAdapter):
    """Klipper/Moonraker adapter with REST + WebSocket"""

    STATE_MAP = {
        "printing": NormalizedPrinterState.PRINTING,
        "paused": NormalizedPrinterState.PAUSED,
        "standby": NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT,
        "idle": NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT,
        "complete": NormalizedPrinterState.TERMINAL,
        "cancelled": NormalizedPrinterState.TERMINAL,
        "error": NormalizedPrinterState.ERROR,
    }

    def __init__(
        self,
        host: str = "localhost",
        port: int = 7125,
        api_key: Optional[str] = None,
        compatibility_mode: bool = False,
    ):
        super().__init__()
        self.host = host
        self.port = port
        self.api_key = api_key
        self.compatibility_mode = compatibility_mode

        self._rest_base_url = f"http://{host}:{port}"
        self._ws_url = f"ws://{host}:{port}/websocket"

        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._connected = False
        self._shutdown_event = asyncio.Event()

        # WebSocket reconnection policy
        self._reconnect_policy = BackoffPolicy(
            base_delay=1.0,
            max_delay=30.0,
            factor=1.8,
            jitter=0.25,
        )
        self._reconnect_attempts = 0

    async def connect(self) -> None:
        """Connect to Moonraker"""
        if self._connected:
            return

        # Create REST session
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        self._session = aiohttp.ClientSession(headers=headers)

        # Start WebSocket connection task
        asyncio.create_task(self._ws_reconnect_loop())

        self._connected = True
        logger.info(f"Connected to Moonraker at {self.host}:{self.port}")

    async def disconnect(self) -> None:
        """Disconnect from Moonraker"""
        if not self._connected:
            return

        self._shutdown_event.set()

        if self._ws:
            await self._ws.close()

        if self._session:
            await self._session.close()

        self._connected = False
        logger.info("Disconnected from Moonraker")

    async def _ws_reconnect_loop(self):
        """WebSocket reconnection loop (OpenClaw SSE reconnect pattern)"""
        while not self._shutdown_event.is_set():
            try:
                await self._ws_connect_and_stream()
                # Stream ended normally
            except (AdapterConnectionError, aiohttp.ClientError) as e:
                logger.warning(f"WebSocket connection error: {e}")
            except Exception as e:
                logger.error(f"WebSocket error: {e}", exc_info=True)

            # Reconnect delay
            delay = min(
                self._reconnect_policy.base_delay
                * (self._reconnect_policy.factor ** self._reconnect_attempts),
                self._reconnect_policy.max_delay,
            )

            if self._reconnect_policy.jitter > 0:
                delay += delay * self._reconnect_policy.jitter * asyncio.get_event_loop().time() % 1

            logger.info(f"Reconnecting in {delay:.1f}s (attempt {self._reconnect_attempts + 1})")
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=delay)
                break
            except asyncio.TimeoutError:
                self._reconnect_attempts += 1

    async def _ws_connect_and_stream(self):
        """Connect WebSocket and stream events"""
        if not self._session:
            raise AdapterConnectionError("REST session not initialized")

        async with self._session.ws_connect(self._ws_url) as ws:
            self._ws = ws
            self._reconnect_attempts = 0  # Reset on successful connection
            await self._emit_connection_restored()

            logger.info("WebSocket connected")

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        await self._handle_ws_message(msg.json())
                    except Exception as e:
                        logger.error(f"Error handling WebSocket message: {e}", exc_info=True)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    raise AdapterConnectionError(f"WebSocket error: {ws.exception()}")

    async def _handle_ws_message(self, data: dict):
        """Handle WebSocket message"""
        method = data.get("method")
        params = data.get("params", [{}])[0]

        if method == "notify_status_update":
            await self._handle_status_update(params)
        elif method == "notify_gcode_output":
            await self._handle_gcode_output(params)

    async def _handle_status_update(self, status: dict):
        """Handle status update from printer"""
        print_stats = status.get("print_stats", {})
        state_str = print_stats.get("state", "unknown").lower()

        normalized_state = self.STATE_MAP.get(state_str, NormalizedPrinterState.UNKNOWN)

        # Emit state changed event if state changed
        if hasattr(self, "_last_state") and self._last_state != normalized_state:
            await self._emit_state_changed(self._last_state, normalized_state)
            logger.debug(
                f"State changed: {self._last_state.name} -> {normalized_state.name}"
            )
        elif not hasattr(self, "_last_state"):
            logger.debug(f"Initial state: {normalized_state.name}")

        self._last_state = normalized_state

        # Parse telemetry data
        telemetry = self._parse_telemetry(status, print_stats)
        if telemetry:
            await self._emit_telemetry_update(telemetry)

    def _parse_telemetry(self, status: dict, print_stats: dict) -> Optional[TelemetryData]:
        """Parse telemetry data from status update"""
        from datetime import datetime

        # Temperature data
        temps = {}
        heater_bed = status.get("heater_bed", {})
        if heater_bed:
            temps["bed"] = heater_bed.get("temperature", 0.0)
            temps["bed_target"] = heater_bed.get("target", 0.0)

        extruder = status.get("extruder", {})
        if extruder:
            temps["extruder"] = extruder.get("temperature", 0.0)
            temps["extruder_target"] = extruder.get("target", 0.0)

        # Position data
        toolhead = status.get("toolhead", {})
        position = None
        if toolhead:
            position = {
                "x": toolhead.get("position", [0, 0, 0])[0],
                "y": toolhead.get("position", [0, 0, 0])[1],
                "z": toolhead.get("position", [0, 0, 0])[2],
            }

        # Print progress
        metadata = {}
        if print_stats:
            metadata["filename"] = print_stats.get("filename", "")
            metadata["progress"] = print_stats.get("progress", 0.0)
            metadata["print_time"] = print_stats.get("print_duration", 0.0)
            metadata["state"] = print_stats.get("state", "")

        if not temps and not position:
            return None

        return TelemetryData(
            timestamp=datetime.now(),
            temperatures=temps,
            position=position,
            metadata=metadata,
        )

    async def _handle_gcode_output(self, params: dict):
        """Handle G-code output"""
        pass

    # --- Commands ---

    async def start_print(self, gcode_path: str) -> CommandResult:
        return await self._rest_request(
            "POST",
            "/printer/print/start",
            json={"filename": gcode_path},
        )

    async def pause_print(self) -> CommandResult:
        return await self._rest_request("POST", "/printer/print/pause")

    async def resume_print(self) -> CommandResult:
        return await self._rest_request("POST", "/printer/print/resume")

    async def cancel_print(self) -> CommandResult:
        return await self._rest_request("POST", "/printer/print/cancel")

    async def emergency_cancel(self) -> CommandResult:
        # Use dedicated session to bypass queue
        return await self._rest_request("POST", "/printer/print/cancel")

    async def upload_file(self, local_path: str, remote_name: str) -> CommandResult:
        """Upload file to printer via Moonraker"""
        from pathlib import Path

        try:
            path = Path(local_path)
            if not path.exists():
                return CommandResult(
                    CommandResultStatus.FAILURE,
                    message=f"File not found: {local_path}",
                )

            # Read file content
            with open(path, "rb") as f:
                file_content = f.read()

            # Moonraker upload endpoint
            url = f"{self._rest_base_url}/server/files/upload"

            # Prepare multipart form data
            data = aiohttp.FormData()
            data.add_field(
                "file",
                file_content,
                filename=remote_name,
                content_type="application/octet-stream",
            )

            # Optional: specify path in printer's file system
            data.add_field("path", "sdcard")  # Default to sdcard directory

            result = await self._rest_request("POST", "/server/files/upload", data=data)

            if result.status == CommandResultStatus.SUCCESS:
                logger.info(f"Uploaded file: {remote_name}")
            return result

        except Exception as e:
            return CommandResult(
                CommandResultStatus.FAILURE,
                message=f"Upload failed: {str(e)}",
            )

    async def get_z_offset(self) -> float:
        """Get current Z offset via gcode"""
        # Query gcode_move object for current Z offset
        result = await self._rest_request(
            "GET", "/printer/objects/query?gcode_move"
        )

        if result.status == CommandResultStatus.SUCCESS and result.data:
            gcode_move = result.data.get("gcode_move", {})
            return gcode_move.get("homing_origin", [0, 0, 0])[2]  # Z component

        raise AdapterCommandError("Failed to query Z offset")

    async def set_z_offset(self, offset_mm: float) -> CommandResult:
        """Set Z offset via SET_GCODE_OFFSET command"""
        gcode = f"SET_GCODE_OFFSET Z_ADJUST={offset_mm:.4f} MOVE=1"
        return await self._rest_request(
            "POST",
            "/printer/gcode/script",
            json={"script": gcode},
        )

    # --- Queries ---

    async def get_printer_state(self) -> NormalizedPrinterState:
        result = await self._rest_request("GET", "/printer/objects/query?print_stats")
        if result.status == CommandResultStatus.SUCCESS and result.data:
            state_str = result.data.get("print_stats", {}).get("state", "unknown")
            return self.STATE_MAP.get(
                state_str.lower(), NormalizedPrinterState.UNKNOWN
            )
        return NormalizedPrinterState.UNKNOWN

    async def get_print_progress(self) -> PrintProgress:
        """Get current print progress and state"""
        # Query comprehensive printer status
        result = await self._rest_request(
            "GET",
            "/printer/objects/query?"
            "print_stats&heater_bed&extruder&toolhead"
        )

        if result.status != CommandResultStatus.SUCCESS or not result.data:
            return PrintProgress(state=NormalizedPrinterState.UNKNOWN)

        data = result.data

        # Parse state
        print_stats = data.get("print_stats", {})
        state_str = print_stats.get("state", "unknown").lower()
        state = self.STATE_MAP.get(state_str, NormalizedPrinterState.UNKNOWN)

        # Parse progress
        progress = print_stats.get("progress")
        print_time = print_stats.get("print_duration")

        # Parse temperatures
        heater_bed = data.get("heater_bed", {})
        extruder = data.get("extruder", {})

        # Parse layer info from filename parsing (if available)
        filename = print_stats.get("filename", "")
        current_layer = None
        total_layers = None

        # Try to extract layer from metadata if available
        metadata = print_stats.get("info", {})
        if metadata:
            current_layer = metadata.get("current_layer")
            total_layers = metadata.get("total_layer")

        return PrintProgress(
            state=state,
            filename=filename,
            progress=progress,
            print_time=print_time,
            current_layer=current_layer,
            total_layers=total_layers,
            nozzle_temp=extruder.get("temperature"),
            nozzle_target=extruder.get("target"),
            bed_temp=heater_bed.get("temperature"),
            bed_target=heater_bed.get("target"),
        )

    async def is_connected(self) -> bool:
        return self._connected and self._session is not None

    # --- Internal ---

    async def _rest_request(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> CommandResult:
        """Make REST request to Moonraker"""
        if not self._session:
            return CommandResult(
                CommandResultStatus.FAILURE, message="Not connected"
            )

        url = f"{self._rest_base_url}{path}"

        try:
            async with self._session.request(method, url, **kwargs) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return CommandResult(
                        CommandResultStatus.SUCCESS,
                        data=data.get("result"),
                    )
                else:
                    error_text = await resp.text()
                    return CommandResult(
                        CommandResultStatus.FAILURE,
                        message=f"HTTP {resp.status}: {error_text}",
                    )
        except Exception as e:
            return CommandResult(
                CommandResultStatus.FAILURE,
                message=f"Request failed: {str(e)}",
            )
