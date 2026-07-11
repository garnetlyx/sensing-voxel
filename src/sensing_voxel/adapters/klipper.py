"""
Klipper/Moonraker adapter

Adapted from OpenClaw's skills/lucakaufmann/snapmaker/scripts/snapmaker.py
"""

import asyncio
import logging
import math
import random
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import aiohttp


def _normalize_filename(filename: str) -> str:
    """Replace spaces and special characters with dashes to avoid Moonraker path issues."""
    import os
    import re
    name, ext = os.path.splitext(filename)
    name = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-")
    return f"{name}{ext}"

from sensing_voxel.adapters.base import PrinterAdapter
from sensing_voxel.core.models import (
    AdapterCommandError,
    AdapterConnectionError,
    CommandResult,
    CommandResultStatus,
    NormalizedPrinterState,
    PrintProgress,
    TelemetryData,
)
from sensing_voxel.infra.gcode import _strip_gcode_comment, validate_gcode_command
from sensing_voxel.infra.retry import BackoffPolicy

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
        "ready": NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT,  # Moonraker ready state
        "startup": NormalizedPrinterState.UNKNOWN,  # During startup
        "shutdown": NormalizedPrinterState.UNKNOWN,  # During shutdown
        "disconnecting": NormalizedPrinterState.UNKNOWN,  # Disconnecting
    }

    def __init__(
        self,
        host: str = "localhost",
        port: int = 7125,
        api_key: str | None = None,
        compatibility_mode: bool = False,
        base_url: str | None = None,
    ):
        super().__init__()
        self.host = host
        self.port = port
        self.api_key = api_key
        self.compatibility_mode = compatibility_mode

        if base_url:
            self._rest_base_url = base_url.rstrip("/")
            ws_base = self._rest_base_url.replace("https://", "wss://").replace("http://", "ws://")
            self._ws_url = f"{ws_base}/websocket"
        else:
            self._rest_base_url = f"http://{host}:{port}"
            self._ws_url = f"ws://{host}:{port}/websocket"

        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
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

        # Accumulated status from incremental WebSocket updates
        self._cached_status: dict[str, dict] = {}

    async def connect(self) -> None:
        """Connect to Moonraker"""
        if self._connected:
            return

        # Create REST session
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        self._session = aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(total=30))

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
        # Maximum reconnection attempts before giving up
        max_reconnect_attempts = 100

        while not self._shutdown_event.is_set():
            try:
                await self._ws_connect_and_stream()
                # Stream ended normally
            except (AdapterConnectionError, aiohttp.ClientError) as e:
                logger.warning(f"WebSocket connection error: {e}")
            except Exception as e:
                logger.error(f"WebSocket error: {e}", exc_info=True)

            # Give up if we've exceeded maximum attempts
            if self._reconnect_attempts >= max_reconnect_attempts:
                logger.error(
                    f"WebSocket reconnection failed after {self._reconnect_attempts} attempts. "
                    "Giving up."
                )
                # Emit connection lost event
                await self._emit_connection_lost()
                break

            # Reconnect delay - cap attempts to prevent overflow
            delay = self._calculate_backoff_delay()

            logger.info(
                f"Reconnecting in {delay:.1f}s (attempt {self._reconnect_attempts + 1}/{max_reconnect_attempts})"
            )
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=delay)
                break
            except TimeoutError:
                self._reconnect_attempts += 1

    def _calculate_backoff_delay(self) -> float:
        """Calculate reconnect delay with capped exponential backoff."""
        max_attempts_before_max_delay = (
            int(
                math.log(self._reconnect_policy.max_delay / self._reconnect_policy.base_delay)
                / math.log(self._reconnect_policy.factor)
            )
            + 1
        )
        capped_attempts = min(self._reconnect_attempts, max_attempts_before_max_delay)

        delay = min(
            self._reconnect_policy.base_delay
            * (self._reconnect_policy.factor**capped_attempts),
            self._reconnect_policy.max_delay,
        )

        # Add random jitter to prevent thundering herd.
        if self._reconnect_policy.jitter > 0:
            delay += delay * self._reconnect_policy.jitter * random.random()

        return delay

    async def _ws_connect_and_stream(self):
        """Connect WebSocket and stream events"""
        if not self._session:
            raise AdapterConnectionError("REST session not initialized")

        async with self._session.ws_connect(self._ws_url) as ws:
            self._ws = ws
            self._reconnect_attempts = 0  # Reset on successful connection
            self._cached_status = {}  # Reset accumulated status on reconnect

            # Subscribe to printer objects for live status updates
            await self._ws_subscribe(ws)

            # Check printer state on reconnection to detect changes during disconnection
            await self._check_state_after_reconnect()

            await self._emit_connection_restored()

            logger.info("WebSocket connected and subscribed")

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        await self._handle_ws_message(msg.json())
                    except Exception as e:
                        logger.error(f"Error handling WebSocket message: {e}", exc_info=True)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    raise AdapterConnectionError(f"WebSocket error: {ws.exception()}")

    async def _ws_subscribe(self, ws: aiohttp.ClientWebSocketResponse):
        """Subscribe to Moonraker printer objects for live updates."""
        await ws.send_json(
            {
                "jsonrpc": "2.0",
                "method": "printer.objects.subscribe",
                "params": {
                    "objects": {
                        "print_stats": None,
                        "heater_bed": None,
                        "extruder": None,
                        "toolhead": None,
                        "display_status": None,
                    }
                },
                "id": 1,
            }
        )
        logger.debug("Subscribed to printer objects")

    async def _handle_ws_message(self, data: dict):
        """Handle WebSocket message"""
        # Validate message structure
        if not isinstance(data, dict):
            logger.warning(f"Invalid WebSocket message type: {type(data)}")
            return

        method = data.get("method")
        if method is None:
            # JSON-RPC response — parse subscription initial state
            result = data.get("result", {})
            if isinstance(result, dict):
                status = result.get("status", {})
                if status:
                    for key, value in status.items():
                        if isinstance(value, dict):
                            self._cached_status[key] = value
                    # Emit initial telemetry from subscription snapshot
                    print_stats = status.get("print_stats", {})
                    telemetry = self._parse_telemetry(self._cached_status, print_stats)
                    if telemetry:
                        await self._emit_telemetry_update(telemetry)
            return

        params = data.get("params")
        if params is None:
            params = [{}]
        elif not isinstance(params, list) or len(params) == 0:
            logger.warning(f"Invalid 'params' field in WebSocket message: {params}")
            params = [{}]

        # Extract first params dict
        try:
            params_dict = params[0] if len(params) > 0 else {}
        except (TypeError, IndexError):
            params_dict = {}

        if method == "notify_status_update":
            await self._handle_status_update(params_dict)
        elif method == "notify_gcode_output":
            self._handle_gcode_output(params_dict)

    async def _handle_status_update(self, status: dict):
        """Handle incremental status update from printer.

        Moonraker sends partial updates — each message only contains changed
        fields.  We merge into _cached_status so that _parse_telemetry always
        sees the full picture.
        """
        # Merge incremental update into accumulated cache
        for key, value in status.items():
            if isinstance(value, dict) and key in self._cached_status:
                self._cached_status[key].update(value)
            else:
                self._cached_status[key] = value

        print_stats = self._cached_status.get("print_stats", {})
        state_str = print_stats.get("state", "unknown").lower()

        normalized_state = self.STATE_MAP.get(state_str, NormalizedPrinterState.UNKNOWN)

        # Emit state changed event if state changed
        if hasattr(self, "_last_state") and self._last_state != normalized_state:
            await self._emit_state_changed(self._last_state, normalized_state)
            logger.debug(f"State changed: {self._last_state.name} -> {normalized_state.name}")
        elif not hasattr(self, "_last_state"):
            logger.debug(f"Initial state: {normalized_state.name}")

        self._last_state = normalized_state

        # Parse telemetry from the full accumulated status
        telemetry = self._parse_telemetry(self._cached_status, print_stats)
        if telemetry:
            await self._emit_telemetry_update(telemetry)

    def _handle_gcode_output(self, params: dict):
        """Handle G-code output notification from Klipper"""
        # TODO: parse LAYER_CHANGE markers for monitoring orchestrator (ARCHITECTURE.md §4.8)
        message = params.get("message", "")
        if message:
            logger.debug(f"G-code output: {message}")

    def _parse_telemetry(self, status: dict, print_stats: dict) -> TelemetryData | None:
        """Parse telemetry data from status update"""
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
            metadata["print_time"] = print_stats.get("print_duration", 0.0)
            metadata["state"] = print_stats.get("state", "")

            # Prefer display_status.progress (updated by M73 / virtual_sdcard),
            # fall back to print_stats.progress
            display = status.get("display_status", {})
            metadata["progress"] = display.get("progress") or print_stats.get("progress") or 0.0

        if not temps and not position:
            return None

        return TelemetryData(
            timestamp=datetime.now(),
            temperatures=temps,
            position=position,
            metadata=metadata,
        )

    async def _check_state_after_reconnect(self):
        """Query full printer state on (re)connect and seed the telemetry cache."""
        try:
            result = await self._rest_request(
                "GET",
                "/printer/objects/query?print_stats&heater_bed&extruder&toolhead&display_status",
            )

            if result.status != CommandResultStatus.SUCCESS or not result.data:
                logger.warning("Failed to query printer state after reconnect")
                return

            data = result.data

            # Moonraker REST wraps objects under {"status": {...}, "eventtime": ...}
            status = data.get("status", data)

            # Seed the accumulated cache with full current state
            for key, value in status.items():
                if isinstance(value, dict):
                    self._cached_status[key] = value

            print_stats = status.get("print_stats", {})
            state_str = print_stats.get("state", "unknown").lower()
            filename = print_stats.get("filename", "")
            total_duration = print_stats.get("total_duration", 0)

            normalized_state = self.STATE_MAP.get(state_str, NormalizedPrinterState.UNKNOWN)

            # Check if we have a previous state to compare against
            if hasattr(self, "_last_state"):
                if self._last_state != normalized_state:
                    logger.info(
                        f"Detected state change during disconnection: "
                        f"{self._last_state.name} -> {normalized_state.name}"
                    )
                    await self._emit_state_changed(self._last_state, normalized_state)

                    if (
                        self._last_state
                        in [
                            NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT,
                            NormalizedPrinterState.UNKNOWN,
                        ]
                        and normalized_state == NormalizedPrinterState.PRINTING
                    ):
                        logger.warning(
                            f"Print started during disconnection: {filename} "
                            f"(duration: {total_duration:.1f}s)"
                        )

                    if self._last_state == NormalizedPrinterState.PRINTING and normalized_state in [
                        NormalizedPrinterState.TERMINAL,
                        NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT,
                    ]:
                        logger.warning(
                            f"Print completed during disconnection: {filename} "
                            f"(total duration: {total_duration:.1f}s)"
                        )

            self._last_state = normalized_state

            # Emit initial telemetry so the context collector has data immediately
            telemetry = self._parse_telemetry(self._cached_status, print_stats)
            if telemetry:
                await self._emit_telemetry_update(telemetry)

        except Exception as e:
            logger.error(f"Error checking state after reconnect: {e}", exc_info=True)

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
        """Emergency cancel with dedicated session and aggressive timeout"""
        # Create dedicated session with aggressive timeout for immediate response
        # This bypasses the normal queue to ensure emergency commands execute ASAP
        try:
            timeout = aiohttp.ClientTimeout(total=5)  # 5s aggressive timeout
            async with aiohttp.ClientSession(timeout=timeout) as emergency_session:
                headers = {}
                if self.api_key:
                    headers["X-API-Key"] = self.api_key

                url = f"{self._rest_base_url}/printer/print/cancel"

                async with emergency_session.post(url, headers=headers) as resp:
                    if resp.status == 200:
                        logger.warning(
                            f"EMERGENCY CANCEL executed for printer at {self.host}:{self.port}"
                        )
                        return CommandResult(CommandResultStatus.SUCCESS)
                    else:
                        error_text = await resp.text()
                        logger.error(f"EMERGENCY CANCEL failed: HTTP {resp.status}")
                        return CommandResult(
                            CommandResultStatus.FAILURE,
                            message=f"HTTP {resp.status}: {error_text}",
                        )
        except Exception as e:
            logger.error(f"EMERGENCY CANCEL exception: {e}")
            return CommandResult(
                CommandResultStatus.FAILURE,
                message=f"Emergency cancel failed: {str(e)}",
            )

    async def upload_file(self, local_path: str, remote_name: str) -> CommandResult:
        """Upload file to printer via Moonraker"""
        try:
            path = Path(local_path)
            if not path.exists():
                return CommandResult(
                    CommandResultStatus.FAILURE,
                    message=f"File not found: {local_path}",
                )

            remote_name = _normalize_filename(remote_name)

            with open(path, "rb") as f:
                file_content = f.read()

            data = aiohttp.FormData()
            data.add_field(
                "file",
                file_content,
                filename=remote_name,
                content_type="application/octet-stream",
            )

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
        result = await self._rest_request("GET", "/printer/objects/query?gcode_move")

        if result.status == CommandResultStatus.SUCCESS and result.data:
            status = result.data.get("status", result.data)
            gcode_move = status.get("gcode_move", {})
            return gcode_move.get("homing_origin", [0, 0, 0])[2]  # Z component

        raise AdapterCommandError("Failed to query Z offset")

    async def set_z_offset(self, offset_mm: float) -> CommandResult:
        """Set Z offset via SET_GCODE_OFFSET command"""
        # Validate offset is numeric to prevent G-code injection
        try:
            offset_value = float(offset_mm)
        except (ValueError, TypeError) as e:
            return CommandResult(
                CommandResultStatus.FAILURE, message=f"Invalid offset value: {str(e)}"
            )

        # Ensure offset is within reasonable bounds
        if offset_value < -10.0 or offset_value > 10.0:
            return CommandResult(
                CommandResultStatus.FAILURE,
                message=f"Offset out of range: {offset_value}. Must be between -10mm and +10mm",
            )

        # Format as G-code command with only numeric value
        gcode = f"SET_GCODE_OFFSET Z_ADJUST={offset_value:.4f} MOVE=1"
        return await self._rest_request(
            "POST",
            "/printer/gcode/script",
            json={"script": gcode},
        )

    async def set_nozzle_temp(self, temperature: float) -> CommandResult:
        temp = int(temperature)
        return await self._rest_request(
            "POST", "/printer/gcode/script", json={"script": f"M104 S{temp}"}
        )

    async def set_bed_temp(self, temperature: float) -> CommandResult:
        temp = int(temperature)
        return await self._rest_request(
            "POST", "/printer/gcode/script", json={"script": f"M140 S{temp}"}
        )

    async def home_axes(self, axes: str = "") -> CommandResult:
        axes_arg = f" {axes.upper()}" if axes.strip() else ""
        return await self._rest_request(
            "POST", "/printer/gcode/script", json={"script": f"G28{axes_arg}"}
        )

    async def set_fan_speed(self, speed: int) -> CommandResult:
        if speed <= 0:
            gcode = "M107"
        else:
            gcode = f"M106 S{min(speed, 255)}"
        return await self._rest_request("POST", "/printer/gcode/script", json={"script": gcode})

    async def run_gcode(self, command: str) -> CommandResult:
        validation_error = validate_gcode_command(command)
        if validation_error:
            return CommandResult(CommandResultStatus.FAILURE, message=validation_error)

        safe_command = _strip_gcode_comment(command)
        upper_command = safe_command.upper()
        if upper_command.startswith("M114") or upper_command.startswith("GET_POSITION"):
            position = await self.get_current_position()
            if position is None:
                return CommandResult(
                    CommandResultStatus.FAILURE,
                    message="Current position is unavailable",
                )
            return CommandResult(
                CommandResultStatus.SUCCESS,
                message=(
                    f"X:{position['x']:.3f} Y:{position['y']:.3f} Z:{position['z']:.3f}"
                ),
                data={"response_type": "position", "position": position},
            )
        if upper_command.startswith("GCODE_LIST") or upper_command.startswith("HELP"):
            return await self._get_gcode_help()
        if upper_command.startswith("QUERY_ENDSTOPS"):
            return await self._query_endstops()

        return await self._rest_request(
            "POST", "/printer/gcode/script", json={"script": safe_command}
        )

    async def _get_gcode_help(self) -> CommandResult:
        result = await self._rest_request("GET", "/printer/gcode/help")
        if result.status != CommandResultStatus.SUCCESS or not result.data:
            return result
        commands = result.data.get("commands") if isinstance(result.data, dict) else None
        if commands is None:
            commands = result.data
        if isinstance(commands, dict):
            command_names = sorted(commands.keys())
        elif isinstance(commands, list):
            command_names = sorted(str(item) for item in commands)
        else:
            command_names = []
        return CommandResult(
            CommandResultStatus.SUCCESS,
            message=f"Found {len(command_names)} available G-code commands",
            data={"response_type": "gcode_list", "commands": command_names},
        )

    async def _query_endstops(self) -> CommandResult:
        result = await self._rest_request("GET", "/printer/objects/query?query_endstops")
        if result.status != CommandResultStatus.SUCCESS or not result.data:
            return result
        status = result.data.get("status", result.data)
        query = status.get("query_endstops", {})
        last_query = query.get("last_query")
        return CommandResult(
            CommandResultStatus.SUCCESS,
            message="Queried endstop status",
            data={"response_type": "endstops", "endstops": last_query or query},
        )

    async def list_files(self) -> CommandResult:
        return await self._rest_request("GET", "/server/files/list?root=gcodes")

    async def list_config_files(self) -> CommandResult:
        """List Klipper/Moonraker configuration files via Moonraker's config root."""
        return await self._rest_request("GET", "/server/files/list?root=config")

    async def get_config_file(
        self, filename: str = "printer.cfg", max_bytes: int = 50000
    ) -> CommandResult:
        """Read a Klipper configuration file from Moonraker's config root.

        This reads the remote printer's config files, not files from the
        Sensing Voxel container.  It is intentionally read-only.
        """
        if not self._session:
            return CommandResult(CommandResultStatus.FAILURE, message="Not connected")

        safe_filename = filename.strip().lstrip("/") or "printer.cfg"
        if ".." in Path(safe_filename).parts:
            return CommandResult(
                CommandResultStatus.FAILURE,
                message="filename must stay under Moonraker config root",
            )

        limit = max(1, min(int(max_bytes), 200000))
        encoded = quote(safe_filename, safe="/")
        url = f"{self._rest_base_url}/server/files/config/{encoded}"

        try:
            async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    return CommandResult(
                        CommandResultStatus.FAILURE,
                        message=f"HTTP {resp.status}: {error_text}",
                    )

                raw = await resp.read()
                truncated = len(raw) > limit
                content = raw[:limit].decode(errors="replace")
                return CommandResult(
                    CommandResultStatus.SUCCESS,
                    data={
                        "filename": safe_filename,
                        "content": content,
                        "size_bytes": len(raw),
                        "truncated": truncated,
                    },
                )
        except TimeoutError:
            return CommandResult(
                CommandResultStatus.FAILURE, message="Request timeout after 10s"
            )
        except Exception as e:
            return CommandResult(
                CommandResultStatus.FAILURE,
                message=f"Request failed: {str(e)}",
            )

    async def write_config_file(self, filename: str, content: str) -> CommandResult:
        """Write a config file to Moonraker's config root. Auto-creates a backup first."""
        safe_filename = filename.strip().lstrip("/") or "printer.cfg"
        if ".." in Path(safe_filename).parts:
            return CommandResult(
                CommandResultStatus.FAILURE,
                message="filename must stay under Moonraker config root",
            )

        backup_name = f"sv-backup-{datetime.now().strftime('%Y%m%d_%H%M%S')}-{safe_filename}"
        backup_result = await self.get_config_file(safe_filename, max_bytes=200000)
        if backup_result.status == CommandResultStatus.SUCCESS:
            backup_content = backup_result.data.get("content", "")
            try:
                backup_data = aiohttp.FormData()
                backup_data.add_field(
                    "file",
                    backup_content.encode(),
                    filename=backup_name,
                    content_type="text/plain",
                )
                backup_data.add_field("root", "config")
                await self._rest_request("POST", "/server/files/upload", data=backup_data)
                logger.info(f"Created config backup: {backup_name}")
            except Exception as e:
                logger.warning(f"Backup upload failed (proceeding): {e}")

        data = aiohttp.FormData()
        data.add_field(
            "file",
            content.encode(),
            filename=safe_filename,
            content_type="text/plain",
        )
        data.add_field("root", "config")
        result = await self._rest_request("POST", "/server/files/upload", data=data)
        if result.status == CommandResultStatus.SUCCESS:
            logger.info(f"Wrote config file: {safe_filename}")
        return result

    async def delete_file(self, filename: str) -> CommandResult:
        return await self._rest_request("DELETE", f"/server/files/gcodes/{filename}")

    async def get_file_metadata(self, filename: str) -> CommandResult:
        """Get file metadata — parse G-code header if Moonraker metadata is sparse."""
        # First try Moonraker's metadata endpoint
        result = await self._rest_request("GET", f"/server/files/metadata?filename={filename}")
        if result.status == CommandResultStatus.SUCCESS and result.data:
            meta = result.data
            # If Moonraker already has layer_count, return as-is
            if meta.get("layer_count"):
                return result

        # Moonraker metadata is sparse — download G-code header and parse
        return await self._parse_gcode_header(filename)

    async def _parse_gcode_header(self, filename: str) -> CommandResult:
        """Download first 10KB of G-code file and parse slicer comments."""
        import re

        if not self._session:
            return CommandResult(CommandResultStatus.FAILURE, message="Not connected")

        url = f"{self._rest_base_url}/server/files/gcodes/{filename}"
        try:
            headers = {"Range": "bytes=0-10240"}
            async with self._session.get(
                url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status not in (200, 206):
                    return CommandResult(
                        CommandResultStatus.FAILURE,
                        message=f"Failed to download G-code header: HTTP {resp.status}",
                    )
                header_text = await resp.text(errors="replace")
        except Exception as e:
            return CommandResult(
                CommandResultStatus.FAILURE,
                message=f"Failed to download G-code header: {e}",
            )

        meta: dict = {"filename": filename}

        # OrcaSlicer / PrusaSlicer / Cura header patterns
        patterns = {
            "total_layers": [
                r";\s*total layer(?:s| number)\s*[:=]\s*(\d+)",
                r";\s*LAYER_COUNT\s*[:=]\s*(\d+)",  # Cura
            ],
            "layer_height": [
                r";\s*layer_height\s*[:=]\s*([\d.]+)",
            ],
            "first_layer_height": [
                r";\s*first_layer_height\s*[:=]\s*([\d.]+)",
            ],
            "object_height": [
                r";\s*max_z_height\s*[:=]\s*([\d.]+)",
                r";\s*object_height\s*[:=]\s*([\d.]+)",
            ],
            "filament_type": [
                r";\s*filament_type\s*[:=]\s*(\S+)",
            ],
            "estimated_time": [
                r";\s*estimated_time\s*[:=]\s*(\d+)",
                r";\s*TIME\s*[:=]\s*(\d+)",  # Cura
            ],
            "slicer": [
                r";\s*generated by\s+(.+?)(?:\s+on\s|\n)",
            ],
        }

        for key, regexes in patterns.items():
            for regex in regexes:
                m = re.search(regex, header_text, re.IGNORECASE)
                if m:
                    val = m.group(1).strip()
                    # Convert numeric values
                    if key in ("total_layers", "estimated_time"):
                        meta[key] = int(val)
                    elif key in ("layer_height", "first_layer_height", "object_height"):
                        meta[key] = float(val)
                    else:
                        meta[key] = val
                    break

        return CommandResult(CommandResultStatus.SUCCESS, data=meta)

    async def move_axis(self, axis: str, distance: float, speed: int = 3000) -> CommandResult:
        axis = axis.upper()
        if axis not in ("X", "Y", "Z"):
            return CommandResult(CommandResultStatus.FAILURE, message=f"Invalid axis: {axis}")
        gcode = f"G91\nG1 {axis}{distance:.3f} F{speed}\nG90"
        return await self._rest_request("POST", "/printer/gcode/script", json={"script": gcode})

    async def extrude(self, length: float, speed: int = 300) -> CommandResult:
        gcode = f"M83\nG1 E{length:.2f} F{speed}\nM82"
        return await self._rest_request("POST", "/printer/gcode/script", json={"script": gcode})

    async def set_speed_override(self, percent: int) -> CommandResult:
        return await self._rest_request(
            "POST",
            "/printer/gcode/script",
            json={"script": f"M220 S{percent}"},
        )

    async def set_flow_override(self, percent: int) -> CommandResult:
        return await self._rest_request(
            "POST",
            "/printer/gcode/script",
            json={"script": f"M221 S{percent}"},
        )

    async def list_macros(self) -> CommandResult:
        result = await self._rest_request("GET", "/printer/objects/list")
        if result.status != CommandResultStatus.SUCCESS or not result.data:
            return result
        objects = result.data.get("objects", [])
        macros = [
            obj.replace("gcode_macro ", "") for obj in objects if obj.startswith("gcode_macro ")
        ]
        return CommandResult(CommandResultStatus.SUCCESS, data=macros)

    async def execute_macro(self, name: str) -> CommandResult:
        return await self._rest_request("POST", "/printer/gcode/script", json={"script": name})

    async def restart_firmware(self) -> CommandResult:
        return await self._rest_request("POST", "/printer/firmware_restart")

    async def restart_service(self) -> CommandResult:
        return await self._rest_request("POST", "/printer/restart")

    async def get_system_info(self) -> CommandResult:
        result = await self._rest_request("GET", "/printer/objects/query?system_stats&webhooks")
        if result.status != CommandResultStatus.SUCCESS or not result.data:
            return result
        status = result.data.get("status", result.data)
        sys_stats = status.get("system_stats", {})
        webhooks = status.get("webhooks", {})
        return CommandResult(
            CommandResultStatus.SUCCESS,
            data={
                "cpu_load": sys_stats.get("sysload"),
                "memory_available_kb": sys_stats.get("memavail"),
                "klipper_state": webhooks.get("state"),
                "klipper_message": webhooks.get("state_message"),
            },
        )

    async def get_hardware_info(self) -> CommandResult:
        """Fetch hardware fingerprint (MAC address, hostname, CPU info) from Moonraker."""
        result = await self._rest_request("GET", "/machine/system_info")
        if result.status != CommandResultStatus.SUCCESS or not result.data:
            return result

        data = result.data.get("result", result.data)
        system_info = data.get("system_info", {})
        network = system_info.get("network", {})

        mac_address = None
        ip_address = None
        interface = None
        for iface_name, iface_info in network.items():
            mac = iface_info.get("mac_address")
            if mac and mac != "00:00:00:00:00:00":
                mac_address = mac
                interface = iface_name
                ips = iface_info.get("ip_addresses", [])
                for ip_entry in ips:
                    if ip_entry.get("family") == "ipv4" and not ip_entry.get("is_link_local"):
                        ip_address = ip_entry.get("address")
                        break
                if ip_address:
                    break

        cpu_info = system_info.get("cpu_info", {})
        return CommandResult(
            CommandResultStatus.SUCCESS,
            data={
                "mac_address": mac_address,
                "ip_address": ip_address,
                "network_interface": interface,
                "hostname": system_info.get("hostname"),
                "cpu_count": cpu_info.get("cpu_count"),
                "processor": cpu_info.get("processor"),
                "total_memory_kb": cpu_info.get("total_memory"),
            },
        )

    async def motors_off(self) -> CommandResult:
        return await self._rest_request("POST", "/printer/gcode/script", json={"script": "M84"})

    # --- Queries ---

    async def get_printer_state(self) -> NormalizedPrinterState:
        result = await self._rest_request("GET", "/printer/objects/query?print_stats")
        if result.status == CommandResultStatus.SUCCESS and result.data:
            status = result.data.get("status", result.data)
            state_str = status.get("print_stats", {}).get("state", "unknown")
            return self.STATE_MAP.get(state_str.lower(), NormalizedPrinterState.UNKNOWN)
        return NormalizedPrinterState.UNKNOWN

    async def get_print_progress(self) -> PrintProgress:
        """Get current print progress and state"""
        # Query comprehensive printer status
        result = await self._rest_request(
            "GET", "/printer/objects/query?print_stats&heater_bed&extruder&toolhead&display_status"
        )

        if result.status != CommandResultStatus.SUCCESS or not result.data:
            return PrintProgress(state=NormalizedPrinterState.UNKNOWN)

        data = result.data
        status = data.get("status", data)

        # Parse state
        print_stats = status.get("print_stats", {})
        state_str = print_stats.get("state", "unknown").lower()
        state = self.STATE_MAP.get(state_str, NormalizedPrinterState.UNKNOWN)

        # Parse progress
        progress = print_stats.get("progress")
        print_time = print_stats.get("print_duration")

        # Parse temperatures
        heater_bed = status.get("heater_bed", {})
        extruder = status.get("extruder", {})

        # Parse layer info from Klipper stats or metadata fallback
        filename = print_stats.get("filename", "")
        current_layer = None
        total_layers = None

        # Try to extract layer from print_stats.info (requires SET_PRINT_STATS_INFO in gcode)
        info = print_stats.get("info") or {}
        if info:
            current_layer = info.get("current_layer")
            total_layers = info.get("total_layer")

        toolhead = status.get("toolhead", {})
        position = toolhead.get("position", [0.0, 0.0, 0.0]) if toolhead else [0.0, 0.0, 0.0]
        current_z = position[2] if len(position) > 2 else 0.0

        # If Klipper doesn't have layer info, estimate from file metadata and current Z/progress
        if total_layers is None and filename and progress is not None:
            file_meta = await self.get_file_metadata(filename)
            if file_meta.status == CommandResultStatus.SUCCESS and file_meta.data:
                total_layers = file_meta.data.get("total_layers")
                layer_height = file_meta.data.get("layer_height")
                first_layer_height = file_meta.data.get("first_layer_height")
                if total_layers and current_layer is None:
                    if layer_height and current_z > 0:
                        base_height = first_layer_height or layer_height
                        if current_z <= base_height:
                            current_layer = 1
                        else:
                            current_layer = 1 + max(
                                0,
                                int(round((current_z - base_height) / layer_height)),
                            )
                            current_layer = min(current_layer, total_layers)
                    else:
                        current_layer = max(1, int(progress * total_layers))

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

    async def get_current_position(self) -> dict[str, float] | None:
        """Get current position from Klipper toolhead state."""
        result = await self._rest_request("GET", "/printer/objects/query?toolhead")
        if result.status != CommandResultStatus.SUCCESS or not result.data:
            return None
        status = result.data.get("status", result.data)
        toolhead = status.get("toolhead", {})
        position = toolhead.get("position")
        if not position or len(position) < 3:
            return None
        return {
            "x": float(position[0]),
            "y": float(position[1]),
            "z": float(position[2]),
        }

    async def is_connected(self) -> bool:
        return self._connected and self._session is not None

    # --- Internal ---

    async def _rest_request(
        self,
        method: str,
        path: str,
        timeout: int | None = 30,
        **kwargs,
    ) -> CommandResult:
        """Make REST request to Moonraker"""
        if not self._session:
            return CommandResult(CommandResultStatus.FAILURE, message="Not connected")

        url = f"{self._rest_base_url}{path}"

        # Enforce timeout to prevent indefinite hangs (default 30s)
        effective_timeout = timeout if timeout else 30
        client_timeout = aiohttp.ClientTimeout(total=effective_timeout)

        try:
            async with self._session.request(method, url, timeout=client_timeout, **kwargs) as resp:
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
        except TimeoutError:
            return CommandResult(
                CommandResultStatus.FAILURE, message=f"Request timeout after {timeout}s"
            )
        except Exception as e:
            return CommandResult(
                CommandResultStatus.FAILURE,
                message=f"Request failed: {str(e)}",
            )
