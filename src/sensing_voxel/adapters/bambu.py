"""
Bambu Lab printer adapter with MQTT + FTP/FTPS

Full-featured adapter for Bambu printers with:
- Local MQTT for status + control
- FTP/FTPS for file upload/management
- Complete print lifecycle control
- Emergency stop via high-priority MQTT channel
- Z-offset control via MQTT
- Rich telemetry parsing (temps, layers, filament, etc.)
- Full file management (list, delete, metadata)
- Robust reconnection with exponential backoff

Based on Bambu MQTT protocol documentation and community reverse engineering.
"""

import asyncio
import ftplib
import json
import logging
import math
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from paho.mqtt.client import Client as MqttClient, MQTT_ERR_SUCCESS

from sensing_voxel.adapters.base import (
    NotSupportedError,
    PrinterAdapter,
)
from sensing_voxel.core.models import (
    AdapterCommandError,
    AdapterConnectionError,
    CommandResult,
    CommandResultStatus,
    NormalizedPrinterState,
    PrintProgress,
    TelemetryData,
)
from sensing_voxel.infra.retry import BackoffPolicy

logger = logging.getLogger(__name__)


@dataclass
class BambuPrintStats:
    """Internal print statistics tracking"""

    filename: str = ""
    progress: float = 0.0
    print_time: int = 0
    total_time: int = 0
    remaining_time: int = 0
    current_layer: int = 0
    total_layers: int = 0
    state: NormalizedPrinterState = NormalizedPrinterState.UNKNOWN


@dataclass
class BambuFilamentInfo:
    """Filament usage information"""

    tray_type: str = ""  # e.g., "PLA", "PETG"
    tray_color: str = ""  # e.g., "FF0000"
    nozzle_temp_min: int = 0
    nozzle_temp_max: int = 0
    tray_weight: float = 0.0  # grams remaining


class BambuAdapter(PrinterAdapter):
    """Bambu Lab printer adapter with full feature parity"""

    # MQTT topic patterns for Bambu printers
    STATUS_TOPIC = "device/{serial}/report"
    CONTROL_TOPIC = "device/{serial}/request"
    P2P_TOPIC = "p2p/report"

    # State mapping from Bambu MQTT status codes
    # Bambu uses numeric status codes in MQTT messages
    STATE_MAP = {
        # From Bambu MQTT protocol
        0: NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT,  # Idle
        1: NormalizedPrinterState.PRINTING,  # Printing
        2: NormalizedPrinterState.PAUSED,  # Pause
        3: NormalizedPrinterState.PRINTING,  # Preparing (treated as printing)
        4: NormalizedPrinterState.PRINTING,  # Running (M108 resume)
        5: NormalizedPrinterState.TERMINAL,  # Complete
        6: NormalizedPrinterState.TERMINAL,  # Failed
        7: NormalizedPrinterState.ERROR,  # Error
        8: NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT,  # Offline
        9: NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT,  # Upgrade (maintenance)
        10: NormalizedPrinterState.PRINTING,  # Filament change (treated as paused/printing)
        11: NormalizedPrinterState.PAUSED,  # Pausing
        12: NormalizedPrinterState.PRINTING,  # Resuming
        13: NormalizedPrinterState.TERMINAL,  # Finishing
        14: NormalizedPrinterState.PAUSED,  # Paused for filament runout
        15: NormalizedPrinterState.ERROR,  # Heating error
        16: NormalizedPrinterState.PRINTING,  # Calibrating
        17: NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT,  # Scanning bed
        18: NormalizedPrinterState.PRINTING,  # First layer inspection
    }

    def __init__(
        self,
        host: str = "localhost",
        mqtt_port: int = 8883,  # Bambu default is TLS on 8883
        ftp_port: int = 990,  # Bambu default is FTPS on 990
        serial_number: str = "",
        access_code: str = "",
        use_tls: bool = True,
        ftp_timeout: int = 30,
    ):
        super().__init__()
        self.host = host
        self.mqtt_port = mqtt_port
        self.ftp_port = ftp_port
        self.serial_number = serial_number
        self.access_code = access_code
        self.use_tls = use_tls
        self._ftp_timeout = ftp_timeout

        # MQTT client
        self._mqtt_client: MqttClient | None = None
        self._mqtt_connected = False

        # FTP client
        self._ftp_client: ftplib.FTP | ftplib.FTP_TLS | None = None
        self._ftp_connected = False

        self._connected = False
        self._shutdown_event = asyncio.Event()

        # Track last known state
        self._last_state: NormalizedPrinterState | None = None

        # Enhanced print metadata
        self._print_stats = BambuPrintStats()
        self._filament_info: list[BambuFilamentInfo] = []

        # Cached telemetry data
        self._cached_telemetry: dict[str, Any] = {}
        self._last_position: dict[str, float] | None = None

        # Z-offset cache (Bambu reports this in telemetry)
        self._z_offset: float = 0.0
        self._z_offset_calibrated: bool = False

        # System info cache
        self._system_info: dict[str, Any] = {}

        # MQTT reconnection policy
        self._reconnect_policy = BackoffPolicy(
            base_delay=1.0,
            max_delay=30.0,
            factor=1.8,
            jitter=0.25,
        )
        self._reconnect_attempts = 0

        # Command timeout
        self._command_timeout = 10.0

    # --- Connection Lifecycle ---

    async def connect(self) -> None:
        """Connect to Bambu printer via MQTT and FTP"""
        if self._connected:
            return

        # Create MQTT client
        client = MqttClient(
            client_id=f"bambu_adapter_{self.serial_number}",
        )
        self._mqtt_client = client

        if self.access_code:
            self._mqtt_client.username_pw_set("bblp", self.access_code)

        # Configure TLS if needed
        if self.use_tls:
            self._mqtt_client.tls_set()

        # Set callbacks
        self._mqtt_client.on_connect = self._on_mqtt_connect
        self._mqtt_client.on_disconnect = self._on_mqtt_disconnect
        self._mqtt_client.on_message = self._on_mqtt_message

        # Start MQTT reconnection loop
        asyncio.create_task(self._mqtt_reconnect_loop())

        self._connected = True
        logger.info(f"Started MQTT reconnection loop for Bambu printer {self.serial_number}")

    async def disconnect(self) -> None:
        """Disconnect from Bambu printer"""
        if not self._connected:
            return

        self._shutdown_event.set()

        # Disconnect MQTT
        if self._mqtt_client:
            self._mqtt_client.loop_stop()
            self._mqtt_client.disconnect()
            logger.info("MQTT client disconnected")

        # Disconnect FTP
        await self._disconnect_ftp()

        self._connected = False
        logger.info("Disconnected from Bambu printer")

    async def _disconnect_ftp(self) -> None:
        """Disconnect FTP client"""
        if self._ftp_client:
            try:
                self._ftp_client.quit()
                logger.debug("FTP client disconnected")
            except Exception as e:
                logger.warning(f"Error disconnecting FTP: {e}")
            finally:
                self._ftp_client = None
                self._ftp_connected = False

    # --- MQTT Reconnection ---

    async def _mqtt_reconnect_loop(self):
        """MQTT reconnection loop with exponential backoff"""
        max_reconnect_attempts = 100

        while not self._shutdown_event.is_set():
            try:
                await self._mqtt_connect_and_subscribe()
            except (AdapterConnectionError, ConnectionError) as e:
                logger.warning(f"MQTT connection error: {e}")
            except Exception as e:
                logger.error(f"MQTT error: {e}", exc_info=True)

            if self._reconnect_attempts >= max_reconnect_attempts:
                logger.error(
                    f"MQTT reconnection failed after {self._reconnect_attempts} attempts. Giving up."
                )
                await self._emit_connection_lost()
                break

            delay = self._calculate_backoff_delay()
            logger.info(
                f"Reconnecting MQTT in {delay:.1f}s "
                f"(attempt {self._reconnect_attempts + 1}/{max_reconnect_attempts})"
            )
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=delay)
                break
            except TimeoutError:
                self._reconnect_attempts += 1

    def _calculate_backoff_delay(self) -> float:
        """Calculate reconnection delay with exponential backoff and jitter"""
        max_attempts_before_max_delay = (
            int(
                math.log(self._reconnect_policy.max_delay / self._reconnect_policy.base_delay)
                / math.log(self._reconnect_policy.factor)
            )
            + 1
        )
        capped_attempts = min(self._reconnect_attempts, max_attempts_before_max_delay)

        delay = min(
            self._reconnect_policy.base_delay * (self._reconnect_policy.factor**capped_attempts),
            self._reconnect_policy.max_delay,
        )

        if self._reconnect_policy.jitter > 0:
            delay += delay * self._reconnect_policy.jitter * random.random()

        return delay

    async def _mqtt_connect_and_subscribe(self):
        """Connect MQTT client and subscribe to topics"""
        if not self._mqtt_client:
            raise AdapterConnectionError("MQTT client not initialized")

        self._mqtt_client.connect(self.host, self.mqtt_port, keepalive=60)
        self._mqtt_client.loop_start()

        await self._wait_for_mqtt_connection(timeout=10)

        logger.info(f"MQTT connected to {self.host}:{self.mqtt_port}")
        self._reconnect_attempts = 0

        await self._emit_connection_restored()

        while not self._shutdown_event.is_set() and self._mqtt_connected:
            await asyncio.sleep(1)

    async def _wait_for_mqtt_connection(self, timeout: int = 10):
        """Wait for MQTT connection to establish"""
        start_time = asyncio.get_event_loop().time()
        while not self._mqtt_connected:
            if asyncio.get_event_loop().time() - start_time > timeout:
                raise AdapterConnectionError("MQTT connection timeout")
            await asyncio.sleep(0.1)

    # --- MQTT Callbacks ---

    def _on_mqtt_connect(self, client, userdata, flags, reason_code, properties):
        """MQTT on_connect callback"""
        if reason_code == 0:
            self._mqtt_connected = True
            logger.info("MQTT connected successfully")

            if client:
                status_topic = self.STATUS_TOPIC.format(serial=self.serial_number)
                client.subscribe(status_topic, qos=0)
                logger.info(f"Subscribed to {status_topic}")
                client.subscribe(self.P2P_TOPIC, qos=0)
        else:
            logger.error(f"MQTT connection failed with code {reason_code}")

    def _on_mqtt_disconnect(self, client, userdata, flags, reason_code, properties):
        """MQTT on_disconnect callback"""
        self._mqtt_connected = False
        logger.warning(f"MQTT disconnected (code {reason_code})")
        asyncio.create_task(self._emit_connection_lost())

    def _on_mqtt_message(self, client, userdata, msg):
        """MQTT on_message callback"""
        try:
            payload = json.loads(msg.payload.decode())

            if msg.topic.endswith("/report"):
                self._handle_report_message(payload)
            elif "p2p" in msg.topic:
                self._handle_p2p_message(payload)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to decode MQTT message: {e}")
        except Exception as e:
            logger.error(f"Error handling MQTT message: {e}", exc_info=True)

    def _handle_p2p_message(self, payload: dict):
        """Handle P2P (peer-to-peer) messages"""
        # P2P messages may contain additional status info
        # Currently just log for debugging
        logger.debug(f"P2P message received: {payload}")

    def _handle_report_message(self, payload: dict):
        """Handle status report from Bambu printer"""
        try:
            # Update cached telemetry
            self._cached_telemetry.update(payload)

            # Extract print status
            print_status = payload.get("print", {})
            status_code = print_status.get("status", 0)

            # Map to normalized state
            normalized_state = self.STATE_MAP.get(status_code, NormalizedPrinterState.UNKNOWN)

            # Update print stats
            self._update_print_stats(print_status)

            # Emit state changed event if state changed
            if self._last_state is not None and self._last_state != normalized_state:
                asyncio.create_task(self._emit_state_changed(self._last_state, normalized_state))
                logger.debug(f"State changed: {self._last_state.name} -> {normalized_state.name}")
            elif self._last_state is None:
                logger.debug(f"Initial state: {normalized_state.name}")

            self._last_state = normalized_state
            self._print_stats.state = normalized_state

            # Parse telemetry data
            telemetry = self._parse_telemetry(payload)
            if telemetry:
                asyncio.create_task(self._emit_telemetry_update(telemetry))

        except Exception as e:
            logger.error(f"Error handling report message: {e}", exc_info=True)

    def _update_print_stats(self, print_status: dict):
        """Update internal print statistics from MQTT data"""
        self._print_stats.filename = print_status.get("subtask_name", "")
        self._print_stats.progress = print_status.get("mc_percent", 0) / 100.0
        self._print_stats.print_time = print_status.get("mc_print_time", 0)
        self._print_stats.total_time = print_status.get("mc_total_time", 0)
        self._print_stats.remaining_time = print_status.get("mc_remaining_time", 0)
        self._print_stats.current_layer = print_status.get("layer_num", 0)
        self._print_stats.total_layers = print_status.get("total_layer_num", 0)

        # Update Z-offset info
        self._z_offset = print_status.get("z_offset", 0.0)
        self._z_offset_calibrated = print_status.get("bed_leveling_calibrated", False)

        # Update filament info
        self._update_filament_info(print_status)

        # Update system info
        self._system_info.update(
            {
                "bed_temp": print_status.get("bed_temper", 0),
                "nozzle_temp": print_status.get("nozzle_temper", 0),
                "chamber_temp": print_status.get("chamber_temper", 0),
                "aux_fan_speed": print_status.get("big_fan1_speed", 0),
                "chamber_fan_speed": print_status.get("big_fan2_speed", 0),
                "cooling_fan_speed": print_status.get("cooling_fan_speed", 0),
                "fan_speed": print_status.get("fan_speed", 0),
                "speed_level": print_status.get("spd_lvl", 0),  # Speed level 1-4
                "print_speed": print_status.get("spd_mag", 100),  # Speed percentage
            }
        )

    def _update_filament_info(self, print_status: dict):
        """Update filament information from MQTT data"""
        self._filament_info = []

        # Bambu supports up to 4 filament trays (AMS)
        for i in range(4):
            tray_key = f"ams_tray{i}"
            if tray_key in print_status:
                tray_data = print_status[tray_key]
                if tray_data:
                    self._filament_info.append(
                        BambuFilamentInfo(
                            tray_type=tray_data.get("tray_type", ""),
                            tray_color=tray_data.get("tray_color", ""),
                            nozzle_temp_min=tray_data.get("nozzle_temp_min", 0),
                            nozzle_temp_max=tray_data.get("nozzle_temp_max", 0),
                            tray_weight=tray_data.get("tray_weight", 0.0),
                        )
                    )

        # Also check for external spool info
        if "vt_tray" in print_status:
            vt_tray = print_status["vt_tray"]
            if vt_tray:
                self._filament_info.append(
                    BambuFilamentInfo(
                        tray_type=vt_tray.get("tray_type", ""),
                        tray_color=vt_tray.get("tray_color", ""),
                        nozzle_temp_min=vt_tray.get("nozzle_temp_min", 0),
                        nozzle_temp_max=vt_tray.get("nozzle_temp_max", 0),
                        tray_weight=vt_tray.get("tray_weight", 0.0),
                    )
                )

    def _parse_telemetry(self, payload: dict) -> TelemetryData | None:
        """Parse telemetry data from Bambu MQTT payload"""
        try:
            temps = {}
            position = None
            metadata = {}

            # Print status data
            print_data = payload.get("print", {})

            # Temperature data - prefer direct print_status fields
            if print_data:
                temps["bed"] = print_data.get("bed_temper", 0.0)
                temps["bed_target"] = print_data.get("bed_target_temper", 0.0)
                temps["extruder"] = print_data.get("nozzle_temper", 0.0)
                temps["extruder_target"] = print_data.get("nozzle_target_temper", 0.0)
                temps["chamber"] = print_data.get("chamber_temper", 0.0)

            # Fallback to temps block
            temps_data = payload.get("temps", {})
            if temps_data:
                if "bed" not in temps:
                    temps["bed"] = temps_data.get("bed_temp", 0.0)
                    temps["bed_target"] = temps_data.get("bed_temp_type", 0.0)
                if "extruder" not in temps:
                    temps["extruder"] = temps_data.get("nozzle_temp", 0.0)
                    temps["extruder_target"] = temps_data.get("nozzle_temp_type", 0.0)

            # Position data
            pos_data = payload.get("pos", {})
            if pos_data:
                position = {
                    "x": pos_data.get("x", 0.0),
                    "y": pos_data.get("y", 0.0),
                    "z": pos_data.get("z", 0.0),
                }
            elif print_data:
                # Some reports have position in print data
                position = {
                    "x": print_data.get("x", 0.0),
                    "y": print_data.get("y", 0.0),
                    "z": print_data.get("z", 0.0),
                }

            # Print metadata
            if print_data:
                metadata["filename"] = print_data.get("subtask_name", "")
                metadata["progress"] = print_data.get("mc_percent", 0) / 100.0
                metadata["print_time"] = print_data.get("mc_print_time", 0)
                metadata["remaining_time"] = print_data.get("mc_remaining_time", 0)
                metadata["total_time"] = print_data.get("mc_total_time", 0)
                metadata["status"] = print_data.get("status", 0)
                metadata["current_layer"] = print_data.get("layer_num", 0)
                metadata["total_layers"] = print_data.get("total_layer_num", 0)
                metadata["z_offset"] = print_data.get("z_offset", 0.0)
                metadata["bed_leveling_calibrated"] = print_data.get(
                    "bed_leveling_calibrated", False
                )

                # Fan speeds
                metadata["fan_speed"] = print_data.get("fan_speed", 0)
                metadata["cooling_fan_speed"] = print_data.get("cooling_fan_speed", 0)
                metadata["aux_fan_speed"] = print_data.get("big_fan1_speed", 0)
                metadata["chamber_fan_speed"] = print_data.get("big_fan2_speed", 0)

                # Speed/flow overrides
                metadata["speed_level"] = print_data.get("spd_lvl", 0)
                metadata["print_speed_pct"] = print_data.get("spd_mag", 100)

                # Filament info
                if self._filament_info:
                    metadata["filament_type"] = self._filament_info[0].tray_type
                    metadata["filament_color"] = self._filament_info[0].tray_color

            if not temps and not position:
                return None

            if position:
                self._last_position = {
                    "x": float(position["x"]),
                    "y": float(position["y"]),
                    "z": float(position["z"]),
                }

            return TelemetryData(
                timestamp=datetime.now(),
                temperatures=temps,
                position=position,
                metadata=metadata,
            )
        except Exception as e:
            logger.error(f"Error parsing telemetry: {e}", exc_info=True)
            return None

    # --- FTP Connection Helpers ---

    async def _ensure_ftp_connected(self) -> ftplib.FTP | ftplib.FTP_TLS:
        """Ensure FTP connection is established"""
        if self._ftp_client and self._ftp_connected:
            try:
                # Test connection with NOOP
                self._ftp_client.voidcmd("NOOP")
                return self._ftp_client
            except Exception:
                logger.debug("FTP connection stale, reconnecting")
                await self._disconnect_ftp()

        try:
            if self.use_tls:
                ftp = ftplib.FTP_TLS(timeout=self._ftp_timeout)
                ftp.connect(self.host, self.ftp_port)
                ftp.login(self.serial_number, self.access_code)
                ftp.prot_p()  # Secure data connection
            else:
                ftp = ftplib.FTP(timeout=self._ftp_timeout)
                ftp.connect(self.host, self.ftp_port)
                ftp.login(self.serial_number, self.access_code)

            self._ftp_client = ftp
            self._ftp_connected = True
            logger.debug("FTP connection established")
            return ftp
        except Exception as e:
            raise AdapterConnectionError(f"Failed to connect to FTP: {e}")

    # --- Command Helpers ---

    async def _send_mqtt_command(self, command: dict, qos: int = 1) -> CommandResult:
        """Send MQTT command to printer"""
        if not self._mqtt_client or not self._mqtt_connected:
            return CommandResult(
                CommandResultStatus.FAILURE,
                message="MQTT client not connected",
            )

        try:
            control_topic = self.CONTROL_TOPIC.format(serial=self.serial_number)
            payload = json.dumps(command)

            result = self._mqtt_client.publish(control_topic, payload, qos=qos)

            if result.rc == MQTT_ERR_SUCCESS:
                return CommandResult(CommandResultStatus.SUCCESS)
            else:
                return CommandResult(
                    CommandResultStatus.FAILURE,
                    message=f"MQTT publish failed with code {result.rc}",
                )
        except Exception as e:
            return CommandResult(
                CommandResultStatus.FAILURE,
                message=f"Failed to send MQTT command: {e}",
            )

    def _validate_temperature(
        self, temperature: float, min_temp: float = 0, max_temp: float = 400
    ) -> str | None:
        """Validate temperature value"""
        try:
            temp = float(temperature)
            if temp < min_temp or temp > max_temp:
                return f"Temperature {temp} out of range [{min_temp}, {max_temp}]"
            return None
        except (ValueError, TypeError):
            return f"Invalid temperature value: {temperature}"

    def _validate_axis(self, axis: str) -> str | None:
        """Validate axis parameter"""
        axis = axis.upper().strip()
        if not axis:
            return None  # Empty = home all
        valid_axes = {"X", "Y", "Z", "XY", "XZ", "YZ", "XYZ"}
        if axis not in valid_axes:
            return f"Invalid axis: {axis}. Valid: X, Y, Z, XY, XZ, YZ, XYZ"
        return None

    # --- Print Lifecycle Commands ---

    async def start_print(self, gcode_path: str) -> CommandResult:
        """Start print from uploaded file"""
        command = {
            "print": {
                "command": "project_file",
                "name": gcode_path,
            }
        }
        result = await self._send_mqtt_command(command, qos=1)
        if result.status == CommandResultStatus.SUCCESS:
            logger.info(f"Sent print start command for {gcode_path}")
        return result

    async def pause_print(self) -> CommandResult:
        """Pause current print"""
        command = {"print": {"command": "pause"}}
        result = await self._send_mqtt_command(command, qos=1)
        if result.status == CommandResultStatus.SUCCESS:
            logger.info("Sent pause command")
        return result

    async def resume_print(self) -> CommandResult:
        """Resume paused print"""
        command = {"print": {"command": "resume"}}
        result = await self._send_mqtt_command(command, qos=1)
        if result.status == CommandResultStatus.SUCCESS:
            logger.info("Sent resume command")
        return result

    async def cancel_print(self) -> CommandResult:
        """Cancel current print"""
        command = {"print": {"command": "stop"}}
        result = await self._send_mqtt_command(command, qos=1)
        if result.status == CommandResultStatus.SUCCESS:
            logger.info("Sent cancel command")
        return result

    async def emergency_cancel(self) -> CommandResult:
        """Emergency cancel with highest QoS"""
        command = {"print": {"command": "stop"}}
        result = await self._send_mqtt_command(command, qos=2)
        if result.status == CommandResultStatus.SUCCESS:
            logger.warning(f"EMERGENCY CANCEL executed for Bambu printer {self.serial_number}")
        return result

    # --- File Management ---

    async def upload_file(self, local_path: str, remote_name: str) -> CommandResult:
        """Upload file to printer via FTP/FTPS"""
        path = Path(local_path)
        if not path.exists():
            return CommandResult(
                CommandResultStatus.FAILURE,
                message=f"File not found: {local_path}",
            )

        try:
            ftp = await self._ensure_ftp_connected()

            with open(path, "rb") as f:
                ftp.storbinary(f"STOR {remote_name}", f)

            logger.info(f"Uploaded file: {remote_name}")
            return CommandResult(CommandResultStatus.SUCCESS)

        except Exception as e:
            return CommandResult(
                CommandResultStatus.FAILURE,
                message=f"Upload failed: {e}",
            )

    async def list_files(self) -> CommandResult:
        """List files on printer SD card via FTP"""
        try:
            ftp = await self._ensure_ftp_connected()

            files = []
            ftp.dir(files.append)

            # Parse FTP directory listing
            parsed_files = []
            for line in files:
                # Parse typical FTP LIST output
                # -rw-rw-rw- 1 user group 123456 Jan 01 12:00 filename.3mf
                parts = line.split()
                if len(parts) >= 9:
                    filename = parts[-1]
                    size = parts[4] if parts[4].isdigit() else "0"
                    parsed_files.append(
                        {
                            "name": filename,
                            "size": int(size),
                            "raw": line,
                        }
                    )

            return CommandResult(CommandResultStatus.SUCCESS, data={"files": parsed_files})

        except Exception as e:
            return CommandResult(
                CommandResultStatus.FAILURE,
                message=f"Failed to list files: {e}",
            )

    async def delete_file(self, filename: str) -> CommandResult:
        """Delete file from printer SD card via FTP"""
        try:
            ftp = await self._ensure_ftp_connected()
            ftp.delete(filename)
            logger.info(f"Deleted file: {filename}")
            return CommandResult(CommandResultStatus.SUCCESS)

        except Exception as e:
            return CommandResult(
                CommandResultStatus.FAILURE,
                message=f"Failed to delete file: {e}",
            )

    async def get_file_metadata(self, filename: str) -> CommandResult:
        """Get file metadata - download header and parse"""
        try:
            ftp = await self._ensure_ftp_connected()

            # Get file size
            size = ftp.size(filename)

            # Download first 10KB for header parsing
            header_data = bytearray()

            def collect_header(data):
                header_data.extend(data)
                if len(header_data) >= 10240:
                    raise ftplib.error_temp("Header collected")

            try:
                ftp.retrbinary(f"RETR {filename}", collect_header, blocksize=1024)
            except ftplib.error_temp:
                pass  # Expected when we abort after collecting header

            header_text = header_data.decode("utf-8", errors="replace")

            meta = {"filename": filename, "size": size}

            # Parse 3MF/G-code header
            patterns = {
                "total_layers": [
                    r";\s*total layer(?:s| number)\s*[:=]\s*(\d+)",
                    r";\s*LAYER_COUNT\s*[:=]\s*(\d+)",
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
                    r";\s*TIME\s*[:=]\s*(\d+)",
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
                        if key in ("total_layers", "estimated_time"):
                            meta[key] = int(val)
                        elif key in ("layer_height", "first_layer_height", "object_height"):
                            meta[key] = float(val)
                        else:
                            meta[key] = val
                        break

            return CommandResult(CommandResultStatus.SUCCESS, data=meta)

        except Exception as e:
            return CommandResult(
                CommandResultStatus.FAILURE,
                message=f"Failed to get file metadata: {e}",
            )

    # --- Z-Offset Control ---

    async def get_z_offset(self) -> float:
        """Get current Z offset from cached telemetry"""
        return self._z_offset

    async def set_z_offset(self, offset_mm: float) -> CommandResult:
        """Set Z offset via Bambu MQTT command"""
        # Validate offset
        try:
            offset_value = float(offset_mm)
        except (ValueError, TypeError) as e:
            return CommandResult(CommandResultStatus.FAILURE, message=f"Invalid offset value: {e}")

        # Bambu Z-offset range is typically -0.5mm to +0.5mm
        if offset_value < -0.5 or offset_value > 0.5:
            return CommandResult(
                CommandResultStatus.FAILURE,
                message=f"Offset {offset_value} out of range [-0.5, +0.5]",
            )

        # Send Z-offset command via MQTT
        command = {
            "print": {
                "command": "set_z_offset",
                "z_offset": round(offset_value, 3),
            }
        }

        result = await self._send_mqtt_command(command, qos=1)
        if result.status == CommandResultStatus.SUCCESS:
            self._z_offset = offset_value
            logger.info(f"Set Z-offset to {offset_mm}mm")
        return result

    # --- Temperature Control ---

    async def set_nozzle_temp(self, temperature: float) -> CommandResult:
        """Set nozzle/extruder temperature"""
        error = self._validate_temperature(temperature, 0, 350)
        if error:
            return CommandResult(CommandResultStatus.FAILURE, message=error)

        command = {
            "print": {
                "command": "set_nozzle_temp",
                "nozzle_temp": int(temperature),
            }
        }
        return await self._send_mqtt_command(command, qos=1)

    async def set_bed_temp(self, temperature: float) -> CommandResult:
        """Set heated bed temperature"""
        error = self._validate_temperature(temperature, 0, 120)
        if error:
            return CommandResult(CommandResultStatus.FAILURE, message=error)

        command = {
            "print": {
                "command": "set_bed_temp",
                "bed_temp": int(temperature),
            }
        }
        return await self._send_mqtt_command(command, qos=1)

    # --- Motion Control ---

    async def home_axes(self, axes: str = "") -> CommandResult:
        """Home printer axes via MQTT"""
        error = self._validate_axis(axes)
        if error:
            return CommandResult(CommandResultStatus.FAILURE, message=error)

        command = {"print": {"command": "home"}}
        return await self._send_mqtt_command(command, qos=1)

    async def move_axis(self, axis: str, distance: float, speed: int = 3000) -> CommandResult:
        """Move an axis by relative distance - not fully supported on Bambu"""
        # Bambu doesn't support direct G-code movement via MQTT
        # This would require using the jog commands if available
        return CommandResult(
            CommandResultStatus.FAILURE,
            message="Direct axis movement not supported via Bambu MQTT. Use printer's LCD.",
        )

    async def extrude(self, length: float, speed: int = 300) -> CommandResult:
        """Extrude or retract filament via load/unload commands"""
        if length > 0:
            command = {"print": {"command": "load_filament"}}
        else:
            command = {"print": {"command": "unload_filament"}}

        return await self._send_mqtt_command(command, qos=1)

    # --- Fan Control ---

    async def set_fan_speed(self, speed: int) -> CommandResult:
        """Set part cooling fan speed (0-255)"""
        if not isinstance(speed, int) or speed < 0 or speed > 255:
            return CommandResult(
                CommandResultStatus.FAILURE,
                message="Fan speed must be 0-255",
            )

        # Convert 0-255 to percentage for Bambu
        speed_pct = int((speed / 255.0) * 100)

        command = {
            "print": {
                "command": "set_cooling_fan_speed",
                "fan_speed": speed_pct,
            }
        }
        return await self._send_mqtt_command(command, qos=1)

    # --- Speed/Flow Overrides ---

    async def set_speed_override(self, percent: int) -> CommandResult:
        """Set print speed override percentage"""
        if not isinstance(percent, int) or percent < 10 or percent > 500:
            return CommandResult(
                CommandResultStatus.FAILURE,
                message="Speed override must be 10-500%",
            )

        command = {
            "print": {
                "command": "set_print_speed",
                "speed": percent,
            }
        }
        return await self._send_mqtt_command(command, qos=1)

    async def set_flow_override(self, percent: int) -> CommandResult:
        """Set flow/extrusion multiplier - not directly supported"""
        # Bambu doesn't expose flow override via MQTT
        # This would require firmware-level support
        return CommandResult(
            CommandResultStatus.FAILURE,
            message="Flow override not supported via Bambu MQTT",
        )

    # --- G-Code Commands ---

    async def run_gcode(self, command: str) -> CommandResult:
        """Run arbitrary G-code - limited support on Bambu"""
        # Bambu supports limited G-code commands via gcode_line
        # Only safe, non-movement commands should be used
        safe_commands = {"M104", "M140", "M190", "M109", "M106", "M107"}

        cmd_upper = command.strip().upper().split()[0] if command.strip() else ""

        if cmd_upper not in safe_commands:
            return CommandResult(
                CommandResultStatus.FAILURE,
                message=f"G-code command '{command}' not in safe list for Bambu: {safe_commands}",
            )

        mqtt_command = {
            "print": {
                "command": "gcode_line",
                "gcode": command.strip(),
            }
        }
        return await self._send_mqtt_command(mqtt_command, qos=1)

    # --- Macros (Not Applicable to Bambu) ---

    async def list_macros(self) -> CommandResult:
        """Bambu does not support macros like Klipper"""
        return CommandResult(
            CommandResultStatus.SUCCESS,
            data=[],
            message="Bambu printers do not support user-defined macros",
        )

    async def execute_macro(self, name: str) -> CommandResult:
        """Bambu does not support macros like Klipper"""
        return CommandResult(
            CommandResultStatus.FAILURE,
            message="Bambu printers do not support user-defined macros",
        )

    # --- System Control ---

    async def restart_firmware(self) -> CommandResult:
        """Restart printer firmware via MQTT"""
        command = {"print": {"command": "system"}}
        return await self._send_mqtt_command(command, qos=1)

    async def restart_service(self) -> CommandResult:
        """Restart service - not applicable to Bambu"""
        return CommandResult(
            CommandResultStatus.FAILURE,
            message="Service restart not applicable to Bambu printers",
        )

    async def get_system_info(self) -> CommandResult:
        """Get system info from cached telemetry"""
        info = {
            "serial_number": self.serial_number,
            "host": self.host,
            "mqtt_connected": self._mqtt_connected,
            "ftp_connected": self._ftp_connected,
            "z_offset": self._z_offset,
            "z_offset_calibrated": self._z_offset_calibrated,
        }

        # Add cached telemetry data
        if self._system_info:
            info.update(self._system_info)

        return CommandResult(CommandResultStatus.SUCCESS, data=info)

    async def motors_off(self) -> CommandResult:
        """Turn off stepper motors - not directly supported via MQTT"""
        return CommandResult(
            CommandResultStatus.FAILURE,
            message="Motors off command not supported via Bambu MQTT. Use printer's LCD.",
        )

    # --- Queries ---

    async def get_printer_state(self) -> NormalizedPrinterState:
        """Get current printer state"""
        if self._last_state is not None:
            return self._last_state
        return NormalizedPrinterState.UNKNOWN

    async def get_print_progress(self) -> PrintProgress:
        """Get current print progress with rich telemetry"""
        state = self._last_state if self._last_state is not None else NormalizedPrinterState.UNKNOWN

        # Get latest temps from cached telemetry
        bed_temp = self._system_info.get("bed_temp")
        nozzle_temp = self._system_info.get("nozzle_temp")
        bed_target = self._system_info.get("bed_target")
        nozzle_target = self._system_info.get("nozzle_target")

        return PrintProgress(
            state=state,
            filename=self._print_stats.filename,
            progress=self._print_stats.progress,
            print_time=self._print_stats.print_time if self._print_stats.print_time > 0 else None,
            current_layer=self._print_stats.current_layer
            if self._print_stats.current_layer > 0
            else None,
            total_layers=self._print_stats.total_layers
            if self._print_stats.total_layers > 0
            else None,
            nozzle_temp=nozzle_temp,
            nozzle_target=nozzle_target,
            bed_temp=bed_temp,
            bed_target=bed_target,
        )

    async def get_current_position(self) -> dict[str, float] | None:
        """Get last known position from cached telemetry."""
        return dict(self._last_position) if self._last_position else None

    async def is_connected(self) -> bool:
        """Check if adapter is connected"""
        return self._connected and self._mqtt_connected
