"""
Bambu Lab printer adapter with MQTT + FTP/FTPS

Implements minimal MVP adapter for Bambu printers with:
- Local MQTT for status + control
- FTP/FTPS for file upload
- Basic print lifecycle control
- Emergency stop via high-priority MQTT channel

Z-offset commands return NotSupported in MVP.
Bambu prints are NOTHING-only monitoring until full adapter parity.
"""

import asyncio
import ftplib
import logging
from datetime import datetime
from pathlib import Path

import paho.mqtt.client as mqtt

from sensing_voxel.adapters.base import (
    NotSupportedError,
    PrinterAdapter,
)
from sensing_voxel.core.models import (
    AdapterConnectionError,
    CommandResult,
    CommandResultStatus,
    NormalizedPrinterState,
    PrintProgress,
    TelemetryData,
)
from sensing_voxel.infra.retry import BackoffPolicy

logger = logging.getLogger(__name__)


class BambuAdapter(PrinterAdapter):
    """Bambu Lab printer adapter"""

    # MQTT topic patterns for Bambu printers
    STATUS_TOPIC = "device/+/report"
    CONTROL_TOPIC = "device/+/request"
    P2P_TOPIC = "p2p/report"  # P2P mode status

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
    }

    def __init__(
        self,
        host: str = "localhost",
        mqtt_port: int = 1883,
        ftp_port: int = 21,
        serial_number: str = "",  # Bambu printer serial number
        access_code: str = "",  # Bambu printer access code
        use_tls: bool = False,
    ):
        super().__init__()
        self.host = host
        self.mqtt_port = mqtt_port
        self.ftp_port = ftp_port
        self.serial_number = serial_number
        self.access_code = access_code
        self.use_tls = use_tls

        # MQTT client
        self._mqtt_client: mqtt.Client | None = None
        self._mqtt_connected = False

        # FTP client
        self._ftp_client: ftplib.FTP | None = None
        self._ftp_connected = False

        self._connected = False
        self._shutdown_event = asyncio.Event()

        # Track last known state
        self._last_state: NormalizedPrinterState | None = None

        # Current print metadata
        self._current_filename: str = ""
        self._current_progress: float = 0.0

        # MQTT reconnection policy
        self._reconnect_policy = BackoffPolicy(
            base_delay=1.0,
            max_delay=30.0,
            factor=1.8,
            jitter=0.25,
        )
        self._reconnect_attempts = 0

    async def connect(self) -> None:
        """Connect to Bambu printer via MQTT and FTP"""
        if self._connected:
            return

        # Create MQTT client
        self._mqtt_client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"bambu_adapter_{self.serial_number}",
        )

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
        if self._ftp_client:
            try:
                self._ftp_client.quit()
            except Exception as e:
                logger.warning(f"Error disconnecting FTP: {e}")
            finally:
                self._ftp_client = None

        self._connected = False
        logger.info("Disconnected from Bambu printer")

    async def _mqtt_reconnect_loop(self):
        """MQTT reconnection loop with exponential backoff (OpenClaw SSE pattern)"""
        # Maximum reconnection attempts before giving up
        max_reconnect_attempts = 100

        while not self._shutdown_event.is_set():
            try:
                await self._mqtt_connect_and_subscribe()
                # Connection successful, wait for disconnect
            except (AdapterConnectionError, ConnectionError) as e:
                logger.warning(f"MQTT connection error: {e}")
            except Exception as e:
                logger.error(f"MQTT error: {e}", exc_info=True)

            # Give up if we've exceeded maximum attempts
            if self._reconnect_attempts >= max_reconnect_attempts:
                logger.error(
                    f"MQTT reconnection failed after {self._reconnect_attempts} attempts. "
                    "Giving up."
                )
                await self._emit_connection_lost()
                break

            # Reconnect delay with exponential backoff and jitter
            import math
            import random

            # Calculate max useful attempts before reaching max_delay
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

            # Add random jitter to prevent thundering herd
            if self._reconnect_policy.jitter > 0:
                delay += delay * self._reconnect_policy.jitter * random.random()

            logger.info(
                f"Reconnecting MQTT in {delay:.1f}s "
                f"(attempt {self._reconnect_attempts + 1}/{max_reconnect_attempts})"
            )
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=delay)
                break
            except TimeoutError:
                self._reconnect_attempts += 1

    async def _mqtt_connect_and_subscribe(self):
        """Connect MQTT client and subscribe to topics"""
        if not self._mqtt_client:
            raise AdapterConnectionError("MQTT client not initialized")

        # Connect to MQTT broker
        self._mqtt_client.connect(self.host, self.mqtt_port, keepalive=60)
        self._mqtt_client.loop_start()

        # Wait for connection
        await self._wait_for_mqtt_connection(timeout=10)

        logger.info(f"MQTT connected to {self.host}:{self.mqtt_port}")
        self._reconnect_attempts = 0  # Reset on successful connection

        # Check printer state after reconnection to detect changes during disconnection
        # Note: Bambu MQTT doesn't have a REST API like Moonraker, so we rely on
        # the first status report after reconnection to update state
        # The adapter will receive a report message shortly after subscription

        # Emit connection restored event
        await self._emit_connection_restored()

        # Connection successful - wait for disconnect (loop will handle)
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

            # Subscribe to status topics
            status_topic = self.STATUS_TOPIC.replace("+", self.serial_number)
            self._mqtt_client.subscribe(status_topic, qos=0)
            logger.info(f"Subscribed to {status_topic}")

            # Also subscribe to P2P topic for completeness
            p2p_topic = self.P2P_TOPIC.replace("+", self.serial_number)
            self._mqtt_client.subscribe(p2p_topic, qos=0)
        else:
            logger.error(f"MQTT connection failed with code {reason_code}")

    def _on_mqtt_disconnect(self, client, userdata, flags, reason_code, properties):
        """MQTT on_disconnect callback"""
        self._mqtt_connected = False
        logger.warning(f"MQTT disconnected (code {reason_code})")
        # Emit connection lost event
        asyncio.create_task(self._emit_connection_lost())

    def _on_mqtt_message(self, client, userdata, msg):
        """MQTT on_message callback"""
        try:
            # Parse message payload as JSON
            import json

            payload = json.loads(msg.payload.decode())

            # Route based on topic
            if "report" in msg.topic:
                self._handle_report_message(payload)
        except Exception as e:
            logger.error(f"Error handling MQTT message: {e}", exc_info=True)

    def _handle_report_message(self, payload: dict):
        """Handle status report from Bambu printer"""
        try:
            # Extract print status
            print_status = payload.get("print", {})
            status_code = print_status.get("status", 0)

            # Map to normalized state
            normalized_state = self.STATE_MAP.get(status_code, NormalizedPrinterState.UNKNOWN)

            # Track filename and progress
            self._current_filename = print_status.get("subtask_name", "")
            self._current_progress = print_status.get("mc_percent", 0) / 100.0

            # Emit state changed event if state changed
            if self._last_state is not None and self._last_state != normalized_state:
                asyncio.create_task(self._emit_state_changed(self._last_state, normalized_state))
                logger.debug(f"State changed: {self._last_state.name} -> {normalized_state.name}")
            elif self._last_state is None:
                logger.debug(f"Initial state: {normalized_state.name}")

            self._last_state = normalized_state

            # Parse telemetry data
            telemetry = self._parse_telemetry(payload)
            if telemetry:
                asyncio.create_task(self._emit_telemetry_update(telemetry))

        except Exception as e:
            logger.error(f"Error handling report message: {e}", exc_info=True)

    def _parse_telemetry(self, payload: dict) -> TelemetryData | None:
        """Parse telemetry data from Bambu MQTT payload"""
        try:
            temps = {}
            position = None
            metadata = {}

            # Temperature data
            temps_data = payload.get("temps", {})
            if temps_data:
                temps["bed"] = temps_data.get("bed_temp", 0.0)
                temps["bed_target"] = temps_data.get("bed_temp_type", 0.0)
                temps["extruder"] = temps_data.get("nozzle_temp", 0.0)
                temps["extruder_target"] = temps_data.get("nozzle_temp_type", 0.0)

            # Position data (if available)
            pos_data = payload.get("pos", {})
            if pos_data:
                position = {
                    "x": pos_data.get("x", 0.0),
                    "y": pos_data.get("y", 0.0),
                    "z": pos_data.get("z", 0.0),
                }

            # Print metadata
            print_data = payload.get("print", {})
            if print_data:
                metadata["filename"] = print_data.get("subtask_name", "")
                metadata["progress"] = print_data.get("mc_percent", 0) / 100.0
                metadata["print_time"] = print_data.get("mc_print_time", 0)
                metadata["status"] = print_data.get("status", 0)

            if not temps and not position:
                return None

            return TelemetryData(
                timestamp=datetime.now(),
                temperatures=temps,
                position=position,
                metadata=metadata,
            )
        except Exception as e:
            logger.error(f"Error parsing telemetry: {e}", exc_info=True)
            return None

    # --- Commands ---

    async def start_print(self, gcode_path: str) -> CommandResult:
        """Start print from uploaded file"""
        # Bambu requires sending MQTT command to start print
        # Format: {"print": {"command": "project_file", "name": "filename"}}
        try:
            import json

            command = {
                "print": {
                    "command": "project_file",
                    "name": gcode_path,
                }
            }

            control_topic = self.CONTROL_TOPIC.replace("+", self.serial_number)
            payload = json.dumps(command)

            if self._mqtt_client:
                self._mqtt_client.publish(control_topic, payload, qos=1)
                logger.info(f"Sent print start command for {gcode_path}")
                return CommandResult(CommandResultStatus.SUCCESS)
            else:
                return CommandResult(
                    CommandResultStatus.FAILURE,
                    message="MQTT client not connected",
                )
        except Exception as e:
            return CommandResult(
                CommandResultStatus.FAILURE,
                message=f"Start print failed: {str(e)}",
            )

    async def pause_print(self) -> CommandResult:
        """Pause current print"""
        try:
            import json

            command = {"print": {"command": "pause"}}

            control_topic = self.CONTROL_TOPIC.replace("+", self.serial_number)
            payload = json.dumps(command)

            if self._mqtt_client:
                self._mqtt_client.publish(control_topic, payload, qos=1)
                logger.info("Sent pause command")
                return CommandResult(CommandResultStatus.SUCCESS)
            else:
                return CommandResult(
                    CommandResultStatus.FAILURE,
                    message="MQTT client not connected",
                )
        except Exception as e:
            return CommandResult(
                CommandResultStatus.FAILURE,
                message=f"Pause failed: {str(e)}",
            )

    async def resume_print(self) -> CommandResult:
        """Resume paused print"""
        try:
            import json

            command = {"print": {"command": "resume"}}

            control_topic = self.CONTROL_TOPIC.replace("+", self.serial_number)
            payload = json.dumps(command)

            if self._mqtt_client:
                self._mqtt_client.publish(control_topic, payload, qos=1)
                logger.info("Sent resume command")
                return CommandResult(CommandResultStatus.SUCCESS)
            else:
                return CommandResult(
                    CommandResultStatus.FAILURE,
                    message="MQTT client not connected",
                )
        except Exception as e:
            return CommandResult(
                CommandResultStatus.FAILURE,
                message=f"Resume failed: {str(e)}",
            )

    async def cancel_print(self) -> CommandResult:
        """Cancel current print"""
        try:
            import json

            command = {"print": {"command": "stop"}}

            control_topic = self.CONTROL_TOPIC.replace("+", self.serial_number)
            payload = json.dumps(command)

            if self._mqtt_client:
                self._mqtt_client.publish(control_topic, payload, qos=1)
                logger.info("Sent cancel command")
                return CommandResult(CommandResultStatus.SUCCESS)
            else:
                return CommandResult(
                    CommandResultStatus.FAILURE,
                    message="MQTT client not connected",
                )
        except Exception as e:
            return CommandResult(
                CommandResultStatus.FAILURE,
                message=f"Cancel failed: {str(e)}",
            )

    async def emergency_cancel(self) -> CommandResult:
        """Emergency cancel with high-priority MQTT"""
        # Use highest QoS and immediate publish
        try:
            import json

            command = {"print": {"command": "stop"}}

            control_topic = self.CONTROL_TOPIC.replace("+", self.serial_number)
            payload = json.dumps(command)

            if self._mqtt_client:
                # Use QoS 2 for highest reliability
                self._mqtt_client.publish(control_topic, payload, qos=2)
                logger.warning(f"EMERGENCY CANCEL executed for Bambu printer {self.serial_number}")
                return CommandResult(CommandResultStatus.SUCCESS)
            else:
                return CommandResult(
                    CommandResultStatus.FAILURE,
                    message="MQTT client not connected",
                )
        except Exception as e:
            return CommandResult(
                CommandResultStatus.FAILURE,
                message=f"Emergency cancel failed: {str(e)}",
            )

    async def upload_file(self, local_path: str, remote_name: str) -> CommandResult:
        """Upload file to printer via FTP/FTPS"""
        try:
            path = Path(local_path)
            if not path.exists():
                return CommandResult(
                    CommandResultStatus.FAILURE,
                    message=f"File not found: {local_path}",
                )

            # Connect to FTP server
            if not self._ftp_client:
                if self.use_tls:
                    self._ftp_client = ftplib.FTP_TLS()
                    self._ftp_client.connect(self.host, self.ftp_port)
                    self._ftp_client.login(self.serial_number, self.access_code)
                    self._ftp_client.prot_p()  # Secure data connection
                else:
                    self._ftp_client = ftplib.FTP()
                    self._ftp_client.connect(self.host, self.ftp_port)
                    self._ftp_client.login(self.serial_number, self.access_code)
                self._ftp_connected = True
                logger.info("FTP connection established")

            # Upload file
            with open(path, "rb") as f:
                self._ftp_client.storbinary(f"STOR {remote_name}", f)

            logger.info(f"Uploaded file: {remote_name}")
            return CommandResult(CommandResultStatus.SUCCESS)

        except Exception as e:
            return CommandResult(
                CommandResultStatus.FAILURE,
                message=f"Upload failed: {str(e)}",
            )

    async def get_z_offset(self) -> float:
        """Z-offset not supported in MVP"""
        raise NotSupportedError("Z-offset commands not supported for Bambu printers in MVP")

    async def set_z_offset(self, offset_mm: float) -> CommandResult:
        """Z-offset not supported in MVP"""
        return CommandResult(
            CommandResultStatus.FAILURE,
            message="Z-offset commands not supported for Bambu printers in MVP",
        )

    # --- Queries ---

    async def get_printer_state(self) -> NormalizedPrinterState:
        """Get current printer state"""
        if self._last_state is not None:
            return self._last_state
        return NormalizedPrinterState.UNKNOWN

    async def get_print_progress(self) -> PrintProgress:
        """Get current print progress"""
        state = self._last_state if self._last_state is not None else NormalizedPrinterState.UNKNOWN

        return PrintProgress(
            state=state,
            filename=self._current_filename,
            progress=self._current_progress,
            print_time=0.0,  # Bambu MQTT provides this but we'd need to track it
        )

    async def is_connected(self) -> bool:
        """Check if adapter is connected"""
        return self._connected and self._mqtt_connected
