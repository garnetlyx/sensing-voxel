"""
Abstract base class for printer adapters
"""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable

from sensing_voxel.core.models import (
    AdapterCommandError,
    CommandResult,
    NormalizedPrinterState,
    PrintProgress,
    TelemetryData,
)


class PrinterAdapter(ABC):
    """Abstract base class for printer adapters"""

    def __init__(self):
        self._state_changed_callbacks: list[
            Callable[[NormalizedPrinterState, NormalizedPrinterState], Awaitable[None]]
        ] = []
        self._telemetry_callbacks: list[Callable[[TelemetryData], Awaitable[None]]] = []
        self._connection_lost_callbacks: list[Callable[[], Awaitable[None]]] = []
        self._connection_restored_callbacks: list[Callable[[], Awaitable[None]]] = []

    # --- Commands ---

    @abstractmethod
    async def start_print(self, gcode_path: str) -> CommandResult:
        """Start printing a G-code file"""
        pass

    @abstractmethod
    async def pause_print(self) -> CommandResult:
        """Pause current print"""
        pass

    @abstractmethod
    async def resume_print(self) -> CommandResult:
        """Resume paused print"""
        pass

    @abstractmethod
    async def cancel_print(self) -> CommandResult:
        """Cancel current print"""
        pass

    @abstractmethod
    async def emergency_cancel(self) -> CommandResult:
        """Emergency cancel - bypasses command queue"""
        pass

    @abstractmethod
    async def upload_file(self, local_path: str, remote_name: str) -> CommandResult:
        """Upload file to printer"""
        pass

    @abstractmethod
    async def get_z_offset(self) -> float:
        """Get current Z-offset"""
        pass

    @abstractmethod
    async def set_z_offset(self, offset_mm: float) -> CommandResult:
        """Set Z-offset"""
        pass

    @abstractmethod
    async def set_nozzle_temp(self, temperature: float) -> CommandResult:
        """Set nozzle/extruder temperature"""
        pass

    @abstractmethod
    async def set_bed_temp(self, temperature: float) -> CommandResult:
        """Set heated bed temperature"""
        pass

    @abstractmethod
    async def home_axes(self, axes: str = "") -> CommandResult:
        """Home printer axes. Empty string = home all."""
        pass

    @abstractmethod
    async def set_fan_speed(self, speed: int) -> CommandResult:
        """Set part cooling fan speed (0-255)"""
        pass

    @abstractmethod
    async def run_gcode(self, command: str) -> CommandResult:
        """Run arbitrary G-code command"""
        pass

    @abstractmethod
    async def list_files(self) -> CommandResult:
        """List G-code files on the printer"""
        pass

    @abstractmethod
    async def delete_file(self, filename: str) -> CommandResult:
        """Delete a G-code file from the printer"""
        pass

    @abstractmethod
    async def get_file_metadata(self, filename: str) -> CommandResult:
        """Get G-code file metadata (layers, height, etc.)"""
        pass

    @abstractmethod
    async def move_axis(self, axis: str, distance: float, speed: int = 3000) -> CommandResult:
        """Move an axis by relative distance (mm)"""
        pass

    @abstractmethod
    async def extrude(self, length: float, speed: int = 300) -> CommandResult:
        """Extrude or retract filament (negative = retract)"""
        pass

    @abstractmethod
    async def set_speed_override(self, percent: int) -> CommandResult:
        """Set speed override percentage (e.g. 100 = normal)"""
        pass

    @abstractmethod
    async def set_flow_override(self, percent: int) -> CommandResult:
        """Set flow/extrusion multiplier percentage"""
        pass

    @abstractmethod
    async def list_macros(self) -> CommandResult:
        """List available Klipper macros"""
        pass

    @abstractmethod
    async def execute_macro(self, name: str) -> CommandResult:
        """Execute a Klipper macro by name"""
        pass

    @abstractmethod
    async def restart_firmware(self) -> CommandResult:
        """Restart Klipper firmware"""
        pass

    @abstractmethod
    async def restart_service(self) -> CommandResult:
        """Restart Klipper service"""
        pass

    @abstractmethod
    async def get_system_info(self) -> CommandResult:
        """Get system info (CPU, memory, etc.)"""
        pass

    @abstractmethod
    async def motors_off(self) -> CommandResult:
        """Turn off all stepper motors"""
        pass

    # --- Queries ---

    @abstractmethod
    async def get_printer_state(self) -> NormalizedPrinterState:
        """Get current printer state"""
        pass

    @abstractmethod
    async def get_print_progress(self) -> PrintProgress:
        """Get current print progress"""
        pass

    @abstractmethod
    async def is_connected(self) -> bool:
        """Check if adapter is connected"""
        pass

    # --- Lifecycle ---

    @abstractmethod
    async def connect(self) -> None:
        """Connect to printer"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from printer"""
        pass

    # --- Events (adapter emits these) ---

    def on_state_changed(
        self,
        callback: Callable[[NormalizedPrinterState, NormalizedPrinterState], Awaitable[None]],
    ):
        """Register callback for state changes"""
        self._state_changed_callbacks.append(callback)

    def remove_state_changed_handler(
        self,
        callback: Callable[[NormalizedPrinterState, NormalizedPrinterState], Awaitable[None]],
    ):
        """Remove state changed callback"""
        if callback in self._state_changed_callbacks:
            self._state_changed_callbacks.remove(callback)

    def on_telemetry_update(self, callback: Callable[[TelemetryData], Awaitable[None]]):
        """Register callback for telemetry updates"""
        self._telemetry_callbacks.append(callback)

    def remove_telemetry_handler(self, callback: Callable[[TelemetryData], Awaitable[None]]):
        """Remove telemetry callback"""
        if callback in self._telemetry_callbacks:
            self._telemetry_callbacks.remove(callback)

    def on_connection_lost(self, callback: Callable[[], Awaitable[None]]):
        """Register callback for connection loss"""
        self._connection_lost_callbacks.append(callback)

    def remove_connection_lost_handler(self, callback: Callable[[], Awaitable[None]]):
        """Remove connection lost callback"""
        if callback in self._connection_lost_callbacks:
            self._connection_lost_callbacks.remove(callback)

    def on_connection_restored(self, callback: Callable[[], Awaitable[None]]):
        """Register callback for connection restore"""
        self._connection_restored_callbacks.append(callback)

    def remove_connection_restored_handler(self, callback: Callable[[], Awaitable[None]]):
        """Remove connection restored callback"""
        if callback in self._connection_restored_callbacks:
            self._connection_restored_callbacks.remove(callback)

    async def _emit_state_changed(
        self,
        old_state: NormalizedPrinterState,
        new_state: NormalizedPrinterState,
    ):
        """Emit state changed event"""
        for callback in self._state_changed_callbacks:
            await callback(old_state, new_state)

    async def _emit_telemetry_update(self, data: TelemetryData):
        """Emit telemetry update event"""
        for callback in self._telemetry_callbacks:
            await callback(data)

    async def _emit_connection_lost(self):
        """Emit connection lost event"""
        for callback in self._connection_lost_callbacks:
            await callback()

    async def _emit_connection_restored(self):
        """Emit connection restored event"""
        for callback in self._connection_restored_callbacks:
            await callback()


class NotSupportedError(AdapterCommandError):
    """Command not supported by this adapter"""

    def __init__(self, message: str = "Command not supported"):
        super().__init__(message, retryable=False)
