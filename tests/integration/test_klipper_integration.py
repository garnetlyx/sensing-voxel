"""
Integration test skeleton for Klipper adapter

Note: These tests require a real Moonraker instance to be running.
They are skipped by default and can be run with:
    pytest tests/integration/test_klipper_integration.py -m integration
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from sensing_voxel.adapters.klipper import KlipperAdapter
from sensing_voxel.core.models import NormalizedPrinterState


@pytest.mark.integration
@pytest.mark.skip(reason="Requires real Moonraker instance")
class TestKlipperIntegration:
    """Integration tests requiring real Moonraker connection"""

    @pytest.fixture
    async def adapter(self):
        """Create adapter connected to real Moonraker"""
        # Update these to match your test printer
        adapter = KlipperAdapter(
            host="localhost",
            port=7125,
            api_key=None,  # Or your API key
        )
        await adapter.connect()
        yield adapter
        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_connect_disconnect(self, adapter):
        """Test basic connection"""
        assert await adapter.is_connected()
        await adapter.disconnect()
        assert not await adapter.is_connected()

    @pytest.mark.asyncio
    async def test_get_printer_state(self, adapter):
        """Test querying printer state"""
        state = await adapter.get_printer_state()
        assert isinstance(state, NormalizedPrinterState)
        # State should be one of the valid states
        assert state in [
            NormalizedPrinterState.PRINTING,
            NormalizedPrinterState.PAUSED,
            NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT,
            NormalizedPrinterState.TERMINAL,
            NormalizedPrinterState.ERROR,
            NormalizedPrinterState.UNKNOWN,
        ]

    @pytest.mark.asyncio
    async def test_get_print_progress(self, adapter):
        """Test querying print progress"""
        progress = await adapter.get_print_progress()
        # Should return a PrintProgress object
        assert progress is not None
        assert hasattr(progress, 'state')

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires G-code file and will start a print")
    async def test_full_print_workflow(self, adapter):
        """
        Test complete print workflow:
        1. Upload G-code
        2. Start print
        3. Monitor progress
        4. Cancel print
        """
        # This is a skeleton - implement when ready to test with real printer
        pass

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Will move printer - ensure bed is clear")
    async def test_z_offset_commands(self, adapter):
        """
        Test Z-offset get/set commands
        WARNING: This will move the printer!
        """
        # Get current Z offset
        current_offset = await adapter.get_z_offset()
        assert isinstance(current_offset, float)

        # Set a small offset (safe for testing)
        test_offset = current_offset + 0.01
        result = await adapter.set_z_offset(test_offset)
        assert result.status.name == "SUCCESS"

        # Verify it was set
        new_offset = await adapter.get_z_offset()
        # Should be close to what we set (within 0.005mm tolerance)
        assert abs(new_offset - test_offset) < 0.005

        # Restore original offset
        await adapter.set_z_offset(current_offset)


class TestKlipperMockIntegration:
    """
    Integration-like tests using mocked Moonraker responses.
    These verify the adapter works correctly with realistic Moonraker API responses.
    """

    @pytest.fixture
    def mock_session(self):
        """Create mocked aiohttp session"""
        session = Mock()
        session.post = AsyncMock()
        session.get = AsyncMock()
        return session

    @pytest.mark.asyncio
    async def test_parse_realistic_moonraker_response(self):
        """
        Test parsing actual Moonraker API responses.
        This helps catch issues with field name changes in Moonraker updates.
        """
        from sensing_voxel.adapters.klipper import KlipperAdapter

        adapter = KlipperAdapter()

        # Realistic Moonraker response for print_stats query
        realistic_response = {
            "eventtime": 12345.678,
            "status": {
                "print_stats": {
                    "state": "printing",
                    "filename": "test.gcode",
                    "progress": 0.45,
                    "print_duration": 1234.5,
                    "total_duration": 2000.0,
                    "info": {
                        "current_layer": 15,
                        "total_layer": 100,
                    }
                },
                "heater_bed": {
                    "temperature": 60.5,
                    "target": 60.0,
                },
                "extruder": {
                    "temperature": 210.0,
                    "target": 210.0,
                },
                "toolhead": {
                    "position": [100.0, 100.0, 5.0, 0.0],
                },
                "gcode_move": {
                    "gcode_position": [100.0, 100.0, 5.0, 0.0],
                    "homing_origin": [0.0, 0.0, 0.0, 0.0],
                }
            }
        }

        # Parse telemetry from this response
        status = realistic_response["status"]
        print_stats = status["print_stats"]
        telemetry = adapter._parse_telemetry(status, print_stats)

        assert telemetry is not None
        assert "bed" in telemetry.temperatures
        assert "extruder" in telemetry.temperatures
        assert telemetry.position is not None
        assert telemetry.position["x"] == 100.0
        assert telemetry.metadata["filename"] == "test.gcode"
        assert telemetry.metadata["progress"] == 0.45

    @pytest.mark.asyncio
    async def test_state_map_comprehensive(self):
        """
        Test that all common Moonraker states are handled.
        This prevents UNKNOWN state for normal printer operations.
        """
        from sensing_voxel.adapters.klipper import KlipperAdapter
        from sensing_voxel.core.models import NormalizedPrinterState

        adapter = KlipperAdapter()

        # Test all states mentioned in Moonraker docs
        test_cases = [
            ("printing", NormalizedPrinterState.PRINTING),
            ("paused", NormalizedPrinterState.PAUSED),
            ("standby", NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT),
            ("idle", NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT),
            ("complete", NormalizedPrinterState.TERMINAL),
            ("cancelled", NormalizedPrinterState.TERMINAL),
            ("error", NormalizedPrinterState.ERROR),
            ("ready", NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT),
            ("startup", NormalizedPrinterState.UNKNOWN),  # Startup is transient
            ("shutdown", NormalizedPrinterState.UNKNOWN),  # Shutdown is transient
            ("disconnecting", NormalizedPrinterState.UNKNOWN),  # Disconnecting is transient
        ]

        for moonraker_state, expected_normalized in test_cases:
            result = adapter.STATE_MAP.get(moonraker_state, NormalizedPrinterState.UNKNOWN)
            assert result == expected_normalized, \
                f"State '{moonraker_state}' should map to {expected_normalized.name}, got {result.name}"
