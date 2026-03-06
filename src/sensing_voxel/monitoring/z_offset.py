"""
Z-offset controller for first-layer correction.

Applies bounded Z-offset adjustments based on first-layer quality analysis.
Implements hysteresis, clamping, and saturation exit.
"""

from __future__ import annotations

import logging
from datetime import datetime

from sensing_voxel.adapters.base import PrinterAdapter
from sensing_voxel.core.models import (
    FirstLayerAnalysisResult,
    FirstLayerQuality,
    MonitoringConfig,
    ZOffsetState,
    ZOffsetStatus,
)

logger = logging.getLogger(__name__)


class ZOffsetController:
    """
    Bounded Z-offset correction based on first-layer analysis.

    Corrections use ±step_mm increments clamped to ±max_correction_mm.
    Requires 2 consecutive same-direction signals before acting (hysteresis).
    Exits with pause after saturation_count consecutive POOR at clamp limit.
    """

    def __init__(self, config: MonitoringConfig | None = None):
        self.config = config or MonitoringConfig()
        self.state = ZOffsetState()

        # Hysteresis tracking
        self._last_direction: str | None = None  # "up" or "down"
        self._consecutive_same_direction = 0

        # Saturation tracking
        self._consecutive_poor_at_clamp = 0

        # Cooldown
        self._last_correction_time: datetime | None = None

        self._running = False

    async def start(self, adapter: PrinterAdapter):
        """Capture baseline Z-offset at print start."""
        try:
            baseline = await adapter.get_z_offset()
            self.state.baseline_offset = baseline
            self.state.current_offset = 0.0
            self.state.status = ZOffsetStatus.CLEAN
            self._running = True
            logger.info(f"Z-offset controller started, baseline={baseline:.3f}mm")
        except Exception as e:
            logger.error(f"Failed to capture Z-offset baseline: {e}")
            self._running = False

    def evaluate(self, analysis: FirstLayerAnalysisResult) -> float | None:
        """
        Evaluate first-layer analysis and return Z adjustment.

        Returns:
            Adjustment in mm (positive = up, negative = down) or None
        """
        if not self._running:
            return None

        # Check cooldown
        if self._last_correction_time:
            elapsed = (datetime.now() - self._last_correction_time).total_seconds()
            if elapsed < self.config.cooldown_seconds:
                return None

        # Skip low-confidence results
        if analysis.confidence < 0.5:
            return None

        # Determine direction from defects
        direction: str | None = None
        if "gaps" in analysis.defects:
            direction = "down"  # Z too high → move nozzle down
        elif "ridges" in analysis.defects:
            direction = "up"  # Z too low → move nozzle up
        elif analysis.quality in (FirstLayerQuality.EXCELLENT, FirstLayerQuality.GOOD):
            self._last_direction = None
            self._consecutive_same_direction = 0
            self._consecutive_poor_at_clamp = 0
            return None

        if direction is None:
            return None

        # Hysteresis: require consecutive same-direction signals
        if direction == self._last_direction:
            self._consecutive_same_direction += 1
        else:
            self._last_direction = direction
            self._consecutive_same_direction = 1

        if self._consecutive_same_direction < self.config.min_consecutive_evaluations:
            return None

        # Calculate adjustment
        step = self.config.z_offset_step_mm
        adjustment = -step if direction == "down" else step

        # Check clamp
        new_total = self.state.current_offset + adjustment
        max_corr = self.config.z_offset_max_correction_mm

        if abs(new_total) > max_corr:
            # At clamp limit
            self._consecutive_poor_at_clamp += 1
            logger.warning(
                f"Z-offset at clamp limit ({new_total:.3f}mm), "
                f"poor count: {self._consecutive_poor_at_clamp}/{self.config.z_offset_saturation_count}"
            )
            return None

        self._consecutive_poor_at_clamp = 0
        return adjustment

    async def apply_correction(self, adapter: PrinterAdapter, adjustment: float):
        """Apply Z-offset correction to printer."""
        try:
            result = await adapter.set_z_offset(
                (self.state.baseline_offset or 0) + self.state.current_offset + adjustment
            )

            from sensing_voxel.core.models import CommandResultStatus

            if result.status == CommandResultStatus.SUCCESS:
                self.state.current_offset += adjustment
                self.state.status = ZOffsetStatus.DIRTY
                self.state.dirty_since = self.state.dirty_since or datetime.now()
                self._last_correction_time = datetime.now()
                logger.info(
                    f"Z-offset corrected by {adjustment:+.3f}mm "
                    f"(total: {self.state.current_offset:+.3f}mm)"
                )
            else:
                logger.error(f"Z-offset correction failed: {result.message}")
        except Exception as e:
            logger.error(f"Z-offset correction error: {e}")

    @property
    def is_saturated(self) -> bool:
        """Check if corrections are saturated (should pause and notify)."""
        return self._consecutive_poor_at_clamp >= self.config.z_offset_saturation_count

    async def restore_baseline(self, adapter: PrinterAdapter):
        """Restore original Z-offset baseline on print end."""
        if self.state.baseline_offset is None:
            return

        try:
            await adapter.set_z_offset(self.state.baseline_offset)
            logger.info(
                f"Z-offset restored to baseline: {self.state.baseline_offset:.3f}mm"
            )

            # Verify
            current = await adapter.get_z_offset()
            diff = abs(current - self.state.baseline_offset)
            if diff > 0.005:
                logger.warning(
                    f"Z-offset verification failed: expected {self.state.baseline_offset:.3f}, "
                    f"got {current:.3f} (diff={diff:.4f}mm)"
                )
            else:
                self.state.status = ZOffsetStatus.CLEAN

        except Exception as e:
            logger.error(f"Failed to restore Z-offset baseline: {e}")

    async def stop(self, adapter: PrinterAdapter):
        """Stop controller and restore baseline."""
        if self._running:
            await self.restore_baseline(adapter)
            self._running = False
            logger.info("Z-offset controller stopped")

    def reset(self):
        """Reset controller state (for /reset_z_offset)."""
        self.state = ZOffsetState()
        self._last_direction = None
        self._consecutive_same_direction = 0
        self._consecutive_poor_at_clamp = 0
        self._last_correction_time = None
