"""
Unit tests for first-layer classifier and Z-offset controller.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

import numpy as np

from sensing_voxel.core.models import (
    CommandResult,
    CommandResultStatus,
    FirstLayerAnalysisResult,
    FirstLayerBounds,
    FirstLayerQuality,
    HomographyMatrix,
    MonitoringConfig,
    VisionFrame,
    ZOffsetStatus,
)
from sensing_voxel.vision.first_layer import FirstLayerClassifier
from sensing_voxel.monitoring.z_offset import ZOffsetController


class TestFirstLayerClassifier:

    @pytest.fixture
    def classifier(self):
        return FirstLayerClassifier()

    @pytest.fixture
    def homography(self):
        return HomographyMatrix(
            matrix=np.eye(3).tolist(),
            mm_per_pixel=1.0,
        )

    @pytest.fixture
    def bounds(self):
        return FirstLayerBounds(min_x=50, max_x=150, min_y=50, max_y=150)

    def test_compute_roi_identity_homography(self, bounds, homography):
        roi = FirstLayerClassifier.compute_roi(bounds, homography)
        x, y, w, h = roi
        # With identity homography, pixel coords ≈ bed coords
        assert x <= 50
        assert y <= 50
        assert w >= 100
        assert h >= 100

    def test_analyze_uniform_gray(self, classifier):
        """Uniform gray image should classify as GOOD or EXCELLENT."""
        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        frame = VisionFrame(frame_data=img, width=200, height=200)
        result = classifier.analyze(frame, (10, 10, 180, 180))
        assert result.quality in (FirstLayerQuality.GOOD, FirstLayerQuality.EXCELLENT)

    def test_analyze_small_roi(self, classifier):
        """Very small ROI should be skipped."""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        frame = VisionFrame(frame_data=img, width=100, height=100)
        result = classifier.analyze(frame, (0, 0, 5, 5))
        assert "roi_too_small" in result.defects

    def test_analyze_low_coverage(self, classifier):
        """Mostly black image should be flagged as low coverage."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        frame = VisionFrame(frame_data=img, width=200, height=200)
        result = classifier.analyze(frame, (0, 0, 200, 200))
        assert "low_coverage" in result.defects

    def test_analyze_returns_roi(self, classifier):
        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        frame = VisionFrame(frame_data=img, width=200, height=200)
        result = classifier.analyze(frame, (10, 20, 100, 80))
        assert result.roi == (10, 20, 100, 80)


class TestZOffsetController:

    @pytest.fixture
    def config(self):
        return MonitoringConfig(
            z_offset_step_mm=0.01,
            z_offset_max_correction_mm=0.05,
            z_offset_saturation_count=3,
            min_consecutive_evaluations=2,
            cooldown_seconds=0,  # No cooldown for tests
        )

    @pytest.fixture
    def controller(self, config):
        return ZOffsetController(config)

    @pytest.fixture
    def mock_adapter(self):
        adapter = MagicMock()
        adapter.get_z_offset = AsyncMock(return_value=0.0)
        adapter.set_z_offset = AsyncMock(
            return_value=CommandResult(status=CommandResultStatus.SUCCESS)
        )
        return adapter

    @pytest.mark.asyncio
    async def test_start_captures_baseline(self, controller, mock_adapter):
        mock_adapter.get_z_offset = AsyncMock(return_value=0.15)
        await controller.start(mock_adapter)
        assert controller.state.baseline_offset == 0.15
        assert controller._running

    def test_evaluate_returns_none_when_not_running(self, controller):
        analysis = FirstLayerAnalysisResult(
            quality=FirstLayerQuality.POOR,
            confidence=0.8,
            defects=["gaps"],
        )
        assert controller.evaluate(analysis) is None

    def test_evaluate_returns_none_for_good_quality(self, controller):
        controller._running = True
        analysis = FirstLayerAnalysisResult(
            quality=FirstLayerQuality.GOOD,
            confidence=0.8,
        )
        assert controller.evaluate(analysis) is None

    def test_evaluate_hysteresis(self, controller):
        """Should require 2 consecutive same-direction signals."""
        controller._running = True

        # First gap detection → no action (hysteresis not met)
        analysis = FirstLayerAnalysisResult(
            quality=FirstLayerQuality.POOR,
            confidence=0.8,
            defects=["gaps"],
        )
        result = controller.evaluate(analysis)
        assert result is None  # Only 1 consecutive

        # Second gap detection → action
        result = controller.evaluate(analysis)
        assert result == -0.01  # Down step for gaps

    def test_evaluate_direction_change_resets_hysteresis(self, controller):
        controller._running = True

        gap_analysis = FirstLayerAnalysisResult(
            quality=FirstLayerQuality.POOR,
            confidence=0.8,
            defects=["gaps"],
        )
        ridge_analysis = FirstLayerAnalysisResult(
            quality=FirstLayerQuality.POOR,
            confidence=0.8,
            defects=["ridges"],
        )

        controller.evaluate(gap_analysis)  # 1x down
        controller.evaluate(ridge_analysis)  # Resets to 1x up
        result = controller.evaluate(ridge_analysis)  # 2x up → action
        assert result == 0.01  # Up step for ridges

    def test_evaluate_clamp(self, controller):
        controller._running = True
        controller.state.current_offset = 0.05  # Already at max

        analysis = FirstLayerAnalysisResult(
            quality=FirstLayerQuality.POOR,
            confidence=0.8,
            defects=["ridges"],
        )
        controller.evaluate(analysis)
        result = controller.evaluate(analysis)
        assert result is None  # Clamped

    def test_saturation_detection(self, controller):
        controller._running = True
        controller.state.current_offset = 0.05

        analysis = FirstLayerAnalysisResult(
            quality=FirstLayerQuality.POOR,
            confidence=0.8,
            defects=["ridges"],
        )

        for _ in range(6):  # Need consecutive evaluations at clamp
            controller.evaluate(analysis)

        assert controller.is_saturated

    @pytest.mark.asyncio
    async def test_apply_correction(self, controller, mock_adapter):
        controller._running = True
        controller.state.baseline_offset = 0.0

        await controller.apply_correction(mock_adapter, -0.01)

        assert controller.state.current_offset == -0.01
        assert controller.state.status == ZOffsetStatus.DIRTY
        mock_adapter.set_z_offset.assert_called_once_with(-0.01)

    @pytest.mark.asyncio
    async def test_restore_baseline(self, controller, mock_adapter):
        controller.state.baseline_offset = 0.15
        controller.state.current_offset = -0.03
        controller._running = True

        mock_adapter.get_z_offset = AsyncMock(return_value=0.15)
        await controller.restore_baseline(mock_adapter)

        mock_adapter.set_z_offset.assert_called_with(0.15)
        assert controller.state.status == ZOffsetStatus.CLEAN

    def test_reset_clears_state(self, controller):
        controller.state.current_offset = 0.03
        controller._last_direction = "up"
        controller._consecutive_same_direction = 2

        controller.reset()

        assert controller.state.current_offset == 0.0
        assert controller._last_direction is None
        assert controller._consecutive_same_direction == 0
