"""
Tests for closed-loop calibration automation module.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from sensing_voxel.calibration.config_injector import (
    ConfigChange,
    ConfigInjector,
    ConfigParameterType,
    ConfigProposal,
    SaveConfigError,
)
from sensing_voxel.calibration.tower_generator import (
    FlowTowerConfig,
    GeneratedTower,
    TempTowerConfig,
    TowerGenerator,
    TowerSection,
    TowerType,
)
from sensing_voxel.calibration.visual_scoring import (
    ScoringConfig,
    StringingLevel,
    SurfaceQuality,
    TowerZone,
    VisualScorer,
    ZoneScore,
)
from sensing_voxel.core.models import CommandResult, CommandResultStatus


class TestTempTowerConfig:
    def test_default_values(self):
        config = TempTowerConfig()
        assert config.start_temp == 220
        assert config.end_temp == 180
        assert config.step_temp == 5
        assert config.base_height == 2.0
        assert config.section_height == 7.0

    def test_temperatures_calculation(self):
        config = TempTowerConfig(start_temp=210, end_temp=190, step_temp=10)
        assert config.temperatures == [210, 200, 190]

    def test_num_sections(self):
        config = TempTowerConfig(start_temp=220, end_temp=200, step_temp=10)
        assert config.num_sections == 3

    def test_total_height(self):
        config = TempTowerConfig(start_temp=220, end_temp=200, step_temp=10)
        assert config.total_height == 2.0 + (3 * 7.0)

    def test_validation_start_end_order(self):
        with pytest.raises(ValueError, match="start_temp must be greater"):
            TempTowerConfig(start_temp=180, end_temp=220)

    def test_validation_step_positive(self):
        with pytest.raises(ValueError, match="step_temp must be positive"):
            TempTowerConfig(step_temp=0)

    def test_validation_divisible(self):
        with pytest.raises(ValueError, match="temperature range must be divisible"):
            TempTowerConfig(start_temp=220, end_temp=181, step_temp=5)


class TestFlowTowerConfig:
    def test_default_values(self):
        config = FlowTowerConfig()
        assert config.start_flow == 90
        assert config.end_flow == 110
        assert config.step_flow == 5

    def test_flows_calculation(self):
        config = FlowTowerConfig(start_flow=95, end_flow=105, step_flow=5)
        assert config.flows == [95, 100, 105]

    def test_num_sections(self):
        config = FlowTowerConfig(start_flow=90, end_flow=110, step_flow=10)
        assert config.num_sections == 3

    def test_validation_start_end_order(self):
        with pytest.raises(ValueError, match="start_flow must be less"):
            FlowTowerConfig(start_flow=110, end_flow=90)

    def test_validation_step_positive(self):
        with pytest.raises(ValueError, match="step_flow must be positive"):
            FlowTowerConfig(step_flow=0)

    def test_validation_divisible(self):
        with pytest.raises(ValueError, match="flow range must be divisible"):
            FlowTowerConfig(start_flow=90, end_flow=111, step_flow=5)


class TestTowerGenerator:
    def test_generate_temp_tower(self):
        generator = TowerGenerator()
        config = TempTowerConfig(start_temp=210, end_temp=190, step_temp=10)
        tower = generator.generate_temp_tower(config)

        assert tower.tower_type == TowerType.TEMPERATURE
        assert len(tower.sections) == 3
        assert tower.sections[0].value == 210
        assert tower.sections[2].value == 190
        assert "M104 S210" in tower.gcode
        assert "M104 S200" in tower.gcode
        assert "M104 S190" in tower.gcode

    def test_generate_flow_tower(self):
        generator = TowerGenerator()
        config = FlowTowerConfig(start_flow=95, end_flow=105, step_flow=5)
        tower = generator.generate_flow_tower(config)

        assert tower.tower_type == TowerType.FLOW
        assert len(tower.sections) == 3
        assert tower.sections[0].value == 95
        assert tower.sections[2].value == 105
        assert "M221 S95" in tower.gcode
        assert "M221 S100" in tower.gcode
        assert "M221 S105" in tower.gcode

    def test_gcode_structure(self):
        generator = TowerGenerator()
        tower = generator.generate_temp_tower()

        assert tower.gcode.startswith("; Temperature Tower")
        assert "G28" in tower.gcode
        assert "M104" in tower.gcode
        assert "M140" in tower.gcode
        assert "G1" in tower.gcode
        assert "M117 Zone:" in tower.gcode

    def test_section_properties(self):
        generator = TowerGenerator()
        config = TempTowerConfig(start_temp=220, end_temp=200, step_temp=10)
        tower = generator.generate_temp_tower(config)

        section = tower.sections[0]
        assert section.index == 0
        assert section.value == 220
        assert section.height == pytest.approx(config.section_height)
        assert section.mid_z == pytest.approx(section.start_z + section.height / 2)

    def test_metadata(self):
        generator = TowerGenerator()
        tower = generator.generate_temp_tower()

        assert tower.metadata["type"] == "temperature"
        assert "start_temp" in tower.metadata
        assert "total_sections" in tower.metadata
        assert tower.metadata["total_sections"] == len(tower.sections)

    def test_save_tower(self):
        generator = TowerGenerator()
        tower = generator.generate_temp_tower()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".gcode", delete=False) as f:
            path = f.name

        try:
            tower.save(path)
            saved_content = Path(path).read_text()
            assert saved_content == tower.gcode
        finally:
            Path(path).unlink(missing_ok=True)

    def test_custom_start_gcode(self):
        custom_start = "G28\nG1 Z10\nM104 S{temp}\n"
        generator = TowerGenerator(start_gcode=custom_start)
        tower = generator.generate_temp_tower()

        assert "G1 Z10" in tower.gcode

    def test_extrusion_calculation(self):
        generator = TowerGenerator()
        config = TempTowerConfig()
        extrusion = generator._calculate_extrusion(config)

        assert extrusion > 0


class TestScoringConfig:
    def test_default_weights_sum_to_one(self):
        config = ScoringConfig()
        total = config.surface_weight + config.stringing_weight + config.adhesion_weight
        assert 0.99 <= total <= 1.01

    def test_validation_weights_sum(self):
        with pytest.raises(ValueError, match="weights must sum to 1.0"):
            ScoringConfig(surface_weight=0.5, stringing_weight=0.5, adhesion_weight=0.5)


class TestVisualScorer:
    @pytest.fixture
    def scorer(self):
        return VisualScorer()

    @pytest.fixture
    def sample_zone(self):
        return TowerZone(
            section_index=0,
            value=200,
            center_x_mm=30,
            center_y_mm=30,
            width_mm=20,
            height_mm=7,
            layer_start=10,
            layer_end=45,
        )

    @pytest.fixture
    def sample_image(self):
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_score_zone_returns_zone_score(self, scorer, sample_zone, sample_image):
        score = scorer.score_zone(sample_zone, sample_image)

        assert isinstance(score, ZoneScore)
        assert score.zone == sample_zone
        assert 0 <= score.surface_score <= 1
        assert 0 <= score.stringing_score <= 1
        assert 0 <= score.adhesion_score <= 1
        assert 0 <= score.overall_score <= 1

    def test_score_zone_surface_quality(self, scorer, sample_zone, sample_image):
        score = scorer.score_zone(sample_zone, sample_image)

        assert isinstance(score.surface_quality, SurfaceQuality)
        assert score.surface_quality in list(SurfaceQuality)

    def test_score_zone_stringing_level(self, scorer, sample_zone, sample_image):
        score = scorer.score_zone(sample_zone, sample_image)

        assert isinstance(score.stringing_level, StringingLevel)
        assert score.stringing_level in list(StringingLevel)

    def test_score_all_zones(self, scorer, sample_image):
        zones = [
            TowerZone(0, 200, 30, 30, 20, 7, 10, 45),
            TowerZone(1, 210, 30, 30, 20, 7, 46, 80),
            TowerZone(2, 220, 30, 30, 20, 7, 81, 115),
        ]
        images = [sample_image for _ in zones]

        scores = scorer.score_all_zones(zones, images)

        assert len(scores) == 3
        for score in scores:
            assert isinstance(score, ZoneScore)

    def test_score_zone_with_error(self, scorer, sample_zone):
        invalid_image = np.array([])
        score = scorer.score_zone(sample_zone, invalid_image)

        assert score.surface_quality == SurfaceQuality.POOR
        assert score.stringing_level == StringingLevel.SEVERE
        assert score.overall_score == 0.0

    def test_recommend_optimal_basic(self, scorer, sample_zone, sample_image):
        scores = [
            ZoneScore(
                zone=TowerZone(0, 200, 30, 30, 20, 7, 10, 45),
                surface_quality=SurfaceQuality.GOOD,
                surface_score=0.8,
                stringing_level=StringingLevel.NONE,
                stringing_score=0.9,
                adhesion_score=0.85,
                overall_score=0.85,
            ),
            ZoneScore(
                zone=TowerZone(1, 210, 30, 30, 20, 7, 46, 80),
                surface_quality=SurfaceQuality.EXCELLENT,
                surface_score=0.95,
                stringing_level=StringingLevel.NONE,
                stringing_score=0.95,
                adhesion_score=0.9,
                overall_score=0.93,
            ),
        ]

        value, best_score, comparison = scorer.recommend_optimal(scores)

        assert value == 210
        assert best_score.overall_score == 0.93
        assert len(comparison) == 2

    def test_recommend_optimal_empty(self, scorer):
        value, best_score, comparison = scorer.recommend_optimal([])

        assert value == 0
        assert best_score is None
        assert comparison == []

    def test_recommend_optimal_low_scores(self, scorer):
        scores = [
            ZoneScore(
                zone=TowerZone(0, 200, 30, 30, 20, 7, 10, 45),
                surface_quality=SurfaceQuality.POOR,
                surface_score=0.3,
                stringing_level=StringingLevel.SEVERE,
                stringing_score=0.2,
                adhesion_score=0.3,
                overall_score=0.25,
            ),
        ]

        value, best_score, comparison = scorer.recommend_optimal(scores, min_acceptable_score=0.5)

        assert value == 200
        assert best_score is not None


class TestConfigInjector:
    @pytest.fixture
    def mock_adapter(self):
        adapter = MagicMock()
        adapter.run_gcode = AsyncMock()
        adapter.is_connected = AsyncMock(return_value=True)
        return adapter

    @pytest.fixture
    def injector(self, mock_adapter):
        return ConfigInjector(mock_adapter)

    @pytest.mark.asyncio
    async def test_propose_temperature_changes(self, injector):
        scores = {"best_score": 0.85}
        proposal = await injector.propose_changes(200, "temperature", scores)

        assert isinstance(proposal, ConfigProposal)
        assert proposal.source == "temperature_tower_calibration"
        assert len(proposal.changes) == 1
        assert proposal.changes[0].section == "extruder"
        assert proposal.changes[0].option == "target_temperature"
        assert proposal.changes[0].value == 200
        assert proposal.confidence == 0.85

    @pytest.mark.asyncio
    async def test_propose_flow_changes(self, injector):
        scores = {"best_score": 0.9}
        proposal = await injector.propose_changes(105, "flow", scores)

        assert isinstance(proposal, ConfigProposal)
        assert proposal.source == "flow_tower_calibration"
        assert len(proposal.changes) == 1
        assert proposal.changes[0].option == "flow_multiplier"
        assert proposal.changes[0].value == 105

    @pytest.mark.asyncio
    async def test_propose_unknown_type_raises(self, injector):
        with pytest.raises(ValueError, match="unknown calibration type"):
            await injector.propose_changes(100, "unknown", {})

    @pytest.mark.asyncio
    async def test_inject_config_success(self, injector, mock_adapter):
        mock_adapter.run_gcode.return_value = CommandResult(CommandResultStatus.SUCCESS)

        proposal = ConfigProposal(
            changes=[
                ConfigChange(
                    section="extruder",
                    option="target_temperature",
                    value=200,
                    parameter_type=ConfigParameterType.TEMPERATURE,
                )
            ],
            source="test",
            timestamp="2024-01-01T00:00:00",
            confidence=0.9,
        )

        result = await injector.inject_config(proposal)

        assert result.status == CommandResultStatus.SUCCESS
        assert mock_adapter.run_gcode.call_count == 2

    @pytest.mark.asyncio
    async def test_inject_config_failure(self, injector, mock_adapter):
        mock_adapter.run_gcode.return_value = CommandResult(
            CommandResultStatus.FAILURE, message="G-code error"
        )

        proposal = ConfigProposal(
            changes=[
                ConfigChange(
                    section="extruder",
                    option="target_temperature",
                    value=200,
                    parameter_type=ConfigParameterType.TEMPERATURE,
                )
            ],
            source="test",
            timestamp="2024-01-01T00:00:00",
            confidence=0.9,
        )

        result = await injector.inject_config(proposal)

        assert result.status == CommandResultStatus.FAILURE

    @pytest.mark.asyncio
    async def test_inject_config_dry_run(self, injector, mock_adapter):
        proposal = ConfigProposal(
            changes=[
                ConfigChange(
                    section="extruder",
                    option="target_temperature",
                    value=200,
                    parameter_type=ConfigParameterType.TEMPERATURE,
                )
            ],
            source="test",
            timestamp="2024-01-01T00:00:00",
            confidence=0.9,
        )

        result = await injector.inject_config(proposal, dry_run=True)

        assert result.status == CommandResultStatus.SUCCESS
        assert result.data["dry_run"] is True
        mock_adapter.run_gcode.assert_not_called()

    @pytest.mark.asyncio
    async def test_inject_temperature(self, injector, mock_adapter):
        mock_adapter.run_gcode.return_value = CommandResult(CommandResultStatus.SUCCESS)

        result = await injector.inject_temperature(205)

        assert result.status == CommandResultStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_inject_flow(self, injector, mock_adapter):
        mock_adapter.run_gcode.return_value = CommandResult(CommandResultStatus.SUCCESS)

        result = await injector.inject_flow(98)

        assert result.status == CommandResultStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_validate_connection(self, injector, mock_adapter):
        mock_adapter.is_connected.return_value = True

        is_valid = await injector.validate_connection()

        assert is_valid is True

    @pytest.mark.asyncio
    async def test_get_current_config(self, injector, mock_adapter):
        mock_adapter.run_gcode.return_value = CommandResult(
            CommandResultStatus.SUCCESS, data={"extruder": {"temperature": 200}}
        )

        config = await injector.get_current_config()

        assert config == {"extruder": {"temperature": 200}}


class TestIntegration:
    def test_full_temp_tower_workflow(self):
        generator = TowerGenerator()
        config = TempTowerConfig(start_temp=220, end_temp=200, step_temp=10)
        tower = generator.generate_temp_tower(config)

        assert len(tower.sections) == 3

        for section in tower.sections:
            zone = TowerZone(
                section_index=section.index,
                value=section.value,
                center_x_mm=section.center_x,
                center_y_mm=section.center_y,
                width_mm=section.size,
                height_mm=section.height,
                layer_start=int(section.start_z / 0.2),
                layer_end=int(section.end_z / 0.2),
            )
            assert zone.value in [220, 210, 200]

    def test_full_flow_tower_workflow(self):
        generator = TowerGenerator()
        config = FlowTowerConfig(start_flow=90, end_flow=110, step_flow=10)
        tower = generator.generate_flow_tower(config)

        assert len(tower.sections) == 3

        for section in tower.sections:
            zone = TowerZone(
                section_index=section.index,
                value=section.value,
                center_x_mm=section.center_x,
                center_y_mm=section.center_y,
                width_mm=section.size,
                height_mm=section.height,
                layer_start=int(section.start_z / 0.2),
                layer_end=int(section.end_z / 0.2),
            )
            assert zone.value in [90, 100, 110]

    def test_proposal_summary(self):
        proposal = ConfigProposal(
            changes=[
                ConfigChange(
                    section="extruder",
                    option="target_temperature",
                    value=200,
                    parameter_type=ConfigParameterType.TEMPERATURE,
                    description="Optimal temp",
                )
            ],
            source="temperature_tower_calibration",
            timestamp="2024-01-01T00:00:00",
            confidence=0.9,
        )

        summary = proposal.summary

        assert "Configuration Proposal" in summary
        assert "Confidence: 90.0%" in summary
        assert "[extruder] target_temperature = 200" in summary


class TestSurfaceQuality:
    def test_quality_ordering(self):
        assert SurfaceQuality.EXCELLENT
        assert SurfaceQuality.GOOD
        assert SurfaceQuality.FAIR
        assert SurfaceQuality.POOR


class TestStringingLevel:
    def test_level_ordering(self):
        assert StringingLevel.NONE
        assert StringingLevel.MINIMAL
        assert StringingLevel.MODERATE
        assert StringingLevel.SEVERE
