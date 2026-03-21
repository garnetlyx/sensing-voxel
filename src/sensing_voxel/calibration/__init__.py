"""
Closed-loop calibration automation module.

This module provides automated temperature and flow tower calibration
with vision-based scoring and config injection via Moonraker.

Components:
    - tower_generator: Generate G-code for temperature/flow towers
    - visual_scoring: Vision-based quality scoring of tower zones
    - config_injector: Moonraker SAVE_CONFIG integration

Example workflow:
    1. Generate tower G-code using TowerGenerator
    2. Print the tower and capture images of each zone
    3. Score zones using VisualScorer
    4. Get optimal values recommendation
    5. Inject config via ConfigInjector on user approval
"""

from sensing_voxel.calibration.config_injector import ConfigInjector, SaveConfigError
from sensing_voxel.calibration.tower_generator import (
    FlowTowerConfig,
    TowerGenerator,
    TempTowerConfig,
)
from sensing_voxel.calibration.visual_scoring import (
    ScoringConfig,
    TowerZone,
    VisualScorer,
    ZoneScore,
)

__all__ = [
    # Tower Generator
    "TowerGenerator",
    "TempTowerConfig",
    "FlowTowerConfig",
    # Visual Scoring
    "VisualScorer",
    "TowerZone",
    "ZoneScore",
    "ScoringConfig",
    # Config Injector
    "ConfigInjector",
    "SaveConfigError",
]
