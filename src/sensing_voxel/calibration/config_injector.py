"""
Moonraker SAVE_CONFIG integration for calibration automation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import aiohttp

from sensing_voxel.adapters.klipper import KlipperAdapter
from sensing_voxel.core.models import CommandResult, CommandResultStatus

logger = logging.getLogger(__name__)


class ConfigParameterType(Enum):
    TEMPERATURE = auto()
    FLOW = auto()
    PRESSURE_ADVANCE = auto()
    RETRACTION = auto()


class SaveConfigError(Exception):
    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


@dataclass
class ConfigChange:
    section: str
    option: str
    value: str | int | float
    parameter_type: ConfigParameterType
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "section": self.section,
            "option": self.option,
            "value": self.value,
            "type": self.parameter_type.name,
            "description": self.description,
        }


@dataclass
class ConfigProposal:
    changes: list[ConfigChange]
    source: str
    timestamp: str
    confidence: float
    scores: dict[str, Any] | None = None

    @property
    def summary(self) -> str:
        lines = [f"Configuration Proposal ({self.source})"]
        lines.append(f"Confidence: {self.confidence:.1%}")
        lines.append("")
        for change in self.changes:
            desc = f" - {change.description}" if change.description else ""
            lines.append(f"  [{change.section}] {change.option} = {change.value}{desc}")
        return "\n".join(lines)


class ConfigInjector:
    def __init__(self, adapter: KlipperAdapter):
        self.adapter = adapter

    async def propose_changes(
        self,
        optimal_value: int,
        calibration_type: str,
        scores: dict[str, Any],
    ) -> ConfigProposal:
        if calibration_type == "temperature":
            changes = [
                ConfigChange(
                    section="extruder",
                    option="target_temperature",
                    value=optimal_value,
                    parameter_type=ConfigParameterType.TEMPERATURE,
                    description=f"Optimal temperature from tower calibration",
                )
            ]
        elif calibration_type == "flow":
            changes = [
                ConfigChange(
                    section="extruder",
                    option="flow_multiplier",
                    value=optimal_value,
                    parameter_type=ConfigParameterType.FLOW,
                    description=f"Optimal flow percentage from tower calibration",
                )
            ]
        else:
            raise ValueError(f"unknown calibration type: {calibration_type}")

        from datetime import datetime

        return ConfigProposal(
            changes=changes,
            source=f"{calibration_type}_tower_calibration",
            timestamp=datetime.now().isoformat(),
            confidence=scores.get("best_score", 0.0),
            scores=scores,
        )

    async def inject_config(
        self,
        proposal: ConfigProposal,
        dry_run: bool = False,
    ) -> CommandResult:
        if dry_run:
            logger.info(f"DRY RUN: Would apply {len(proposal.changes)} config changes")
            for change in proposal.changes:
                logger.info(f"  [{change.section}] {change.option} = {change.value}")
            return CommandResult(
                CommandResultStatus.SUCCESS,
                data={"dry_run": True, "changes": [c.to_dict() for c in proposal.changes]},
            )

        for change in proposal.changes:
            result = await self._set_config_value(change)
            if result.status != CommandResultStatus.SUCCESS:
                logger.error(f"Failed to set {change.section}.{change.option}: {result.message}")
                return result

        save_result = await self._save_config()
        if save_result.status != CommandResultStatus.SUCCESS:
            logger.error(f"Failed to save config: {save_result.message}")
            return save_result

        logger.info(f"Successfully applied {len(proposal.changes)} config changes")
        return CommandResult(
            CommandResultStatus.SUCCESS,
            data={"changes_applied": len(proposal.changes)},
        )

    async def inject_temperature(self, temperature: int, dry_run: bool = False) -> CommandResult:
        proposal = await self.propose_changes(
            optimal_value=temperature,
            calibration_type="temperature",
            scores={"best_score": 0.9},
        )
        return await self.inject_config(proposal, dry_run)

    async def inject_flow(self, flow_percent: int, dry_run: bool = False) -> CommandResult:
        proposal = await self.propose_changes(
            optimal_value=flow_percent,
            calibration_type="flow",
            scores={"best_score": 0.9},
        )
        return await self.inject_config(proposal, dry_run)

    async def _set_config_value(self, change: ConfigChange) -> CommandResult:
        if change.parameter_type == ConfigParameterType.TEMPERATURE:
            gcode = f"M104 S{int(change.value)}"
        elif change.parameter_type == ConfigParameterType.FLOW:
            gcode = f"M221 S{int(change.value)}"
        else:
            return CommandResult(
                CommandResultStatus.FAILURE,
                message=f"Unsupported parameter type: {change.parameter_type}",
            )

        result = await self.adapter.run_gcode(gcode)

        if result.status == CommandResultStatus.SUCCESS:
            logger.info(f"Set {change.section}.{change.option} = {change.value}")

        return result

    async def _save_config(self) -> CommandResult:
        return await self.adapter.run_gcode("SAVE_CONFIG")

    async def validate_connection(self) -> bool:
        try:
            return await self.adapter.is_connected()
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False

    async def get_current_config(self, section: str | None = None) -> dict[str, Any]:
        result = await self.adapter.run_gcode("QUERY_CONFIG")

        if result.status != CommandResultStatus.SUCCESS:
            logger.error(f"Failed to query config: {result.message}")
            return {}

        return result.data or {}
