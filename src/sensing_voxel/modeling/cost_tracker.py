"""
Cost tracking for image-to-3D generations.

Tracks API costs per user and job for billing and monitoring.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from sensing_voxel.modeling.image_to_3d import ImageTo3DProvider

logger = logging.getLogger(__name__)


@dataclass
class GenerationCost:
    """Cost information for a single generation."""

    user_id: int | str
    job_id: str
    provider: ImageTo3DProvider
    timestamp: datetime
    success: bool
    estimated_cost_usd: float = 0.0
    api_calls: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "user_id": str(self.user_id),
            "job_id": self.job_id,
            "provider": self.provider.name,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "estimated_cost_usd": self.estimated_cost_usd,
            "api_calls": self.api_calls,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenerationCost:
        """Create from dictionary."""
        return cls(
            user_id=data["user_id"],
            job_id=data["job_id"],
            provider=ImageTo3DProvider[data["provider"]],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            success=data["success"],
            estimated_cost_usd=data.get("estimated_cost_usd", 0.0),
            api_calls=data.get("api_calls", 1),
            metadata=data.get("metadata", {}),
        )


class CostTracker:
    """Track costs for image-to-3D generations."""

    PROVIDER_COSTS: dict[ImageTo3DProvider, float] = {
        ImageTo3DProvider.TRIPOSR: 0.02,
        ImageTo3DProvider.MESHY: 0.01,
        ImageTo3DProvider.TRIPO3D: 0.015,
    }

    def __init__(
        self,
        storage_path: Path | str | None = None,
        provider_costs: dict[ImageTo3DProvider, float] | None = None,
    ):
        self.storage_path = Path(storage_path) if storage_path else None
        self.provider_costs = provider_costs or self.PROVIDER_COSTS.copy()
        self._costs: list[GenerationCost] = []
        self._load_costs()

    def _load_costs(self) -> None:
        """Load existing costs from storage."""
        if not self.storage_path or not self.storage_path.exists():
            return

        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    try:
                        self._costs.append(GenerationCost.from_dict(item))
                    except (KeyError, ValueError) as e:
                        logger.warning(f"Failed to load cost entry: {e}")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load cost data: {e}")

    def _save_costs(self) -> None:
        """Save costs to storage."""
        if not self.storage_path:
            return

        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump([c.to_dict() for c in self._costs], f, indent=2)
        except OSError as e:
            logger.warning(f"Failed to save cost data: {e}")

    async def record_generation(
        self,
        user_id: int | str,
        job_id: str,
        provider: ImageTo3DProvider,
        success: bool,
        metadata: dict[str, Any] | None = None,
    ) -> GenerationCost:
        """Record a generation and its cost."""
        base_cost = self.provider_costs.get(provider, 0.0)

        if metadata and metadata.get("multiview"):
            base_cost *= 1.5

        cost = GenerationCost(
            user_id=user_id,
            job_id=job_id,
            provider=provider,
            timestamp=datetime.now(),
            success=success,
            estimated_cost_usd=base_cost,
            metadata=metadata or {},
        )

        self._costs.append(cost)
        self._save_costs()

        logger.info(
            f"Recorded generation cost: {cost.estimated_cost_usd:.4f} USD "
            f"for user {user_id}, job {job_id}"
        )

        return cost

    def get_user_costs(
        self,
        user_id: int | str,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[GenerationCost]:
        """Get all costs for a specific user."""
        costs = [c for c in self._costs if str(c.user_id) == str(user_id)]

        if since:
            costs = [c for c in costs if c.timestamp >= since]
        if until:
            costs = [c for c in costs if c.timestamp <= until]

        return sorted(costs, key=lambda c: c.timestamp, reverse=True)

    def get_job_cost(self, job_id: str) -> GenerationCost | None:
        """Get cost for a specific job."""
        for cost in reversed(self._costs):
            if cost.job_id == job_id:
                return cost
        return None

    def get_total_cost(
        self,
        user_id: int | str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> float:
        """Get total cost for a user or all users."""
        costs = self._costs

        if user_id is not None:
            costs = [c for c in costs if str(c.user_id) == str(user_id)]
        if since:
            costs = [c for c in costs if c.timestamp >= since]
        if until:
            costs = [c for c in costs if c.timestamp <= until]

        return sum(c.estimated_cost_usd for c in costs)

    def get_provider_usage(
        self,
        provider: ImageTo3DProvider,
        since: datetime | None = None,
    ) -> dict[str, Any]:
        """Get usage statistics for a specific provider."""
        costs = [c for c in self._costs if c.provider == provider]

        if since:
            costs = [c for c in costs if c.timestamp >= since]

        successful = [c for c in costs if c.success]
        failed = [c for c in costs if not c.success]

        return {
            "total_calls": len(costs),
            "successful_calls": len(successful),
            "failed_calls": len(failed),
            "total_cost": sum(c.estimated_cost_usd for c in costs),
            "success_rate": len(successful) / len(costs) if costs else 0.0,
        }

    def get_user_summary(self, user_id: int | str) -> dict[str, Any]:
        """Get cost summary for a user."""
        user_costs = self.get_user_costs(user_id)

        if not user_costs:
            return {
                "user_id": str(user_id),
                "total_generations": 0,
                "total_cost": 0.0,
                "providers_used": [],
            }

        providers_used = list(set(c.provider.name for c in user_costs))
        total_cost = sum(c.estimated_cost_usd for c in user_costs)
        successful = sum(1 for c in user_costs if c.success)

        return {
            "user_id": str(user_id),
            "total_generations": len(user_costs),
            "successful_generations": successful,
            "failed_generations": len(user_costs) - successful,
            "total_cost": total_cost,
            "average_cost": total_cost / len(user_costs) if user_costs else 0.0,
            "providers_used": providers_used,
        }

    def clear_history(self, user_id: int | str | None = None) -> int:
        """Clear cost history for a user or all users."""
        if user_id is None:
            count = len(self._costs)
            self._costs.clear()
        else:
            user_str = str(user_id)
            original_count = len(self._costs)
            self._costs = [c for c in self._costs if str(c.user_id) != user_str]
            count = original_count - len(self._costs)

        self._save_costs()
        return count
