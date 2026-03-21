"""
Scheduling algorithms for multi-printer job assignment

Provides various scheduling strategies: load balancing, priority-based,
filament matching, and deadline-aware scheduling.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sensing_voxel.queue.models import (
    AssignmentResult,
    AssignmentStrategy,
    PrinterCapabilities,
    PrinterLoad,
    PrinterStatus,
    QueueConfiguration,
    QueueJobStatus,
    QueuePriority,
    QueuedJob,
    ScheduleWindow,
)

logger = logging.getLogger(__name__)


@dataclass
class ScheduleScore:
    """Score for a printer-job assignment"""

    printer_id: str
    score: float
    breakdown: dict[str, float]


class Scheduler:
    """
    Multi-strategy job scheduler

    Provides various scheduling algorithms for job-printer assignment.
    """

    def __init__(self, config: QueueConfiguration | None = None):
        self.config = config or QueueConfiguration()

    def schedule(
        self,
        job: QueuedJob,
        printers: dict[str, tuple[PrinterCapabilities, PrinterLoad]],
        strategy: AssignmentStrategy | None = None,
    ) -> AssignmentResult:
        """
        Schedule a job to an available printer using specified strategy.

        Args:
            job: The job to schedule
            printers: Dict of printer_id -> (capabilities, load)
            strategy: Scheduling strategy to use

        Returns:
            AssignmentResult with success status and assigned printer
        """
        strategy = strategy or self.config.assignment_strategy

        available = self._filter_available_printers(printers)
        if not available:
            return AssignmentResult(
                success=False,
                job_id=job.queue_job_id,
                reason="No available printers",
            )

        if strategy == AssignmentStrategy.FIRST_AVAILABLE:
            return self._first_available(job, available)

        if strategy == AssignmentStrategy.ROUND_ROBIN:
            return self._round_robin(job, available)

        if strategy == AssignmentStrategy.LEAST_LOADED:
            return self._least_loaded(job, available)

        if strategy == AssignmentStrategy.FILAMENT_MATCH:
            return self._filament_match(job, available)

        if strategy == AssignmentStrategy.BED_SIZE_MATCH:
            return self._bed_size_match(job, available)

        if strategy == AssignmentStrategy.OPTIMAL:
            return self._optimal(job, available)

        return self._first_available(job, available)

    def _filter_available_printers(
        self,
        printers: dict[str, tuple[PrinterCapabilities, PrinterLoad]],
    ) -> dict[str, tuple[PrinterCapabilities, PrinterLoad]]:
        """Filter to only available printers"""
        return {
            pid: (cap, load)
            for pid, (cap, load) in printers.items()
            if load.status == PrinterStatus.AVAILABLE
        }

    def _first_available(
        self,
        job: QueuedJob,
        available: dict[str, tuple[PrinterCapabilities, PrinterLoad]],
    ) -> AssignmentResult:
        """Assign to first available printer"""
        printer_id = next(iter(available.keys()))
        return AssignmentResult(
            success=True,
            job_id=job.queue_job_id,
            printer_id=printer_id,
            strategy_used=AssignmentStrategy.FIRST_AVAILABLE,
        )

    def _round_robin(
        self,
        job: QueuedJob,
        available: dict[str, tuple[PrinterCapabilities, PrinterLoad]],
    ) -> AssignmentResult:
        """Round-robin assignment based on completed job count"""
        printer_id = min(
            available.keys(),
            key=lambda pid: available[pid][1].jobs_completed_today,
        )
        return AssignmentResult(
            success=True,
            job_id=job.queue_job_id,
            printer_id=printer_id,
            strategy_used=AssignmentStrategy.ROUND_ROBIN,
        )

    def _least_loaded(
        self,
        job: QueuedJob,
        available: dict[str, tuple[PrinterCapabilities, PrinterLoad]],
    ) -> AssignmentResult:
        """Assign to least utilized printer"""
        printer_id = min(
            available.keys(),
            key=lambda pid: available[pid][1].utilization_percent,
        )
        return AssignmentResult(
            success=True,
            job_id=job.queue_job_id,
            printer_id=printer_id,
            strategy_used=AssignmentStrategy.LEAST_LOADED,
        )

    def _filament_match(
        self,
        job: QueuedJob,
        available: dict[str, tuple[PrinterCapabilities, PrinterLoad]],
    ) -> AssignmentResult:
        """Assign to printer matching filament type"""
        if not job.filament_type:
            return self._first_available(job, available)

        matching = [
            pid for pid, (cap, _) in available.items() if cap.supports_filament(job.filament_type)
        ]

        if matching:
            printer_id = matching[0]
            return AssignmentResult(
                success=True,
                job_id=job.queue_job_id,
                printer_id=printer_id,
                strategy_used=AssignmentStrategy.FILAMENT_MATCH,
            )

        return AssignmentResult(
            success=False,
            job_id=job.queue_job_id,
            reason=f"No printer supports filament: {job.filament_type}",
        )

    def _bed_size_match(
        self,
        job: QueuedJob,
        available: dict[str, tuple[PrinterCapabilities, PrinterLoad]],
    ) -> AssignmentResult:
        """Assign to printer with adequate bed size"""
        if not job.slicing_result or not job.slicing_result.first_layer_bounds:
            return self._first_available(job, available)

        bounds = job.slicing_result.first_layer_bounds

        matching = [pid for pid, (cap, _) in available.items() if cap.can_fit_bounds(bounds)]

        if matching:
            # Prefer smallest printer that fits (bed efficiency)
            printer_id = min(
                matching,
                key=lambda pid: available[pid][0].bed_area_mm2,
            )
            return AssignmentResult(
                success=True,
                job_id=job.queue_job_id,
                printer_id=printer_id,
                strategy_used=AssignmentStrategy.BED_SIZE_MATCH,
            )

        return AssignmentResult(
            success=False,
            job_id=job.queue_job_id,
            reason="No printer can fit the job bounds",
        )

    def _optimal(
        self,
        job: QueuedJob,
        available: dict[str, tuple[PrinterCapabilities, PrinterLoad]],
    ) -> AssignmentResult:
        """Optimal assignment with multi-factor scoring"""
        scores = self._calculate_scores(job, available)

        if not scores:
            return AssignmentResult(
                success=False,
                job_id=job.queue_job_id,
                reason="No suitable printer found",
            )

        best = max(scores, key=lambda s: s.score)
        return AssignmentResult(
            success=True,
            job_id=job.queue_job_id,
            printer_id=best.printer_id,
            strategy_used=AssignmentStrategy.OPTIMAL,
        )

    def _calculate_scores(
        self,
        job: QueuedJob,
        available: dict[str, tuple[PrinterCapabilities, PrinterLoad]],
    ) -> list[ScheduleScore]:
        """Calculate scores for each printer"""
        scores: list[ScheduleScore] = []

        for pid, (cap, load) in available.items():
            breakdown: dict[str, float] = {}
            score = 100.0

            # Utilization penalty (lower is better)
            utilization_penalty = load.utilization_percent * 0.5
            score -= utilization_penalty
            breakdown["utilization"] = -utilization_penalty

            # Idle time bonus
            if load.last_activity:
                idle_minutes = (datetime.now() - load.last_activity).total_seconds() / 60
                idle_bonus = min(idle_minutes * 2, 50)  # Cap at 50 points
                score += idle_bonus
                breakdown["idle_time"] = idle_bonus

            # Jobs completed today penalty (load balancing)
            jobs_penalty = load.jobs_completed_today * 5
            score -= jobs_penalty
            breakdown["jobs_today"] = -jobs_penalty

            # Filament match bonus
            if job.filament_type and cap.supports_filament(job.filament_type):
                filament_bonus = 30
                score += filament_bonus
                breakdown["filament_match"] = filament_bonus

            # Bed size efficiency
            if job.slicing_result and job.slicing_result.first_layer_bounds:
                bounds = job.slicing_result.first_layer_bounds
                if cap.can_fit_bounds(bounds):
                    area_ratio = bounds.area / cap.bed_area_mm2
                    efficiency_bonus = (1 - area_ratio) * 20
                    score += efficiency_bonus
                    breakdown["bed_efficiency"] = efficiency_bonus
                else:
                    continue  # Skip printers that cannot fit the job

            # Temperature compatibility
            nozzle_temp_ok = True
            bed_temp_ok = True

            if job.required_nozzle_temp and cap.max_nozzle_temp:
                if job.required_nozzle_temp > cap.max_nozzle_temp:
                    nozzle_temp_ok = False

            if job.required_bed_temp and cap.max_bed_temp:
                if job.required_bed_temp > cap.max_bed_temp:
                    bed_temp_ok = False

            if not nozzle_temp_ok or not bed_temp_ok:
                continue  # Skip printers with insufficient temperature

            # Error count penalty
            error_penalty = load.error_count * 10
            score -= error_penalty
            breakdown["error_count"] = -error_penalty

            # Maintenance penalty
            if load.maintenance_due:
                maintenance_penalty = 20
                score -= maintenance_penalty
                breakdown["maintenance"] = -maintenance_penalty

            scores.append(ScheduleScore(printer_id=pid, score=score, breakdown=breakdown))

        return scores


class PriorityScheduler:
    """Priority-based scheduling with deadline awareness"""

    def __init__(self, config: QueueConfiguration | None = None):
        self.config = config or QueueConfiguration()

    def sort_by_priority(self, jobs: list[QueuedJob]) -> list[QueuedJob]:
        """Sort jobs by dynamic priority score"""
        return sorted(jobs, key=lambda j: j.priority_score)

    def calculate_earliest_deadline(
        self,
        jobs: list[QueuedJob],
    ) -> list[QueuedJob]:
        """EDF (Earliest Deadline First) ordering"""

        def edf_key(job: QueuedJob) -> tuple:
            if job.deadline:
                return (0, job.deadline)
            return (1, datetime.max)

        return sorted(jobs, key=edf_key)

    def prioritize_with_deadlines(
        self,
        jobs: list[QueuedJob],
    ) -> list[QueuedJob]:
        """
        Combine priority and deadline scheduling.

        Critical/high priority jobs are ordered first.
        Among same priority, deadline-aware ordering is applied.
        """
        by_priority: dict[QueuePriority, list[QueuedJob]] = {p: [] for p in QueuePriority}

        for job in jobs:
            by_priority[job.priority].append(job)

        result: list[QueuedJob] = []
        for priority in [
            QueuePriority.CRITICAL,
            QueuePriority.HIGH,
            QueuePriority.NORMAL,
            QueuePriority.LOW,
            QueuePriority.BACKGROUND,
        ]:
            priority_jobs = by_priority[priority]
            if priority_jobs:
                # Within same priority, sort by deadline if exists
                sorted_priority = self.calculate_earliest_deadline(priority_jobs)
                result.extend(sorted_priority)

        return result

    def get_urgency_score(self, job: QueuedJob) -> int:
        """Calculate urgency score (higher = more urgent)"""
        if job.priority == QueuePriority.CRITICAL:
            return 1000

        score = (QueuePriority.NORMAL.value - job.priority.value) * 100

        if job.deadline:
            time_to_deadline = (job.deadline - datetime.now()).total_seconds()
            if time_to_deadline < 0:
                score += 500  # Overdue
            elif time_to_deadline < 3600:  # Less than 1 hour
                score += 300
            elif time_to_deadline < 86400:  # Less than 24 hours
                score += 100

        return score


class LoadBalancer:
    """Load balancing across multiple printers"""

    def __init__(self, config: QueueConfiguration | None = None):
        self.config = config or QueueConfiguration()
        self._printer_weights: dict[str, float] = {}

    def calculate_utilization(
        self,
        printer_id: str,
        printers: dict[str, tuple[PrinterCapabilities, PrinterLoad]],
        jobs: dict[str, QueuedJob],
    ) -> float:
        """Calculate printer utilization (0.0 to 1.0)"""
        if printer_id not in printers:
            return 0.0

        _, load = printers[printer_id]
        base_utilization = load.utilization_percent / 100.0

        # Add pending jobs contribution
        pending_jobs = [
            j
            for j in jobs.values()
            if j.assigned_printer_id == printer_id
            and j.status in {QueueJobStatus.PENDING, QueueJobStatus.SCHEDULED}
        ]

        # Estimate additional load from pending jobs
        additional_load = len(pending_jobs) * 0.1  # 10% per pending job

        return min(base_utilization + additional_load, 1.0)

    def get_balanced_assignment(
        self,
        job: QueuedJob,
        printers: dict[str, tuple[PrinterCapabilities, PrinterLoad]],
        jobs: dict[str, QueuedJob],
    ) -> str | None:
        """Get printer assignment with load balancing"""
        available = {
            pid: data for pid, data in printers.items() if data[1].status == PrinterStatus.AVAILABLE
        }

        if not available:
            return None

        # Calculate utilization for each printer
        utilizations = {pid: self.calculate_utilization(pid, printers, jobs) for pid in available}

        # Pick printer with lowest utilization
        return min(utilizations.keys(), key=lambda pid: utilizations[pid])

    def rebalance_queue(
        self,
        printers: dict[str, tuple[PrinterCapabilities, PrinterLoad]],
        jobs: dict[str, QueuedJob],
    ) -> list[tuple[str, str]]:
        """
        Suggest reassignments to balance load.

        Returns list of (job_id, new_printer_id) tuples.
        """
        if not self.config.enable_load_balancing:
            return []

        # Calculate utilization for each printer
        utilizations: dict[str, float] = {}
        job_counts: dict[str, int] = {}

        for pid in printers:
            utilizations[pid] = self.calculate_utilization(pid, printers, jobs)
            job_counts[pid] = sum(
                1
                for j in jobs.values()
                if j.assigned_printer_id == pid and j.status == QueueJobStatus.PENDING
            )

        if not utilizations:
            return []

        avg_utilization = sum(utilizations.values()) / len(utilizations)
        threshold = 0.2  # Rebalance if difference > 20%

        overloaded = [
            pid for pid, util in utilizations.items() if util > avg_utilization + threshold
        ]
        underloaded = [
            pid for pid, util in utilizations.items() if util < avg_utilization - threshold
        ]

        reassignments: list[tuple[str, str]] = []

        for overloaded_pid in overloaded:
            # Find jobs to move from overloaded to underloaded
            overloaded_jobs = [
                j
                for j in jobs.values()
                if j.assigned_printer_id == overloaded_pid and j.status == QueueJobStatus.PENDING
            ]

            for job in overloaded_jobs:
                if underloaded:
                    # Move to least loaded printer
                    new_printer = min(underloaded, key=lambda pid: utilizations[pid])
                    reassignments.append((job.queue_job_id, new_printer))

                    # Update counts
                    utilizations[overloaded_pid] -= 0.1
                    utilizations[new_printer] += 0.1

                    if utilizations[overloaded_pid] <= avg_utilization:
                        break

        return reassignments


class DeadlineAwareScheduler:
    """Scheduler that respects deadlines and SLAs"""

    def __init__(self, config: QueueConfiguration | None = None):
        self.config = config or QueueConfiguration()

    def can_meet_deadline(
        self,
        job: QueuedJob,
        printer_id: str,
        printers: dict[str, tuple[PrinterCapabilities, PrinterLoad]],
        current_time: datetime | None = None,
    ) -> bool:
        """Check if job can meet deadline on specified printer"""
        if not job.deadline:
            return True

        if not job.estimated_duration:
            return True  # Assume can meet if no estimate

        current_time = current_time or datetime.now()

        # Get estimated completion time
        estimated_completion = self._estimate_completion(job, printer_id, printers, current_time)

        return estimated_completion <= job.deadline

    def _estimate_completion(
        self,
        job: QueuedJob,
        printer_id: str,
        printers: dict[str, tuple[PrinterCapabilities, PrinterLoad]],
        current_time: datetime,
    ) -> datetime:
        """Estimate when job would complete on printer"""
        if printer_id not in printers:
            return current_time + (job.estimated_duration or timedelta(hours=1))

        _, load = printers[printer_id]

        # Current load on printer
        if load.status == PrinterStatus.PRINTING and load.estimated_completion:
            start_time = load.estimated_completion
        else:
            start_time = current_time

        # Add time for pending jobs ahead in queue
        return start_time + (job.estimated_duration or timedelta(hours=1))

    def find_viable_printers(
        self,
        job: QueuedJob,
        printers: dict[str, tuple[PrinterCapabilities, PrinterLoad]],
    ) -> list[str]:
        """Find printers that can meet job deadline"""
        viable = []

        for pid, (cap, load) in printers.items():
            if load.status not in {PrinterStatus.AVAILABLE, PrinterStatus.PRINTING}:
                continue

            if self.can_meet_deadline(job, pid, printers):
                viable.append(pid)

        return viable

    def schedule_with_sla(
        self,
        job: QueuedJob,
        printers: dict[str, tuple[PrinterCapabilities, PrinterLoad]],
    ) -> AssignmentResult:
        """Schedule job ensuring SLA/deadline is met"""
        viable = self.find_viable_printers(job, printers)

        if not viable:
            if job.deadline:
                return AssignmentResult(
                    success=False,
                    job_id=job.queue_job_id,
                    reason="No printer can meet deadline",
                )
            return AssignmentResult(
                success=False,
                job_id=job.queue_job_id,
                reason="No available printers",
            )

        # Among viable printers, pick least loaded
        best_printer = min(
            viable,
            key=lambda pid: printers[pid][1].utilization_percent,
        )

        return AssignmentResult(
            success=True,
            job_id=job.queue_job_id,
            printer_id=best_printer,
            strategy_used=AssignmentStrategy.OPTIMAL,
        )


class CompositeScheduler:
    """Composite scheduler combining multiple strategies"""

    def __init__(self, config: QueueConfiguration | None = None):
        self.config = config or QueueConfiguration()
        self.base_scheduler = Scheduler(config)
        self.priority_scheduler = PriorityScheduler(config)
        self.load_balancer = LoadBalancer(config)
        self.deadline_scheduler = DeadlineAwareScheduler(config)

    def schedule_batch(
        self,
        jobs: list[QueuedJob],
        printers: dict[str, tuple[PrinterCapabilities, PrinterLoad]],
    ) -> list[AssignmentResult]:
        """Schedule multiple jobs optimally"""
        # Sort by priority with deadline awareness
        sorted_jobs = self.priority_scheduler.prioritize_with_deadlines(jobs)

        results: list[AssignmentResult] = []
        assigned_printers: set[str] = set()

        for job in sorted_jobs:
            # Filter out already assigned printers for this batch
            available_printers = {
                pid: data
                for pid, data in printers.items()
                if pid not in assigned_printers and data[1].status == PrinterStatus.AVAILABLE
            }

            if not available_printers:
                results.append(
                    AssignmentResult(
                        success=False,
                        job_id=job.queue_job_id,
                        reason="No available printers",
                    )
                )
                continue

            # Try deadline-aware first
            if job.deadline:
                result = self.deadline_scheduler.schedule_with_sla(job, available_printers)
                if result.success:
                    assigned_printers.add(result.printer_id)
                    results.append(result)
                    continue

            # Fall back to optimal scheduling
            result = self.base_scheduler.schedule(
                job, available_printers, AssignmentStrategy.OPTIMAL
            )
            if result.success:
                assigned_printers.add(result.printer_id)

            results.append(result)

        return results

    def get_schedule_recommendations(
        self,
        jobs: list[QueuedJob],
        printers: dict[str, tuple[PrinterCapabilities, PrinterLoad]],
    ) -> dict[str, Any]:
        """Get scheduling recommendations and insights"""
        recommendations: dict[str, Any] = {
            "total_jobs": len(jobs),
            "available_printers": len(
                [
                    pid
                    for pid, (_, load) in printers.items()
                    if load.status == PrinterStatus.AVAILABLE
                ]
            ),
            "can_schedule_all": False,
            "suggested_order": [],
            "deadline_risks": [],
            "load_balance_score": 0.0,
        }

        # Check if all jobs can be scheduled
        sorted_jobs = self.priority_scheduler.prioritize_with_deadlines(jobs)
        available_count = recommendations["available_printers"]

        recommendations["can_schedule_all"] = len(jobs) <= available_count

        # Identify deadline risks
        for job in jobs:
            if job.deadline:
                viable = self.deadline_scheduler.find_viable_printers(job, printers)
                if not viable:
                    recommendations["deadline_risks"].append(
                        {
                            "job_id": job.queue_job_id,
                            "reason": "No printer can meet deadline",
                        }
                    )

        # Calculate load balance score
        if printers:
            utilizations = [load.utilization_percent for _, load in printers.values()]
            if utilizations:
                avg = sum(utilizations) / len(utilizations)
                variance = sum((u - avg) ** 2 for u in utilizations) / len(utilizations)
                recommendations["load_balance_score"] = max(0, 100 - variance)

        return recommendations
