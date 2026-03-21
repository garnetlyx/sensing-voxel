"""
Queue manager for multi-printer job scheduling

Manages job queue, printer registration, and job lifecycle.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
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
    QueueStats,
    QueuedJob,
    ScheduleWindow,
    generate_queue_job_id,
)

logger = logging.getLogger(__name__)


@dataclass
class PrinterSlot:
    """Internal printer registration"""

    capabilities: PrinterCapabilities
    load: PrinterLoad = field(init=False)
    adapter: Any = field(default=None, repr=False)
    registered_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        self.load = PrinterLoad(printer_id=self.capabilities.printer_id)


class QueueManager:
    """
    Multi-printer queue manager

    Handles job queueing, printer registration, and assignment coordination.
    """

    def __init__(
        self,
        config: QueueConfiguration | None = None,
        scheduler: Any | None = None,
    ):
        self.config = config or QueueConfiguration()
        self.scheduler = scheduler

        # Job queue: ordered by priority_score
        self._jobs: dict[str, QueuedJob] = {}
        self._job_order: list[str] = []  # Ordered queue_job_ids

        # Printer registry
        self._printers: dict[str, PrinterSlot] = {}

        # Callbacks
        self._status_callbacks: list[
            Callable[[QueuedJob, QueueJobStatus, QueueJobStatus], Any]
        ] = []
        self._assignment_callbacks: list[Callable[[str, str], Any]] = []

        # Background task
        self._running = False
        self._scheduler_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    # --- Printer Registration ---

    async def register_printer(
        self,
        capabilities: PrinterCapabilities,
        adapter: Any | None = None,
    ) -> bool:
        """Register a printer with the queue"""
        async with self._lock:
            if capabilities.printer_id in self._printers:
                logger.warning(f"Printer {capabilities.printer_id} already registered")
                return False

            slot = PrinterSlot(capabilities=capabilities, adapter=adapter)
            self._printers[capabilities.printer_id] = slot

            logger.info(f"Registered printer: {capabilities.printer_id}")
            return True

    async def unregister_printer(self, printer_id: str) -> bool:
        """Unregister a printer from the queue"""
        async with self._lock:
            if printer_id not in self._printers:
                return False

            # Cancel any running jobs
            slot = self._printers[printer_id]
            if slot.load.current_job_id:
                job = self._jobs.get(slot.load.current_job_id)
                if job:
                    await self._update_job_status(job, QueueJobStatus.CANCELLED)

            del self._printers[printer_id]
            logger.info(f"Unregistered printer: {printer_id}")
            return True

    async def update_printer_status(
        self,
        printer_id: str,
        status: PrinterStatus,
        current_job_id: str | None = None,
    ) -> bool:
        """Update printer status from adapter"""
        async with self._lock:
            if printer_id not in self._printers:
                return False

            slot = self._printers[printer_id]
            slot.load.status = status
            slot.load.current_job_id = current_job_id
            slot.load.last_activity = datetime.now()
            slot.last_heartbeat = datetime.now()

            # Update job status if transitioning
            if current_job_id and current_job_id in self._jobs:
                job = self._jobs[current_job_id]
                new_status = self._map_printer_status(status)
                if new_status != job.status:
                    await self._update_job_status(job, new_status)

            return True

    def _map_printer_status(self, printer_status: PrinterStatus) -> QueueJobStatus:
        """Map printer status to job status"""
        mapping = {
            PrinterStatus.PRINTING: QueueJobStatus.RUNNING,
            PrinterStatus.PAUSED: QueueJobStatus.RUNNING,
            PrinterStatus.BUSY: QueueJobStatus.STARTING,
            PrinterStatus.ERROR: QueueJobStatus.FAILED,
        }
        return mapping.get(printer_status, QueueJobStatus.WAITING_FOR_PRINTER)

    async def update_printer_heartbeat(self, printer_id: str) -> bool:
        """Update printer heartbeat timestamp"""
        async with self._lock:
            if printer_id not in self._printers:
                return False
            self._printers[printer_id].last_heartbeat = datetime.now()
            return True

    # --- Job Management ---

    async def submit_job(
        self,
        job_context: Any,
        priority: QueuePriority = QueuePriority.NORMAL,
        deadline: datetime | None = None,
        filament_type: str | None = None,
        user_id: str | None = None,
        tags: set[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> QueuedJob | None:
        """Submit a new job to the queue"""
        async with self._lock:
            if len(self._jobs) >= self.config.max_queue_size:
                logger.error("Queue is full, cannot accept new job")
                return None

            queue_job_id = generate_queue_job_id()
            job = QueuedJob(
                queue_job_id=queue_job_id,
                job_context=job_context,
                priority=priority,
                deadline=deadline,
                filament_type=filament_type,
                user_id=user_id,
                tags=tags or set(),
                metadata=metadata or {},
            )

            self._jobs[queue_job_id] = job
            self._insert_job_in_order(queue_job_id)

            await self._update_job_status(job, QueueJobStatus.PENDING)

            logger.info(f"Submitted job {queue_job_id} with priority {priority.name}")
            return job

    def _insert_job_in_order(self, queue_job_id: str):
        """Insert job into ordered list based on priority score"""
        job = self._jobs[queue_job_id]
        score = job.priority_score

        # Insert in order of ascending priority score
        for i, existing_id in enumerate(self._job_order):
            existing = self._jobs[existing_id]
            if score < existing.priority_score:
                self._job_order.insert(i, queue_job_id)
                return
        self._job_order.append(queue_job_id)

    async def cancel_job(self, queue_job_id: str) -> bool:
        """Cancel a queued job"""
        async with self._lock:
            if queue_job_id not in self._jobs:
                return False

            job = self._jobs[queue_job_id]
            if job.status in {
                QueueJobStatus.COMPLETED,
                QueueJobStatus.FAILED,
                QueueJobStatus.CANCELLED,
            }:
                return False

            await self._update_job_status(job, QueueJobStatus.CANCELLED)

            if queue_job_id in self._job_order:
                self._job_order.remove(queue_job_id)

            logger.info(f"Cancelled job: {queue_job_id}")
            return True

    async def remove_job(self, queue_job_id: str) -> bool:
        """Remove a job from the queue (completed/failed/cancelled only)"""
        async with self._lock:
            if queue_job_id not in self._jobs:
                return False

            job = self._jobs[queue_job_id]
            if job.status not in {
                QueueJobStatus.COMPLETED,
                QueueJobStatus.FAILED,
                QueueJobStatus.CANCELLED,
            }:
                return False

            del self._jobs[queue_job_id]
            if queue_job_id in self._job_order:
                self._job_order.remove(queue_job_id)

            return True

    async def get_job(self, queue_job_id: str) -> QueuedJob | None:
        """Get a job by ID"""
        async with self._lock:
            return self._jobs.get(queue_job_id)

    async def list_jobs(
        self,
        status: QueueJobStatus | None = None,
        user_id: str | None = None,
        printer_id: str | None = None,
    ) -> list[QueuedJob]:
        """List jobs with optional filters"""
        async with self._lock:
            jobs = list(self._jobs.values())

            if status:
                jobs = [j for j in jobs if j.status == status]
            if user_id:
                jobs = [j for j in jobs if j.user_id == user_id]
            if printer_id:
                jobs = [j for j in jobs if j.assigned_printer_id == printer_id]

            return sorted(jobs, key=lambda j: j.priority_score)

    # --- Assignment Logic ---

    async def assign_job(
        self,
        queue_job_id: str,
        printer_id: str | None = None,
        strategy: AssignmentStrategy | None = None,
    ) -> AssignmentResult:
        """Assign a job to a printer"""
        async with self._lock:
            if queue_job_id not in self._jobs:
                return AssignmentResult(
                    success=False,
                    job_id=queue_job_id,
                    reason="Job not found",
                )

            job = self._jobs[queue_job_id]

            if job.status not in {QueueJobStatus.PENDING, QueueJobStatus.SCHEDULED}:
                return AssignmentResult(
                    success=False,
                    job_id=queue_job_id,
                    reason=f"Job is in {job.status.name} state",
                )

            # Use provided printer_id or find one
            if printer_id:
                if printer_id not in self._printers:
                    return AssignmentResult(
                        success=False,
                        job_id=queue_job_id,
                        reason=f"Printer {printer_id} not found",
                    )
            else:
                printer_id = await self._find_best_printer(job, strategy)
                if not printer_id:
                    return AssignmentResult(
                        success=False,
                        job_id=queue_job_id,
                        reason="No suitable printer available",
                    )

            # Assign the job
            job.assigned_printer_id = printer_id
            job.scheduled_at = datetime.now()
            await self._update_job_status(job, QueueJobStatus.SCHEDULED)

            # Update printer load
            slot = self._printers[printer_id]
            slot.load.queue_position = self._get_queue_position(queue_job_id)

            # Notify callbacks
            await self._notify_assignment(queue_job_id, printer_id)

            logger.info(f"Assigned job {queue_job_id} to printer {printer_id}")
            return AssignmentResult(
                success=True,
                job_id=queue_job_id,
                printer_id=printer_id,
                strategy_used=strategy or self.config.assignment_strategy,
            )

    async def _find_best_printer(
        self,
        job: QueuedJob,
        strategy: AssignmentStrategy | None = None,
    ) -> str | None:
        """Find the best printer for a job"""
        strategy = strategy or self.config.assignment_strategy

        # Filter available printers
        available = [
            pid
            for pid, slot in self._printers.items()
            if slot.load.status == PrinterStatus.AVAILABLE
        ]

        if not available:
            return None

        if strategy == AssignmentStrategy.FIRST_AVAILABLE:
            return available[0]

        if strategy == AssignmentStrategy.FILAMENT_MATCH and job.filament_type:
            for pid in available:
                slot = self._printers[pid]
                if slot.capabilities.supports_filament(job.filament_type):
                    return pid
            return available[0]

        if strategy == AssignmentStrategy.LEAST_LOADED:
            return min(
                available,
                key=lambda pid: self._printers[pid].load.utilization_percent,
            )

        if strategy == AssignmentStrategy.ROUND_ROBIN:
            return available[0]

        if strategy == AssignmentStrategy.OPTIMAL:
            return self._optimal_assignment(job, available)

        return available[0]

    def _optimal_assignment(self, job: QueuedJob, available: list[str]) -> str | None:
        """Optimal assignment considering multiple factors"""
        if not available:
            return None

        scores: dict[str, float] = {}

        for pid in available:
            slot = self._printers[pid]
            score = 0.0

            # Prefer lower utilization
            score -= slot.load.utilization_percent * 0.3

            # Prefer filament match
            if job.filament_type and slot.capabilities.supports_filament(job.filament_type):
                score += 50

            # Prefer fewer completed jobs today (load balancing)
            score -= slot.load.jobs_completed_today * 10

            # Prefer printers with longer idle time
            if slot.load.last_activity:
                idle_time = (datetime.now() - slot.load.last_activity).total_seconds()
                score += idle_time / 60  # Per minute

            scores[pid] = score

        return max(scores, key=lambda pid: scores[pid])

    def _get_queue_position(self, queue_job_id: str) -> int:
        """Get position in ordered queue"""
        try:
            return self._job_order.index(queue_job_id)
        except ValueError:
            return len(self._job_order)

    # --- Callbacks ---

    def on_job_status_change(
        self,
        callback: Callable[[QueuedJob, QueueJobStatus, QueueJobStatus], Any],
    ):
        """Register callback for job status changes"""
        self._status_callbacks.append(callback)

    def remove_status_callback(
        self,
        callback: Callable[[QueuedJob, QueueJobStatus, QueueJobStatus], Any],
    ):
        """Remove status callback"""
        if callback in self._status_callbacks:
            self._status_callbacks.remove(callback)

    def on_job_assignment(self, callback: Callable[[str, str], Any]):
        """Register callback for job assignments"""
        self._assignment_callbacks.append(callback)

    def remove_assignment_callback(self, callback: Callable[[str, str], Any]):
        """Remove assignment callback"""
        if callback in self._assignment_callbacks:
            self._assignment_callbacks.remove(callback)

    async def _update_job_status(self, job: QueuedJob, new_status: QueueJobStatus):
        """Update job status and notify callbacks"""
        old_status = job.status
        job.status = new_status

        if new_status == QueueJobStatus.RUNNING and not job.started_at:
            job.started_at = datetime.now()
        elif new_status in {
            QueueJobStatus.COMPLETED,
            QueueJobStatus.FAILED,
            QueueJobStatus.CANCELLED,
        }:
            job.completed_at = datetime.now()

        for callback in self._status_callbacks:
            try:
                await callback(job, old_status, new_status)
            except Exception as e:
                logger.error(f"Status callback error: {e}")

    async def _notify_assignment(self, job_id: str, printer_id: str):
        """Notify assignment callbacks"""
        for callback in self._assignment_callbacks:
            try:
                await callback(job_id, printer_id)
            except Exception as e:
                logger.error(f"Assignment callback error: {e}")

    # --- Background Scheduling ---

    async def start(self):
        """Start background scheduler"""
        if self._running:
            return

        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Queue manager started")

    async def stop(self):
        """Stop background scheduler"""
        self._running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
            self._scheduler_task = None
        logger.info("Queue manager stopped")

    async def _scheduler_loop(self):
        """Background scheduler loop"""
        while self._running:
            try:
                await self._process_scheduling()
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(5)

    async def _process_scheduling(self):
        """Process scheduled jobs and assignments"""
        async with self._lock:
            # Update priority scores and reorder
            self._reorder_queue()

            # Check for jobs that need assignment
            if self.config.auto_assignment:
                pending = [
                    jid
                    for jid in self._job_order
                    if self._jobs[jid].status == QueueJobStatus.PENDING
                ]

                for queue_job_id in pending:
                    job = self._jobs[queue_job_id]

                    # Skip if in maintenance window
                    if self._is_maintenance_window():
                        continue

                    # Try to assign
                    available = [
                        pid
                        for pid, slot in self._printers.items()
                        if slot.load.status == PrinterStatus.AVAILABLE
                    ]

                    if available:
                        printer_id = self._find_best_printer_for_job(job, available)
                        if printer_id:
                            await self._do_assignment(queue_job_id, printer_id)

            # Check for stale jobs
            await self._check_stale_jobs()

            # Check for offline printers
            await self._check_printer_health()

    def _reorder_queue(self):
        """Reorder queue based on current priority scores"""
        self._job_order.sort(key=lambda jid: self._jobs[jid].priority_score)

        # Update queue positions
        for position, queue_job_id in enumerate(self._job_order):
            job = self._jobs[queue_job_id]
            if job.assigned_printer_id:
                slot = self._printers.get(job.assigned_printer_id)
                if slot:
                    slot.load.queue_position = position

    def _find_best_printer_for_job(self, job: QueuedJob, available: list[str]) -> str | None:
        """Find best printer for a specific job"""
        return self._optimal_assignment(job, available)

    async def _do_assignment(self, queue_job_id: str, printer_id: str):
        """Perform actual assignment"""
        job = self._jobs[queue_job_id]
        job.assigned_printer_id = printer_id
        job.scheduled_at = datetime.now()
        await self._update_job_status(job, QueueJobStatus.SCHEDULED)
        await self._notify_assignment(queue_job_id, printer_id)

    async def _check_stale_jobs(self):
        """Check for jobs that have been waiting too long"""
        now = datetime.now()
        for job in self._jobs.values():
            if job.status != QueueJobStatus.PENDING:
                continue

            if job.wait_time.total_seconds() > self.config.job_timeout_seconds:
                await self._update_job_status(job, QueueJobStatus.FAILED)
                job.metadata["failure_reason"] = "Timeout: exceeded max wait time"
                logger.warning(f"Job {job.queue_job_id} timed out")

    async def _check_printer_health(self):
        """Check printer heartbeats and mark offline if stale"""
        now = datetime.now()
        timeout = timedelta(seconds=self.config.printer_timeout_seconds)

        for pid, slot in self._printers.items():
            if now - slot.last_heartbeat > timeout:
                if slot.load.status != PrinterStatus.OFFLINE:
                    slot.load.status = PrinterStatus.OFFLINE
                    logger.warning(f"Printer {pid} marked offline due to missing heartbeat")

    def _is_maintenance_window(self) -> bool:
        """Check if currently in maintenance window"""
        if not self.config.maintenance_window:
            return False
        return self.config.maintenance_window.contains(datetime.now())

    # --- Statistics ---

    async def get_stats(self) -> QueueStats:
        """Get current queue statistics"""
        async with self._lock:
            stats = QueueStats(last_updated=datetime.now())

            for job in self._jobs.values():
                stats.total_jobs += 1
                if job.status == QueueJobStatus.PENDING:
                    stats.pending_jobs += 1
                elif job.status == QueueJobStatus.RUNNING:
                    stats.running_jobs += 1
                elif job.status == QueueJobStatus.COMPLETED:
                    stats.completed_jobs += 1
                elif job.status == QueueJobStatus.FAILED:
                    stats.failed_jobs += 1

            # Calculate averages
            completed = [j for j in self._jobs.values() if j.status == QueueJobStatus.COMPLETED]
            if completed:
                total_wait = sum(
                    (j.started_at - j.scheduled_at).total_seconds()
                    for j in completed
                    if j.started_at and j.scheduled_at
                )
                total_process = sum(
                    (j.completed_at - j.started_at).total_seconds()
                    for j in completed
                    if j.completed_at and j.started_at
                )
                stats.avg_wait_time = timedelta(seconds=total_wait / len(completed))
                stats.avg_processing_time = timedelta(seconds=total_process / len(completed))

            # Printer utilization
            for pid, slot in self._printers.items():
                stats.printer_utilization[pid] = slot.load.utilization_percent

            return stats

    # --- Utility ---

    async def get_printers(self) -> list[PrinterCapabilities]:
        """Get all registered printer capabilities"""
        async with self._lock:
            return [slot.capabilities for slot in self._printers.values()]

    async def get_printer_load(self, printer_id: str) -> PrinterLoad | None:
        """Get printer load information"""
        async with self._lock:
            slot = self._printers.get(printer_id)
            return slot.load if slot else None

    async def update_job_progress(
        self,
        queue_job_id: str,
        progress: float,
    ) -> bool:
        """Update job progress"""
        async with self._lock:
            if queue_job_id not in self._jobs:
                return False

            job = self._jobs[queue_job_id]
            job.metadata["progress"] = progress

            # Update printer utilization if running
            if job.assigned_printer_id and job.status == QueueJobStatus.RUNNING:
                slot = self._printers.get(job.assigned_printer_id)
                if slot:
                    slot.load.utilization_percent = progress * 100

            return True

    async def complete_job(self, queue_job_id: str, success: bool = True) -> bool:
        """Mark a job as completed or failed"""
        async with self._lock:
            if queue_job_id not in self._jobs:
                return False

            job = self._jobs[queue_job_id]
            new_status = QueueJobStatus.COMPLETED if success else QueueJobStatus.FAILED
            await self._update_job_status(job, new_status)

            # Update printer stats
            if job.assigned_printer_id:
                slot = self._printers.get(job.assigned_printer_id)
                if slot:
                    slot.load.jobs_completed_today += 1
                    if job.started_at:
                        duration = datetime.now() - job.started_at
                        slot.load.total_print_time_today += duration
                    slot.load.current_job_id = None
                    slot.load.utilization_percent = 0.0

            return True
