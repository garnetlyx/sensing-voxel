"""Simple job queue for multi-printer scheduling."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    WAITING = auto()
    ASSIGNED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class PrinterInfo:
    printer_id: str
    bed_width_mm: float = 220.0
    bed_depth_mm: float = 220.0
    supported_filaments: set[str] = field(default_factory=lambda: {"PLA", "PETG", "ABS"})
    is_available: bool = True

    def supports_filament(self, filament_type: str) -> bool:
        return filament_type.upper() in {f.upper() for f in self.supported_filaments}


@dataclass
class Job:
    job_id: str
    file: Path
    printer_id: str | None = None
    priority: int = 0
    status: JobStatus = JobStatus.WAITING
    filament_type: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


class JobQueue:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._printers: dict[str, PrinterInfo] = {}
        self._counter: int = 0

    def _generate_job_id(self) -> str:
        self._counter += 1
        return f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._counter}"

    def add_job(
        self,
        file: str | Path,
        priority: int = 0,
        filament_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Job:
        job_id = self._generate_job_id()
        job = Job(
            job_id=job_id,
            file=Path(file),
            priority=priority,
            filament_type=filament_type,
            metadata=metadata or {},
        )
        self._jobs[job_id] = job
        logger.info(f"Added job {job_id} to queue: {file}")
        return job

    def get_job(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list_jobs(
        self,
        status: JobStatus | None = None,
        printer_id: str | None = None,
    ) -> list[Job]:
        jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        if printer_id:
            jobs = [j for j in jobs if j.printer_id == printer_id]
        return sorted(jobs, key=lambda j: (-j.priority, j.created_at))

    def cancel_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False
        if job.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}:
            return False
        job.status = JobStatus.CANCELLED
        logger.info(f"Cancelled job: {job_id}")
        return True

    def complete_job(self, job_id: str, success: bool = True) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False
        job.status = JobStatus.COMPLETED if success else JobStatus.FAILED
        logger.info(f"Job {job_id} marked as {job.status.name}")
        return True

    def register_printer(
        self,
        printer_id: str,
        bed_width_mm: float = 220.0,
        bed_depth_mm: float = 220.0,
        supported_filaments: set[str] | None = None,
    ) -> None:
        self._printers[printer_id] = PrinterInfo(
            printer_id=printer_id,
            bed_width_mm=bed_width_mm,
            bed_depth_mm=bed_depth_mm,
            supported_filaments=supported_filaments or {"PLA", "PETG", "ABS"},
            is_available=True,
        )
        logger.info(f"Registered printer: {printer_id}")

    def set_printer_available(self, printer_id: str) -> bool:
        if printer_id not in self._printers:
            return False
        self._printers[printer_id].is_available = True
        return True

    def set_printer_busy(self, printer_id: str) -> bool:
        if printer_id not in self._printers:
            return False
        self._printers[printer_id].is_available = False
        return True

    def get_printer_for_job(self, job: Job) -> str | None:
        available = [p for p in self._printers.values() if p.is_available]
        if not available:
            return None

        best_printer = None
        best_score = float("-inf")

        for printer in available:
            score = 0.0
            if job.filament_type and printer.supports_filament(job.filament_type):
                score += 100
            score -= (
                len(
                    [
                        j
                        for j in self._jobs.values()
                        if j.printer_id == printer.printer_id and j.status == JobStatus.RUNNING
                    ]
                )
                * 50
            )

            if score > best_score:
                best_score = score
                best_printer = printer.printer_id

        return best_printer

    def assign_next_job(self, printer_id: str) -> Job | None:
        if printer_id not in self._printers:
            return None

        waiting = [j for j in self._jobs.values() if j.status == JobStatus.WAITING]
        if not waiting:
            return None

        waiting.sort(key=lambda j: (-j.priority, j.created_at))
        printer = self._printers[printer_id]

        for job in waiting:
            if job.filament_type and not printer.supports_filament(job.filament_type):
                continue
            job.printer_id = printer_id
            job.status = JobStatus.ASSIGNED
            logger.info(f"Assigned job {job.job_id} to printer {printer_id}")
            return job

        return None

    def assign_job_to_printer(self, job_id: str, printer_id: str) -> bool:
        if printer_id not in self._printers:
            return False
        job = self._jobs.get(job_id)
        if not job or job.status != JobStatus.WAITING:
            return False
        job.printer_id = printer_id
        job.status = JobStatus.ASSIGNED
        logger.info(f"Manually assigned job {job_id} to printer {printer_id}")
        return True

    def start_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job or job.status != JobStatus.ASSIGNED:
            return False
        job.status = JobStatus.RUNNING
        self.set_printer_busy(job.printer_id)
        logger.info(f"Started job: {job_id}")
        return True

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_jobs": len(self._jobs),
            "waiting": sum(1 for j in self._jobs.values() if j.status == JobStatus.WAITING),
            "assigned": sum(1 for j in self._jobs.values() if j.status == JobStatus.ASSIGNED),
            "running": sum(1 for j in self._jobs.values() if j.status == JobStatus.RUNNING),
            "completed": sum(1 for j in self._jobs.values() if j.status == JobStatus.COMPLETED),
            "failed": sum(1 for j in self._jobs.values() if j.status == JobStatus.FAILED),
        }

    def get_next_job(self) -> Job | None:
        waiting = [j for j in self._jobs.values() if j.status == JobStatus.WAITING]
        if not waiting:
            return None
        waiting.sort(key=lambda j: (-j.priority, j.created_at))
        return waiting[0]

    def clear_completed(self) -> int:
        to_remove = [
            job_id
            for job_id, job in self._jobs.items()
            if job.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
        ]
        for job_id in to_remove:
            del self._jobs[job_id]
        logger.info(f"Cleared {len(to_remove)} completed jobs from queue")
        return len(to_remove)
