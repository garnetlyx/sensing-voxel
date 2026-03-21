"""
Job repository for Sensing Voxel

CRUD operations and query API for job history.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Any

from sensing_voxel.storage.database import Database


class JobStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class JobRecord:
    id: int | None
    job_id: str
    printer_id: str
    file_name: str
    status: JobStatus
    started_at: datetime
    completed_at: datetime | None
    duration_seconds: float | None
    filament_used_mm: float | None
    notes: str | None
    metadata: dict[str, Any]
    created_at: datetime | None
    updated_at: datetime | None


class JobRepositoryError(Exception):
    pass


class JobRepository:
    def __init__(self, db: Database):
        self.db = db

    def _status_to_db(self, status: JobStatus) -> str:
        return status.name.lower()

    def _status_from_db(self, status_str: str) -> JobStatus:
        return JobStatus[status_str.upper()]

    def create(
        self,
        job_id: str,
        printer_id: str,
        file_name: str,
        status: JobStatus = JobStatus.PENDING,
        started_at: datetime | None = None,
        notes: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> JobRecord:
        if started_at is None:
            started_at = datetime.now()

        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO jobs (job_id, printer_id, file_name, status, started_at, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (job_id, printer_id, file_name, self._status_to_db(status), started_at, notes),
            )

            job_db_id = cursor.lastrowid

            if metadata:
                for key, value in metadata.items():
                    conn.execute(
                        "INSERT INTO job_metadata (job_id, key, value) VALUES (?, ?, ?)",
                        (job_id, key, str(value)),
                    )

            return JobRecord(
                id=job_db_id,
                job_id=job_id,
                printer_id=printer_id,
                file_name=file_name,
                status=status,
                started_at=started_at,
                completed_at=None,
                duration_seconds=None,
                filament_used_mm=None,
                notes=notes,
                metadata=metadata or {},
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

    def get_by_id(self, job_id: str) -> JobRecord | None:
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()

            if not row:
                return None

            metadata = self._get_metadata(conn, job_id)
            return self._row_to_record(row, metadata)

    def get_by_db_id(self, db_id: int) -> JobRecord | None:
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE id = ?",
                (db_id,),
            ).fetchone()

            if not row:
                return None

            metadata = self._get_metadata(conn, row["job_id"])
            return self._row_to_record(row, metadata)

    def _get_metadata(self, conn, job_id: str) -> dict[str, Any]:
        rows = conn.execute(
            "SELECT key, value FROM job_metadata WHERE job_id = ?",
            (job_id,),
        ).fetchall()
        return {row["key"]: row["value"] for row in rows}

    def _row_to_record(self, row: Any, metadata: dict[str, Any]) -> JobRecord:
        return JobRecord(
            id=row["id"],
            job_id=row["job_id"],
            printer_id=row["printer_id"],
            file_name=row["file_name"],
            status=self._status_from_db(row["status"]),
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            duration_seconds=row["duration_seconds"],
            filament_used_mm=row["filament_used_mm"],
            notes=row["notes"],
            metadata=metadata,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        completed_at: datetime | None = None,
        duration_seconds: float | None = None,
        filament_used_mm: float | None = None,
        notes: str | None = None,
    ) -> bool:
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs
                SET status = ?, completed_at = ?, duration_seconds = ?,
                    filament_used_mm = ?, notes = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (
                    self._status_to_db(status),
                    completed_at,
                    duration_seconds,
                    filament_used_mm,
                    notes,
                    datetime.now(),
                    job_id,
                ),
            )
            return cursor.rowcount > 0

    def complete(
        self,
        job_id: str,
        duration_seconds: float | None = None,
        filament_used_mm: float | None = None,
        notes: str | None = None,
    ) -> bool:
        return self.update_status(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            completed_at=datetime.now(),
            duration_seconds=duration_seconds,
            filament_used_mm=filament_used_mm,
            notes=notes,
        )

    def fail(
        self,
        job_id: str,
        duration_seconds: float | None = None,
        notes: str | None = None,
    ) -> bool:
        return self.update_status(
            job_id=job_id,
            status=JobStatus.FAILED,
            completed_at=datetime.now(),
            duration_seconds=duration_seconds,
            notes=notes,
        )

    def cancel(
        self,
        job_id: str,
        duration_seconds: float | None = None,
        notes: str | None = None,
    ) -> bool:
        return self.update_status(
            job_id=job_id,
            status=JobStatus.CANCELLED,
            completed_at=datetime.now(),
            duration_seconds=duration_seconds,
            notes=notes,
        )

    def list_jobs(
        self,
        printer_id: str | None = None,
        status: JobStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[JobRecord]:
        query = "SELECT * FROM jobs WHERE 1=1"
        params: list[Any] = []

        if printer_id:
            query += " AND printer_id = ?"
            params.append(printer_id)

        if status:
            query += " AND status = ?"
            params.append(self._status_to_db(status))

        query += " ORDER BY started_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self.db.connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [
                self._row_to_record(row, self._get_metadata(conn, row["job_id"])) for row in rows
            ]

    def get_recent_jobs(
        self,
        hours: int = 24,
        printer_id: str | None = None,
    ) -> list[JobRecord]:
        since = datetime.now() - __import__("datetime").timedelta(hours=hours)

        query = "SELECT * FROM jobs WHERE started_at >= ?"
        params: list[Any] = [since]

        if printer_id:
            query += " AND printer_id = ?"
            params.append(printer_id)

        query += " ORDER BY started_at DESC"

        with self.db.connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [
                self._row_to_record(row, self._get_metadata(conn, row["job_id"])) for row in rows
            ]

    def get_job_stats(
        self,
        printer_id: str | None = None,
        days: int = 30,
    ) -> dict[str, Any]:
        since = datetime.now() - __import__("datetime").timedelta(days=days)

        query = """
            SELECT
                status,
                COUNT(*) as count,
                AVG(duration_seconds) as avg_duration,
                SUM(filament_used_mm) as total_filament
            FROM jobs
            WHERE started_at >= ?
        """
        params: list[Any] = [since]

        if printer_id:
            query += " AND printer_id = ?"
            params.append(printer_id)

        query += " GROUP BY status"

        with self.db.connection() as conn:
            rows = conn.execute(query, params).fetchall()

        stats = {
            "total_jobs": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "success_rate": 0.0,
            "avg_duration_seconds": 0.0,
            "total_filament_mm": 0.0,
            "period_days": days,
        }

        total_duration = 0.0
        duration_count = 0

        for row in rows:
            count = row["count"]
            status = row["status"]
            avg_dur = row["avg_duration"] or 0
            filament = row["total_filament"] or 0

            stats["total_jobs"] += count
            stats["total_filament_mm"] += filament

            if status == "completed":
                stats["completed"] = count
            elif status == "failed":
                stats["failed"] = count
            elif status == "cancelled":
                stats["cancelled"] = count

            if avg_dur:
                total_duration += avg_dur * count
                duration_count += count

        if stats["total_jobs"] > 0:
            stats["success_rate"] = stats["completed"] / stats["total_jobs"]

        if duration_count > 0:
            stats["avg_duration_seconds"] = total_duration / duration_count

        return stats

    def delete(self, job_id: str) -> bool:
        with self.db.transaction() as conn:
            cursor = conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            return cursor.rowcount > 0

    def delete_old_jobs(self, days: int = 90) -> int:
        cutoff = datetime.now() - __import__("datetime").timedelta(days=days)

        with self.db.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM jobs WHERE started_at < ?",
                (cutoff,),
            )
            return cursor.rowcount
