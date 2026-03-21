"""
Calibration repository for Sensing Voxel

CRUD operations and query API for calibration records.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Any

from sensing_voxel.storage.database import Database


class CalibrationType(Enum):
    CAMERA_HOMOGRAPHY = auto()
    EXTRUDER = auto()
    BED_LEVEL = auto()
    Z_OFFSET = auto()
    TEMPERATURE = auto()
    SHAPER = auto()


@dataclass
class CalibrationRecord:
    id: int | None
    calibration_id: str
    printer_id: str
    calibration_type: CalibrationType
    values: dict[str, Any]
    performed_at: datetime
    results: dict[str, Any] | None
    notes: str | None
    created_at: datetime | None
    updated_at: datetime | None


class CalibrationRepositoryError(Exception):
    pass


class CalibrationRepository:
    def __init__(self, db: Database):
        self.db = db

    def _type_to_db(self, cal_type: CalibrationType) -> str:
        return cal_type.name.lower()

    def _type_from_db(self, type_str: str) -> CalibrationType:
        return CalibrationType[type_str.upper()]

    def create(
        self,
        calibration_id: str,
        printer_id: str,
        calibration_type: CalibrationType,
        values: dict[str, Any],
        performed_at: datetime | None = None,
        results: dict[str, Any] | None = None,
        notes: str | None = None,
    ) -> CalibrationRecord:
        if performed_at is None:
            performed_at = datetime.now()

        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO calibrations
                (calibration_id, printer_id, calibration_type, values_json,
                 performed_at, results_json, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    calibration_id,
                    printer_id,
                    self._type_to_db(calibration_type),
                    json.dumps(values),
                    performed_at,
                    json.dumps(results) if results else None,
                    notes,
                ),
            )

            return CalibrationRecord(
                id=cursor.lastrowid,
                calibration_id=calibration_id,
                printer_id=printer_id,
                calibration_type=calibration_type,
                values=values,
                performed_at=performed_at,
                results=results,
                notes=notes,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

    def get_by_id(self, calibration_id: str) -> CalibrationRecord | None:
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM calibrations WHERE calibration_id = ?",
                (calibration_id,),
            ).fetchone()

            if not row:
                return None

            return self._row_to_record(row)

    def get_by_db_id(self, db_id: int) -> CalibrationRecord | None:
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM calibrations WHERE id = ?",
                (db_id,),
            ).fetchone()

            if not row:
                return None

            return self._row_to_record(row)

    def _row_to_record(self, row: Any) -> CalibrationRecord:
        return CalibrationRecord(
            id=row["id"],
            calibration_id=row["calibration_id"],
            printer_id=row["printer_id"],
            calibration_type=self._type_from_db(row["calibration_type"]),
            values=json.loads(row["values_json"]),
            performed_at=row["performed_at"],
            results=json.loads(row["results_json"]) if row["results_json"] else None,
            notes=row["notes"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def update(
        self,
        calibration_id: str,
        values: dict[str, Any] | None = None,
        results: dict[str, Any] | None = None,
        notes: str | None = None,
    ) -> bool:
        with self.db.transaction() as conn:
            updates = []
            params: list[Any] = []

            if values is not None:
                updates.append("values_json = ?")
                params.append(json.dumps(values))

            if results is not None:
                updates.append("results_json = ?")
                params.append(json.dumps(results))

            if notes is not None:
                updates.append("notes = ?")
                params.append(notes)

            if not updates:
                return False

            updates.append("updated_at = ?")
            params.append(datetime.now())
            params.append(calibration_id)

            query = f"UPDATE calibrations SET {', '.join(updates)} WHERE calibration_id = ?"
            cursor = conn.execute(query, params)
            return cursor.rowcount > 0

    def list_calibrations(
        self,
        printer_id: str | None = None,
        calibration_type: CalibrationType | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[CalibrationRecord]:
        query = "SELECT * FROM calibrations WHERE 1=1"
        params: list[Any] = []

        if printer_id:
            query += " AND printer_id = ?"
            params.append(printer_id)

        if calibration_type:
            query += " AND calibration_type = ?"
            params.append(self._type_to_db(calibration_type))

        query += " ORDER BY performed_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self.db.connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_record(row) for row in rows]

    def get_latest_calibration(
        self,
        printer_id: str,
        calibration_type: CalibrationType,
    ) -> CalibrationRecord | None:
        with self.db.connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM calibrations
                WHERE printer_id = ? AND calibration_type = ?
                ORDER BY performed_at DESC
                LIMIT 1
                """,
                (printer_id, self._type_to_db(calibration_type)),
            ).fetchone()

            if not row:
                return None

            return self._row_to_record(row)

    def get_calibration_history(
        self,
        printer_id: str,
        calibration_type: CalibrationType,
        days: int = 90,
    ) -> list[CalibrationRecord]:
        since = datetime.now() - __import__("datetime").timedelta(days=days)

        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM calibrations
                WHERE printer_id = ? AND calibration_type = ? AND performed_at >= ?
                ORDER BY performed_at DESC
                """,
                (printer_id, self._type_to_db(calibration_type), since),
            ).fetchall()

            return [self._row_to_record(row) for row in rows]

    def get_calibration_stats(
        self,
        printer_id: str | None = None,
        days: int = 30,
    ) -> dict[str, Any]:
        since = datetime.now() - __import__("datetime").timedelta(days=days)

        query = """
            SELECT calibration_type, COUNT(*) as count
            FROM calibrations
            WHERE performed_at >= ?
        """
        params: list[Any] = [since]

        if printer_id:
            query += " AND printer_id = ?"
            params.append(printer_id)

        query += " GROUP BY calibration_type"

        with self.db.connection() as conn:
            rows = conn.execute(query, params).fetchall()

        stats = {
            "total_calibrations": 0,
            "by_type": {},
            "period_days": days,
        }

        for row in rows:
            count = row["count"]
            cal_type = row["calibration_type"]
            stats["total_calibrations"] += count
            stats["by_type"][cal_type] = count

        return stats

    def delete(self, calibration_id: str) -> bool:
        with self.db.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM calibrations WHERE calibration_id = ?",
                (calibration_id,),
            )
            return cursor.rowcount > 0

    def delete_old_calibrations(self, days: int = 365) -> int:
        cutoff = datetime.now() - __import__("datetime").timedelta(days=days)

        with self.db.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM calibrations WHERE performed_at < ?",
                (cutoff,),
            )
            return cursor.rowcount

    def has_valid_calibration(
        self,
        printer_id: str,
        calibration_type: CalibrationType,
        max_age_days: int = 30,
    ) -> bool:
        since = datetime.now() - __import__("datetime").timedelta(days=max_age_days)

        with self.db.connection() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) as count FROM calibrations
                WHERE printer_id = ? AND calibration_type = ? AND performed_at >= ?
                """,
                (printer_id, self._type_to_db(calibration_type), since),
            ).fetchone()

            return row["count"] > 0 if row else False
