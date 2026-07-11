"""
Printer memo repository for Sensing Voxel.

Key-value storage for printer attributes that cannot be read from Klipper
(e.g. nozzle diameter, bed surface type, filament brand). Stored locally
in SQLite, not on the printer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sensing_voxel.storage.database import Database


@dataclass
class MemoRecord:
    printer_id: str
    key: str
    value: str
    updated_at: datetime | None = None


class PrinterMemoRepository:
    def __init__(self, db: Database):
        self.db = db

    def get(self, printer_id: str, key: str) -> MemoRecord | None:
        with self.db.connection() as conn:
            cursor = conn.execute(
                "SELECT printer_id, key, value, updated_at "
                "FROM printer_memos WHERE printer_id = ? AND key = ?",
                (printer_id, key),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return MemoRecord(
                printer_id=row["printer_id"],
                key=row["key"],
                value=row["value"],
                updated_at=row["updated_at"],
            )

    def get_all(self, printer_id: str) -> list[MemoRecord]:
        with self.db.connection() as conn:
            cursor = conn.execute(
                "SELECT printer_id, key, value, updated_at "
                "FROM printer_memos WHERE printer_id = ? ORDER BY key",
                (printer_id,),
            )
            return [
                MemoRecord(
                    printer_id=row["printer_id"],
                    key=row["key"],
                    value=row["value"],
                    updated_at=row["updated_at"],
                )
                for row in cursor.fetchall()
            ]

    def get_as_dict(self, printer_id: str) -> dict[str, str]:
        return {r.key: r.value for r in self.get_all(printer_id)}

    def upsert(self, printer_id: str, key: str, value: str) -> MemoRecord:
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO printer_memos (printer_id, key, value, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(printer_id, key) DO UPDATE SET value = excluded.value, "
                "updated_at = excluded.updated_at",
                (printer_id, key, value, datetime.now().isoformat()),
            )
        return MemoRecord(printer_id=printer_id, key=key, value=value)

    def delete(self, printer_id: str, key: str) -> bool:
        with self.db.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM printer_memos WHERE printer_id = ? AND key = ?",
                (printer_id, key),
            )
            return cursor.rowcount > 0

    def list_printers_with_memos(self) -> list[str]:
        with self.db.connection() as conn:
            cursor = conn.execute(
                "SELECT DISTINCT printer_id FROM printer_memos ORDER BY printer_id"
            )
            return [row["printer_id"] for row in cursor.fetchall()]
