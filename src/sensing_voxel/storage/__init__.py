"""
Storage module for Sensing Voxel

SQLite-based persistence for job history and printer memos.
"""

from sensing_voxel.storage.database import (
    Database,
    DatabaseConfig,
    DatabaseConnectionError,
    DatabaseError,
    DatabaseSchemaError,
    create_database,
    get_default_db_path,
)
from sensing_voxel.storage.job_repository import (
    JobRecord,
    JobRepository,
    JobStatus,
)
from sensing_voxel.storage.printer_memo_repository import (
    MemoRecord,
    PrinterMemoRepository,
)

__all__ = [
    "Database",
    "DatabaseConfig",
    "DatabaseError",
    "DatabaseConnectionError",
    "DatabaseSchemaError",
    "create_database",
    "get_default_db_path",
    "JobRepository",
    "JobRecord",
    "JobStatus",
    "PrinterMemoRepository",
    "MemoRecord",
]
