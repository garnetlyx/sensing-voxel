"""
SQLite database module for Sensing Voxel

Provides database connection management, schema initialization,
and utilities for job history and calibration storage.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class DatabaseConfig:
    """Database configuration"""

    db_path: Path
    auto_migrate: bool = True


class DatabaseError(Exception):
    """Base database error"""

    pass


class DatabaseConnectionError(DatabaseError):
    """Database connection error"""

    pass


class DatabaseSchemaError(DatabaseError):
    """Database schema error"""

    pass


class Database:
    """
    SQLite database manager for Sensing Voxel

    Manages connections, schema, and provides query utilities.
    Thread-safe for single-threaded asyncio usage.
    """

    SCHEMA_VERSION = 1

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Ensure database directory exists"""
        self.config.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """
        Get database connection context manager

        Yields:
            sqlite3.Connection: Database connection
        """
        conn = None
        try:
            conn = sqlite3.connect(
                self.config.db_path,
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
                timeout=30.0,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            yield conn
        except sqlite3.Error as e:
            raise DatabaseConnectionError(f"Failed to connect to database: {e}") from e
        finally:
            if conn:
                conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """
        Get transaction context manager

        Automatically commits on success, rolls back on error.

        Yields:
            sqlite3.Connection: Database connection
        """
        with self.connection() as conn:
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def initialize(self) -> None:
        """Initialize database schema"""
        with self.transaction() as conn:
            self._create_schema(conn)
            self._set_version(conn)

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        """Create database schema"""
        # Jobs table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT UNIQUE NOT NULL,
                printer_id TEXT NOT NULL,
                file_name TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TIMESTAMP NOT NULL,
                completed_at TIMESTAMP,
                duration_seconds REAL,
                filament_used_mm REAL,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Job metadata table (flexible key-value storage)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS job_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
                key TEXT NOT NULL,
                value TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(job_id, key)
            )
            """
        )

        # Calibrations table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS calibrations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                calibration_id TEXT UNIQUE NOT NULL,
                printer_id TEXT NOT NULL,
                calibration_type TEXT NOT NULL,
                values_json TEXT NOT NULL,
                performed_at TIMESTAMP NOT NULL,
                results_json TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Create indexes for common queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_printer ON jobs(printer_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_started_at ON jobs(started_at)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_calibrations_printer ON calibrations(printer_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_calibrations_type ON calibrations(calibration_type)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_calibrations_performed ON calibrations(performed_at)"
        )

        # Printer memos table (SVA notes about physical printer attributes)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS printer_memos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                printer_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(printer_id, key)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_printer_memos_printer ON printer_memos(printer_id)"
        )

        # Schema version table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    def _set_version(self, conn: sqlite3.Connection) -> None:
        """Set schema version"""
        conn.execute(
            "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
            (self.SCHEMA_VERSION,),
        )

    def get_version(self) -> int:
        """Get current schema version"""
        with self.connection() as conn:
            cursor = conn.execute("SELECT version FROM schema_version LIMIT 1")
            row = cursor.fetchone()
            return row[0] if row else 0

    def is_initialized(self) -> bool:
        """Check if database is initialized"""
        try:
            return self.get_version() >= self.SCHEMA_VERSION
        except DatabaseError:
            return False

    def close(self) -> None:
        """Close database (no-op for SQLite, kept for API compatibility)"""
        pass


def adapt_datetime(val: datetime) -> str:
    """Adapt datetime for SQLite storage"""
    return val.isoformat()


def convert_datetime(val: bytes) -> datetime:
    """Convert SQLite datetime string to datetime"""
    return datetime.fromisoformat(val.decode())


def adapt_json(val: dict[str, Any] | list[Any]) -> str:
    """Adapt JSON-serializable object for SQLite storage"""
    return json.dumps(val)


def convert_json(val: bytes) -> dict[str, Any] | list[Any]:
    """Convert SQLite JSON string to Python object"""
    return json.loads(val.decode())


# Register adapters and converters
sqlite3.register_adapter(datetime, adapt_datetime)
sqlite3.register_converter("TIMESTAMP", convert_datetime)


def get_default_db_path(data_dir: Path | str) -> Path:
    """Get default database path from data directory"""
    return Path(data_dir) / "sensing_voxel.db"


def create_database(data_dir: Path | str | None = None) -> Database:
    """
    Create and initialize database

    Args:
        data_dir: Data directory path (default: ./data)

    Returns:
        Database: Initialized database instance
    """
    if data_dir is None:
        data_dir = Path("data")

    db_path = get_default_db_path(data_dir)
    config = DatabaseConfig(db_path=db_path)
    db = Database(config)

    if not db.is_initialized():
        db.initialize()

    return db
