"""
Storage module for Sensing Voxel

SQLite-based persistence for job history and calibration records.
"""

from sensing_voxel.storage.calibration_repository import (
    CalibrationRecord,
    CalibrationRepository,
    CalibrationType,
)
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

__all__ = [
    # Database
    "Database",
    "DatabaseConfig",
    "DatabaseError",
    "DatabaseConnectionError",
    "DatabaseSchemaError",
    "create_database",
    "get_default_db_path",
    # Job Repository
    "JobRepository",
    "JobRecord",
    "JobStatus",
    # Calibration Repository
    "CalibrationRepository",
    "CalibrationRecord",
    "CalibrationType",
]
