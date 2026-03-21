"""
Unit tests for storage module

Tests database initialization, job repository, and calibration repository.
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path

from sensing_voxel.storage import (
    Database,
    DatabaseConfig,
    DatabaseConnectionError,
    CalibrationRecord,
    CalibrationRepository,
    CalibrationType,
    JobRecord,
    JobRepository,
    JobStatus,
    create_database,
    get_default_db_path,
)


@pytest.fixture
def temp_db_path(tmp_path):
    return tmp_path / "test.db"


@pytest.fixture
def database(temp_db_path):
    config = DatabaseConfig(db_path=temp_db_path)
    db = Database(config)
    db.initialize()
    return db


@pytest.fixture
def job_repo(database):
    return JobRepository(database)


@pytest.fixture
def calibration_repo(database):
    return CalibrationRepository(database)


class TestDatabase:
    def test_database_initializes_on_first_run(self, temp_db_path):
        config = DatabaseConfig(db_path=temp_db_path)
        db = Database(config)
        assert not db.is_initialized()
        db.initialize()
        assert db.is_initialized()
        assert temp_db_path.exists()

    def test_database_version(self, database):
        version = database.get_version()
        assert version == 1

    def test_connection_context_manager(self, database):
        with database.connection() as conn:
            cursor = conn.execute("SELECT 1")
            assert cursor.fetchone()[0] == 1

    def test_transaction_commit(self, database):
        with database.transaction() as conn:
            conn.execute(
                "INSERT INTO jobs (job_id, printer_id, file_name, status, started_at) VALUES (?, ?, ?, ?, ?)",
                ("test_job", "printer1", "test.gcode", "pending", datetime.now()),
            )

        with database.connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM jobs WHERE job_id = ?", ("test_job",))
            assert cursor.fetchone()[0] == 1

    def test_transaction_rollback_on_error(self, database):
        try:
            with database.transaction() as conn:
                conn.execute(
                    "INSERT INTO jobs (job_id, printer_id, file_name, status, started_at) VALUES (?, ?, ?, ?, ?)",
                    ("rollback_test", "printer1", "test.gcode", "pending", datetime.now()),
                )
                raise ValueError("Test error")
        except ValueError:
            pass

        with database.connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM jobs WHERE job_id = ?", ("rollback_test",))
            assert cursor.fetchone()[0] == 0

    def test_get_default_db_path(self, tmp_path):
        path = get_default_db_path(tmp_path)
        assert path == tmp_path / "sensing_voxel.db"

    def test_create_database_auto_initializes(self, tmp_path):
        data_dir = tmp_path / "data"
        db = create_database(data_dir)
        assert db.is_initialized()


class TestJobRepository:
    def test_create_job(self, job_repo):
        job = job_repo.create(
            job_id="job_001",
            printer_id="printer1",
            file_name="test.gcode",
            status=JobStatus.PENDING,
        )
        assert job.id is not None
        assert job.job_id == "job_001"
        assert job.printer_id == "printer1"
        assert job.file_name == "test.gcode"
        assert job.status == JobStatus.PENDING

    def test_create_job_with_metadata(self, job_repo):
        job = job_repo.create(
            job_id="job_002",
            printer_id="printer1",
            file_name="test.gcode",
            metadata={"layer_height": "0.2", "material": "PLA"},
        )
        retrieved = job_repo.get_by_id("job_002")
        assert retrieved is not None
        assert retrieved.metadata["layer_height"] == "0.2"
        assert retrieved.metadata["material"] == "PLA"

    def test_get_by_id_not_found(self, job_repo):
        result = job_repo.get_by_id("nonexistent")
        assert result is None

    def test_get_by_db_id(self, job_repo):
        job = job_repo.create(
            job_id="job_003",
            printer_id="printer1",
            file_name="test.gcode",
        )
        retrieved = job_repo.get_by_db_id(job.id)
        assert retrieved is not None
        assert retrieved.job_id == "job_003"

    def test_complete_job(self, job_repo):
        job_repo.create(
            job_id="job_004",
            printer_id="printer1",
            file_name="test.gcode",
            status=JobStatus.RUNNING,
        )
        result = job_repo.complete(
            job_id="job_004",
            duration_seconds=3600.0,
            filament_used_mm=1000.0,
            notes="Completed successfully",
        )
        assert result is True

        job = job_repo.get_by_id("job_004")
        assert job.status == JobStatus.COMPLETED
        assert job.completed_at is not None
        assert job.duration_seconds == 3600.0
        assert job.filament_used_mm == 1000.0
        assert job.notes == "Completed successfully"

    def test_fail_job(self, job_repo):
        job_repo.create(
            job_id="job_005",
            printer_id="printer1",
            file_name="test.gcode",
            status=JobStatus.RUNNING,
        )
        result = job_repo.fail(
            job_id="job_005",
            duration_seconds=1800.0,
            notes="Failed due to NOTHING",
        )
        assert result is True

        job = job_repo.get_by_id("job_005")
        assert job.status == JobStatus.FAILED
        assert job.notes == "Failed due to NOTHING"

    def test_cancel_job(self, job_repo):
        job_repo.create(
            job_id="job_006",
            printer_id="printer1",
            file_name="test.gcode",
            status=JobStatus.RUNNING,
        )
        result = job_repo.cancel(
            job_id="job_006",
            notes="User cancelled",
        )
        assert result is True

        job = job_repo.get_by_id("job_006")
        assert job.status == JobStatus.CANCELLED

    def test_list_jobs(self, job_repo):
        for i in range(5):
            job_repo.create(
                job_id=f"job_list_{i}",
                printer_id="printer1" if i < 3 else "printer2",
                file_name="test.gcode",
                status=JobStatus.COMPLETED if i % 2 == 0 else JobStatus.FAILED,
            )

        all_jobs = job_repo.list_jobs()
        assert len(all_jobs) == 5

        printer1_jobs = job_repo.list_jobs(printer_id="printer1")
        assert len(printer1_jobs) == 3

        completed_jobs = job_repo.list_jobs(status=JobStatus.COMPLETED)
        assert len(completed_jobs) == 3

    def test_list_jobs_pagination(self, job_repo):
        for i in range(10):
            job_repo.create(
                job_id=f"job_page_{i}",
                printer_id="printer1",
                file_name="test.gcode",
            )

        page1 = job_repo.list_jobs(limit=5, offset=0)
        assert len(page1) == 5

        page2 = job_repo.list_jobs(limit=5, offset=5)
        assert len(page2) == 5

    def test_get_recent_jobs(self, job_repo):
        old_time = datetime.now() - timedelta(hours=25)

        job_repo.create(
            job_id="old_job",
            printer_id="printer1",
            file_name="test.gcode",
            started_at=old_time,
        )

        job_repo.create(
            job_id="new_job",
            printer_id="printer1",
            file_name="test.gcode",
        )

        recent = job_repo.get_recent_jobs(hours=24)
        assert len(recent) == 1
        assert recent[0].job_id == "new_job"

    def test_get_job_stats(self, job_repo):
        for i in range(10):
            job = job_repo.create(
                job_id=f"stat_job_{i}",
                printer_id="printer1",
                file_name="test.gcode",
            )
            if i < 6:
                job_repo.complete(
                    job_id=f"stat_job_{i}",
                    duration_seconds=3600.0,
                    filament_used_mm=1000.0,
                )
            elif i < 8:
                job_repo.fail(job_id=f"stat_job_{i}")
            else:
                job_repo.cancel(job_id=f"stat_job_{i}")

        stats = job_repo.get_job_stats(printer_id="printer1", days=30)
        assert stats["total_jobs"] == 10
        assert stats["completed"] == 6
        assert stats["failed"] == 2
        assert stats["cancelled"] == 2
        assert stats["success_rate"] == 0.6
        assert stats["avg_duration_seconds"] == 3600.0
        assert stats["total_filament_mm"] == 6000.0

    def test_delete_job(self, job_repo):
        job_repo.create(
            job_id="delete_me",
            printer_id="printer1",
            file_name="test.gcode",
        )
        result = job_repo.delete("delete_me")
        assert result is True
        assert job_repo.get_by_id("delete_me") is None

    def test_delete_old_jobs(self, job_repo):
        old_time = datetime.now() - timedelta(days=100)
        recent_time = datetime.now() - timedelta(days=10)

        job_repo.create(
            job_id="old_delete",
            printer_id="printer1",
            file_name="test.gcode",
            started_at=old_time,
        )

        job_repo.create(
            job_id="recent_keep",
            printer_id="printer1",
            file_name="test.gcode",
            started_at=recent_time,
        )

        deleted_count = job_repo.delete_old_jobs(days=90)
        assert deleted_count == 1
        assert job_repo.get_by_id("old_delete") is None
        assert job_repo.get_by_id("recent_keep") is not None


class TestCalibrationRepository:
    def test_create_calibration(self, calibration_repo):
        cal = calibration_repo.create(
            calibration_id="cal_001",
            printer_id="printer1",
            calibration_type=CalibrationType.CAMERA_HOMOGRAPHY,
            values={"matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        )
        assert cal.id is not None
        assert cal.calibration_id == "cal_001"
        assert cal.calibration_type == CalibrationType.CAMERA_HOMOGRAPHY

    def test_create_calibration_with_results(self, calibration_repo):
        cal = calibration_repo.create(
            calibration_id="cal_002",
            printer_id="printer1",
            calibration_type=CalibrationType.Z_OFFSET,
            values={"offset": 0.05},
            results={"accuracy": 0.99, "error_mm": 0.01},
            notes="Manual calibration",
        )
        retrieved = calibration_repo.get_by_id("cal_002")
        assert retrieved is not None
        assert retrieved.results["accuracy"] == 0.99
        assert retrieved.notes == "Manual calibration"

    def test_get_by_id_not_found(self, calibration_repo):
        result = calibration_repo.get_by_id("nonexistent")
        assert result is None

    def test_update_calibration(self, calibration_repo):
        calibration_repo.create(
            calibration_id="cal_003",
            printer_id="printer1",
            calibration_type=CalibrationType.Z_OFFSET,
            values={"offset": 0.05},
        )
        result = calibration_repo.update(
            calibration_id="cal_003",
            values={"offset": 0.06},
            results={"verified": True},
            notes="Updated",
        )
        assert result is True

        cal = calibration_repo.get_by_id("cal_003")
        assert cal.values["offset"] == 0.06
        assert cal.results["verified"] is True
        assert cal.notes == "Updated"

    def test_list_calibrations(self, calibration_repo):
        for i in range(5):
            calibration_repo.create(
                calibration_id=f"cal_list_{i}",
                printer_id="printer1" if i < 3 else "printer2",
                calibration_type=CalibrationType.CAMERA_HOMOGRAPHY
                if i % 2 == 0
                else CalibrationType.Z_OFFSET,
                values={"test": i},
            )

        all_cals = calibration_repo.list_calibrations()
        assert len(all_cals) == 5

        printer1_cals = calibration_repo.list_calibrations(printer_id="printer1")
        assert len(printer1_cals) == 3

        camera_cals = calibration_repo.list_calibrations(
            calibration_type=CalibrationType.CAMERA_HOMOGRAPHY
        )
        assert len(camera_cals) == 3

    def test_get_latest_calibration(self, calibration_repo):
        old_time = datetime.now() - timedelta(days=7)
        recent_time = datetime.now() - timedelta(days=1)

        calibration_repo.create(
            calibration_id="cal_old",
            printer_id="printer1",
            calibration_type=CalibrationType.CAMERA_HOMOGRAPHY,
            values={"version": 1},
            performed_at=old_time,
        )

        calibration_repo.create(
            calibration_id="cal_recent",
            printer_id="printer1",
            calibration_type=CalibrationType.CAMERA_HOMOGRAPHY,
            values={"version": 2},
            performed_at=recent_time,
        )

        latest = calibration_repo.get_latest_calibration(
            printer_id="printer1",
            calibration_type=CalibrationType.CAMERA_HOMOGRAPHY,
        )
        assert latest is not None
        assert latest.calibration_id == "cal_recent"
        assert latest.values["version"] == 2

    def test_get_calibration_history(self, calibration_repo):
        for i in range(5):
            calibration_repo.create(
                calibration_id=f"cal_hist_{i}",
                printer_id="printer1",
                calibration_type=CalibrationType.Z_OFFSET,
                values={"offset": i * 0.01},
            )

        history = calibration_repo.get_calibration_history(
            printer_id="printer1",
            calibration_type=CalibrationType.Z_OFFSET,
            days=90,
        )
        assert len(history) == 5

    def test_get_calibration_stats(self, calibration_repo):
        for i in range(10):
            cal_type = CalibrationType.CAMERA_HOMOGRAPHY if i < 4 else CalibrationType.Z_OFFSET
            calibration_repo.create(
                calibration_id=f"cal_stat_{i}",
                printer_id="printer1",
                calibration_type=cal_type,
                values={"test": i},
            )

        stats = calibration_repo.get_calibration_stats(printer_id="printer1", days=30)
        assert stats["total_calibrations"] == 10
        assert stats["by_type"]["camera_homography"] == 4
        assert stats["by_type"]["z_offset"] == 6

    def test_delete_calibration(self, calibration_repo):
        calibration_repo.create(
            calibration_id="cal_delete",
            printer_id="printer1",
            calibration_type=CalibrationType.Z_OFFSET,
            values={"offset": 0.05},
        )
        result = calibration_repo.delete("cal_delete")
        assert result is True
        assert calibration_repo.get_by_id("cal_delete") is None

    def test_delete_old_calibrations(self, calibration_repo):
        old_time = datetime.now() - timedelta(days=400)
        recent_time = datetime.now() - timedelta(days=10)

        calibration_repo.create(
            calibration_id="cal_old_del",
            printer_id="printer1",
            calibration_type=CalibrationType.Z_OFFSET,
            values={"offset": 0.05},
            performed_at=old_time,
        )

        calibration_repo.create(
            calibration_id="cal_recent_keep",
            printer_id="printer1",
            calibration_type=CalibrationType.Z_OFFSET,
            values={"offset": 0.06},
            performed_at=recent_time,
        )

        deleted_count = calibration_repo.delete_old_calibrations(days=365)
        assert deleted_count == 1
        assert calibration_repo.get_by_id("cal_old_del") is None
        assert calibration_repo.get_by_id("cal_recent_keep") is not None

    def test_has_valid_calibration(self, calibration_repo):
        old_time = datetime.now() - timedelta(days=40)
        recent_time = datetime.now() - timedelta(days=5)

        calibration_repo.create(
            calibration_id="cal_stale",
            printer_id="printer1",
            calibration_type=CalibrationType.CAMERA_HOMOGRAPHY,
            values={"version": 1},
            performed_at=old_time,
        )

        calibration_repo.create(
            calibration_id="cal_fresh",
            printer_id="printer1",
            calibration_type=CalibrationType.Z_OFFSET,
            values={"version": 2},
            performed_at=recent_time,
        )

        assert not calibration_repo.has_valid_calibration(
            printer_id="printer1",
            calibration_type=CalibrationType.CAMERA_HOMOGRAPHY,
            max_age_days=30,
        )
        assert calibration_repo.has_valid_calibration(
            printer_id="printer1",
            calibration_type=CalibrationType.Z_OFFSET,
            max_age_days=30,
        )
