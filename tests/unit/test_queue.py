"""
Unit tests for multi-printer queue scheduling
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest

from sensing_voxel.core.models import JobContext, generate_job_id
from sensing_voxel.queue import (
    AssignmentResult,
    AssignmentStrategy,
    CompositeScheduler,
    DeadlineAwareScheduler,
    LoadBalancer,
    PriorityScheduler,
    PrinterCapabilities,
    PrinterLoad,
    PrinterStatus,
    QueueConfiguration,
    QueueJobStatus,
    QueueManager,
    QueuePriority,
    QueueStats,
    QueuedJob,
    ScheduleWindow,
    Scheduler,
    generate_queue_job_id,
)


class TestQueueModels:
    """Tests for queue data models"""

    def test_printer_capabilities_bed_area(self):
        cap = PrinterCapabilities(
            printer_id="printer1",
            bed_width_mm=220,
            bed_depth_mm=220,
        )
        assert cap.bed_area_mm2 == 48400

    def test_printer_capabilities_can_fit_bounds(self):
        from sensing_voxel.core.models import FirstLayerBounds

        cap = PrinterCapabilities(
            printer_id="printer1",
            bed_width_mm=220,
            bed_depth_mm=220,
        )
        bounds = FirstLayerBounds(min_x=0, max_x=100, min_y=0, max_y=100)
        assert cap.can_fit_bounds(bounds) is True

        large_bounds = FirstLayerBounds(min_x=0, max_x=300, min_y=0, max_y=300)
        assert cap.can_fit_bounds(large_bounds) is False

    def test_printer_capabilities_supports_filament(self):
        cap = PrinterCapabilities(
            printer_id="printer1",
            supported_filaments={"PLA", "PETG"},
        )
        assert cap.supports_filament("PLA") is True
        assert cap.supports_filament("pla") is True  # Case insensitive
        assert cap.supports_filament("ABS") is False

    def test_queued_job_wait_time(self):
        job = QueuedJob(
            queue_job_id="test1",
            job_context=None,
            created_at=datetime.now() - timedelta(minutes=10),
        )
        assert job.wait_time.total_seconds() >= 600

    def test_queued_job_is_overdue(self):
        job = QueuedJob(
            queue_job_id="test1",
            job_context=None,
            deadline=datetime.now() - timedelta(minutes=1),
        )
        assert job.is_overdue is True

        job2 = QueuedJob(
            queue_job_id="test2",
            job_context=None,
            deadline=datetime.now() + timedelta(hours=1),
        )
        assert job2.is_overdue is False

    def test_queued_job_priority_score_basic(self):
        job = QueuedJob(
            queue_job_id="test1",
            job_context=None,
            priority=QueuePriority.NORMAL,
            created_at=datetime.now(),
        )
        score = job.priority_score
        assert score > 190

    def test_queued_job_priority_score_with_wait_time(self):
        job = QueuedJob(
            queue_job_id="test1",
            job_context=None,
            priority=QueuePriority.NORMAL,
            created_at=datetime.now() - timedelta(minutes=10),
        )
        score_before = QueuedJob(
            queue_job_id="test2",
            job_context=None,
            priority=QueuePriority.NORMAL,
            created_at=datetime.now(),
        ).priority_score
        score_after = job.priority_score

        # Longer wait time should lower score (higher priority)
        assert score_after < score_before

    def test_queued_job_priority_score_with_deadline(self):
        job_overdue = QueuedJob(
            queue_job_id="test1",
            job_context=None,
            priority=QueuePriority.NORMAL,
            created_at=datetime.now(),
            deadline=datetime.now() - timedelta(minutes=1),
        )
        job_urgent = QueuedJob(
            queue_job_id="test2",
            job_context=None,
            priority=QueuePriority.NORMAL,
            created_at=datetime.now(),
            deadline=datetime.now() + timedelta(minutes=30),
        )
        job_normal = QueuedJob(
            queue_job_id="test3",
            job_context=None,
            priority=QueuePriority.NORMAL,
            created_at=datetime.now(),
            deadline=datetime.now() + timedelta(days=2),
        )

        # Overdue jobs should have lowest score (highest priority)
        assert job_overdue.priority_score < job_urgent.priority_score
        assert job_urgent.priority_score < job_normal.priority_score


class TestQueueManager:
    """Tests for QueueManager"""

    @pytest.fixture
    def manager(self):
        return QueueManager()

    @pytest.fixture
    def printer_caps(self):
        return PrinterCapabilities(
            printer_id="printer1",
            bed_width_mm=220,
            bed_depth_mm=220,
            supported_filaments={"PLA", "PETG"},
        )

    @pytest.fixture
    def job_context(self):
        return JobContext(
            job_id=generate_job_id(),
            job_file="test.gcode",
            uploaded_at=datetime.now(),
        )

    @pytest.mark.asyncio
    async def test_register_printer(self, manager, printer_caps):
        result = await manager.register_printer(printer_caps)
        assert result is True

        printers = await manager.get_printers()
        assert len(printers) == 1
        assert printers[0].printer_id == "printer1"

    @pytest.mark.asyncio
    async def test_register_duplicate_printer(self, manager, printer_caps):
        await manager.register_printer(printer_caps)
        result = await manager.register_printer(printer_caps)
        assert result is False

    @pytest.mark.asyncio
    async def test_unregister_printer(self, manager, printer_caps):
        await manager.register_printer(printer_caps)
        result = await manager.unregister_printer("printer1")
        assert result is True

        printers = await manager.get_printers()
        assert len(printers) == 0

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_printer(self, manager):
        result = await manager.unregister_printer("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_submit_job(self, manager, job_context):
        job = await manager.submit_job(job_context)
        assert job is not None
        assert job.queue_job_id is not None
        assert job.status == QueueJobStatus.PENDING

    @pytest.mark.asyncio
    async def test_submit_job_exceeds_queue_size(self, manager, job_context):
        manager.config.max_queue_size = 2
        await manager.submit_job(job_context)
        await manager.submit_job(job_context)
        job3 = await manager.submit_job(job_context)
        assert job3 is None

    @pytest.mark.asyncio
    async def test_cancel_job(self, manager, job_context):
        job = await manager.submit_job(job_context)
        result = await manager.cancel_job(job.queue_job_id)
        assert result is True

        updated_job = await manager.get_job(job.queue_job_id)
        assert updated_job.status == QueueJobStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_completed_job(self, manager, job_context):
        job = await manager.submit_job(job_context)
        await manager.cancel_job(job.queue_job_id)
        result = await manager.cancel_job(job.queue_job_id)
        assert result is False

    @pytest.mark.asyncio
    async def test_list_jobs(self, manager, job_context):
        job1 = await manager.submit_job(job_context, user_id="user1")
        job2 = await manager.submit_job(job_context, user_id="user1")
        await manager.submit_job(job_context, user_id="user2")

        user_jobs = await manager.list_jobs(user_id="user1")
        assert len(user_jobs) == 2

        pending_jobs = await manager.list_jobs(status=QueueJobStatus.PENDING)
        assert len(pending_jobs) == 3

    @pytest.mark.asyncio
    async def test_update_printer_status(self, manager, printer_caps):
        await manager.register_printer(printer_caps)
        result = await manager.update_printer_status(
            "printer1", PrinterStatus.PRINTING, current_job_id="job1"
        )
        assert result is True

        load = await manager.get_printer_load("printer1")
        assert load.status == PrinterStatus.PRINTING
        assert load.current_job_id == "job1"

    @pytest.mark.asyncio
    async def test_job_status_callbacks(self, manager, job_context):
        callbacks = []

        async def status_callback(job, old_status, new_status):
            callbacks.append((job.queue_job_id, old_status, new_status))

        manager.on_job_status_change(status_callback)

        job = await manager.submit_job(job_context)
        await manager.cancel_job(job.queue_job_id)

        assert len(callbacks) >= 1
        assert callbacks[-1][2] == QueueJobStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_assignment_callbacks(self, manager, printer_caps, job_context):
        callbacks = []

        async def assignment_callback(job_id, printer_id):
            callbacks.append((job_id, printer_id))

        manager.on_job_assignment(assignment_callback)
        await manager.register_printer(printer_caps)
        await manager.update_printer_status("printer1", PrinterStatus.AVAILABLE)

        job = await manager.submit_job(job_context)
        result = await manager.assign_job(job.queue_job_id)

        assert result.success is True
        assert result.printer_id == "printer1"
        assert len(callbacks) == 1
        assert callbacks[0][1] == "printer1"
        assert callbacks[0][1] == "printer1"

    @pytest.mark.asyncio
    async def test_get_stats(self, manager, printer_caps, job_context):
        await manager.register_printer(printer_caps)
        await manager.update_printer_status("printer1", PrinterStatus.AVAILABLE)

        job = await manager.submit_job(job_context)
        await manager.assign_job(job.queue_job_id)

        stats = await manager.get_stats()
        assert stats.total_jobs == 1
        assert "printer1" in stats.printer_utilization


class TestScheduler:
    """Tests for Scheduler"""

    @pytest.fixture
    def scheduler(self):
        return Scheduler()

    @pytest.fixture
    def printer1(self):
        cap = PrinterCapabilities(
            printer_id="printer1",
            bed_width_mm=220,
            bed_depth_mm=220,
            supported_filaments={"PLA", "PETG"},
        )
        load = PrinterLoad(printer_id="printer1", status=PrinterStatus.AVAILABLE)
        return cap, load

    @pytest.fixture
    def printer2(self):
        cap = PrinterCapabilities(
            printer_id="printer2",
            bed_width_mm=300,
            bed_depth_mm=300,
            supported_filaments={"PLA", "ABS"},
        )
        load = PrinterLoad(printer_id="printer2", status=PrinterStatus.AVAILABLE)
        return cap, load

    @pytest.fixture
    def job(self):
        return QueuedJob(
            queue_job_id="job1",
            job_context=None,
            filament_type="PLA",
        )

    def test_schedule_first_available(self, scheduler, printer1, job):
        printers = {"printer1": printer1}
        result = scheduler.schedule(job, printers, AssignmentStrategy.FIRST_AVAILABLE)

        assert result.success is True
        assert result.printer_id == "printer1"

    def test_schedule_no_available_printers(self, scheduler, job):
        result = scheduler.schedule(job, {}, AssignmentStrategy.FIRST_AVAILABLE)
        assert result.success is False

    def test_schedule_least_loaded(self, scheduler, printer1, printer2, job):
        printer1[1].utilization_percent = 50
        printer2[1].utilization_percent = 20

        printers = {"printer1": printer1, "printer2": printer2}
        result = scheduler.schedule(job, printers, AssignmentStrategy.LEAST_LOADED)

        assert result.success is True
        assert result.printer_id == "printer2"

    def test_schedule_filament_match(self, scheduler, printer1, printer2, job):
        job.filament_type = "PETG"
        printers = {"printer1": printer1, "printer2": printer2}
        result = scheduler.schedule(job, printers, AssignmentStrategy.FILAMENT_MATCH)

        assert result.success is True
        assert result.printer_id == "printer1"

    def test_schedule_filament_no_match(self, scheduler, printer2, job):
        job.filament_type = "PETG"
        printers = {"printer2": printer2}
        result = scheduler.schedule(job, printers, AssignmentStrategy.FILAMENT_MATCH)

        assert result.success is False

    def test_schedule_optimal_prefers_idle(self, scheduler, printer1, printer2, job):
        printer1[1].last_activity = datetime.now() - timedelta(hours=2)
        printer2[1].last_activity = datetime.now() - timedelta(minutes=5)

        printers = {"printer1": printer1, "printer2": printer2}
        result = scheduler.schedule(job, printers, AssignmentStrategy.OPTIMAL)

        assert result.success is True
        assert result.printer_id == "printer1"

    def test_schedule_optimal_prefers_filament_match(self, scheduler, printer1, printer2, job):
        job.filament_type = "PETG"
        printers = {"printer1": printer1, "printer2": printer2}
        result = scheduler.schedule(job, printers, AssignmentStrategy.OPTIMAL)

        assert result.success is True
        assert result.printer_id == "printer1"


class TestPriorityScheduler:
    """Tests for PriorityScheduler"""

    @pytest.fixture
    def priority_scheduler(self):
        return PriorityScheduler()

    def test_sort_by_priority(self, priority_scheduler):
        jobs = [
            QueuedJob("j1", None, priority=QueuePriority.LOW),
            QueuedJob("j2", None, priority=QueuePriority.HIGH),
            QueuedJob("j3", None, priority=QueuePriority.NORMAL),
        ]

        sorted_jobs = priority_scheduler.sort_by_priority(jobs)
        assert sorted_jobs[0].queue_job_id == "j2"
        assert sorted_jobs[1].queue_job_id == "j3"
        assert sorted_jobs[2].queue_job_id == "j1"

    def test_prioritize_with_deadlines(self, priority_scheduler):
        jobs = [
            QueuedJob(
                "j1",
                None,
                priority=QueuePriority.NORMAL,
                deadline=datetime.now() + timedelta(hours=1),
            ),
            QueuedJob(
                "j2",
                None,
                priority=QueuePriority.HIGH,
                deadline=datetime.now() + timedelta(hours=2),
            ),
            QueuedJob(
                "j3",
                None,
                priority=QueuePriority.NORMAL,
                deadline=datetime.now() + timedelta(minutes=30),
            ),
        ]

        sorted_jobs = priority_scheduler.prioritize_with_deadlines(jobs)
        assert sorted_jobs[0].queue_job_id == "j2"  # High priority
        # Normal priority with closer deadline first
        assert sorted_jobs[1].queue_job_id == "j3"
        assert sorted_jobs[2].queue_job_id == "j1"

    def test_get_urgency_score(self, priority_scheduler):
        critical = QueuedJob("j1", None, priority=QueuePriority.CRITICAL)
        normal = QueuedJob("j2", None, priority=QueuePriority.NORMAL)
        low = QueuedJob("j3", None, priority=QueuePriority.LOW)

        assert priority_scheduler.get_urgency_score(critical) == 1000
        assert priority_scheduler.get_urgency_score(normal) > priority_scheduler.get_urgency_score(
            low
        )

    def test_get_urgency_score_with_deadline(self, priority_scheduler):
        overdue = QueuedJob(
            "j1",
            None,
            priority=QueuePriority.NORMAL,
            deadline=datetime.now() - timedelta(minutes=1),
        )
        soon = QueuedJob(
            "j2",
            None,
            priority=QueuePriority.NORMAL,
            deadline=datetime.now() + timedelta(minutes=30),
        )
        later = QueuedJob(
            "j3", None, priority=QueuePriority.NORMAL, deadline=datetime.now() + timedelta(days=2)
        )

        assert priority_scheduler.get_urgency_score(overdue) > priority_scheduler.get_urgency_score(
            soon
        )
        assert priority_scheduler.get_urgency_score(soon) > priority_scheduler.get_urgency_score(
            later
        )


class TestLoadBalancer:
    """Tests for LoadBalancer"""

    @pytest.fixture
    def load_balancer(self):
        return LoadBalancer()

    @pytest.fixture
    def printer1(self):
        cap = PrinterCapabilities(printer_id="printer1")
        load = PrinterLoad(printer_id="printer1", status=PrinterStatus.AVAILABLE)
        return cap, load

    @pytest.fixture
    def printer2(self):
        cap = PrinterCapabilities(printer_id="printer2")
        load = PrinterLoad(printer_id="printer2", status=PrinterStatus.AVAILABLE)
        return cap, load

    def test_calculate_utilization_empty(self, load_balancer, printer1):
        printers = {"printer1": printer1}
        util = load_balancer.calculate_utilization("printer1", printers, {})
        assert util == 0.0

    def test_calculate_utilization_with_pending_jobs(self, load_balancer, printer1):
        printers = {"printer1": printer1}
        job = QueuedJob("j1", None, assigned_printer_id="printer1", status=QueueJobStatus.PENDING)
        jobs = {"j1": job}

        util = load_balancer.calculate_utilization("printer1", printers, jobs)
        assert util > 0.0

    def test_get_balanced_assignment(self, load_balancer, printer1, printer2):
        printer1[1].utilization_percent = 80
        printer2[1].utilization_percent = 20

        printers = {"printer1": printer1, "printer2": printer2}
        job = QueuedJob("j1", None)

        assigned = load_balancer.get_balanced_assignment(job, printers, {})
        assert assigned == "printer2"

    def test_get_balanced_assignment_no_available(self, load_balancer, printer1):
        printer1[1].status = PrinterStatus.BUSY
        printers = {"printer1": printer1}
        job = QueuedJob("j1", None)

        assigned = load_balancer.get_balanced_assignment(job, printers, {})
        assert assigned is None

    def test_rebalance_queue(self, load_balancer, printer1, printer2):
        printer1[1].utilization_percent = 90
        printer1[1].status = PrinterStatus.AVAILABLE
        printer2[1].utilization_percent = 10
        printer2[1].status = PrinterStatus.AVAILABLE

        printers = {"printer1": printer1, "printer2": printer2}

        job = QueuedJob("j1", None, assigned_printer_id="printer1", status=QueueJobStatus.PENDING)
        jobs = {"j1": job}

        reassignments = load_balancer.rebalance_queue(printers, jobs)
        assert len(reassignments) > 0
        assert reassignments[0] == ("j1", "printer2")


class TestDeadlineAwareScheduler:
    """Tests for DeadlineAwareScheduler"""

    @pytest.fixture
    def deadline_scheduler(self):
        return DeadlineAwareScheduler()

    @pytest.fixture
    def printer1(self):
        cap = PrinterCapabilities(printer_id="printer1")
        load = PrinterLoad(printer_id="printer1", status=PrinterStatus.AVAILABLE)
        return cap, load

    def test_can_meet_deadline_no_deadline(self, deadline_scheduler, printer1):
        job = QueuedJob("j1", None, estimated_duration=timedelta(hours=1))
        printers = {"printer1": printer1}

        can_meet = deadline_scheduler.can_meet_deadline(job, "printer1", printers)
        assert can_meet is True

    def test_can_meet_deadline_with_sufficient_time(self, deadline_scheduler, printer1):
        job = QueuedJob(
            "j1",
            None,
            deadline=datetime.now() + timedelta(hours=2),
            estimated_duration=timedelta(hours=1),
        )
        printers = {"printer1": printer1}

        can_meet = deadline_scheduler.can_meet_deadline(job, "printer1", printers)
        assert can_meet is True

    def test_can_meet_deadline_insufficient_time(self, deadline_scheduler, printer1):
        job = QueuedJob(
            "j1",
            None,
            deadline=datetime.now() + timedelta(minutes=30),
            estimated_duration=timedelta(hours=1),
        )
        printer1[1].status = PrinterStatus.PRINTING
        printer1[1].estimated_completion = datetime.now() + timedelta(minutes=20)
        printers = {"printer1": printer1}

        can_meet = deadline_scheduler.can_meet_deadline(job, "printer1", printers)
        assert can_meet is False

    def test_find_viable_printers(self, deadline_scheduler, printer1):
        job = QueuedJob(
            "j1",
            None,
            deadline=datetime.now() + timedelta(hours=1),
            estimated_duration=timedelta(minutes=30),
        )
        printers = {"printer1": printer1}

        viable = deadline_scheduler.find_viable_printers(job, printers)
        assert "printer1" in viable

    def test_schedule_with_sla_success(self, deadline_scheduler, printer1):
        job = QueuedJob(
            "j1",
            None,
            deadline=datetime.now() + timedelta(hours=1),
            estimated_duration=timedelta(minutes=30),
        )
        printers = {"printer1": printer1}

        result = deadline_scheduler.schedule_with_sla(job, printers)
        assert result.success is True
        assert result.printer_id == "printer1"

    def test_schedule_with_sla_no_printers(self, deadline_scheduler):
        job = QueuedJob("j1", None, deadline=datetime.now() + timedelta(hours=1))
        result = deadline_scheduler.schedule_with_sla(job, {})
        assert result.success is False


class TestCompositeScheduler:
    """Tests for CompositeScheduler"""

    @pytest.fixture
    def composite_scheduler(self):
        return CompositeScheduler()

    @pytest.fixture
    def printer1(self):
        cap = PrinterCapabilities(printer_id="printer1")
        load = PrinterLoad(printer_id="printer1", status=PrinterStatus.AVAILABLE)
        return cap, load

    @pytest.fixture
    def printer2(self):
        cap = PrinterCapabilities(printer_id="printer2")
        load = PrinterLoad(printer_id="printer2", status=PrinterStatus.AVAILABLE)
        return cap, load

    def test_schedule_batch(self, composite_scheduler, printer1, printer2):
        jobs = [
            QueuedJob("j1", None, priority=QueuePriority.HIGH),
            QueuedJob("j2", None, priority=QueuePriority.NORMAL),
        ]
        printers = {"printer1": printer1, "printer2": printer2}

        results = composite_scheduler.schedule_batch(jobs, printers)
        assert len(results) == 2
        assert all(r.success for r in results)

    def test_schedule_batch_more_jobs_than_printers(self, composite_scheduler, printer1):
        jobs = [
            QueuedJob("j1", None),
            QueuedJob("j2", None),
            QueuedJob("j3", None),
        ]
        printers = {"printer1": printer1}

        results = composite_scheduler.schedule_batch(jobs, printers)
        assert len(results) == 3
        success_count = sum(1 for r in results if r.success)
        assert success_count == 1

    def test_get_schedule_recommendations(self, composite_scheduler, printer1):
        future_deadline = datetime.now() + timedelta(minutes=5)
        jobs = [
            QueuedJob("j1", None, deadline=future_deadline, estimated_duration=timedelta(hours=2)),
            QueuedJob("j2", None),
        ]
        printer1[1].status = PrinterStatus.PRINTING
        printer1[1].estimated_completion = datetime.now() + timedelta(minutes=30)
        printers = {"printer1": printer1}

        recommendations = composite_scheduler.get_schedule_recommendations(jobs, printers)
        assert recommendations["total_jobs"] == 2
        assert recommendations["available_printers"] == 0
        assert len(recommendations["deadline_risks"]) == 1


class TestQueueConfiguration:
    """Tests for QueueConfiguration"""

    def test_default_configuration(self):
        config = QueueConfiguration()
        assert config.max_queue_size == 100
        assert config.default_priority == QueuePriority.NORMAL
        assert config.auto_assignment is True
        assert config.enable_load_balancing is True

    def test_schedule_window_contains(self):
        window = ScheduleWindow(
            start=datetime.now() - timedelta(hours=1), end=datetime.now() + timedelta(hours=1)
        )
        assert window.contains(datetime.now()) is True
        assert window.contains(datetime.now() + timedelta(hours=2)) is False


class TestQueueIntegration:
    """Integration tests for queue scheduling"""

    @pytest.mark.asyncio
    async def test_full_job_lifecycle(self):
        manager = QueueManager()

        # Register printer
        caps = PrinterCapabilities(
            printer_id="neptune4",
            bed_width_mm=225,
            bed_depth_mm=225,
            supported_filaments={"PLA", "PETG", "ABS"},
        )
        await manager.register_printer(caps)
        await manager.update_printer_status("neptune4", PrinterStatus.AVAILABLE)

        # Submit job
        job_context = JobContext(
            job_id=generate_job_id(),
            job_file="model.gcode",
            uploaded_at=datetime.now(),
        )
        job = await manager.submit_job(
            job_context,
            priority=QueuePriority.HIGH,
            filament_type="PETG",
        )
        assert job.status == QueueJobStatus.PENDING

        # Assign job
        result = await manager.assign_job(job.queue_job_id)
        assert result.success is True
        assert result.printer_id == "neptune4"

        # Update progress
        await manager.update_job_progress(job.queue_job_id, 0.5)

        # Complete job
        await manager.complete_job(job.queue_job_id, success=True)

        completed_job = await manager.get_job(job.queue_job_id)
        assert completed_job.status == QueueJobStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_priority_queue_ordering(self):
        manager = QueueManager()

        caps = PrinterCapabilities(printer_id="printer1")
        await manager.register_printer(caps)
        await manager.update_printer_status("printer1", PrinterStatus.AVAILABLE)

        # Submit jobs in reverse priority order
        job_context = JobContext(
            job_id=generate_job_id(),
            job_file="test.gcode",
            uploaded_at=datetime.now(),
        )

        low_job = await manager.submit_job(job_context, priority=QueuePriority.LOW)
        normal_job = await manager.submit_job(job_context, priority=QueuePriority.NORMAL)
        high_job = await manager.submit_job(job_context, priority=QueuePriority.HIGH)

        pending = await manager.list_jobs(status=QueueJobStatus.PENDING)
        assert pending[0].queue_job_id == high_job.queue_job_id
        assert pending[1].queue_job_id == normal_job.queue_job_id
        assert pending[2].queue_job_id == low_job.queue_job_id

    @pytest.mark.asyncio
    async def test_auto_assignment(self):
        config = QueueConfiguration(auto_assignment=True)
        manager = QueueManager(config=config)

        caps = PrinterCapabilities(printer_id="auto_printer")
        await manager.register_printer(caps)
        await manager.update_printer_status("auto_printer", PrinterStatus.AVAILABLE)

        job_context = JobContext(
            job_id=generate_job_id(),
            job_file="test.gcode",
            uploaded_at=datetime.now(),
        )

        job = await manager.submit_job(job_context)

        # Start scheduler loop briefly
        await manager.start()
        await asyncio.sleep(0.1)
        await manager.stop()

        # Job should be assigned
        updated = await manager.get_job(job.queue_job_id)
        assert updated.status == QueueJobStatus.SCHEDULED
        assert updated.assigned_printer_id == "auto_printer"

    @pytest.mark.asyncio
    async def test_load_balancing_across_printers(self):
        manager = QueueManager()

        # Register two printers
        caps1 = PrinterCapabilities(printer_id="printer1")
        caps2 = PrinterCapabilities(printer_id="printer2")
        await manager.register_printer(caps1)
        await manager.register_printer(caps2)
        await manager.update_printer_status("printer1", PrinterStatus.AVAILABLE)
        await manager.update_printer_status("printer2", PrinterStatus.AVAILABLE)

        # Set different utilization
        await manager.update_printer_status("printer1", PrinterStatus.PRINTING)
        slot1 = manager._printers["printer1"]
        slot1.load.utilization_percent = 80

        slot2 = manager._printers["printer2"]
        slot2.load.utilization_percent = 10

        job_context = JobContext(
            job_id=generate_job_id(),
            job_file="test.gcode",
            uploaded_at=datetime.now(),
        )

        job = await manager.submit_job(job_context)
        result = await manager.assign_job(job.queue_job_id)

        # Should assign to less loaded printer
        assert result.printer_id == "printer2"
