"""
Unit tests for photo-to-print pipeline.

Tests the 8-stage orchestration:
- Stage 1: Image-to-3D generation
- Stage 2: Mesh repair
- Stage 3: Optimize orientation
- Stage 4: Analyze support
- Stage 5: Preview rendering
- Stage 6: Telegram approval
- Stage 7: Slicing
- Stage 8: Print start
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest
from PIL import Image

from sensing_voxel.adapters.base import PrinterAdapter
from sensing_voxel.core.models import CommandResult, CommandResultStatus
from sensing_voxel.modeling.image_to_3d import GenerationResult, ImageTo3DService
from sensing_voxel.modeling.orientation import OrientationResult
from sensing_voxel.modeling.pipeline import (
    PhotoToPrintPipeline,
    PipelineJob,
    PipelineResult,
    PipelineStage,
    PipelineStatus,
    process_photo_to_print,
)
from sensing_voxel.modeling.support_analysis import SupportAnalysisResult
from sensing_voxel.slicer.pipeline import SlicerPipeline


@pytest.fixture
def mock_image():
    """Create a mock PIL Image."""
    return Image.new("RGB", (512, 512), color="red")


@pytest.fixture
def temp_image_file(tmp_path):
    """Create a temporary image file."""
    img_path = tmp_path / "test_image.png"
    img = Image.new("RGB", (512, 512), color="red")
    img.save(img_path)
    return img_path


@pytest.fixture
def mock_image_to_3d_service():
    """Create a mock ImageTo3DService."""
    service = MagicMock(spec=ImageTo3DService)
    service.generate = AsyncMock()
    return service


@pytest.fixture
def mock_slicer_pipeline():
    """Create a mock SlicerPipeline."""
    from sensing_voxel.core.models import SlicingResult

    pipeline = MagicMock(spec=SlicerPipeline)
    slicing_result = SlicingResult(
        gcode_path=Path("/tmp/test.gcode"),
        gcode_hash="abc123",
    )
    pipeline.process_model = AsyncMock(return_value=slicing_result)
    return pipeline


@pytest.fixture
def mock_printer_adapter():
    """Create a mock PrinterAdapter."""
    adapter = MagicMock(spec=PrinterAdapter)
    adapter.upload_file = AsyncMock(return_value=CommandResult(status=CommandResultStatus.SUCCESS))
    adapter.start_print = AsyncMock(return_value=CommandResult(status=CommandResultStatus.SUCCESS))
    return adapter


@pytest.fixture
def mock_telegram_bot():
    """Create a mock Telegram bot."""
    bot = MagicMock()
    bot.send_photo = AsyncMock(return_value=MagicMock(message_id=12345))
    return bot


@pytest.fixture
def pipeline(tmp_path, mock_image_to_3d_service, mock_slicer_pipeline):
    """Create a PhotoToPrintPipeline with mocked dependencies."""
    return PhotoToPrintPipeline(
        image_to_3d_service=mock_image_to_3d_service,
        slicer_pipeline=mock_slicer_pipeline,
        output_dir=tmp_path / "pipeline",
    )


@pytest.fixture
def pipeline_with_printer(
    tmp_path, mock_image_to_3d_service, mock_slicer_pipeline, mock_printer_adapter
):
    """Create a PhotoToPrintPipeline with printer adapter."""
    return PhotoToPrintPipeline(
        image_to_3d_service=mock_image_to_3d_service,
        slicer_pipeline=mock_slicer_pipeline,
        printer_adapter=mock_printer_adapter,
        output_dir=tmp_path / "pipeline",
    )


class TestPipelineJob:
    """Tests for PipelineJob dataclass."""

    def test_job_creation(self, tmp_path):
        """Test basic job creation."""
        job = PipelineJob(
            job_id="test123",
            user_id="user_1",
            chat_id=123456789,
            image_path=tmp_path / "test.png",
        )

        assert job.job_id == "test123"
        assert job.user_id == "user_1"
        assert job.chat_id == 123456789
        assert job.status == PipelineStatus.PENDING
        assert job.stage == PipelineStage.PENDING
        assert len(job.errors) == 0
        assert len(job.warnings) == 0

    def test_job_to_dict(self, tmp_path):
        """Test job serialization to dict."""
        job = PipelineJob(
            job_id="test123",
            user_id="user_1",
            chat_id=123456789,
            image_path=tmp_path / "test.png",
        )

        data = job.to_dict()

        assert data["job_id"] == "test123"
        assert data["user_id"] == "user_1"
        assert data["chat_id"] == 123456789
        assert data["status"] == "pending"
        assert data["stage"] == "PENDING"


class TestPhotoToPrintPipeline:
    """Tests for PhotoToPrintPipeline."""

    @pytest.mark.asyncio
    async def test_stage_1_success(self, pipeline, temp_image_file, mock_image_to_3d_service):
        """Test successful image-to-3D stage."""
        # Mock successful generation
        stl_data = b"mock stl data"
        mock_image_to_3d_service.generate.return_value = GenerationResult(
            success=True,
            model_data=stl_data,
            generation_id="gen_123",
        )

        job = PipelineJob(
            job_id="test123",
            user_id="user_1",
            chat_id=123456789,
            image_path=temp_image_file,
        )

        result = await pipeline._stage_1_image_to_3d(job)

        assert result is True
        assert job.stl_path is not None
        assert job.stl_path.exists()
        assert job.stl_path.read_bytes() == stl_data
        assert job.stage == PipelineStage.IMAGE_TO_3D

    @pytest.mark.asyncio
    async def test_stage_1_failure(self, pipeline, temp_image_file, mock_image_to_3d_service):
        """Test failed image-to-3D stage."""
        mock_image_to_3d_service.generate.return_value = GenerationResult(
            success=False,
            error_message="API rate limit exceeded",
        )

        job = PipelineJob(
            job_id="test123",
            user_id="user_1",
            chat_id=123456789,
            image_path=temp_image_file,
        )

        result = await pipeline._stage_1_image_to_3d(job)

        assert result is False
        assert job.status == PipelineStatus.FAILED
        assert "API rate limit exceeded" in job.errors[0]

    @pytest.mark.asyncio
    async def test_stage_1_retry_and_fail(
        self, pipeline, temp_image_file, mock_image_to_3d_service
    ):
        """Test image-to-3D stage with retries."""
        mock_image_to_3d_service.generate.return_value = GenerationResult(
            success=False,
            error_message="Temporary error",
        )

        job = PipelineJob(
            job_id="test123",
            user_id="user_1",
            chat_id=123456789,
            image_path=temp_image_file,
        )

        # Reduce retries for faster test
        pipeline.max_retries = 1
        result = await pipeline._stage_1_image_to_3d(job)

        assert result is False
        # Should have been called twice (initial + 1 retry)
        assert mock_image_to_3d_service.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_stage_2_success(self, pipeline, tmp_path):
        """Test successful mesh repair stage."""
        # Create a simple STL file
        stl_path = tmp_path / "test.stl"
        stl_path.write_bytes(b"solid test\nendsolid test")

        job = PipelineJob(
            job_id="test123",
            user_id="user_1",
            chat_id=123456789,
            image_path=tmp_path / "test.png",
        )
        job.stl_path = stl_path

        # Mock repair_mesh to return True
        with patch("sensing_voxel.modeling.pipeline.repair_mesh", return_value=True):
            result = await pipeline._stage_2_repair_mesh(job)

        assert result is True
        assert job.repaired_stl_path is not None

    @pytest.mark.asyncio
    async def test_stage_2_failure_no_stl(self, pipeline, tmp_path):
        """Test mesh repair stage when no STL exists."""
        job = PipelineJob(
            job_id="test123",
            user_id="user_1",
            chat_id=123456789,
            image_path=tmp_path / "test.png",
        )

        result = await pipeline._stage_2_repair_mesh(job)

        assert result is False
        assert job.status == PipelineStatus.FAILED
        assert "No STL file" in job.errors[0]

    @pytest.mark.asyncio
    async def test_stage_3_success(self, pipeline, tmp_path):
        """Test successful orientation optimization stage."""
        # Create a mock STL file
        stl_path = tmp_path / "repaired.stl"
        stl_path.write_bytes(b"mock stl content")

        job = PipelineJob(
            job_id="test123",
            user_id="user_1",
            chat_id=123456789,
            image_path=tmp_path / "test.png",
        )
        job.repaired_stl_path = stl_path

        # Mock orientation optimizer
        mock_result = OrientationResult(
            success=True,
            score=0.85,
            support_volume=100.0,
            overhang_area=50.0,
            print_height=20.0,
            base_normal=np.array([0, 0, -1]),
        )

        with (
            patch.object(pipeline.orientation_optimizer, "optimize", return_value=mock_result),
            patch.object(
                pipeline.orientation_optimizer, "apply_orientation", return_value=MagicMock()
            ) as mock_apply,
        ):
            mock_mesh = MagicMock()
            mock_apply.return_value = mock_mesh

            result = await pipeline._stage_3_optimize_orientation(job)

        assert result is True
        assert job.oriented_stl_path is not None
        assert job.orientation_result == mock_result

    @pytest.mark.asyncio
    async def test_stage_3_optimization_failure_non_fatal(self, pipeline, tmp_path):
        """Test that orientation failure is non-fatal."""
        stl_path = tmp_path / "repaired.stl"
        stl_path.write_bytes(b"mock stl content")

        job = PipelineJob(
            job_id="test123",
            user_id="user_1",
            chat_id=123456789,
            image_path=tmp_path / "test.png",
        )
        job.repaired_stl_path = stl_path

        # Mock failed optimization
        mock_result = OrientationResult(success=False, errors=["Optimization failed"])

        with patch.object(pipeline.orientation_optimizer, "optimize", return_value=mock_result):
            result = await pipeline._stage_3_optimize_orientation(job)

        # Should still succeed (non-fatal)
        assert result is True
        assert job.warnings  # Should have warning
        assert job.oriented_stl_path == job.repaired_stl_path  # Falls back to repaired

    @pytest.mark.asyncio
    async def test_stage_4_success(self, pipeline, tmp_path):
        """Test successful support analysis stage."""
        stl_path = tmp_path / "oriented.stl"
        stl_path.write_bytes(b"mock stl content")

        job = PipelineJob(
            job_id="test123",
            user_id="user_1",
            chat_id=123456789,
            image_path=tmp_path / "test.png",
        )
        job.oriented_stl_path = stl_path

        # Mock support analysis result
        mock_result = SupportAnalysisResult(
            success=True,
            total_support_volume=50.0,
            total_contact_area=10.0,
            overhang_area=20.0,
            printability_score=0.85,
            is_printable=True,
        )

        with patch.object(pipeline.support_analyzer, "analyze", return_value=mock_result):
            result = await pipeline._stage_4_analyze_support(job)

        assert result is True
        assert job.support_result == mock_result
        assert job.metadata["support_analysis"]["printability_score"] == 0.85

    @pytest.mark.asyncio
    async def test_stage_4_not_printable(self, pipeline, tmp_path):
        """Test support analysis when model is not printable."""
        stl_path = tmp_path / "oriented.stl"
        stl_path.write_bytes(b"mock stl content")

        job = PipelineJob(
            job_id="test123",
            user_id="user_1",
            chat_id=123456789,
            image_path=tmp_path / "test.png",
        )
        job.oriented_stl_path = stl_path

        # Mock unprintable result
        mock_result = SupportAnalysisResult(
            success=True,
            total_support_volume=1000.0,
            printability_score=0.2,
            is_printable=False,
            failure_reasons=["Overhang angle too steep"],
        )

        with patch.object(pipeline.support_analyzer, "analyze", return_value=mock_result):
            result = await pipeline._stage_4_analyze_support(job)

        assert result is False
        assert job.status == PipelineStatus.FAILED
        assert "not printable" in job.errors[0]

    @pytest.mark.asyncio
    async def test_stage_5_success(self, pipeline, tmp_path):
        """Test successful preview rendering stage."""
        # Create a mock STL file
        stl_path = tmp_path / "oriented.stl"
        stl_path.write_bytes(b"mock stl content")

        job = PipelineJob(
            job_id="test123",
            user_id="user_1",
            chat_id=123456789,
            image_path=tmp_path / "test.png",
        )
        job.oriented_stl_path = stl_path
        job.repaired_stl_path = stl_path

        # Mock the preview functions
        mock_thumbnail = tmp_path / "test123_preview.png"
        mock_thumbnail.write_bytes(b"mock image data")

        with (
            patch("sensing_voxel.modeling.pipeline.render_thumbnail") as mock_render,
            patch("sensing_voxel.modeling.pipeline.calculate_estimates") as mock_estimates,
        ):
            mock_render.return_value = mock_thumbnail
            mock_estimates.return_value = Mock(
                time_minutes=60.0,
                filament_mm=1000.0,
                filament_grams=25.0,
                layer_height=0.2,
            )

            result = await pipeline._stage_5_render_preview(job)

        assert result is True
        assert job.thumbnail_path is not None
        assert job.estimates["time_minutes"] == 60.0

    @pytest.mark.asyncio
    async def test_stage_6_no_bot(self, pipeline, tmp_path):
        """Test approval stage without telegram bot."""
        job = PipelineJob(
            job_id="test123",
            user_id="user_1",
            chat_id=123456789,
            image_path=tmp_path / "test.png",
        )
        preview_path = tmp_path / "preview.png"
        preview_path.write_bytes(b"mock image")
        job.thumbnail_path = preview_path
        job.estimates = {"time_minutes": 60.0, "filament_mm": 1000.0}

        result = await pipeline._stage_6_wait_for_approval(job)

        assert result is False
        assert "Telegram bot not available" in job.errors[0]

    @pytest.mark.asyncio
    async def test_stage_6_success(self, pipeline, tmp_path, mock_telegram_bot):
        """Test successful approval stage."""
        pipeline.telegram_bot = mock_telegram_bot

        job = PipelineJob(
            job_id="test123",
            user_id="user_1",
            chat_id=123456789,
            image_path=tmp_path / "test.png",
        )
        # Create actual file
        preview_path = tmp_path / "preview.png"
        preview_path.write_bytes(b"mock image")
        job.thumbnail_path = preview_path
        job.estimates = {"time_minutes": 60.0, "filament_mm": 1000.0, "filament_grams": 25.0}
        job.support_result = SupportAnalysisResult(
            success=True,
            total_support_volume=100.0,
            printability_score=0.9,
        )

        result = await pipeline._stage_6_wait_for_approval(job)

        assert result is True
        assert job.approval_message_id == 12345
        assert job.status == PipelineStatus.WAITING_APPROVAL

    @pytest.mark.asyncio
    async def test_stage_7_success(self, pipeline, tmp_path, mock_slicer_pipeline):
        """Test successful slicing stage."""
        from sensing_voxel.core.models import SlicingResult

        stl_path = tmp_path / "oriented.stl"
        stl_path.write_bytes(b"mock stl")

        gcode_path = tmp_path / "test.gcode"
        gcode_path.write_text("G1 X10")

        mock_slicer_pipeline.process_model.return_value = SlicingResult(
            gcode_path=gcode_path,
            gcode_hash="abc123",
        )

        job = PipelineJob(
            job_id="test123",
            user_id="user_1",
            chat_id=123456789,
            image_path=tmp_path / "test.png",
        )
        job.oriented_stl_path = stl_path
        job.repaired_stl_path = stl_path
        job.approved = True

        result = await pipeline._stage_7_slice_model(job)

        assert result is True
        assert job.gcode_path == gcode_path

    @pytest.mark.asyncio
    async def test_stage_7_failure(self, pipeline, tmp_path, mock_slicer_pipeline):
        """Test slicing stage failure."""
        from sensing_voxel.core.models import SlicingResult

        stl_path = tmp_path / "oriented.stl"
        stl_path.write_bytes(b"mock stl")

        mock_slicer_pipeline.process_model.return_value = SlicingResult(
            gcode_path=Path(""),
            gcode_hash="",
            errors=["Slicer not found"],
        )

        job = PipelineJob(
            job_id="test123",
            user_id="user_1",
            chat_id=123456789,
            image_path=tmp_path / "test.png",
        )
        job.oriented_stl_path = stl_path
        job.repaired_stl_path = stl_path

        result = await pipeline._stage_7_slice_model(job)

        assert result is False
        assert job.status == PipelineStatus.FAILED
        assert "Slicing failed" in job.errors[0]

    @pytest.mark.asyncio
    async def test_stage_8_success(self, pipeline_with_printer, tmp_path, mock_printer_adapter):
        """Test successful print start stage."""
        gcode_path = tmp_path / "test.gcode"
        gcode_path.write_text("G1 X10")

        job = PipelineJob(
            job_id="test123",
            user_id="user_1",
            chat_id=123456789,
            image_path=tmp_path / "test.png",
        )
        job.gcode_path = gcode_path

        result = await pipeline_with_printer._stage_8_start_print(job)

        assert result is True
        assert job.status == PipelineStatus.COMPLETED
        mock_printer_adapter.upload_file.assert_called_once()
        mock_printer_adapter.start_print.assert_called_once()

    @pytest.mark.asyncio
    async def test_stage_8_no_adapter(self, pipeline, tmp_path):
        """Test print start without printer adapter."""
        gcode_path = tmp_path / "test.gcode"
        gcode_path.write_text("G1 X10")

        job = PipelineJob(
            job_id="test123",
            user_id="user_1",
            chat_id=123456789,
            image_path=tmp_path / "test.png",
        )
        job.gcode_path = gcode_path

        result = await pipeline._stage_8_start_print(job)

        assert result is False
        assert "Printer adapter not available" in job.errors[0]

    @pytest.mark.asyncio
    async def test_full_pipeline_through_approval(
        self, pipeline, temp_image_file, mock_image_to_3d_service, tmp_path
    ):
        """Test full pipeline through stage 6 (approval)."""
        # Setup mocks
        mock_image_to_3d_service.generate.return_value = GenerationResult(
            success=True,
            model_data=b"mock stl",
            generation_id="gen_123",
        )

        def mock_repair(input_path, output_path):
            output_path.write_bytes(b"mock stl content")
            return True

        def mock_render(input_path, output_path):
            output_path.write_bytes(b"mock image data")
            return output_path

        with (
            patch("sensing_voxel.modeling.pipeline.repair_mesh", side_effect=mock_repair),
            patch("sensing_voxel.modeling.pipeline.render_thumbnail") as mock_render_fn,
            patch("sensing_voxel.modeling.pipeline.calculate_estimates") as mock_estimates,
        ):
            mock_render_fn.side_effect = mock_render
            mock_estimates.return_value = Mock(
                time_minutes=60.0,
                filament_mm=1000.0,
                filament_grams=25.0,
                layer_height=0.2,
            )

            # Add telegram bot to pipeline
            pipeline.telegram_bot = MagicMock()
            pipeline.telegram_bot.send_photo = AsyncMock(return_value=MagicMock(message_id=12345))

            result = await pipeline.process_photo_to_print(
                image_path=temp_image_file,
                telegram_chat_id=123456789,
                user_id="user_1",
            )

        assert result.success is True
        assert result.job.status == PipelineStatus.WAITING_APPROVAL
        assert result.job.approval_message_id == 12345

    @pytest.mark.asyncio
    async def test_pipeline_with_missing_image(self, pipeline, tmp_path):
        """Test pipeline with non-existent image file."""
        result = await pipeline.process_photo_to_print(
            image_path=tmp_path / "nonexistent.png",
            telegram_chat_id=123456789,
            user_id="user_1",
        )

        assert result.success is False
        assert "not found" in result.message

    @pytest.mark.asyncio
    async def test_pipeline_auto_approve(
        self, pipeline_with_printer, temp_image_file, mock_image_to_3d_service, tmp_path
    ):
        """Test pipeline with auto_approve enabled."""
        mock_image_to_3d_service.generate.return_value = GenerationResult(
            success=True,
            model_data=b"mock stl",
            generation_id="gen_123",
        )

        gcode_path = tmp_path / "test.gcode"
        gcode_path.write_text("G1 X10")

        from sensing_voxel.core.models import SlicingResult

        pipeline_with_printer.slicer_pipeline.process_model.return_value = SlicingResult(
            gcode_path=gcode_path,
            gcode_hash="abc123",
        )

        def mock_repair(input_path, output_path):
            output_path.write_bytes(b"mock stl content")
            return True

        def mock_render(input_path, output_path):
            output_path.write_bytes(b"mock image data")
            return output_path

        with (
            patch("sensing_voxel.modeling.pipeline.repair_mesh", side_effect=mock_repair),
            patch("sensing_voxel.modeling.pipeline.render_thumbnail") as mock_render_fn,
            patch("sensing_voxel.modeling.pipeline.calculate_estimates") as mock_estimates,
        ):
            mock_render_fn.side_effect = mock_render
            mock_estimates.return_value = Mock(
                time_minutes=60.0,
                filament_mm=1000.0,
                filament_grams=25.0,
                layer_height=0.2,
            )

            result = await pipeline_with_printer.process_photo_to_print(
                image_path=temp_image_file,
                telegram_chat_id=123456789,
                user_id="user_1",
                auto_approve=True,
            )

        assert result.success is True
        assert result.job.status == PipelineStatus.COMPLETED

    def test_get_job(self, pipeline, tmp_path):
        """Test retrieving a job by ID."""
        job = PipelineJob(
            job_id="test123",
            user_id="user_1",
            chat_id=123456789,
            image_path=tmp_path / "test.png",
        )
        pipeline._jobs["test123"] = job

        retrieved = pipeline.get_job("test123")
        assert retrieved == job

        not_found = pipeline.get_job("nonexistent")
        assert not_found is None

    def test_list_jobs(self, pipeline, tmp_path):
        """Test listing all jobs."""
        job1 = PipelineJob(
            job_id="job1",
            user_id="user_1",
            chat_id=123456789,
            image_path=tmp_path / "test1.png",
        )
        job2 = PipelineJob(
            job_id="job2",
            user_id="user_2",
            chat_id=123456789,
            image_path=tmp_path / "test2.png",
        )
        pipeline._jobs["job1"] = job1
        pipeline._jobs["job2"] = job2

        all_jobs = pipeline.list_jobs()
        assert len(all_jobs) == 2

        user_jobs = pipeline.list_jobs(user_id="user_1")
        assert len(user_jobs) == 1
        assert user_jobs[0].job_id == "job1"

    def test_get_pending_approvals(self, pipeline, tmp_path):
        """Test getting jobs waiting for approval."""
        job1 = PipelineJob(
            job_id="job1",
            user_id="user_1",
            chat_id=123456789,
            image_path=tmp_path / "test1.png",
        )
        job1.status = PipelineStatus.WAITING_APPROVAL

        job2 = PipelineJob(
            job_id="job2",
            user_id="user_1",
            chat_id=123456789,
            image_path=tmp_path / "test2.png",
        )
        job2.status = PipelineStatus.COMPLETED

        pipeline._jobs["job1"] = job1
        pipeline._jobs["job2"] = job2

        pending = pipeline.get_pending_approvals()
        assert len(pending) == 1
        assert pending[0].job_id == "job1"

    @pytest.mark.asyncio
    async def test_approve_job(self, pipeline_with_printer, tmp_path):
        """Test approving a job."""
        stl_path = tmp_path / "repaired.stl"
        stl_path.write_bytes(b"mock stl")

        gcode_path = tmp_path / "test.gcode"
        gcode_path.write_text("G1 X10")

        from sensing_voxel.core.models import SlicingResult

        pipeline_with_printer.slicer_pipeline.process_model.return_value = SlicingResult(
            gcode_path=gcode_path,
            gcode_hash="abc123",
        )

        job = PipelineJob(
            job_id="test123",
            user_id="user_1",
            chat_id=123456789,
            image_path=tmp_path / "test.png",
        )
        job.status = PipelineStatus.WAITING_APPROVAL
        job.oriented_stl_path = stl_path
        job.repaired_stl_path = stl_path
        pipeline_with_printer._jobs["test123"] = job
        pipeline_with_printer._approval_events["test123"] = asyncio.Event()

        result = await pipeline_with_printer.approve_job("test123", scale_factor=1.5)

        assert result.success is True
        assert job.approved is True
        assert job.scale_factor == 1.5

    def test_reject_job(self, pipeline, tmp_path):
        """Test rejecting a job."""
        job = PipelineJob(
            job_id="test123",
            user_id="user_1",
            chat_id=123456789,
            image_path=tmp_path / "test.png",
        )
        job.status = PipelineStatus.WAITING_APPROVAL
        pipeline._jobs["test123"] = job
        pipeline._approval_events["test123"] = asyncio.Event()

        result = pipeline.reject_job("test123")

        assert result is True
        assert job.rejected is True
        assert job.status == PipelineStatus.REJECTED

    def test_cancel_job(self, pipeline, tmp_path):
        """Test cancelling a job."""
        job = PipelineJob(
            job_id="test123",
            user_id="user_1",
            chat_id=123456789,
            image_path=tmp_path / "test.png",
        )
        job.status = PipelineStatus.WAITING_APPROVAL
        pipeline._jobs["test123"] = job
        pipeline._approval_events["test123"] = asyncio.Event()

        result = pipeline.cancel_job("test123")

        assert result is True
        assert job.cancelled is True
        assert job.status == PipelineStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_approve_nonexistent_job(self, pipeline):
        """Test approving a job that doesn't exist."""
        result = await pipeline.approve_job("nonexistent")

        assert result.success is False
        assert "not found" in result.message

    def test_get_job_summary(self, pipeline, tmp_path):
        """Test getting job summary."""
        job = PipelineJob(
            job_id="test123",
            user_id="user_1",
            chat_id=123456789,
            image_path=tmp_path / "test.png",
        )
        job.estimates = {"time_minutes": 60.0}
        pipeline._jobs["test123"] = job

        summary = pipeline.get_job_summary("test123")

        assert summary is not None
        assert summary["job_id"] == "test123"
        assert summary["status"] == "pending"
        assert summary["estimates"]["time_minutes"] == 60.0

    def test_get_job_summary_not_found(self, pipeline):
        """Test getting summary for non-existent job."""
        summary = pipeline.get_job_summary("nonexistent")
        assert summary is None


class TestProcessPhotoToPrintFunction:
    """Tests for the standalone process_photo_to_print function."""

    @pytest.mark.asyncio
    async def test_standalone_function(
        self, tmp_path, mock_image_to_3d_service, mock_printer_adapter
    ):
        """Test the standalone convenience function."""
        img_path = tmp_path / "test.png"
        Image.new("RGB", (512, 512), color="red").save(img_path)

        mock_image_to_3d_service.generate.return_value = GenerationResult(
            success=True,
            model_data=b"mock stl",
            generation_id="gen_123",
        )

        gcode_path = tmp_path / "test.gcode"
        gcode_path.write_text("G1 X10")

        from sensing_voxel.core.models import SlicingResult

        mock_slicer = MagicMock(spec=SlicerPipeline)
        mock_slicer.process_model = AsyncMock(
            return_value=SlicingResult(gcode_path=gcode_path, gcode_hash="abc123")
        )

        def mock_repair(input_path, output_path):
            output_path.write_bytes(b"mock stl content")
            return True

        def mock_render(input_path, output_path):
            output_path.write_bytes(b"mock image data")
            return output_path

        with (
            patch("sensing_voxel.modeling.pipeline.repair_mesh", side_effect=mock_repair),
            patch("sensing_voxel.modeling.pipeline.render_thumbnail") as mock_render_fn,
            patch("sensing_voxel.modeling.pipeline.calculate_estimates") as mock_estimates,
            patch("sensing_voxel.modeling.pipeline.OrientationOptimizer") as mock_optimizer_class,
            patch("sensing_voxel.modeling.pipeline.SupportAnalyzer") as mock_analyzer_class,
        ):
            mock_render_fn.side_effect = mock_render
            mock_estimates.return_value = Mock(
                time_minutes=60.0,
                filament_mm=1000.0,
                filament_grams=25.0,
                layer_height=0.2,
            )

            # Mock orientation optimizer
            mock_optimizer = MagicMock()
            mock_optimizer.optimize.return_value = OrientationResult(success=True, score=15.0)

            def mock_export(path):
                path.write_bytes(b"mock oriented stl content")

            mock_oriented_mesh = MagicMock()
            mock_oriented_mesh.export = mock_export
            mock_optimizer.apply_orientation.return_value = mock_oriented_mesh
            mock_optimizer_class.return_value = mock_optimizer

            # Mock support analyzer
            mock_analyzer = MagicMock()
            mock_analyzer.analyze.return_value = SupportAnalysisResult(
                success=True,
                total_support_volume=100.0,
                printability_score=0.85,
                is_printable=True,
            )
            mock_analyzer_class.return_value = mock_analyzer

            result = await process_photo_to_print(
                image_path=img_path,
                telegram_chat_id=123456789,
                user_id="user_1",
                image_to_3d_service=mock_image_to_3d_service,
                slicer_pipeline=mock_slicer,
                printer_adapter=mock_printer_adapter,
                telegram_bot=MagicMock(),
                output_dir=tmp_path / "output",
                auto_approve=True,
            )

        assert result.success is True
        assert result.job.status == PipelineStatus.COMPLETED


class TestErrorRecovery:
    """Tests for error handling and recovery."""

    @pytest.mark.asyncio
    async def test_telegram_notification_on_failure(
        self, pipeline, temp_image_file, mock_image_to_3d_service
    ):
        """Test that Telegram notification is sent on stage failure."""
        mock_image_to_3d_service.generate.return_value = GenerationResult(
            success=False,
            error_message="API error",
        )

        notifications = []

        def capture_notification(chat_id, message):
            notifications.append((chat_id, message))

        pipeline.on_telegram_notification = capture_notification

        job = PipelineJob(
            job_id="test123",
            user_id="user_1",
            chat_id=123456789,
            image_path=temp_image_file,
        )

        result = await pipeline._stage_1_image_to_3d(job)

        assert result is False
        assert len(notifications) == 1
        assert "❌" in notifications[0][1]

    @pytest.mark.asyncio
    async def test_status_update_callback(
        self, pipeline, temp_image_file, mock_image_to_3d_service
    ):
        """Test status update callback is called."""
        mock_image_to_3d_service.generate.return_value = GenerationResult(
            success=True,
            model_data=b"mock stl",
            generation_id="gen_123",
        )

        status_updates = []

        def capture_status(job, message):
            status_updates.append((job.job_id, message))

        pipeline.on_status_update = capture_status

        job = PipelineJob(
            job_id="test123",
            user_id="user_1",
            chat_id=123456789,
            image_path=temp_image_file,
        )

        result = await pipeline._stage_1_image_to_3d(job)

        assert result is True
        assert len(status_updates) >= 2  # Starting + completed

    @pytest.mark.asyncio
    async def test_job_warnings_accumulation(self, pipeline, tmp_path):
        """Test that warnings accumulate on non-failures."""
        job = PipelineJob(
            job_id="test123",
            user_id="user_1",
            chat_id=123456789,
            image_path=tmp_path / "test.png",
        )
        job.warnings.append("Warning 1")
        job.warnings.append("Warning 2")

        assert len(job.warnings) == 2
        assert "Warning 1" in job.warnings
