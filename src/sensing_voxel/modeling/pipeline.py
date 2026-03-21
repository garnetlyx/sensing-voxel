"""
End-to-End Photo-to-Print Pipeline for Sensing Voxel.

8-Stage Orchestration:
- Stage 1: Image-to-3D generation (image_to_3d.generate)
- Stage 2: Mesh repair (repair.repair_mesh)
- Stage 3: Optimize orientation (orientation.optimize)
- Stage 4: Analyze support requirements (support_analysis.analyze)
- Stage 5: Generate preview (preview.render_thumbnail)
- Stage 6: Wait for approval (Telegram interaction)
- Stage 7: Slice model (slicer.process_model)
- Stage 8: Start print (printer adapter)

Error handling and recovery at each stage with Telegram notifications.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from sensing_voxel.adapters.base import PrinterAdapter
from sensing_voxel.modeling.image_to_3d import GenerationResult, ImageTo3DService
from sensing_voxel.modeling.orientation import OrientationOptimizer, OrientationResult
from sensing_voxel.modeling.preview import (
    calculate_estimates,
    render_thumbnail,
    send_approval_message,
)
from sensing_voxel.modeling.repair import RepairResult, repair_mesh
from sensing_voxel.modeling.support_analysis import (
    SupportAnalysisResult,
    SupportAnalyzer,
)
from sensing_voxel.slicer.pipeline import SlicerPipeline

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline execution stages."""

    PENDING = auto()
    IMAGE_TO_3D = auto()
    REPAIR_MESH = auto()
    OPTIMIZE_ORIENTATION = auto()
    ANALYZE_SUPPORT = auto()
    RENDER_PREVIEW = auto()
    WAITING_APPROVAL = auto()
    SLICE_MODEL = auto()
    START_PRINT = auto()
    COMPLETED = auto()
    FAILED = auto()


class PipelineStatus(Enum):
    """Pipeline execution status."""

    PENDING = "pending"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineJob:
    """Tracks a photo-to-print pipeline job."""

    job_id: str
    user_id: int | str
    chat_id: int
    image_path: Path
    status: PipelineStatus = PipelineStatus.PENDING
    stage: PipelineStage = PipelineStage.PENDING
    stl_path: Path | None = None
    repaired_stl_path: Path | None = None
    oriented_stl_path: Path | None = None
    thumbnail_path: Path | None = None
    gcode_path: Path | None = None
    estimates: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    approval_message_id: int | None = None
    approved: bool = False
    rejected: bool = False
    cancelled: bool = False
    scale_factor: float = 1.0
    metadata: dict = field(default_factory=dict)
    orientation_result: OrientationResult | None = None
    support_result: SupportAnalysisResult | None = None
    repair_result: RepairResult | None = None

    def to_dict(self) -> dict:
        """Convert job to dictionary for serialization."""
        return {
            "job_id": self.job_id,
            "user_id": str(self.user_id),
            "chat_id": self.chat_id,
            "image_path": str(self.image_path),
            "status": self.status.value,
            "stage": self.stage.name,
            "stl_path": str(self.stl_path) if self.stl_path else None,
            "repaired_stl_path": str(self.repaired_stl_path) if self.repaired_stl_path else None,
            "oriented_stl_path": str(self.oriented_stl_path) if self.oriented_stl_path else None,
            "thumbnail_path": str(self.thumbnail_path) if self.thumbnail_path else None,
            "gcode_path": str(self.gcode_path) if self.gcode_path else None,
            "estimates": self.estimates,
            "errors": self.errors,
            "warnings": self.warnings,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "approval_message_id": self.approval_message_id,
            "approved": self.approved,
            "rejected": self.rejected,
            "cancelled": self.cancelled,
            "scale_factor": self.scale_factor,
            "metadata": self.metadata,
        }


@dataclass
class PipelineResult:
    """Result of pipeline execution."""

    success: bool
    job: PipelineJob
    message: str = ""
    recovery_attempted: bool = False
    recovery_successful: bool = False


class PhotoToPrintPipeline:
    """
    End-to-end photo-to-print pipeline orchestrator.

    Orchestrates the 8-stage workflow:
    1. Image-to-3D generation
    2. Mesh repair
    3. Optimize orientation
    4. Analyze support
    5. Generate preview
    6. Wait for approval
    7. Slice model
    8. Start print
    """

    def __init__(
        self,
        image_to_3d_service: ImageTo3DService,
        slicer_pipeline: SlicerPipeline,
        printer_adapter: PrinterAdapter | None = None,
        telegram_bot: Any | None = None,
        output_dir: Path | None = None,
        on_status_update: Callable[[PipelineJob, str], None] | None = None,
        on_telegram_notification: Callable[[int, str], None] | None = None,
        max_retries: int = 2,
    ):
        self.image_to_3d_service = image_to_3d_service
        self.slicer_pipeline = slicer_pipeline
        self.printer_adapter = printer_adapter
        self.telegram_bot = telegram_bot
        self.output_dir = output_dir or Path("data/pipeline")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.on_status_update = on_status_update
        self.on_telegram_notification = on_telegram_notification
        self.max_retries = max_retries

        # Job tracking
        self._jobs: dict[str, PipelineJob] = {}
        self._approval_events: dict[str, asyncio.Event] = {}

        # Optimizers
        self.orientation_optimizer = OrientationOptimizer()
        self.support_analyzer = SupportAnalyzer()

    def _update_status(self, job: PipelineJob, message: str) -> None:
        """Update job status and notify callback if set."""
        job.updated_at = datetime.now()
        logger.info(f"[Job {job.job_id}] {message}")
        if self.on_status_update:
            try:
                self.on_status_update(job, message)
            except Exception as e:
                logger.warning(f"Status update callback failed: {e}")

    def _set_error(self, job: PipelineJob, error: str) -> None:
        """Set job error state."""
        job.errors.append(error)
        job.status = PipelineStatus.FAILED
        job.stage = PipelineStage.FAILED
        self._update_status(job, f"ERROR: {error}")

    async def _send_telegram_notification(self, chat_id: int, message: str) -> None:
        """Send Telegram notification if bot is available."""
        if self.on_telegram_notification:
            try:
                self.on_telegram_notification(chat_id, message)
            except Exception as e:
                logger.warning(f"Telegram notification failed: {e}")

    async def _stage_1_image_to_3d(self, job: PipelineJob) -> bool:
        """
        Stage 1: Generate 3D model from image.

        Uses image_to_3d.generate() to create an STL from the input image.
        """
        job.stage = PipelineStage.IMAGE_TO_3D
        self._update_status(job, "Stage 1/8: Generating 3D model from image...")

        for attempt in range(self.max_retries + 1):
            try:
                # Load image
                image = Image.open(job.image_path)

                # Generate 3D model
                result: GenerationResult = await self.image_to_3d_service.generate(
                    image=image,
                    user_id=job.user_id,
                    job_id=job.job_id,
                )

                if not result.success:
                    if attempt < self.max_retries:
                        await asyncio.sleep(2**attempt)  # Exponential backoff
                        continue
                    self._set_error(job, f"Image-to-3D generation failed: {result.error_message}")
                    await self._send_telegram_notification(
                        job.chat_id, f"❌ Image-to-3D generation failed: {result.error_message}"
                    )
                    return False

                # Save model data to STL file
                stl_path = self.output_dir / f"{job.job_id}_generated.stl"
                if result.model_data:
                    stl_path.write_bytes(result.model_data)
                elif result.model_url:
                    # Download from URL if needed
                    import aiohttp

                    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                        async with session.get(result.model_url) as response:
                            if response.status == 200:
                                stl_path.write_bytes(await response.read())
                            else:
                                if attempt < self.max_retries:
                                    await asyncio.sleep(2**attempt)
                                    continue
                                self._set_error(
                                    job, f"Failed to download model: HTTP {response.status}"
                                )
                                await self._send_telegram_notification(
                                    job.chat_id,
                                    f"❌ Failed to download model: HTTP {response.status}",
                                )
                                return False
                else:
                    if attempt < self.max_retries:
                        await asyncio.sleep(2**attempt)
                        continue
                    self._set_error(job, "No model data or URL returned from image-to-3D service")
                    await self._send_telegram_notification(
                        job.chat_id, "❌ No model data received from image-to-3D service"
                    )
                    return False

                job.stl_path = stl_path
                job.metadata["generation_result"] = {
                    "success": result.success,
                    "provider": result.provider.name if result.provider else None,
                    "generation_id": result.generation_id,
                }
                self._update_status(job, f"Stage 1 complete: STL saved to {stl_path.name}")
                await self._send_telegram_notification(
                    job.chat_id, "✅ Stage 1/8: 3D model generated successfully"
                )
                return True

            except Exception as e:
                if attempt < self.max_retries:
                    await asyncio.sleep(2**attempt)
                    continue
                self._set_error(job, f"Image-to-3D stage failed: {e}")
                await self._send_telegram_notification(
                    job.chat_id, f"❌ Image-to-3D stage failed: {str(e)[:100]}"
                )
                return False

        return False

    async def _stage_2_repair_mesh(self, job: PipelineJob) -> bool:
        """
        Stage 2: Repair the generated mesh.

        Uses repair.repair_mesh() to fix mesh issues.
        """
        job.stage = PipelineStage.REPAIR_MESH
        self._update_status(job, "Stage 2/8: Repairing mesh...")

        if not job.stl_path or not job.stl_path.exists():
            self._set_error(job, "No STL file available for repair")
            await self._send_telegram_notification(
                job.chat_id, "❌ No STL file available for repair"
            )
            return False

        try:
            # Repair mesh using existing function
            repaired_path = self.output_dir / f"{job.job_id}_repaired.stl"
            success = repair_mesh(job.stl_path, repaired_path)

            if not success:
                self._set_error(job, "Mesh repair failed - model may not be printable")
                await self._send_telegram_notification(
                    job.chat_id, "❌ Mesh repair failed - model may not be printable"
                )
                return False

            job.repaired_stl_path = repaired_path
            self._update_status(job, "Stage 2 complete: Repaired mesh saved")
            await self._send_telegram_notification(
                job.chat_id, "✅ Stage 2/8: Mesh repaired successfully"
            )
            return True

        except Exception as e:
            self._set_error(job, f"Mesh repair stage failed: {e}")
            await self._send_telegram_notification(
                job.chat_id, f"❌ Mesh repair stage failed: {str(e)[:100]}"
            )
            return False

    async def _stage_3_optimize_orientation(self, job: PipelineJob) -> bool:
        """
        Stage 3: Optimize orientation for printing.

        Uses orientation optimizer to find best print direction.
        """
        job.stage = PipelineStage.OPTIMIZE_ORIENTATION
        self._update_status(job, "Stage 3/8: Optimizing orientation...")

        if not job.repaired_stl_path or not job.repaired_stl_path.exists():
            self._set_error(job, "No repaired STL available for orientation optimization")
            await self._send_telegram_notification(
                job.chat_id, "❌ No repaired STL available for orientation optimization"
            )
            return False

        try:
            import trimesh

            # Load mesh and optimize
            mesh = trimesh.load_mesh(job.repaired_stl_path, force="mesh")
            result: OrientationResult = self.orientation_optimizer.optimize(mesh)

            if not result.success:
                # Non-fatal: continue with original orientation
                job.warnings.append(f"Orientation optimization failed: {result.errors}")
                job.oriented_stl_path = job.repaired_stl_path
                self._update_status(
                    job, "Stage 3: Using original orientation (optimization failed)"
                )
                await self._send_telegram_notification(
                    job.chat_id,
                    "⚠️ Stage 3/8: Orientation optimization skipped, using original orientation",
                )
                return True

            # Apply optimal orientation
            oriented_mesh = self.orientation_optimizer.apply_orientation(mesh, result)
            oriented_path = self.output_dir / f"{job.job_id}_oriented.stl"
            oriented_mesh.export(oriented_path)

            job.oriented_stl_path = oriented_path
            job.orientation_result = result
            job.metadata["orientation"] = {
                "score": result.score,
                "support_volume": result.support_volume,
                "overhang_area": result.overhang_area,
                "print_height": result.print_height,
            }

            self._update_status(
                job,
                f"Stage 3 complete: Orientation optimized (score: {result.score:.2f})",
            )
            await self._send_telegram_notification(
                job.chat_id,
                f"✅ Stage 3/8: Orientation optimized (support volume: {result.support_volume:.1f}mm³)",
            )
            return True

        except Exception as e:
            # Non-fatal: continue with original orientation
            job.warnings.append(f"Orientation optimization error: {e}")
            job.oriented_stl_path = job.repaired_stl_path
            self._update_status(job, f"Stage 3: Using original orientation (error: {e})")
            await self._send_telegram_notification(
                job.chat_id,
                "⚠️ Stage 3/8: Orientation optimization skipped, using original orientation",
            )
            return True

    async def _stage_4_analyze_support(self, job: PipelineJob) -> bool:
        """
        Stage 4: Analyze support requirements.

        Uses support analyzer to check printability.
        """
        job.stage = PipelineStage.ANALYZE_SUPPORT
        self._update_status(job, "Stage 4/8: Analyzing support requirements...")

        if not job.oriented_stl_path or not job.oriented_stl_path.exists():
            self._set_error(job, "No STL available for support analysis")
            await self._send_telegram_notification(
                job.chat_id, "❌ No STL available for support analysis"
            )
            return False

        try:
            # Analyze support requirements
            base_normal = None
            if job.orientation_result and job.orientation_result.base_normal is not None:
                base_normal = job.orientation_result.base_normal

            result: SupportAnalysisResult = self.support_analyzer.analyze(
                job.oriented_stl_path,
                base_normal=base_normal,
            )

            job.support_result = result
            job.metadata["support_analysis"] = {
                "total_support_volume": result.total_support_volume,
                "total_contact_area": result.total_contact_area,
                "overhang_area": result.overhang_area,
                "printability_score": result.printability_score,
                "is_printable": result.is_printable,
            }

            # Add recommendations as warnings if printability is low
            if result.recommendations:
                job.warnings.extend(result.recommendations[:3])

            if not result.is_printable:
                # Critical failure - cannot print
                error_msg = f"Model not printable: {', '.join(result.failure_reasons)}"
                self._set_error(job, error_msg)
                await self._send_telegram_notification(job.chat_id, f"❌ {error_msg}")
                return False

            self._update_status(
                job,
                f"Stage 4 complete: Support analysis done (printability: {result.printability_score:.2f})",
            )
            await self._send_telegram_notification(
                job.chat_id,
                f"✅ Stage 4/8: Support analysis complete - model is printable!",
            )
            return True

        except Exception as e:
            # Non-fatal: continue without analysis
            job.warnings.append(f"Support analysis error: {e}")
            self._update_status(job, f"Stage 4: Support analysis skipped (error: {e})")
            await self._send_telegram_notification(
                job.chat_id,
                "⚠️ Stage 4/8: Support analysis skipped",
            )
            return True

    async def _stage_5_render_preview(self, job: PipelineJob) -> bool:
        """
        Stage 5: Render thumbnail preview.

        Uses preview.render_thumbnail() to create a preview image.
        """
        job.stage = PipelineStage.RENDER_PREVIEW
        self._update_status(job, "Stage 5/8: Rendering preview...")

        stl_to_render = job.oriented_stl_path or job.repaired_stl_path
        if not stl_to_render or not stl_to_render.exists():
            self._set_error(job, "No STL available for preview")
            await self._send_telegram_notification(
                job.chat_id, "❌ No STL available for preview rendering"
            )
            return False

        try:
            # Render thumbnail
            thumbnail_path = self.output_dir / f"{job.job_id}_preview.png"
            render_thumbnail(stl_to_render, thumbnail_path)

            # Calculate print estimates
            estimates = calculate_estimates(stl_to_render)
            job.estimates = {
                "time_minutes": estimates.time_minutes,
                "filament_mm": estimates.filament_mm,
                "filament_grams": estimates.filament_grams,
                "layer_height": estimates.layer_height,
            }

            job.thumbnail_path = thumbnail_path
            self._update_status(job, "Stage 5 complete: Preview rendered")
            await self._send_telegram_notification(
                job.chat_id,
                f"✅ Stage 5/8: Preview ready - Est. time: {estimates.time_minutes:.0f}min",
            )
            return True

        except Exception as e:
            self._set_error(job, f"Preview rendering stage failed: {e}")
            await self._send_telegram_notification(
                job.chat_id, f"❌ Preview rendering failed: {str(e)[:100]}"
            )
            return False

    async def _stage_6_wait_for_approval(self, job: PipelineJob) -> bool:
        """
        Stage 6: Send Telegram approval message and wait for user response.

        Uses preview.send_approval_message() to request user approval.
        """
        job.stage = PipelineStage.WAITING_APPROVAL
        self._update_status(job, "Stage 6/8: Sending approval request...")

        if not self.telegram_bot:
            self._set_error(job, "Telegram bot not available for approval")
            await self._send_telegram_notification(
                job.chat_id, "❌ Telegram bot not available for approval"
            )
            return False

        if not job.thumbnail_path or not job.thumbnail_path.exists():
            self._set_error(job, "No preview image available for approval")
            await self._send_telegram_notification(
                job.chat_id, "❌ No preview image available for approval"
            )
            return False

        try:
            from sensing_voxel.modeling.preview import PrintEstimates

            estimates = PrintEstimates(**job.estimates)

            # Build detailed caption with support info
            caption_lines = [
                f"📐 *{job.image_path.stem}*\n",
                f"⏱️ Print time: *{estimates.time_minutes:.0f} min*",
                f"🧵 Filament: *{estimates.filament_mm:.0f}mm*",
            ]
            if estimates.filament_grams:
                caption_lines.append(f"⚖️ Weight: *{estimates.filament_grams:.1f}g*")
            if job.support_result:
                caption_lines.append(
                    f"📊 Printability: *{job.support_result.printability_score:.0%}*"
                )
                if job.support_result.total_support_volume > 0:
                    caption_lines.append(
                        f"🔧 Support volume: *{job.support_result.total_support_volume / 1000:.1f}cm³*"
                    )
            if job.warnings:
                caption_lines.append(f"\n⚠️ Warnings: {len(job.warnings)}")
            caption_lines.append("\nApprove to start printing?")

            caption = "\n".join(caption_lines)

            # Send approval message with photo
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup

            keyboard = [
                [
                    InlineKeyboardButton("✅ Approve", callback_data=f"approve:{job.job_id}"),
                    InlineKeyboardButton("❌ Reject", callback_data=f"reject:{job.job_id}"),
                ],
                [
                    InlineKeyboardButton("🔍 Scale 150%", callback_data=f"scale:{job.job_id}:1.5"),
                    InlineKeyboardButton("🔍 Scale 75%", callback_data=f"scale:{job.job_id}:0.75"),
                ],
            ]

            with open(job.thumbnail_path, "rb") as photo:
                message = await self.telegram_bot.send_photo(
                    chat_id=job.chat_id,
                    photo=photo,
                    caption=caption,
                    parse_mode="Markdown",
                    reply_markup=InlineKeyboardMarkup(keyboard),
                )

            job.approval_message_id = message.message_id
            job.status = PipelineStatus.WAITING_APPROVAL
            self._update_status(
                job, f"Stage 6 complete: Approval message sent (ID: {message.message_id})"
            )
            return True

        except Exception as e:
            self._set_error(job, f"Approval request failed: {e}")
            await self._send_telegram_notification(
                job.chat_id, f"❌ Approval request failed: {str(e)[:100]}"
            )
            return False

    async def _stage_7_slice_model(self, job: PipelineJob) -> bool:
        """
        Stage 7: Slice the model to G-code.

        Uses slicer pipeline to generate G-code.
        """
        job.stage = PipelineStage.SLICE_MODEL
        self._update_status(job, "Stage 7/8: Slicing model...")

        stl_to_slice = job.oriented_stl_path or job.repaired_stl_path
        if not stl_to_slice or not stl_to_slice.exists():
            self._set_error(job, "No STL available for slicing")
            await self._send_telegram_notification(job.chat_id, "❌ No STL available for slicing")
            return False

        try:
            # Slice the model
            result = await self.slicer_pipeline.process_model(
                model_path=stl_to_slice,
                output_name=job.job_id,
            )

            if result.errors:
                error_msg = f"Slicing failed: {'; '.join(result.errors)}"
                self._set_error(job, error_msg)
                await self._send_telegram_notification(job.chat_id, f"❌ {error_msg[:200]}")
                return False

            job.gcode_path = result.gcode_path
            self._update_status(
                job, f"Stage 7 complete: G-code generated at {result.gcode_path.name}"
            )
            await self._send_telegram_notification(
                job.chat_id,
                f"✅ Stage 7/8: G-code generated successfully",
            )
            return True

        except Exception as e:
            self._set_error(job, f"Slicing stage failed: {e}")
            await self._send_telegram_notification(
                job.chat_id, f"❌ Slicing stage failed: {str(e)[:100]}"
            )
            return False

    async def _stage_8_start_print(self, job: PipelineJob) -> bool:
        """
        Stage 8: Upload G-code and start print.

        Uses printer adapter to upload file and start printing.
        """
        job.stage = PipelineStage.START_PRINT
        self._update_status(job, "Stage 8/8: Starting print...")

        if not job.gcode_path or not job.gcode_path.exists():
            self._set_error(job, "No G-code available for print")
            await self._send_telegram_notification(job.chat_id, "❌ No G-code available for print")
            return False

        if not self.printer_adapter:
            self._set_error(job, "Printer adapter not available")
            await self._send_telegram_notification(job.chat_id, "❌ Printer adapter not available")
            return False

        try:
            from sensing_voxel.core.models import CommandResultStatus

            # Upload file to printer
            remote_name = f"{job.job_id}.gcode"
            upload_result = await self.printer_adapter.upload_file(str(job.gcode_path), remote_name)

            if upload_result.status != CommandResultStatus.SUCCESS:
                if upload_result.status == CommandResultStatus.NOT_SUPPORTED:
                    # Some adapters don't support file upload - try direct print
                    logger.info("File upload not supported, attempting direct print")
                else:
                    error_msg = f"Failed to upload G-code: {upload_result.message}"
                    self._set_error(job, error_msg)
                    await self._send_telegram_notification(job.chat_id, f"❌ {error_msg[:200]}")
                    return False

            # Start print
            start_result = await self.printer_adapter.start_print(remote_name)

            if start_result.status != CommandResultStatus.SUCCESS:
                error_msg = f"Failed to start print: {start_result.message}"
                self._set_error(job, error_msg)
                await self._send_telegram_notification(job.chat_id, f"❌ {error_msg[:200]}")
                return False

            job.status = PipelineStatus.COMPLETED
            job.stage = PipelineStage.COMPLETED
            self._update_status(job, "Stage 8 complete: Print started successfully!")
            await self._send_telegram_notification(
                job.chat_id,
                f"🎉 Stage 8/8: Print started successfully!\nJob ID: `{job.job_id}`",
            )
            return True

        except Exception as e:
            self._set_error(job, f"Print start failed: {e}")
            await self._send_telegram_notification(
                job.chat_id, f"❌ Print start failed: {str(e)[:100]}"
            )
            return False

    async def process_photo_to_print(
        self,
        image_path: Path | str,
        telegram_chat_id: int,
        user_id: int | str,
        context: dict[str, Any] | None = None,
        auto_approve: bool = False,
    ) -> PipelineResult:
        """
        Process a photo through the full photo-to-print pipeline.

        Args:
            image_path: Path to the input image
            telegram_chat_id: Telegram chat ID for approval messages
            user_id: User ID for tracking and cost attribution
            context: Optional additional context (e.g., scale factor, print settings)
            auto_approve: If True, skip approval stage (for testing)

        Returns:
            PipelineResult with job status and details

        Example:
            result = await pipeline.process_photo_to_print(
                image_path="/path/to/photo.jpg",
                telegram_chat_id=123456789,
                user_id="user_123",
            )
            if result.success:
                print(f"Job {result.job.job_id} created, waiting for approval")
        """
        image_path = Path(image_path)

        if not image_path.exists():
            return PipelineResult(
                success=False,
                job=PipelineJob(
                    job_id="",
                    user_id=user_id,
                    chat_id=telegram_chat_id,
                    image_path=image_path,
                ),
                message=f"Image file not found: {image_path}",
            )

        # Create job
        job = PipelineJob(
            job_id=str(uuid.uuid4())[:8],
            user_id=user_id,
            chat_id=telegram_chat_id,
            image_path=image_path,
            metadata=context or {},
        )
        self._jobs[job.job_id] = job
        self._approval_events[job.job_id] = asyncio.Event()

        self._update_status(job, f"Starting photo-to-print pipeline for {image_path.name}")
        await self._send_telegram_notification(
            job.chat_id,
            f"🚀 Starting photo-to-print pipeline for `{image_path.name}`\nJob ID: `{job.job_id}`",
        )

        # Stage 1: Image to 3D
        if not await self._stage_1_image_to_3d(job):
            return PipelineResult(success=False, job=job, message="Stage 1 (Image-to-3D) failed")

        # Stage 2: Repair mesh
        if not await self._stage_2_repair_mesh(job):
            return PipelineResult(success=False, job=job, message="Stage 2 (Mesh Repair) failed")

        # Stage 3: Optimize orientation
        if not await self._stage_3_optimize_orientation(job):
            # Non-fatal - continue
            pass

        # Stage 4: Analyze support
        if not await self._stage_4_analyze_support(job):
            # Non-fatal - continue
            pass

        # Stage 5: Render preview
        if not await self._stage_5_render_preview(job):
            return PipelineResult(success=False, job=job, message="Stage 5 (Preview) failed")

        # Stage 6: Wait for approval (unless auto_approve)
        if auto_approve:
            job.approved = True
            job.status = PipelineStatus.APPROVED
            self._update_status(job, "Auto-approved (testing mode)")
        else:
            if not await self._stage_6_wait_for_approval(job):
                return PipelineResult(success=False, job=job, message="Stage 6 (Approval) failed")

            # Pipeline waits for approval - return success for stages 1-5
            return PipelineResult(
                success=True,
                job=job,
                message=f"Pipeline complete through stage 5. Waiting for user approval (Job ID: {job.job_id})",
            )

        # Stage 7: Slice model
        if not await self._stage_7_slice_model(job):
            return PipelineResult(success=False, job=job, message="Stage 7 (Slicing) failed")

        # Stage 8: Start print
        if not await self._stage_8_start_print(job):
            return PipelineResult(success=False, job=job, message="Stage 8 (Print Start) failed")

        return PipelineResult(
            success=True,
            job=job,
            message=f"Pipeline complete! Print started for job {job.job_id}",
        )

    async def approve_job(
        self, job_id: str, scale_factor: float = 1.0, start_print: bool = True
    ) -> PipelineResult:
        """
        Approve a job and proceed to slicing/printing.

        Args:
            job_id: The job ID to approve
            scale_factor: Optional scaling factor for the model
            start_print: If True, proceed to stage 7-8 (slice and print)

        Returns:
            PipelineResult with final status
        """
        job = self._jobs.get(job_id)
        if not job:
            return PipelineResult(
                success=False,
                job=PipelineJob(job_id=job_id, user_id="", chat_id=0, image_path=Path()),
                message=f"Job {job_id} not found",
            )

        if job.status != PipelineStatus.WAITING_APPROVAL:
            return PipelineResult(
                success=False,
                job=job,
                message=f"Job {job_id} is not waiting for approval (status: {job.status.value})",
            )

        job.approved = True
        job.scale_factor = scale_factor
        job.status = PipelineStatus.APPROVED
        self._update_status(job, f"Job approved with scale factor {scale_factor}")
        await self._send_telegram_notification(
            job.chat_id,
            f"✅ Job approved! Proceeding with scale factor {scale_factor}x",
        )

        # Signal the waiting event
        if job_id in self._approval_events:
            self._approval_events[job_id].set()

        if start_print:
            # Stage 7: Slice model
            if not await self._stage_7_slice_model(job):
                return PipelineResult(success=False, job=job, message="Stage 7 (Slicing) failed")

            # Stage 8: Start print
            if not await self._stage_8_start_print(job):
                return PipelineResult(
                    success=False, job=job, message="Stage 8 (Print Start) failed"
                )

        return PipelineResult(
            success=True,
            job=job,
            message=f"Job {job_id} completed successfully!",
        )

    def reject_job(self, job_id: str) -> bool:
        """
        Reject a job.

        Args:
            job_id: The job ID to reject

        Returns:
            True if job was found and rejected
        """
        job = self._jobs.get(job_id)
        if not job:
            return False

        job.rejected = True
        job.status = PipelineStatus.REJECTED
        self._update_status(job, "Job rejected by user")

        # Signal the waiting event
        if job_id in self._approval_events:
            self._approval_events[job_id].set()

        return True

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running or waiting job.

        Args:
            job_id: The job ID to cancel

        Returns:
            True if job was found and cancelled
        """
        job = self._jobs.get(job_id)
        if not job:
            return False

        job.cancelled = True
        job.status = PipelineStatus.CANCELLED
        self._update_status(job, "Job cancelled")

        # Signal the waiting event
        if job_id in self._approval_events:
            self._approval_events[job_id].set()

        return True

    def get_job(self, job_id: str) -> PipelineJob | None:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def list_jobs(self, user_id: int | str | None = None) -> list[PipelineJob]:
        """
        List all jobs, optionally filtered by user.

        Args:
            user_id: Optional user ID to filter by

        Returns:
            List of PipelineJob objects
        """
        jobs = list(self._jobs.values())
        if user_id is not None:
            jobs = [j for j in jobs if j.user_id == user_id]
        return jobs

    def get_pending_approvals(self, user_id: int | str | None = None) -> list[PipelineJob]:
        """
        Get jobs waiting for approval.

        Args:
            user_id: Optional user ID to filter by

        Returns:
            List of PipelineJob objects waiting for approval
        """
        jobs = [j for j in self._jobs.values() if j.status == PipelineStatus.WAITING_APPROVAL]
        if user_id is not None:
            jobs = [j for j in jobs if j.user_id == user_id]
        return jobs

    def get_job_summary(self, job_id: str) -> dict[str, Any] | None:
        """
        Get a summary of a job with key details.

        Args:
            job_id: The job ID

        Returns:
            Dictionary with job summary or None if not found
        """
        job = self._jobs.get(job_id)
        if not job:
            return None

        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "stage": job.stage.name,
            "image_name": job.image_path.name,
            "estimates": job.estimates,
            "errors": job.errors[:5],  # First 5 errors
            "warnings": job.warnings[:5],  # First 5 warnings
            "created_at": job.created_at.isoformat(),
            "updated_at": job.updated_at.isoformat(),
            "printability_score": (
                job.support_result.printability_score if job.support_result else None
            ),
            "support_volume_mm3": (
                job.support_result.total_support_volume if job.support_result else None
            ),
        }


# Convenience function for direct usage
async def process_photo_to_print(
    image_path: Path | str,
    telegram_chat_id: int,
    user_id: int | str,
    image_to_3d_service: ImageTo3DService,
    slicer_pipeline: SlicerPipeline,
    printer_adapter: PrinterAdapter | None = None,
    telegram_bot: Any | None = None,
    output_dir: Path | None = None,
    context: dict[str, Any] | None = None,
    auto_approve: bool = False,
) -> PipelineResult:
    """
    Standalone function to process a photo through the full pipeline.

    Creates a temporary pipeline instance and runs the full workflow.

    Args:
        image_path: Path to the input image
        telegram_chat_id: Telegram chat ID for approval messages
        user_id: User ID for tracking
        image_to_3d_service: Configured ImageTo3DService instance
        slicer_pipeline: Configured SlicerPipeline instance
        printer_adapter: Optional printer adapter for starting print
        telegram_bot: Optional Telegram bot instance for approval messages
        output_dir: Optional output directory for generated files
        context: Optional additional context
        auto_approve: If True, skip approval stage

    Returns:
        PipelineResult with job status
    """
    pipeline = PhotoToPrintPipeline(
        image_to_3d_service=image_to_3d_service,
        slicer_pipeline=slicer_pipeline,
        printer_adapter=printer_adapter,
        telegram_bot=telegram_bot,
        output_dir=output_dir,
    )

    return await pipeline.process_photo_to_print(
        image_path=image_path,
        telegram_chat_id=telegram_chat_id,
        user_id=user_id,
        context=context,
        auto_approve=auto_approve,
    )
