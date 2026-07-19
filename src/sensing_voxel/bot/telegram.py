"""
Telegram bot integration

Adapted from OpenClaw's telegram patterns
"""

import asyncio
import html
import json
import logging
import tempfile
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import (
    BadRequest,
    Forbidden,
    NetworkError,
    RetryAfter,
    TimedOut,
)
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from sensing_voxel.adapters.base import PrinterAdapter
from sensing_voxel.config import Config
from sensing_voxel.core.decision import DecisionEngine
from sensing_voxel.core.models import (
    AgentState,
    CommandResultStatus,
    ConsumeResult,
    DecisionAction,
    FileValidationStatus,
    JobContext,
    generate_decision_token,
    generate_job_id,
)
from sensing_voxel.core.persistence import StatePersistence
from sensing_voxel.core.state_machine import AgentStateMachine
from sensing_voxel.llm.chat import ChatManager
from sensing_voxel.llm.runtime import ChatRequestContext
from sensing_voxel.safety.models import PrintRequest
from sensing_voxel.safety.service import PrintSafetyService
from sensing_voxel.slicer import GcodeMetadataExtractor

# --- Error Classification ---


def is_recoverable_error(error: Exception) -> bool:
    """
    Classify Telegram errors as recoverable or fatal (OpenClaw pattern)

    Recoverable errors can be retried with backoff.
    Fatal errors should not be retried.
    """
    # Check specific error types first (before NetworkError which is a parent class)
    if isinstance(error, BadRequest):
        # Some BadRequest errors are recoverable (e.g., message too long can be split)
        error_message = str(error).lower()
        if "message is too long" in error_message:
            return True
        return False
    elif isinstance(error, Forbidden):
        # Bot blocked or chat not found - do not retry
        return False
    elif isinstance(error, (NetworkError, TimedOut, RetryAfter)):
        # Network issues, timeouts, rate limits - retry
        return True
    return False


# --- Send Functions ---

logger = logging.getLogger(__name__)


class TelegramBot:
    """Telegram bot for Sensing Voxel"""

    def __init__(
        self,
        token: str,
        allowed_users: list[int],
        state_machine: AgentStateMachine,
        adapter: PrinterAdapter | None = None,
        config: Config | None = None,
        persistence: StatePersistence | None = None,
        chat_manager: ChatManager | None = None,
        safety_services: dict[str, PrintSafetyService] | None = None,
        active_printer: str | None = None,
    ):
        self.token = token
        self.allowed_users = set(allowed_users)
        self.state_machine = state_machine
        self.adapter = adapter
        self.config = config
        self.persistence = persistence
        self.chat_manager = chat_manager
        self.safety_services = safety_services or {}
        self.active_printer = active_printer or next(iter(self.safety_services), None)
        self.decision_engine = DecisionEngine()

        self._app = Application.builder().token(token).build()

        # Register handlers
        self._register_handlers()

        self._decision_callbacks: dict[str, asyncio.Future] = {}
        self._chat_id: int | None = None  # Store chat ID for notifications

    def _register_handlers(self):
        """Register command and message handlers"""
        # Command handlers
        self._app.add_handler(CommandHandler("start", self._handle_start))
        self._app.add_handler(CommandHandler("status", self._handle_status))
        self._app.add_handler(CommandHandler("reset_z_offset", self._handle_reset_z_offset))
        self._app.add_handler(CommandHandler("calibrate_camera", self._handle_calibrate_camera))
        self._app.add_handler(CommandHandler("ask", self._handle_ask))
        self._app.add_handler(CommandHandler("clear_chat", self._handle_clear_chat))
        self._app.add_handler(CommandHandler("ctx", self._handle_context_status))
        self._app.add_handler(CommandHandler("audit", self._handle_audit))
        self._app.add_handler(CommandHandler("tasks", self._handle_tasks))
        self._app.add_handler(CommandHandler("approvals", self._handle_approvals))

        # Message handlers
        self._app.add_handler(MessageHandler(filters.Document.ALL, self._handle_file_upload))

        # Callback query handler (inline keyboard buttons)
        self._app.add_handler(CallbackQueryHandler(self._handle_callback_query))

        # Free-text messages → AI chat (registered LAST as catch-all)
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_chat_message)
        )

    async def _check_auth(self, update: Update) -> bool:
        """Check if user is authorized"""
        user_id = update.effective_user.id if update.effective_user else None

        if user_id not in self.allowed_users:
            logger.warning(f"Unauthorized access attempt by user {user_id}")
            if update.message:
                await update.message.reply_text(
                    "❌ Unauthorized. Please contact the administrator."
                )
            return False
        return True

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        if not await self._check_auth(update):
            return

        message = (
            "Welcome to Sensing Voxel!\n\n"
            "AI-powered 3D printer monitoring and control\n\n"
            "Send me an STL file to start a print\n"
            "/status - check current state\n"
            "/reset_z_offset - reset Z-offset\n"
            "/calibrate_camera - calibrate camera\n"
            "/ask <question> - ask AI about printing\n"
            "/clear_chat - reset AI conversation\n\n"
            "Or just send a text message to chat with AI!"
        )

        await update.message.reply_text(message)

    async def _handle_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        if not await self._check_auth(update):
            return

        state = self.state_machine.current_state
        job = self.state_machine.job_context

        message = f"📊 <b>Current State:</b> <code>{html.escape(state.name)}</code>\n\n"

        if job:
            message += f"🖨️ <b>Job ID:</b> <code>{html.escape(job.job_id)}</code>\n"
            message += f"📁 <b>File:</b> <code>{html.escape(job.job_file)}</code>\n"
            if job.decision_consumed:
                message += f"✅ <b>Decision:</b> {html.escape(job.decision_action.name)}\n"

        if self.state_machine.z_offset_dirty:
            message += "\n⚠️ <b>Z-offset is dirty</b> - prints blocked until reset\n"

        await update.message.reply_text(message, parse_mode=ParseMode.HTML)

    GCODE_EXTENSIONS = {".gcode", ".g", ".gco"}
    STL_EXTENSIONS = {".stl"}
    STEP_EXTENSIONS = {".step", ".stp"}
    THREE_MF_EXTENSIONS = {".3mf"}
    MODEL_EXTENSIONS = STL_EXTENSIONS | STEP_EXTENSIONS | THREE_MF_EXTENSIONS
    ALLOWED_EXTENSIONS = MODEL_EXTENSIONS | GCODE_EXTENSIONS

    @staticmethod
    def _update_chat_id(update: Update) -> int | None:
        """Return the chat that originated this update."""
        chat = update.effective_chat
        if chat is not None:
            return chat.id
        query = update.callback_query
        message = query.message if query is not None else None
        message_chat = message.chat if message is not None else None
        return message_chat.id if message_chat is not None else None

    @staticmethod
    def _job_chat_id(job: JobContext) -> int | None:
        """Return the persisted routing destination for a job."""
        chat_id = job.metadata.get("telegram_chat_id")
        return chat_id if isinstance(chat_id, int) and not isinstance(chat_id, bool) else None

    @staticmethod
    def _job_owner_user_id(job: JobContext) -> str | None:
        """Return the persisted Telegram owner for a job."""
        user_id = job.metadata.get("telegram_user_id")
        if isinstance(user_id, (str, int)) and not isinstance(user_id, bool):
            return str(user_id)
        return None

    def _authorize_job_owner(self, job: JobContext, *, user_id: str) -> bool:
        """Authorize control actions against the persisted job owner."""
        owner_id = self._job_owner_user_id(job)
        return owner_id is not None and owner_id == user_id

    def _authorize_staged_print_owner(
        self, job: JobContext, *, user_id: str
    ) -> tuple[bool, PrintRequest | None]:
        """Resolve the current safety request and authorize its binding owner."""
        safety = job.metadata.get("safety")
        if not isinstance(safety, dict):
            return False, None
        printer_id = safety.get("printer_id")
        request_id = safety.get("request_id")
        operation_id = safety.get("operation_id")
        if not isinstance(printer_id, str) or not isinstance(request_id, str):
            return False, None
        service = self.safety_services.get(printer_id)
        if service is None:
            return False, None
        try:
            request = service.get_request(request_id)
        except Exception:
            logger.warning("Unable to resolve safety request for job %s", job.job_id)
            return False, None
        authorized = (
            request.operation_id == operation_id
            and request.binding.user_id == user_id
            and self._authorize_job_owner(job, user_id=user_id)
        )
        return authorized, request if authorized else None

    async def _handle_file_upload(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle file upload (STL or G-code)"""
        if not await self._check_auth(update):
            return

        document = update.message.document
        if not document:
            return

        # Check file type
        ext = Path(document.file_name).suffix.lower() if document.file_name else ""
        if ext not in self.ALLOWED_EXTENSIONS:
            await update.message.reply_text(
                "Supported file types: STL (.stl), STEP (.step, .stp), 3MF (.3mf), "
                "and G-code (.gcode, .g, .gco)"
            )
            return

        is_gcode = ext in self.GCODE_EXTENSIONS
        max_size = 200 * 1024 * 1024 if is_gcode else 100 * 1024 * 1024

        # Check file size
        if document.file_size and document.file_size > max_size:
            limit_mb = max_size // (1024 * 1024)
            await update.message.reply_text(f"File too large (max {limit_mb}MB)")
            return

        if is_gcode:
            await update.message.reply_text("Processing your G-code file...")
            # Store chat ID for future notifications
            self._chat_id = update.effective_chat.id

        # Check if we're already in a job
        if self.state_machine.current_state != AgentState.IDLE:
            await update.message.reply_text(
                f"Agent is busy in state {self.state_machine.current_state.name}. "
                "Please wait or cancel current job first."
            )
            return

        if not is_gcode:
            await update.message.reply_text(
                "Model slicing is currently unavailable. Please upload pre-sliced "
                "G-code (.gcode, .g, or .gco)."
            )
            return

        # Download and stage file
        try:
            file = await document.get_file()
            uploads_dir = self.config.uploads_path if self.config else Path("data/uploads")
            uploads_dir.mkdir(parents=True, exist_ok=True)

            job_id = generate_job_id()
            staged_path = uploads_dir / f"{job_id}{ext}"

            # Security check: validate resolved path is within uploads directory
            resolved = staged_path.resolve()
            try:
                resolved.relative_to(uploads_dir.resolve())
            except ValueError:
                await update.message.reply_text("Invalid file path")
                logger.error(f"Path traversal attempt detected: {staged_path}")
                return

            await file.download_to_drive(str(staged_path))
            await self._handle_gcode_upload(update, document, staged_path, job_id)

        except Exception as e:
            logger.error(f"Error processing file upload: {e}", exc_info=True)
            await update.message.reply_text(f"Error processing file: {str(e)}")

    async def _handle_gcode_upload(self, update, document, gcode_path, job_id):
        """Validate uploaded G-code and present its immutable safety preview for approval."""
        printer_id = self.active_printer
        service = self.safety_services.get(printer_id or "")
        user = update.effective_user
        if service is None or printer_id is None or user is None:
            await update.message.reply_text(
                "Print safety service is unavailable; the G-code cannot be approved."
            )
            gcode_path.unlink(missing_ok=True)
            return

        try:
            binding = await service.create_binding(
                user_id=str(user.id), source_trust="user_gcode"
            )
            request = await service.request_from_path(
                gcode_path,
                operation_id=uuid.uuid4().hex,
                binding=binding,
                printer_file=Path(gcode_path).name,
            )
        except Exception as exc:
            await update.message.reply_text(f"G-code safety validation failed: {exc}")
            gcode_path.unlink(missing_ok=True)
            return

        confirmation = request.confirmation
        metadata = self._parse_gcode_metadata(gcode_path)
        metadata["telegram_user_id"] = user.id
        metadata["telegram_chat_id"] = self._update_chat_id(update)
        metadata["safety"] = {
            "printer_id": printer_id,
            "request_id": request.request_id,
            "operation_id": request.operation_id,
            "artifact_sha256": request.artifact.sha256,
            "preview_sha256": request.preview.sha256,
            "rendered_preview_sha256": request.preview.rendered_sha256,
            "validation_sha256": request.validation.result_sha256,
            "confirmation_sha256": request.confirmation.confirmation_sha256,
            "binding": {
                "target_printer": request.binding.target_printer,
                "hardware_digest": request.binding.hardware_digest,
                "profile_digest": request.binding.profile_digest,
                "config_digest": request.binding.config_digest,
                "macro_digest": request.binding.macro_digest,
                "source_trust": request.binding.source_trust,
                "user_id": request.binding.user_id,
                "probe_z_offset": request.binding.probe_z_offset,
                "homing_origin_z": request.binding.homing_origin_z,
            },
            "confirmation": {
                "material": request.confirmation.material,
                "build_plate_surface": request.confirmation.build_plate_surface,
                "hotend_nozzle_size_mm": request.confirmation.hotend_nozzle_size_mm,
                "estimated_print_time_seconds": request.confirmation.estimated_print_time_seconds,
                "filament_usage_mm": request.confirmation.filament_usage_mm,
                "z_offset_mm": request.confirmation.z_offset_mm,
                "artifact_sha256": request.confirmation.artifact_sha256,
                "preview_sha256": request.confirmation.preview_sha256,
                "rendered_preview_sha256": request.confirmation.rendered_preview_sha256,
                "confirmation_sha256": request.confirmation.confirmation_sha256,
            },
            "preview": json.loads(request.preview.content),
            "rendered_preview": request.preview.rendered_content.decode("utf-8"),
            "rendered_preview_media_type": request.preview.rendered_media_type,

        }

        job_context = JobContext(
            job_id=job_id,
            job_file=str(gcode_path),
            uploaded_at=datetime.now(),
            decision_token=generate_decision_token(),
            gcode_path=str(gcode_path),
            metadata=metadata,
            file_validation_status=FileValidationStatus.VALID,
            is_external=False,
            external_monitoring=False,
        )

        self.state_machine.job_context = job_context
        success = await self.state_machine.transition_to(
            AgentState.AWAITING_APPROVAL, persistence=self.persistence
        )

        if success:
            meta_lines = []
            if metadata.get("total_layers"):
                meta_lines.append(f"Layers: {metadata['total_layers']}")
            if metadata.get("estimated_time"):
                minutes = metadata["estimated_time"] // 60
                meta_lines.append(f"Est. time: {minutes}min")
            if metadata.get("filament_type"):
                meta_lines.append(f"Filament: {metadata['filament_type']}")
            if metadata.get("layer_height"):
                meta_lines.append(f"Layer height: {metadata['layer_height']}mm")

            meta_text = "\n".join(meta_lines) if meta_lines else "No metadata found"

            keyboard = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton("Approve", callback_data=f"apr_{job_id}"),
                        InlineKeyboardButton("Reject", callback_data=f"rej_{job_id}"),
                    ]
                ]
            )

            preview_file = BytesIO(request.preview.rendered_content)
            preview_file.name = f"{Path(document.file_name).stem}-preview.svg"
            await update.message.reply_document(
                document=preview_file,
                caption=(
                    "Deterministic toolpath preview for the exact artifact below.\n"
                    f"SVG SHA-256: `{request.preview.rendered_sha256}`"
                ),
            )
            await update.message.reply_text(
                f"G-code file received: {document.file_name}\n"
                f"Job ID: `{job_id}`\n\n"
                f"{meta_text}\n\n"
                f"Material: `{confirmation.material}`\n"
                f"Build plate: `{confirmation.build_plate_surface}`\n"
                f"Nozzle: `{confirmation.hotend_nozzle_size_mm:g} mm`\n"
                f"Estimated time: `{confirmation.estimated_print_time_seconds} s`\n"
                f"Filament: `{confirmation.filament_usage_mm:g} mm`\n"
                f"Live Z offset: `{confirmation.z_offset_mm:g} mm`\n\n"
                f"Artifact: `{request.artifact.sha256}`\n"
                f"JSON preview: `{request.preview.sha256}`\n"
                f"SVG preview: `{request.preview.rendered_sha256}`\n"
                f"Validation: `{request.validation.result_sha256}`\n"
                f"Confirmation: `{confirmation.confirmation_sha256}`\n"
                f"Printer: `{request.binding.target_printer}`\n"
                f"Profile: `{request.binding.profile_digest}`\n"
                f"Config: `{request.binding.config_digest}`\n"
                f"Macros: `{request.binding.macro_digest}`\n"
                f"Hardware: `{request.binding.hardware_digest}`\n"
                f"Probe Z: `{request.binding.probe_z_offset:g} mm`\n"
                f"Homing Z: `{request.binding.homing_origin_z:g} mm`\n\n"
                "Approve this exact validated artifact, both previews, confirmation, and "
                "binding to upload and start print?",
                reply_markup=keyboard,
            )

            logger.info(f"gcode_upload_staged job_id={job_id} file={document.file_name}")
        else:
            await update.message.reply_text("Failed to start job. Please try again.")
            gcode_path.unlink(missing_ok=True)

    @staticmethod
    def _parse_gcode_metadata(gcode_path: Path) -> dict:
        """Extract display metadata through the shared exact-byte parser."""
        return GcodeMetadataExtractor.extract_metadata(gcode_path)

    async def _handle_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline keyboard callback"""
        if not await self._check_auth(update):
            return

        query = update.callback_query
        if not query:
            return

        await query.answer()

        # Parse callback data (OpenClaw pattern: prefix_jobid)
        data = query.data
        if not data:
            return

        parts = data.split("_", 1)
        if len(parts) != 2:
            return

        prefix, job_id = parts

        # Find matching job - MUST match full job ID
        job = self.state_machine.job_context
        if not job or job.job_id != job_id:
            await query.edit_message_text("❌ Job not found or expired")
            return

        action_map = {
            "apr": DecisionAction.APPROVE,
            "rej": DecisionAction.REJECT,
            "res": DecisionAction.RESUME,
            "can": DecisionAction.CANCEL,
        }
        expected_state_map = {
            DecisionAction.APPROVE: AgentState.AWAITING_APPROVAL,
            DecisionAction.REJECT: AgentState.AWAITING_APPROVAL,
            DecisionAction.RESUME: AgentState.PAUSED_PENDING_USER,
            DecisionAction.CANCEL: AgentState.PAUSED_PENDING_USER,
        }
        action = action_map.get(prefix)
        if action is None:
            return

        user = update.effective_user
        if user is None:
            return
        user_id = str(user.id)
        callback_chat_id = self._update_chat_id(update) or self._job_chat_id(job)
        print_request: PrintRequest | None = None
        if action in {DecisionAction.APPROVE, DecisionAction.REJECT}:
            authorized, print_request = self._authorize_staged_print_owner(
                job, user_id=user_id
            )
        else:
            authorized = self._authorize_job_owner(job, user_id=user_id)

        if not authorized:
            logger.warning(
                "Rejected %s for job %s from non-owner user %s",
                action.name,
                job.job_id,
                user_id,
            )
            if callback_chat_id is not None:
                await self._app.bot.send_message(
                    chat_id=callback_chat_id,
                    text="This job belongs to another user; the decision is still pending.",
                )
            return

        decision_token = job.decision_token
        if not decision_token:
            await query.edit_message_text("❌ Decision token is missing or expired")
            return

        result = await self.decision_engine.consume_decision(
            job_context=job,
            decision_token=decision_token,
            action=action,
            expected_state=expected_state_map[action],
            current_state=self.state_machine.current_state,
        )

        # Execute side effects based on action
        if result == ConsumeResult.CONSUMED:
            await query.edit_message_text(
                f"✅ Decision recorded: {action.name}\nJob ID: `{job.job_id[:12]}`"
            )

            # Handle action-specific transitions
            if action == DecisionAction.APPROVE:
                if callback_chat_id is not None:
                    await self._app.bot.send_message(
                        chat_id=callback_chat_id, text="Starting print..."
                    )
                await self._execute_print_approval(
                    job, user_id=user_id, chat_id=callback_chat_id
                )
            elif action == DecisionAction.REJECT:
                assert print_request is not None
                safety = job.metadata["safety"]
                service = self.safety_services[str(safety["printer_id"])]
                service.reject(
                    print_request.request_id, binding=print_request.binding
                )
                # Cleanup staged file
                if job.gcode_path:
                    Path(job.gcode_path).unlink(missing_ok=True)
                await self.state_machine.transition_to(
                    AgentState.IDLE, persistence=self.persistence
                )
                if callback_chat_id is not None:
                    await self._app.bot.send_message(
                        chat_id=callback_chat_id, text="Job rejected. Back to idle."
                    )
                logger.info(f"gcode_upload_rejected job_id={job.job_id}")
            elif action == DecisionAction.RESUME:
                # Resume print
                if self.adapter:
                    await self.adapter.resume_print()
                await self.state_machine.transition_to(
                    AgentState.MONITORING, persistence=self.persistence
                )
                if callback_chat_id is not None:
                    await self._app.bot.send_message(
                        chat_id=callback_chat_id, text="▶️ Print resumed."
                    )
            elif action == DecisionAction.CANCEL:
                if self.adapter:
                    await self.adapter.cancel_print()
                if await self.state_machine.transition_to(
                    AgentState.FAILED, persistence=self.persistence
                ):
                    await self.state_machine.transition_to(
                        AgentState.IDLE, persistence=self.persistence
                    )
                if callback_chat_id is not None:
                    await self._app.bot.send_message(
                        chat_id=callback_chat_id, text="⏹️ Print cancelled."
                    )

        elif result == ConsumeResult.ALREADY_HANDLED:
            await query.edit_message_text("ℹ️ Decision already processed for this job.")
        else:
            await query.edit_message_text(f"⚠️ Decision rejected: {result.name}")

    async def _execute_print_approval(
        self, job: JobContext, *, user_id: str, chat_id: int | None
    ):
        """Upload approved immutable bytes and execute through the safety service."""
        safety = job.metadata.get("safety", {})
        printer_id = safety.get("printer_id")
        request_id = safety.get("request_id")
        operation_id = safety.get("operation_id")
        service = self.safety_services.get(printer_id or "")
        adapter = self.adapter
        if service is None or adapter is None or not request_id or not operation_id:
            if chat_id is not None:
                await self._app.bot.send_message(
                    chat_id=chat_id,
                    text="Print safety approval is unavailable or incomplete.",
                )
            return

        try:
            request = service.get_request(request_id)
            if request.operation_id != operation_id:
                raise ValueError("stored operation does not match the approved request")
            if request.binding.user_id != user_id:
                raise ValueError("approving user does not match the staged request")
            if request.binding.source_trust != "user_gcode":
                raise ValueError("unexpected print source trust")

            await self.state_machine.transition_to(
                AgentState.UPLOADING, persistence=self.persistence
            )

            with tempfile.NamedTemporaryFile(suffix=".gcode") as staged:
                staged.write(request.artifact.content)
                staged.flush()
                upload_result = await adapter.upload_file(
                    staged.name, request.printer_file
                )

            if upload_result.status != CommandResultStatus.SUCCESS:
                if chat_id is not None:
                    await self._app.bot.send_message(
                        chat_id=chat_id,
                        text=f"Upload failed: {upload_result.message}",
                    )
                await self.state_machine.transition_to(
                    AgentState.ERROR, persistence=self.persistence
                )
                return

            approval = service.approve(request.request_id, binding=request.binding)
            receipt = await service.execute(
                approval.token,
                operation_id=request.operation_id,
                binding=request.binding,
            )
            if not receipt.started:
                if chat_id is not None:
                    await self._app.bot.send_message(
                        chat_id=chat_id,
                        text=f"Failed to start print: {receipt.message}",
                    )
                await self.state_machine.transition_to(
                    AgentState.ERROR, persistence=self.persistence
                )
                return

            await self.state_machine.transition_to(
                AgentState.PRINTING, persistence=self.persistence
            )
            if chat_id is not None:
                await self._app.bot.send_message(
                    chat_id=chat_id, text="Print started successfully!"
                )
            logger.info("gcode_upload_approved job_id=%s", job.job_id)

        except Exception as exc:
            logger.error("Error executing print approval: %s", exc, exc_info=True)
            if chat_id is not None:
                await self._app.bot.send_message(
                    chat_id=chat_id, text=f"Error starting print: {exc}"
                )
            await self.state_machine.transition_to(
                AgentState.ERROR, persistence=self.persistence
            )

    async def _handle_reset_z_offset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /reset_z_offset command"""
        if not await self._check_auth(update):
            return

        # Parse optional target Z offset from arguments
        args = context.args or []
        target_mm = None

        if args:
            try:
                target_mm = float(args[0])
            except ValueError:
                await update.message.reply_text(
                    "❌ Invalid Z offset value. Use: `/reset_z_offset [target_mm]`"
                )
                return

        # Check current state - cannot reset during active print
        if self.state_machine.current_state not in (AgentState.IDLE, AgentState.ERROR):
            await update.message.reply_text(
                "⚠️ Cannot reset Z-offset while printer is active. "
                "Please wait for current job to complete or cancel it first."
            )
            return

        # If no target specified and baseline exists, use baseline
        if target_mm is None and self.state_machine.last_known_z_offset_baseline is not None:
            target_mm = self.state_machine.last_known_z_offset_baseline
            reset_type = "baseline"
        elif target_mm is None:
            target_mm = 0.0
            reset_type = "zero"
        else:
            reset_type = f"{target_mm:.3f}mm"

        # Apply Z offset reset via adapter
        if self.adapter:
            try:
                result = await self.adapter.set_z_offset(target_mm)
                if result.status == CommandResultStatus.SUCCESS:
                    self.state_machine.z_offset_dirty = False
                    self.state_machine.last_known_z_offset_baseline = target_mm

                    # Persist state
                    # Persist state if persistence is available
                    if self.persistence:
                        self.persistence.save(self.state_machine)

                    await update.message.reply_text(
                        f"✅ Z-offset reset to {reset_type} ({target_mm:.3f}mm)\n"
                        f"State: {'CLEAN' if not self.state_machine.z_offset_dirty else 'DIRTY'}"
                    )
                else:
                    await update.message.reply_text(
                        f"❌ Failed to reset Z-offset: {result.message}"
                    )
            except Exception as e:
                await update.message.reply_text(f"❌ Error resetting Z-offset: {str(e)}")
        else:
            # No adapter configured - just update state
            self.state_machine.z_offset_dirty = False
            self.state_machine.last_known_z_offset_baseline = target_mm

            await update.message.reply_text(
                f"⚠️ No printer adapter configured. "
                f"Z-offset state reset to {reset_type} ({target_mm:.3f}mm) in memory only."
            )

    async def _handle_calibrate_camera(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /calibrate_camera command"""
        if not await self._check_auth(update):
            return

        # Check if camera is configured
        if not self.config or not self.config.camera:
            await update.message.reply_text(
                "❌ Camera not configured. Please configure camera in config file."
            )
            return

        await update.message.reply_text(
            "📹 **Camera Calibration**\n\n"
            "1. Place a checkerboard pattern on the print bed\n"
            "2. Ensure good lighting\n"
            "3. Reply with 'START' to begin calibration\n\n"
            "⚠️ Note: This will override existing calibration data"
        )

        # Store state for follow-up message
        context.user_data["awaiting_calibration"] = True

    async def send(
        self,
        chat_id: int,
        content: str,
        content_type: str = "text",
        caption: str | None = None,
        parse_mode: ParseMode = ParseMode.HTML,
        reply_markup: InlineKeyboardMarkup | None = None,
        max_retries: int = 3,
    ):
        """
        Unified send function for text, photos, and documents (OpenClaw pattern)

        Args:
            chat_id: Target chat ID
            content: Content to send (text, photo path, or document path)
            content_type: "text", "photo", or "document"
            caption: Caption for photo/document
            parse_mode: Parse mode for text
            reply_markup: Inline keyboard markup
            max_retries: Maximum retry attempts for recoverable errors

        Returns:
            Result from Telegram API
        """
        for attempt in range(max_retries):
            try:
                if content_type == "text":
                    return await self._app.bot.send_message(
                        chat_id=chat_id,
                        text=content,
                        parse_mode=parse_mode,
                        reply_markup=reply_markup,
                    )
                elif content_type == "photo":
                    return await self._app.bot.send_photo(
                        chat_id=chat_id,
                        photo=content,
                        caption=caption,
                        parse_mode=parse_mode,
                        reply_markup=reply_markup,
                    )
                elif content_type == "document":
                    return await self._app.bot.send_document(
                        chat_id=chat_id,
                        document=content,
                        caption=caption,
                        parse_mode=parse_mode,
                        reply_markup=reply_markup,
                    )
                else:
                    raise ValueError(f"Unknown content_type: {content_type}")
            except Exception as e:
                if attempt < max_retries - 1 and is_recoverable_error(e):
                    # Wait before retry with exponential backoff
                    delay = 2**attempt
                    logger.warning(
                        f"Recoverable error on attempt {attempt + 1}/{max_retries}, "
                        f"retrying in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    # Fatal error or last attempt - re-raise
                    if is_recoverable_error(e):
                        logger.error(f"All {max_retries} retry attempts failed: {e}")
                    raise

    async def send_message(
        self,
        chat_id: int,
        text: str,
        parse_mode: ParseMode = ParseMode.HTML,
        reply_markup: InlineKeyboardMarkup | None = None,
    ):
        """Send message with HTML fallback (OpenClaw pattern)"""
        try:
            await self._app.bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
            )
        except Exception as e:
            # Fallback to plain text if HTML parse fails
            if parse_mode == ParseMode.HTML:
                logger.warning(f"HTML parse failed, retrying as plain text: {e}")
                await self._app.bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    parse_mode=None,
                    reply_markup=reply_markup,
                )
            else:
                raise

    async def send_photo(
        self,
        chat_id: int,
        photo_path: str,
        caption: str | None = None,
        parse_mode: ParseMode = ParseMode.HTML,
        reply_markup: InlineKeyboardMarkup | None = None,
    ):
        """Send photo with error handling and fallback"""
        try:
            await self._app.bot.send_photo(
                chat_id=chat_id,
                photo=photo_path,
                caption=caption,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
            )
        except Exception as e:
            logger.error(f"Failed to send photo: {e}")
            # Fallback to sending caption as text
            if caption:
                await self.send_message(
                    chat_id=chat_id,
                    text=caption,
                    parse_mode=parse_mode,
                    reply_markup=reply_markup,
                )

    def _notification_chat_id(self, *, job_id: str | None = None) -> int | None:
        if job_id is None:
            return self._chat_id
        job = self.state_machine.job_context
        if job is None or job.job_id != job_id:
            return None
        return self._job_chat_id(job)

    async def notify_print_complete(self, job_context: JobContext):
        """Notify user that print is complete"""
        chat_id = self._job_chat_id(job_context)
        if chat_id is None:
            return

        message = (
            "✅ <b>Print Complete!</b>\n\n"
            f"Job ID: <code>{job_context.job_id}</code>\n"
            f"File: <code>{job_context.job_file}</code>\n"
            f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        try:
            await self.send(
                chat_id=chat_id,
                content=message,
                content_type="text",
                parse_mode=ParseMode.HTML,
            )
        except Exception as e:
            logger.error(f"Failed to send print complete notification: {e}")

    async def notify_error(self, error_message: str, job_id: str | None = None):
        """Notify user of error"""
        chat_id = self._notification_chat_id(job_id=job_id)
        if chat_id is None:
            return

        message = "❌ <b>Error</b>\n\n" + error_message
        if job_id:
            message += f"\n\nJob ID: <code>{job_id}</code>"

        try:
            await self.send(
                chat_id=chat_id,
                content=message,
                content_type="text",
                parse_mode=ParseMode.HTML,
            )
        except Exception as e:
            logger.error(f"Failed to send error notification: {e}")

    async def notify_warning(self, warning_message: str, job_id: str | None = None):
        """Notify user of warning"""
        chat_id = self._notification_chat_id(job_id=job_id)
        if chat_id is None:
            return

        message = "⚠️ <b>Warning</b>\n\n" + warning_message
        if job_id:
            message += f"\n\nJob ID: <code>{job_id}</code>"

        try:
            await self.send(
                chat_id=chat_id,
                content=message,
                content_type="text",
                parse_mode=ParseMode.HTML,
            )
        except Exception as e:
            logger.error(f"Failed to send warning notification: {e}")

    async def _handle_chat_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle free-text messages as AI chat"""
        if not await self._check_auth(update):
            return

        await self._send_chat_response(update, update.message.text)

    async def _send_chat_response(self, update: Update, message: str):
        """Send a message to the LLM and reply with the response."""
        if not self.chat_manager:
            await update.message.reply_text(
                "AI chat is not configured. Set up LLM in config/local.yaml."
            )
            return

        user_id = update.effective_user.id
        request_context = ChatRequestContext(
            logical_user_id=str(user_id),
            channel="telegram",
            channel_user_id=str(user_id),
            chat_id=update.effective_chat.id if update.effective_chat else None,
        )

        await update.effective_chat.send_action("typing")

        async def send_tool_progress(progress_message: str) -> None:
            await update.message.reply_text(progress_message)

        try:
            response = await self.chat_manager.chat(
                user_id,
                message,
                request_context=request_context,
                tool_progress_callback=send_tool_progress,
            )
            await update.message.reply_text(response)
        except Exception as e:
            logger.error(f"LLM chat error: {e}", exc_info=True)
            error_detail = str(e)[:200] if str(e) else type(e).__name__
            error_msg = (
                f"⚠️ I ran into an error: {type(e).__name__}: {error_detail}\n\n"
                "This may be a temporary issue (network timeout, camera unreachable, "
                "or LLM service down). You can try again, or ask me to check "
                "printer status instead."
            )
            session = self.chat_manager._get_or_create_session(user_id)
            if session:
                session.add_assistant_message(error_msg)
            self.chat_manager.save_history()
            await update.message.reply_text(error_msg)

    async def _handle_ask(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /ask command for explicit AI chat"""
        if not await self._check_auth(update):
            return

        if not context.args:
            await update.message.reply_text("Usage: /ask <your question>")
            return

        question = " ".join(context.args)
        await self._send_chat_response(update, question)

    async def _handle_clear_chat(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /clear_chat to reset conversation history"""
        if not await self._check_auth(update):
            return

        if self.chat_manager:
            self.chat_manager.clear_session(update.effective_user.id)

        await update.message.reply_text("Chat history cleared.")

    async def _handle_context_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show context/session diagnostics."""
        if not await self._check_auth(update):
            return
        if not self.chat_manager:
            await update.message.reply_text("AI chat is not configured.")
            return

        user_id = update.effective_user.id
        request_context = ChatRequestContext(
            logical_user_id=str(user_id),
            channel="telegram",
            channel_user_id=str(user_id),
            chat_id=update.effective_chat.id if update.effective_chat else None,
        )
        status = self.chat_manager.get_context_status(str(user_id), request_context)
        summary = status.get("summary") or "(none)"
        message = (
            f"Model: <code>{html.escape(str(status.get('model')))}</code>\n"
            f"Estimated tokens: <code>{status.get('estimated_tokens')}</code>\n"
            f"Context window: <code>{status.get('context_window_tokens')}</code>\n"
            f"Messages: <code>{status.get('message_count')}</code>\n"
            f"Last compacted: <code>{html.escape(str(status.get('last_compacted_at')))}</code>\n"
            f"Active printer: <code>{html.escape(str(status.get('active_printer')))}</code>\n\n"
            f"Summary:\n<code>{html.escape(summary[:1500])}</code>"
        )
        await update.message.reply_text(message, parse_mode=ParseMode.HTML)

    async def _handle_audit(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show recent tool audit records."""
        if not await self._check_auth(update):
            return
        if not self.chat_manager:
            await update.message.reply_text("AI chat is not configured.")
            return

        records = self.chat_manager.get_audit_records(limit=5)
        if not records:
            await update.message.reply_text("No tool audit records yet.")
            return

        lines = []
        for record in records:
            lines.append(
                f"{record['started_at']} | {record['tool_name']} | {record['status']} | {record['channel']}"
            )
        await update.message.reply_text("\n".join(lines))

    async def _handle_tasks(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show recent background tasks."""
        if not await self._check_auth(update):
            return
        if not self.chat_manager:
            await update.message.reply_text("AI chat is not configured.")
            return

        user_id = update.effective_user.id
        tasks = self.chat_manager.get_background_tasks(str(user_id), limit=5)
        if not tasks:
            await update.message.reply_text("No background tasks yet.")
            return

        lines = [
            f"{task['task_id']} | {task['task_name']} | {task['status']}"
            for task in tasks
        ]
        await update.message.reply_text("\n".join(lines))

    async def _handle_approvals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show pending approvals."""
        if not await self._check_auth(update):
            return
        if not self.chat_manager:
            await update.message.reply_text("AI chat is not configured.")
            return

        user_id = update.effective_user.id
        approvals = self.chat_manager.get_pending_approvals(str(user_id), limit=5)
        if not approvals:
            await update.message.reply_text("No pending approvals.")
            return

        lines = [
            f"{record['approval_token']} | {record['tool_name']} | {record['reason']}"
            for record in approvals
        ]
        await update.message.reply_text("\n".join(lines))

    def create_decision_keyboard(self, job_id: str, state: AgentState) -> InlineKeyboardMarkup:
        """Create inline keyboard for decisions (OpenClaw pattern)"""
        buttons = []

        if state == AgentState.AWAITING_APPROVAL:
            buttons = [
                [
                    InlineKeyboardButton("✅ Approve", callback_data=f"apr_{job_id}"),
                    InlineKeyboardButton("❌ Reject", callback_data=f"rej_{job_id}"),
                ]
            ]
        elif state == AgentState.PAUSED_PENDING_USER:
            buttons = [
                [
                    InlineKeyboardButton("▶️ Resume", callback_data=f"res_{job_id}"),
                    InlineKeyboardButton("⏹️ Cancel", callback_data=f"can_{job_id}"),
                ]
            ]

        return InlineKeyboardMarkup(buttons)

    async def start(self):
        """Start the bot"""
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()

        logger.info("Telegram bot started")

    async def stop(self):
        """Stop the bot"""
        await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()

        logger.info("Telegram bot stopped")
