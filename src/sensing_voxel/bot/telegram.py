"""
Telegram bot integration

Adapted from OpenClaw's telegram patterns
"""

import asyncio
import html
import logging
from datetime import datetime
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
from sensing_voxel.llm.chat import ChatManager
from sensing_voxel.core.decision import DecisionEngine
from sensing_voxel.core.models import (
    AgentState,
    CommandResultStatus,
    ConsumeResult,
    DecisionAction,
    FileValidationStatus,
    JobContext,
    ModelFileType,
    generate_decision_token,
    generate_job_id,
)
from sensing_voxel.core.persistence import StatePersistence
from sensing_voxel.core.state_machine import AgentStateMachine
from sensing_voxel.slicer import FileFormatValidator, Step3MFConverter

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
    ):
        self.token = token
        self.allowed_users = set(allowed_users)
        self.state_machine = state_machine
        self.adapter = adapter
        self.config = config
        self.persistence = persistence
        self.chat_manager = chat_manager
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
        is_step_3mf = ext in self.STEP_EXTENSIONS or ext in self.THREE_MF_EXTENSIONS
        max_size = 200 * 1024 * 1024 if is_gcode else 100 * 1024 * 1024

        # Check file size
        if document.file_size and document.file_size > max_size:
            limit_mb = max_size // (1024 * 1024)
            await update.message.reply_text(f"File too large (max {limit_mb}MB)")
            return

        file_type = "G-code" if is_gcode else ("STEP/3MF" if is_step_3mf else "STL")
        await update.message.reply_text(f"Processing your {file_type} file...")

        # Store chat ID for future notifications
        self._chat_id = update.effective_chat.id

        # Check if we're already in a job
        if self.state_machine.current_state != AgentState.IDLE:
            await update.message.reply_text(
                f"Agent is busy in state {self.state_machine.current_state.name}. "
                "Please wait or cancel current job first."
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

            # Validate STEP/3MF format before proceeding
            if is_step_3mf:
                validator = FileFormatValidator()
                detected = validator.detect_format(staged_path)
                if not detected:
                    await update.message.reply_text(
                        "Invalid file format. Could not validate as STEP or 3MF."
                    )
                    staged_path.unlink(missing_ok=True)
                    return

                await self._handle_model_upload(
                    update, document, staged_path, job_id, is_converted=True
                )
            elif is_gcode:
                await self._handle_gcode_upload(update, document, staged_path, job_id)
            else:
                await self._handle_model_upload(update, document, staged_path, job_id)

        except Exception as e:
            logger.error(f"Error processing file upload: {e}", exc_info=True)
            await update.message.reply_text(f"Error processing file: {str(e)}")

    async def _handle_model_upload(self, update, document, model_path, job_id, is_converted=False):
        """Handle model file upload (STL, STEP, 3MF) — enters slicing pipeline"""
        job_context = JobContext(
            job_id=job_id,
            job_file=str(model_path),
            uploaded_at=datetime.now(),
            decision_token=generate_decision_token(),
            file_validation_status=FileValidationStatus.VALID,
            is_external=False,
            external_monitoring=False,
            metadata={"is_converted": is_converted},
        )

        self.state_machine.job_context = job_context
        success = await self.state_machine.transition_to(
            AgentState.RECEIVING, persistence=self.persistence
        )

        if success:
            extra_msg = " (will be converted to STL)" if is_converted else ""
            await update.message.reply_text(
                f"File received: {document.file_name}{extra_msg}\n"
                f"Job ID: `{job_id}`\n\n"
                "Starting slicing pipeline..."
            )
        else:
            await update.message.reply_text("Failed to start job. Please try again.")

    async def _handle_stl_upload(self, update, document, stl_path, job_id):
        """Handle STL file upload — enters slicing pipeline"""
        await self._handle_model_upload(update, document, stl_path, job_id)

    async def _handle_gcode_upload(self, update, document, gcode_path, job_id):
        """Handle G-code file upload — bypasses slicing, goes to approval"""
        metadata = self._parse_gcode_metadata(gcode_path)

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

            await update.message.reply_text(
                f"G-code file received: {document.file_name}\n"
                f"Job ID: `{job_id}`\n\n"
                f"{meta_text}\n\n"
                "Approve to upload and start print?",
                reply_markup=keyboard,
            )

            logger.info(f"gcode_upload_staged job_id={job_id} file={document.file_name}")
        else:
            await update.message.reply_text("Failed to start job. Please try again.")
            gcode_path.unlink(missing_ok=True)

    @staticmethod
    def _parse_gcode_metadata(gcode_path: Path) -> dict:
        """Parse metadata from G-code file header (first 10KB)"""
        import re

        try:
            with open(gcode_path, "r", errors="replace") as f:
                header = f.read(10240)
        except OSError:
            return {}

        metadata = {}
        patterns = {
            "total_layers": r";\s*total layer(?:s| number|_count)\s*[:=]\s*(\d+)",
            "estimated_time": r";\s*(?:estimated[_ ]?(?:printing[_ ]?)?time|TIME)\s*[:=]\s*(\d+)",
            "filament_type": r";\s*filament_type\s*[:=]\s*(\S+)",
            "layer_height": r";\s*layer_height\s*[:=]\s*([\d.]+)",
            "filament_used_mm": r";\s*filament used\s*\[mm\]\s*[:=]\s*([\d.]+)",
        }

        for key, regex in patterns.items():
            m = re.search(regex, header, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                if key in ("total_layers", "estimated_time"):
                    metadata[key] = int(val)
                elif key in ("layer_height", "filament_used_mm"):
                    metadata[key] = float(val)
                else:
                    metadata[key] = val

        return metadata

    async def _handle_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline keyboard callback"""
        if not await self._check_auth(update):
            return

        # Ensure chat_id is set from callback context (handles bot restart before first upload)
        if self._chat_id is None:
            self._chat_id = update.effective_chat.id

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

        action: DecisionAction | None = None

        if prefix == "apr":
            action = DecisionAction.APPROVE
        elif prefix == "rej":
            action = DecisionAction.REJECT
        elif prefix == "res":
            action = DecisionAction.RESUME
        elif prefix == "can":
            action = DecisionAction.CANCEL

        if not action:
            return

        # Consume decision via decision engine
        result = await self.decision_engine.consume_decision(
            job_context=job,
            decision_token=job.decision_token,
            action=action,
            expected_state=self.state_machine.current_state,
            current_state=self.state_machine.current_state,
        )

        # Execute side effects based on action
        if result == ConsumeResult.CONSUMED:
            await query.edit_message_text(
                f"✅ Decision recorded: {action.name}\nJob ID: `{job.job_id[:12]}`"
            )

            # Handle action-specific transitions
            if action == DecisionAction.APPROVE:
                await self._app.bot.send_message(chat_id=self._chat_id, text="Starting print...")
                await self._execute_print_approval(job)
            elif action == DecisionAction.REJECT:
                # Cleanup staged file
                if job.gcode_path:
                    Path(job.gcode_path).unlink(missing_ok=True)
                await self.state_machine.transition_to(
                    AgentState.IDLE, persistence=self.persistence
                )
                await self._app.bot.send_message(
                    chat_id=self._chat_id, text="Job rejected. Back to idle."
                )
                logger.info(f"gcode_upload_rejected job_id={job.job_id}")
            elif action == DecisionAction.RESUME:
                # Resume print
                if self.adapter:
                    await self.adapter.resume_print()
                await self.state_machine.transition_to(
                    AgentState.MONITORING, persistence=self.persistence
                )
                await self._app.bot.send_message(chat_id=self._chat_id, text="▶️ Print resumed.")
            elif action == DecisionAction.CANCEL:
                # Cancel print
                if self.adapter:
                    await self.adapter.cancel_print()
                await self.state_machine.transition_to(
                    AgentState.IDLE, persistence=self.persistence
                )
                await self._app.bot.send_message(chat_id=self._chat_id, text="⏹️ Print cancelled.")

        elif result == ConsumeResult.ALREADY_HANDLED:
            await query.edit_message_text("ℹ️ Decision already processed for this job.")
        else:
            await query.edit_message_text(f"⚠️ Decision rejected: {result.name}")

    async def _execute_print_approval(self, job: JobContext):
        """Upload G-code to printer and start print after approval"""
        if not self.adapter:
            await self._app.bot.send_message(chat_id=self._chat_id, text="No printer connected.")
            return

        gcode_path = job.gcode_path or job.job_file
        if not gcode_path:
            await self._app.bot.send_message(
                chat_id=self._chat_id, text="No G-code file found for this job."
            )
            return

        try:
            # Transition to UPLOADING
            await self.state_machine.transition_to(
                AgentState.UPLOADING, persistence=self.persistence
            )

            # Upload file to printer
            remote_name = Path(gcode_path).name
            upload_result = await self.adapter.upload_file(gcode_path, remote_name)

            if upload_result.status != CommandResultStatus.SUCCESS:
                await self._app.bot.send_message(
                    chat_id=self._chat_id,
                    text=f"Upload failed: {upload_result.message}",
                )
                await self.state_machine.transition_to(
                    AgentState.ERROR, persistence=self.persistence
                )
                return

            # Start print
            start_result = await self.adapter.start_print(remote_name)

            if start_result.status != CommandResultStatus.SUCCESS:
                await self._app.bot.send_message(
                    chat_id=self._chat_id,
                    text=f"Failed to start print: {start_result.message}",
                )
                await self.state_machine.transition_to(
                    AgentState.ERROR, persistence=self.persistence
                )
                return

            # Transition to PRINTING
            await self.state_machine.transition_to(
                AgentState.PRINTING, persistence=self.persistence
            )
            await self._app.bot.send_message(
                chat_id=self._chat_id, text="Print started successfully!"
            )
            logger.info(f"gcode_upload_approved job_id={job.job_id}")

        except Exception as e:
            logger.error(f"Error executing print approval: {e}", exc_info=True)
            await self._app.bot.send_message(
                chat_id=self._chat_id, text=f"Error starting print: {e}"
            )
            await self.state_machine.transition_to(AgentState.ERROR, persistence=self.persistence)

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

    async def notify_print_complete(self, job_context: JobContext):
        """Notify user that print is complete"""
        if not self._chat_id:
            return

        message = (
            "✅ <b>Print Complete!</b>\n\n"
            f"Job ID: <code>{job_context.job_id}</code>\n"
            f"File: <code>{job_context.job_file}</code>\n"
            f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        try:
            await self.send(
                chat_id=self._chat_id,
                content=message,
                content_type="text",
                parse_mode=ParseMode.HTML,
            )
        except Exception as e:
            logger.error(f"Failed to send print complete notification: {e}")

    async def notify_error(self, error_message: str, job_id: str | None = None):
        """Notify user of error"""
        if not self._chat_id:
            return

        message = "❌ <b>Error</b>\n\n" + error_message
        if job_id:
            message += f"\n\nJob ID: <code>{job_id}</code>"

        try:
            await self.send(
                chat_id=self._chat_id,
                content=message,
                content_type="text",
                parse_mode=ParseMode.HTML,
            )
        except Exception as e:
            logger.error(f"Failed to send error notification: {e}")

    async def notify_warning(self, warning_message: str, job_id: str | None = None):
        """Notify user of warning"""
        if not self._chat_id:
            return

        message = "⚠️ <b>Warning</b>\n\n" + warning_message
        if job_id:
            message += f"\n\nJob ID: <code>{job_id}</code>"

        try:
            await self.send(
                chat_id=self._chat_id,
                content=message,
                content_type="text",
                parse_mode=ParseMode.HTML,
            )
        except Exception as e:
            logger.error(f"Failed to send warning notification: {e}")

    async def NOTHING(self, confidence: float, job_id: str):
        """Notify user that NOTHING has been detected"""
        if not self._chat_id:
            return

        message = (
            "🍝 <b>NOTHING Detected!</b>\n\n"
            f"Confidence: {confidence:.1%}\n"
            f"Job ID: <code>{job_id}</code>\n\n"
            "Bot has paused the print. Please check the printer."
        )

        try:
            await self.send(
                chat_id=self._chat_id,
                content=message,
                content_type="text",
                parse_mode=ParseMode.HTML,
            )
        except Exception as e:
            logger.error(f"Failed to send NOTHING notification: {e}")

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

        await update.effective_chat.send_action("typing")

        try:
            response = await self.chat_manager.chat(user_id, message)
            await update.message.reply_text(response)
        except Exception as e:
            logger.error(f"LLM chat error: {e}", exc_info=True)
            await update.message.reply_text(
                "Sorry, I encountered an error processing your request. Please try again later."
            )

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
