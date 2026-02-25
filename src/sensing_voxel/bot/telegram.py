"""
Telegram bot integration

Adapted from OpenClaw's telegram patterns
"""

import logging
from typing import Optional, Callable
from datetime import datetime
import asyncio

from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode

from sensing_voxel.core.models import (
    AgentState,
    DecisionAction,
    JobContext,
    generate_job_id,
)
from sensing_voxel.core.state_machine import AgentStateMachine
from sensing_voxel.core.decision import DecisionEngine
from sensing_voxel.adapters.base import PrinterAdapter

logger = logging.getLogger(__name__)


class TelegramBot:
    """Telegram bot for Sensing Voxel"""

    def __init__(
        self,
        token: str,
        allowed_users: list[int],
        state_machine: AgentStateMachine,
        adapter: Optional[PrinterAdapter] = None,
    ):
        self.token = token
        self.allowed_users = set(allowed_users)
        self.state_machine = state_machine
        self.adapter = adapter
        self.decision_engine = DecisionEngine()

        self._app = Application.builder().token(token).build()

        # Register handlers
        self._register_handlers()

        self._decision_callbacks: dict[str, asyncio.Future] = {}
        self._chat_id: Optional[int] = None  # Store chat ID for notifications

    def _register_handlers(self):
        """Register command and message handlers"""
        # Command handlers
        self._app.add_handler(CommandHandler("start", self._handle_start))
        self._app.add_handler(CommandHandler("status", self._handle_status))
        self._app.add_handler(CommandHandler("reset_z_offset", self._handle_reset_z_offset))
        self._app.add_handler(CommandHandler("calibrate_camera", self._handle_calibrate_camera))

        # Message handlers
        self._app.add_handler(
            MessageHandler(filters.Document.ALL, self._handle_file_upload)
        )

        # Callback query handler (inline keyboard buttons)
        self._app.add_handler(CallbackQueryHandler(self._handle_callback_query))

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
            "👋 Welcome to Sensing Voxel!\n\n"
            "🤖 AI-powered 3D printer monitoring and control\n\n"
            "📤 Send me an STL file to start a print\n"
            "📊 Use /status to check current state\n"
            "⚙️ Use /reset_z_offset to reset Z-offset\n"
            "📹 Use /calibrate_camera to calibrate camera\n"
        )

        await update.message.reply_text(message)

    async def _handle_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        if not await self._check_auth(update):
            return

        state = self.state_machine.current_state
        job = self.state_machine.job_context

        message = f"📊 **Current State:** `{state.name}`\n\n"

        if job:
            message += f"🖨️ **Job ID:** `{job.job_id}`\n"
            message += f"📁 **File:** `{job.job_file}`\n"
            if job.decision_consumed:
                message += f"✅ **Decision:** {job.decision_action.name}\n"

        if self.state_machine.z_offset_dirty:
            message += "\n⚠️ **Z-offset is dirty** - prints blocked until reset\n"

        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)

    async def _handle_file_upload(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle file upload"""
        if not await self._check_auth(update):
            return

        document = update.message.document
        if not document:
            return

        # Check file type
        if not document.file_name.lower().endswith(".stl"):
            await update.message.reply_text("❌ Only STL files are supported")
            return

        # Check file size (limit to 100MB)
        if document.file_size > 100 * 1024 * 1024:
            await update.message.reply_text("❌ File too large (max 100MB)")
            return

        await update.message.reply_text("🔄 Processing your STL file...")

        # Store chat ID for future notifications
        self._chat_id = update.effective_chat.id

        # Check if we're already in a job
        if self.state_machine.current_state != AgentState.IDLE:
            await update.message.reply_text(
                f"⚠️ Agent is busy in state {self.state_machine.current_state.name}. "
                "Please wait or cancel current job first."
            )
            return

        # Download and stage file
        try:
            file = await document.get_file()
            uploads_dir = self.state_machine.config.uploads_path if hasattr(self.state_machine, "config") else Path("data/uploads")
            uploads_dir.mkdir(parents=True, exist_ok=True)

            # Generate job ID and file path
            job_id = generate_job_id()
            stl_path = uploads_dir / f"{job_id}.stl"

            # Download file
            await file.download_to_drive(str(stl_path))

            # Create job context
            from sensing_voxel.core.models import FileValidationStatus
            job_context = JobContext(
                job_id=job_id,
                job_file=str(stl_path),
                uploaded_at=datetime.now(),
                decision_token=generate_decision_token(),
                file_validation_status=FileValidationStatus.VALID,
                is_external=False,
                external_monitoring=False,
            )

            # Transition to RECEIVING state
            self.state_machine.job_context = job_context
            success = await self.state_machine.transition_to(
                AgentState.RECEIVING,
                persistence=self.state_machine.persistence if hasattr(self.state_machine, "persistence") else None
            )

            if success:
                await update.message.reply_text(
                    f"✅ File received: {document.file_name}\n"
                    f"Job ID: `{job_id}`\n\n"
                    "🔄 Starting slicing pipeline..."
                )
                # TODO: Trigger slicing pipeline
            else:
                await update.message.reply_text(
                    "❌ Failed to start job. Please try again."
                )

        except Exception as e:
            logger.error(f"Error processing file upload: {e}", exc_info=True)
            await update.message.reply_text(
                f"❌ Error processing file: {str(e)}"
            )

    async def _handle_callback_query(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
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

        prefix, job_id_short = parts

        # Find matching job
        job = self.state_machine.job_context
        if not job or not job.job_id.startswith(job_id_short):
            await query.edit_message_text("❌ Job not found or expired")
            return

        action: Optional[DecisionAction] = None

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
        from sensing_voxel.core.models import ConsumeResult
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
                f"✅ Decision recorded: {action.name}\n"
                f"Job ID: `{job.job_id[:12]}`"
            )

            # Handle action-specific transitions
            if action == DecisionAction.APPROVE:
                # TODO: Start upload and print
                await self._app.bot.send_message(
                    chat_id=self._chat_id,
                    text="🚀 Starting print..."
                )
                # Transition to UPLOADING
                await self.state_machine.transition_to(
                    AgentState.UPLOADING,
                    persistence=self.state_machine.persistence if hasattr(self.state_machine, "persistence") else None
                )
            elif action == DecisionAction.REJECT:
                # Transition to IDLE
                await self.state_machine.transition_to(
                    AgentState.IDLE,
                    persistence=self.state_machine.persistence if hasattr(self.state_machine, "persistence") else None
                )
                await self._app.bot.send_message(
                    chat_id=self._chat_id,
                    text="🛑 Job rejected. Back to idle."
                )
            elif action == DecisionAction.RESUME:
                # Resume print
                if self.adapter:
                    await self.adapter.resume_print()
                await self.state_machine.transition_to(
                    AgentState.MONITORING,
                    persistence=self.state_machine.persistence if hasattr(self.state_machine, "persistence") else None
                )
                await self._app.bot.send_message(
                    chat_id=self._chat_id,
                    text="▶️ Print resumed."
                )
            elif action == DecisionAction.CANCEL:
                # Cancel print
                if self.adapter:
                    await self.adapter.cancel_print()
                await self.state_machine.transition_to(
                    AgentState.IDLE,
                    persistence=self.state_machine.persistence if hasattr(self.state_machine, "persistence") else None
                )
                await self._app.bot.send_message(
                    chat_id=self._chat_id,
                    text="⏹️ Print cancelled."
                )

        elif result == ConsumeResult.ALREADY_HANDLED:
            await query.edit_message_text(
                "ℹ️ Decision already processed for this job."
            )
        else:
            await query.edit_message_text(
                f"⚠️ Decision rejected: {result.name}"
            )

    async def _handle_reset_z_offset(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
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

        # Check current state
        if self.state_machine.z_offset_dirty and self.state_machine.current_state != AgentState.IDLE:
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
                    if hasattr(self.state_machine, "persistence"):
                        self.state_machine.persistence.save(self.state_machine)

                    await update.message.reply_text(
                        f"✅ Z-offset reset to {reset_type} ({target_mm:.3f}mm)\n"
                        f"State: {'CLEAN' if not self.state_machine.z_offset_dirty else 'DIRTY'}"
                    )
                else:
                    await update.message.reply_text(
                        f"❌ Failed to reset Z-offset: {result.message}"
                    )
            except Exception as e:
                await update.message.reply_text(
                    f"❌ Error resetting Z-offset: {str(e)}"
                )
        else:
            # No adapter configured - just update state
            self.state_machine.z_offset_dirty = False
            self.state_machine.last_known_z_offset_baseline = target_mm

            await update.message.reply_text(
                f"⚠️ No printer adapter configured. "
                f"Z-offset state reset to {reset_type} ({target_mm:.3f}mm) in memory only."
            )

    async def _handle_calibrate_camera(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /calibrate_camera command"""
        if not await self._check_auth(update):
            return

        # Check if camera is configured
        if not hasattr(self.state_machine, "config") or not self.state_machine.config.camera:
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

    async def send_message(
        self,
        chat_id: int,
        text: str,
        parse_mode: ParseMode = ParseMode.HTML,
        reply_markup: Optional[InlineKeyboardMarkup] = None,
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

    def create_decision_keyboard(
        self, job_id: str, state: AgentState
    ) -> InlineKeyboardMarkup:
        """Create inline keyboard for decisions (OpenClaw pattern)"""
        buttons = []

        if state == AgentState.AWAITING_APPROVAL:
            buttons = [
                [
                    InlineKeyboardButton(
                        "✅ Approve", callback_data=f"apr_{job_id[:8]}"
                    ),
                    InlineKeyboardButton(
                        "❌ Reject", callback_data=f"rej_{job_id[:8]}"
                    ),
                ]
            ]
        elif state == AgentState.PAUSED_PENDING_USER:
            buttons = [
                [
                    InlineKeyboardButton(
                        "▶️ Resume", callback_data=f"res_{job_id[:8]}"
                    ),
                    InlineKeyboardButton(
                        "⏹️ Cancel", callback_data=f"can_{job_id[:8]}"
                    ),
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
