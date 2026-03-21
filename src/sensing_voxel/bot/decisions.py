"""
Telegram bot decision UI module

Manages inline keyboards and callback handling for agent decisions.
Implements callback_data protocol with prefix-based encoding
to stay within Telegram's 64-byte limit (ref: OpenClaw model-buttons.ts).
"""

import logging
from collections.abc import Callable
from datetime import datetime

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from sensing_voxel.core.models import (
    DecisionAction,
    JobContext,
)

logger = logging.getLogger(__name__)


class DecisionUI:
    """
    Decision UI manager for Telegram inline keyboards.

    Encodes decision actions and job IDs into compact callback_data strings.
    Prefix scheme (max 64 bytes per Telegram spec):
    - "apr_{job_id[:8]}" - Approve
    - "rej_{job_id[:8]}" - Reject
    - "res_{job_id[:8]}" - Resume
    - "can_{job_id[:8]}" - Cancel
    - "to_a_{job_id[:8]}" - Timeout approval
    - "to_s_{job_id[:8]}" - Timeout NOTHING pause
    """

    # Callback data prefixes (must be <= 64 bytes total with job ID)
    PREFIX_APPROVE = "apr_"
    PREFIX_REJECT = "rej_"
    PREFIX_RESUME = "res_"
    PREFIX_CANCEL = "can_"
    PREFIX_TIMEOUT_APPROVAL = "to_a_"
    NOTHING = "to_s_"

    MAX_JOB_ID_LENGTH = 56  # 64 - len(longest_prefix)

    def __init__(self):
        # Callback handlers for each decision type
        self._handlers: dict[str, Callable] = {}
        self._default_handler: Callable | None = None

    def create_approval_keyboard(self, job_context: JobContext) -> InlineKeyboardMarkup:
        """
        Create approval/rejection keyboard for print job.

        Args:
            job_context: Job context with decision token

        Returns:
            Inline keyboard markup
        """
        job_id_prefix = job_context.job_id[: self.MAX_JOB_ID_LENGTH]

        keyboard = [
            [
                InlineKeyboardButton(
                    "✅ Approve",
                    callback_data=f"{self.PREFIX_APPROVE}{job_id_prefix}",
                ),
                InlineKeyboardButton(
                    "❌ Reject",
                    callback_data=f"{self.PREFIX_REJECT}{job_id_prefix}",
                ),
            ]
        ]

        return InlineKeyboardMarkup(keyboard)

    def create_pause_keyboard(self, job_context: JobContext) -> InlineKeyboardMarkup:
        """
        Create resume/cancel keyboard for paused print.

        Args:
            job_context: Job context with decision token

        Returns:
            Inline keyboard markup
        """
        job_id_prefix = job_context.job_id[: self.MAX_JOB_ID_LENGTH]

        keyboard = [
            [
                InlineKeyboardButton(
                    "⏯️ Resume",
                    callback_data=f"{self.PREFIX_RESUME}{job_id_prefix}",
                ),
                InlineKeyboardButton(
                    "⏹️ Cancel",
                    callback_data=f"{self.PREFIX_CANCEL}{job_id_prefix}",
                ),
            ]
        ]

        return InlineKeyboardMarkup(keyboard)

    def parse_callback_data(self, callback_data: str) -> tuple[DecisionAction | None, str | None]:
        """
        Parse callback_data to extract decision action and job ID.

        Args:
            callback_data: Raw callback data from Telegram

        Returns:
            Tuple of (DecisionAction or None, job_id_prefix or None)
        """
        if callback_data.startswith(self.PREFIX_APPROVE):
            action = DecisionAction.APPROVE
            job_id_prefix = callback_data[len(self.PREFIX_APPROVE) :]
        elif callback_data.startswith(self.PREFIX_REJECT):
            action = DecisionAction.REJECT
            job_id_prefix = callback_data[len(self.PREFIX_REJECT) :]
        elif callback_data.startswith(self.PREFIX_RESUME):
            action = DecisionAction.RESUME
            job_id_prefix = callback_data[len(self.PREFIX_RESUME) :]
        elif callback_data.startswith(self.PREFIX_CANCEL):
            action = DecisionAction.CANCEL
            job_id_prefix = callback_data[len(self.PREFIX_CANCEL) :]
        elif callback_data.startswith(self.PREFIX_TIMEOUT_APPROVAL):
            action = DecisionAction.TIMEOUT_APPROVAL
            job_id_prefix = callback_data[len(self.PREFIX_TIMEOUT_APPROVAL) :]
        elif callback_data.startswith(self.NOTHING):
            action = DecisionAction.NOTHING
            job_id_prefix = callback_data[len(self.NOTHING) :]
        else:
            logger.warning(f"Unknown callback data prefix: {callback_data[:20]}")
            return None, None

        return action, job_id_prefix

    def register_handler(self, action: DecisionAction, handler: Callable) -> None:
        """
        Register handler for decision action.

        Args:
            action: Decision action type
            handler: Async callback function(job_id_prefix: str, query, context)
        """
        prefix_map = {
            DecisionAction.APPROVE: self.PREFIX_APPROVE,
            DecisionAction.REJECT: self.PREFIX_REJECT,
            DecisionAction.RESUME: self.PREFIX_RESUME,
            DecisionAction.CANCEL: self.PREFIX_CANCEL,
            DecisionAction.TIMEOUT_APPROVAL: self.PREFIX_TIMEOUT_APPROVAL,
            DecisionAction.NOTHING: self.NOTHING,
        }

        prefix = prefix_map.get(action)
        if prefix:
            self._handlers[prefix] = handler
            logger.debug(f"Registered handler for {action.name}")

    def register_default_handler(self, handler: Callable) -> None:
        """
        Register default handler for unrecognized callback data.

        Args:
            handler: Async callback function(callback_data: str, query, context)
        """
        self._default_handler = handler
        logger.debug("Registered default handler")

    async def handle_callback(self, callback_data: str, query, context) -> bool:
        """
        Route callback to appropriate handler.

        Args:
            callback_data: Raw callback data
            query: Telegram update query
            context: Telegram context

        Returns:
            True if handled, False otherwise
        """
        action, job_id_prefix = self.parse_callback_data(callback_data)

        if action is None:
            if self._default_handler:
                await self._default_handler(callback_data, query, context)
                return True
            logger.warning(f"No handler for callback data: {callback_data[:20]}")
            return False

        prefix_map = {
            DecisionAction.APPROVE: self.PREFIX_APPROVE,
            DecisionAction.REJECT: self.PREFIX_REJECT,
            DecisionAction.RESUME: self.PREFIX_RESUME,
            DecisionAction.CANCEL: self.PREFIX_CANCEL,
        }

        prefix = prefix_map.get(action)
        if prefix and prefix in self._handlers:
            handler = self._handlers[prefix]
            await handler(job_id_prefix, query, context)
            return True

        if self._default_handler:
            await self._default_handler(callback_data, query, context)
            return True

        logger.warning(f"No handler registered for action {action.name} (prefix {prefix})")
        return False

    def get_action_description(self, action: DecisionAction) -> str:
        """
        Get human-readable description for decision action.

        Args:
            action: Decision action

        Returns:
            Description string
        """
        descriptions = {
            DecisionAction.APPROVE: "Approve print job",
            DecisionAction.REJECT: "Reject print job",
            DecisionAction.RESUME: "Resume paused print",
            DecisionAction.CANCEL: "Cancel paused print",
            DecisionAction.TIMEOUT_APPROVAL: "Approval timeout",
            DecisionAction.NOTHING: "NOTHING pause timeout",
        }
        return descriptions.get(action, action.name)

    @staticmethod
    def generate_approval_message(
        job_context: JobContext,
        job_summary: str,
        expires_at: datetime | None = None,
        slicer_backend: str | None = None,
        slicer_version: str | None = None,
    ) -> str:
        """
        Generate approval message text.

        Args:
            job_context: Job context
            job_summary: Formatted job summary from slicer
            expires_at: Optional expiration timestamp
            slicer_backend: Optional slicer backend name (e.g., 'orcaslicer', 'prusaslicer')
            slicer_version: Optional slicer version string

        Returns:
            Message text with HTML formatting
        """
        message = (
            "<b>🖨️ New Print Job Ready</b>\n\n"
            f"<b>File:</b> {job_context.job_file}\n"
            f"<b>Job ID:</b> <code>{job_context.job_id[:8]}</code>\n"
        )

        # Add slicer info if available
        if slicer_backend:
            slicer_name = "OrcaSlicer" if slicer_backend == "orcaslicer" else "PrusaSlicer"
            if slicer_version:
                message += f"<b>Sliced with:</b> {slicer_name} v{slicer_version}\n"
            else:
                message += f"<b>Sliced with:</b> {slicer_name}\n"

        message += (
            f"\n{job_summary}\n\n<i>Use the buttons below to approve or reject this print.</i>"
        )

        if expires_at:
            now = datetime.now()
            remaining = int((expires_at - now).total_seconds())
            if remaining > 0:
                message += f"\n\n⏱️ <i>Auto-rejects in {remaining}s if no response</i>"

        return message

    @staticmethod
    def generate_pause_message(
        job_context: JobContext,
        reason: str,
        current_state: str = "",
    ) -> str:
        """
        Generate pause message text.

        Args:
            job_context: Job context
            reason: Reason for pause (e.g., "NOTHING detected")
            current_state: Optional current printer state info

        Returns:
            Message text with HTML formatting
        """
        message = (
            f"<b>⏸️ Print Paused</b>\n\n"
            f"<b>Job:</b> {job_context.job_file}\n"
            f"<b>Job ID:</b> <code>{job_context.job_id[:8]}</code>\n\n"
            f"<b>Reason:</b> {reason}\n"
        )

        if current_state:
            message += f"\n<b>Status:</b> {current_state}\n"

        message += "\n<i>Use the buttons below to resume or cancel the print.</i>"

        return message

    @staticmethod
    def generate_timeout_message(action: DecisionAction, job_context: JobContext) -> str:
        """
        Generate timeout message text.

        Args:
            action: Timeout action type
            job_context: Job context

        Returns:
            Message text
        """
        if action == DecisionAction.TIMEOUT_APPROVAL:
            return (
                "⏰ <b>Approval Timeout</b>\n\n"
                f"Print job <code>{job_context.job_id[:8]}</code> was automatically "
                "rejected due to no response."
            )
        elif action == DecisionAction.NOTHING:
            return (
                "⏰ <b>Pause Timeout</b>\n\n"
                f"Paused print <code>{job_context.job_id[:8]}</code> was automatically "
                "cancelled due to no response."
            )
        return f"Timeout: {action.name}"

    @staticmethod
    def generate_decision_consumed_message(action: DecisionAction, job_context: JobContext) -> str:
        """
        Generate message for already-consumed decision.

        Args:
            action: Attempted action
            job_context: Job context

        Returns:
            Message text
        """
        return (
            "⚠️ <b>Already Handled</b>\n\n"
            f"This decision for job <code>{job_context.job_id[:8]}</code> "
            "has already been processed.\n\n"
            "<i>You may have clicked twice or this is a stale button.</i>"
        )


# Singleton instance for convenience
decision_ui = DecisionUI()
