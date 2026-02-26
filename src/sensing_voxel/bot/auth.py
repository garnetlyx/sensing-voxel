"""
Telegram bot authentication module

Manages user allowlist and authentication for all bot interactions.
"""

import logging

logger = logging.getLogger(__name__)


class TelegramAuth:
    """Telegram bot authentication manager"""

    def __init__(self, allowed_user_ids: set[int] | None = None):
        """
        Initialize authentication manager.

        Args:
            allowed_user_ids: Set of authorized Telegram user/chat IDs.
                             If None or empty, all users are allowed (dev mode).
        """
        self.allowed_user_ids = allowed_user_ids or set()
        self._dev_mode = len(self.allowed_user_ids) == 0

        if self._dev_mode:
            logger.warning(
                "Telegram authentication in dev mode - all users allowed. "
                "Configure allowed_user_ids in production."
            )
        else:
            logger.info(f"Telegram authentication enabled for {len(self.allowed_user_ids)} users")

    def is_authorized(self, user_id: int) -> bool:
        """
        Check if user is authorized to interact with bot.

        Args:
            user_id: Telegram user or chat ID

        Returns:
            True if authorized, False otherwise
        """
        if self._dev_mode:
            return True
        return user_id in self.allowed_user_ids

    def authorize_user(self, user_id: int):
        """
        Add user to allowlist.

        Args:
            user_id: Telegram user or chat ID to authorize
        """
        self.allowed_user_ids.add(user_id)
        self._dev_mode = False
        logger.info(f"Authorized user {user_id}")

    def revoke_user(self, user_id: int):
        """
        Remove user from allowlist.

        Args:
            user_id: Telegram user or chat ID to revoke
        """
        if user_id in self.allowed_user_ids:
            self.allowed_user_ids.remove(user_id)
            logger.info(f"Revoked user {user_id}")
        else:
            logger.warning(f"User {user_id} not in allowlist")

    def get_authorized_users(self) -> set[int]:
        """
        Get set of authorized user IDs.

        Returns:
            Set of authorized user IDs
        """
        return self.allowed_user_ids.copy()

    def is_dev_mode(self) -> bool:
        """
        Check if authentication is in dev mode (no restrictions).

        Returns:
            True if dev mode, False otherwise
        """
        return self._dev_mode

    def require_auth(self, user_id: int, action: str = "action") -> bool:
        """
        Require authentication and log rejection if unauthorized.

        Args:
            user_id: Telegram user or chat ID
            action: Action being attempted (for logging)

        Returns:
            True if authorized, False otherwise
        """
        if not self.is_authorized(user_id):
            logger.warning(
                f"Unauthorized {action} attempt from user {user_id}. "
                f"Authorized users: {self.allowed_user_ids}"
            )
            return False
        return True

    def get_rejection_message(self) -> str:
        """
        Get standard rejection message for unauthorized users.

        Returns:
            Rejection message
        """
        return (
            "🚫 Access denied. You are not authorized to use this bot.\n\n"
            "Please contact the administrator to be added to the allowlist."
        )
