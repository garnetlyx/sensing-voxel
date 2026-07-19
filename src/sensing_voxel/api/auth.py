"""Bearer token authentication middleware for dashboard API"""

import hmac
import logging

from aiohttp import web

logger = logging.getLogger(__name__)

PUBLIC_PATHS = {"/", "/favicon.ico", "/health"}
WEBSOCKET_PATH = "/api/v1/ws"


def _is_public_path(path: str) -> bool:
    return path in PUBLIC_PATHS or path.startswith("/static/")


def create_auth_middleware(api_token: str | None):
    """Create fail-closed auth middleware for protected HTTP routes.

    The WebSocket endpoint performs its own upgrade authentication so browser
    clients can supply credentials through ``Sec-WebSocket-Protocol``.
    """

    @web.middleware
    async def auth_middleware(request: web.Request, handler):
        if _is_public_path(request.path) or request.path == WEBSOCKET_PATH:
            return await handler(request)

        if not api_token:
            raise web.HTTPUnauthorized(
                text="Dashboard authentication is not configured",
                headers={"WWW-Authenticate": "Bearer"},
            )

        auth_header = request.headers.get("Authorization", "")
        scheme, separator, token = auth_header.partition(" ")
        if separator != " " or scheme.lower() != "bearer" or not token:
            raise web.HTTPUnauthorized(
                text="Missing or invalid Authorization header",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if not hmac.compare_digest(token.encode(), api_token.encode()):
            logger.warning("Invalid API token from %s", request.remote)
            raise web.HTTPUnauthorized(
                text="Invalid API token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return await handler(request)

    return auth_middleware
