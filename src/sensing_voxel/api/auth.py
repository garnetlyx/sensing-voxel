"""Bearer token authentication middleware for dashboard API"""

import logging

from aiohttp import web

logger = logging.getLogger(__name__)

# Paths that skip authentication
PUBLIC_PATHS = {"/health", "/", "/favicon.ico"}
PUBLIC_PREFIXES = ("/static/",)


def create_auth_middleware(api_token: str | None):
    """Create auth middleware. If api_token is None, auth is disabled (dev mode)."""

    @web.middleware
    async def auth_middleware(request: web.Request, handler):
        # Skip auth if no token configured (dev mode)
        if api_token is None:
            return await handler(request)

        # Skip auth for public paths
        path = request.path
        if path in PUBLIC_PATHS:
            return await handler(request)
        if any(path.startswith(prefix) for prefix in PUBLIC_PREFIXES):
            return await handler(request)

        # Check Authorization header
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise web.HTTPUnauthorized(
                text="Missing or invalid Authorization header",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = auth_header[7:]  # Strip "Bearer "
        if token != api_token:
            logger.warning(f"Invalid API token from {request.remote}")
            raise web.HTTPUnauthorized(
                text="Invalid API token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return await handler(request)

    return auth_middleware
