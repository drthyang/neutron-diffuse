"""FastAPI dependencies."""

from __future__ import annotations

from fastapi import Request

from nebula3d.server.config import ServerConfig


def get_config(request: Request) -> ServerConfig:
    """Return the :class:`ServerConfig` stored on the app at startup."""
    return request.app.state.config  # type: ignore[no-any-return]
