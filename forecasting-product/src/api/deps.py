"""Shared FastAPI dependencies for router endpoints."""

from __future__ import annotations

from fastapi import Request


def get_app_state(request: Request):
    """Provide app.state to router endpoints via dependency injection."""
    return request.app.state
