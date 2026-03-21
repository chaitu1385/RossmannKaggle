"""
RBAC enforcement for FastAPI endpoints.

Provides FastAPI dependency functions that extract the current user
from a JWT token and check permissions before the endpoint executes.
"""

import logging
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .models import Permission, Role, User
from .token import decode_token

logger = logging.getLogger(__name__)

_security = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_security),
) -> User:
    """
    FastAPI dependency: extract and validate the current user from JWT.

    If no auth is configured (e.g. development mode), returns a default
    admin user to preserve backward compatibility.
    """
    # Check if auth is enabled (set on app.state by create_app)
    auth_enabled = getattr(request.app.state, "auth_enabled", False)

    if not auth_enabled:
        # Development mode: return default admin
        return User(
            user_id="system",
            email="system@localhost",
            role=Role.ADMIN,
        )

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide a Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    secret_key = getattr(request.app.state, "jwt_secret", "")
    payload = decode_token(credentials.credentials, secret_key)

    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        role = Role(payload.get("role", "viewer"))
    except ValueError:
        role = Role.VIEWER

    user = User(
        user_id=payload.get("user_id", "unknown"),
        email=payload.get("email", ""),
        role=role,
        is_active=payload.get("is_active", True),
    )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated.",
        )

    return user


def require_permission(permission: Permission):
    """
    FastAPI dependency factory: ensure the current user has a specific permission.

    Usage::

        @app.get("/admin/users")
        def list_users(user: User = Depends(require_permission(Permission.MANAGE_USERS))):
            ...
    """
    async def _check(user: User = Depends(get_current_user)) -> User:
        if not user.has_permission(permission):
            logger.warning(
                "Permission denied: user=%s role=%s permission=%s",
                user.user_id, user.role.value, permission.value,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission.value}' required. "
                       f"Your role '{user.role.value}' does not have this permission.",
            )
        return user

    return _check


def require_role(*roles: Role):
    """
    FastAPI dependency factory: ensure the current user has one of the specified roles.

    Usage::

        @app.post("/models/promote")
        def promote(user: User = Depends(require_role(Role.ADMIN, Role.DATA_SCIENTIST))):
            ...
    """
    async def _check(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            role_names = ", ".join(r.value for r in roles)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"One of roles [{role_names}] required. "
                       f"Your role: '{user.role.value}'.",
            )
        return user

    return _check
