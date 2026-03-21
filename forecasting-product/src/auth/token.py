"""
JWT token creation and validation.

Pluggable token provider — swap for OAuth2/SAML by replacing
create_token / decode_token while keeping the same interface.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# PyJWT is optional — graceful degradation
try:
    import jwt as pyjwt
    _HAS_JWT = True
except ImportError:
    _HAS_JWT = False


DEFAULT_EXPIRY_HOURS = 24


def create_token(
    user_id: str,
    email: str,
    role: str,
    secret_key: str,
    expiry_hours: int = DEFAULT_EXPIRY_HOURS,
    extra_claims: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Create a signed JWT token.

    Parameters
    ----------
    user_id:
        Unique user identifier.
    email:
        User email.
    role:
        User role string (e.g. 'admin', 'planner').
    secret_key:
        HMAC secret for signing.
    expiry_hours:
        Token validity in hours.
    extra_claims:
        Additional claims to embed in the token.

    Returns
    -------
    Encoded JWT string.
    """
    if not _HAS_JWT:
        raise RuntimeError(
            "PyJWT is required for token operations. Install with: pip install PyJWT"
        )

    now = datetime.now(timezone.utc)
    payload = {
        "user_id": user_id,
        "email": email,
        "role": role,
        "is_active": True,
        "iat": now,
        "exp": now + timedelta(hours=expiry_hours),
    }
    if extra_claims:
        payload.update(extra_claims)

    return pyjwt.encode(payload, secret_key, algorithm="HS256")


def decode_token(
    token: str,
    secret_key: str,
) -> Optional[Dict[str, Any]]:
    """
    Decode and validate a JWT token.

    Returns None if the token is invalid or expired.
    """
    if not _HAS_JWT:
        logger.error("PyJWT not installed — cannot decode tokens.")
        return None

    try:
        payload = pyjwt.decode(token, secret_key, algorithms=["HS256"])
        return payload
    except pyjwt.ExpiredSignatureError:
        logger.warning("Token expired.")
        return None
    except pyjwt.InvalidTokenError as e:
        logger.warning("Invalid token: %s", e)
        return None
