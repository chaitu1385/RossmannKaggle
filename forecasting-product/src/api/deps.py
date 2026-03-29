"""Shared FastAPI dependencies for router endpoints."""

from __future__ import annotations

import re

from fastapi import HTTPException, UploadFile

# Regex for safe path-segment identifiers (LOB names, series IDs, model names).
# Allows alphanumeric, hyphens, underscores, and dots (no slashes, no ..).
_SAFE_PATH_SEGMENT = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")

# Maximum upload file size: 100 MB
MAX_UPLOAD_BYTES = 100 * 1024 * 1024


def validate_path_param(value: str, name: str = "parameter") -> str:
    """Validate a path parameter is a safe filesystem identifier.

    Rejects path traversal attempts (../, /, etc.) and empty strings.
    Returns the validated value unchanged.
    """
    if not value or not _SAFE_PATH_SEGMENT.match(value):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid {name}: '{value}'. "
                f"Must be 1-128 characters, alphanumeric/hyphens/underscores/dots, "
                f"and cannot start with a dot or hyphen."
            ),
        )
    return value


async def validate_upload_size(file: UploadFile, max_bytes: int = MAX_UPLOAD_BYTES) -> bytes:
    """Read an uploaded file and enforce a size limit.

    Returns the file content as bytes.
    """
    content = await file.read()
    if len(content) > max_bytes:
        mb = max_bytes / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content) / (1024*1024):.1f} MB). Maximum allowed: {mb:.0f} MB.",
        )
    return content
