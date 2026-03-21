"""
StructuredLogger — JSON logging with pipeline context.

Wraps the standard ``logging`` module to emit JSON-formatted log records
that include ``run_id``, ``lob``, and arbitrary tags from the active
``PipelineContext``.

Usage
-----
>>> from src.observability.context import PipelineContext
>>> from src.observability.logging import StructuredLogger, setup_logging
>>> setup_logging(format="json", level="INFO")
>>> ctx = PipelineContext(lob="rossmann")
>>> log = StructuredLogger("my_module", context=ctx)
>>> log.info("Model trained", model="lgbm_direct", series_count=1200)
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from typing import Any, Optional

from .context import PipelineContext


class StructuredLogger:
    """
    Logger that enriches every record with pipeline context fields.

    Parameters
    ----------
    name:
        Logger name (usually ``__name__``).
    context:
        Active ``PipelineContext`` (optional — can be set later via
        ``set_context()``).
    """

    def __init__(self, name: str, context: Optional[PipelineContext] = None):
        self.logger = logging.getLogger(name)
        self.context = context

    def set_context(self, context: PipelineContext) -> None:
        """Attach or update the pipeline context."""
        self.context = context

    def _build_record(self, level: str, msg: str, **extra: Any) -> str:
        record = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "msg": msg,
        }
        if self.context:
            record["run_id"] = self.context.run_id
            record["lob"] = self.context.lob
            if self.context.parent_run_id:
                record["parent_run_id"] = self.context.parent_run_id
        record.update(extra)
        return json.dumps(record, default=str)

    def debug(self, msg: str, **extra: Any) -> None:
        """Emit a DEBUG-level structured log."""
        self.logger.debug(self._build_record("DEBUG", msg, **extra))

    def info(self, msg: str, **extra: Any) -> None:
        """Emit an INFO-level structured log."""
        self.logger.info(self._build_record("INFO", msg, **extra))

    def warning(self, msg: str, **extra: Any) -> None:
        """Emit a WARNING-level structured log."""
        self.logger.warning(self._build_record("WARNING", msg, **extra))

    def error(self, msg: str, **extra: Any) -> None:
        """Emit an ERROR-level structured log."""
        self.logger.error(self._build_record("ERROR", msg, **extra))

    def exception(self, msg: str, **extra: Any) -> None:
        """Emit an ERROR-level structured log with exception info."""
        self.logger.exception(self._build_record("ERROR", msg, **extra))


def setup_logging(
    format: str = "text",
    level: str = "INFO",
) -> None:
    """
    Configure root logging format.

    Parameters
    ----------
    format:
        ``"text"`` — standard human-readable format.
        ``"json"`` — one JSON object per line (for log aggregators).
    level:
        Log level string (``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``).
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicates
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))

    if format == "json":
        # Minimal formatter — StructuredLogger already produces JSON
        handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))

    root.addHandler(handler)
