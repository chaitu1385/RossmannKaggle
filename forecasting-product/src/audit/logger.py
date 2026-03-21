"""
Append-only audit log backed by Parquet files.

Writes are append-only (no UPDATE, no DELETE). Files are partitioned
by date for efficient archival and querying.
"""

import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import List, Optional

import polars as pl

from .schemas import AuditEvent

logger = logging.getLogger(__name__)

# Schema for the audit log Parquet files
AUDIT_SCHEMA = {
    "audit_id": pl.Utf8,
    "timestamp": pl.Datetime,
    "user_id": pl.Utf8,
    "user_email": pl.Utf8,
    "user_role": pl.Utf8,
    "action": pl.Utf8,
    "resource_type": pl.Utf8,
    "resource_id": pl.Utf8,
    "status": pl.Utf8,
    "old_value": pl.Utf8,
    "new_value": pl.Utf8,
    "ip_address": pl.Utf8,
    "request_id": pl.Utf8,
    "error_message": pl.Utf8,
}


class AuditLogger:
    """
    Append-only Parquet-backed audit log.

    Files are partitioned by date::

        audit_log/
        └── date=2026-03-13/
            └── audit_20260313_143022.parquet

    Usage
    -----
    >>> audit = AuditLogger("data/audit_log/")
    >>> audit.log(AuditEvent(
    ...     action="create_override",
    ...     resource_type="override",
    ...     resource_id="OVR-12345678",
    ...     user_id="jane.doe",
    ... ))
    """

    def __init__(self, base_path: str = "data/audit_log/"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def log(self, event: AuditEvent) -> None:
        """Append a single audit event."""
        self._write_events([event])

    def log_batch(self, events: List[AuditEvent]) -> None:
        """Append multiple audit events at once."""
        if events:
            self._write_events(events)

    def _write_events(self, events: List[AuditEvent]) -> None:
        """Write events to a date-partitioned Parquet file."""
        rows = [e.to_dict() for e in events]
        df = pl.DataFrame(rows)

        today = date.today().isoformat()
        partition_dir = self.base_path / f"date={today}"
        partition_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        filename = f"audit_{timestamp}.parquet"
        path = partition_dir / filename

        df.write_parquet(str(path))
        logger.debug("Audit: %d events written to %s", len(events), path)

    def query(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        status: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 1000,
    ) -> pl.DataFrame:
        """
        Query the audit log with optional filters.

        Returns events sorted by timestamp descending (most recent first).
        """
        pattern = str(self.base_path / "**" / "*.parquet")
        try:
            df = pl.read_parquet(pattern)
        except Exception:
            return pl.DataFrame(schema=AUDIT_SCHEMA)

        if user_id:
            df = df.filter(pl.col("user_id") == user_id)
        if action:
            df = df.filter(pl.col("action") == action)
        if resource_type:
            df = df.filter(pl.col("resource_type") == resource_type)
        if status:
            df = df.filter(pl.col("status") == status)
        if start_date and "timestamp" in df.columns:
            df = df.filter(pl.col("timestamp") >= datetime(start_date.year, start_date.month, start_date.day))
        if end_date and "timestamp" in df.columns:
            df = df.filter(pl.col("timestamp") <= datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59))

        return df.sort("timestamp", descending=True).head(limit)

    def count_by_action(self) -> pl.DataFrame:
        """Summary: count of events grouped by action and status."""
        pattern = str(self.base_path / "**" / "*.parquet")
        try:
            df = pl.read_parquet(pattern)
        except Exception:
            return pl.DataFrame(schema={"action": pl.Utf8, "status": pl.Utf8, "count": pl.UInt32})

        return (
            df.group_by(["action", "status"])
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )
