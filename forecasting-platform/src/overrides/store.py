"""
Planner override persistence via DuckDB.

Stores human-in-the-loop overrides for product transitions:
  - Proportion adjustments
  - Scenario overrides (force Scenario A/B/C/manual)
  - Ramp shape changes

DuckDB is used because it's zero-server, file-based, SQL-queryable,
and natively interops with Polars via Arrow.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl

# DuckDB is optional — graceful fallback to CSV if unavailable
try:
    import duckdb
    _HAS_DUCKDB = True
except ImportError:
    _HAS_DUCKDB = False


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS transition_overrides (
    override_id     VARCHAR PRIMARY KEY,
    old_sku         VARCHAR NOT NULL,
    new_sku         VARCHAR NOT NULL,
    effective_date  DATE,
    scenario        VARCHAR DEFAULT 'manual',
    proportion      DOUBLE,
    ramp_shape      VARCHAR DEFAULT 'linear',
    created_by      VARCHAR,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes           VARCHAR,
    status          VARCHAR DEFAULT 'approved',
    approved_by     VARCHAR,
    approved_at     TIMESTAMP
)
"""


class OverrideStore:
    """
    Persistent store for planner transition overrides.

    If DuckDB is available, uses a file-based DuckDB database.
    Otherwise, falls back to reading/writing a CSV file.
    """

    def __init__(self, db_path: str = "data/overrides.duckdb"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        if _HAS_DUCKDB:
            self._conn = duckdb.connect(str(self.db_path))
            self._conn.execute(_CREATE_TABLE_SQL)
        else:
            self._conn = None
            self._csv_path = self.db_path.with_suffix(".csv")

    def add_override(
        self,
        old_sku: str,
        new_sku: str,
        proportion: float,
        scenario: str = "manual",
        ramp_shape: str = "linear",
        effective_date: Optional[str] = None,
        created_by: Optional[str] = None,
        notes: Optional[str] = None,
        approval_threshold: float = 0.0,
    ) -> str:
        """
        Add a planner override.

        Parameters
        ----------
        approval_threshold:
            If proportion exceeds this threshold, the override is created
            with status 'pending_approval' and requires manager sign-off.
            A threshold of 0.0 means all overrides are auto-approved.

        Returns the generated override_id.
        """
        override_id = f"OVR-{uuid.uuid4().hex[:8].upper()}"

        # Determine approval status based on threshold
        if approval_threshold > 0.0 and proportion > approval_threshold:
            status = "pending_approval"
        else:
            status = "approved"

        if self._conn:
            self._conn.execute(
                """
                INSERT INTO transition_overrides
                (override_id, old_sku, new_sku, effective_date, scenario,
                 proportion, ramp_shape, created_by, notes, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    override_id, old_sku, new_sku, effective_date,
                    scenario, proportion, ramp_shape, created_by, notes,
                    status,
                ],
            )
        else:
            # CSV fallback
            record = pl.DataFrame({
                "override_id": [override_id],
                "old_sku": [old_sku],
                "new_sku": [new_sku],
                "effective_date": [effective_date],
                "scenario": [scenario],
                "proportion": [proportion],
                "ramp_shape": [ramp_shape],
                "created_by": [created_by],
                "created_at": [datetime.now().isoformat()],
                "notes": [notes],
                "status": [status],
                "approved_by": [None],
                "approved_at": [None],
            })
            if self._csv_path.exists():
                existing = pl.read_csv(str(self._csv_path))
                combined = pl.concat(
                    [existing, record], how="vertical_relaxed"
                )
                combined.write_csv(str(self._csv_path))
            else:
                record.write_csv(str(self._csv_path))

        return override_id

    def get_overrides(
        self,
        old_sku: Optional[str] = None,
        new_sku: Optional[str] = None,
    ) -> pl.DataFrame:
        """Retrieve overrides, optionally filtered by SKU."""
        if self._conn:
            query = "SELECT * FROM transition_overrides WHERE 1=1"
            params = []
            if old_sku:
                query += " AND old_sku = ?"
                params.append(old_sku)
            if new_sku:
                query += " AND new_sku = ?"
                params.append(new_sku)
            result = self._conn.execute(query, params).pl()
            return result
        else:
            if not self._csv_path.exists():
                return pl.DataFrame()
            df = pl.read_csv(str(self._csv_path))
            if old_sku:
                df = df.filter(pl.col("old_sku") == old_sku)
            if new_sku:
                df = df.filter(pl.col("new_sku") == new_sku)
            return df

    def get_all(self) -> pl.DataFrame:
        """Retrieve all overrides."""
        return self.get_overrides()

    def delete_override(self, override_id: str) -> bool:
        """Remove an override by ID."""
        if self._conn:
            self._conn.execute(
                "DELETE FROM transition_overrides WHERE override_id = ?",
                [override_id],
            )
            return True
        return False

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
