"""Serialise MappingRecord objects to Polars DataFrames and CSV files."""

from pathlib import Path
from typing import List

import polars as pl

from ..data.schemas import MappingRecord


class MappingWriter:
    """Converts ``MappingRecord`` objects to the full output schema (§5)."""

    # Column order follows the design document §5 exactly
    _COLUMNS = [
        "mapping_id",
        "old_sku",
        "new_sku",
        "mapping_type",
        "proportion",
        "confidence_score",
        "confidence_level",
        "methods_matched",
        "transition_start_week",
        "transition_end_week",
        "old_sku_lifecycle_stage",
        "validation_status",
        "validated_by",
        "validation_date",
        "notes",
    ]

    def to_polars(self, records: List[MappingRecord]) -> pl.DataFrame:
        """Convert a list of ``MappingRecord`` objects to a Polars DataFrame."""
        if not records:
            return pl.DataFrame(schema=self._schema())

        rows = [self._record_to_dict(r) for r in records]
        return pl.DataFrame(rows, schema=self._schema())

    def save_csv(self, df: pl.DataFrame, path: str) -> None:
        """Write the mapping DataFrame to a CSV file."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Serialise list column to comma-separated string for CSV compatibility
        serialised = df.with_columns(
            pl.col("methods_matched")
            .list.join("|")
            .alias("methods_matched")
        )
        serialised.write_csv(str(out))

    def save_excel(self, df: pl.DataFrame, path: str) -> None:
        """
        Write the mapping DataFrame to an Excel file.

        Requires ``openpyxl`` (``pip install openpyxl``).
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.write_excel(str(out))

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _record_to_dict(self, r: MappingRecord) -> dict:
        return {
            "mapping_id": r.mapping_id,
            "old_sku": r.old_sku,
            "new_sku": r.new_sku,
            "mapping_type": r.mapping_type,
            "proportion": r.proportion,
            "confidence_score": r.confidence_score,
            "confidence_level": r.confidence_level,
            "methods_matched": r.methods_matched,
            "transition_start_week": r.transition_start_week,
            "transition_end_week": r.transition_end_week,
            "old_sku_lifecycle_stage": r.old_sku_lifecycle_stage,
            "validation_status": r.validation_status,
            "validated_by": r.validated_by,
            "validation_date": r.validation_date,
            "notes": r.notes,
        }

    @staticmethod
    def _schema() -> dict:
        return {
            "mapping_id": pl.Utf8,
            "old_sku": pl.Utf8,
            "new_sku": pl.Utf8,
            "mapping_type": pl.Utf8,
            "proportion": pl.Float64,
            "confidence_score": pl.Float64,
            "confidence_level": pl.Utf8,
            "methods_matched": pl.List(pl.Utf8),
            "transition_start_week": pl.Date,
            "transition_end_week": pl.Date,
            "old_sku_lifecycle_stage": pl.Utf8,
            "validation_status": pl.Utf8,
            "validated_by": pl.Utf8,
            "validation_date": pl.Date,
            "notes": pl.Utf8,
        }
