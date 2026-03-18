"""
MultiFileMerger — join key detection and multi-file merge for forecasting.

Takes a :class:`ClassificationResult` from :class:`FileClassifier` and produces
a single merged DataFrame suitable for :class:`DataAnalyzer` and the forecast
pipeline.

Merge order:
1. Start with the primary time-series DataFrame.
2. Left-join each dimension table on shared ID columns.
3. Left-join each regressor table on ``[time_col]`` or ``[time_col, id_col]``.
4. Resolve duplicate column names with a ``_<filename_stem>`` suffix.
5. Fill nulls in regressor columns with ``0``.

Usage
-----
>>> from src.data.file_classifier import FileClassifier
>>> from src.data.file_merger import MultiFileMerger
>>> result = FileClassifier().classify_files(files)
>>> merger = MultiFileMerger()
>>> merge_result = merger.merge(result)
>>> merged_df = merge_result.df
"""

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import List, Optional

import polars as pl

from .file_classifier import ClassificationResult, FileProfile


# --------------------------------------------------------------------------- #
#  Result dataclasses
# --------------------------------------------------------------------------- #

@dataclass
class JoinSpec:
    """Specification for joining two files."""

    left_file: str
    right_file: str
    join_keys: List[str]
    join_type: str = "left"
    key_overlap_pct: float = 0.0
    warnings: List[str] = field(default_factory=list)


@dataclass
class MergePreview:
    """Preview of the merge result before committing."""

    sample_rows: pl.DataFrame
    total_rows: int
    total_columns: int
    matched_rows: int
    unmatched_primary_keys: int
    null_fill_columns: List[str]
    column_name_conflicts: List[str]
    warnings: List[str] = field(default_factory=list)
    join_specs: List[JoinSpec] = field(default_factory=list)


@dataclass
class MergeResult:
    """Final merged DataFrame with metadata."""

    df: pl.DataFrame
    preview: MergePreview
    join_specs: List[JoinSpec]


# --------------------------------------------------------------------------- #
#  MultiFileMerger
# --------------------------------------------------------------------------- #

class MultiFileMerger:
    """Detect join keys and merge classified files into a single DataFrame."""

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def detect_join_keys(
        self,
        primary: FileProfile,
        secondary: FileProfile,
    ) -> JoinSpec:
        """Find the best join keys between *primary* and *secondary*.

        Parameters
        ----------
        primary, secondary : FileProfile
            Classified file profiles.

        Returns
        -------
        JoinSpec
        """
        warnings: List[str] = []

        if secondary.role == "dimension":
            keys = self._dimension_join_keys(primary, secondary)
        else:
            keys = self._regressor_join_keys(primary, secondary)

        if not keys:
            warnings.append(
                f"No overlapping join keys found between '{primary.filename}' "
                f"and '{secondary.filename}'."
            )
            return JoinSpec(
                left_file=primary.filename,
                right_file=secondary.filename,
                join_keys=[],
                key_overlap_pct=0.0,
                warnings=warnings,
            )

        # Compute overlap %
        overlap_pct = self._compute_key_overlap(primary.df, secondary.df, keys)
        if overlap_pct < 0.5:
            warnings.append(
                f"Low key overlap ({overlap_pct:.0%}) between "
                f"'{primary.filename}' and '{secondary.filename}' on {keys}."
            )

        return JoinSpec(
            left_file=primary.filename,
            right_file=secondary.filename,
            join_keys=keys,
            key_overlap_pct=round(overlap_pct, 3),
            warnings=warnings,
        )

    def preview_merge(
        self,
        classification: ClassificationResult,
    ) -> MergePreview:
        """Generate a merge preview without persisting state.

        Parameters
        ----------
        classification : ClassificationResult

        Returns
        -------
        MergePreview
        """
        primary = classification.primary_file
        if primary is None:
            return MergePreview(
                sample_rows=pl.DataFrame(),
                total_rows=0,
                total_columns=0,
                matched_rows=0,
                unmatched_primary_keys=0,
                null_fill_columns=[],
                column_name_conflicts=[],
                warnings=["No primary time-series file identified."],
            )

        secondaries = classification.dimension_files + classification.regressor_files
        if not secondaries:
            df = primary.df
            return MergePreview(
                sample_rows=df.head(10),
                total_rows=df.height,
                total_columns=len(df.columns),
                matched_rows=df.height,
                unmatched_primary_keys=0,
                null_fill_columns=[],
                column_name_conflicts=[],
            )

        # Perform the merge
        result = self._execute_merge(primary, secondaries)
        return result.preview

    def merge(
        self,
        classification: ClassificationResult,
    ) -> MergeResult:
        """Execute the full merge and return the result.

        Parameters
        ----------
        classification : ClassificationResult

        Returns
        -------
        MergeResult
        """
        primary = classification.primary_file
        if primary is None:
            empty_preview = MergePreview(
                sample_rows=pl.DataFrame(),
                total_rows=0,
                total_columns=0,
                matched_rows=0,
                unmatched_primary_keys=0,
                null_fill_columns=[],
                column_name_conflicts=[],
                warnings=["No primary time-series file identified."],
            )
            return MergeResult(df=pl.DataFrame(), preview=empty_preview, join_specs=[])

        secondaries = classification.dimension_files + classification.regressor_files
        if not secondaries:
            preview = MergePreview(
                sample_rows=primary.df.head(10),
                total_rows=primary.df.height,
                total_columns=len(primary.df.columns),
                matched_rows=primary.df.height,
                unmatched_primary_keys=0,
                null_fill_columns=[],
                column_name_conflicts=[],
            )
            return MergeResult(df=primary.df, preview=preview, join_specs=[])

        return self._execute_merge(primary, secondaries)

    # ------------------------------------------------------------------ #
    #  Join key detection
    # ------------------------------------------------------------------ #

    @staticmethod
    def _dimension_join_keys(
        primary: FileProfile,
        secondary: FileProfile,
    ) -> List[str]:
        """Join keys for a dimension table: shared ID columns (no time)."""
        shared = [
            c for c in secondary.df.columns
            if c in primary.id_columns
        ]
        return shared

    @staticmethod
    def _regressor_join_keys(
        primary: FileProfile,
        secondary: FileProfile,
    ) -> List[str]:
        """Join keys for a regressor table: time column + optional shared IDs."""
        keys: List[str] = []

        # Time column (must match by name or both be present)
        if (
            secondary.time_column
            and primary.time_column
            and secondary.time_column == primary.time_column
        ):
            keys.append(primary.time_column)
        elif secondary.time_column and primary.time_column:
            # Different names — still try if both are date columns
            keys.append(primary.time_column)

        # Shared ID columns
        shared_ids = [
            c for c in secondary.id_columns
            if c in primary.id_columns
        ]
        keys.extend(shared_ids)

        return keys

    @staticmethod
    def _compute_key_overlap(
        left: pl.DataFrame,
        right: pl.DataFrame,
        keys: List[str],
    ) -> float:
        """Fraction of left key values that exist in right."""
        # Only use columns present in both
        usable_keys = [k for k in keys if k in left.columns and k in right.columns]
        if not usable_keys:
            return 0.0

        left_keys = left.select(usable_keys).unique()
        right_keys = right.select(usable_keys).unique()

        if left_keys.height == 0:
            return 0.0

        joined = left_keys.join(right_keys, on=usable_keys, how="inner")
        return joined.height / left_keys.height

    # ------------------------------------------------------------------ #
    #  Merge execution
    # ------------------------------------------------------------------ #

    def _execute_merge(
        self,
        primary: FileProfile,
        secondaries: List[FileProfile],
    ) -> MergeResult:
        """Perform left joins and build merge metadata."""
        merged = primary.df.clone()
        join_specs: List[JoinSpec] = []
        all_conflicts: List[str] = []
        null_fill_cols: List[str] = []
        all_warnings: List[str] = []

        original_height = merged.height

        for sec in secondaries:
            spec = self.detect_join_keys(primary, sec)
            join_specs.append(spec)
            all_warnings.extend(spec.warnings)

            if not spec.join_keys:
                all_warnings.append(
                    f"Skipping '{sec.filename}' — no join keys detected."
                )
                continue

            # Prepare right DataFrame
            right = sec.df

            # Handle regressor with different time column name
            if (
                sec.role == "regressor"
                and sec.time_column
                and primary.time_column
                and sec.time_column != primary.time_column
                and primary.time_column in spec.join_keys
            ):
                right = right.rename({sec.time_column: primary.time_column})

            # Resolve column conflicts before joining
            conflicts, right = self._resolve_column_conflicts(
                merged, right, sec.filename, spec.join_keys,
            )
            all_conflicts.extend(conflicts)

            # Only use join keys that exist in both DataFrames
            usable_keys = [
                k for k in spec.join_keys
                if k in merged.columns and k in right.columns
            ]
            if not usable_keys:
                all_warnings.append(
                    f"Skipping '{sec.filename}' — join keys not in both DataFrames."
                )
                continue

            # Left join
            merged = merged.join(right, on=usable_keys, how="left", suffix="_dup")

            # Drop any _dup columns that snuck through
            dup_cols = [c for c in merged.columns if c.endswith("_dup")]
            if dup_cols:
                merged = merged.drop(dup_cols)

            # Fill nulls in regressor numeric columns with 0
            if sec.role == "regressor":
                new_cols = [
                    c for c in right.columns
                    if c not in usable_keys and c in merged.columns
                ]
                for c in new_cols:
                    if merged[c].dtype in (
                        pl.Float32, pl.Float64,
                        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                    ):
                        merged = merged.with_columns(pl.col(c).fill_null(0))
                        null_fill_cols.append(c)

        # Compute stats
        # Count rows where at least one joined column has a non-null value
        matched_rows = merged.height  # left join preserves all primary rows
        # Unmatched = rows where ALL joined columns are null
        joined_cols = [
            c for c in merged.columns if c not in primary.df.columns
        ]
        if joined_cols:
            all_null_mask = pl.all_horizontal(
                pl.col(c).is_null() for c in joined_cols
            )
            unmatched = merged.filter(all_null_mask).height
        else:
            unmatched = 0

        preview = MergePreview(
            sample_rows=merged.head(10),
            total_rows=merged.height,
            total_columns=len(merged.columns),
            matched_rows=merged.height - unmatched,
            unmatched_primary_keys=unmatched,
            null_fill_columns=null_fill_cols,
            column_name_conflicts=all_conflicts,
            warnings=all_warnings,
            join_specs=join_specs,
        )

        return MergeResult(df=merged, preview=preview, join_specs=join_specs)

    @staticmethod
    def _resolve_column_conflicts(
        left: pl.DataFrame,
        right: pl.DataFrame,
        right_filename: str,
        join_keys: List[str],
    ) -> tuple:
        """Rename conflicting non-key columns in *right*.

        Returns
        -------
        tuple of (conflicts_list, renamed_right_df)
        """
        stem = PurePosixPath(right_filename).stem
        conflicts: List[str] = []
        renames: dict = {}

        for c in right.columns:
            if c in join_keys:
                continue
            if c in left.columns:
                new_name = f"{c}_{stem}"
                renames[c] = new_name
                conflicts.append(f"'{c}' renamed to '{new_name}' (from {right_filename})")

        if renames:
            right = right.rename(renames)

        return conflicts, right
