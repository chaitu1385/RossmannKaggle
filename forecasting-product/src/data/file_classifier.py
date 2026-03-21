"""
FileClassifier — automatic role classification for multi-file uploads.

Given N uploaded DataFrames, classifies each into one of:
- ``time_series``: primary demand/sales data (date + target + IDs)
- ``dimension``: lookup/attribute table (IDs + categorical attributes, no date)
- ``regressor``: external feature table (date + numeric features, joinable)
- ``unknown``: does not match any expected pattern

Classification is a two-pass process:
1. **Isolation pass** — score each file independently for ``time_series`` signals.
2. **Resolution pass** — pick the best primary, then re-evaluate remaining files
   as ``dimension`` or ``regressor`` relative to the primary.

Usage
-----
>>> from src.data.file_classifier import FileClassifier
>>> classifier = FileClassifier()
>>> result = classifier.classify_files({"sales.csv": df_sales, "stores.csv": df_stores})
>>> print(result.primary_file.filename)
'sales.csv'
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import polars as pl

# Reuse heuristic constants from DataAnalyzer
_TIME_COLUMN_PATTERNS = {"week", "date", "ds", "time", "timestamp", "period", "day"}
_TARGET_COLUMN_PATTERNS = {
    "quantity", "sales", "demand", "revenue", "volume", "units",
    "target", "value", "amount", "qty", "count",
}

_NUMERIC_TYPES = (
    pl.Float32, pl.Float64,
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
)


# --------------------------------------------------------------------------- #
#  Result dataclasses
# --------------------------------------------------------------------------- #

@dataclass
class FileProfile:
    """Profile of a single uploaded file."""

    filename: str
    df: pl.DataFrame
    role: str  # "time_series" | "dimension" | "regressor" | "unknown"
    confidence: float  # 0.0 – 1.0
    time_column: Optional[str]  # None for dimension / unknown files
    id_columns: List[str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    n_rows: int
    n_columns: int
    reasoning: List[str] = field(default_factory=list)


@dataclass
class ClassificationResult:
    """Result of classifying all uploaded files."""

    profiles: List[FileProfile]
    primary_file: Optional[FileProfile]
    dimension_files: List[FileProfile]
    regressor_files: List[FileProfile]
    unknown_files: List[FileProfile]
    warnings: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
#  FileClassifier
# --------------------------------------------------------------------------- #

class FileClassifier:
    """Classify uploaded DataFrames into pipeline roles.

    Parameters
    ----------
    id_cardinality_cap : int
        Maximum unique values for a column to be treated as an ID (not free text).
    """

    def __init__(self, id_cardinality_cap: int = 10_000):
        self.id_cardinality_cap = id_cardinality_cap

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def classify_files(
        self,
        files: Dict[str, pl.DataFrame],
    ) -> ClassificationResult:
        """Classify all files and resolve roles.

        Parameters
        ----------
        files : Dict[str, pl.DataFrame]
            Mapping of ``filename -> DataFrame``.

        Returns
        -------
        ClassificationResult
        """
        if not files:
            return ClassificationResult(
                profiles=[], primary_file=None,
                dimension_files=[], regressor_files=[],
                unknown_files=[], warnings=["No files provided"],
            )

        # Pass 1: classify each file in isolation (looking for time_series signals)
        profiles = [self.classify_single(name, df) for name, df in files.items()]

        # Pass 2: cross-file resolution
        return self._resolve_roles(profiles)

    def classify_single(
        self,
        filename: str,
        df: pl.DataFrame,
    ) -> FileProfile:
        """Classify a single file in isolation.

        In isolation we can only detect ``time_series`` signals. Dimension
        and regressor roles require a primary to compare against, so those
        are assigned in :meth:`_resolve_roles`.
        """
        reasoning: List[str] = []
        dtypes = {c: df[c].dtype for c in df.columns}

        time_col = self._find_time_column(df, dtypes)
        target_col = self._find_target_column(df, dtypes, time_col)
        id_cols = self._find_id_columns(df, dtypes, time_col, target_col)
        numeric_cols = self._find_numeric_columns(df, dtypes, time_col, target_col)
        cat_cols = self._find_categorical_columns(df, dtypes, time_col, target_col)

        # Score for time_series role
        ts_score = self._score_time_series(
            df, time_col, target_col, id_cols, dtypes, reasoning,
        )

        role = "time_series" if ts_score >= 0.4 else "unknown"

        return FileProfile(
            filename=filename,
            df=df,
            role=role,
            confidence=round(min(ts_score, 1.0), 3),
            time_column=time_col,
            id_columns=id_cols,
            numeric_columns=numeric_cols,
            categorical_columns=cat_cols,
            n_rows=df.height,
            n_columns=len(df.columns),
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------ #
    #  Column detection helpers (mirroring DataAnalyzer)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _find_time_column(
        df: pl.DataFrame,
        dtypes: Dict[str, pl.DataType],
    ) -> Optional[str]:
        """Find a date/time column, or ``None`` if absent."""
        cols = df.columns
        # 1. Actual Date/Datetime types
        for c in cols:
            if dtypes[c] in (pl.Date, pl.Datetime):
                return c
        # 2. Name matching
        for c in cols:
            if c.lower() in _TIME_COLUMN_PATTERNS:
                return c
        # 3. Try parsing string columns
        for c in cols:
            if dtypes[c] in (pl.Utf8, pl.String):
                try:
                    parsed = df[c].str.to_date(strict=False)
                    if parsed.null_count() < df.height * 0.5:
                        return c
                except Exception:
                    continue
        return None

    @staticmethod
    def _find_target_column(
        df: pl.DataFrame,
        dtypes: Dict[str, pl.DataType],
        time_col: Optional[str],
    ) -> Optional[str]:
        """Find the most likely target column, or ``None``."""
        numeric_cols = [
            c for c in df.columns
            if c != time_col and dtypes[c] in _NUMERIC_TYPES
        ]
        if not numeric_cols:
            return None
        for c in numeric_cols:
            if c.lower() in _TARGET_COLUMN_PATTERNS:
                return c
        return max(numeric_cols, key=lambda c: df.height - df[c].null_count())

    def _find_id_columns(
        self,
        df: pl.DataFrame,
        dtypes: Dict[str, pl.DataType],
        time_col: Optional[str],
        target_col: Optional[str],
    ) -> List[str]:
        """Find categorical columns with moderate cardinality (likely IDs)."""
        result = []
        for c in df.columns:
            if c in (time_col, target_col):
                continue
            if dtypes[c] in (pl.Utf8, pl.Categorical, pl.String):
                n_unique = df[c].n_unique()
                if 1 < n_unique <= self.id_cardinality_cap:
                    result.append(c)
        return result

    @staticmethod
    def _find_numeric_columns(
        df: pl.DataFrame,
        dtypes: Dict[str, pl.DataType],
        time_col: Optional[str],
        target_col: Optional[str],
    ) -> List[str]:
        """Return numeric columns excluding time and target."""
        return [
            c for c in df.columns
            if c not in (time_col, target_col) and dtypes[c] in _NUMERIC_TYPES
        ]

    @staticmethod
    def _find_categorical_columns(
        df: pl.DataFrame,
        dtypes: Dict[str, pl.DataType],
        time_col: Optional[str],
        target_col: Optional[str],
    ) -> List[str]:
        """Return string/categorical columns excluding time and target."""
        return [
            c for c in df.columns
            if c not in (time_col, target_col)
            and dtypes[c] in (pl.Utf8, pl.Categorical, pl.String)
        ]

    # ------------------------------------------------------------------ #
    #  Scoring helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _score_time_series(
        df: pl.DataFrame,
        time_col: Optional[str],
        target_col: Optional[str],
        id_cols: List[str],
        dtypes: Dict[str, pl.DataType],
        reasoning: List[str],
    ) -> float:
        """Confidence score for the ``time_series`` role."""
        score = 0.0

        # +0.30  Date/Datetime column present
        if time_col is not None:
            if dtypes[time_col] in (pl.Date, pl.Datetime):
                score += 0.30
                reasoning.append(f"Date-typed column '{time_col}' (+0.30)")
            else:
                score += 0.15
                reasoning.append(f"Parseable date column '{time_col}' (+0.15)")
        else:
            reasoning.append("No date column detected (+0.00)")

        # +0.20  Target column name matches known patterns
        if target_col is not None and target_col.lower() in _TARGET_COLUMN_PATTERNS:
            score += 0.20
            reasoning.append(f"Target column '{target_col}' matches known pattern (+0.20)")
        elif target_col is not None:
            score += 0.05
            reasoning.append(f"Numeric column '{target_col}' found, no name match (+0.05)")

        # +0.20  ID columns with repeating values
        if id_cols and df.height > 0:
            max_cardinality = max(df[c].n_unique() for c in id_cols)
            if max_cardinality < df.height / 5:
                score += 0.20
                reasoning.append(f"ID columns have repeating values (max cardinality {max_cardinality}) (+0.20)")
            elif max_cardinality < df.height:
                score += 0.10
                reasoning.append(f"ID columns have some repetition (+0.10)")
        elif not id_cols:
            reasoning.append("No ID columns detected (+0.00)")

        # +0.15  Substantial row count
        if df.height > 50:
            score += 0.15
            reasoning.append(f"Substantial data ({df.height} rows) (+0.15)")
        elif df.height > 10:
            score += 0.05
            reasoning.append(f"Small data ({df.height} rows) (+0.05)")

        # +0.15  Time column name matches common patterns
        if time_col is not None and time_col.lower() in _TIME_COLUMN_PATTERNS:
            score += 0.15
            reasoning.append(f"Time column name '{time_col}' matches pattern (+0.15)")

        return score

    def _score_dimension(
        self,
        profile: FileProfile,
        primary: FileProfile,
        reasoning: List[str],
    ) -> float:
        """Confidence score for the ``dimension`` role."""
        score = 0.0

        # +0.30  No time column
        if profile.time_column is None:
            score += 0.30
            reasoning.append("No time column detected (+0.30)")
        else:
            reasoning.append("Has a time column — less likely to be dimension (+0.00)")

        # +0.30  ID overlap with primary
        overlap = set(profile.id_columns) & set(primary.id_columns)
        if overlap:
            score += 0.30
            reasoning.append(f"ID columns overlap with primary: {overlap} (+0.30)")
        else:
            # Check column name overlap (might use different dtype)
            name_overlap = set(c.lower() for c in profile.df.columns) & set(
                c.lower() for c in primary.id_columns
            )
            if name_overlap:
                score += 0.15
                reasoning.append(f"Column name overlap with primary IDs: {name_overlap} (+0.15)")

        # +0.20  Mostly categorical columns
        total_non_id = len(profile.df.columns) - (1 if profile.time_column else 0)
        if total_non_id > 0:
            cat_ratio = len(profile.categorical_columns) / total_non_id
            if cat_ratio > 0.5:
                score += 0.20
                reasoning.append(f"Mostly categorical ({cat_ratio:.0%}) (+0.20)")
            elif cat_ratio > 0.3:
                score += 0.10
                reasoning.append(f"Partially categorical ({cat_ratio:.0%}) (+0.10)")

        # +0.20  Cardinality matches primary ID count
        if overlap and primary.id_columns:
            for col in overlap:
                if col in profile.df.columns and col in primary.df.columns:
                    dim_unique = profile.df[col].n_unique()
                    primary_unique = primary.df[col].n_unique()
                    ratio = dim_unique / max(primary_unique, 1)
                    if 0.5 <= ratio <= 2.0:
                        score += 0.20
                        reasoning.append(
                            f"Cardinality of '{col}' roughly matches primary "
                            f"({dim_unique} vs {primary_unique}) (+0.20)"
                        )
                        break

        return score

    def _score_regressor(
        self,
        profile: FileProfile,
        primary: FileProfile,
        reasoning: List[str],
    ) -> float:
        """Confidence score for the ``regressor`` role."""
        score = 0.0

        # +0.30  Has a time column
        if profile.time_column is not None:
            score += 0.30
            reasoning.append(f"Has time column '{profile.time_column}' (+0.30)")

        # +0.30  Has numeric features (beyond time)
        if profile.numeric_columns:
            score += 0.30
            reasoning.append(f"Has {len(profile.numeric_columns)} numeric columns (+0.30)")

        # +0.20  Time range overlaps with primary
        if profile.time_column and primary.time_column:
            try:
                p_min = primary.df[primary.time_column].min()
                p_max = primary.df[primary.time_column].max()
                s_min = profile.df[profile.time_column].min()
                s_max = profile.df[profile.time_column].max()
                if s_min <= p_max and s_max >= p_min:
                    score += 0.20
                    reasoning.append("Time range overlaps with primary (+0.20)")
                else:
                    reasoning.append("Time range does NOT overlap with primary (+0.00)")
            except Exception:
                reasoning.append("Could not compare time ranges (+0.00)")

        # +0.20  ID column overlap with primary, or broadcastable (no IDs)
        id_overlap = set(profile.id_columns) & set(primary.id_columns)
        if id_overlap:
            score += 0.20
            reasoning.append(f"ID columns overlap with primary: {id_overlap} (+0.20)")
        elif not profile.id_columns:
            # Broadcastable: global features with just a time column
            score += 0.10
            reasoning.append("No ID columns — broadcastable feature (+0.10)")

        return score

    # ------------------------------------------------------------------ #
    #  Cross-file resolution
    # ------------------------------------------------------------------ #

    def _resolve_roles(
        self,
        profiles: List[FileProfile],
    ) -> ClassificationResult:
        """Pick the best primary and re-evaluate remaining files."""
        warnings: List[str] = []

        # If only one file, just return it as-is
        if len(profiles) == 1:
            p = profiles[0]
            primary = p if p.role == "time_series" else None
            if primary is None:
                warnings.append(
                    f"'{p.filename}' was not confidently classified as time series "
                    f"(confidence={p.confidence:.2f}). Expected a file with a date "
                    f"column, a numeric target (e.g. 'quantity', 'sales'), and ID columns."
                )
            return ClassificationResult(
                profiles=profiles,
                primary_file=primary,
                dimension_files=[],
                regressor_files=[],
                unknown_files=[p] if primary is None else [],
                warnings=warnings,
            )

        # Find the best time_series candidate
        ts_candidates = [p for p in profiles if p.role == "time_series"]

        if not ts_candidates:
            warnings.append(
                "No file was classified as a time series. At least one file must "
                "contain a date column, a numeric target, and identifier columns."
            )
            return ClassificationResult(
                profiles=profiles,
                primary_file=None,
                dimension_files=[],
                regressor_files=[],
                unknown_files=profiles,
                warnings=warnings,
            )

        # Pick highest confidence
        primary = max(ts_candidates, key=lambda p: p.confidence)

        # Re-evaluate remaining files relative to primary
        dimension_files: List[FileProfile] = []
        regressor_files: List[FileProfile] = []
        unknown_files: List[FileProfile] = []

        for profile in profiles:
            if profile is primary:
                continue

            reasoning: List[str] = []
            dim_score = self._score_dimension(profile, primary, reasoning)
            reg_score = self._score_regressor(profile, primary, reasoning)

            if dim_score > reg_score and dim_score >= 0.3:
                profile.role = "dimension"
                profile.confidence = round(min(dim_score, 1.0), 3)
                profile.reasoning = reasoning
                dimension_files.append(profile)
            elif reg_score >= 0.3:
                profile.role = "regressor"
                profile.confidence = round(min(reg_score, 1.0), 3)
                profile.reasoning = reasoning
                regressor_files.append(profile)
            else:
                profile.role = "unknown"
                profile.confidence = round(max(dim_score, reg_score), 3)
                profile.reasoning = reasoning
                unknown_files.append(profile)
                warnings.append(
                    f"'{profile.filename}' could not be confidently classified "
                    f"(dim={dim_score:.2f}, reg={reg_score:.2f})."
                )

        return ClassificationResult(
            profiles=profiles,
            primary_file=primary,
            dimension_files=dimension_files,
            regressor_files=regressor_files,
            unknown_files=unknown_files,
            warnings=warnings,
        )
