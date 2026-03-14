"""
Schema validation for ingested DataFrames.

Validates column presence, data types, nullability, and value constraints
before data enters the pipeline.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import polars as pl

logger = logging.getLogger(__name__)

# Map config dtype names to Polars type classes
def _build_dtype_map() -> Dict[str, Any]:
    """Build dtype map, handling Polars version differences."""
    m: Dict[str, Any] = {
        "Utf8": pl.Utf8,
        "Float64": pl.Float64,
        "float": pl.Float64,
        "Float32": pl.Float32,
        "Int64": pl.Int64,
        "int": pl.Int64,
        "Int32": pl.Int32,
        "Date": pl.Date,
        "Datetime": pl.Datetime,
        "Boolean": pl.Boolean,
        "bool": pl.Boolean,
    }
    # Polars >= 1.0 renamed Utf8 → String
    if hasattr(pl, "String"):
        m["String"] = pl.String
        m["str"] = pl.String
    else:
        m["String"] = pl.Utf8
        m["str"] = pl.Utf8
    return m


_DTYPE_MAP = _build_dtype_map()


@dataclass
class ColumnSpec:
    """Specification for a single column in the expected schema."""

    name: str
    dtype: str = "Utf8"  # Polars dtype name
    required: bool = True
    nullable: bool = True
    allowed_values: Optional[List[Any]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None


@dataclass
class ValidationResult:
    """Outcome of schema validation."""

    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    @property
    def summary(self) -> str:
        status = "PASS" if self.is_valid else "FAIL"
        return (
            f"Schema validation: {status} "
            f"({len(self.errors)} errors, {len(self.warnings)} warnings)"
        )


class SchemaValidator:
    """
    Validates a DataFrame against a list of ColumnSpec definitions.

    Usage
    -----
    >>> specs = [
    ...     ColumnSpec(name="series_id", dtype="Utf8", required=True),
    ...     ColumnSpec(name="week", dtype="Date", required=True),
    ...     ColumnSpec(name="quantity", dtype="Float64", required=True, nullable=False, min_value=0),
    ... ]
    >>> validator = SchemaValidator(specs)
    >>> result = validator.validate(df)
    """

    def __init__(
        self,
        columns: List[ColumnSpec],
        allow_extra_columns: bool = True,
    ):
        self.columns = {c.name: c for c in columns}
        self.allow_extra_columns = allow_extra_columns

    def validate(self, df: pl.DataFrame) -> ValidationResult:
        result = ValidationResult()
        df_columns: Set[str] = set(df.columns)

        # Check required columns exist
        for spec in self.columns.values():
            if spec.required and spec.name not in df_columns:
                result.add_error(f"Required column '{spec.name}' is missing")

        # Check for unexpected columns
        if not self.allow_extra_columns:
            expected = set(self.columns.keys())
            extra = df_columns - expected
            if extra:
                result.add_warning(
                    f"Unexpected columns found: {sorted(extra)}"
                )

        # Per-column checks (only for columns that exist)
        for col_name in df_columns:
            if col_name not in self.columns:
                continue

            spec = self.columns[col_name]
            series = df[col_name]

            # Dtype check
            self._check_dtype(spec, series, result)

            # Nullability
            if not spec.nullable:
                null_count = series.null_count()
                if null_count > 0:
                    result.add_error(
                        f"Column '{col_name}' has {null_count} nulls "
                        f"but nullable=False"
                    )

            # Allowed values
            if spec.allowed_values is not None:
                allowed_set = set(spec.allowed_values)
                unique_vals = set(series.drop_nulls().unique().to_list())
                invalid = unique_vals - allowed_set
                if invalid:
                    examples = sorted(str(v) for v in list(invalid)[:5])
                    result.add_error(
                        f"Column '{col_name}' has invalid values: {examples}. "
                        f"Allowed: {sorted(str(v) for v in spec.allowed_values)}"
                    )

            # Numeric range checks
            if spec.min_value is not None:
                self._check_min(spec, series, result)
            if spec.max_value is not None:
                self._check_max(spec, series, result)

        logger.info(result.summary)
        return result

    def _check_dtype(
        self, spec: ColumnSpec, series: pl.Series, result: ValidationResult
    ) -> None:
        expected_type = _DTYPE_MAP.get(spec.dtype)
        if expected_type is None:
            result.add_warning(
                f"Unknown dtype '{spec.dtype}' for column '{spec.name}'; "
                f"skipping type check"
            )
            return

        # Check compatibility
        actual = series.dtype

        # Build equivalence sets for types that are the same across Polars versions
        string_types = {pl.Utf8}
        if hasattr(pl, "String"):
            string_types.add(pl.String)

        # Direct equality or both are string types
        is_match = (actual == expected_type)
        if not is_match and actual in string_types and expected_type in string_types:
            is_match = True

        if not is_match:
            # Allow numeric coercion (Int → Float is fine)
            if actual.is_numeric() and expected_type in (
                pl.Float64, pl.Float32, pl.Int64, pl.Int32,
            ):
                result.add_warning(
                    f"Column '{spec.name}' has dtype {actual} "
                    f"(expected {spec.dtype}); numeric coercion will apply"
                )
            else:
                result.add_error(
                    f"Column '{spec.name}' has dtype {actual} "
                    f"(expected {spec.dtype})"
                )

    def _check_min(
        self, spec: ColumnSpec, series: pl.Series, result: ValidationResult
    ) -> None:
        if not series.dtype.is_numeric():
            return
        actual_min = series.drop_nulls().min()
        if actual_min is not None and actual_min < spec.min_value:
            result.add_error(
                f"Column '{spec.name}' min value {actual_min} "
                f"is below minimum {spec.min_value}"
            )

    def _check_max(
        self, spec: ColumnSpec, series: pl.Series, result: ValidationResult
    ) -> None:
        if not series.dtype.is_numeric():
            return
        actual_max = series.drop_nulls().max()
        if actual_max is not None and actual_max > spec.max_value:
            result.add_error(
                f"Column '{spec.name}' max value {actual_max} "
                f"is above maximum {spec.max_value}"
            )


def build_column_specs(raw_list: List[Dict[str, Any]]) -> List[ColumnSpec]:
    """Build ColumnSpec list from config dicts."""
    specs = []
    for raw in raw_list:
        specs.append(
            ColumnSpec(
                name=raw["name"],
                dtype=raw.get("dtype", "Utf8"),
                required=raw.get("required", True),
                nullable=raw.get("nullable", True),
                allowed_values=raw.get("allowed_values"),
                min_value=raw.get("min_value"),
                max_value=raw.get("max_value"),
            )
        )
    return specs
