"""
Data quality scoring engine.

Runs configurable quality checks against ingested DataFrames and produces
a QualityReport with per-check results and an overall score.

Each check can be set to "block", "warn", or "info" severity.
The pipeline halts only when a "block"-severity check fails.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class QualityCheckConfig:
    """Configuration for a single quality check."""

    name: str
    severity: str = "warn"  # "block" | "warn" | "info"
    threshold: Optional[float] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckResult:
    """Result of a single quality check."""

    check_name: str
    passed: bool
    severity: str
    score: float  # 0-100
    detail: str

    @property
    def blocks(self) -> bool:
        return self.severity == "block" and not self.passed


@dataclass
class QualityReport:
    """Aggregated quality report across all checks."""

    overall_score: float  # 0-100, weighted average
    passed: bool  # True if no "block" checks failed
    check_results: List[CheckResult] = field(default_factory=list)

    @property
    def blocking_failures(self) -> List[CheckResult]:
        return [r for r in self.check_results if r.blocks]

    @property
    def warnings(self) -> List[CheckResult]:
        return [
            r for r in self.check_results
            if r.severity == "warn" and not r.passed
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": round(self.overall_score, 2),
            "passed": self.passed,
            "n_checks": len(self.check_results),
            "n_passed": sum(1 for r in self.check_results if r.passed),
            "n_blocked": len(self.blocking_failures),
            "n_warnings": len(self.warnings),
            "checks": [
                {
                    "name": r.check_name,
                    "passed": r.passed,
                    "severity": r.severity,
                    "score": round(r.score, 2),
                    "detail": r.detail,
                }
                for r in self.check_results
            ],
        }


class DataQualityScorer:
    """
    Runs quality checks on a DataFrame and produces a QualityReport.

    Built-in checks:
      - completeness: % non-null per column (threshold = min %)
      - uniqueness: duplicate row detection
      - freshness: most recent date within expected window
      - volume: row count within expected range
      - outlier: IQR-based outlier percentage
      - schema_drift: unexpected new or missing columns
      - value_range: values within configured min/max bounds
    """

    def __init__(self, checks: List[QualityCheckConfig]):
        self.checks = checks

    def score(
        self,
        df: pl.DataFrame,
        expected_columns: Optional[List[str]] = None,
        expected_row_count: Optional[int] = None,
        time_column: Optional[str] = None,
    ) -> QualityReport:
        """Run all configured checks and return a quality report."""
        results: List[CheckResult] = []

        for check in self.checks:
            name = check.name
            if name == "completeness":
                results.append(self._check_completeness(df, check))
            elif name == "uniqueness":
                results.append(self._check_uniqueness(df, check))
            elif name == "freshness":
                results.append(
                    self._check_freshness(df, check, time_column)
                )
            elif name == "volume":
                results.append(
                    self._check_volume(df, check, expected_row_count)
                )
            elif name == "outlier":
                results.append(self._check_outliers(df, check))
            elif name == "schema_drift":
                results.append(
                    self._check_schema_drift(df, check, expected_columns)
                )
            elif name == "value_range":
                results.append(self._check_value_range(df, check))
            else:
                logger.warning("Unknown quality check: %s (skipping)", name)

        # Compute overall score (average of individual scores)
        if results:
            overall = sum(r.score for r in results) / len(results)
        else:
            overall = 100.0

        passed = not any(r.blocks for r in results)

        report = QualityReport(
            overall_score=overall,
            passed=passed,
            check_results=results,
        )

        logger.info(
            "Quality report: score=%.1f, passed=%s, %d checks",
            overall, passed, len(results),
        )
        return report

    # ── Built-in checks ───────────────────────────────────────────────

    def _check_completeness(
        self, df: pl.DataFrame, check: QualityCheckConfig
    ) -> CheckResult:
        """Check that columns have non-null rates above threshold."""
        threshold = check.threshold if check.threshold is not None else 95.0
        n_rows = len(df)
        if n_rows == 0:
            return CheckResult(
                check_name="completeness",
                passed=False,
                severity=check.severity,
                score=0.0,
                detail="DataFrame is empty",
            )

        col_rates = {}
        for col in df.columns:
            non_null = n_rows - df[col].null_count()
            col_rates[col] = (non_null / n_rows) * 100

        min_rate = min(col_rates.values())
        worst_cols = [
            f"{c}={r:.1f}%"
            for c, r in sorted(col_rates.items(), key=lambda x: x[1])
            if r < threshold
        ]

        passed = min_rate >= threshold
        detail = (
            f"All columns ≥{threshold}% complete"
            if passed
            else f"Below threshold: {', '.join(worst_cols[:5])}"
        )

        return CheckResult(
            check_name="completeness",
            passed=passed,
            severity=check.severity,
            score=min_rate,
            detail=detail,
        )

    def _check_uniqueness(
        self, df: pl.DataFrame, check: QualityCheckConfig
    ) -> CheckResult:
        """Check for duplicate rows."""
        n_rows = len(df)
        if n_rows == 0:
            return CheckResult(
                check_name="uniqueness",
                passed=True,
                severity=check.severity,
                score=100.0,
                detail="DataFrame is empty",
            )

        n_unique = df.unique().height
        dup_count = n_rows - n_unique
        dup_pct = (dup_count / n_rows) * 100
        score = 100.0 - dup_pct

        threshold = check.threshold if check.threshold is not None else 0.0
        passed = dup_pct <= threshold

        detail = (
            f"No duplicate rows"
            if dup_count == 0
            else f"{dup_count} duplicate rows ({dup_pct:.1f}%)"
        )

        return CheckResult(
            check_name="uniqueness",
            passed=passed,
            severity=check.severity,
            score=score,
            detail=detail,
        )

    def _check_freshness(
        self,
        df: pl.DataFrame,
        check: QualityCheckConfig,
        time_column: Optional[str],
    ) -> CheckResult:
        """Check that the most recent date is within the expected window."""
        max_stale_days = check.threshold if check.threshold is not None else 7.0

        if not time_column or time_column not in df.columns:
            return CheckResult(
                check_name="freshness",
                passed=True,
                severity=check.severity,
                score=100.0,
                detail="No time column specified; skipping freshness check",
            )

        series = df[time_column]
        if series.dtype == pl.Date:
            max_date = series.drop_nulls().max()
            today = date.today()
            if max_date is None:
                return CheckResult(
                    check_name="freshness",
                    passed=False,
                    severity=check.severity,
                    score=0.0,
                    detail="Time column is all nulls",
                )
            days_stale = (today - max_date).days
        elif series.dtype == pl.Datetime:
            max_dt = series.drop_nulls().max()
            if max_dt is None:
                return CheckResult(
                    check_name="freshness",
                    passed=False,
                    severity=check.severity,
                    score=0.0,
                    detail="Time column is all nulls",
                )
            days_stale = (datetime.now() - max_dt).days
        else:
            return CheckResult(
                check_name="freshness",
                passed=True,
                severity=check.severity,
                score=100.0,
                detail=f"Column '{time_column}' is not a date type; skipping",
            )

        passed = days_stale <= max_stale_days
        score = max(0.0, 100.0 - (days_stale / max(max_stale_days, 1)) * 100)

        return CheckResult(
            check_name="freshness",
            passed=passed,
            severity=check.severity,
            score=min(score, 100.0),
            detail=f"Most recent date: {max_date if series.dtype == pl.Date else max_dt} "
                   f"({days_stale} days ago, max allowed: {max_stale_days})",
        )

    def _check_volume(
        self,
        df: pl.DataFrame,
        check: QualityCheckConfig,
        expected_row_count: Optional[int],
    ) -> CheckResult:
        """Check row count is within expected range (±threshold %)."""
        tolerance_pct = check.threshold if check.threshold is not None else 20.0
        n_rows = len(df)

        if expected_row_count is None or expected_row_count == 0:
            return CheckResult(
                check_name="volume",
                passed=True,
                severity=check.severity,
                score=100.0,
                detail=f"Row count: {n_rows} (no expected count provided)",
            )

        deviation_pct = abs(n_rows - expected_row_count) / expected_row_count * 100
        passed = deviation_pct <= tolerance_pct
        score = max(0.0, 100.0 - deviation_pct)

        return CheckResult(
            check_name="volume",
            passed=passed,
            severity=check.severity,
            score=min(score, 100.0),
            detail=f"Row count: {n_rows} (expected ~{expected_row_count}, "
                   f"deviation: {deviation_pct:.1f}%, tolerance: ±{tolerance_pct}%)",
        )

    def _check_outliers(
        self, df: pl.DataFrame, check: QualityCheckConfig
    ) -> CheckResult:
        """Check that outlier percentage is below threshold (IQR method)."""
        max_outlier_pct = check.threshold if check.threshold is not None else 5.0
        iqr_multiplier = check.params.get("iqr_multiplier", 1.5)

        numeric_cols = [
            c for c in df.columns if df[c].dtype.is_numeric()
        ]

        if not numeric_cols:
            return CheckResult(
                check_name="outlier",
                passed=True,
                severity=check.severity,
                score=100.0,
                detail="No numeric columns to check",
            )

        total_values = 0
        total_outliers = 0

        for col in numeric_cols:
            series = df[col].drop_nulls()
            if len(series) < 4:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr
            n_outliers = ((series < lower) | (series > upper)).sum()
            total_values += len(series)
            total_outliers += n_outliers

        if total_values == 0:
            outlier_pct = 0.0
        else:
            outlier_pct = (total_outliers / total_values) * 100

        passed = outlier_pct <= max_outlier_pct
        score = max(0.0, 100.0 - outlier_pct * (100 / max(max_outlier_pct, 1)))

        return CheckResult(
            check_name="outlier",
            passed=passed,
            severity=check.severity,
            score=min(score, 100.0),
            detail=f"{total_outliers} outliers across {len(numeric_cols)} numeric columns "
                   f"({outlier_pct:.1f}%, max allowed: {max_outlier_pct}%)",
        )

    def _check_schema_drift(
        self,
        df: pl.DataFrame,
        check: QualityCheckConfig,
        expected_columns: Optional[List[str]],
    ) -> CheckResult:
        """Check for new or missing columns vs expected schema."""
        if expected_columns is None:
            return CheckResult(
                check_name="schema_drift",
                passed=True,
                severity=check.severity,
                score=100.0,
                detail="No expected columns specified; skipping drift check",
            )

        expected_set = set(expected_columns)
        actual_set = set(df.columns)

        missing = expected_set - actual_set
        extra = actual_set - expected_set

        passed = len(missing) == 0
        issues = []
        if missing:
            issues.append(f"missing: {sorted(missing)}")
        if extra:
            issues.append(f"new: {sorted(extra)}")

        score = (
            (len(expected_set & actual_set) / len(expected_set)) * 100
            if expected_set
            else 100.0
        )

        return CheckResult(
            check_name="schema_drift",
            passed=passed,
            severity=check.severity,
            score=score,
            detail="; ".join(issues) if issues else "Schema matches expected columns",
        )

    def _check_value_range(
        self, df: pl.DataFrame, check: QualityCheckConfig
    ) -> CheckResult:
        """Check that configured columns are within min/max bounds."""
        column = check.params.get("column")
        min_val = check.params.get("min_value")
        max_val = check.params.get("max_value")

        if not column or column not in df.columns:
            return CheckResult(
                check_name="value_range",
                passed=True,
                severity=check.severity,
                score=100.0,
                detail=f"Column '{column}' not found; skipping range check",
            )

        series = df[column].drop_nulls()
        violations = 0
        detail_parts = []

        if min_val is not None and series.dtype.is_numeric():
            below = (series < min_val).sum()
            if below > 0:
                violations += below
                detail_parts.append(f"{below} values below {min_val}")

        if max_val is not None and series.dtype.is_numeric():
            above = (series > max_val).sum()
            if above > 0:
                violations += above
                detail_parts.append(f"{above} values above {max_val}")

        passed = violations == 0
        score = (
            max(0.0, (1 - violations / max(len(series), 1)) * 100)
            if len(series) > 0
            else 100.0
        )

        return CheckResult(
            check_name="value_range",
            passed=passed,
            severity=check.severity,
            score=score,
            detail=f"Column '{column}': "
                   + ("; ".join(detail_parts) if detail_parts else "within range"),
        )


def build_quality_checks(
    raw_list: List[Dict[str, Any]],
) -> List[QualityCheckConfig]:
    """Build QualityCheckConfig list from config dicts."""
    return [
        QualityCheckConfig(
            name=raw["name"],
            severity=raw.get("severity", "warn"),
            threshold=raw.get("threshold"),
            params=raw.get("params", {}),
        )
        for raw in raw_list
    ]
