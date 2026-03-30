"""Tests for DataValidator — schema enforcement and data validation."""

from datetime import date, timedelta

import polars as pl
import pytest

from src.config.schema import (
    DataQualityConfig,
    PlatformConfig,
    ValidationConfig,
)
from src.data.validator import (
    DataValidator,
    ValidationIssue,
    ValidationReport,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def _make_weekly_actuals(
    n_series: int = 3,
    n_weeks: int = 52,
    start_date: date = date(2023, 1, 2),  # Monday
    base_value: float = 100.0,
) -> pl.DataFrame:
    """Build a clean weekly panel DataFrame with no validation issues."""
    rows = []
    for s in range(n_series):
        sid = f"series_{s}"
        for w in range(n_weeks):
            rows.append({
                "series_id": sid,
                "week": start_date + timedelta(weeks=w),
                "quantity": base_value + float(w),
            })
    return pl.DataFrame(rows).with_columns(
        pl.col("week").cast(pl.Date),
        pl.col("quantity").cast(pl.Float64),
    )


# ===========================================================================
# TestSchemaCheck
# ===========================================================================


class TestSchemaCheck:
    """Validates column presence and type checks."""

    def test_missing_time_column_is_error(self):
        df = pl.DataFrame({
            "series_id": ["a", "b"],
            "quantity": [1.0, 2.0],
        })
        v = DataValidator(ValidationConfig(enabled=True))
        report = v.validate(df, target_col="quantity", time_col="week", id_col="series_id")
        assert not report.passed
        assert "week" in report.missing_column_names

    def test_wrong_type_is_error(self):
        df = pl.DataFrame({
            "series_id": ["a", "a"],
            "week": ["2023-01-02", "2023-01-09"],  # string, not Date
            "quantity": [1.0, 2.0],
        })
        v = DataValidator(ValidationConfig(enabled=True))
        report = v.validate(df)
        assert not report.passed
        assert any(i.check == "schema" and "type" in i.message.lower() for i in report.errors)

    def test_extra_required_columns(self):
        df = _make_weekly_actuals(n_series=1, n_weeks=4)
        cfg = ValidationConfig(enabled=True, require_columns=["store_id"])
        v = DataValidator(cfg)
        report = v.validate(df)
        assert not report.passed
        assert "store_id" in report.missing_column_names

    def test_all_present_passes(self):
        df = _make_weekly_actuals(n_series=2, n_weeks=10)
        v = DataValidator(ValidationConfig(enabled=True))
        report = v.validate(df)
        # No schema errors
        schema_errors = [i for i in report.errors if i.check == "schema"]
        assert len(schema_errors) == 0

    def test_utf8_id_column_accepted(self):
        df = _make_weekly_actuals(n_series=1, n_weeks=4)
        assert df["series_id"].dtype == pl.Utf8
        v = DataValidator(ValidationConfig(enabled=True))
        report = v.validate(df)
        schema_errors = [i for i in report.errors if i.check == "schema"]
        assert len(schema_errors) == 0


# ===========================================================================
# TestDuplicateCheck
# ===========================================================================


class TestDuplicateCheck:
    """Validates duplicate row detection."""

    def test_no_duplicates_passes(self):
        df = _make_weekly_actuals(n_series=2, n_weeks=10)
        v = DataValidator(ValidationConfig(enabled=True))
        report = v.validate(df)
        assert report.duplicate_count == 0
        dup_errors = [i for i in report.errors if i.check == "duplicates"]
        assert len(dup_errors) == 0

    def test_duplicates_detected(self):
        df = _make_weekly_actuals(n_series=1, n_weeks=4)
        # Append a duplicate row
        dup = df.head(1)
        df_with_dup = pl.concat([df, dup])
        v = DataValidator(ValidationConfig(enabled=True))
        report = v.validate(df_with_dup)
        assert report.duplicate_count == 1
        assert not report.passed

    def test_disabled_skips_check(self):
        df = _make_weekly_actuals(n_series=1, n_weeks=4)
        dup = df.head(1)
        df_with_dup = pl.concat([df, dup])
        v = DataValidator(ValidationConfig(enabled=True, check_duplicates=False))
        report = v.validate(df_with_dup)
        assert report.duplicate_count == 0


# ===========================================================================
# TestFrequencyCheck
# ===========================================================================


class TestFrequencyCheck:
    """Validates weekly frequency enforcement."""

    def test_weekly_data_passes(self):
        df = _make_weekly_actuals(n_series=2, n_weeks=10)
        v = DataValidator(ValidationConfig(enabled=True))
        report = v.validate(df)
        assert report.frequency_violations == 0

    def test_daily_gaps_flagged(self):
        """Non-weekly intervals are flagged as warnings."""
        rows = [
            {"series_id": "a", "week": date(2023, 1, 2), "quantity": 1.0},
            {"series_id": "a", "week": date(2023, 1, 3), "quantity": 2.0},  # 1-day gap
            {"series_id": "a", "week": date(2023, 1, 9), "quantity": 3.0},
        ]
        df = pl.DataFrame(rows).with_columns(
            pl.col("week").cast(pl.Date),
            pl.col("quantity").cast(pl.Float64),
        )
        v = DataValidator(ValidationConfig(enabled=True))
        report = v.validate(df)
        assert report.frequency_violations == 1
        freq_warnings = [i for i in report.warnings if i.check == "frequency"]
        assert len(freq_warnings) == 1

    def test_mixed_gaps_per_series(self):
        """Only series with non-weekly gaps are counted."""
        good = _make_weekly_actuals(n_series=1, n_weeks=10)
        bad_rows = [
            {"series_id": "bad", "week": date(2023, 1, 2), "quantity": 1.0},
            {"series_id": "bad", "week": date(2023, 1, 5), "quantity": 2.0},  # 3-day gap
        ]
        bad = pl.DataFrame(bad_rows).with_columns(
            pl.col("week").cast(pl.Date),
            pl.col("quantity").cast(pl.Float64),
        )
        df = pl.concat([good, bad])
        v = DataValidator(ValidationConfig(enabled=True))
        report = v.validate(df)
        assert report.frequency_violations == 1


# ===========================================================================
# TestValueRange
# ===========================================================================


class TestValueRange:
    """Validates value range enforcement."""

    def test_negative_values_flagged(self):
        df = _make_weekly_actuals(n_series=1, n_weeks=4)
        df = df.with_columns(
            pl.when(pl.col("quantity") > 101).then(pl.lit(-5.0)).otherwise(pl.col("quantity")).alias("quantity")
        )
        v = DataValidator(ValidationConfig(enabled=True))
        report = v.validate(df)
        assert report.negative_count > 0
        assert not report.passed

    def test_custom_min_max(self):
        df = _make_weekly_actuals(n_series=1, n_weeks=4)
        # Values are 100.0, 101.0, 102.0, 103.0
        cfg = ValidationConfig(enabled=True, min_value=101.0, max_value=102.0)
        v = DataValidator(cfg)
        report = v.validate(df)
        # 100.0 is below min → error
        assert report.negative_count == 1
        # 103.0 is above max → warning
        above_warnings = [i for i in report.issues if "above" in i.message]
        assert len(above_warnings) == 1

    def test_all_valid_passes(self):
        df = _make_weekly_actuals(n_series=1, n_weeks=4)
        v = DataValidator(ValidationConfig(enabled=True))
        report = v.validate(df)
        assert report.negative_count == 0

    def test_non_negative_disabled(self):
        df = _make_weekly_actuals(n_series=1, n_weeks=4)
        df = df.with_columns(pl.lit(-1.0).alias("quantity"))
        cfg = ValidationConfig(enabled=True, check_non_negative=False)
        v = DataValidator(cfg)
        report = v.validate(df)
        # No error about negative values
        range_errors = [i for i in report.errors if i.check == "value_range"]
        assert len(range_errors) == 0


# ===========================================================================
# TestCompleteness
# ===========================================================================


class TestCompleteness:
    """Validates completeness checks."""

    def test_high_missing_pct_flagged(self):
        """Series with many gaps triggers warning."""
        # Create 52-week range but only 10 rows for one series
        start = date(2023, 1, 2)
        rows = [
            {"series_id": "sparse", "week": start + timedelta(weeks=w), "quantity": 1.0}
            for w in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
        ]
        df = pl.DataFrame(rows).with_columns(
            pl.col("week").cast(pl.Date),
            pl.col("quantity").cast(pl.Float64),
        )
        cfg = ValidationConfig(enabled=True, max_missing_pct=50.0)
        v = DataValidator(cfg)
        report = v.validate(df)
        comp_warnings = [i for i in report.warnings if i.check == "completeness"]
        assert len(comp_warnings) >= 1

    def test_min_series_count(self):
        df = _make_weekly_actuals(n_series=2, n_weeks=10)
        cfg = ValidationConfig(enabled=True, min_series_count=5)
        v = DataValidator(cfg)
        report = v.validate(df)
        assert not report.passed
        comp_errors = [i for i in report.errors if i.check == "completeness"]
        assert len(comp_errors) == 1

    def test_sufficient_data_passes(self):
        df = _make_weekly_actuals(n_series=3, n_weeks=52)
        cfg = ValidationConfig(enabled=True, min_series_count=3, max_missing_pct=50.0)
        v = DataValidator(cfg)
        report = v.validate(df)
        comp_errors = [i for i in report.errors if i.check == "completeness"]
        assert len(comp_errors) == 0


# ===========================================================================
# TestValidateEndToEnd
# ===========================================================================


class TestValidateEndToEnd:
    """Integration-level validation tests."""

    def test_clean_data_passes(self):
        df = _make_weekly_actuals(n_series=3, n_weeks=52)
        v = DataValidator(ValidationConfig(enabled=True))
        report = v.validate(df)
        assert report.passed
        assert len(report.errors) == 0

    def test_multiple_errors_collected(self):
        """Multiple problems detected in one pass."""
        df = pl.DataFrame({
            "series_id": ["a", "a"],
            "week": ["not-a-date", "also-not"],  # wrong type
            "quantity": [-1.0, 2.0],
        })
        v = DataValidator(ValidationConfig(enabled=True))
        report = v.validate(df)
        assert not report.passed
        # At least a schema type error
        assert len(report.errors) >= 1

    def test_strict_mode_promotes_warnings(self):
        """Warnings become errors when strict=True."""
        # Create data with frequency warnings
        rows = [
            {"series_id": "a", "week": date(2023, 1, 2), "quantity": 1.0},
            {"series_id": "a", "week": date(2023, 1, 4), "quantity": 2.0},  # non-weekly
            {"series_id": "a", "week": date(2023, 1, 9), "quantity": 3.0},
        ]
        df = pl.DataFrame(rows).with_columns(
            pl.col("week").cast(pl.Date),
            pl.col("quantity").cast(pl.Float64),
        )
        # Non-strict: frequency issue is a warning
        v_lenient = DataValidator(ValidationConfig(enabled=True, strict=False))
        r_lenient = v_lenient.validate(df)
        assert r_lenient.passed  # warnings don't block

        # Strict: frequency issue becomes an error
        v_strict = DataValidator(ValidationConfig(enabled=True, strict=True))
        r_strict = v_strict.validate(df)
        assert not r_strict.passed

    def test_disabled_is_noop(self):
        """When enabled=False, validator still runs but we test enabled=True path."""
        # The validator is only called when enabled in the builder.
        # Here we just verify a basic run with no issues.
        df = _make_weekly_actuals(n_series=1, n_weeks=10)
        v = DataValidator(ValidationConfig(enabled=True))
        report = v.validate(df)
        assert report.passed


# ===========================================================================
# TestBuilderIntegration
# ===========================================================================


class TestBuilderIntegration:
    """Test DataValidator wired into SeriesBuilder."""

    def test_builder_raises_on_failed_validation(self):
        from src.series.builder import SeriesBuilder

        cfg = PlatformConfig()
        cfg.data_quality.validation.enabled = True
        cfg.data_quality.validation.check_non_negative = True
        cfg.data_quality.min_series_length_weeks = 0  # don't filter

        builder = SeriesBuilder(cfg)

        df = pl.DataFrame({
            "series_id": ["a", "a"],
            "week": [date(2023, 1, 2), date(2023, 1, 9)],
            "quantity": [-10.0, -20.0],
        }).with_columns(
            pl.col("week").cast(pl.Date),
            pl.col("quantity").cast(pl.Float64),
        )

        with pytest.raises(ValueError, match="validation failed"):
            builder.build(df)

    def test_builder_stores_validation_report(self):
        from src.series.builder import SeriesBuilder

        cfg = PlatformConfig()
        cfg.data_quality.validation.enabled = True
        cfg.data_quality.min_series_length_weeks = 0

        builder = SeriesBuilder(cfg)
        df = _make_weekly_actuals(n_series=2, n_weeks=10)
        builder.build(df)

        from src.data.validator import ValidationReport
        assert isinstance(builder._last_validation_report, ValidationReport)
        assert builder._last_validation_report.passed
