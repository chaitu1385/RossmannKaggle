"""Tests for regressor variance screening."""

from datetime import date, timedelta

import polars as pl
import pytest

from src.config.schema import (
    ExternalRegressorConfig,
    ForecastConfig,
    PlatformConfig,
    RegressorScreenConfig,
)
from src.data.regressor_screen import RegressorScreenReport, screen_regressors
from src.series.builder import SeriesBuilder

from conftest import make_actuals as _make_actuals


def _make_df_with_features(n_rows: int = 100) -> pl.DataFrame:
    """DataFrame with target and several feature columns."""
    import random
    random.seed(42)
    return pl.DataFrame({
        "series_id": ["S001"] * n_rows,
        "week": [date(2024, 1, 1) + timedelta(weeks=w) for w in range(n_rows)],
        "quantity": [float(100 + w * 2 + (w % 13) * 5) for w in range(n_rows)],
        "good_feature": [float(w % 13) for w in range(n_rows)],
        "constant_feature": [1.0] * n_rows,
        "near_constant": [1.0 + (1e-8 if w == 0 else 0.0) for w in range(n_rows)],
        "noisy_feature": [random.random() for _ in range(n_rows)],
    })


class TestVarianceCheck:
    def test_constant_column_dropped(self):
        df = _make_df_with_features()
        config = RegressorScreenConfig(enabled=True)
        report = screen_regressors(
            df, ["constant_feature", "good_feature"], "quantity", config
        )
        assert "constant_feature" in report.dropped_columns
        assert "constant_feature" in report.low_variance_columns
        assert "good_feature" not in report.dropped_columns

    def test_near_zero_variance_flagged(self):
        df = _make_df_with_features()
        config = RegressorScreenConfig(enabled=True, variance_threshold=1e-4)
        report = screen_regressors(
            df, ["near_constant"], "quantity", config
        )
        assert "near_constant" in report.dropped_columns

    def test_normal_variance_passes(self):
        df = _make_df_with_features()
        config = RegressorScreenConfig(enabled=True)
        report = screen_regressors(
            df, ["good_feature"], "quantity", config
        )
        assert len(report.dropped_columns) == 0
        assert "good_feature" in report.per_column_stats
        assert report.per_column_stats["good_feature"]["variance"] > 0


class TestCorrelationCheck:
    def test_high_correlation_warned(self):
        """Two columns that are 99%+ correlated should trigger a warning."""
        n = 100
        df = pl.DataFrame({
            "quantity": [float(i) for i in range(n)],
            "feat_a": [float(i * 2) for i in range(n)],
            "feat_b": [float(i * 2 + 0.01) for i in range(n)],  # nearly identical
        })
        config = RegressorScreenConfig(enabled=True, correlation_threshold=0.95)
        report = screen_regressors(df, ["feat_a", "feat_b"], "quantity", config)
        assert len(report.high_correlation_pairs) > 0
        pair = report.high_correlation_pairs[0]
        assert pair["col_a"] == "feat_a"
        assert pair["col_b"] == "feat_b"
        assert abs(pair["correlation"]) > 0.95

    def test_low_correlation_passes(self):
        """Uncorrelated features should not trigger warnings."""
        n = 100
        import random
        random.seed(123)
        df = pl.DataFrame({
            "quantity": [float(i) for i in range(n)],
            "feat_a": [float(i) for i in range(n)],
            "feat_b": [random.random() for _ in range(n)],
        })
        config = RegressorScreenConfig(enabled=True, correlation_threshold=0.95)
        report = screen_regressors(df, ["feat_a", "feat_b"], "quantity", config)
        assert len(report.high_correlation_pairs) == 0

    def test_single_feature_no_correlation_check(self):
        df = _make_df_with_features()
        config = RegressorScreenConfig(enabled=True)
        report = screen_regressors(df, ["good_feature"], "quantity", config)
        assert len(report.high_correlation_pairs) == 0


class TestMutualInformation:
    def test_mi_disabled_by_default(self):
        df = _make_df_with_features()
        config = RegressorScreenConfig(enabled=True, mi_enabled=False)
        report = screen_regressors(df, ["good_feature"], "quantity", config)
        assert len(report.low_mi_columns) == 0

    def test_mi_check_runs_when_enabled(self):
        """With MI enabled, constant noise should have low MI."""
        n = 200
        import random
        random.seed(99)
        df = pl.DataFrame({
            "quantity": [float(i * 3 + (i % 13) * 5) for i in range(n)],
            "signal": [float(i * 3) for i in range(n)],  # correlated with target
            "noise": [random.random() for _ in range(n)],  # no signal
        })
        config = RegressorScreenConfig(
            enabled=True, mi_enabled=True, mi_threshold=0.1
        )
        report = screen_regressors(df, ["signal", "noise"], "quantity", config)
        # noise should have low MI; signal should be fine
        assert "noise" in report.low_mi_columns
        # signal should not be flagged
        assert "signal" not in report.low_mi_columns


class TestAutoDrop:
    def test_auto_drop_removes_from_dropped_list(self):
        df = _make_df_with_features()
        config = RegressorScreenConfig(enabled=True, auto_drop=True)
        report = screen_regressors(
            df, ["constant_feature", "good_feature"], "quantity", config
        )
        assert "constant_feature" in report.dropped_columns

    def test_auto_drop_false_still_flags(self):
        """auto_drop=False still populates dropped_columns for the caller to handle."""
        df = _make_df_with_features()
        config = RegressorScreenConfig(enabled=True, auto_drop=False)
        report = screen_regressors(
            df, ["constant_feature", "good_feature"], "quantity", config
        )
        # The report still lists them as dropped (caller decides what to do)
        assert "constant_feature" in report.dropped_columns


class TestEdgeCases:
    def test_empty_feature_list(self):
        df = _make_df_with_features()
        config = RegressorScreenConfig(enabled=True)
        report = screen_regressors(df, [], "quantity", config)
        assert len(report.dropped_columns) == 0
        assert len(report.warnings) == 0

    def test_missing_columns_ignored(self):
        df = _make_df_with_features()
        config = RegressorScreenConfig(enabled=True)
        report = screen_regressors(
            df, ["nonexistent_col"], "quantity", config
        )
        assert len(report.dropped_columns) == 0

    def test_default_config(self):
        config = RegressorScreenConfig()
        assert config.enabled is False
        assert config.variance_threshold == 1e-6
        assert config.correlation_threshold == 0.95
        assert config.mi_enabled is False
        assert config.auto_drop is True

    def test_report_per_column_stats_populated(self):
        df = _make_df_with_features()
        config = RegressorScreenConfig(enabled=True)
        report = screen_regressors(
            df, ["good_feature", "constant_feature"], "quantity", config
        )
        assert "good_feature" in report.per_column_stats
        assert "constant_feature" in report.per_column_stats
        assert "variance" in report.per_column_stats["good_feature"]


class TestSeriesBuilderIntegration:
    def test_screening_in_series_builder(self):
        """End-to-end: constant feature should be dropped when screening enabled."""
        actuals = _make_actuals(n_weeks=52, n_series=2)
        base = date(2024, 1, 1)

        # External features with one constant column
        rows = []
        for s in range(2):
            for w in range(52):
                rows.append({
                    "series_id": f"S{s:03d}",
                    "week": base + timedelta(weeks=w),
                    "useful_promo": 1 if w % 8 == 0 else 0,
                    "useless_constant": 1.0,
                })
        features = pl.DataFrame(rows)

        config = PlatformConfig(
            forecast=ForecastConfig(
                external_regressors=ExternalRegressorConfig(
                    enabled=True,
                    feature_columns=["useful_promo", "useless_constant"],
                    screen=RegressorScreenConfig(enabled=True, auto_drop=True),
                ),
            ),
        )
        builder = SeriesBuilder(config)
        result = builder.build(actuals, external_features=features)

        # Constant column should be dropped
        assert "useless_constant" not in result.columns
        # Useful column should survive
        assert "useful_promo" in result.columns
        # Report should be available
        assert builder._last_regressor_screen_report is not None
        assert "useless_constant" in builder._last_regressor_screen_report.dropped_columns

    def test_screening_disabled_keeps_all(self):
        """When screening is disabled, all features pass through."""
        actuals = _make_actuals(n_weeks=52, n_series=1)
        base = date(2024, 1, 1)
        features = pl.DataFrame({
            "series_id": ["S000"] * 52,
            "week": [base + timedelta(weeks=w) for w in range(52)],
            "constant_col": [1.0] * 52,
        })
        config = PlatformConfig(
            forecast=ForecastConfig(
                external_regressors=ExternalRegressorConfig(
                    enabled=True,
                    feature_columns=["constant_col"],
                    screen=RegressorScreenConfig(enabled=False),
                ),
            ),
        )
        builder = SeriesBuilder(config)
        result = builder.build(actuals, external_features=features)
        # Constant column should still be present (screening disabled)
        assert "constant_col" in result.columns
