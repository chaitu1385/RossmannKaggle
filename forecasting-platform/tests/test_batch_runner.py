"""
Tests for BatchInferenceRunner — partitioned parallel forecasting.

Covers:
  - All-at-once (batch_size=0) single-process path
  - Series partitioning logic
  - Sequential multi-batch (n_workers=1)
  - ParallelismConfig dataclass defaults
"""

from datetime import date, timedelta

import polars as pl
import pytest


def _make_actuals(n_series: int = 3, n_weeks: int = 60) -> pl.DataFrame:
    """Create a synthetic actuals DataFrame for testing."""
    rows = []
    base_date = date(2023, 1, 2)
    for i in range(n_series):
        sid = f"series_{i}"
        for w in range(n_weeks):
            rows.append({
                "series_id": sid,
                "week": base_date + timedelta(weeks=w),
                "quantity": float(10 + i + (w % 7)),
            })
    return pl.DataFrame(rows)


class TestBatchInferenceRunner:
    """Test BatchInferenceRunner execution paths."""

    def test_all_at_once(self):
        """batch_size=0 runs the standard fit/predict without partitioning."""
        from src.pipeline.batch_runner import BatchInferenceRunner
        from src.forecasting.naive import SeasonalNaiveForecaster

        runner = BatchInferenceRunner(n_workers=1, batch_size=0)
        actuals = _make_actuals(n_series=2, n_weeks=60)
        forecaster = SeasonalNaiveForecaster(season_length=52)

        result = runner.run_forecast(actuals, forecaster, horizon=4)

        assert "forecast" in result.columns
        assert "series_id" in result.columns
        assert "week" in result.columns
        # 2 series × 4 horizon = 8 rows
        assert len(result) == 8

    def test_partitioning(self):
        """_partition_series splits correctly."""
        from src.pipeline.batch_runner import BatchInferenceRunner

        runner = BatchInferenceRunner(batch_size=2)
        actuals = _make_actuals(n_series=5, n_weeks=10)

        batches = runner._partition_series(actuals, "series_id")
        # 5 series / 2 per batch = 3 batches
        assert len(batches) == 3
        # Each batch has exactly the right number of unique series
        assert batches[0]["series_id"].n_unique() == 2
        assert batches[1]["series_id"].n_unique() == 2
        assert batches[2]["series_id"].n_unique() == 1

    def test_sequential_multi_batch(self):
        """n_workers=1 with batch_size > 0 runs sequentially."""
        from src.pipeline.batch_runner import BatchInferenceRunner
        from src.forecasting.naive import SeasonalNaiveForecaster

        runner = BatchInferenceRunner(n_workers=1, batch_size=2)
        actuals = _make_actuals(n_series=4, n_weeks=60)
        forecaster = SeasonalNaiveForecaster(season_length=52)

        result = runner.run_forecast(actuals, forecaster, horizon=3)

        assert len(result) == 12  # 4 series × 3 weeks
        # All series represented
        assert result["series_id"].n_unique() == 4

    def test_empty_actuals(self):
        """Handles empty input gracefully."""
        from src.pipeline.batch_runner import BatchInferenceRunner
        from src.forecasting.naive import SeasonalNaiveForecaster

        runner = BatchInferenceRunner(n_workers=1, batch_size=0)
        actuals = pl.DataFrame(schema={
            "series_id": pl.Utf8, "week": pl.Date, "quantity": pl.Float64
        })
        forecaster = SeasonalNaiveForecaster(season_length=52)

        result = runner.run_forecast(actuals, forecaster, horizon=4)
        assert len(result) == 0

    def test_single_series(self):
        """Single series should work correctly."""
        from src.pipeline.batch_runner import BatchInferenceRunner
        from src.forecasting.naive import SeasonalNaiveForecaster

        runner = BatchInferenceRunner(n_workers=1, batch_size=1)
        actuals = _make_actuals(n_series=1, n_weeks=60)
        forecaster = SeasonalNaiveForecaster(season_length=52)

        result = runner.run_forecast(actuals, forecaster, horizon=5)
        assert len(result) == 5
        assert result["series_id"].n_unique() == 1


class TestParallelismConfig:
    """Test ParallelismConfig dataclass."""

    def test_defaults(self):
        from src.config.schema import ParallelismConfig
        cfg = ParallelismConfig()
        assert cfg.backend == "local"
        assert cfg.n_workers == -1
        assert cfg.n_jobs_statsforecast == -1
        assert cfg.num_threads_mlforecast == -1
        assert cfg.batch_size == 0
        assert cfg.gpu is False

    def test_custom_values(self):
        from src.config.schema import ParallelismConfig
        cfg = ParallelismConfig(
            backend="spark",
            n_workers=4,
            batch_size=100,
            gpu=True,
        )
        assert cfg.backend == "spark"
        assert cfg.n_workers == 4
        assert cfg.batch_size == 100
        assert cfg.gpu is True

    def test_in_platform_config(self):
        """ParallelismConfig should be accessible from PlatformConfig."""
        from src.config.schema import PlatformConfig
        cfg = PlatformConfig()
        assert hasattr(cfg, "parallelism")
        assert cfg.parallelism.backend == "local"
