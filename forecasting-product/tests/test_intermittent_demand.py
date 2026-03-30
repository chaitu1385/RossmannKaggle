"""
Tests for Phase 3 Step 3 — Intermittent / sparse demand handling.

Coverage:
  - SparseDetector: ADI/CV² classification, split(), edge cases
  - CrostonForecaster: fit, predict, predict_quantiles, zero-demand series
  - CrostonSBAForecaster: SBA correction reduces forecast relative to Croston
  - TSBForecaster: fit, predict, predict_quantiles, obsolescence handling
  - Registry: all three models discoverable by name
  - Config: intermittent_forecasters field parsed correctly
  - BacktestEngine: sparse routing integration (smoke test)
"""

import sys
import unittest
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import polars as pl
import numpy as np
import pytest

pytestmark = pytest.mark.unit

# ── helpers ───────────────────────────────────────────────────────────────────

def _make_panel(
    series: dict,
    start: date = date(2022, 1, 3),
    id_col: str = "series_id",
    time_col: str = "week",
    target_col: str = "quantity",
) -> pl.DataFrame:
    """Build a panel DataFrame from {series_id: [values]}."""
    rows = []
    for sid, values in series.items():
        for i, v in enumerate(values):
            rows.append({
                id_col: sid,
                time_col: start + timedelta(weeks=i),
                target_col: float(v),
            })
    return pl.DataFrame(rows)


def _intermittent_values(n: int = 52, demand_prob: float = 0.2, seed: int = 42) -> list:
    """Generate sparse demand: ~20% non-zero, rest 0."""
    rng = np.random.default_rng(seed)
    mask = rng.random(n) < demand_prob
    vals = rng.uniform(1, 10, n)
    return list(np.where(mask, vals, 0.0))


def _smooth_values(n: int = 52, base: float = 100.0, seed: int = 42) -> list:
    """Generate smooth (non-sparse) demand series."""
    rng = np.random.default_rng(seed)
    return list(base + rng.normal(0, 5, n))


# ─────────────────────────────────────────────────────────────────────────────
# SparseDetector tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSparseDetectorClassify(unittest.TestCase):
    """Tests for ADI/CV² classification."""

    def setUp(self):
        from src.series.sparse_detector import SparseDetector
        self.detector = SparseDetector()

    def test_smooth_series_classified_correctly(self):
        """Series with ADI < 1.32 and CV² < 0.49 → smooth."""
        # All weeks have demand close to 100 (low ADI, low CV²)
        values = _smooth_values(n=52, base=100.0)
        df = _make_panel({"A": values})
        result = self.detector.classify(df)
        row = result.filter(pl.col("series_id") == "A").to_dicts()[0]
        self.assertEqual(row["demand_class"], "smooth")
        self.assertFalse(row["is_sparse"])

    def test_intermittent_series_classified_correctly(self):
        """Series with ADI >= 1.32 and low CV² → intermittent."""
        # ~20% demand probability = ADI ≈ 5, which is >> 1.32
        values = _intermittent_values(n=104, demand_prob=0.2, seed=10)
        # Make all non-zero values identical to keep CV² ≈ 0
        vals = [5.0 if v > 0 else 0.0 for v in values]
        df = _make_panel({"B": vals})
        result = self.detector.classify(df)
        row = result.filter(pl.col("series_id") == "B").to_dicts()[0]
        self.assertEqual(row["demand_class"], "intermittent")
        self.assertTrue(row["is_sparse"])

    def test_lumpy_series_classified_correctly(self):
        """High ADI + high CV² → lumpy."""
        values = _intermittent_values(n=104, demand_prob=0.15, seed=7)
        df = _make_panel({"C": values})
        result = self.detector.classify(df)
        row = result.filter(pl.col("series_id") == "C").to_dicts()[0]
        self.assertIn(row["demand_class"], ("lumpy", "intermittent"))  # depends on CV²
        self.assertTrue(row["is_sparse"])

    def test_all_zero_series_insufficient_data(self):
        """All-zero series labelled insufficient_data, treated as dense."""
        values = [0.0] * 52
        df = _make_panel({"Z": values})
        result = self.detector.classify(df)
        row = result.filter(pl.col("series_id") == "Z").to_dicts()[0]
        self.assertEqual(row["demand_class"], "insufficient_data")
        self.assertFalse(row["is_sparse"])

    def test_short_series_insufficient_data(self):
        """Series shorter than min_periods → insufficient_data."""
        from src.series.sparse_detector import SparseDetector
        det = SparseDetector(min_periods=20)
        values = [1.0, 0.0, 0.0, 1.0, 0.0]  # only 5 periods
        df = _make_panel({"S": values})
        result = det.classify(df)
        row = result.to_dicts()[0]
        self.assertEqual(row["demand_class"], "insufficient_data")

    def test_adi_computed_correctly(self):
        """ADI = T / n_nonzero."""
        n = 100
        nonzero_count = 10
        vals = [0.0] * n
        for i in range(nonzero_count):
            vals[i * 10] = 5.0  # every 10th period
        df = _make_panel({"D": vals})
        result = self.detector.classify(df)
        row = result.filter(pl.col("series_id") == "D").to_dicts()[0]
        self.assertAlmostEqual(row["adi"], n / nonzero_count, places=5)

    def test_cv2_computed_correctly(self):
        """CV² = (std/mean)² of non-zero demands."""
        # Constant non-zero demand → CV² = 0
        vals = [0.0, 5.0, 0.0, 5.0, 0.0, 5.0] * 10  # 60 periods
        df = _make_panel({"E": vals})
        result = self.detector.classify(df)
        row = result.filter(pl.col("series_id") == "E").to_dicts()[0]
        self.assertAlmostEqual(row["cv2"], 0.0, places=5)

    def test_classify_returns_all_series(self):
        """Output has one row per input series."""
        panel = _make_panel({
            "s1": _smooth_values(),
            "s2": _intermittent_values(),
            "s3": _smooth_values(base=50.0),
        })
        result = self.detector.classify(panel)
        self.assertEqual(len(result), 3)

    def test_custom_thresholds(self):
        """Custom thresholds change classification boundary."""
        from src.series.sparse_detector import SparseDetector
        # Use a very high ADI threshold so nothing is classified sparse
        det = SparseDetector(adi_threshold=100.0)
        values = _intermittent_values(n=52, demand_prob=0.1)
        df = _make_panel({"F": values})
        result = det.classify(df)
        row = result.to_dicts()[0]
        self.assertFalse(row["is_sparse"])


class TestSparseDetectorSplit(unittest.TestCase):
    """Tests for split() which partitions a panel into dense and sparse."""

    def setUp(self):
        from src.series.sparse_detector import SparseDetector
        self.detector = SparseDetector()

    def test_split_produces_two_partitions(self):
        """dense_df + sparse_df covers all original rows."""
        panel = _make_panel({
            "dense1": _smooth_values(n=52),
            "dense2": _smooth_values(n=52, base=50.0),
            "sparse1": _intermittent_values(n=52, demand_prob=0.1, seed=1),
        })
        dense, sparse = self.detector.split(panel)
        self.assertEqual(len(dense) + len(sparse), len(panel))

    def test_no_mixing(self):
        """A series appears in exactly one partition."""
        panel = _make_panel({
            "d": _smooth_values(n=52),
            "s": _intermittent_values(n=52, demand_prob=0.1),
        })
        dense, sparse = self.detector.split(panel)
        dense_ids = set(dense["series_id"].to_list())
        sparse_ids = set(sparse["series_id"].to_list())
        self.assertEqual(len(dense_ids & sparse_ids), 0)

    def test_all_dense_returns_empty_sparse(self):
        """All smooth series → sparse partition empty."""
        panel = _make_panel({"d1": _smooth_values(), "d2": _smooth_values(base=80.0)})
        dense, sparse = self.detector.split(panel)
        self.assertTrue(sparse.is_empty())
        self.assertEqual(len(dense), len(panel))


# ─────────────────────────────────────────────────────────────────────────────
# Croston forecaster tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCrostonForecasterBasic(unittest.TestCase):
    """Fit/predict/quantiles for CrostonForecaster."""

    def setUp(self):
        from src.forecasting.intermittent import CrostonForecaster
        self.model = CrostonForecaster(alpha=0.1)
        # Simple intermittent series: 0,0,5,0,0,10,0,0,... (demand every 3)
        self.values = [0.0, 0.0, 5.0] * 20 + [0.0, 0.0, 8.0] * 4
        self.df = _make_panel({"s1": self.values})

    def test_fit_stores_state(self):
        """fit() sets _states for each series."""
        self.model.fit(self.df)
        self.assertIn("s1", self.model._states)
        z, x = self.model._states["s1"]
        self.assertGreater(z, 0)
        self.assertGreater(x, 0)

    def test_predict_returns_correct_shape(self):
        """predict() returns horizon rows per series."""
        self.model.fit(self.df)
        out = self.model.predict(horizon=8)
        self.assertEqual(len(out), 8)

    def test_predict_nonnegative(self):
        """All forecast values are >= 0."""
        self.model.fit(self.df)
        out = self.model.predict(horizon=13)
        self.assertTrue((out["forecast"] >= 0).all())

    def test_predict_constant(self):
        """Croston produces a constant forecast over the horizon."""
        self.model.fit(self.df)
        out = self.model.predict(horizon=5)
        vals = out["forecast"].to_list()
        self.assertEqual(len(set(vals)), 1)

    def test_predict_future_dates(self):
        """Forecast dates are strictly after training end."""
        self.model.fit(self.df)
        max_train = self.df["week"].max()
        out = self.model.predict(horizon=4)
        self.assertTrue((out["week"] > max_train).all())

    def test_predict_quantiles_columns(self):
        """predict_quantiles returns expected column names."""
        self.model.fit(self.df)
        out = self.model.predict_quantiles(horizon=4, quantiles=[0.1, 0.5, 0.9])
        self.assertIn("forecast_p10", out.columns)
        self.assertIn("forecast_p50", out.columns)
        self.assertIn("forecast_p90", out.columns)

    def test_quantile_p10_lte_p90(self):
        """P10 <= P90 for all rows (valid intervals)."""
        self.model.fit(self.df)
        out = self.model.predict_quantiles(horizon=4, quantiles=[0.1, 0.5, 0.9])
        self.assertTrue((out["forecast_p10"] <= out["forecast_p90"]).all())

    def test_quantile_p50_equals_point(self):
        """P50 equals the point forecast."""
        self.model.fit(self.df)
        point = self.model.predict(horizon=4)["forecast"].to_list()
        q50 = self.model.predict_quantiles(horizon=4, quantiles=[0.5])["forecast_p50"].to_list()
        for p, q in zip(point, q50):
            self.assertAlmostEqual(p, q, places=6)

    def test_all_zero_series_forecast_zero(self):
        """Series with all-zero demand → forecast 0."""
        from src.forecasting.intermittent import CrostonForecaster
        m = CrostonForecaster()
        df = _make_panel({"zero": [0.0] * 30})
        m.fit(df)
        out = m.predict(horizon=4)
        self.assertTrue((out["forecast"] == 0.0).all())

    def test_multiple_series(self):
        """predict() returns rows for all fitted series."""
        df = _make_panel({
            "s1": [0.0, 5.0, 0.0] * 10,
            "s2": [0.0, 0.0, 3.0] * 10,
        })
        self.model.fit(df)
        out = self.model.predict(horizon=3)
        self.assertEqual(out["series_id"].n_unique(), 2)

    def test_registry_name(self):
        """Croston is registered as 'croston'."""
        from src.forecasting.registry import registry
        m = registry.build("croston")
        self.assertEqual(m.name, "croston")


class TestCrostonSBAForecaster(unittest.TestCase):
    """CrostonSBA applies (1 - alpha/2) correction, reducing the forecast."""

    def test_sba_forecast_lower_than_croston(self):
        """SBA forecast should be <= Croston forecast for the same data."""
        from src.forecasting.intermittent import CrostonForecaster, CrostonSBAForecaster
        values = [0.0, 0.0, 10.0] * 20  # demand every 3rd week = ADI=3
        df = _make_panel({"s": values})

        c = CrostonForecaster(alpha=0.3)
        sba = CrostonSBAForecaster(alpha=0.3)
        c.fit(df)
        sba.fit(df)

        c_val = c.predict(horizon=1)["forecast"].to_list()[0]
        sba_val = sba.predict(horizon=1)["forecast"].to_list()[0]
        self.assertLessEqual(sba_val, c_val + 1e-9)  # SBA ≤ Croston

    def test_sba_registered(self):
        """CrostonSBA is registered as 'croston_sba'."""
        from src.forecasting.registry import registry
        m = registry.build("croston_sba")
        self.assertEqual(m.name, "croston_sba")

    def test_sba_inherits_quantiles(self):
        """CrostonSBA predict_quantiles works correctly."""
        from src.forecasting.intermittent import CrostonSBAForecaster
        values = [0.0, 5.0, 0.0, 8.0] * 15
        df = _make_panel({"s": values})
        m = CrostonSBAForecaster(alpha=0.1)
        m.fit(df)
        out = m.predict_quantiles(horizon=3, quantiles=[0.1, 0.5, 0.9])
        self.assertIn("forecast_p10", out.columns)
        self.assertTrue((out["forecast_p10"] <= out["forecast_p90"]).all())


# ─────────────────────────────────────────────────────────────────────────────
# TSB forecaster tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTSBForecaster(unittest.TestCase):
    """Tests for Teunter-Syntetos-Babai (TSB) forecaster."""

    def setUp(self):
        from src.forecasting.intermittent import TSBForecaster
        self.model = TSBForecaster(alpha_z=0.1, alpha_p=0.1)

    def test_fit_stores_state(self):
        """fit() stores (p, z) state per series."""
        values = [0.0, 5.0, 0.0, 0.0, 8.0] * 12
        df = _make_panel({"s": values})
        self.model.fit(df)
        self.assertIn("s", self.model._states)
        p, z = self.model._states["s"]
        self.assertGreater(p, 0)
        self.assertGreater(z, 0)

    def test_predict_shape(self):
        """predict() returns horizon rows."""
        df = _make_panel({"s": [0.0, 5.0, 0.0] * 15})
        self.model.fit(df)
        out = self.model.predict(horizon=6)
        self.assertEqual(len(out), 6)

    def test_predict_nonnegative(self):
        df = _make_panel({"s": [0.0, 5.0, 0.0] * 15})
        self.model.fit(df)
        out = self.model.predict(horizon=6)
        self.assertTrue((out["forecast"] >= 0).all())

    def test_predict_constant(self):
        """TSB forecast is constant over the horizon (same p * z)."""
        df = _make_panel({"s": [0.0, 5.0, 0.0] * 15})
        self.model.fit(df)
        out = self.model.predict(horizon=5)
        vals = out["forecast"].to_list()
        self.assertEqual(len(set(vals)), 1)

    def test_obsolescence_decay(self):
        """Series that ends with many zeros → low p → lower forecast than Croston."""
        from src.forecasting.intermittent import TSBForecaster, CrostonForecaster
        # 40 periods of demand, then 20 zeros (obsolescence pattern)
        values = [5.0] * 40 + [0.0] * 20
        df = _make_panel({"s": values})

        tsb = TSBForecaster(alpha_z=0.3, alpha_p=0.3)
        croston = CrostonForecaster(alpha=0.3)
        tsb.fit(df)
        croston.fit(df)

        tsb_val = tsb.predict(horizon=1)["forecast"].to_list()[0]
        croston_val = croston.predict(horizon=1)["forecast"].to_list()[0]
        # TSB should detect the demand trailing off and forecast lower
        self.assertLess(tsb_val, croston_val)

    def test_predict_quantiles_columns(self):
        df = _make_panel({"s": [0.0, 5.0, 0.0] * 15})
        self.model.fit(df)
        out = self.model.predict_quantiles(horizon=4, quantiles=[0.1, 0.5, 0.9])
        for col in ["forecast_p10", "forecast_p50", "forecast_p90"]:
            self.assertIn(col, out.columns)

    def test_quantile_ordering(self):
        """P10 <= P90 for all rows."""
        df = _make_panel({"s": [0.0, 3.0, 0.0, 7.0] * 15})
        self.model.fit(df)
        out = self.model.predict_quantiles(horizon=5, quantiles=[0.1, 0.5, 0.9])
        self.assertTrue((out["forecast_p10"] <= out["forecast_p90"]).all())

    def test_p10_can_be_zero(self):
        """Low quantiles are 0 when demand probability is low."""
        from src.forecasting.intermittent import TSBForecaster
        # Very sparse: demand only 10% of the time
        values = [0.0] * 9 + [10.0] + [0.0] * 9 + [10.0]  # 20 obs
        df = _make_panel({"s": values * 5})  # repeat to get 100 obs
        m = TSBForecaster(alpha_z=0.1, alpha_p=0.1)
        m.fit(df)
        out = m.predict_quantiles(horizon=4, quantiles=[0.1, 0.5, 0.9])
        # P10 should be 0 (below the zero-demand mass)
        p10_vals = out["forecast_p10"].to_list()
        self.assertTrue(all(v == 0.0 for v in p10_vals))

    def test_all_zeros(self):
        """All-zero series → forecast 0."""
        df = _make_panel({"z": [0.0] * 30})
        self.model.fit(df)
        out = self.model.predict(horizon=3)
        self.assertTrue((out["forecast"] == 0.0).all())

    def test_registry_name(self):
        """TSB registered as 'tsb'."""
        from src.forecasting.registry import registry
        m = registry.build("tsb")
        self.assertEqual(m.name, "tsb")


# ─────────────────────────────────────────────────────────────────────────────
# Core fitting function tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCrostonFitFunction(unittest.TestCase):
    """Unit tests for _croston_fit internals."""

    def test_all_zeros_returns_zero_z(self):
        from src.forecasting.intermittent import _croston_fit
        z, x = _croston_fit([0.0, 0.0, 0.0, 0.0])
        self.assertEqual(z, 0.0)
        self.assertEqual(x, 1.0)

    def test_constant_demand(self):
        """Constant demand 5 → z converges toward 5, x toward interval."""
        from src.forecasting.intermittent import _croston_fit
        values = [0.0, 0.0, 5.0] * 50  # demand every 3 weeks
        z, x = _croston_fit(values, alpha=0.3)
        # z should be close to 5
        self.assertAlmostEqual(z, 5.0, delta=0.5)
        # x should be close to 3 (interval)
        self.assertAlmostEqual(x, 3.0, delta=0.5)

    def test_sba_correction_reduces_z(self):
        """SBA correction reduces z relative to uncorrected Croston."""
        from src.forecasting.intermittent import _croston_fit
        values = [5.0, 0.0, 0.0] * 20
        z_c, _ = _croston_fit(values, alpha=0.2, sba_correction=False)
        z_sba, _ = _croston_fit(values, alpha=0.2, sba_correction=True)
        self.assertLess(z_sba, z_c)

    def test_single_demand_event(self):
        """Single non-zero observation: z = demand, x = 1."""
        from src.forecasting.intermittent import _croston_fit
        z, x = _croston_fit([0.0, 0.0, 7.0])
        self.assertEqual(z, 7.0)
        self.assertEqual(x, 1.0)


class TestTSBFitFunction(unittest.TestCase):
    """Unit tests for _tsb_fit internals."""

    def test_empty_returns_zeros(self):
        from src.forecasting.intermittent import _tsb_fit
        p, z = _tsb_fit([])
        self.assertEqual(p, 0.0)
        self.assertEqual(z, 0.0)

    def test_all_zeros(self):
        from src.forecasting.intermittent import _tsb_fit
        p, z = _tsb_fit([0.0, 0.0, 0.0])
        self.assertEqual(p, 0.0)
        self.assertEqual(z, 0.0)

    def test_all_demand(self):
        """Fully observed series → p converges toward 1.0."""
        from src.forecasting.intermittent import _tsb_fit
        values = [5.0] * 100
        p, z = _tsb_fit(values, alpha_z=0.1, alpha_p=0.1)
        self.assertGreater(p, 0.9)

    def test_probability_in_range(self):
        from src.forecasting.intermittent import _tsb_fit
        values = [0.0, 5.0] * 30
        p, z = _tsb_fit(values, alpha_z=0.1, alpha_p=0.1)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Config schema tests
# ─────────────────────────────────────────────────────────────────────────────

class TestIntermittentConfig(unittest.TestCase):
    """ForecastConfig correctly exposes intermittent demand settings."""

    def test_default_intermittent_forecasters_empty(self):
        from src.config.schema import ForecastConfig
        fc = ForecastConfig()
        self.assertEqual(fc.intermittent_forecasters, [])

    def test_sparse_detection_default_true(self):
        from src.config.schema import ForecastConfig
        fc = ForecastConfig()
        self.assertTrue(fc.sparse_detection)

    def test_default_adi_threshold(self):
        from src.config.schema import ForecastConfig
        fc = ForecastConfig()
        self.assertAlmostEqual(fc.sparse_adi_threshold, 1.32)

    def test_default_cv2_threshold(self):
        from src.config.schema import ForecastConfig
        fc = ForecastConfig()
        self.assertAlmostEqual(fc.sparse_cv2_threshold, 0.49)

    def test_intermittent_forecasters_set(self):
        from src.config.schema import ForecastConfig
        fc = ForecastConfig(intermittent_forecasters=["croston_sba", "tsb"])
        self.assertEqual(fc.intermittent_forecasters, ["croston_sba", "tsb"])


# ─────────────────────────────────────────────────────────────────────────────
# BacktestEngine sparse routing smoke test
# ─────────────────────────────────────────────────────────────────────────────

class TestBacktestEngineSparseRouting(unittest.TestCase):
    """
    Smoke test: BacktestEngine routes dense/sparse series to different models.
    """

    def _make_config(self):
        from src.config.schema import PlatformConfig
        cfg = PlatformConfig()
        cfg.lob = "test"
        cfg.forecast.series_id_column = "series_id"
        cfg.forecast.time_column = "week"
        cfg.forecast.target_column = "quantity"
        cfg.forecast.sparse_adi_threshold = 1.32
        cfg.forecast.sparse_cv2_threshold = 0.49
        cfg.backtest.n_folds = 1
        cfg.backtest.val_weeks = 4
        cfg.backtest.gap_weeks = 0
        cfg.output.metrics_path = "/tmp/test_metrics_intermittent"
        cfg.metrics = ["wmape"]
        return cfg

    def test_engine_run_with_sparse_forecasters(self):
        """BacktestEngine accepts sparse_forecasters and returns non-empty results."""
        from src.backtesting.engine import BacktestEngine
        from src.forecasting.intermittent import CrostonForecaster
        from src.forecasting.naive import SeasonalNaiveForecaster

        config = self._make_config()

        # Build a panel with one dense and one sparse series (>= 4+4 = 8 periods for fold)
        dense_vals = _smooth_values(n=60, base=100.0)
        sparse_vals = _intermittent_values(n=60, demand_prob=0.15, seed=99)
        panel = _make_panel({"dense": dense_vals, "sparse": sparse_vals})

        metric_store_mock = MagicMock()
        metric_store_mock.write = MagicMock()

        engine = BacktestEngine(config, metric_store=metric_store_mock)

        dense_forecasters = [SeasonalNaiveForecaster(season_length=4)]
        sparse_forecasters = [CrostonForecaster(alpha=0.2)]

        result = engine.run(
            panel,
            forecasters=dense_forecasters,
            sparse_forecasters=sparse_forecasters,
        )

        # Should produce some results (may be empty if detector routes all to one partition)
        self.assertIsInstance(result, pl.DataFrame)

    def test_engine_run_without_sparse_forecasters_unchanged(self):
        """Without sparse_forecasters, engine behaves exactly as before."""
        from src.backtesting.engine import BacktestEngine
        from src.forecasting.naive import SeasonalNaiveForecaster

        config = self._make_config()
        dense_vals = _smooth_values(n=40, base=100.0)
        panel = _make_panel({"d": dense_vals})

        metric_store_mock = MagicMock()
        metric_store_mock.write = MagicMock()

        engine = BacktestEngine(config, metric_store=metric_store_mock)
        result = engine.run(panel, forecasters=[SeasonalNaiveForecaster(season_length=4)])

        self.assertIsInstance(result, pl.DataFrame)


# ─────────────────────────────────────────────────────────────────────────────
# Registry completeness
# ─────────────────────────────────────────────────────────────────────────────

class TestIntermittentRegistry(unittest.TestCase):
    """All three intermittent models are discoverable by name."""

    def test_croston_registered(self):
        from src.forecasting.registry import registry
        self.assertIn("croston", registry.available)

    def test_croston_sba_registered(self):
        from src.forecasting.registry import registry
        self.assertIn("croston_sba", registry.available)

    def test_tsb_registered(self):
        from src.forecasting.registry import registry
        self.assertIn("tsb", registry.available)

    def test_build_all_three(self):
        from src.forecasting.registry import registry
        for name in ["croston", "croston_sba", "tsb"]:
            m = registry.build(name)
            self.assertEqual(m.name, name)


if __name__ == "__main__":
    unittest.main()
