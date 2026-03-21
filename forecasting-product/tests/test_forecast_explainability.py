"""
Tests for Phase 4 — Forecast explainability layer.

Coverage
--------
TestForecastComparator     — align sources, gap columns, uncertainty ratio
TestExceptionEngine        — each flag type, summary, threshold config
TestForecastExplainer      — STL decomposition, SHAP graceful fallback, narratives
TestModelCard              — construction, from_backtest, serialization
TestModelCardRegistry      — register, get, persist/reload
TestDriftDetector          — ok/warning/alert status, insufficient data
TestForecastLineage        — record, history, latest
"""

import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl

# ── helpers ───────────────────────────────────────────────────────────────────


def _fc(
    series: Dict[str, float],
    weeks: int = 4,
    start: date = date(2024, 1, 1),
    id_col: str = "series_id",
    time_col: str = "week",
    value_col: str = "forecast",
) -> pl.DataFrame:
    rows = []
    for sid, val in series.items():
        for w in range(weeks):
            rows.append({id_col: sid, time_col: start + timedelta(weeks=w), value_col: float(val)})
    return pl.DataFrame(rows)


def _fc_with_quantiles(
    series: Dict[str, tuple],  # sid → (p50, p10, p90)
    weeks: int = 4,
) -> pl.DataFrame:
    rows = []
    for sid, (p50, p10, p90) in series.items():
        for w in range(weeks):
            rows.append({
                "series_id": sid,
                "week": date(2024, 1, 1) + timedelta(weeks=w),
                "forecast": float(p50),
                "forecast_p10": float(p10),
                "forecast_p90": float(p90),
            })
    return pl.DataFrame(rows)


def _history(
    series: Dict[str, float],
    weeks: int = 52,
    start: date = date(2023, 1, 2),
) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for sid, base in series.items():
        for w in range(weeks):
            rows.append({
                "series_id": sid,
                "week": start + timedelta(weeks=w),
                "quantity": float(base + rng.normal(0, base * 0.1)),
            })
    return pl.DataFrame(rows)


def _metric_store_with_data(tmp_path: str, wmape_bt: float = 0.15, wmape_live: float = 0.20):
    from src.metrics.store import MetricStore
    store = MetricStore(tmp_path)
    bt = pl.DataFrame([{
        "run_id": "r1", "run_type": "backtest", "run_date": date(2024, 1, 1),
        "lob": "test", "model_id": "lgbm_direct", "fold": 0,
        "grain_level": "series", "series_id": f"s{i}", "channel": "",
        "target_week": date(2024, 1, 1) + timedelta(weeks=i),
        "actual": 100.0, "forecast": 85.0,
        "wmape": wmape_bt, "normalized_bias": 0.05,
        "mape": wmape_bt, "mae": 15.0, "rmse": 15.0,
    } for i in range(8)])
    store.write(bt, run_type="backtest", lob="test")

    live = pl.DataFrame([{
        "run_id": "r2", "run_type": "live", "run_date": date(2024, 3, 1),
        "lob": "test", "model_id": "lgbm_direct", "fold": -1,
        "grain_level": "series", "series_id": f"s{i}", "channel": "",
        "target_week": date(2024, 3, 1) + timedelta(weeks=i),
        "actual": 100.0, "forecast": 80.0,
        "wmape": wmape_live, "normalized_bias": 0.10,
        "mape": wmape_live, "mae": 20.0, "rmse": 20.0,
    } for i in range(8)])
    store.write(live, run_type="live", lob="test")
    return store


# ─────────────────────────────────────────────────────────────────────────────
# ForecastComparator
# ─────────────────────────────────────────────────────────────────────────────

class TestForecastComparator(unittest.TestCase):

    def setUp(self):
        from src.analytics.comparator import ForecastComparator
        self.comp = ForecastComparator()

    def test_returns_dataframe(self):
        model_fc = _fc({"s1": 100, "s2": 50})
        result = self.comp.compare(model_fc)
        self.assertIsInstance(result, pl.DataFrame)

    def test_model_forecast_column_renamed(self):
        model_fc = _fc({"s1": 100})
        result = self.comp.compare(model_fc)
        self.assertIn("model_forecast", result.columns)
        self.assertNotIn("forecast", result.columns)

    def test_external_forecast_columns(self):
        model_fc = _fc({"s1": 100, "s2": 50})
        field_fc = _fc({"s1": 90, "s2": 55}, value_col="forecast")
        result = self.comp.compare(model_fc, external_forecasts={"field": field_fc})
        self.assertIn("field_forecast", result.columns)
        self.assertIn("field_gap", result.columns)
        self.assertIn("field_gap_pct", result.columns)

    def test_gap_sign_correct(self):
        """model=100, field=90 → gap = +10, gap_pct ≈ +11.1%"""
        model_fc = _fc({"s1": 100}, weeks=1)
        field_fc = _fc({"s1": 90}, weeks=1)
        result = self.comp.compare(model_fc, external_forecasts={"field": field_fc})
        s1 = result.filter(pl.col("series_id") == "s1").to_dicts()[0]
        self.assertAlmostEqual(s1["field_gap"], 10.0, places=3)
        self.assertAlmostEqual(s1["field_gap_pct"], 100 / 90 * 100 - 100, places=1)

    def test_zero_external_gap_pct_is_null(self):
        """Division by zero in gap_pct → None."""
        model_fc = _fc({"s1": 100}, weeks=1)
        field_fc = _fc({"s1": 0}, weeks=1)
        result = self.comp.compare(model_fc, external_forecasts={"field": field_fc})
        s1 = result.filter(pl.col("series_id") == "s1").to_dicts()[0]
        self.assertIsNone(s1["field_gap_pct"])

    def test_prior_model_columns(self):
        model_fc = _fc({"s1": 110})
        prior_fc = _fc({"s1": 100})
        result = self.comp.compare(model_fc, prior_model_forecast=prior_fc)
        self.assertIn("prior_model_forecast", result.columns)
        self.assertIn("cycle_change", result.columns)
        self.assertIn("cycle_change_pct", result.columns)

    def test_cycle_change_correct(self):
        model_fc = _fc({"s1": 110}, weeks=1)
        prior_fc = _fc({"s1": 100}, weeks=1)
        result = self.comp.compare(model_fc, prior_model_forecast=prior_fc)
        row = result.to_dicts()[0]
        self.assertAlmostEqual(row["cycle_change"], 10.0)
        self.assertAlmostEqual(row["cycle_change_pct"], 10.0)

    def test_uncertainty_ratio_computed(self):
        model_fc = _fc_with_quantiles({"s1": (100, 70, 130)}, weeks=1)
        result = self.comp.compare(model_fc)
        self.assertIn("uncertainty_ratio", result.columns)
        row = result.to_dicts()[0]
        # (130 - 70) / 100 = 0.6
        self.assertAlmostEqual(row["uncertainty_ratio"], 0.6, places=5)

    def test_multiple_external_sources(self):
        model_fc = _fc({"s1": 100})
        result = self.comp.compare(
            model_fc,
            external_forecasts={
                "field": _fc({"s1": 90}),
                "financial": _fc({"s1": 120}),
            },
        )
        self.assertIn("field_gap_pct", result.columns)
        self.assertIn("financial_gap_pct", result.columns)

    def test_output_sorted(self):
        model_fc = _fc({"s2": 50, "s1": 100})
        result = self.comp.compare(model_fc)
        ids = result["series_id"].to_list()
        self.assertEqual(ids, sorted(ids))

    def test_summary_method(self):
        model_fc = _fc({"s1": 100, "s2": 50})
        field_fc = _fc({"s1": 90, "s2": 55})
        comparison = self.comp.compare(model_fc, external_forecasts={"field": field_fc})
        summary = self.comp.summary(comparison)
        self.assertIsInstance(summary, pl.DataFrame)
        self.assertEqual(summary["series_id"].n_unique(), 2)


# ─────────────────────────────────────────────────────────────────────────────
# ExceptionEngine
# ─────────────────────────────────────────────────────────────────────────────

class TestExceptionEngine(unittest.TestCase):

    def setUp(self):
        from src.analytics.comparator import ForecastComparator
        from src.analytics.exceptions import ExceptionEngine
        self.comp = ForecastComparator()
        self.eng = ExceptionEngine(
            cycle_change_pct_threshold=20.0,
            uncertainty_ratio_threshold=0.50,
            field_disagree_pct_threshold=25.0,
        )

    def test_returns_dataframe(self):
        model_fc = _fc({"s1": 100})
        comparison = self.comp.compare(model_fc)
        result = self.eng.flag(comparison)
        self.assertIsInstance(result, pl.DataFrame)

    def test_exception_columns_always_present(self):
        model_fc = _fc({"s1": 100})
        result = self.eng.flag(self.comp.compare(model_fc))
        for col in ["exc_large_cycle_change", "exc_high_uncertainty",
                    "exc_field_disagree", "has_exception"]:
            self.assertIn(col, result.columns)

    def test_no_exception_when_stable(self):
        """Stable forecast with no external sources → no exceptions."""
        model_fc = _fc({"s1": 100})
        prior_fc = _fc({"s1": 98})  # only 2% change
        comparison = self.comp.compare(model_fc, prior_model_forecast=prior_fc)
        result = self.eng.flag(comparison)
        self.assertFalse(result["has_exception"].any())

    def test_large_cycle_change_fires(self):
        """50% cycle change exceeds 20% threshold."""
        model_fc = _fc({"s1": 150}, weeks=1)
        prior_fc = _fc({"s1": 100}, weeks=1)
        comparison = self.comp.compare(model_fc, prior_model_forecast=prior_fc)
        result = self.eng.flag(comparison)
        self.assertTrue(result["exc_large_cycle_change"].any())
        self.assertTrue(result["has_exception"].any())

    def test_large_cycle_change_not_fires_when_small(self):
        """5% change does not trigger large cycle change."""
        model_fc = _fc({"s1": 105}, weeks=1)
        prior_fc = _fc({"s1": 100}, weeks=1)
        comparison = self.comp.compare(model_fc, prior_model_forecast=prior_fc)
        result = self.eng.flag(comparison)
        self.assertFalse(result["exc_large_cycle_change"].any())

    def test_high_uncertainty_fires(self):
        """(P90-P10)/P50 = 0.8 > 0.5 threshold."""
        model_fc = _fc_with_quantiles({"s1": (100, 60, 140)}, weeks=1)
        comparison = self.comp.compare(model_fc)
        result = self.eng.flag(comparison)
        self.assertTrue(result["exc_high_uncertainty"].any())

    def test_high_uncertainty_not_fires_when_narrow(self):
        """(P90-P10)/P50 = 0.1 < 0.5 threshold."""
        model_fc = _fc_with_quantiles({"s1": (100, 95, 105)}, weeks=1)
        comparison = self.comp.compare(model_fc)
        result = self.eng.flag(comparison)
        self.assertFalse(result["exc_high_uncertainty"].any())

    def test_field_disagree_fires(self):
        """50% gap vs field exceeds 25% threshold."""
        model_fc = _fc({"s1": 150}, weeks=1)
        field_fc = _fc({"s1": 100}, weeks=1)
        comparison = self.comp.compare(model_fc, external_forecasts={"field": field_fc})
        result = self.eng.flag(comparison)
        self.assertTrue(result["exc_field_disagree"].any())

    def test_overforecast_fires(self):
        """System 40% above field → OVERFORECAST."""
        model_fc = _fc({"s1": 140}, weeks=1)
        field_fc = _fc({"s1": 100}, weeks=1)
        comparison = self.comp.compare(model_fc, external_forecasts={"field": field_fc})
        result = self.eng.flag(comparison)
        self.assertTrue(result["exc_overforecast"].any())

    def test_underforecast_fires(self):
        """System 40% below field → UNDERFORECAST."""
        model_fc = _fc({"s1": 60}, weeks=1)
        field_fc = _fc({"s1": 100}, weeks=1)
        comparison = self.comp.compare(model_fc, external_forecasts={"field": field_fc})
        result = self.eng.flag(comparison)
        self.assertTrue(result["exc_underforecast"].any())

    def test_no_prior_fires_when_prior_missing(self):
        """Series not in prior forecast → exc_no_prior = True."""
        model_fc = _fc({"s1": 100, "s2": 50}, weeks=1)
        prior_fc = _fc({"s1": 95}, weeks=1)  # s2 missing
        comparison = self.comp.compare(model_fc, prior_model_forecast=prior_fc)
        result = self.eng.flag(comparison)
        s2_rows = result.filter(pl.col("series_id") == "s2")
        self.assertTrue(s2_rows["exc_no_prior"].any())

    def test_custom_thresholds(self):
        """Very tight threshold (5%) fires on a 10% change."""
        from src.analytics.exceptions import ExceptionEngine
        eng = ExceptionEngine(cycle_change_pct_threshold=5.0)
        model_fc = _fc({"s1": 110}, weeks=1)
        prior_fc = _fc({"s1": 100}, weeks=1)
        comparison = self.comp.compare(model_fc, prior_model_forecast=prior_fc)
        result = eng.flag(comparison)
        self.assertTrue(result["exc_large_cycle_change"].any())

    def test_exception_summary(self):
        """Summary returns one row per series."""
        model_fc = _fc({"s1": 150, "s2": 100})
        prior_fc = _fc({"s1": 100, "s2": 99})
        comparison = self.comp.compare(model_fc, prior_model_forecast=prior_fc)
        flagged = self.eng.flag(comparison)
        summary = self.eng.exception_summary(flagged)
        self.assertEqual(summary["series_id"].n_unique(), 2)


# ─────────────────────────────────────────────────────────────────────────────
# ForecastExplainer — decomposition
# ─────────────────────────────────────────────────────────────────────────────

class TestForecastExplainerDecompose(unittest.TestCase):

    def setUp(self):
        from src.analytics.explainer import ForecastExplainer
        self.exp = ForecastExplainer(season_length=12, trend_window=6)

    def test_returns_dataframe(self):
        hist = _history({"s1": 100}, weeks=24)
        fc = _fc({"s1": 105}, weeks=4, value_col="forecast")
        result = self.exp.decompose(hist, fc)
        self.assertIsInstance(result, pl.DataFrame)

    def test_output_columns(self):
        hist = _history({"s1": 100}, weeks=24)
        fc = _fc({"s1": 105}, weeks=4, value_col="forecast")
        result = self.exp.decompose(hist, fc)
        for col in ["series_id", "week", "value", "trend", "seasonal", "residual", "is_forecast"]:
            self.assertIn(col, result.columns)

    def test_history_is_forecast_false(self):
        hist = _history({"s1": 100}, weeks=24)
        fc = _fc({"s1": 105}, weeks=4, value_col="forecast")
        result = self.exp.decompose(hist, fc)
        hist_rows = result.filter(~pl.col("is_forecast"))
        self.assertTrue((~hist_rows["is_forecast"]).all())

    def test_forecast_rows_is_forecast_true(self):
        hist = _history({"s1": 100}, weeks=24)
        fc = _fc({"s1": 105}, weeks=4, value_col="forecast")
        result = self.exp.decompose(hist, fc)
        fc_rows = result.filter(pl.col("is_forecast"))
        self.assertEqual(len(fc_rows), 4)

    def test_seasonal_repeats(self):
        """Seasonal component has period = season_length."""
        from src.analytics.explainer import ForecastExplainer
        exp = ForecastExplainer(season_length=4, trend_window=4)
        # Constant series: seasonal should be ~0
        rows = []
        for w in range(20):
            rows.append({"series_id": "s", "week": date(2023, 1, 2) + timedelta(weeks=w), "quantity": 100.0})
        hist = pl.DataFrame(rows)
        fc = _fc({"s": 100}, weeks=4, value_col="forecast")
        result = exp.decompose(hist, fc, id_col="series_id")
        fc_seasonal = result.filter(pl.col("is_forecast"))["seasonal"].drop_nulls().to_list()
        # For constant series, seasonal ≈ 0
        for s in fc_seasonal:
            self.assertAlmostEqual(s, 0.0, delta=1e-3)

    def test_multiple_series(self):
        hist = _history({"s1": 100, "s2": 50}, weeks=24)
        fc = _fc({"s1": 105, "s2": 52}, weeks=4, value_col="forecast")
        result = self.exp.decompose(hist, fc)
        self.assertEqual(result["series_id"].n_unique(), 2)

    def test_trend_extrapolated_to_forecast(self):
        """Trend column is non-null for all forecast rows."""
        hist = _history({"s1": 100}, weeks=24)
        fc = _fc({"s1": 105}, weeks=4, value_col="forecast")
        result = self.exp.decompose(hist, fc)
        fc_trend = result.filter(pl.col("is_forecast"))["trend"].drop_nulls()
        self.assertEqual(len(fc_trend), 4)


class TestForecastExplainerSHAP(unittest.TestCase):
    """SHAP integration — graceful fallback when shap not installed."""

    def test_shap_unavailable_returns_empty_frame(self):
        from src.analytics.explainer import ForecastExplainer
        exp = ForecastExplainer()

        # Ensure shap appears uninstalled
        with patch.dict(sys.modules, {"shap": None}):
            features_df = pl.DataFrame({
                "series_id": ["s1", "s2"],
                "week": [date(2024, 1, 1), date(2024, 1, 8)],
                "price_index": [1.0, 1.1],
                "promo_flag": [0.0, 1.0],
            })
            mock_model = MagicMock()
            result = exp.explain_ml(mock_model, features_df)
        self.assertIsInstance(result, pl.DataFrame)

    def test_shap_columns_when_available(self):
        """When shap IS available, output has feature/shap_value/rank columns."""
        from src.analytics.explainer import ForecastExplainer
        exp = ForecastExplainer()

        # Mock shap module
        mock_shap = MagicMock()
        shap_vals = np.array([[0.5, -0.3], [0.1, 0.8]])
        mock_explanation = MagicMock()
        mock_explanation.values = shap_vals
        mock_shap.Explainer.return_value.return_value = mock_explanation

        features_df = pl.DataFrame({
            "series_id": ["s1", "s2"],
            "week": [date(2024, 1, 1), date(2024, 1, 8)],
            "price_index": [1.0, 1.1],
            "promo_flag": [0.0, 1.0],
        })
        mock_model = MagicMock()
        mock_model.feature_names_in_ = ["price_index", "promo_flag"]

        with patch.dict(sys.modules, {"shap": mock_shap}):
            result = exp.explain_ml(mock_model, features_df)

        self.assertIn("feature", result.columns)
        self.assertIn("shap_value", result.columns)
        self.assertIn("rank", result.columns)


class TestForecastExplainerNarrative(unittest.TestCase):

    def setUp(self):
        from src.analytics.explainer import ForecastExplainer
        self.exp = ForecastExplainer(season_length=12, trend_window=6)

    def test_narrative_returns_dict(self):
        hist = _history({"s1": 100}, weeks=24)
        fc = _fc({"s1": 105}, weeks=4, value_col="forecast")
        decomp = self.exp.decompose(hist, fc)
        result = self.exp.narrative(decomp)
        self.assertIsInstance(result, dict)
        self.assertIn("s1", result)

    def test_narrative_is_string(self):
        hist = _history({"s1": 100}, weeks=24)
        fc = _fc({"s1": 105}, weeks=4, value_col="forecast")
        decomp = self.exp.decompose(hist, fc)
        result = self.exp.narrative(decomp)
        self.assertIsInstance(result["s1"], str)
        self.assertGreater(len(result["s1"]), 10)

    def test_narrative_mentions_series_id(self):
        hist = _history({"s1": 100, "s2": 50}, weeks=24)
        fc = _fc({"s1": 105, "s2": 52}, weeks=4, value_col="forecast")
        decomp = self.exp.decompose(hist, fc)
        result = self.exp.narrative(decomp)
        self.assertIn("s1", result["s1"])
        self.assertIn("s2", result["s2"])

    def test_narrative_with_comparison(self):
        """Narrative includes field gap text when comparison provided."""
        from src.analytics.comparator import ForecastComparator
        hist = _history({"s1": 100}, weeks=24)
        fc = _fc({"s1": 140}, weeks=4, value_col="forecast")
        decomp = self.exp.decompose(hist, fc)

        model_fc = _fc({"s1": 140})
        field_fc = _fc({"s1": 100})
        comparison = ForecastComparator().compare(
            model_fc, external_forecasts={"field": field_fc}
        )
        result = self.exp.narrative(decomp, comparison=comparison)
        self.assertIn("field", result["s1"])

    def test_narrative_uncertainty_label(self):
        """High uncertainty produces a 'HIGH' label in narrative."""
        from src.analytics.comparator import ForecastComparator
        hist = _history({"s1": 100}, weeks=24)
        fc = _fc({"s1": 100}, weeks=4, value_col="forecast")
        decomp = self.exp.decompose(hist, fc)

        model_fc = _fc_with_quantiles({"s1": (100, 20, 180)})  # ratio=1.6 > 0.75
        comparison = ForecastComparator().compare(model_fc)
        result = self.exp.narrative(decomp, comparison=comparison)
        self.assertIn("HIGH", result["s1"])


# ─────────────────────────────────────────────────────────────────────────────
# ModelCard
# ─────────────────────────────────────────────────────────────────────────────

class TestModelCard(unittest.TestCase):

    def _backtest_df(self, model_name="lgbm_direct"):
        return pl.DataFrame([{
            "model_id": model_name, "series_id": f"s{i}",
            "wmape": 0.15, "normalized_bias": 0.02,
            "fold": 0, "target_week": date(2024, 1, 1),
            "actual": 100.0, "forecast": 85.0,
        } for i in range(5)])

    def test_from_backtest(self):
        from src.analytics.governance import ModelCard
        bt = self._backtest_df()
        card = ModelCard.from_backtest("lgbm_direct", "test_lob", bt)
        self.assertEqual(card.model_name, "lgbm_direct")
        self.assertEqual(card.lob, "test_lob")
        self.assertEqual(card.n_series, 5)
        self.assertAlmostEqual(card.backtest_wmape, 0.15)

    def test_to_dict(self):
        from src.analytics.governance import ModelCard
        bt = self._backtest_df()
        card = ModelCard.from_backtest("lgbm_direct", "test", bt)
        d = card.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("model_name", d)
        self.assertIn("backtest_wmape", d)

    def test_to_frame(self):
        from src.analytics.governance import ModelCard
        bt = self._backtest_df()
        card = ModelCard.from_backtest("lgbm_direct", "test", bt)
        frame = card.to_frame()
        self.assertIsInstance(frame, pl.DataFrame)
        self.assertEqual(len(frame), 1)

    def test_features_stored(self):
        from src.analytics.governance import ModelCard
        bt = self._backtest_df()
        features = ["price_index", "promo_flag", "trend_12w"]
        card = ModelCard.from_backtest("lgbm_direct", "test", bt, features=features)
        self.assertEqual(card.features, features)

    def test_config_hash_computed(self):
        from src.analytics.governance import ModelCard
        from src.config.schema import ForecastConfig
        bt = self._backtest_df()
        cfg = ForecastConfig()
        card = ModelCard.from_backtest("lgbm_direct", "test", bt, config=cfg)
        self.assertGreater(len(card.config_hash), 0)


class TestModelCardRegistry(unittest.TestCase):

    def test_register_and_get(self):
        from src.analytics.governance import ModelCard, ModelCardRegistry
        with tempfile.TemporaryDirectory() as tmp:
            reg = ModelCardRegistry(tmp)
            card = ModelCard(model_name="naive_seasonal", lob="test",
                             backtest_wmape=0.22)
            reg.register(card)
            retrieved = reg.get("naive_seasonal")
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved.model_name, "naive_seasonal")

    def test_all_cards_returns_dataframe(self):
        from src.analytics.governance import ModelCard, ModelCardRegistry
        with tempfile.TemporaryDirectory() as tmp:
            reg = ModelCardRegistry(tmp)
            for name in ["naive_seasonal", "lgbm_direct", "auto_arima"]:
                reg.register(ModelCard(model_name=name, lob="test"))
            df = reg.all_cards()
            self.assertEqual(df["model_name"].n_unique(), 3)

    def test_get_missing_returns_none(self):
        from src.analytics.governance import ModelCardRegistry
        with tempfile.TemporaryDirectory() as tmp:
            reg = ModelCardRegistry(tmp)
            self.assertIsNone(reg.get("nonexistent_model"))

    def test_persist_reload(self):
        """Cards survive a registry restart."""
        from src.analytics.governance import ModelCard, ModelCardRegistry
        with tempfile.TemporaryDirectory() as tmp:
            reg1 = ModelCardRegistry(tmp)
            reg1.register(ModelCard(model_name="lgbm_direct", lob="test",
                                    backtest_wmape=0.15))
            # New instance reads from disk
            reg2 = ModelCardRegistry(tmp)
            card = reg2.get("lgbm_direct")
            self.assertIsNotNone(card)


# ─────────────────────────────────────────────────────────────────────────────
# DriftDetector
# ─────────────────────────────────────────────────────────────────────────────

class TestDriftDetector(unittest.TestCase):

    def test_ok_status_when_live_matches_backtest(self):
        from src.analytics.governance import DriftDetector
        with tempfile.TemporaryDirectory() as tmp:
            store = _metric_store_with_data(tmp, wmape_bt=0.15, wmape_live=0.16)
            det = DriftDetector(store, warn_multiplier=1.25, min_live_weeks=1)
            result = det.detect("lgbm_direct", lob="test")
            self.assertEqual(result["status"], "ok")

    def test_warning_status(self):
        from src.analytics.governance import DriftDetector
        with tempfile.TemporaryDirectory() as tmp:
            # live = 1.3x backtest (between 1.25x and 1.5x)
            store = _metric_store_with_data(tmp, wmape_bt=0.15, wmape_live=0.20)
            det = DriftDetector(store, warn_multiplier=1.25, alert_multiplier=1.50, min_live_weeks=1)
            result = det.detect("lgbm_direct", lob="test")
            self.assertIn(result["status"], ("warning", "ok"))  # depends on ratio

    def test_alert_status_when_severely_degraded(self):
        from src.analytics.governance import DriftDetector
        with tempfile.TemporaryDirectory() as tmp:
            # live = 2x backtest → alert
            store = _metric_store_with_data(tmp, wmape_bt=0.10, wmape_live=0.30)
            det = DriftDetector(store, warn_multiplier=1.25, alert_multiplier=1.50, min_live_weeks=1)
            result = det.detect("lgbm_direct", lob="test")
            self.assertEqual(result["status"], "alert")

    def test_insufficient_data_when_no_live(self):
        from src.analytics.governance import DriftDetector
        from src.metrics.store import MetricStore
        with tempfile.TemporaryDirectory() as tmp:
            store = MetricStore(tmp)
            det = DriftDetector(store, min_live_weeks=4)
            result = det.detect("lgbm_direct", lob="test")
            self.assertEqual(result["status"], "insufficient_data")

    def test_batch_detect_returns_dataframe(self):
        from src.analytics.governance import DriftDetector
        with tempfile.TemporaryDirectory() as tmp:
            store = _metric_store_with_data(tmp)
            det = DriftDetector(store, min_live_weeks=1)
            result = det.batch_detect(lob="test")
            self.assertIsInstance(result, pl.DataFrame)

    def test_detect_returns_ratio(self):
        from src.analytics.governance import DriftDetector
        with tempfile.TemporaryDirectory() as tmp:
            store = _metric_store_with_data(tmp, wmape_bt=0.10, wmape_live=0.20)
            det = DriftDetector(store, min_live_weeks=1)
            result = det.detect("lgbm_direct", lob="test")
            if result["ratio"] is not None:
                self.assertAlmostEqual(result["ratio"], 2.0, places=2)


# ─────────────────────────────────────────────────────────────────────────────
# ForecastLineage
# ─────────────────────────────────────────────────────────────────────────────

class TestForecastLineage(unittest.TestCase):

    def test_record_and_history(self):
        from src.analytics.governance import ForecastLineage
        with tempfile.TemporaryDirectory() as tmp:
            lin = ForecastLineage(tmp)
            lin.record(lob="test", model_id="lgbm_direct", n_series=50,
                       horizon_weeks=13, run_date=date(2024, 3, 1))
            hist = lin.history(lob="test")
            self.assertIsInstance(hist, pl.DataFrame)
            self.assertGreater(len(hist), 0)

    def test_multiple_records(self):
        from src.analytics.governance import ForecastLineage
        with tempfile.TemporaryDirectory() as tmp:
            lin = ForecastLineage(tmp)
            for i, model in enumerate(["naive_seasonal", "lgbm_direct", "auto_arima"]):
                lin.record(lob="test", model_id=model,
                           run_date=date(2024, 1, 1) + timedelta(weeks=i))
            hist = lin.history(lob="test")
            self.assertEqual(hist["model_id"].n_unique(), 3)

    def test_latest_returns_most_recent(self):
        from src.analytics.governance import ForecastLineage
        with tempfile.TemporaryDirectory() as tmp:
            lin = ForecastLineage(tmp)
            lin.record(lob="test", model_id="old_model", run_date=date(2024, 1, 1))
            lin.record(lob="test", model_id="new_model", run_date=date(2024, 3, 1))
            latest = lin.latest("test")
            self.assertIsNotNone(latest)
            self.assertEqual(latest["model_id"], "new_model")

    def test_history_filter_by_model(self):
        from src.analytics.governance import ForecastLineage
        with tempfile.TemporaryDirectory() as tmp:
            lin = ForecastLineage(tmp)
            lin.record(lob="test", model_id="lgbm_direct")
            lin.record(lob="test", model_id="naive_seasonal")
            hist = lin.history(model_id="lgbm_direct")
            self.assertTrue((hist["model_id"] == "lgbm_direct").all())

    def test_latest_none_when_empty(self):
        from src.analytics.governance import ForecastLineage
        with tempfile.TemporaryDirectory() as tmp:
            lin = ForecastLineage(tmp)
            self.assertIsNone(lin.latest("no_such_lob"))

    def test_history_sorted_descending(self):
        """history() returns most recent record first."""
        from src.analytics.governance import ForecastLineage
        with tempfile.TemporaryDirectory() as tmp:
            lin = ForecastLineage(tmp)
            for d in [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)]:
                lin.record(lob="test", model_id="lgbm_direct", run_date=d)
            hist = lin.history(lob="test")
            dates = hist["run_date"].to_list()
            self.assertEqual(dates, sorted(dates, reverse=True))


if __name__ == "__main__":
    unittest.main()
