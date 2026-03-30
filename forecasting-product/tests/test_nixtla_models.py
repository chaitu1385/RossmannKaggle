"""
Tests for expanded Nixtla model portfolio.

Covers:
  - AutoTheta and MSTL (statsforecast-based): full fit/predict integration tests
  - N-BEATS, NHITS, TFT (neuralforecast-based): mocked tests (neuralforecast optional)
  - Registry integration for all five new models
"""

import random
import unittest
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import polars as pl

from conftest import make_weekly_series as _make_weekly_series
import pytest

pytestmark = pytest.mark.unit


# ═════════════════════════════════════════════════════════════════════════════════
# AutoTheta tests (statsforecast)
# ═════════════════════════════════════════════════════════════════════════════════

class TestAutoThetaForecaster(unittest.TestCase):

    def _get_forecaster(self):
        from src.forecasting.statistical import AutoThetaForecaster
        return AutoThetaForecaster()

    def test_fit_predict_shape(self):
        f = self._get_forecaster()
        df = _make_weekly_series()
        f.fit(df)
        out = f.predict(horizon=8)
        self.assertIn("series_id", out.columns)
        self.assertIn("week", out.columns)
        self.assertIn("forecast", out.columns)
        self.assertEqual(out.shape[0], 2 * 8)
        self.assertFalse(out["forecast"].is_null().any())

    def test_predict_quantiles(self):
        f = self._get_forecaster()
        df = _make_weekly_series()
        f.fit(df)
        out = f.predict_quantiles(horizon=4, quantiles=[0.1, 0.5, 0.9])
        self.assertIn("forecast_p10", out.columns)
        self.assertIn("forecast_p50", out.columns)
        self.assertIn("forecast_p90", out.columns)
        for row in out.iter_rows(named=True):
            self.assertLessEqual(row["forecast_p10"], row["forecast_p90"])

    def test_predict_before_fit_raises(self):
        f = self._get_forecaster()
        with self.assertRaises(RuntimeError):
            f.predict(horizon=4)

    def test_predict_quantiles_before_fit_raises(self):
        f = self._get_forecaster()
        with self.assertRaises(RuntimeError):
            f.predict_quantiles(horizon=4, quantiles=[0.1, 0.5, 0.9])

    def test_get_params(self):
        f = self._get_forecaster()
        p = f.get_params()
        self.assertEqual(p["model"], "AutoTheta")
        self.assertEqual(p["season_length"], 52)
        self.assertEqual(p["decomposition_type"], "multiplicative")

    def test_custom_decomposition_type(self):
        from src.forecasting.statistical import AutoThetaForecaster
        f = AutoThetaForecaster(decomposition_type="additive")
        self.assertEqual(f.get_params()["decomposition_type"], "additive")

    def test_date_column_type(self):
        f = self._get_forecaster()
        df = _make_weekly_series()
        f.fit(df)
        out = f.predict(horizon=4)
        self.assertEqual(out["week"].dtype, pl.Date)

    def test_registry_name(self):
        from src.forecasting.registry import registry
        self.assertIn("auto_theta", registry.available)
        f = registry.build("auto_theta")
        self.assertEqual(f.name, "auto_theta")


# ═════════════════════════════════════════════════════════════════════════════════
# MSTL tests (statsforecast)
# ═════════════════════════════════════════════════════════════════════════════════

class TestMSTLForecaster(unittest.TestCase):

    def _get_forecaster(self):
        from src.forecasting.statistical import MSTLForecaster
        return MSTLForecaster()

    def test_fit_predict_shape(self):
        f = self._get_forecaster()
        df = _make_weekly_series()
        f.fit(df)
        out = f.predict(horizon=8)
        self.assertIn("series_id", out.columns)
        self.assertIn("week", out.columns)
        self.assertIn("forecast", out.columns)
        self.assertEqual(out.shape[0], 2 * 8)
        self.assertFalse(out["forecast"].is_null().any())

    def test_predict_quantiles(self):
        f = self._get_forecaster()
        df = _make_weekly_series()
        f.fit(df)
        out = f.predict_quantiles(horizon=4, quantiles=[0.1, 0.5, 0.9])
        self.assertIn("forecast_p10", out.columns)
        self.assertIn("forecast_p50", out.columns)
        self.assertIn("forecast_p90", out.columns)

    def test_predict_before_fit_raises(self):
        f = self._get_forecaster()
        with self.assertRaises(RuntimeError):
            f.predict(horizon=4)

    def test_get_params(self):
        f = self._get_forecaster()
        p = f.get_params()
        self.assertEqual(p["model"], "MSTL")
        self.assertEqual(p["season_length"], 52)
        self.assertIsNone(p["secondary_season_length"])

    def test_secondary_season_length(self):
        from src.forecasting.statistical import MSTLForecaster
        f = MSTLForecaster(secondary_season_length=13)
        p = f.get_params()
        self.assertEqual(p["secondary_season_length"], 13)

    def test_date_column_type(self):
        f = self._get_forecaster()
        df = _make_weekly_series()
        f.fit(df)
        out = f.predict(horizon=4)
        self.assertEqual(out["week"].dtype, pl.Date)

    def test_registry_name(self):
        from src.forecasting.registry import registry
        self.assertIn("mstl", registry.available)
        f = registry.build("mstl")
        self.assertEqual(f.name, "mstl")


# ═════════════════════════════════════════════════════════════════════════════════
# Neural model tests (mocked — neuralforecast is an optional dependency)
# ═════════════════════════════════════════════════════════════════════════════════

def _make_mock_nf_predict(n_series=2, horizon=13, model_name="NBEATS"):
    """Create a mock NeuralForecast.predict() return value (pandas DataFrame)."""
    import pandas as pd
    rows = []
    for s in range(1, n_series + 1):
        for h in range(horizon):
            rows.append({
                "unique_id": f"S{s}",
                "ds": pd.Timestamp("2025-01-06") + pd.Timedelta(weeks=h),
                model_name: 100.0 + h * 0.5,
            })
    return pd.DataFrame(rows)


def _make_mock_nf_predict_with_levels(n_series=2, horizon=13, model_name="NBEATS"):
    """Mock NeuralForecast.predict(level=[80]) return value with intervals."""
    import pandas as pd
    rows = []
    for s in range(1, n_series + 1):
        for h in range(horizon):
            val = 100.0 + h * 0.5
            rows.append({
                "unique_id": f"S{s}",
                "ds": pd.Timestamp("2025-01-06") + pd.Timedelta(weeks=h),
                model_name: val,
                f"{model_name}-lo-80": val - 20,
                f"{model_name}-hi-80": val + 20,
            })
    return pd.DataFrame(rows)


class TestNBEATSForecaster(unittest.TestCase):

    @patch.dict("sys.modules", {
        "neuralforecast": MagicMock(),
        "neuralforecast.models": MagicMock(),
    })
    def test_get_params(self):
        # Re-import to pick up mock
        import importlib
        import src.forecasting.neural as mod
        importlib.reload(mod)
        f = mod.NBEATSForecaster(max_steps=200, learning_rate=0.01)
        p = f.get_params()
        self.assertEqual(p["model"], "NBEATS")
        self.assertEqual(p["max_steps"], 200)
        self.assertEqual(p["learning_rate"], 0.01)

    @patch.dict("sys.modules", {
        "neuralforecast": MagicMock(),
        "neuralforecast.models": MagicMock(),
    })
    def test_predict_shape(self):
        import importlib
        import src.forecasting.neural as mod
        importlib.reload(mod)

        f = mod.NBEATSForecaster()
        mock_nf = MagicMock()
        mock_nf.predict.return_value = _make_mock_nf_predict(
            n_series=2, horizon=13, model_name="NBEATS"
        )
        f._nf = mock_nf
        f._horizon = 13

        out = f.predict(horizon=8)
        self.assertIn("forecast", out.columns)
        self.assertIn("series_id", out.columns)
        self.assertIn("week", out.columns)
        # Trimmed to 8 from 13
        self.assertEqual(out.shape[0], 2 * 8)

    @patch.dict("sys.modules", {
        "neuralforecast": MagicMock(),
        "neuralforecast.models": MagicMock(),
    })
    def test_predict_quantiles(self):
        import importlib
        import src.forecasting.neural as mod
        importlib.reload(mod)

        f = mod.NBEATSForecaster()
        mock_nf = MagicMock()
        mock_nf.predict.return_value = _make_mock_nf_predict_with_levels(
            n_series=2, horizon=13, model_name="NBEATS"
        )
        f._nf = mock_nf
        f._horizon = 13

        out = f.predict_quantiles(horizon=8, quantiles=[0.1, 0.5, 0.9])
        self.assertIn("forecast_p10", out.columns)
        self.assertIn("forecast_p50", out.columns)
        self.assertIn("forecast_p90", out.columns)
        self.assertEqual(out.shape[0], 2 * 8)

    @patch.dict("sys.modules", {
        "neuralforecast": MagicMock(),
        "neuralforecast.models": MagicMock(),
    })
    def test_predict_before_fit_raises(self):
        import importlib
        import src.forecasting.neural as mod
        importlib.reload(mod)

        f = mod.NBEATSForecaster()
        with self.assertRaises(RuntimeError):
            f.predict(horizon=4)

    def test_registry_name(self):
        from src.forecasting.registry import registry
        self.assertIn("nbeats", registry.available)


class TestNHITSForecaster(unittest.TestCase):

    @patch.dict("sys.modules", {
        "neuralforecast": MagicMock(),
        "neuralforecast.models": MagicMock(),
    })
    def test_get_params(self):
        import importlib
        import src.forecasting.neural as mod
        importlib.reload(mod)
        f = mod.NHITSForecaster(max_steps=300)
        p = f.get_params()
        self.assertEqual(p["model"], "NHITS")
        self.assertEqual(p["max_steps"], 300)

    @patch.dict("sys.modules", {
        "neuralforecast": MagicMock(),
        "neuralforecast.models": MagicMock(),
    })
    def test_predict_shape(self):
        import importlib
        import src.forecasting.neural as mod
        importlib.reload(mod)

        f = mod.NHITSForecaster()
        mock_nf = MagicMock()
        mock_nf.predict.return_value = _make_mock_nf_predict(
            n_series=2, horizon=13, model_name="NHITS"
        )
        f._nf = mock_nf
        f._horizon = 13

        out = f.predict(horizon=13)
        self.assertIn("forecast", out.columns)
        self.assertEqual(out.shape[0], 2 * 13)

    def test_registry_name(self):
        from src.forecasting.registry import registry
        self.assertIn("nhits", registry.available)


class TestTFTForecaster(unittest.TestCase):

    @patch.dict("sys.modules", {
        "neuralforecast": MagicMock(),
        "neuralforecast.models": MagicMock(),
    })
    def test_get_params(self):
        import importlib
        import src.forecasting.neural as mod
        importlib.reload(mod)
        f = mod.TFTForecaster(hidden_size=128, n_head=8)
        p = f.get_params()
        self.assertEqual(p["model"], "TFT")
        self.assertEqual(p["hidden_size"], 128)
        self.assertEqual(p["n_head"], 8)

    @patch.dict("sys.modules", {
        "neuralforecast": MagicMock(),
        "neuralforecast.models": MagicMock(),
    })
    def test_predict_shape(self):
        import importlib
        import src.forecasting.neural as mod
        importlib.reload(mod)

        f = mod.TFTForecaster()
        mock_nf = MagicMock()
        mock_nf.predict.return_value = _make_mock_nf_predict(
            n_series=2, horizon=13, model_name="TFT"
        )
        f._nf = mock_nf
        f._horizon = 13

        out = f.predict(horizon=10)
        self.assertIn("forecast", out.columns)
        self.assertEqual(out.shape[0], 2 * 10)

    def test_registry_name(self):
        from src.forecasting.registry import registry
        self.assertIn("tft", registry.available)


# ═════════════════════════════════════════════════════════════════════════════════
# Registry integration: all new models
# ═════════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("model_name", ["auto_theta", "mstl", "nbeats", "nhits", "tft"])
def test_model_registered(model_name):
    from src.forecasting.registry import registry
    assert model_name in registry.available


class TestExpandedRegistry(unittest.TestCase):
    """Verify registry integration for new models."""

    def test_build_from_config_with_new_models(self):
        from src.forecasting.registry import registry
        # Build statistical models (these work without neuralforecast)
        forecasters = registry.build_from_config(["auto_theta", "mstl"])
        self.assertEqual(len(forecasters), 2)
        self.assertEqual(forecasters[0].name, "auto_theta")
        self.assertEqual(forecasters[1].name, "mstl")

    def test_total_model_count(self):
        """Platform should now have at least 15 registered models."""
        from src.forecasting.registry import registry
        # Original: naive_seasonal, auto_arima, auto_ets, lgbm_direct,
        #           xgboost_direct, croston, croston_sba, tsb,
        #           chronos, timegpt, weighted_ensemble
        # New: auto_theta, mstl, nbeats, nhits, tft
        self.assertGreaterEqual(len(registry.available), 15)


if __name__ == "__main__":
    unittest.main()
