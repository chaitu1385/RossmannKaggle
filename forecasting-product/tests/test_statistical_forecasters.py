"""
Tests for statistical forecasters (AutoARIMA, AutoETS) — src/forecasting/statistical.py.

Covers:
  - fit / predict shape and column names
  - predict_quantiles output structure and ordering
  - predict before fit raises RuntimeError
  - get_params returns correct metadata
  - custom season_length
  - output date type enforcement
"""

import random
import unittest
from datetime import date, timedelta

import polars as pl


def _make_weekly_series(n_series: int = 2, n_weeks: int = 104, seed: int = 42):
    random.seed(seed)
    rows = []
    for s in range(1, n_series + 1):
        sid = f"S{s}"
        base = 100.0 + s * 20
        for w in range(n_weeks):
            rows.append({
                "series_id": sid,
                "week": date(2023, 1, 1) + timedelta(weeks=w),  # Sunday
                "quantity": base + random.gauss(0, 10),
            })
    return pl.DataFrame(rows).with_columns(pl.col("quantity").cast(pl.Float64))


class TestAutoARIMAForecaster(unittest.TestCase):

    def _get_forecaster(self):
        from src.forecasting.statistical import AutoARIMAForecaster
        return AutoARIMAForecaster()

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
        # p10 <= p90 row-wise
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
        self.assertEqual(p["model"], "AutoARIMA")
        self.assertEqual(p["season_length"], 52)

    def test_date_column_type(self):
        f = self._get_forecaster()
        df = _make_weekly_series()
        f.fit(df)
        out = f.predict(horizon=4)
        self.assertEqual(out["week"].dtype, pl.Date)


class TestAutoETSForecaster(unittest.TestCase):

    def _get_forecaster(self):
        from src.forecasting.statistical import AutoETSForecaster
        return AutoETSForecaster()

    def test_fit_predict_shape(self):
        f = self._get_forecaster()
        df = _make_weekly_series()
        f.fit(df)
        out = f.predict(horizon=8)
        self.assertIn("forecast", out.columns)
        self.assertEqual(out.shape[0], 2 * 8)

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
        self.assertEqual(p["model"], "AutoETS")
        self.assertEqual(p["season_length"], 52)

    def test_custom_season_length(self):
        from src.forecasting.statistical import AutoETSForecaster
        f = AutoETSForecaster(season_length=12)
        self.assertEqual(f.get_params()["season_length"], 12)

    def test_date_column_type(self):
        f = self._get_forecaster()
        df = _make_weekly_series()
        f.fit(df)
        out = f.predict(horizon=4)
        self.assertEqual(out["week"].dtype, pl.Date)


if __name__ == "__main__":
    unittest.main()
