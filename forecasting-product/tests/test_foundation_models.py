"""
Tests for foundation model forecasters (ChronosForecaster, TimeGPTForecaster).

Because Chronos requires model weights (~200 MB download) and TimeGPT requires
a paid API key, both backends are fully mocked.  The tests verify:

  - fit() stores context correctly (zero-shot: no model loaded)
  - predict() returns well-formed DataFrame (correct schema, row count)
  - predict_quantiles() returns correct columns and sensible values
  - P10 ≤ P90 for every row
  - Both models are discoverable via the model registry
  - Graceful error handling: missing library, missing API key

Mock strategy:
  - Inject ``f._pipeline`` / ``f._client`` directly before calling predict()
    (bypasses the import + download step entirely)
  - Use a fixed random-seed NumPy array as the fake Chronos sample tensor
  - Use a fixed pandas DataFrame as the fake TimeGPT API response
"""
from datetime import date, timedelta
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from src.forecasting.foundation import ChronosForecaster, TimeGPTForecaster
from src.forecasting.registry import registry

from conftest import make_weekly_series as _make_weekly_series

pytestmark = pytest.mark.unit

def _make_torch_forecast_tensor(n_series: int, num_samples: int, horizon: int):
    """Return a fake Chronos forecast tensor (PyTorch-free mock)."""
    rng = np.random.default_rng(42)
    data = rng.uniform(30.0, 80.0, size=(n_series, num_samples, horizon)).astype(np.float32)
    mock = MagicMock()
    mock.numpy.return_value = data
    return mock


def _mock_torch_tensor():
    """Return a mock torch module that passes context tensor creation."""
    mock_torch = MagicMock()
    mock_torch.float32 = np.float32  # dtype used in torch.tensor(... dtype=torch.float32)
    mock_torch.tensor.side_effect = lambda x, dtype=None: np.array(x, dtype=np.float32)
    mock_torch.bfloat16 = "bfloat16"
    return mock_torch


def _make_timegpt_response(
    series_ids: List[str],
    horizon: int,
    levels: List[int] = None,
    start: date = date(2024, 1, 8),
) -> "pd.DataFrame":
    import pandas as pd

    rows = []
    for sid in series_ids:
        for h in range(horizon):
            ds = start + timedelta(weeks=h)
            row = {"unique_id": sid, "ds": ds, "TimeGPT": 60.0 + h}
            if levels:
                for lvl in levels:
                    row[f"TimeGPT-lo-{lvl}"] = 60.0 + h - 5.0
                    row[f"TimeGPT-hi-{lvl}"] = 60.0 + h + 5.0
            rows.append(row)
    return pd.DataFrame(rows)


# ── ChronosForecaster: basic behaviour ────────────────────────────────────────


class TestChronosForecasterFit:

    def test_fit_stores_context_for_all_series(self):
        f = ChronosForecaster()
        data = _make_weekly_series(n_series=3, n_weeks=52)
        f.fit(data, target_col="quantity", time_col="week", id_col="series_id")
        assert len(f._context) == 3
        assert all(len(v) == 52 for v in f._context.values())

    def test_fit_stores_last_dates(self):
        f = ChronosForecaster()
        data = _make_weekly_series(n_series=2, n_weeks=10)
        f.fit(data)
        for sid, last_date in f._last_dates.items():
            assert isinstance(last_date, date)

    def test_fit_does_not_load_pipeline(self):
        """fit() must not trigger any model download."""
        f = ChronosForecaster()
        data = _make_weekly_series(n_series=1)
        f.fit(data)
        assert f._pipeline is None

    def test_fit_clears_previous_context(self):
        f = ChronosForecaster()
        data_a = _make_weekly_series(n_series=3, n_weeks=52)
        data_b = _make_weekly_series(n_series=1, n_weeks=52)
        f.fit(data_a)
        f.fit(data_b)
        assert len(f._context) == 1  # overwritten by second fit


class TestChronosPredictPoint:

    def _fit_and_inject(self, n_series=2, n_weeks=52, horizon=13, num_samples=20):
        f = ChronosForecaster(num_samples=num_samples)
        data = _make_weekly_series(n_series=n_series, n_weeks=n_weeks)
        f.fit(data)
        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = _make_torch_forecast_tensor(
            n_series, num_samples, horizon
        )
        f._pipeline = mock_pipeline
        return f, data

    def _predict(self, f, **kwargs):
        """Call f.predict() with torch patched out (not installed in CI)."""
        with patch.dict("sys.modules", {"torch": _mock_torch_tensor()}):
            return f.predict(**kwargs)

    def test_predict_returns_correct_columns(self):
        f, _ = self._fit_and_inject()
        result = self._predict(f, horizon=13)
        assert "series_id" in result.columns
        assert "week" in result.columns
        assert "forecast" in result.columns

    def test_predict_returns_correct_row_count(self):
        f, _ = self._fit_and_inject(n_series=3, horizon=8)
        result = self._predict(f, horizon=8)
        assert len(result) == 3 * 8

    def test_predict_dates_are_future_weeks(self):
        """Every forecast date should be strictly after the last training date."""
        f, data = self._fit_and_inject(n_series=1, n_weeks=52, horizon=4)
        last_train_date = data.filter(pl.col("series_id") == "SKU-000")["week"].max()
        result = self._predict(f, horizon=4)
        skus = result.filter(pl.col("series_id") == "SKU-000")
        assert skus["week"].min() > last_train_date

    def test_predict_dates_are_weekly_spaced(self):
        f, _ = self._fit_and_inject(n_series=1, horizon=5)
        result = self._predict(f, horizon=5).filter(
            pl.col("series_id") == "SKU-000"
        ).sort("week")
        dates = result["week"].to_list()
        for i in range(1, len(dates)):
            assert (dates[i] - dates[i - 1]).days == 7

    def test_predict_no_nulls(self):
        f, _ = self._fit_and_inject()
        result = self._predict(f, horizon=13)
        assert result["forecast"].null_count() == 0

    def test_predict_calls_pipeline_once(self):
        f, _ = self._fit_and_inject(n_series=2, horizon=13)
        self._predict(f, horizon=13)
        # All series should be batched into a single pipeline.predict() call
        assert f._pipeline.predict.call_count == 1

    def test_predict_empty_context_returns_empty(self):
        f = ChronosForecaster()
        mock_pipeline = MagicMock()
        f._pipeline = mock_pipeline
        with patch.dict("sys.modules", {"torch": _mock_torch_tensor()}):
            result = f.predict(horizon=4)
        assert result.is_empty()
        mock_pipeline.predict.assert_not_called()


class TestChronosPredictQuantiles:

    def _fit_and_inject(self, n_series=2, horizon=8, num_samples=20):
        f = ChronosForecaster(num_samples=num_samples)
        data = _make_weekly_series(n_series=n_series, n_weeks=52)
        f.fit(data)
        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = _make_torch_forecast_tensor(
            n_series, num_samples, horizon
        )
        f._pipeline = mock_pipeline
        return f

    def _predict_q(self, f, **kwargs):
        with patch.dict("sys.modules", {"torch": _mock_torch_tensor()}):
            return f.predict_quantiles(**kwargs)

    def _predict_pt(self, f, **kwargs):
        with patch.dict("sys.modules", {"torch": _mock_torch_tensor()}):
            return f.predict(**kwargs)

    def test_quantile_columns_present(self):
        f = self._fit_and_inject()
        qdf = self._predict_q(f, horizon=8, quantiles=[0.1, 0.5, 0.9])
        assert "forecast_p10" in qdf.columns
        assert "forecast_p50" in qdf.columns
        assert "forecast_p90" in qdf.columns

    def test_no_point_forecast_column(self):
        """predict_quantiles() should NOT include a bare 'forecast' column."""
        f = self._fit_and_inject()
        qdf = self._predict_q(f, horizon=8, quantiles=[0.1, 0.5, 0.9])
        assert "forecast" not in qdf.columns

    def test_row_count(self):
        f = self._fit_and_inject(n_series=3, horizon=5)
        qdf = self._predict_q(f, horizon=5, quantiles=[0.1, 0.5, 0.9])
        assert len(qdf) == 3 * 5

    def test_p10_le_p90(self):
        """P10 ≤ P90 must hold for every row (guaranteed by sorted quantiles)."""
        f = self._fit_and_inject(n_series=2, horizon=13, num_samples=50)
        qdf = self._predict_q(f, horizon=13, quantiles=[0.1, 0.5, 0.9])
        violations = (qdf["forecast_p10"] > qdf["forecast_p90"] + 1e-6).sum()
        assert violations == 0

    def test_p50_is_median_of_point(self):
        """P50 from predict_quantiles should be close to median from predict."""
        f = self._fit_and_inject(n_series=1, horizon=4, num_samples=20)
        # Both calls use the same mocked tensor so results are deterministic
        point = self._predict_pt(f, horizon=4)
        qdf = self._predict_q(f, horizon=4, quantiles=[0.5])
        merged = point.join(qdf, on=["series_id", "week"])
        diff = (merged["forecast"] - merged["forecast_p50"]).abs().max()
        assert diff == pytest.approx(0.0, abs=1e-5)

    def test_arbitrary_quantiles(self):
        """Any quantile set should produce the right column names."""
        f = self._fit_and_inject()
        qdf = self._predict_q(f, horizon=4, quantiles=[0.05, 0.25, 0.75, 0.95])
        for q in [5, 25, 75, 95]:
            assert f"forecast_p{q}" in qdf.columns


# ── ChronosForecaster: error handling ─────────────────────────────────────────


class TestChronosErrorHandling:

    def test_missing_library_raises_import_error(self):
        """If chronos-forecasting is not installed, a clear ImportError is raised."""
        f = ChronosForecaster()
        f._context = {"A": [1.0, 2.0, 3.0]}
        f._last_dates = {"A": date(2024, 1, 1)}
        # f._pipeline is None → will try to load → import chronos fails
        with patch.dict("sys.modules", {"chronos": None, "torch": None}):
            with pytest.raises(ImportError, match="chronos-forecasting"):
                f._load_pipeline()

    def test_get_params(self):
        f = ChronosForecaster(model_name="amazon/chronos-t5-small", num_samples=30)
        p = f.get_params()
        assert p["model_name"] == "amazon/chronos-t5-small"
        assert p["num_samples"] == 30


# ── TimeGPTForecaster: basic behaviour ────────────────────────────────────────


class TestTimeGPTForecasterFit:

    def test_fit_stores_polars_df_with_nixtla_schema(self):
        """fit() stores data as Polars with unique_id/ds/y columns (no pandas conversion yet)."""
        f = TimeGPTForecaster(api_key="dummy")
        data = _make_weekly_series(n_series=2, n_weeks=52)
        f.fit(data)
        assert isinstance(f._train_df, pl.DataFrame)
        assert "unique_id" in f._train_df.columns
        assert "ds" in f._train_df.columns
        assert "y" in f._train_df.columns

    def test_fit_does_not_initialise_client(self):
        """fit() must not call the API."""
        f = TimeGPTForecaster(api_key="dummy")
        data = _make_weekly_series(n_series=1)
        f.fit(data)
        assert f._client is None

    def test_fit_stores_series_count(self):
        f = TimeGPTForecaster(api_key="dummy")
        data = _make_weekly_series(n_series=4, n_weeks=52)
        f.fit(data)
        assert f._train_df["unique_id"].n_unique() == 4


class TestTimeGPTPredictPoint:
    """
    TimeGPT predict tests mock ``_call_api`` directly to avoid the
    ``to_pandas()`` call which requires ``pyarrow`` (not installed in CI).
    """

    def _fit_and_mock_api(self, n_series=2, horizon=8):
        f = TimeGPTForecaster(api_key="test-key")
        data = _make_weekly_series(n_series=n_series, n_weeks=52)
        f.fit(data)
        series_ids = [f"SKU-{i:03d}" for i in range(n_series)]
        # Build the Polars frame that _call_api would normally return
        rows = []
        for sid in series_ids:
            for h in range(horizon):
                rows.append({
                    "series_id": sid,
                    "week": date(2024, 1, 8) + timedelta(weeks=h),
                    "forecast": 60.0 + h,
                })
        api_response = pl.DataFrame(rows)
        return f, api_response, series_ids

    def test_predict_returns_correct_columns(self):
        f, api_resp, _ = self._fit_and_mock_api()
        with patch.object(f, "_call_api", return_value=api_resp):
            result = f.predict(horizon=8)
        assert "series_id" in result.columns
        assert "week" in result.columns
        assert "forecast" in result.columns

    def test_predict_row_count(self):
        f, api_resp, _ = self._fit_and_mock_api(n_series=3, horizon=5)
        with patch.object(f, "_call_api", return_value=api_resp):
            result = f.predict(horizon=5)
        assert len(result) == 3 * 5

    def test_predict_no_nulls(self):
        f, api_resp, _ = self._fit_and_mock_api()
        with patch.object(f, "_call_api", return_value=api_resp):
            result = f.predict(horizon=8)
        assert result["forecast"].null_count() == 0


class TestTimeGPTPredictQuantiles:

    def _fit_and_mock_api(self, n_series=2, horizon=8):
        f = TimeGPTForecaster(api_key="test-key")
        data = _make_weekly_series(n_series=n_series, n_weeks=52)
        f.fit(data)
        series_ids = [f"SKU-{i:03d}" for i in range(n_series)]
        # Include interval columns in the mock API response
        rows = []
        for sid in series_ids:
            for h in range(horizon):
                rows.append({
                    "series_id": sid,
                    "week": date(2024, 1, 8) + timedelta(weeks=h),
                    "TimeGPT": 60.0 + h,
                    "TimeGPT-lo-80": 55.0 + h,
                    "TimeGPT-hi-80": 65.0 + h,
                })
        api_response = pl.DataFrame(rows)
        return f, api_response, series_ids

    def test_quantile_columns_present(self):
        f, api_resp, _ = self._fit_and_mock_api()
        with patch.object(f, "_call_api", return_value=api_resp):
            qdf = f._map_to_quantile_frame(
                horizon=8, quantiles=[0.1, 0.5, 0.9],
                levels=[80], id_col="series_id", time_col="week",
            )
        assert "forecast_p10" in qdf.columns
        assert "forecast_p50" in qdf.columns
        assert "forecast_p90" in qdf.columns

    def test_p10_le_p90(self):
        f, api_resp, _ = self._fit_and_mock_api()
        with patch.object(f, "_call_api", return_value=api_resp):
            qdf = f._map_to_quantile_frame(
                horizon=8, quantiles=[0.1, 0.5, 0.9],
                levels=[80], id_col="series_id", time_col="week",
            )
        violations = (qdf["forecast_p10"] > qdf["forecast_p90"] + 1e-6).sum()
        assert violations == 0

    def test_row_count(self):
        f, api_resp, _ = self._fit_and_mock_api(n_series=2, horizon=4)
        with patch.object(f, "_call_api", return_value=api_resp):
            qdf = f._map_to_quantile_frame(
                horizon=4, quantiles=[0.1, 0.5, 0.9],
                levels=[80], id_col="series_id", time_col="week",
            )
        assert len(qdf) == 2 * 4


# ── TimeGPTForecaster: error handling ─────────────────────────────────────────


class TestTimeGPTErrorHandling:

    def test_predict_without_fit_raises_runtime_error(self):
        f = TimeGPTForecaster(api_key="dummy")
        f._client = MagicMock()  # inject mock so API key check passes
        # _train_df is None → RuntimeError expected
        with pytest.raises(RuntimeError, match="fit()"):
            f.predict(horizon=4)

    def test_missing_api_key_raises_value_error(self):
        import os

        old = os.environ.pop("NIXTLA_API_KEY", None)
        try:
            f = TimeGPTForecaster()  # no key, no env var
            f._train_pdf = MagicMock()  # bypass fit check
            with pytest.raises(ValueError, match="NIXTLA_API_KEY"):
                f._get_client()
        finally:
            if old is not None:
                os.environ["NIXTLA_API_KEY"] = old

    def test_missing_library_raises_import_error(self):
        f = TimeGPTForecaster(api_key="dummy")
        f._train_pdf = MagicMock()
        with patch.dict("sys.modules", {"nixtla": None}):
            with pytest.raises(ImportError, match="nixtla"):
                f._get_client()

    def test_get_params(self):
        f = TimeGPTForecaster(api_key="k", model="timegpt-1-long-horizon")
        p = f.get_params()
        assert p["timegpt_model"] == "timegpt-1-long-horizon"


# ── Registry discovery ────────────────────────────────────────────────────────


class TestFoundationModelRegistry:

    def test_chronos_registered(self):
        assert "chronos" in registry.available

    def test_timegpt_registered(self):
        assert "timegpt" in registry.available

    def test_registry_builds_chronos(self):
        f = registry.build("chronos")
        assert isinstance(f, ChronosForecaster)
        assert f.name == "chronos"

    def test_registry_builds_timegpt(self):
        f = registry.build("timegpt")
        assert isinstance(f, TimeGPTForecaster)
        assert f.name == "timegpt"

    def test_chronos_custom_params(self):
        f = registry.build("chronos", model_name="amazon/chronos-t5-small", num_samples=50)
        assert f.model_name == "amazon/chronos-t5-small"
        assert f.num_samples == 50

    def test_timegpt_accepts_api_key(self):
        f = registry.build("timegpt", api_key="test-key-123")
        assert f.api_key == "test-key-123"


# ── Zero-shot: fit() is truly a no-op for model weights ──────────────────────


class TestZeroShotProperty:

    def test_chronos_fit_twice_replaces_context(self):
        """Re-fitting replaces the context with the new data (stateless w.r.t. model)."""
        f = ChronosForecaster()
        f.fit(_make_weekly_series(n_series=5))
        assert len(f._context) == 5
        f.fit(_make_weekly_series(n_series=2))
        assert len(f._context) == 2

    def test_timegpt_fit_twice_replaces_dataframe(self):
        f = TimeGPTForecaster(api_key="k")
        f.fit(_make_weekly_series(n_series=5))
        n_before = f._train_df["unique_id"].n_unique()
        f.fit(_make_weekly_series(n_series=2))
        n_after = f._train_df["unique_id"].n_unique()
        assert n_before == 5
        assert n_after == 2

    def test_chronos_is_zero_shot_label(self):
        """The model's zero-shot nature should be documented in the class."""
        assert "zero-shot" in ChronosForecaster._ZERO_SHOT_MSG.lower()
        assert "zero-shot" in TimeGPTForecaster._ZERO_SHOT_MSG.lower()
