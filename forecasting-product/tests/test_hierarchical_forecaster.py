"""
Tests for hierarchical forecaster and neural model training notes.
"""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from src.forecasting.hierarchical import (
    HierarchicalForecaster,
    build_hierarchy_tags,
    build_summing_matrix_df,
)
from src.forecasting.neural import _NeuralforecastBase
from src.forecasting.registry import registry


# ═══════════════════════════════════════════════════════════════════════════════
# Test data generators
# ═══════════════════════════════════════════════════════════════════════════════

def _make_hierarchical_actuals(
    n_weeks: int = 52,
    start_date: date = date(2022, 1, 3),
) -> pl.DataFrame:
    """Synthetic data with geography hierarchy: region → store."""
    rng = np.random.RandomState(42)
    stores = {
        "East": ["store_E1", "store_E2"],
        "West": ["store_W1", "store_W2"],
    }
    rows = []
    for region, store_list in stores.items():
        for store in store_list:
            base = 100 + rng.randint(0, 50)
            for w in range(n_weeks):
                week_date = start_date + timedelta(weeks=w)
                seasonal = 1.3 if week_date.month >= 10 else 1.0
                noise = rng.normal(0, base * 0.1)
                quantity = max(0, base * seasonal + noise)
                rows.append({
                    "series_id": store,
                    "region": region,
                    "week": week_date,
                    "quantity": round(quantity, 2),
                })
    return pl.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Hierarchy Tag Building
# ═══════════════════════════════════════════════════════════════════════════════

class TestHierarchyTags:

    def test_build_tags_basic(self):
        df = _make_hierarchical_actuals()
        tags = build_hierarchy_tags(df, ["region", "series_id"], id_col="series_id")

        assert "region" in tags
        assert "series_id" in tags
        assert len(tags["series_id"]) == 4  # 4 stores
        assert set(tags["region"]) == {"East", "West"}

    def test_build_tags_missing_levels(self):
        df = _make_hierarchical_actuals()
        with pytest.raises(ValueError, match="None of the hierarchy levels"):
            build_hierarchy_tags(df, ["nonexistent"], id_col="series_id")


class TestSummingMatrix:

    def test_summing_matrix_shape(self):
        df = _make_hierarchical_actuals()
        tags = build_hierarchy_tags(df, ["region", "series_id"], id_col="series_id")
        S = build_summing_matrix_df(tags, ["region", "series_id"])

        # 2 regions + 4 stores = 6 rows
        assert S.height == 6
        # unique_id + 4 bottom-level columns
        assert len(S.columns) == 5

    def test_summing_matrix_identity_at_bottom(self):
        df = _make_hierarchical_actuals()
        tags = build_hierarchy_tags(df, ["region", "series_id"], id_col="series_id")
        S = build_summing_matrix_df(tags, ["region", "series_id"])

        # Bottom 4 rows should form identity
        bottom = S.tail(4)
        bottom_ids = sorted(tags["series_id"])
        for i, bid in enumerate(bottom_ids):
            row = bottom.filter(pl.col("unique_id") == bid)
            assert row[bid][0] == 1.0

    def test_summing_matrix_aggregation(self):
        df = _make_hierarchical_actuals()
        tags = build_hierarchy_tags(df, ["region", "series_id"], id_col="series_id")
        S = build_summing_matrix_df(tags, ["region", "series_id"])

        # East region should sum store_E1 + store_E2
        east = S.filter(pl.col("unique_id") == "region/East")
        assert east["store_E1"][0] == 1.0
        assert east["store_E2"][0] == 1.0
        assert east["store_W1"][0] == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: HierarchicalForecaster
# ═══════════════════════════════════════════════════════════════════════════════

class TestHierarchicalForecaster:

    def test_registered_in_registry(self):
        assert "hierarchical_reconciliation" in registry.available

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown hierarchical method"):
            HierarchicalForecaster(method="invalid")

    def test_valid_methods(self):
        for method in ["bottom_up", "top_down", "middle_out", "mint_shrink",
                        "mint_cov", "ols", "wls_struct", "erm"]:
            f = HierarchicalForecaster(method=method)
            assert f.method == method

    def test_get_params(self):
        f = HierarchicalForecaster(method="mint_shrink")
        params = f.get_params()
        assert params["method"] == "mint_shrink"
        assert params["model"] == "HierarchicalReconciliation"

    def test_fit_predict_basic(self):
        """Test that fit/predict works without hierarchicalforecast library.

        The HierarchicalForecaster falls back to naive seasonal
        when hierarchicalforecast is not installed.
        """
        df = _make_hierarchical_actuals(n_weeks=52)

        try:
            f = HierarchicalForecaster(method="mint_shrink")
            f.fit(
                df,
                hierarchy_levels=["region", "series_id"],
                id_col="series_id",
            )
            result = f.predict(horizon=4, id_col="series_id")

            assert isinstance(result, pl.DataFrame)
            assert "forecast" in result.columns
            assert "series_id" in result.columns
            assert "week" in result.columns
        except ImportError:
            pytest.skip("hierarchicalforecast not installed")

    def test_predict_before_fit_raises(self):
        f = HierarchicalForecaster(method="bottom_up")
        with pytest.raises(RuntimeError, match="Call fit"):
            f.predict(horizon=4)

    def test_predict_horizon_correct(self):
        df = _make_hierarchical_actuals(n_weeks=52)
        try:
            f = HierarchicalForecaster(method="bottom_up")
            f.fit(df, hierarchy_levels=["region", "series_id"], id_col="series_id")
            result = f.predict(horizon=8, id_col="series_id")

            # Each leaf series should have exactly `horizon` forecast rows
            for sid in result["series_id"].unique().to_list():
                n = result.filter(pl.col("series_id") == sid).height
                assert n == 8, f"Expected 8 rows for {sid}, got {n}"
        except ImportError:
            pytest.skip("hierarchicalforecast not installed")


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Neural Model Training Notes
# ═══════════════════════════════════════════════════════════════════════════════

class TestNeuralTrainingNotes:

    def test_cpu_default_notes(self):
        """CPU defaults should flag as non-production."""
        # Create a minimal concrete subclass for testing
        from src.forecasting.neural import NBEATSForecaster

        f = NBEATSForecaster(max_steps=200, accelerator="cpu")
        notes = f.training_notes()

        assert notes["is_production_quality"] is False
        assert "CPU defaults" in notes["recommendation"]
        assert "max_steps" in notes["current_settings"]
        assert notes["current_settings"]["accelerator"] == "cpu"

    def test_gpu_production_notes(self):
        """GPU with high max_steps should flag as production-ready."""
        from src.forecasting.neural import NHITSForecaster

        f = NHITSForecaster(max_steps=3000, accelerator="gpu")
        notes = f.training_notes()

        assert notes["is_production_quality"] is True
        assert "Production-ready" in notes["recommendation"]

    def test_gpu_recommended_defaults(self):
        from src.forecasting.neural import TFTForecaster

        f = TFTForecaster()
        notes = f.training_notes()
        gpu_rec = notes["gpu_recommended"]

        assert gpu_rec["max_steps"] >= 2000
        assert gpu_rec["accelerator"] == "gpu"

    def test_training_notes_model_name(self):
        from src.forecasting.neural import NBEATSForecaster

        f = NBEATSForecaster()
        notes = f.training_notes()
        assert notes["model"] == "nbeats"


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: BacktestEngine neural notes integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestBacktestNeuralNotes:

    def test_backtest_engine_has_neural_notes_property(self):
        from src.backtesting.engine import BacktestEngine
        from src.config.schema import PlatformConfig

        config = PlatformConfig()
        engine = BacktestEngine(config)
        # Before any run, notes should be empty
        assert engine.neural_training_notes == []
