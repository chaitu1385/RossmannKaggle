"""Tests for src/models/ — base.py, xgboost_model.py, lightgbm_model.py."""

from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from src.models.base import BaseForecaster
from src.models.lightgbm_model import LightGBMForecaster
from src.models.xgboost_model import XGBoostForecaster


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_feature_df(n: int = 50, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "lag_1": rng.normal(100, 10, n).tolist(),
        "lag_2": rng.normal(100, 10, n).tolist(),
        "month": (rng.integers(1, 13, n)).tolist(),
    })


def _make_target(n: int = 50, seed: int = 1) -> pl.Series:
    rng = np.random.default_rng(seed)
    return pl.Series("quantity", rng.normal(100, 15, n).tolist())


# ---------------------------------------------------------------------------
# base.py
# ---------------------------------------------------------------------------

class ConcreteForecaster(BaseForecaster):
    """Minimal concrete subclass for testing BaseForecaster."""

    def fit(self, X: pl.DataFrame, y: pl.Series) -> "ConcreteForecaster":
        self.is_fitted = True
        self.feature_names = X.columns
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


class TestBaseForecaster:
    def test_instantiation_defaults(self):
        m = ConcreteForecaster(name="test_model")
        assert m.name == "test_model"
        assert m.params == {}
        assert m.model is None
        assert m.feature_names == []
        assert m.is_fitted is False

    def test_instantiation_with_params(self):
        m = ConcreteForecaster(name="m", params={"n": 10, "lr": 0.1})
        assert m.params == {"n": 10, "lr": 0.1}

    def test_repr_unfitted(self):
        m = ConcreteForecaster(name="mymodel")
        assert "ConcreteForecaster" in repr(m)
        assert "mymodel" in repr(m)
        assert "False" in repr(m)

    def test_repr_fitted(self):
        m = ConcreteForecaster(name="mymodel")
        m.fit(_make_feature_df(), _make_target())
        assert "True" in repr(m)

    def test_get_feature_importance_returns_none_by_default(self):
        m = ConcreteForecaster(name="m")
        assert m.get_feature_importance() is None

    def test_save_and_load(self, tmp_path):
        m = ConcreteForecaster(name="save_test")
        m.fit(_make_feature_df(), _make_target())
        m.save(str(tmp_path))
        loaded = ConcreteForecaster.load(str(tmp_path), "save_test")
        assert loaded.name == "save_test"
        assert loaded.is_fitted is True

    def test_save_creates_directory(self, tmp_path):
        m = ConcreteForecaster(name="dirtest")
        nested = tmp_path / "a" / "b" / "c"
        m.save(str(nested))
        assert (nested / "dirtest.pkl").exists()


# ---------------------------------------------------------------------------
# xgboost_model.py
# ---------------------------------------------------------------------------

class TestXGBoostForecaster:
    def test_default_params_merged(self):
        m = XGBoostForecaster()
        assert m.params["n_estimators"] == 1000
        assert m.params["objective"] == "reg:squarederror"

    def test_custom_params_override_defaults(self):
        m = XGBoostForecaster(params={"n_estimators": 50})
        assert m.params["n_estimators"] == 50
        # Other defaults preserved
        assert m.params["max_depth"] == 6

    def test_early_stopping_excluded_from_model_params(self):
        m = XGBoostForecaster()
        model_params = m._get_model_params()
        assert "early_stopping_rounds" not in model_params

    def test_fit_sets_is_fitted(self):
        m = XGBoostForecaster(params={"n_estimators": 10, "early_stopping_rounds": 5})
        X, y = _make_feature_df(), _make_target()
        m.fit(X, y)
        assert m.is_fitted is True

    def test_fit_stores_feature_names(self):
        m = XGBoostForecaster(params={"n_estimators": 10})
        X, y = _make_feature_df(), _make_target()
        m.fit(X, y)
        assert set(m.feature_names) == {"lag_1", "lag_2", "month"}

    def test_predict_returns_array(self):
        m = XGBoostForecaster(params={"n_estimators": 10})
        X, y = _make_feature_df(), _make_target()
        m.fit(X, y)
        preds = m.predict(X)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(X),)

    def test_predict_before_fit_raises(self):
        m = XGBoostForecaster()
        with pytest.raises(RuntimeError, match="fitted"):
            m.predict(_make_feature_df())

    def test_feature_importance_before_fit_raises(self):
        m = XGBoostForecaster()
        with pytest.raises(RuntimeError, match="fitted"):
            m.get_feature_importance()

    def test_feature_importance_shape(self):
        m = XGBoostForecaster(params={"n_estimators": 10})
        X, y = _make_feature_df(), _make_target()
        m.fit(X, y)
        fi = m.get_feature_importance()
        assert isinstance(fi, pl.DataFrame)
        assert set(fi.columns) == {"feature", "importance"}
        assert fi.shape[0] == 3  # 3 features

    def test_feature_importance_sorted_descending(self):
        m = XGBoostForecaster(params={"n_estimators": 10})
        X, y = _make_feature_df(), _make_target()
        m.fit(X, y)
        fi = m.get_feature_importance()
        importances = fi["importance"].to_list()
        assert importances == sorted(importances, reverse=True)

    def test_feature_cols_subset(self):
        m = XGBoostForecaster(params={"n_estimators": 10}, feature_cols=["lag_1", "lag_2"])
        X, y = _make_feature_df(), _make_target()
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(X),)
        assert m.feature_names == ["lag_1", "lag_2"]

    def test_fit_with_validation_set(self):
        m = XGBoostForecaster(params={"n_estimators": 10, "early_stopping_rounds": 3})
        X_tr = _make_feature_df(40)
        y_tr = _make_target(40)
        X_val = _make_feature_df(10, seed=99)
        y_val = _make_target(10, seed=88)
        m.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
        assert m.is_fitted is True

    def test_name_and_repr(self):
        m = XGBoostForecaster(name="my_xgb")
        assert m.name == "my_xgb"
        assert "my_xgb" in repr(m)


# ---------------------------------------------------------------------------
# lightgbm_model.py
# ---------------------------------------------------------------------------

class TestLightGBMForecaster:
    def test_default_params_merged(self):
        m = LightGBMForecaster()
        assert m.params["n_estimators"] == 1000
        assert m.params["objective"] == "regression"

    def test_custom_params_override_defaults(self):
        m = LightGBMForecaster(params={"n_estimators": 30, "num_leaves": 15})
        assert m.params["n_estimators"] == 30
        assert m.params["num_leaves"] == 15
        # Other defaults preserved
        assert m.params["max_depth"] == 6

    def test_early_stopping_excluded_from_model_params(self):
        m = LightGBMForecaster()
        model_params = m._get_model_params()
        assert "early_stopping_rounds" not in model_params

    def test_fit_sets_is_fitted(self):
        m = LightGBMForecaster(params={"n_estimators": 10, "verbose": -1})
        X, y = _make_feature_df(), _make_target()
        m.fit(X, y)
        assert m.is_fitted is True

    def test_fit_stores_feature_names(self):
        m = LightGBMForecaster(params={"n_estimators": 10, "verbose": -1})
        X, y = _make_feature_df(), _make_target()
        m.fit(X, y)
        assert set(m.feature_names) == {"lag_1", "lag_2", "month"}

    def test_predict_returns_array(self):
        m = LightGBMForecaster(params={"n_estimators": 10, "verbose": -1})
        X, y = _make_feature_df(), _make_target()
        m.fit(X, y)
        preds = m.predict(X)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(X),)

    def test_predict_before_fit_raises(self):
        m = LightGBMForecaster()
        with pytest.raises(RuntimeError, match="fitted"):
            m.predict(_make_feature_df())

    def test_feature_importance_before_fit_raises(self):
        m = LightGBMForecaster()
        with pytest.raises(RuntimeError, match="fitted"):
            m.get_feature_importance()

    def test_feature_importance_shape(self):
        m = LightGBMForecaster(params={"n_estimators": 10, "verbose": -1})
        X, y = _make_feature_df(), _make_target()
        m.fit(X, y)
        fi = m.get_feature_importance()
        assert isinstance(fi, pl.DataFrame)
        assert set(fi.columns) == {"feature", "importance"}
        assert fi.shape[0] == 3

    def test_feature_importance_sorted_descending(self):
        m = LightGBMForecaster(params={"n_estimators": 10, "verbose": -1})
        X, y = _make_feature_df(), _make_target()
        m.fit(X, y)
        fi = m.get_feature_importance()
        importances = fi["importance"].to_list()
        assert importances == sorted(importances, reverse=True)

    def test_feature_cols_subset(self):
        m = LightGBMForecaster(
            params={"n_estimators": 10, "verbose": -1},
            feature_cols=["lag_1", "lag_2"],
        )
        X, y = _make_feature_df(), _make_target()
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(X),)
        assert m.feature_names == ["lag_1", "lag_2"]

    def test_fit_with_validation_set(self):
        m = LightGBMForecaster(params={"n_estimators": 20, "early_stopping_rounds": 5, "verbose": -1})
        X_tr = _make_feature_df(40)
        y_tr = _make_target(40)
        X_val = _make_feature_df(10, seed=99)
        y_val = _make_target(10, seed=88)
        m.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
        assert m.is_fitted is True

    def test_name_and_repr(self):
        m = LightGBMForecaster(name="my_lgbm")
        assert m.name == "my_lgbm"
        assert "my_lgbm" in repr(m)
