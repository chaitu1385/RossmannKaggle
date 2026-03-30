"""Tests for evaluation metrics."""

import numpy as np
import pytest
from src.evaluation.metrics import rmspe, rmse, mae, mape

pytestmark = pytest.mark.unit


def test_rmse_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0


def test_rmse_basic():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])
    assert rmse(y_true, y_pred) == pytest.approx(1.0)


def test_mae_basic():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 2.0])
    assert mae(y_true, y_pred) == pytest.approx(2.0 / 3.0)


def test_rmspe_perfect():
    y = np.array([100.0, 200.0, 300.0])
    assert rmspe(y, y) == pytest.approx(0.0)


def test_rmspe_basic():
    y_true = np.array([100.0])
    y_pred = np.array([110.0])
    expected = np.sqrt(((10.0 / 100.0) ** 2))
    assert rmspe(y_true, y_pred) == pytest.approx(expected)


def test_mape_basic():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 180.0])
    expected = np.mean([10.0 / 100.0, 20.0 / 200.0]) * 100
    assert mape(y_true, y_pred) == pytest.approx(expected)
