"""
Tests for ModelEvaluator — src/evaluation/evaluator.py.

Covers:
  - evaluate returns all default metrics
  - evaluate with specific metric subset
  - unknown metrics are silently ignored
  - compare_models returns DataFrame sorted by rmspe
  - metric values are reasonable (perfect predictions → ~0 error)
"""

import unittest

import numpy as np
import pandas as pd


class _MockModel:
    """Minimal model satisfying BaseForecaster interface for evaluation."""

    def __init__(self, name: str, predictions: np.ndarray):
        self.name = name
        self._predictions = predictions

    def predict(self, X):
        return self._predictions


class TestModelEvaluator(unittest.TestCase):

    def _get_evaluator(self):
        from src.evaluation.evaluator import ModelEvaluator
        return ModelEvaluator()

    def test_evaluate_all_metrics(self):
        ev = self._get_evaluator()
        y_true = np.array([100.0, 200.0, 150.0, 300.0])
        y_pred = np.array([110.0, 190.0, 160.0, 280.0])
        model = _MockModel("test", y_pred)
        X = pd.DataFrame({"f1": [1, 2, 3, 4]})
        y = pd.Series(y_true)

        result = ev.evaluate(model, X, y)
        self.assertIn("rmspe", result)
        self.assertIn("rmse", result)
        self.assertIn("mae", result)
        self.assertIn("mape", result)
        for v in result.values():
            self.assertIsInstance(v, float)
            self.assertGreater(v, 0)

    def test_evaluate_specific_metrics(self):
        ev = self._get_evaluator()
        y_true = np.array([100.0, 200.0])
        model = _MockModel("test", np.array([110.0, 190.0]))
        X = pd.DataFrame({"f1": [1, 2]})
        y = pd.Series(y_true)

        result = ev.evaluate(model, X, y, metrics=["mae", "rmse"])
        self.assertEqual(set(result.keys()), {"mae", "rmse"})

    def test_unknown_metric_ignored(self):
        ev = self._get_evaluator()
        y_true = np.array([100.0, 200.0])
        model = _MockModel("test", np.array([100.0, 200.0]))
        X = pd.DataFrame({"f1": [1, 2]})
        y = pd.Series(y_true)

        result = ev.evaluate(model, X, y, metrics=["mae", "nonexistent"])
        self.assertIn("mae", result)
        self.assertNotIn("nonexistent", result)

    def test_compare_models(self):
        ev = self._get_evaluator()
        y_true = np.array([100.0, 200.0, 300.0])
        X = pd.DataFrame({"f1": [1, 2, 3]})
        y = pd.Series(y_true)

        good = _MockModel("good", np.array([100.0, 200.0, 300.0]))
        bad = _MockModel("bad", np.array([200.0, 100.0, 400.0]))

        result = ev.compare_models([good, bad], X, y)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.index.name, "model")
        self.assertIn("good", result.index)
        self.assertIn("bad", result.index)
        # "good" should be first (lowest rmspe)
        self.assertEqual(result.index[0], "good")

    def test_perfect_predictions_low_error(self):
        ev = self._get_evaluator()
        y_true = np.array([100.0, 200.0, 300.0])
        model = _MockModel("perfect", y_true.copy())
        X = pd.DataFrame({"f1": [1, 2, 3]})
        y = pd.Series(y_true)

        result = ev.evaluate(model, X, y)
        for metric, val in result.items():
            self.assertAlmostEqual(val, 0.0, places=5,
                                   msg=f"{metric} should be ~0 for perfect predictions")


if __name__ == "__main__":
    unittest.main()
