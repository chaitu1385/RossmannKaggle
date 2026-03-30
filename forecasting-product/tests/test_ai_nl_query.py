"""Tests for NaturalLanguageQueryEngine."""

import json
import unittest
from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import polars as pl

from src.ai.nl_query import NaturalLanguageQueryEngine, NLQueryResult

import pytest

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
#  Factory helpers
# --------------------------------------------------------------------------- #

def _make_history_df(series_id="series_0", n_weeks=52, seed=42):
    rng = np.random.RandomState(seed)
    base = date(2023, 1, 2)
    return pl.DataFrame({
        "series_id": [series_id] * n_weeks,
        "week": [base + timedelta(weeks=w) for w in range(n_weeks)],
        "quantity": [float(rng.normal(100, 20)) for _ in range(n_weeks)],
    })


def _make_forecast_df(series_id="series_0", n_weeks=13, seed=42):
    rng = np.random.RandomState(seed)
    base = date(2024, 1, 1)
    return pl.DataFrame({
        "series_id": [series_id] * n_weeks,
        "week": [base + timedelta(weeks=w) for w in range(n_weeks)],
        "forecast": [float(rng.normal(110, 15)) for _ in range(n_weeks)],
    })


def _make_metrics_df(series_id="series_0", n_weeks=26, seed=42):
    rng = np.random.RandomState(seed)
    base = date(2023, 7, 1)
    rows = []
    for w in range(n_weeks):
        actual = float(rng.normal(100, 20))
        forecast = actual * (1 + rng.normal(0, 0.1))
        rows.append({
            "series_id": series_id,
            "target_week": base + timedelta(weeks=w),
            "actual": actual,
            "forecast": forecast,
            "wmape": abs(actual - forecast) / max(abs(actual), 1e-9),
            "normalized_bias": (forecast - actual) / max(abs(actual), 1e-9),
        })
    return pl.DataFrame(rows)


_MOCK_NL_RESPONSE = """### ANSWER
Series_0 shows a moderate upward trend in the last 4 weeks, with forecasts averaging 110 units compared to historical mean of 100. The forecast appears to be tracking the recent uptick in demand. Current accuracy (WMAPE 12%) is within acceptable range.

### CONFIDENCE
high

### SUPPORTING_DATA
```json
{"historical_mean": 100.5, "forecast_mean": 110.2, "wmape": 0.12, "trend": "increasing"}
```"""


# --------------------------------------------------------------------------- #
#  Tests
# --------------------------------------------------------------------------- #

class TestNLQueryInit(unittest.TestCase):
    def test_inherits_base(self):
        engine = NaturalLanguageQueryEngine.__new__(NaturalLanguageQueryEngine)
        engine._client = None
        engine._model = "claude-sonnet-4-20250514"
        engine._max_tokens = 2000
        self.assertFalse(engine.available)


class TestNLQueryGracefulDegradation(unittest.TestCase):
    def test_returns_unavailable_when_no_client(self):
        engine = NaturalLanguageQueryEngine.__new__(NaturalLanguageQueryEngine)
        engine._client = None
        engine._model = "claude-sonnet-4-20250514"
        engine._max_tokens = 2000

        result = engine.query(
            series_id="series_0",
            question="Why did the forecast drop?",
            lob="retail",
        )
        self.assertIn("unavailable", result.answer.lower())
        self.assertEqual(result.confidence, "low")


class TestNLQueryContextGathering(unittest.TestCase):
    def setUp(self):
        self.engine = NaturalLanguageQueryEngine.__new__(NaturalLanguageQueryEngine)
        self.engine._client = None
        self.engine._model = "claude-sonnet-4-20250514"
        self.engine._max_tokens = 2000

    def test_gathers_history_context(self):
        history = _make_history_df()
        context, sources = self.engine._gather_context(
            "series_0", history, None, None, None, None, None,
        )
        self.assertIn("history", context)
        self.assertIn("history", sources)
        self.assertIn("mean", context["history"])

    def test_gathers_forecast_context(self):
        forecast = _make_forecast_df()
        context, sources = self.engine._gather_context(
            "series_0", None, forecast, None, None, None, None,
        )
        self.assertIn("forecast", context)
        self.assertIn("forecast", sources)

    def test_gathers_metrics_context(self):
        metrics = _make_metrics_df()
        context, sources = self.engine._gather_context(
            "series_0", None, None, metrics, None, None, None,
        )
        self.assertIn("metrics", context)
        self.assertIn("metrics", sources)
        self.assertIn("wmape", context["metrics"])

    def test_handles_missing_series(self):
        history = _make_history_df("other_series")
        context, sources = self.engine._gather_context(
            "series_0", history, None, None, None, None, None,
        )
        self.assertNotIn("history", context)

    def test_empty_dataframe(self):
        context, sources = self.engine._gather_context(
            "series_0", pl.DataFrame(), None, None, None, None, None,
        )
        self.assertEqual(context, {})
        self.assertEqual(sources, [])

    def test_all_sources(self):
        history = _make_history_df()
        forecast = _make_forecast_df()
        metrics = _make_metrics_df()
        context, sources = self.engine._gather_context(
            "series_0", history, forecast, metrics, None, None, None,
        )
        self.assertEqual(len(sources), 3)


class TestNLQueryPromptConstruction(unittest.TestCase):
    def test_prompt_contains_question(self):
        engine = NaturalLanguageQueryEngine.__new__(NaturalLanguageQueryEngine)
        engine._client = None
        engine._model = "claude-sonnet-4-20250514"
        engine._max_tokens = 2000

        prompt = engine._build_prompt("series_0", "Why did demand drop?", "retail", {"history": {"mean": 100}})
        self.assertIn("Why did demand drop?", prompt)
        self.assertIn("series_0", prompt)
        self.assertIn("retail", prompt)


class TestNLQueryResponseParsing(unittest.TestCase):
    def test_parses_mock_response(self):
        engine = NaturalLanguageQueryEngine.__new__(NaturalLanguageQueryEngine)
        engine._client = MagicMock()
        engine._model = "claude-sonnet-4-20250514"
        engine._max_tokens = 2000

        result = engine._parse_response(_MOCK_NL_RESPONSE, ["history", "forecast"])

        self.assertIn("Series_0", result.answer)
        self.assertEqual(result.confidence, "high")
        self.assertIn("historical_mean", result.supporting_data)
        self.assertEqual(result.sources_used, ["history", "forecast"])

    def test_handles_malformed_response(self):
        engine = NaturalLanguageQueryEngine.__new__(NaturalLanguageQueryEngine)
        engine._client = MagicMock()
        engine._model = "claude-sonnet-4-20250514"
        engine._max_tokens = 2000

        result = engine._parse_response("Random text without any headers", ["history"])
        self.assertIsInstance(result, NLQueryResult)
        self.assertEqual(result.confidence, "low")


class TestNLQueryMockRoundtrip(unittest.TestCase):
    def test_full_roundtrip(self):
        engine = NaturalLanguageQueryEngine.__new__(NaturalLanguageQueryEngine)
        engine._model = "claude-sonnet-4-20250514"
        engine._max_tokens = 2000

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=_MOCK_NL_RESPONSE)]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message
        engine._client = mock_client

        history = _make_history_df()
        forecast = _make_forecast_df()
        result = engine.query(
            series_id="series_0",
            question="Why did demand increase?",
            lob="retail",
            history=history,
            forecast=forecast,
        )

        mock_client.messages.create.assert_called_once()
        self.assertIsInstance(result.answer, str)
        self.assertIn("Series_0", result.answer)
        self.assertEqual(result.confidence, "high")

    def test_api_error_returns_graceful_message(self):
        engine = NaturalLanguageQueryEngine.__new__(NaturalLanguageQueryEngine)
        engine._model = "claude-sonnet-4-20250514"
        engine._max_tokens = 2000

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("Timeout")
        engine._client = mock_client

        result = engine.query(
            series_id="series_0",
            question="Why?",
            lob="retail",
        )
        self.assertIn("unavailable", result.answer.lower())


if __name__ == "__main__":
    unittest.main()
