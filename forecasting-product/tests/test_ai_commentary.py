"""Tests for CommentaryEngine."""

import json
import unittest
from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import polars as pl

from src.ai.commentary import CommentaryEngine, CommentaryResult, KeyMetric
from src.metrics.drift import DriftAlert, DriftSeverity

import pytest

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
#  Factory helpers
# --------------------------------------------------------------------------- #

def _make_metrics_df(n_series=5, n_weeks=26, seed=42):
    """Generate synthetic MetricStore-like DataFrame."""
    rng = np.random.RandomState(seed)
    rows = []
    base = date(2024, 1, 1)
    for i in range(n_series):
        for w in range(n_weeks):
            actual = float(rng.normal(100, 20))
            forecast = actual * (1 + rng.normal(0, 0.15))
            wmape = abs(actual - forecast) / max(abs(actual), 1e-9)
            bias = (forecast - actual) / max(abs(actual), 1e-9)
            rows.append({
                "series_id": f"series_{i}",
                "target_week": base + timedelta(weeks=w),
                "actual": actual,
                "forecast": forecast,
                "wmape": wmape,
                "normalized_bias": bias,
            })
    return pl.DataFrame(rows)


def _make_drift_alerts(n=3):
    alerts = []
    for i in range(n):
        severity = DriftSeverity.CRITICAL if i == 0 else DriftSeverity.WARNING
        alerts.append(DriftAlert(
            series_id=f"series_{i}",
            metric="accuracy",
            severity=severity,
            current_value=0.3 + i * 0.05,
            baseline_value=0.15,
            message=f"Alert for series_{i}",
        ))
    return alerts


_MOCK_COMMENTARY_RESPONSE = """### EXECUTIVE_SUMMARY
Forecast performance for the retail LOB remains stable with an overall WMAPE of 15.2%. The team should focus on three series showing accuracy degradation. Bias is well-controlled at +2.1%, indicating a slight tendency to over-forecast. No systemic issues detected.

### KEY_METRICS
```json
[
  {"name": "Overall WMAPE", "value": 0.152, "unit": "%", "trend": "stable"},
  {"name": "Average Bias", "value": 0.021, "unit": "%", "trend": "improving"},
  {"name": "Series Coverage", "value": 5, "unit": "series", "trend": "stable"}
]
```

### EXCEPTIONS
- series_0: Critical accuracy drift, WMAPE increased by 50% above baseline
- series_1: Warning-level bias shift detected
- series_2: Minor accuracy degradation, monitor next cycle

### ACTION_ITEMS
- Retrain models for series_0 with recent data
- Investigate root cause of bias shift in series_1
- Schedule data quality review for next sprint"""


# --------------------------------------------------------------------------- #
#  Tests
# --------------------------------------------------------------------------- #

class TestCommentaryInit(unittest.TestCase):
    def test_inherits_base(self):
        engine = CommentaryEngine.__new__(CommentaryEngine)
        engine._client = None
        engine._model = "claude-sonnet-4-20250514"
        engine._max_tokens = 2000
        self.assertFalse(engine.available)


class TestCommentaryComputeStats(unittest.TestCase):
    def test_computes_stats_from_metrics(self):
        df = _make_metrics_df()
        stats = CommentaryEngine._compute_stats(df)
        self.assertIn("overall_wmape", stats)
        self.assertIn("overall_bias", stats)
        self.assertIn("n_series", stats)
        self.assertEqual(stats["n_series"], 5)

    def test_empty_dataframe(self):
        df = pl.DataFrame({"wmape": [], "normalized_bias": [], "series_id": [], "target_week": []})
        stats = CommentaryEngine._compute_stats(df)
        # Should not crash, may have no stats
        self.assertIsInstance(stats, dict)


class TestCommentaryGracefulDegradation(unittest.TestCase):
    def test_empty_metrics(self):
        engine = CommentaryEngine.__new__(CommentaryEngine)
        engine._client = None
        engine._model = "claude-sonnet-4-20250514"
        engine._max_tokens = 2000

        result = engine.generate(lob="retail", metrics_df=pl.DataFrame())
        self.assertIn("No metric data", result.executive_summary)

    def test_fallback_when_unavailable(self):
        engine = CommentaryEngine.__new__(CommentaryEngine)
        engine._client = None
        engine._model = "claude-sonnet-4-20250514"
        engine._max_tokens = 2000

        df = _make_metrics_df()
        alerts = _make_drift_alerts()
        result = engine.generate(lob="retail", metrics_df=df, drift_alerts=alerts)

        self.assertIn("unavailable", result.executive_summary.lower())
        self.assertGreaterEqual(len(result.key_metrics), 1)
        self.assertIsInstance(result.key_metrics, list)
        self.assertGreaterEqual(len(result.action_items), 1)
        self.assertIsInstance(result.action_items, list)

    def test_fallback_includes_drift_count(self):
        engine = CommentaryEngine.__new__(CommentaryEngine)
        engine._client = None
        engine._model = "claude-sonnet-4-20250514"
        engine._max_tokens = 2000

        df = _make_metrics_df()
        alerts = _make_drift_alerts(3)
        result = engine.generate(lob="retail", metrics_df=df, drift_alerts=alerts)
        self.assertIn("3", result.executive_summary)

    def test_fallback_exceptions_list(self):
        engine = CommentaryEngine.__new__(CommentaryEngine)
        engine._client = None
        engine._model = "claude-sonnet-4-20250514"
        engine._max_tokens = 2000

        df = _make_metrics_df()
        alerts = _make_drift_alerts(3)
        result = engine.generate(lob="retail", metrics_df=df, drift_alerts=alerts)
        # Only critical alerts should appear in exceptions
        self.assertTrue(len(result.exceptions) >= 1)
        self.assertIn("series_0", result.exceptions[0])


class TestCommentaryPromptConstruction(unittest.TestCase):
    def test_prompt_contains_metrics(self):
        engine = CommentaryEngine.__new__(CommentaryEngine)
        engine._client = None
        engine._model = "claude-sonnet-4-20250514"
        engine._max_tokens = 2000

        df = _make_metrics_df()
        stats = CommentaryEngine._compute_stats(df)
        prompt = engine._build_prompt("retail", stats, None, None, None, None, None, None)
        self.assertIn("overall_wmape", prompt)
        self.assertIn("retail", prompt)

    def test_prompt_includes_drift_alerts(self):
        engine = CommentaryEngine.__new__(CommentaryEngine)
        engine._client = None
        engine._model = "claude-sonnet-4-20250514"
        engine._max_tokens = 2000

        stats = {"overall_wmape": 0.15}
        alerts = _make_drift_alerts(3)
        prompt = engine._build_prompt("retail", stats, alerts, None, None, None, None, None)
        self.assertIn("critical", prompt)
        self.assertIn("series_0", prompt)


class TestCommentaryResponseParsing(unittest.TestCase):
    def test_parses_mock_response(self):
        engine = CommentaryEngine.__new__(CommentaryEngine)
        engine._client = MagicMock()
        engine._model = "claude-sonnet-4-20250514"
        engine._max_tokens = 2000

        stats = {"overall_wmape": 0.15}
        result = engine._parse_response(_MOCK_COMMENTARY_RESPONSE, stats)

        self.assertIn("retail", result.executive_summary.lower())
        self.assertEqual(len(result.key_metrics), 3)
        self.assertEqual(len(result.exceptions), 3)
        self.assertEqual(len(result.action_items), 3)

    def test_handles_malformed_response(self):
        engine = CommentaryEngine.__new__(CommentaryEngine)
        engine._client = MagicMock()
        engine._model = "claude-sonnet-4-20250514"
        engine._max_tokens = 2000

        stats = {"overall_wmape": 0.15}
        result = engine._parse_response("No valid sections", stats)
        # Falls back to template
        self.assertIsInstance(result, CommentaryResult)


class TestCommentaryMockRoundtrip(unittest.TestCase):
    def test_full_roundtrip(self):
        engine = CommentaryEngine.__new__(CommentaryEngine)
        engine._model = "claude-sonnet-4-20250514"
        engine._max_tokens = 2000

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=_MOCK_COMMENTARY_RESPONSE)]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message
        engine._client = mock_client

        df = _make_metrics_df()
        result = engine.generate(lob="retail", metrics_df=df)

        mock_client.messages.create.assert_called_once()
        self.assertIsInstance(result.executive_summary, str)
        self.assertIn("retail", result.executive_summary.lower())
        self.assertEqual(len(result.key_metrics), 3)

    def test_api_error_falls_back(self):
        engine = CommentaryEngine.__new__(CommentaryEngine)
        engine._model = "claude-sonnet-4-20250514"
        engine._max_tokens = 2000

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API error")
        engine._client = mock_client

        df = _make_metrics_df()
        result = engine.generate(lob="retail", metrics_df=df, drift_alerts=_make_drift_alerts())
        self.assertIn("unavailable", result.executive_summary.lower())


if __name__ == "__main__":
    unittest.main()
