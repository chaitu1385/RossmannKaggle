"""Tests for AnomalyTriageEngine."""

import json
import unittest
from unittest.mock import MagicMock

from src.ai.anomaly_triage import AnomalyTriageEngine, TriagedAlert, TriageResult
from src.metrics.drift import DriftAlert, DriftSeverity

import pytest

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
#  Factory helpers
# --------------------------------------------------------------------------- #

def _make_drift_alerts(n=5):
    """Generate synthetic DriftAlert objects."""
    alerts = []
    for i in range(n):
        severity = DriftSeverity.CRITICAL if i < 2 else DriftSeverity.WARNING
        alerts.append(DriftAlert(
            series_id=f"series_{i}",
            metric="accuracy" if i % 2 == 0 else "bias",
            severity=severity,
            current_value=0.3 + i * 0.05,
            baseline_value=0.15 + i * 0.02,
            message=f"Alert message for series_{i}",
        ))
    return alerts


_MOCK_TRIAGE_RESPONSE = """### RANKED_ALERTS
```json
[
  {"series_id": "series_0", "business_impact_score": 90, "suggested_action": "Retrain model immediately", "reasoning": "High-volume series with critical accuracy drift"},
  {"series_id": "series_1", "business_impact_score": 75, "suggested_action": "Review forecast inputs", "reasoning": "Critical bias drift on key series"},
  {"series_id": "series_2", "business_impact_score": 50, "suggested_action": "Monitor for another week", "reasoning": "Warning-level accuracy degradation"},
  {"series_id": "series_3", "business_impact_score": 30, "suggested_action": "Check data pipeline", "reasoning": "Minor bias shift"},
  {"series_id": "series_4", "business_impact_score": 20, "suggested_action": "No immediate action needed", "reasoning": "Low-severity accuracy warning"}
]
```

### EXECUTIVE_SUMMARY
The LOB 'retail' forecast landscape shows 2 critical and 3 warning-level alerts. Series_0 requires immediate attention due to significant accuracy degradation on a high-volume SKU. Overall forecast health is moderate, with the majority of alerts at warning level."""


# --------------------------------------------------------------------------- #
#  Tests
# --------------------------------------------------------------------------- #

class TestAnomalyTriageInit(unittest.TestCase):
    def test_inherits_base(self):
        engine = AnomalyTriageEngine.__new__(AnomalyTriageEngine)
        engine._client = None
        engine._model = "claude-sonnet-4-20250514"
        engine._max_tokens = 2000
        self.assertFalse(engine.available)


class TestTriageGracefulDegradation(unittest.TestCase):
    def test_empty_alerts(self):
        engine = AnomalyTriageEngine.__new__(AnomalyTriageEngine)
        engine._client = None
        engine._model = "claude-sonnet-4-20250514"
        engine._max_tokens = 2000

        result = engine.query(lob="retail", drift_alerts=[])
        self.assertEqual(result.total_alerts, 0)
        self.assertEqual(result.ranked_alerts, [])

    def test_fallback_when_unavailable(self):
        engine = AnomalyTriageEngine.__new__(AnomalyTriageEngine)
        engine._client = None
        engine._model = "claude-sonnet-4-20250514"
        engine._max_tokens = 2000

        alerts = _make_drift_alerts(3)
        result = engine.query(lob="retail", drift_alerts=alerts)

        self.assertEqual(result.total_alerts, 3)
        self.assertIn("unavailable", result.executive_summary.lower())
        self.assertEqual(len(result.ranked_alerts), 3)
        # Check alerts are in original order
        self.assertEqual(result.ranked_alerts[0].series_id, "series_0")

    def test_fallback_preserves_severity_counts(self):
        engine = AnomalyTriageEngine.__new__(AnomalyTriageEngine)
        engine._client = None
        engine._model = "claude-sonnet-4-20250514"
        engine._max_tokens = 2000

        alerts = _make_drift_alerts(5)
        result = engine.query(lob="retail", drift_alerts=alerts)
        self.assertEqual(result.critical_count, 2)
        self.assertEqual(result.warning_count, 3)


class TestTriagePromptConstruction(unittest.TestCase):
    def test_prompt_contains_alert_data(self):
        engine = AnomalyTriageEngine.__new__(AnomalyTriageEngine)
        engine._client = None
        engine._model = "claude-sonnet-4-20250514"
        engine._max_tokens = 2000

        alerts = _make_drift_alerts(3)
        prompt = engine._build_prompt("retail", alerts, None)
        self.assertIn("series_0", prompt)
        self.assertIn("retail", prompt)
        self.assertIn("accuracy", prompt)

    def test_prompt_includes_series_context(self):
        engine = AnomalyTriageEngine.__new__(AnomalyTriageEngine)
        engine._client = None
        engine._model = "claude-sonnet-4-20250514"
        engine._max_tokens = 2000

        alerts = _make_drift_alerts(2)
        context = {"series_0": {"avg_volume": 1000, "revenue_weight": 0.15}}
        prompt = engine._build_prompt("retail", alerts, context)
        self.assertIn("avg_volume", prompt)


class TestTriageResponseParsing(unittest.TestCase):
    def test_parses_mock_response(self):
        engine = AnomalyTriageEngine.__new__(AnomalyTriageEngine)
        engine._client = MagicMock()
        engine._model = "claude-sonnet-4-20250514"
        engine._max_tokens = 2000

        alerts = _make_drift_alerts(5)
        result = engine._parse_response(_MOCK_TRIAGE_RESPONSE, alerts, 2, 3)

        self.assertEqual(result.total_alerts, 5)
        self.assertEqual(result.critical_count, 2)
        self.assertEqual(len(result.ranked_alerts), 5)
        # First alert should have highest impact score
        self.assertEqual(result.ranked_alerts[0].business_impact_score, 90)
        self.assertIn("retail", result.executive_summary)

    def test_handles_malformed_response(self):
        engine = AnomalyTriageEngine.__new__(AnomalyTriageEngine)
        engine._client = MagicMock()
        engine._model = "claude-sonnet-4-20250514"
        engine._max_tokens = 2000

        alerts = _make_drift_alerts(3)
        result = engine._parse_response("Random text without sections", alerts, 1, 2)
        # Should fallback gracefully
        self.assertEqual(len(result.ranked_alerts), 3)


class TestTriageMockRoundtrip(unittest.TestCase):
    def test_full_roundtrip(self):
        engine = AnomalyTriageEngine.__new__(AnomalyTriageEngine)
        engine._model = "claude-sonnet-4-20250514"
        engine._max_tokens = 2000

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=_MOCK_TRIAGE_RESPONSE)]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message
        engine._client = mock_client

        alerts = _make_drift_alerts(5)
        result = engine.query(lob="retail", drift_alerts=alerts)

        mock_client.messages.create.assert_called_once()
        self.assertEqual(result.total_alerts, 5)
        self.assertIsInstance(result.executive_summary, str)
        self.assertIn("retail", result.executive_summary)
        self.assertEqual(result.ranked_alerts[0].series_id, "series_0")

    def test_api_error_falls_back(self):
        engine = AnomalyTriageEngine.__new__(AnomalyTriageEngine)
        engine._model = "claude-sonnet-4-20250514"
        engine._max_tokens = 2000

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("Rate limited")
        engine._client = mock_client

        alerts = _make_drift_alerts(3)
        result = engine.query(lob="retail", drift_alerts=alerts)
        self.assertIn("unavailable", result.executive_summary.lower())


class TestTriageMaxAlerts(unittest.TestCase):
    def test_respects_max_alerts(self):
        engine = AnomalyTriageEngine.__new__(AnomalyTriageEngine)
        engine._client = None
        engine._model = "claude-sonnet-4-20250514"
        engine._max_tokens = 2000

        alerts = _make_drift_alerts(10)
        result = engine.query(lob="retail", drift_alerts=alerts, max_alerts=3)
        self.assertEqual(result.total_alerts, 3)


if __name__ == "__main__":
    unittest.main()
