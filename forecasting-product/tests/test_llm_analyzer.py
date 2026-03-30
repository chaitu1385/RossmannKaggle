"""Tests for LLMAnalyzer — Anthropic Claude integration."""

import unittest
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl

from src.analytics.analyzer import DataAnalyzer
from src.analytics.llm_analyzer import LLMAnalyzer, LLMInsight, _parse_bullets

import pytest

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
#  Factory helpers
# --------------------------------------------------------------------------- #

def _make_flat_data(n_series=5, n_weeks=52, seed=42):
    rng = np.random.RandomState(seed)
    rows = []
    base = date(2022, 1, 3)
    for i in range(n_series):
        sid = f"series_{i}"
        for w in range(n_weeks):
            rows.append({
                "series_id": sid,
                "week": base + timedelta(weeks=w),
                "quantity": float(rng.normal(100, 20)),
            })
    return pl.DataFrame(rows)


def _make_analysis_report():
    """Create a real AnalysisReport from synthetic data."""
    df = _make_flat_data()
    analyzer = DataAnalyzer(lob_name="test")
    return analyzer.analyze(df)


_MOCK_RESPONSE = """### NARRATIVE
This dataset contains 5 weekly time series spanning 52 weeks with moderate
forecastability. The data shows mixed seasonal patterns with some series
exhibiting strong weekly cycles while others are more noisy. A statistical
model approach is recommended as the primary strategy.

### HYPOTHESES
- Series with high seasonal strength (>0.5) are likely driven by recurring promotions or calendar events
- The moderate CV across series suggests stable base demand with periodic fluctuations
- Low approximate entropy indicates the series follow repeatable patterns amenable to time series models
- External regressors are not available, limiting the ability to capture exogenous drivers

### MODEL_RATIONALE
The auto_arima and auto_ets models are well-suited because the forecastability
signals show moderate-to-strong seasonal patterns with low entropy. The naive
seasonal baseline provides a sanity check.

### RISK_FACTORS
- Only 52 weeks of data limits the ability to capture year-over-year patterns
- No external regressors means the models cannot account for promotions or pricing changes
- The small number of series (5) makes cross-learning ML models less effective

### CONFIG_ADJUSTMENTS
- Consider extending the backtest to 3 folds instead of 2 for more robust champion selection
- Add MSTL to the model set for better multi-seasonal decomposition
"""


# --------------------------------------------------------------------------- #
#  Tests: Initialization
# --------------------------------------------------------------------------- #

class TestLLMAnalyzerInit(unittest.TestCase):
    def test_available_with_client(self):
        analyzer = LLMAnalyzer.__new__(LLMAnalyzer)
        analyzer._model = "claude-sonnet-4-20250514"
        analyzer._client = MagicMock()
        self.assertTrue(analyzer.available)

    def test_unavailable_without_client(self):
        analyzer = LLMAnalyzer.__new__(LLMAnalyzer)
        analyzer._model = "claude-sonnet-4-20250514"
        analyzer._client = None
        self.assertFalse(analyzer.available)

    def test_custom_model(self):
        analyzer = LLMAnalyzer.__new__(LLMAnalyzer)
        analyzer._client = None
        analyzer._model = "claude-opus-4-20250514"
        self.assertEqual(analyzer._model, "claude-opus-4-20250514")


# --------------------------------------------------------------------------- #
#  Tests: Prompt Construction
# --------------------------------------------------------------------------- #

class TestPromptConstruction(unittest.TestCase):
    def setUp(self):
        self.analyzer = LLMAnalyzer.__new__(LLMAnalyzer)
        self.analyzer._client = None
        self.analyzer._model = "claude-sonnet-4-20250514"
        self.report = _make_analysis_report()

    def test_prompt_contains_schema(self):
        prompt = self.analyzer._build_prompt(self.report)
        self.assertIn("n_series", prompt)
        self.assertIn("date_range", prompt)
        self.assertIn("frequency", prompt)

    def test_prompt_contains_forecastability(self):
        prompt = self.analyzer._build_prompt(self.report)
        self.assertIn("forecastability", prompt.lower())
        self.assertIn("overall_score", prompt)

    def test_prompt_contains_config(self):
        prompt = self.analyzer._build_prompt(self.report)
        self.assertIn("forecasters", prompt)
        self.assertIn("horizon_weeks", prompt)

    def test_prompt_contains_hypotheses(self):
        prompt = self.analyzer._build_prompt(self.report)
        self.assertIn("statistical_hypotheses", prompt)

    def test_prompt_has_section_instructions(self):
        prompt = self.analyzer._build_prompt(self.report)
        for section in ["NARRATIVE", "HYPOTHESES", "MODEL_RATIONALE", "RISK_FACTORS"]:
            self.assertIn(section, prompt)


# --------------------------------------------------------------------------- #
#  Tests: Response Parsing
# --------------------------------------------------------------------------- #

class TestResponseParsing(unittest.TestCase):
    def test_parses_all_sections(self):
        insight = LLMAnalyzer._parse_response(_MOCK_RESPONSE)
        self.assertIsInstance(insight.narrative, str)
        self.assertIn("5 weekly time series", insight.narrative)
        self.assertEqual(len(insight.hypotheses), 4)
        self.assertIsInstance(insight.model_rationale, str)
        self.assertIn("auto_arima", insight.model_rationale)
        self.assertEqual(len(insight.risk_factors), 3)
        self.assertEqual(len(insight.config_adjustments), 2)

    def test_correct_hypothesis_count(self):
        insight = LLMAnalyzer._parse_response(_MOCK_RESPONSE)
        self.assertEqual(len(insight.hypotheses), 4)

    def test_correct_risk_count(self):
        insight = LLMAnalyzer._parse_response(_MOCK_RESPONSE)
        self.assertEqual(len(insight.risk_factors), 3)

    def test_handles_empty_response(self):
        insight = LLMAnalyzer._parse_response("")
        self.assertEqual(insight.narrative, "")
        self.assertEqual(insight.hypotheses, [])

    def test_handles_malformed_response(self):
        insight = LLMAnalyzer._parse_response("Just some random text without headers")
        # Should not crash, just return empty sections
        self.assertIsInstance(insight, LLMInsight)


# --------------------------------------------------------------------------- #
#  Tests: Bullet Parsing Helper
# --------------------------------------------------------------------------- #

class TestParseBullets(unittest.TestCase):
    def test_dash_bullets(self):
        text = "- First\n- Second\n- Third"
        self.assertEqual(_parse_bullets(text), ["First", "Second", "Third"])

    def test_star_bullets(self):
        text = "* Alpha\n* Beta"
        self.assertEqual(_parse_bullets(text), ["Alpha", "Beta"])

    def test_mixed_bullets(self):
        text = "- One\n* Two\n• Three"
        self.assertEqual(len(_parse_bullets(text)), 3)

    def test_empty_string(self):
        self.assertEqual(_parse_bullets(""), [])


# --------------------------------------------------------------------------- #
#  Tests: Integration with Mock Client
# --------------------------------------------------------------------------- #

class TestLLMIntegration(unittest.TestCase):
    def test_mock_roundtrip(self):
        """Full mock roundtrip: build prompt, call API, parse response."""
        analyzer = LLMAnalyzer.__new__(LLMAnalyzer)
        analyzer._model = "claude-sonnet-4-20250514"

        # Mock the Anthropic client
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=_MOCK_RESPONSE)]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message
        analyzer._client = mock_client

        report = _make_analysis_report()
        insight = analyzer.interpret(report)

        # Verify API was called
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "claude-sonnet-4-20250514")
        self.assertIn("max_tokens", call_kwargs)

        # Verify parsed output
        self.assertIsInstance(insight.narrative, str)
        self.assertIn("5 weekly time series", insight.narrative)
        self.assertEqual(len(insight.hypotheses), 4)

    def test_noop_when_unavailable(self):
        """Returns empty LLMInsight when client is not configured."""
        analyzer = LLMAnalyzer.__new__(LLMAnalyzer)
        analyzer._client = None
        analyzer._model = "claude-sonnet-4-20250514"

        report = _make_analysis_report()
        insight = analyzer.interpret(report)

        self.assertEqual(insight.narrative, "")
        self.assertEqual(insight.hypotheses, [])

    def test_api_error_handling(self):
        """Returns graceful error message on API failure."""
        analyzer = LLMAnalyzer.__new__(LLMAnalyzer)
        analyzer._model = "claude-sonnet-4-20250514"
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("Rate limited")
        analyzer._client = mock_client

        report = _make_analysis_report()
        insight = analyzer.interpret(report)

        self.assertIn("unavailable", insight.narrative.lower())
