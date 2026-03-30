"""Tests for ConfigTunerEngine."""

import json
import unittest
from unittest.mock import MagicMock

import polars as pl

from src.ai.config_tuner import ConfigRecommendation, ConfigTunerEngine, ConfigTuningResult
from src.config.schema import PlatformConfig

import pytest

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
#  Factory helpers
# --------------------------------------------------------------------------- #

def _make_leaderboard():
    return pl.DataFrame({
        "model_id": ["auto_arima", "auto_ets", "lgbm_direct", "naive_seasonal"],
        "wmape": [0.12, 0.14, 0.11, 0.25],
        "normalized_bias": [0.02, -0.01, 0.03, 0.05],
        "rank": [2, 3, 1, 4],
        "n_series": [50, 50, 50, 50],
    })


_MOCK_CONFIG_RESPONSE = """### RECOMMENDATIONS
```json
[
  {
    "field_path": "forecast.forecasters",
    "current_value": ["naive_seasonal"],
    "recommended_value": ["naive_seasonal", "auto_arima", "lgbm_direct"],
    "reasoning": "LightGBM shows best WMAPE (0.11) and AutoARIMA is competitive. Adding both provides robust champion selection.",
    "expected_impact": "5-10% WMAPE improvement over naive baseline",
    "risk": "low"
  },
  {
    "field_path": "backtest.n_folds",
    "current_value": 3,
    "recommended_value": 4,
    "reasoning": "More folds improve champion selection robustness with the current data volume.",
    "expected_impact": "More stable model selection",
    "risk": "low"
  }
]
```

### OVERALL_ASSESSMENT
The current config uses only the naive seasonal baseline, leaving significant accuracy gains on the table. The leaderboard shows LightGBM and AutoARIMA both outperform the baseline by 10+ percentage points. Adding these models is a low-risk, high-impact change.

### RISK_SUMMARY
Both recommended changes are low risk. Adding models increases runtime but improves accuracy. Increasing folds adds marginal compute cost."""


# --------------------------------------------------------------------------- #
#  Tests
# --------------------------------------------------------------------------- #

class TestConfigTunerInit(unittest.TestCase):
    def test_inherits_base(self):
        engine = ConfigTunerEngine.__new__(ConfigTunerEngine)
        engine._client = None
        engine._model = "claude-sonnet-4-20250514"
        engine._max_tokens = 2000
        self.assertFalse(engine.available)


class TestConfigTunerGracefulDegradation(unittest.TestCase):
    def test_returns_unavailable_when_no_client(self):
        engine = ConfigTunerEngine.__new__(ConfigTunerEngine)
        engine._client = None
        engine._model = "claude-sonnet-4-20250514"
        engine._max_tokens = 2000

        config = PlatformConfig(lob="retail")
        result = engine.recommend(lob="retail", current_config=config)
        self.assertIn("unavailable", result.overall_assessment.lower())
        self.assertEqual(result.recommendations, [])


class TestConfigTunerPromptConstruction(unittest.TestCase):
    def test_prompt_contains_config(self):
        engine = ConfigTunerEngine.__new__(ConfigTunerEngine)
        engine._client = None
        engine._model = "claude-sonnet-4-20250514"
        engine._max_tokens = 2000

        config = PlatformConfig(lob="retail")
        prompt = engine._build_prompt("retail", config, None, None, None, None)
        self.assertIn("naive_seasonal", prompt)
        self.assertIn("retail", prompt)

    def test_prompt_includes_leaderboard(self):
        engine = ConfigTunerEngine.__new__(ConfigTunerEngine)
        engine._client = None
        engine._model = "claude-sonnet-4-20250514"
        engine._max_tokens = 2000

        config = PlatformConfig(lob="retail")
        leaderboard = _make_leaderboard()
        prompt = engine._build_prompt("retail", config, leaderboard, None, None, None)
        self.assertIn("auto_arima", prompt)
        self.assertIn("lgbm_direct", prompt)


class TestConfigTunerResponseParsing(unittest.TestCase):
    def test_parses_mock_response(self):
        engine = ConfigTunerEngine.__new__(ConfigTunerEngine)
        engine._client = MagicMock()
        engine._model = "claude-sonnet-4-20250514"
        engine._max_tokens = 2000

        result = engine._parse_response(_MOCK_CONFIG_RESPONSE)

        self.assertEqual(len(result.recommendations), 2)
        self.assertEqual(result.recommendations[0].field_path, "forecast.forecasters")
        self.assertEqual(result.recommendations[0].risk, "low")
        self.assertIn("naive seasonal", result.overall_assessment.lower())
        self.assertIn("low risk", result.risk_summary.lower())

    def test_handles_malformed_response(self):
        engine = ConfigTunerEngine.__new__(ConfigTunerEngine)
        engine._client = MagicMock()
        engine._model = "claude-sonnet-4-20250514"
        engine._max_tokens = 2000

        result = engine._parse_response("No valid sections or JSON")
        self.assertIsInstance(result, ConfigTuningResult)
        self.assertEqual(result.recommendations, [])


class TestConfigTunerMockRoundtrip(unittest.TestCase):
    def test_full_roundtrip(self):
        engine = ConfigTunerEngine.__new__(ConfigTunerEngine)
        engine._model = "claude-sonnet-4-20250514"
        engine._max_tokens = 2000

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=_MOCK_CONFIG_RESPONSE)]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message
        engine._client = mock_client

        config = PlatformConfig(lob="retail")
        leaderboard = _make_leaderboard()
        result = engine.recommend(
            lob="retail",
            current_config=config,
            leaderboard=leaderboard,
        )

        mock_client.messages.create.assert_called_once()
        self.assertEqual(len(result.recommendations), 2)
        self.assertIsInstance(result.overall_assessment, str)
        self.assertIn("naive seasonal", result.overall_assessment.lower())

    def test_api_error_returns_graceful_result(self):
        engine = ConfigTunerEngine.__new__(ConfigTunerEngine)
        engine._model = "claude-sonnet-4-20250514"
        engine._max_tokens = 2000

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API error")
        engine._client = mock_client

        config = PlatformConfig(lob="retail")
        result = engine.recommend(lob="retail", current_config=config)
        self.assertIn("unavailable", result.overall_assessment.lower())


if __name__ == "__main__":
    unittest.main()
