"""Automated Config Tuning — recommend configuration changes using Claude.

Analyzes backtest performance (leaderboard, FVA, forecastability) and
the current platform configuration to suggest specific config changes
with reasoning, expected impact, and risk assessment.

Gracefully degrades to empty recommendations when Claude is unavailable.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import polars as pl

from .base import AIFeatureBase

logger = logging.getLogger(__name__)


@dataclass
class ConfigRecommendation:
    """A single configuration change recommendation."""
    field_path: str                      # e.g. "reconciliation.method"
    current_value: Any = None
    recommended_value: Any = None
    reasoning: str = ""
    expected_impact: str = ""
    risk: str = "low"                    # "low" | "medium" | "high"


@dataclass
class ConfigTuningResult:
    """Result of config tuning analysis."""
    recommendations: List[ConfigRecommendation] = field(default_factory=list)
    overall_assessment: str = ""
    risk_summary: str = ""


class ConfigTunerEngine(AIFeatureBase):
    """Claude-powered configuration tuning recommendations.

    Parameters
    ----------
    api_key : str, optional
        Anthropic API key.
    model : str
        Claude model to use.
    """

    def recommend(
        self,
        lob: str,
        current_config: Any,
        leaderboard: Optional[pl.DataFrame] = None,
        fva_summary: Optional[pl.DataFrame] = None,
        champion_table: Optional[pl.DataFrame] = None,
        forecastability: Optional[Any] = None,
    ) -> ConfigTuningResult:
        """Generate config recommendations from backtest results.

        Parameters
        ----------
        lob : str
            Line of business.
        current_config : PlatformConfig
            Current platform configuration dataclass.
        leaderboard : pl.DataFrame, optional
            Model leaderboard from MetricStore.
        fva_summary : pl.DataFrame, optional
            FVA summary from FVAAnalyzer.
        champion_table : pl.DataFrame, optional
            Champion selection results.
        forecastability : ForecastabilityReport, optional
            Forecastability analysis results.

        Returns
        -------
        ConfigTuningResult
        """
        if not self.available:
            return ConfigTuningResult(overall_assessment="AI analysis unavailable — Claude client not configured.")

        try:
            prompt = self._build_prompt(lob, current_config, leaderboard, fva_summary, champion_table, forecastability)
            text = self._call_claude(self._system_prompt(), prompt, max_tokens=3000)
            return self._parse_response(text)
        except Exception as e:
            logger.error("ConfigTunerEngine: API call failed: %s", e)
            return ConfigTuningResult(overall_assessment=f"AI analysis unavailable: {e}")

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a forecasting platform configuration expert. You analyze "
            "backtest performance data and recommend specific configuration changes "
            "to improve forecast accuracy. Be conservative — only recommend changes "
            "with clear evidence. Specify exact YAML paths and values."
        )

    def _build_prompt(
        self,
        lob: str,
        current_config: Any,
        leaderboard: Optional[pl.DataFrame],
        fva_summary: Optional[pl.DataFrame],
        champion_table: Optional[pl.DataFrame],
        forecastability: Optional[Any],
    ) -> str:
        parts = [f"Recommend config changes for LOB '{lob}'."]

        # Serialize current config
        try:
            config_dict = {
                "forecast": {
                    "forecasters": current_config.forecast.forecasters,
                    "horizon_periods": current_config.forecast.horizon_periods,
                    "frequency": current_config.forecast.frequency,
                    "intermittent_forecasters": current_config.forecast.intermittent_forecasters,
                    "sparse_detection": current_config.forecast.sparse_detection,
                },
                "backtest": {
                    "n_folds": current_config.backtest.n_folds,
                    "val_periods": current_config.backtest.val_periods,
                    "champion_granularity": current_config.backtest.champion_granularity,
                    "primary_metric": current_config.backtest.primary_metric,
                    "selection_strategy": current_config.backtest.selection_strategy,
                },
                "reconciliation": {
                    "method": current_config.reconciliation.method,
                },
                "data_quality": {
                    "cleansing_enabled": current_config.data_quality.cleansing.enabled,
                    "min_series_length": current_config.data_quality.min_series_length,
                },
            }
            parts.append(f"\n## Current Config\n```json\n{json.dumps(config_dict, indent=2)}\n```")
        except Exception:
            parts.append("\n## Current Config\nUnable to serialize — default configuration assumed.")

        if leaderboard is not None and not leaderboard.is_empty():
            rows = leaderboard.head(10).to_dicts()
            parts.append(f"\n## Model Leaderboard\n```json\n{json.dumps(rows, indent=2, default=str)}\n```")

        if fva_summary is not None and not fva_summary.is_empty():
            rows = fva_summary.to_dicts()
            parts.append(f"\n## FVA Summary\n```json\n{json.dumps(rows, indent=2, default=str)}\n```")

        if champion_table is not None and not champion_table.is_empty():
            rows = champion_table.head(10).to_dicts()
            parts.append(f"\n## Champion Selection\n```json\n{json.dumps(rows, indent=2, default=str)}\n```")

        if forecastability is not None:
            try:
                fc_data = {
                    "overall_score": forecastability.overall_score,
                    "score_distribution": forecastability.score_distribution,
                    "demand_class_distribution": forecastability.demand_class_distribution,
                }
                parts.append(f"\n## Forecastability\n```json\n{json.dumps(fc_data, indent=2, default=str)}\n```")
            except Exception:
                pass

        parts.append("""
## Instructions
Respond with these exact section headers:

### RECOMMENDATIONS
A JSON array where each element has:
- "field_path": string (e.g. "forecast.forecasters")
- "current_value": current setting
- "recommended_value": suggested new value
- "reasoning": string (1-2 sentences)
- "expected_impact": string (e.g. "2-5% WMAPE improvement")
- "risk": "low" | "medium" | "high"

Only include recommendations with clear evidence from the data.

### OVERALL_ASSESSMENT
2-3 sentences assessing the current config's effectiveness.

### RISK_SUMMARY
1-2 sentences about the overall risk of the recommended changes.""")

        return "\n".join(parts)

    def _parse_response(self, text: str) -> ConfigTuningResult:
        sections = self._parse_sections(
            text, ["RECOMMENDATIONS", "OVERALL_ASSESSMENT", "RISK_SUMMARY"]
        )

        overall_assessment = sections.get("OVERALL_ASSESSMENT", "")
        risk_summary = sections.get("RISK_SUMMARY", "")

        recommendations = []
        rec_data = self._parse_json_block(sections.get("RECOMMENDATIONS", ""))
        if isinstance(rec_data, list):
            for item in rec_data:
                recommendations.append(ConfigRecommendation(
                    field_path=item.get("field_path", ""),
                    current_value=item.get("current_value"),
                    recommended_value=item.get("recommended_value"),
                    reasoning=item.get("reasoning", ""),
                    expected_impact=item.get("expected_impact", ""),
                    risk=item.get("risk", "low"),
                ))

        return ConfigTuningResult(
            recommendations=recommendations,
            overall_assessment=overall_assessment,
            risk_summary=risk_summary,
        )
