"""
LLMAnalyzer — Anthropic Claude integration for data interpretation.

Takes structured analysis results from DataAnalyzer and uses Claude to:
1. Interpret statistical signals in plain English
2. Generate actionable hypotheses for demand planners
3. Explain model selection rationale
4. Identify risk factors
5. Suggest config refinements

Gracefully degrades to no-op when the ``anthropic`` package is not installed
or no API key is configured.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load .env from project root (if present) so ANTHROPIC_API_KEY is available
try:
    from dotenv import load_dotenv
    # Walk up from this file to find project root .env
    _project_root = Path(__file__).resolve().parents[3]  # src/analytics/ → forecasting-product/ → repo root
    for _candidate in [_project_root, _project_root / "forecasting-product"]:
        _env_path = _candidate / ".env"
        if _env_path.exists():
            load_dotenv(_env_path)
            break
except ImportError:
    pass

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Result dataclass
# --------------------------------------------------------------------------- #

@dataclass
class LLMInsight:
    """Structured output from Claude analysis."""

    narrative: str = ""
    hypotheses: List[str] = field(default_factory=list)
    model_rationale: str = ""
    risk_factors: List[str] = field(default_factory=list)
    config_adjustments: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
#  LLMAnalyzer
# --------------------------------------------------------------------------- #

class LLMAnalyzer:
    """Claude-powered interpretation layer for DataAnalyzer reports.

    Parameters
    ----------
    api_key : str, optional
        Anthropic API key.  Falls back to ``ANTHROPIC_API_KEY`` env var.
    model : str
        Claude model to use.  Default is claude-sonnet-4-20250514.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        self._client = None
        self._model = model
        try:
            import anthropic  # noqa: F811
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if key:
                self._client = anthropic.Anthropic(api_key=key)
            else:
                logger.info("LLMAnalyzer: no API key provided, LLM features disabled")
        except ImportError:
            logger.info("LLMAnalyzer: anthropic package not installed, LLM features disabled")

    @property
    def available(self) -> bool:
        """Whether the Claude client is configured and ready."""
        return self._client is not None

    def interpret(self, report: Any) -> LLMInsight:
        """Send analysis report to Claude for interpretation.

        Parameters
        ----------
        report : AnalysisReport
            Output from ``DataAnalyzer.analyze()``.

        Returns
        -------
        LLMInsight
            Structured Claude interpretation.  Returns empty LLMInsight
            if the client is unavailable.
        """
        if not self.available:
            return LLMInsight()

        prompt = self._build_prompt(report)
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=2000,
                system=self._system_prompt(),
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            return self._parse_response(text)
        except Exception as e:
            logger.error(f"LLMAnalyzer: API call failed: {e}")
            return LLMInsight(narrative=f"LLM analysis unavailable: {e}")

    # ----- Prompt construction --------------------------------------------- #

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a senior demand planning and forecasting expert advising a "
            "retail S&OP team. You are analyzing automated statistical profiling "
            "results for a dataset that will be used in a forecasting product. "
            "Your audience includes both data scientists and non-technical VP-level "
            "stakeholders. Be specific, actionable, and concise. Use bullet points."
        )

    def _build_prompt(self, report: Any) -> str:
        """Construct the analysis prompt with all statistical context."""
        schema = report.schema
        fc = report.forecastability
        config = report.recommended_config
        hierarchy = report.hierarchy

        # Build context dict
        context = {
            "dataset": {
                "n_series": schema.n_series,
                "n_rows": schema.n_rows,
                "date_range": list(schema.date_range),
                "frequency": schema.frequency_guess,
                "time_column": schema.time_column,
                "target_column": schema.target_column,
                "id_columns": schema.id_columns,
                "dimension_columns": schema.dimension_columns,
                "numeric_columns": schema.numeric_columns,
            },
            "hierarchy": {
                "detected": [
                    {"name": h.name, "levels": h.levels, "fixed": h.fixed}
                    for h in hierarchy.hierarchies
                ],
                "reasoning": hierarchy.reasoning,
            },
            "forecastability": {
                "overall_score": round(fc.overall_score, 3),
                "score_distribution": fc.score_distribution,
                "demand_class_distribution": fc.demand_class_distribution,
            },
            "statistical_hypotheses": report.hypotheses,
            "recommended_config": {
                "forecasters": config.forecast.forecasters,
                "intermittent_forecasters": config.forecast.intermittent_forecasters,
                "horizon_weeks": config.forecast.horizon_weeks,
                "n_folds": config.backtest.n_folds,
                "reconciliation_method": config.reconciliation.method,
                "cleansing_enabled": config.data_quality.cleansing.enabled,
            },
            "config_reasoning": report.config_reasoning,
        }

        # Add per-series summary if available
        if fc.per_series is not None:
            ps = fc.per_series
            summary_stats = {
                "cv_mean": round(float(ps["cv"].mean()), 3),
                "cv_median": round(float(ps["cv"].median()), 3),
                "apen_mean": round(float(ps["apen"].mean()), 3),
                "spectral_entropy_mean": round(float(ps["spectral_entropy"].mean()), 3),
                "snr_mean": round(float(ps["snr"].mean()), 3),
                "trend_strength_mean": round(float(ps["trend_strength"].mean()), 3),
                "seasonal_strength_mean": round(float(ps["seasonal_strength"].mean()), 3),
                "forecastability_mean": round(float(ps["forecastability_score"].mean()), 3),
                "forecastability_min": round(float(ps["forecastability_score"].min()), 3),
                "forecastability_max": round(float(ps["forecastability_score"].max()), 3),
            }
            context["signal_summary"] = summary_stats

        context_json = json.dumps(context, indent=2, default=str)

        return f"""Here are the automated analysis results for a retail forecasting dataset.
Please provide your expert interpretation.

## Analysis Context
```json
{context_json}
```

## Instructions
Respond with these exact section headers:

### NARRATIVE
A 3-5 sentence executive summary suitable for a VP presentation. Cover the data's
forecastability, key patterns, and recommended approach.

### HYPOTHESES
Bulleted list of 3-6 actionable data hypotheses. Go beyond the statistical
hypotheses already generated — add business context and cross-signal insights.

### MODEL_RATIONALE
2-3 sentences explaining why the recommended model set is appropriate for this
specific dataset. Reference the forecastability signals.

### RISK_FACTORS
Bulleted list of 2-4 risks or caveats the team should be aware of.

### CONFIG_ADJUSTMENTS
Bulleted list of 0-3 specific config adjustments you would suggest beyond
what was auto-generated. Only include if genuinely warranted by the data."""

    # ----- Response parsing ------------------------------------------------ #

    @staticmethod
    def _parse_response(text: str) -> LLMInsight:
        """Parse Claude's section-headed response into structured fields."""
        insight = LLMInsight()

        sections = {
            "NARRATIVE": "",
            "HYPOTHESES": "",
            "MODEL_RATIONALE": "",
            "RISK_FACTORS": "",
            "CONFIG_ADJUSTMENTS": "",
        }

        current_section = None
        lines = text.split("\n")
        for line in lines:
            stripped = line.strip()
            # Check for section headers
            for key in sections:
                if key in stripped.upper() and stripped.startswith("#"):
                    current_section = key
                    break
            else:
                if current_section is not None:
                    sections[current_section] += line + "\n"

        insight.narrative = sections["NARRATIVE"].strip()
        insight.model_rationale = sections["MODEL_RATIONALE"].strip()

        # Parse bulleted lists
        insight.hypotheses = _parse_bullets(sections["HYPOTHESES"])
        insight.risk_factors = _parse_bullets(sections["RISK_FACTORS"])
        insight.config_adjustments = _parse_bullets(sections["CONFIG_ADJUSTMENTS"])

        return insight


def _parse_bullets(text: str) -> List[str]:
    """Extract bullet points from markdown text."""
    bullets = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith(("-", "*", "•")):
            content = line.lstrip("-*• ").strip()
            if content:
                bullets.append(content)
    return bullets
