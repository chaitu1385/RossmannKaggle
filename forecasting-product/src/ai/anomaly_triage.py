"""Anomaly Triage — rank drift alerts by business impact using Claude.

Takes a list of ``DriftAlert`` objects from ``ForecastDriftDetector`` and
uses Claude to:
1. Rank alerts by estimated business impact
2. Suggest specific actions for each alert
3. Generate an executive summary

Gracefully degrades to returning alerts in their original severity order
when Claude is unavailable.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import AIFeatureBase

logger = logging.getLogger(__name__)


@dataclass
class TriagedAlert:
    """A drift alert enriched with business impact scoring and suggested actions."""
    series_id: str
    metric: str
    severity: str
    business_impact_score: float = 0.0   # 0-100
    suggested_action: str = ""
    reasoning: str = ""
    original_message: str = ""


@dataclass
class TriageResult:
    """Result of anomaly triage analysis."""
    ranked_alerts: List[TriagedAlert] = field(default_factory=list)
    executive_summary: str = ""
    total_alerts: int = 0
    critical_count: int = 0
    warning_count: int = 0


class AnomalyTriageEngine(AIFeatureBase):
    """Claude-powered anomaly triage for drift alerts.

    Parameters
    ----------
    api_key : str, optional
        Anthropic API key.
    model : str
        Claude model to use.
    """

    def query(
        self,
        lob: str,
        drift_alerts: List[Any],
        series_context: Optional[Dict[str, Dict[str, float]]] = None,
        max_alerts: int = 50,
    ) -> TriageResult:
        """Triage drift alerts by business impact.

        Parameters
        ----------
        lob : str
            Line of business identifier.
        drift_alerts : list of DriftAlert
            Raw alerts from ``ForecastDriftDetector.detect()``.
        series_context : dict, optional
            Maps series_id → {"avg_volume": ..., "revenue_weight": ...} for
            business-impact scoring.
        max_alerts : int
            Maximum number of alerts to include in the prompt.

        Returns
        -------
        TriageResult
            Ranked alerts with business impact scores and suggested actions.
        """
        if not drift_alerts:
            return TriageResult()

        alerts_to_process = drift_alerts[:max_alerts]
        critical_count = sum(1 for a in alerts_to_process if a.severity.value == "critical")
        warning_count = sum(1 for a in alerts_to_process if a.severity.value == "warning")

        if not self.available:
            return self._fallback_result(alerts_to_process, critical_count, warning_count)

        try:
            prompt = self._build_prompt(lob, alerts_to_process, series_context)
            text = self._call_claude(self._system_prompt(), prompt)
            return self._parse_response(text, alerts_to_process, critical_count, warning_count)
        except Exception as e:
            logger.error("AnomalyTriageEngine: API call failed: %s", e)
            return self._fallback_result(alerts_to_process, critical_count, warning_count)

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a demand planning triage analyst for a retail S&OP team. "
            "You are analyzing forecast drift alerts and must rank them by business "
            "impact, suggest corrective actions, and provide an executive summary. "
            "Be specific, actionable, and concise."
        )

    def _build_prompt(
        self,
        lob: str,
        alerts: List[Any],
        series_context: Optional[Dict[str, Dict[str, float]]],
    ) -> str:
        alert_data = []
        for a in alerts:
            entry: Dict[str, Any] = {
                "series_id": a.series_id,
                "metric": a.metric,
                "severity": a.severity.value,
                "current_value": round(a.current_value, 4),
                "baseline_value": round(a.baseline_value, 4),
                "message": a.message,
            }
            if series_context and a.series_id in series_context:
                entry["context"] = series_context[a.series_id]
            alert_data.append(entry)

        alerts_json = json.dumps(alert_data, indent=2)

        return f"""Triage the following {len(alerts)} drift alerts for LOB '{lob}'.

## Alerts
```json
{alerts_json}
```

## Instructions
Respond with these exact section headers:

### RANKED_ALERTS
A JSON array where each element has:
- "series_id": string
- "business_impact_score": number (0-100, based on severity + volume context)
- "suggested_action": string (specific corrective action)
- "reasoning": string (1 sentence explaining the ranking)

Sort by business_impact_score descending.

### EXECUTIVE_SUMMARY
A 3-5 sentence paragraph summarizing the alert landscape for a VP audience.
Highlight the most urgent items and overall forecast health."""

    def _parse_response(
        self,
        text: str,
        original_alerts: List[Any],
        critical_count: int,
        warning_count: int,
    ) -> TriageResult:
        sections = self._parse_sections(text, ["RANKED_ALERTS", "EXECUTIVE_SUMMARY"])

        executive_summary = sections.get("EXECUTIVE_SUMMARY", "")

        # Parse ranked alerts JSON
        ranked_data = self._parse_json_block(sections.get("RANKED_ALERTS", ""))

        # Build a lookup from original alerts
        alert_lookup = {a.series_id: a for a in original_alerts}

        ranked_alerts = []
        if isinstance(ranked_data, list):
            for item in ranked_data:
                sid = item.get("series_id", "")
                orig = alert_lookup.get(sid)
                if orig:
                    ranked_alerts.append(TriagedAlert(
                        series_id=sid,
                        metric=orig.metric,
                        severity=orig.severity.value,
                        business_impact_score=float(item.get("business_impact_score", 0)),
                        suggested_action=item.get("suggested_action", ""),
                        reasoning=item.get("reasoning", ""),
                        original_message=orig.message,
                    ))

        # Add any alerts not returned by Claude
        seen = {a.series_id for a in ranked_alerts}
        for orig in original_alerts:
            if orig.series_id not in seen:
                ranked_alerts.append(TriagedAlert(
                    series_id=orig.series_id,
                    metric=orig.metric,
                    severity=orig.severity.value,
                    original_message=orig.message,
                ))

        if not ranked_alerts:
            return self._fallback_result(original_alerts, critical_count, warning_count)

        return TriageResult(
            ranked_alerts=ranked_alerts,
            executive_summary=executive_summary,
            total_alerts=len(original_alerts),
            critical_count=critical_count,
            warning_count=warning_count,
        )

    @staticmethod
    def _fallback_result(
        alerts: List[Any],
        critical_count: int,
        warning_count: int,
    ) -> TriageResult:
        """Return alerts in original order when Claude is unavailable."""
        ranked = [
            TriagedAlert(
                series_id=a.series_id,
                metric=a.metric,
                severity=a.severity.value,
                original_message=a.message,
            )
            for a in alerts
        ]
        summary = (
            f"{len(alerts)} drift alerts detected "
            f"({critical_count} critical, {warning_count} warning). "
            "AI analysis unavailable — alerts shown in default severity order."
        )
        return TriageResult(
            ranked_alerts=ranked,
            executive_summary=summary,
            total_alerts=len(alerts),
            critical_count=critical_count,
            warning_count=warning_count,
        )
