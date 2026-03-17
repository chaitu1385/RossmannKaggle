"""
AlertDispatcher — routes drift alerts to configured channels.

Connects ``ForecastDriftDetector`` output to operational alerting:
  - ``"log"`` — structured log warnings (always available).
  - ``"webhook"`` — POST to Teams / Slack / PagerDuty / generic webhook.

Usage
-----
>>> from src.observability.alerts import AlertDispatcher, AlertConfig
>>> from src.metrics.drift import ForecastDriftDetector, DriftConfig
>>> detector = ForecastDriftDetector(DriftConfig())
>>> alerts = detector.detect(metrics_df)
>>> dispatcher = AlertDispatcher(AlertConfig(channels=["log", "webhook"],
...                                          webhook_url="https://hooks.slack.com/..."))
>>> dispatcher.dispatch(alerts)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AlertConfig:
    """
    Alert routing configuration.

    Attributes
    ----------
    channels:
        List of alert channels: ``"log"`` and/or ``"webhook"``.
    webhook_url:
        Target URL for webhook alerts (Slack, Teams, PagerDuty, etc.).
    min_severity:
        Minimum severity to dispatch: ``"warning"`` or ``"critical"``.
    webhook_timeout:
        HTTP timeout in seconds for webhook calls.
    """
    channels: List[str] = field(default_factory=lambda: ["log"])
    webhook_url: str = ""
    min_severity: str = "warning"
    webhook_timeout: int = 10


class AlertDispatcher:
    """
    Routes drift alerts to configured channels.

    Parameters
    ----------
    config:
        ``AlertConfig`` instance.
    """

    # Severity ordering for min_severity filtering
    _SEVERITY_ORDER = {"warning": 0, "critical": 1}

    def __init__(self, config: Optional[AlertConfig] = None):
        self.config = config or AlertConfig()
        self._min_severity_level = self._SEVERITY_ORDER.get(
            self.config.min_severity, 0
        )
        self._dispatched_count = 0

    def dispatch(self, alerts) -> int:
        """
        Dispatch a list of drift alerts to all configured channels.

        Parameters
        ----------
        alerts:
            List of ``DriftAlert`` objects (from ``ForecastDriftDetector.detect()``).

        Returns
        -------
        int — number of alerts dispatched.
        """
        count = 0
        for alert in alerts:
            severity_level = self._SEVERITY_ORDER.get(
                alert.severity.value if hasattr(alert.severity, "value") else str(alert.severity),
                0,
            )
            if severity_level < self._min_severity_level:
                continue

            if "log" in self.config.channels:
                self._log_alert(alert)

            if "webhook" in self.config.channels and self.config.webhook_url:
                self._send_webhook(alert)

            count += 1

        self._dispatched_count += count
        if count:
            logger.info("Dispatched %d alerts across channels: %s",
                        count, self.config.channels)
        return count

    @property
    def dispatched_count(self) -> int:
        """Total alerts dispatched since this dispatcher was created."""
        return self._dispatched_count

    # ── Channel implementations ───────────────────────────────────────────────

    def _log_alert(self, alert) -> None:
        """Emit a structured log warning for the alert."""
        severity = (
            alert.severity.value
            if hasattr(alert.severity, "value")
            else str(alert.severity)
        )
        record = {
            "alert": True,
            "series_id": alert.series_id,
            "metric": alert.metric,
            "severity": severity,
            "current_value": alert.current_value,
            "baseline_value": alert.baseline_value,
            "message": alert.message,
        }
        logger.warning(json.dumps(record, default=str))

    def _send_webhook(self, alert) -> None:
        """POST alert payload to the configured webhook URL."""
        import urllib.request

        severity = (
            alert.severity.value
            if hasattr(alert.severity, "value")
            else str(alert.severity)
        )
        payload = json.dumps({
            "text": str(alert),
            "severity": severity,
            "series_id": alert.series_id,
            "metric": alert.metric,
            "current_value": alert.current_value,
            "baseline_value": alert.baseline_value,
        }).encode()

        req = urllib.request.Request(
            self.config.webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=self.config.webhook_timeout)
            logger.debug("Webhook sent for %s/%s", alert.series_id, alert.metric)
        except Exception as exc:
            logger.error(
                "Webhook delivery failed for %s/%s: %s",
                alert.series_id, alert.metric, exc,
            )
