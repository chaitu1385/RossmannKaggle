"""
MetricsEmitter — pipeline timing, counters, and gauges.

Emits structured metrics to configurable backends:
  - ``"log"`` — JSON-formatted log lines (default, zero dependencies).
  - ``"statsd"`` — UDP packets to a StatsD daemon (optional).

Instruments key pipeline stages with ``timer`` context managers, tracks
series counts with ``gauge``, and counts events with ``counter``.

Usage
-----
>>> from src.observability.metrics import MetricsEmitter
>>> from src.observability.context import PipelineContext
>>> ctx = PipelineContext(lob="rossmann")
>>> emitter = MetricsEmitter(backend="log", context=ctx)
>>> with emitter.timer("model_fit"):
...     model.fit(data)
>>> emitter.counter("forecast_rows", 5000)
>>> emitter.gauge("series_count", 1200)
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

from .context import PipelineContext

logger = logging.getLogger(__name__)


class MetricsEmitter:
    """
    Emit pipeline metrics to a pluggable backend.

    Parameters
    ----------
    backend:
        ``"log"`` | ``"statsd"``.
    context:
        Active ``PipelineContext`` for tagging metrics with run_id/lob.
    statsd_host:
        StatsD daemon host (only used when ``backend="statsd"``).
    statsd_port:
        StatsD daemon port.
    prefix:
        Metric name prefix (e.g. ``"forecast_platform"``).
    """

    def __init__(
        self,
        backend: str = "log",
        context: Optional[PipelineContext] = None,
        statsd_host: str = "localhost",
        statsd_port: int = 8125,
        prefix: str = "forecast_platform",
    ):
        self.backend = backend
        self.context = context
        self.prefix = prefix
        self._recorded: List[Dict[str, Any]] = []
        self._statsd_addr: Optional[Tuple[str, int]] = None
        self._statsd_sock = None

        if backend == "statsd":
            self._statsd_addr = (statsd_host, statsd_port)

    def set_context(self, context: PipelineContext) -> None:
        """Attach or update the pipeline context."""
        self.context = context

    # ── Core emit ─────────────────────────────────────────────────────────────

    def emit(self, name: str, value: float, metric_type: str = "gauge", **tags: Any) -> None:
        """
        Emit a single metric data point.

        Parameters
        ----------
        name:
            Metric name (e.g. ``"model_fit_duration_seconds"``).
        value:
            Numeric value.
        metric_type:
            ``"gauge"`` | ``"counter"`` | ``"timer"``.
        tags:
            Additional key-value tags.
        """
        full_name = f"{self.prefix}.{name}" if self.prefix else name
        ctx_tags = {}
        if self.context:
            ctx_tags = {"run_id": self.context.run_id, "lob": self.context.lob}

        record = {
            "metric": full_name,
            "value": value,
            "type": metric_type,
            **ctx_tags,
            **tags,
        }
        self._recorded.append(record)

        if self.backend == "log":
            logger.info(json.dumps(record, default=str))
        elif self.backend == "statsd":
            self._send_statsd(full_name, value, metric_type, tags)

    # ── Convenience methods ───────────────────────────────────────────────────

    def counter(self, name: str, value: int = 1, **tags: Any) -> None:
        """Increment a counter metric."""
        self.emit(name, float(value), metric_type="counter", **tags)

    def gauge(self, name: str, value: float, **tags: Any) -> None:
        """Set a gauge metric."""
        self.emit(name, value, metric_type="gauge", **tags)

    @contextmanager
    def timer(self, name: str, **tags: Any):
        """
        Context manager that emits ``{name}_duration_seconds`` on exit.

        >>> with emitter.timer("model_fit", model="lgbm_direct"):
        ...     model.fit(data)
        """
        start = time.perf_counter()
        yield
        duration = time.perf_counter() - start
        self.emit(f"{name}_duration_seconds", duration, metric_type="timer", **tags)

    # ── Recorded metrics (for testing / cost estimation) ──────────────────────

    @property
    def recorded(self) -> List[Dict[str, Any]]:
        """All metrics emitted during the lifetime of this emitter."""
        return list(self._recorded)

    def get_timers(self) -> Dict[str, float]:
        """Return all timer metrics as a {name: seconds} dict."""
        return {
            r["metric"]: r["value"]
            for r in self._recorded
            if r["type"] == "timer"
        }

    def reset(self) -> None:
        """Clear all recorded metrics."""
        self._recorded.clear()

    # ── StatsD backend ────────────────────────────────────────────────────────

    def _send_statsd(self, name: str, value: float, metric_type: str,
                     tags: Dict[str, Any]) -> None:
        """Send a UDP packet to the StatsD daemon."""
        import socket

        if self._statsd_sock is None:
            self._statsd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        type_map = {"counter": "c", "gauge": "g", "timer": "ms"}
        statsd_type = type_map.get(metric_type, "g")

        # Convert tag dict to DogStatsD tag format
        tag_str = ""
        if tags:
            tag_str = "|#" + ",".join(f"{k}:{v}" for k, v in tags.items())

        packet = f"{name}:{value}|{statsd_type}{tag_str}"
        try:
            self._statsd_sock.sendto(
                packet.encode(), self._statsd_addr
            )
        except OSError as exc:
            logger.debug("StatsD send failed: %s", exc)

    def close(self) -> None:
        """Close the StatsD socket if open."""
        if self._statsd_sock:
            self._statsd_sock.close()
            self._statsd_sock = None
