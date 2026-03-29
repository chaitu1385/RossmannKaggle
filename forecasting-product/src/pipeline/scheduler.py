"""
PipelineScheduler — recurring forecast pipeline execution with retry and dead-letter.

Provides a simple scheduler for production forecast runs:
  - Periodic execution (configurable interval in hours).
  - Exponential backoff retry on failure (up to ``max_retries``).
  - Dead-letter logging for permanently failed runs.

For cron-like scheduling (e.g. "every Monday at 6am"), integrate with
APScheduler or external orchestrators (Fabric Pipelines, Azure Data
Factory, Airflow).

Usage
-----
>>> from src.pipeline.scheduler import PipelineScheduler
>>> scheduler = PipelineScheduler(
...     config_path="configs/platform_config.yaml",
...     interval_hours=168,  # weekly
...     max_retries=3,
... )
>>> scheduler.start()  # blocks — runs forever on cadence
"""

from __future__ import annotations

import json
import logging
import sched
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class ScheduledRunResult:
    """Outcome of a single scheduled pipeline execution."""
    run_id: str = ""
    timestamp: str = ""
    status: str = "pending"       # "success" | "failed" | "dead_letter"
    attempts: int = 0
    error: str = ""
    duration_seconds: float = 0.0


class PipelineScheduler:
    """
    Simple scheduler for recurring forecast pipeline runs.

    Parameters
    ----------
    config_path:
        Path to platform YAML config.
    interval_hours:
        Hours between successive pipeline runs (default: 168 = weekly).
    max_retries:
        Maximum retry attempts per run (with exponential backoff).
    dead_letter_path:
        Directory for dead-letter Parquet files (failed runs).
    pipeline_fn:
        Optional custom pipeline function.  If not provided, uses the
        standard ``run_forecast_pipeline`` from ``src.pipeline.forecast``.
    lob:
        Line-of-business for the pipeline run.
    """

    def __init__(
        self,
        config_path: str = "configs/platform_config.yaml",
        interval_hours: int = 168,
        max_retries: int = 3,
        dead_letter_path: str = "data/dead_letter/",
        pipeline_fn: Optional[Callable] = None,
        lob: str = "default",
    ):
        self.config_path = config_path
        self.interval_hours = interval_hours
        self.max_retries = max_retries
        self.dead_letter_path = Path(dead_letter_path)
        self.dead_letter_path.mkdir(parents=True, exist_ok=True)
        self._pipeline_fn = pipeline_fn
        self.lob = lob
        self._history: List[ScheduledRunResult] = []
        self._running = False

    @property
    def history(self) -> List[ScheduledRunResult]:
        """Execution history for all scheduled runs."""
        return list(self._history)

    def run_once(self) -> ScheduledRunResult:
        """
        Execute the pipeline once with retry logic.

        Returns
        -------
        ``ScheduledRunResult`` with status and timing.
        """
        run_id = uuid4().hex[:12].upper()
        result = ScheduledRunResult(
            run_id=run_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

        start = time.perf_counter()

        for attempt in range(1, self.max_retries + 1):
            result.attempts = attempt
            try:
                logger.info(
                    "[%s] Starting pipeline (attempt %d/%d, lob=%s)",
                    run_id, attempt, self.max_retries, self.lob,
                )
                self._execute_pipeline()
                result.status = "success"
                result.duration_seconds = time.perf_counter() - start
                logger.info(
                    "[%s] Pipeline succeeded in %.1fs",
                    run_id, result.duration_seconds,
                )
                break

            except Exception as exc:
                result.error = str(exc)
                logger.error(
                    "[%s] Pipeline failed (attempt %d/%d): %s",
                    run_id, attempt, self.max_retries, exc,
                )
                if attempt < self.max_retries:
                    backoff = 2 ** (attempt - 1)
                    logger.info("[%s] Retrying in %ds...", run_id, backoff)
                    time.sleep(backoff)
                else:
                    result.status = "dead_letter"
                    result.duration_seconds = time.perf_counter() - start
                    self._write_dead_letter(result)

        self._history.append(result)
        return result

    def start(self) -> None:
        """
        Start the scheduler (blocking).  Runs the pipeline on cadence.

        Press Ctrl+C to stop.
        """
        self._running = True
        scheduler = sched.scheduler(time.time, time.sleep)

        def _run() -> None:
            if not self._running:
                return
            self.run_once()
            scheduler.enter(self.interval_hours * 3600, 1, _run)

        scheduler.enter(0, 1, _run)
        logger.info(
            "PipelineScheduler started: interval=%dh, max_retries=%d, lob=%s",
            self.interval_hours, self.max_retries, self.lob,
        )
        try:
            scheduler.run()
        except KeyboardInterrupt:
            logger.info("PipelineScheduler stopped by user.")
            self._running = False

    def stop(self) -> None:
        """Signal the scheduler to stop after the current run completes."""
        self._running = False

    def _execute_pipeline(self) -> None:
        """Run the forecast pipeline (custom function or default)."""
        if self._pipeline_fn:
            self._pipeline_fn()
        else:
            from ..config.loader import load_config
            from .forecast import run_forecast_pipeline

            config = load_config(self.config_path)
            config.lob = self.lob
            run_forecast_pipeline(config)

    def _write_dead_letter(self, result: ScheduledRunResult) -> None:
        """Write failed run metadata to dead-letter JSON for investigation."""
        dl_file = self.dead_letter_path / f"dead_letter_{result.run_id}.json"
        try:
            with open(dl_file, "w") as f:
                json.dump(asdict(result), f, indent=2, default=str)
            logger.warning(
                "[%s] Dead-letter written to %s", result.run_id, dl_file
            )
        except Exception as exc:
            logger.error(
                "[%s] Failed to write dead-letter: %s", result.run_id, exc
            )
