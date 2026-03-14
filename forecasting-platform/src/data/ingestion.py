"""
Ingestion pipeline — orchestrates load → validate → score → report.

Usage
-----
>>> from src.config.schema import IngestionConfig
>>> pipeline = IngestionPipeline(ingestion_config)
>>> result = pipeline.run("actuals")             # single source
>>> results = pipeline.run_all()                  # all configured sources
>>> if result.blocked:
...     raise RuntimeError(f"Data quality gate failed: {result.quality_report.blocking_failures}")
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl

from .quality import DataQualityScorer, QualityCheckConfig, QualityReport, build_quality_checks
from .schema_validator import (
    ColumnSpec,
    SchemaValidator,
    ValidationResult,
    build_column_specs,
)
from .sources import BaseSource, build_source

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Outcome of ingesting a single data source."""

    source_name: str
    data: pl.DataFrame
    schema_result: ValidationResult
    quality_report: QualityReport
    ingested_at: datetime = field(default_factory=datetime.utcnow)
    row_count: int = 0
    blocked: bool = False

    def __post_init__(self) -> None:
        self.row_count = len(self.data)
        self.blocked = (
            not self.schema_result.is_valid or not self.quality_report.passed
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_name": self.source_name,
            "row_count": self.row_count,
            "ingested_at": self.ingested_at.isoformat(),
            "blocked": self.blocked,
            "schema": {
                "is_valid": self.schema_result.is_valid,
                "errors": self.schema_result.errors,
                "warnings": self.schema_result.warnings,
            },
            "quality": self.quality_report.to_dict(),
        }


class IngestionPipeline:
    """
    Orchestrates data ingestion: source → schema validation → quality scoring.

    Parameters
    ----------
    sources : dict
        Mapping of source name → source config dict.
    schemas : dict
        Mapping of source name → list of column spec dicts.
    quality_checks : list
        List of quality check config dicts (shared across all sources).
    report_path : str, optional
        Directory to write JSON quality reports.
    time_column : str, optional
        Name of the time column (used for freshness checks).
    """

    def __init__(
        self,
        sources: Dict[str, Dict[str, Any]],
        schemas: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        quality_checks: Optional[List[Dict[str, Any]]] = None,
        report_path: Optional[str] = None,
        time_column: Optional[str] = None,
    ):
        self._source_configs = sources
        self._schema_configs = schemas or {}
        self._quality_check_configs = quality_checks or []
        self._report_path = Path(report_path) if report_path else None
        self._time_column = time_column

        # Build source connectors
        self._sources: Dict[str, BaseSource] = {}
        for name, cfg in sources.items():
            self._sources[name] = build_source(cfg)

        # Build schema validators
        self._validators: Dict[str, SchemaValidator] = {}
        for name, col_specs_raw in self._schema_configs.items():
            specs = build_column_specs(col_specs_raw)
            self._validators[name] = SchemaValidator(specs)

        # Build quality scorer
        if self._quality_check_configs:
            checks = build_quality_checks(self._quality_check_configs)
            self._quality_scorer = DataQualityScorer(checks)
        else:
            self._quality_scorer = DataQualityScorer([])

    def run(
        self,
        source_name: str,
        expected_row_count: Optional[int] = None,
    ) -> IngestionResult:
        """
        Ingest a single named source.

        Returns an IngestionResult with the data, schema validation result,
        and quality report. If ``blocked`` is True, at least one blocking
        check failed.
        """
        if source_name not in self._sources:
            raise KeyError(
                f"Unknown source '{source_name}'. "
                f"Available: {list(self._sources.keys())}"
            )

        source = self._sources[source_name]

        # Probe connectivity
        if not source.probe():
            logger.error("Source '%s' is not reachable", source_name)
            return IngestionResult(
                source_name=source_name,
                data=pl.DataFrame(),
                schema_result=ValidationResult(
                    is_valid=False,
                    errors=[f"Source '{source_name}' is not reachable"],
                ),
                quality_report=QualityReport(overall_score=0.0, passed=False),
            )

        # Read data
        logger.info("Ingesting source: %s (%s)", source_name, source.source_type)
        df = source.read()

        # Schema validation
        if source_name in self._validators:
            schema_result = self._validators[source_name].validate(df)
        else:
            schema_result = ValidationResult()  # no schema → passes

        # Quality scoring
        expected_columns = None
        if source_name in self._schema_configs:
            expected_columns = [
                c["name"] for c in self._schema_configs[source_name]
            ]

        quality_report = self._quality_scorer.score(
            df,
            expected_columns=expected_columns,
            expected_row_count=expected_row_count,
            time_column=self._time_column,
        )

        result = IngestionResult(
            source_name=source_name,
            data=df,
            schema_result=schema_result,
            quality_report=quality_report,
        )

        # Write report
        if self._report_path:
            self._write_report(result)

        if result.blocked:
            logger.warning(
                "Source '%s' BLOCKED: schema_errors=%d, quality_blocked=%d",
                source_name,
                len(schema_result.errors),
                len(quality_report.blocking_failures),
            )
        else:
            logger.info(
                "Source '%s' OK: %d rows, quality_score=%.1f",
                source_name, result.row_count, quality_report.overall_score,
            )

        return result

    def run_all(
        self,
        expected_row_counts: Optional[Dict[str, int]] = None,
    ) -> Dict[str, IngestionResult]:
        """Ingest all configured sources. Returns dict of name → result."""
        counts = expected_row_counts or {}
        results = {}
        for name in self._sources:
            results[name] = self.run(
                name, expected_row_count=counts.get(name)
            )
        return results

    def probe_all(self) -> Dict[str, bool]:
        """Check connectivity for all sources."""
        return {name: src.probe() for name, src in self._sources.items()}

    def _write_report(self, result: IngestionResult) -> None:
        """Write ingestion report as JSON."""
        self._report_path.mkdir(parents=True, exist_ok=True)
        ts = result.ingested_at.strftime("%Y%m%d_%H%M%S")
        filepath = self._report_path / f"ingestion_{result.source_name}_{ts}.json"
        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        logger.info("Ingestion report written to %s", filepath)
