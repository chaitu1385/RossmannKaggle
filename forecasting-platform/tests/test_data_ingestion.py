"""
Tests for the data ingestion layer:
  - Source connectors (FileSource, DatabaseSource, APISource)
  - Schema validation (SchemaValidator)
  - Data quality scoring (DataQualityScorer)
  - Ingestion orchestrator (IngestionPipeline)
"""

import json
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from src.data.sources import FileSource, DatabaseSource, APISource, build_source
from src.data.schema_validator import (
    ColumnSpec,
    SchemaValidator,
    ValidationResult,
    build_column_specs,
)
from src.data.quality import (
    DataQualityScorer,
    QualityCheckConfig,
    QualityReport,
    CheckResult,
    build_quality_checks,
)
from src.data.ingestion import IngestionPipeline, IngestionResult
from src.config.schema import IngestionConfig, PlatformConfig


# ────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """A well-formed panel DataFrame."""
    today = date.today()
    rows = []
    for sid in ["SKU_001", "SKU_002"]:
        for i in range(10):
            rows.append({
                "series_id": sid,
                "week": today - timedelta(weeks=10 - i),
                "quantity": float(100 + i * 10),
                "lob": "retail",
            })
    return pl.DataFrame(rows)


@pytest.fixture
def csv_path(sample_df, tmp_path):
    """Write sample data to CSV and return path."""
    path = tmp_path / "actuals.csv"
    sample_df.write_csv(path)
    return path


@pytest.fixture
def parquet_path(sample_df, tmp_path):
    """Write sample data to Parquet and return path."""
    path = tmp_path / "actuals.parquet"
    sample_df.write_parquet(path)
    return path


@pytest.fixture
def parquet_dir(sample_df, tmp_path):
    """Write sample data as a directory of Parquet files."""
    dir_path = tmp_path / "actuals_dir"
    dir_path.mkdir()
    half = len(sample_df) // 2
    sample_df[:half].write_parquet(dir_path / "part_0.parquet")
    sample_df[half:].write_parquet(dir_path / "part_1.parquet")
    return dir_path


# ────────────────────────────────────────────────────────────────────
# FileSource tests
# ────────────────────────────────────────────────────────────────────

class TestFileSource:

    def test_read_csv(self, csv_path):
        src = FileSource(str(csv_path))
        df = src.read()
        assert len(df) == 20
        assert "series_id" in df.columns

    def test_read_parquet(self, parquet_path):
        src = FileSource(str(parquet_path))
        df = src.read()
        assert len(df) == 20
        assert "quantity" in df.columns

    def test_read_parquet_dir(self, parquet_dir):
        src = FileSource(str(parquet_dir))
        df = src.read()
        assert len(df) == 20

    def test_probe_exists(self, csv_path):
        src = FileSource(str(csv_path))
        assert src.probe() is True

    def test_probe_missing(self):
        src = FileSource("/nonexistent/path.csv")
        assert src.probe() is False

    def test_source_type(self, csv_path):
        src = FileSource(str(csv_path))
        assert src.source_type == "file"

    def test_explicit_format(self, csv_path):
        src = FileSource(str(csv_path), format="csv")
        df = src.read()
        assert len(df) == 20

    def test_select_columns(self, parquet_path):
        src = FileSource(str(parquet_path), columns=["series_id", "quantity"])
        df = src.read()
        assert set(df.columns) == {"series_id", "quantity"}


# ────────────────────────────────────────────────────────────────────
# DatabaseSource tests
# ────────────────────────────────────────────────────────────────────

class TestDatabaseSource:

    def test_source_type(self):
        src = DatabaseSource("sqlite:///test.db", "SELECT 1")
        assert src.source_type == "database"

    def test_env_var_resolution(self, monkeypatch):
        monkeypatch.setenv("TEST_DB_CONN", "sqlite:///test.db")
        src = DatabaseSource("$TEST_DB_CONN", "SELECT 1")
        assert src.connection_string == "sqlite:///test.db"

    def test_env_var_missing(self):
        src = DatabaseSource("$NONEXISTENT_VAR", "SELECT 1")
        with pytest.raises(ValueError, match="Environment variable"):
            _ = src.connection_string

    def test_probe_missing_env(self):
        src = DatabaseSource("$NONEXISTENT_VAR", "SELECT 1")
        assert src.probe() is False


# ────────────────────────────────────────────────────────────────────
# build_source factory tests
# ────────────────────────────────────────────────────────────────────

class TestBuildSource:

    def test_build_file_source(self, csv_path):
        src = build_source({"type": "file", "path": str(csv_path)})
        assert isinstance(src, FileSource)

    def test_build_database_source(self):
        src = build_source({
            "type": "database",
            "connection_string": "sqlite:///test.db",
            "query": "SELECT 1",
        })
        assert isinstance(src, DatabaseSource)

    def test_build_api_source(self):
        src = build_source({
            "type": "api",
            "url": "https://example.com/data",
        })
        assert isinstance(src, APISource)

    def test_build_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown source type"):
            build_source({"type": "ftp"})

    def test_default_type_is_file(self, csv_path):
        src = build_source({"path": str(csv_path)})
        assert isinstance(src, FileSource)


# ────────────────────────────────────────────────────────────────────
# SchemaValidator tests
# ────────────────────────────────────────────────────────────────────

class TestSchemaValidator:

    def _make_validator(self):
        return SchemaValidator([
            ColumnSpec(name="series_id", dtype="Utf8", required=True),
            ColumnSpec(name="week", dtype="Date", required=True),
            ColumnSpec(name="quantity", dtype="Float64", required=True, nullable=False, min_value=0),
            ColumnSpec(name="lob", dtype="Utf8", required=True),
        ])

    def test_valid_dataframe(self, sample_df):
        v = self._make_validator()
        result = v.validate(sample_df)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_missing_required_column(self, sample_df):
        df = sample_df.drop("lob")
        v = self._make_validator()
        result = v.validate(df)
        assert not result.is_valid
        assert any("lob" in e for e in result.errors)

    def test_nullable_violation(self):
        df = pl.DataFrame({
            "series_id": ["A", "B"],
            "week": [date.today(), date.today()],
            "quantity": [1.0, None],
            "lob": ["retail", "retail"],
        })
        v = self._make_validator()
        result = v.validate(df)
        assert not result.is_valid
        assert any("nullable" in e for e in result.errors)

    def test_min_value_violation(self):
        df = pl.DataFrame({
            "series_id": ["A"],
            "week": [date.today()],
            "quantity": [-5.0],
            "lob": ["retail"],
        })
        v = self._make_validator()
        result = v.validate(df)
        assert not result.is_valid
        assert any("min value" in e for e in result.errors)

    def test_allowed_values(self):
        v = SchemaValidator([
            ColumnSpec(name="status", dtype="Utf8", allowed_values=["active", "inactive"]),
        ])
        df = pl.DataFrame({"status": ["active", "unknown"]})
        result = v.validate(df)
        assert not result.is_valid
        assert any("invalid values" in e for e in result.errors)

    def test_extra_columns_warning(self, sample_df):
        v = SchemaValidator(
            [ColumnSpec(name="series_id", dtype="Utf8")],
            allow_extra_columns=False,
        )
        result = v.validate(sample_df)
        assert len(result.warnings) > 0

    def test_build_column_specs(self):
        raw = [
            {"name": "id", "dtype": "Utf8", "required": True},
            {"name": "value", "dtype": "Float64", "min_value": 0},
        ]
        specs = build_column_specs(raw)
        assert len(specs) == 2
        assert specs[0].name == "id"
        assert specs[1].min_value == 0


# ────────────────────────────────────────────────────────────────────
# DataQualityScorer tests
# ────────────────────────────────────────────────────────────────────

class TestDataQualityScorer:

    def test_completeness_pass(self, sample_df):
        scorer = DataQualityScorer([
            QualityCheckConfig(name="completeness", severity="block", threshold=95),
        ])
        report = scorer.score(sample_df)
        assert report.passed
        assert report.check_results[0].passed

    def test_completeness_fail(self):
        df = pl.DataFrame({
            "a": [1, None, None, None, None],
            "b": [1, 2, 3, 4, 5],
        })
        scorer = DataQualityScorer([
            QualityCheckConfig(name="completeness", severity="block", threshold=90),
        ])
        report = scorer.score(df)
        assert not report.passed
        assert not report.check_results[0].passed

    def test_uniqueness_pass(self, sample_df):
        scorer = DataQualityScorer([
            QualityCheckConfig(name="uniqueness", severity="warn"),
        ])
        report = scorer.score(sample_df)
        assert report.check_results[0].passed

    def test_uniqueness_fail(self):
        df = pl.DataFrame({"a": [1, 1, 1], "b": [2, 2, 2]})
        scorer = DataQualityScorer([
            QualityCheckConfig(name="uniqueness", severity="block"),
        ])
        report = scorer.score(df)
        assert not report.passed

    def test_volume_pass(self, sample_df):
        scorer = DataQualityScorer([
            QualityCheckConfig(name="volume", severity="warn", threshold=20),
        ])
        report = scorer.score(sample_df, expected_row_count=20)
        assert report.check_results[0].passed

    def test_volume_fail(self, sample_df):
        scorer = DataQualityScorer([
            QualityCheckConfig(name="volume", severity="block", threshold=10),
        ])
        report = scorer.score(sample_df, expected_row_count=100)
        assert not report.passed

    def test_outlier_pass(self, sample_df):
        scorer = DataQualityScorer([
            QualityCheckConfig(name="outlier", severity="warn", threshold=10),
        ])
        report = scorer.score(sample_df)
        assert report.check_results[0].passed

    def test_outlier_fail(self):
        # Create data with extreme outliers
        df = pl.DataFrame({"value": [1.0] * 100 + [10000.0] * 20})
        scorer = DataQualityScorer([
            QualityCheckConfig(name="outlier", severity="block", threshold=5),
        ])
        report = scorer.score(df)
        assert not report.passed

    def test_schema_drift_pass(self, sample_df):
        scorer = DataQualityScorer([
            QualityCheckConfig(name="schema_drift", severity="warn"),
        ])
        report = scorer.score(
            sample_df,
            expected_columns=["series_id", "week", "quantity", "lob"],
        )
        assert report.check_results[0].passed

    def test_schema_drift_missing_column(self, sample_df):
        scorer = DataQualityScorer([
            QualityCheckConfig(name="schema_drift", severity="block"),
        ])
        report = scorer.score(
            sample_df,
            expected_columns=["series_id", "week", "quantity", "lob", "extra_col"],
        )
        assert not report.passed

    def test_empty_dataframe(self):
        scorer = DataQualityScorer([
            QualityCheckConfig(name="completeness", severity="block", threshold=95),
        ])
        report = scorer.score(pl.DataFrame())
        assert not report.passed

    def test_multiple_checks(self, sample_df):
        scorer = DataQualityScorer([
            QualityCheckConfig(name="completeness", severity="block", threshold=95),
            QualityCheckConfig(name="uniqueness", severity="warn"),
            QualityCheckConfig(name="outlier", severity="info", threshold=10),
        ])
        report = scorer.score(sample_df)
        assert len(report.check_results) == 3
        assert report.overall_score > 0

    def test_block_severity_gates(self, sample_df):
        scorer = DataQualityScorer([
            QualityCheckConfig(name="completeness", severity="warn", threshold=95),
            QualityCheckConfig(name="volume", severity="block", threshold=5),
        ])
        # Volume check will fail since expected_row_count is way off
        report = scorer.score(sample_df, expected_row_count=1000)
        assert not report.passed
        assert len(report.blocking_failures) == 1

    def test_report_to_dict(self, sample_df):
        scorer = DataQualityScorer([
            QualityCheckConfig(name="completeness", severity="block", threshold=95),
        ])
        report = scorer.score(sample_df)
        d = report.to_dict()
        assert "overall_score" in d
        assert "checks" in d
        assert len(d["checks"]) == 1

    def test_build_quality_checks(self):
        raw = [
            {"name": "completeness", "severity": "block", "threshold": 95},
            {"name": "outlier", "severity": "warn", "threshold": 5},
        ]
        checks = build_quality_checks(raw)
        assert len(checks) == 2
        assert checks[0].severity == "block"


# ────────────────────────────────────────────────────────────────────
# IngestionPipeline tests
# ────────────────────────────────────────────────────────────────────

class TestIngestionPipeline:

    def test_run_file_source(self, parquet_path):
        pipeline = IngestionPipeline(
            sources={"actuals": {"type": "file", "path": str(parquet_path)}},
        )
        result = pipeline.run("actuals")
        assert not result.blocked
        assert result.row_count == 20

    def test_run_with_schema_validation(self, parquet_path):
        pipeline = IngestionPipeline(
            sources={"actuals": {"type": "file", "path": str(parquet_path)}},
            schemas={
                "actuals": [
                    {"name": "series_id", "dtype": "Utf8", "required": True},
                    {"name": "week", "dtype": "Date", "required": True},
                    {"name": "quantity", "dtype": "Float64", "required": True},
                ],
            },
        )
        result = pipeline.run("actuals")
        assert result.schema_result.is_valid
        assert not result.blocked

    def test_run_with_schema_failure(self, parquet_path):
        pipeline = IngestionPipeline(
            sources={"actuals": {"type": "file", "path": str(parquet_path)}},
            schemas={
                "actuals": [
                    {"name": "missing_column", "dtype": "Utf8", "required": True},
                ],
            },
        )
        result = pipeline.run("actuals")
        assert not result.schema_result.is_valid
        assert result.blocked

    def test_run_with_quality_checks(self, parquet_path):
        pipeline = IngestionPipeline(
            sources={"actuals": {"type": "file", "path": str(parquet_path)}},
            quality_checks=[
                {"name": "completeness", "severity": "block", "threshold": 95},
                {"name": "uniqueness", "severity": "warn"},
            ],
        )
        result = pipeline.run("actuals")
        assert result.quality_report.passed
        assert len(result.quality_report.check_results) == 2

    def test_run_unknown_source(self, parquet_path):
        pipeline = IngestionPipeline(
            sources={"actuals": {"type": "file", "path": str(parquet_path)}},
        )
        with pytest.raises(KeyError, match="Unknown source"):
            pipeline.run("nonexistent")

    def test_run_unreachable_source(self):
        pipeline = IngestionPipeline(
            sources={"bad": {"type": "file", "path": "/nonexistent/file.csv"}},
        )
        result = pipeline.run("bad")
        assert result.blocked
        assert result.row_count == 0

    def test_run_all(self, parquet_path, csv_path):
        pipeline = IngestionPipeline(
            sources={
                "actuals": {"type": "file", "path": str(parquet_path)},
                "metadata": {"type": "file", "path": str(csv_path)},
            },
        )
        results = pipeline.run_all()
        assert len(results) == 2
        assert all(not r.blocked for r in results.values())

    def test_probe_all(self, parquet_path):
        pipeline = IngestionPipeline(
            sources={
                "actuals": {"type": "file", "path": str(parquet_path)},
                "missing": {"type": "file", "path": "/no/such/file.csv"},
            },
        )
        probes = pipeline.probe_all()
        assert probes["actuals"] is True
        assert probes["missing"] is False

    def test_write_report(self, parquet_path, tmp_path):
        report_dir = tmp_path / "reports"
        pipeline = IngestionPipeline(
            sources={"actuals": {"type": "file", "path": str(parquet_path)}},
            report_path=str(report_dir),
        )
        result = pipeline.run("actuals")
        assert report_dir.exists()
        report_files = list(report_dir.glob("*.json"))
        assert len(report_files) == 1

        with open(report_files[0]) as f:
            report_data = json.load(f)
        assert report_data["source_name"] == "actuals"
        assert report_data["row_count"] == 20

    def test_ingestion_result_to_dict(self, parquet_path):
        pipeline = IngestionPipeline(
            sources={"actuals": {"type": "file", "path": str(parquet_path)}},
            quality_checks=[
                {"name": "completeness", "severity": "block", "threshold": 95},
            ],
        )
        result = pipeline.run("actuals")
        d = result.to_dict()
        assert "source_name" in d
        assert "schema" in d
        assert "quality" in d
        assert d["row_count"] == 20


# ────────────────────────────────────────────────────────────────────
# Config integration tests
# ────────────────────────────────────────────────────────────────────

class TestIngestionConfig:

    def test_default_ingestion_config(self):
        config = PlatformConfig()
        assert isinstance(config.ingestion, IngestionConfig)
        assert config.ingestion.sources == {}
        assert config.ingestion.quality_checks == []

    def test_ingestion_config_with_sources(self, parquet_path):
        config = PlatformConfig(
            ingestion=IngestionConfig(
                sources={
                    "actuals": {"type": "file", "path": str(parquet_path)},
                },
                schemas={
                    "actuals": [
                        {"name": "series_id", "dtype": "Utf8", "required": True},
                    ],
                },
                quality_checks=[
                    {"name": "completeness", "severity": "block", "threshold": 95},
                ],
                time_column="week",
            )
        )
        assert "actuals" in config.ingestion.sources
        assert len(config.ingestion.quality_checks) == 1

    def test_pipeline_from_config(self, parquet_path):
        config = PlatformConfig(
            ingestion=IngestionConfig(
                sources={
                    "actuals": {"type": "file", "path": str(parquet_path)},
                },
                quality_checks=[
                    {"name": "completeness", "severity": "block", "threshold": 95},
                ],
            )
        )
        pipeline = IngestionPipeline(
            sources=config.ingestion.sources,
            schemas=config.ingestion.schemas,
            quality_checks=config.ingestion.quality_checks,
            report_path=config.ingestion.report_path,
            time_column=config.ingestion.time_column,
        )
        result = pipeline.run("actuals")
        assert not result.blocked
        assert result.row_count == 20
