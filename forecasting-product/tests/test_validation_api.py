"""Phase 2 tests — validation API endpoint, schema extensions, manifest fields."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import polars as pl
import pytest

from conftest import make_actuals

pytestmark = pytest.mark.unit


def _load_schemas():
    """Load src.api.schemas without triggering src.api.__init__ (needs fastapi)."""
    spec = importlib.util.spec_from_file_location(
        "src.api.schemas",
        Path(__file__).resolve().parent.parent / "src" / "api" / "schemas.py",
    )
    mod = importlib.util.module_from_spec(spec)
    # Inject typing names into module's namespace so pydantic can resolve
    # forward refs created by `from __future__ import annotations`.
    import typing
    for name in ("Any", "Dict", "List", "Optional"):
        setattr(mod, name, getattr(typing, name))
    spec.loader.exec_module(mod)
    return mod

_schemas = _load_schemas()
LeaderboardEntry = _schemas.LeaderboardEntry
ValidationResponse = _schemas.ValidationResponse
ValidationLayerResult = _schemas.ValidationLayerResult

# Resolve forward references (needed because schemas.py uses `from __future__ import annotations`)
import typing as _typing
_ns = {n: getattr(_typing, n) for n in ("Any", "Dict", "List", "Optional")}
# Also need pydantic Field
from pydantic import BaseModel as _BM, Field as _F
_ns["BaseModel"] = _BM
_ns["Field"] = _F
_ns["date"] = date
LeaderboardEntry.model_rebuild(_types_namespace=_ns)
ValidationResponse.model_rebuild(_types_namespace=_ns)
ValidationLayerResult.model_rebuild(_types_namespace=_ns)


# --------------------------------------------------------------------------- #
#  Helper data generators
# --------------------------------------------------------------------------- #

def _make_backtest_metrics(n_series=5, n_folds=3, n_models=2):
    """Synthetic backtest metrics DataFrame matching MetricStore schema."""
    rows = []
    base_date = date(2025, 1, 6)
    models = [f"model_{i}" for i in range(n_models)]
    for sid in range(n_series):
        for model_id in models:
            for fold in range(n_folds):
                for step in range(13):
                    actual = 100.0 + sid * 5
                    forecast = actual * 1.02  # small error
                    wmape = abs(forecast - actual) / max(actual, 1e-8)
                    rows.append({
                        "series_id": f"sku_{sid:03d}",
                        "model_id": model_id,
                        "fold": fold,
                        "target_week": base_date + timedelta(weeks=fold * 13 + step),
                        "forecast_step": step + 1,
                        "actual": actual,
                        "forecast": forecast,
                        "wmape": wmape,
                        "normalized_bias": (forecast - actual) / max(actual, 1e-8),
                        "mape": wmape,
                    })
    return pl.DataFrame(rows)


def _make_forecast_df(n_series=5, horizon=13):
    """Clean forecast output."""
    rows = []
    base_date = date(2026, 4, 6)
    for sid in range(n_series):
        for step in range(horizon):
            rows.append({
                "series_id": f"sku_{sid:03d}",
                "week": base_date + timedelta(weeks=step),
                "forecast": 100.0 + sid * 5 + step * 1.5,
            })
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  1. Schema extension tests
# --------------------------------------------------------------------------- #

class TestSchemaExtensions:

    def test_leaderboard_entry_has_validation_fields(self):
        entry = LeaderboardEntry(
            model="lgbm", wmape=0.12, normalized_bias=0.01,
            rank=1, n_series=10,
            validation_grade="B", validation_score=82,
        )
        assert entry.validation_grade == "B"
        assert entry.validation_score == 82

    def test_leaderboard_entry_validation_fields_optional(self):
        entry = LeaderboardEntry(
            model="lgbm", wmape=0.12, normalized_bias=0.01,
            rank=1, n_series=10,
        )
        assert entry.validation_grade is None
        assert entry.validation_score is None

    def test_validation_response_model(self):
        resp = ValidationResponse(
            lob="retail",
            grade="A",
            score=95,
            badge="[A] 95/100",
            layers={"structural": {"checks_run": 5, "checks_passed": 5}},
            confidence={"score": 95, "grade": "A"},
        )
        assert resp.grade == "A"
        assert resp.score == 95
        assert resp.lob == "retail"

    def test_validation_layer_result_model(self):
        layer = ValidationLayerResult(checks_run=5, checks_passed=4)
        assert layer.checks_run == 5
        assert layer.checks_passed == 4
        assert layer.issues == []


# --------------------------------------------------------------------------- #
#  2. Validation endpoint tests (using run_post_validation directly)
# --------------------------------------------------------------------------- #

class TestValidationEndpoint:

    def test_validation_runs_on_good_data(self):
        """Validation produces a valid grade on clean backtest data."""
        from src.pipeline.validation_step import run_post_validation
        from src.config.schema import PostValidationConfig

        df = _make_backtest_metrics()
        config = PostValidationConfig()
        result = run_post_validation(df, config, lob="test_api")

        assert result["grade"] in ("A", "B", "C", "D", "F")
        assert 0 <= result["score"] <= 100
        assert result["badge"]
        assert "layers" in result
        assert "confidence" in result
        assert result["skipped"] is False

    def test_validation_response_shape(self):
        """Result dict can populate ValidationResponse."""
        from src.pipeline.validation_step import run_post_validation
        from src.config.schema import PostValidationConfig

        df = _make_backtest_metrics()
        config = PostValidationConfig()
        result = run_post_validation(df, config, lob="test_api")

        # Should be constructable from result dict
        resp = ValidationResponse(
            lob="test_api",
            grade=result["grade"],
            score=result["score"],
            badge=result["badge"],
            layers=result["layers"],
            confidence=result["confidence"],
            skipped=result.get("skipped", False),
        )
        assert resp.lob == "test_api"


# --------------------------------------------------------------------------- #
#  3. Manifest extension tests
# --------------------------------------------------------------------------- #

class TestManifestExtension:

    def test_manifest_has_post_validation_fields(self):
        from src.pipeline.manifest import PipelineManifest

        m = PipelineManifest()
        assert m.post_validation_grade is None
        assert m.post_validation_score is None

    def test_build_manifest_with_validation_result(self):
        from src.config.schema import PlatformConfig
        from src.pipeline.manifest import build_manifest

        config = PlatformConfig(lob="retail")
        actuals = make_actuals()
        forecast = _make_forecast_df(n_series=2, horizon=4)

        class _MockBuilder:
            _last_validation_report = None
            _last_cleansing_report = None
            _last_regressor_screen_report = None
            _last_quality_report = None

        validation_result = {
            "grade": "B",
            "score": 82,
            "badge": "[B] 82/100",
            "skipped": False,
        }

        manifest = build_manifest(
            run_id="test_val",
            config=config,
            actuals=actuals,
            series_builder=_MockBuilder(),
            champion_model_id="lgbm_direct",
            forecast=forecast,
            forecast_file="forecast_retail.parquet",
            post_validation_result=validation_result,
        )

        assert manifest.post_validation_grade == "B"
        assert manifest.post_validation_score == 82

    def test_build_manifest_without_validation_result(self):
        from src.config.schema import PlatformConfig
        from src.pipeline.manifest import build_manifest

        config = PlatformConfig(lob="retail")
        actuals = make_actuals()
        forecast = _make_forecast_df(n_series=2, horizon=4)

        class _MockBuilder:
            _last_validation_report = None
            _last_cleansing_report = None
            _last_regressor_screen_report = None
            _last_quality_report = None

        manifest = build_manifest(
            run_id="test_no_val",
            config=config,
            actuals=actuals,
            series_builder=_MockBuilder(),
            champion_model_id="lgbm_direct",
            forecast=forecast,
            forecast_file="forecast_retail.parquet",
        )

        assert manifest.post_validation_grade is None
        assert manifest.post_validation_score is None

    def test_build_manifest_skipped_validation(self):
        from src.config.schema import PlatformConfig
        from src.pipeline.manifest import build_manifest

        config = PlatformConfig(lob="retail")
        actuals = make_actuals()
        forecast = _make_forecast_df(n_series=2, horizon=4)

        class _MockBuilder:
            _last_validation_report = None
            _last_cleansing_report = None
            _last_regressor_screen_report = None
            _last_quality_report = None

        skipped_result = {
            "grade": "N/A",
            "score": -1,
            "badge": "[N/A] Skipped",
            "skipped": True,
        }

        manifest = build_manifest(
            run_id="test_skip",
            config=config,
            actuals=actuals,
            series_builder=_MockBuilder(),
            champion_model_id="lgbm_direct",
            forecast=forecast,
            forecast_file="forecast_retail.parquet",
            post_validation_result=skipped_result,
        )

        # Skipped result should NOT populate manifest fields
        assert manifest.post_validation_grade is None
        assert manifest.post_validation_score is None

    def test_manifest_round_trip_with_validation(self):
        """Write manifest with validation fields → read back → fields preserved."""
        import json
        import tempfile
        from src.pipeline.manifest import PipelineManifest, write_manifest, read_manifest

        m = PipelineManifest(
            run_id="rt_test",
            lob="test",
            post_validation_grade="A",
            post_validation_score=95,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_parquet = str(pl.Path(tmpdir) if hasattr(pl, "Path") else __import__("pathlib").Path(tmpdir)) + "/forecast_test.parquet"
            # Create a dummy file so write_manifest can work
            from pathlib import Path
            Path(fake_parquet).touch()
            manifest_path = write_manifest(m, fake_parquet)
            loaded = read_manifest(manifest_path)

        assert loaded.post_validation_grade == "A"
        assert loaded.post_validation_score == 95


# --------------------------------------------------------------------------- #
#  4. End-to-end integration: validation → manifest → API response
# --------------------------------------------------------------------------- #

class TestEndToEndFlow:

    def test_validation_to_manifest_flow(self):
        """Full flow: run validation → feed result to build_manifest → verify."""
        from src.pipeline.validation_step import run_post_validation
        from src.config.schema import PostValidationConfig, PlatformConfig
        from src.pipeline.manifest import build_manifest

        # Step 1: Run validation
        metrics_df = _make_backtest_metrics()
        val_config = PostValidationConfig()
        val_result = run_post_validation(metrics_df, val_config, lob="e2e")

        # Step 2: Build manifest with validation result
        actuals = make_actuals()
        forecast = _make_forecast_df(n_series=2, horizon=4)

        class _MockBuilder:
            _last_validation_report = None
            _last_cleansing_report = None
            _last_regressor_screen_report = None
            _last_quality_report = None

        manifest = build_manifest(
            run_id="e2e_test",
            config=PlatformConfig(lob="e2e"),
            actuals=actuals,
            series_builder=_MockBuilder(),
            champion_model_id="lgbm",
            forecast=forecast,
            forecast_file="forecast_e2e.parquet",
            post_validation_result=val_result,
        )

        # Step 3: Verify
        assert manifest.post_validation_grade == val_result["grade"]
        assert manifest.post_validation_score == val_result["score"]
        assert manifest.post_validation_grade in ("A", "B", "C", "D", "F")
        assert 0 <= manifest.post_validation_score <= 100
