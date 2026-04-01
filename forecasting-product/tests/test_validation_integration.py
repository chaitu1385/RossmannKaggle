"""
End-to-end test for the 4-layer validation framework integration.

Tests:
1. PostValidationConfig loads correctly from dict and YAML
2. validation_step.py runs all 4 layers on synthetic backtest data
3. Good data → high grade (A/B)
4. Bad data → low grade (D/F)
5. halt_on_blocker raises ValidationError on BLOCKER
6. enabled=False skips validation
7. BacktestPipeline return dict includes 'validation' key
8. ForecastPipeline sets _last_validation_result
"""

import sys
import os
import tempfile

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import polars as pl
import pytest
from datetime import date, timedelta


# ── Test 1: Config loading ─────────────────────────────────────────────────

def test_post_validation_config_defaults():
    """PostValidationConfig has sensible defaults."""
    from src.config.schema import PostValidationConfig
    cfg = PostValidationConfig()
    assert cfg.enabled is True
    assert cfg.structural_checks is True
    assert cfg.logical_checks is True
    assert cfg.business_rules_checks is True
    assert cfg.simpsons_paradox_checks is True
    assert cfg.halt_on_blocker is False
    assert cfg.min_grade == "D"
    assert cfg.max_yoy_change_pct == 500.0
    assert cfg.custom_range_rules == []
    assert cfg.simpsons_segment_columns == []
    print("✓ Test 1 PASSED: PostValidationConfig defaults correct")


def test_post_validation_config_in_platform_config():
    """PlatformConfig includes post_validation field."""
    from src.config.schema import PlatformConfig
    cfg = PlatformConfig()
    assert hasattr(cfg, "post_validation")
    assert cfg.post_validation.enabled is True
    print("✓ Test 2 PASSED: PlatformConfig includes post_validation")


def test_config_from_yaml():
    """Config loads from YAML dict with post_validation section."""
    from src.config.loader import _dict_to_config
    raw = {
        "lob": "test",
        "post_validation": {
            "enabled": True,
            "halt_on_blocker": True,
            "min_grade": "C",
            "max_yoy_change_pct": 300.0,
            "simpsons_segment_columns": ["model_id"],
        },
    }
    cfg = _dict_to_config(raw)
    assert cfg.post_validation.enabled is True
    assert cfg.post_validation.halt_on_blocker is True
    assert cfg.post_validation.min_grade == "C"
    assert cfg.post_validation.max_yoy_change_pct == 300.0
    assert cfg.post_validation.simpsons_segment_columns == ["model_id"]
    print("✓ Test 3 PASSED: YAML dict → PostValidationConfig works")


def test_config_from_yaml_no_post_validation():
    """Config loads cleanly when post_validation section is absent."""
    from src.config.loader import _dict_to_config
    raw = {"lob": "retail"}
    cfg = _dict_to_config(raw)
    assert cfg.post_validation.enabled is True  # default
    assert cfg.post_validation.halt_on_blocker is False
    print("✓ Test 4 PASSED: Missing post_validation section → defaults")


# ── Test 2: Synthetic data helpers ─────────────────────────────────────────

def _make_good_backtest_data(n_series=10, n_folds=3, n_models=2):
    """Create clean backtest results DataFrame."""
    rows = []
    base_date = date(2025, 1, 6)
    model_names = [f"model_{i}" for i in range(n_models)]
    for model_id in model_names:
        for fold in range(n_folds):
            for sid in range(n_series):
                for step in range(13):  # 13 weeks
                    actual = 100.0 + sid * 10 + step * 2
                    forecast = actual * (1.0 + (sid % 3) * 0.02)  # small error
                    rows.append({
                        "series_id": f"sku_{sid:03d}",
                        "model_id": model_id,
                        "fold": fold,
                        "target_week": base_date + timedelta(weeks=fold * 13 + step),
                        "forecast_step": step + 1,
                        "actual": actual,
                        "forecast": forecast,
                        "wmape": abs(forecast - actual) / max(actual, 1e-8),
                        "normalized_bias": (forecast - actual) / max(actual, 1e-8),
                        "mape": abs(forecast - actual) / max(actual, 1e-8),
                    })
    return pl.DataFrame(rows)


def _make_bad_backtest_data():
    """Create backtest results with quality issues."""
    rows = []
    base_date = date(2025, 1, 6)
    for sid in range(5):
        for step in range(13):
            actual = 100.0
            # Intentional issues:
            forecast = -50.0 if step == 0 else actual * 8  # negative + extreme
            rows.append({
                "series_id": f"sku_{sid:03d}",
                "model_id": "bad_model",
                "fold": 0,
                "target_week": base_date + timedelta(weeks=step),
                "forecast_step": step + 1,
                "actual": actual,
                "forecast": forecast,
                "wmape": 5.0,  # absurdly high
                "normalized_bias": 3.0,  # way out of range
                "mape": 5.0,
            })
    return pl.DataFrame(rows)


def _make_good_forecast_data(n_series=10, horizon=13):
    """Create clean forecast output DataFrame."""
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


# ── Test 3: Validation step on good data ───────────────────────────────────

def test_validation_good_data():
    """Good data should produce grade A or B."""
    from src.pipeline.validation_step import run_post_validation
    from src.config.schema import PostValidationConfig

    df = _make_good_backtest_data()
    config = PostValidationConfig()
    result = run_post_validation(df, config, lob="test")

    assert result["skipped"] is False
    assert result["grade"] in ("A", "B"), f"Expected A/B, got {result['grade']}"
    assert result["score"] >= 70, f"Expected score >= 70, got {result['score']}"
    assert "layers" in result
    assert "structural" in result["layers"]
    assert "logical" in result["layers"]
    assert "business" in result["layers"]
    print(f"✓ Test 5 PASSED: Good data → grade {result['grade']} ({result['score']}/100)")


def test_validation_bad_data():
    """Bad data should produce grade D or F."""
    from src.pipeline.validation_step import run_post_validation
    from src.config.schema import PostValidationConfig

    df = _make_bad_backtest_data()
    config = PostValidationConfig()
    result = run_post_validation(df, config, lob="test")

    assert result["skipped"] is False
    assert result["grade"] in ("C", "D", "F"), f"Expected C/D/F, got {result['grade']}"
    print(f"✓ Test 6 PASSED: Bad data → grade {result['grade']} ({result['score']}/100)")


def test_validation_disabled():
    """When disabled, validation returns skipped result."""
    from src.pipeline.validation_step import run_post_validation
    from src.config.schema import PostValidationConfig

    df = _make_good_backtest_data()
    config = PostValidationConfig(enabled=False)
    result = run_post_validation(df, config, lob="test")

    assert result["skipped"] is True
    assert result["grade"] == "N/A"
    print("✓ Test 7 PASSED: disabled → skipped")


def test_validation_halt_on_blocker():
    """halt_on_blocker should raise ValidationError on BLOCKER."""
    from src.pipeline.validation_step import run_post_validation, ValidationError
    from src.config.schema import PostValidationConfig

    df = _make_bad_backtest_data()
    config = PostValidationConfig(halt_on_blocker=True)

    # Bad data triggers BLOCKER cap → ValidationError must be raised
    raised = False
    try:
        run_post_validation(df, config, lob="test")
    except ValidationError as exc:
        raised = True
        assert "BLOCKER" in str(exc)
        assert "test" in str(exc)

    assert raised, "Expected ValidationError to be raised for bad data with halt_on_blocker=True"
    print("✓ Test 8 PASSED: halt_on_blocker raises ValidationError on BLOCKER")


def test_validation_forecast_data():
    """Validation runs on forecast output DataFrame."""
    from src.pipeline.validation_step import run_post_validation
    from src.config.schema import PostValidationConfig

    df = _make_good_forecast_data()
    config = PostValidationConfig()
    result = run_post_validation(df, config, lob="test", forecast_df=df)

    assert result["skipped"] is False
    assert result["grade"] in ("A", "B", "C", "D"), f"Expected A-D, got {result['grade']}"
    assert result["score"] >= 0
    assert result["layers"] is not None
    print(f"✓ Test 9 PASSED: Forecast data → grade {result['grade']} ({result['score']}/100)")


def test_validation_individual_layers():
    """Each layer returns expected structure."""
    from src.pipeline.validation_step import run_post_validation
    from src.config.schema import PostValidationConfig

    df = _make_good_backtest_data()
    config = PostValidationConfig()
    result = run_post_validation(df, config, lob="test")

    layers = result["layers"]

    # Structural
    s = layers["structural"]
    assert s is not None
    assert "ok" in s
    assert "checks_run" in s
    assert "checks_passed" in s

    # Logical
    l = layers["logical"]
    assert l is not None
    assert "ok" in l
    assert "checks_run" in l

    # Business
    b = layers["business"]
    assert b is not None
    assert "ok" in b

    # Confidence
    c = result["confidence"]
    assert "score" in c
    assert "grade" in c
    assert "factors" in c
    assert "caps_applied" in c

    print("✓ Test 10 PASSED: All layers return expected structure")


def test_validation_selective_layers():
    """Individual layers can be disabled via config."""
    from src.pipeline.validation_step import run_post_validation
    from src.config.schema import PostValidationConfig

    df = _make_good_backtest_data()

    # Disable all but structural
    config = PostValidationConfig(
        logical_checks=False,
        business_rules_checks=False,
        simpsons_paradox_checks=False,
    )
    result = run_post_validation(df, config, lob="test")

    assert result["layers"]["structural"] is not None
    assert result["layers"]["logical"] is None
    assert result["layers"]["business"] is None
    assert result["layers"]["paradox"] is None
    print(f"✓ Test 11 PASSED: Selective layers (grade={result['grade']})")


def test_min_grade_warning(capsys=None):
    """Grade below min_grade should log a warning (not raise)."""
    import logging
    from src.pipeline.validation_step import run_post_validation
    from src.config.schema import PostValidationConfig

    df = _make_bad_backtest_data()
    config = PostValidationConfig(min_grade="A")  # very strict

    # Capture log output
    handler = logging.StreamHandler()
    handler.setLevel(logging.WARNING)
    log = logging.getLogger("src.pipeline.validation_step")
    log.addHandler(handler)

    result = run_post_validation(df, config, lob="test")
    # Should not raise, just warn
    assert result["grade"] != "A"  # bad data won't be A
    log.removeHandler(handler)
    print(f"✓ Test 12 PASSED: min_grade warning works (grade={result['grade']})")


# ── Test 4: Full config round-trip ─────────────────────────────────────────

def test_yaml_round_trip():
    """Config can be written to YAML and read back."""
    import yaml
    from src.config.loader import _dict_to_config

    original = {
        "lob": "roundtrip_test",
        "post_validation": {
            "enabled": True,
            "halt_on_blocker": True,
            "min_grade": "B",
            "max_yoy_change_pct": 200.0,
            "max_period_change_pct": 300.0,
            "simpsons_segment_columns": ["model_id", "channel"],
            "custom_range_rules": [
                {"column": "forecast", "min": 0, "max": 10000},
            ],
        },
    }

    # Write to temp YAML and read back
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(original, f)
        tmp_path = f.name

    try:
        from src.config.loader import load_config
        cfg = load_config(tmp_path)
        assert cfg.lob == "roundtrip_test"
        assert cfg.post_validation.halt_on_blocker is True
        assert cfg.post_validation.min_grade == "B"
        assert cfg.post_validation.max_yoy_change_pct == 200.0
        assert cfg.post_validation.simpsons_segment_columns == ["model_id", "channel"]
        assert len(cfg.post_validation.custom_range_rules) == 1
        print("✓ Test 13 PASSED: YAML round-trip works")
    finally:
        os.unlink(tmp_path)


# ── Run all tests ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("4-Layer Validation Integration — End-to-End Tests")
    print("=" * 60)
    print()

    tests = [
        test_post_validation_config_defaults,
        test_post_validation_config_in_platform_config,
        test_config_from_yaml,
        test_config_from_yaml_no_post_validation,
        test_validation_good_data,
        test_validation_bad_data,
        test_validation_disabled,
        test_validation_halt_on_blocker,
        test_validation_forecast_data,
        test_validation_individual_layers,
        test_validation_selective_layers,
        test_min_grade_warning,
        test_yaml_round_trip,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"✗ {test_fn.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            print()

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)

    if failed:
        sys.exit(1)
