"""Baseline-driven quality gates for M5 integration tests.

Replaces hand-waved sanity bounds (e.g. ``best_wmape < 2.0``) with
empirical, versioned baselines captured by ``scripts/bless_m5_baseline.py``.

Each baseline JSON records:
  * ``fixture_sha256`` / ``config_sha256`` — detect silent fixture drift
  * ``seed`` — deterministic models only
  * ``per_model_wmape`` — champion + every ranked model
  * ``champion_model`` / ``champion_wmape`` / ``naive_wmape``
  * ``fva_vs_naive`` — championWmape improvement over naive_seasonal
  * Optional calibration coverage fields

Tests import :func:`load_baseline`, :func:`verify_fixture_hash`, and
:func:`assert_no_regression` to assert structural + numerical invariants.

If a baseline file is missing, tests skip (not fail) so a fresh checkout
without a blessed baseline doesn't red-CI-break. Running
``python scripts/bless_m5_baseline.py --frequency daily`` populates it.
"""
from __future__ import annotations

import hashlib
import json
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BASELINES_DIR = Path(__file__).resolve().parent / "baselines"

# ── Tolerances (can be tightened once baselines are stable) ───────────────────
# Champion WMAPE may degrade by at most this fraction before the gate fires.
CHAMPION_REGRESSION_TOLERANCE = 0.05  # 5 % relative
# Any model's WMAPE may degrade by at most this fraction.
PER_MODEL_REGRESSION_TOLERANCE = 0.10  # 10 % relative
# Minimum FVA (naive - champion) that must be preserved; 0.0 means "champion
# must at least match the recorded improvement over naive".
MIN_FVA_RATIO = 0.80  # accept ≥ 80 % of the blessed FVA improvement
# P90 coverage must stay within this many percentage points of baseline.
CALIBRATION_TOLERANCE_PP = 0.05


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Baseline:
    """Parsed baseline JSON with convenience accessors."""

    frequency: str
    fixture_sha256: str
    config_sha256: str
    seed: int
    per_model_wmape: dict[str, float]
    champion_model: str
    champion_wmape: float
    naive_wmape: float | None = None
    fva_vs_naive: float | None = None
    calibration_p50_coverage: float | None = None
    calibration_p90_coverage: float | None = None
    notes: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Baseline":
        return cls(
            frequency=data["frequency"],
            fixture_sha256=data["fixture_sha256"],
            config_sha256=data["config_sha256"],
            seed=int(data.get("seed", 42)),
            per_model_wmape={k: float(v) for k, v in data["per_model_wmape"].items()},
            champion_model=data["champion_model"],
            champion_wmape=float(data["champion_wmape"]),
            naive_wmape=_opt_float(data.get("naive_wmape")),
            fva_vs_naive=_opt_float(data.get("fva_vs_naive")),
            calibration_p50_coverage=_opt_float(data.get("calibration_p50_coverage")),
            calibration_p90_coverage=_opt_float(data.get("calibration_p90_coverage")),
            notes=data.get("notes", ""),
            raw=data,
        )


def _opt_float(v: Any) -> float | None:
    return None if v is None else float(v)


# ── File IO + hashing ─────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file, streamed to tolerate large fixtures."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def baseline_path(frequency: str) -> Path:
    """Return the canonical baseline JSON path for a given frequency."""
    return BASELINES_DIR / f"m5_{frequency}_baseline.json"


def load_baseline(frequency: str) -> Baseline | None:
    """Load the blessed baseline for *frequency* or return ``None`` if absent."""
    p = baseline_path(frequency)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return Baseline.from_dict(data)


def require_baseline(frequency: str, testcase: unittest.TestCase) -> Baseline:
    """Load the baseline or ``testcase.skipTest`` with a helpful message."""
    bl = load_baseline(frequency)
    if bl is None:
        testcase.skipTest(
            f"No baseline at {baseline_path(frequency)}. "
            f"Run: python scripts/bless_m5_baseline.py --frequency {frequency}"
        )
    return bl  # type: ignore[return-value]


# ── Assertion helpers ─────────────────────────────────────────────────────────

def verify_fixture_hash(
    testcase: unittest.TestCase,
    baseline: Baseline,
    fixture_path: Path,
    config_path: Path,
) -> None:
    """Fail if fixture or config contents have drifted from the blessed hashes.

    A hash mismatch means the baseline is stale; re-bless intentionally via
    the bless script rather than weakening the gate silently.
    """
    fixture_hash = sha256_file(fixture_path)
    config_hash = sha256_file(config_path)
    testcase.assertEqual(
        fixture_hash,
        baseline.fixture_sha256,
        f"Fixture {fixture_path.name} has changed since baseline was blessed.\n"
        f"  baseline: {baseline.fixture_sha256}\n"
        f"  current:  {fixture_hash}\n"
        "If intentional, re-run scripts/bless_m5_baseline.py.",
    )
    testcase.assertEqual(
        config_hash,
        baseline.config_sha256,
        f"Config {config_path.name} has changed since baseline was blessed.\n"
        f"  baseline: {baseline.config_sha256}\n"
        f"  current:  {config_hash}\n"
        "If intentional, re-run scripts/bless_m5_baseline.py.",
    )


def assert_no_regression(
    testcase: unittest.TestCase,
    baseline: Baseline,
    observed_champion_wmape: float,
    observed_per_model_wmape: dict[str, float],
) -> None:
    """Layered regression gates against a blessed baseline.

    1. Champion WMAPE must not exceed ``baseline.champion_wmape *
       (1 + CHAMPION_REGRESSION_TOLERANCE)``.
    2. Every blessed model's WMAPE must not exceed its baseline value by more
       than ``PER_MODEL_REGRESSION_TOLERANCE`` (relative). Models absent from
       the observed run are reported but skipped (non-determinism or model
       pruning is caller's responsibility).
    """
    champion_limit = baseline.champion_wmape * (1 + CHAMPION_REGRESSION_TOLERANCE)
    testcase.assertLessEqual(
        observed_champion_wmape,
        champion_limit,
        f"Champion WMAPE regression: observed={observed_champion_wmape:.4f} "
        f"> baseline×{1 + CHAMPION_REGRESSION_TOLERANCE:.2f}={champion_limit:.4f} "
        f"(baseline={baseline.champion_wmape:.4f}).",
    )

    for model, baseline_wmape in baseline.per_model_wmape.items():
        if model not in observed_per_model_wmape:
            continue
        observed = observed_per_model_wmape[model]
        limit = baseline_wmape * (1 + PER_MODEL_REGRESSION_TOLERANCE)
        testcase.assertLessEqual(
            observed,
            limit,
            f"Model '{model}' WMAPE regression: observed={observed:.4f} "
            f"> baseline×{1 + PER_MODEL_REGRESSION_TOLERANCE:.2f}={limit:.4f} "
            f"(baseline={baseline_wmape:.4f}).",
        )


def assert_fva_preserved(
    testcase: unittest.TestCase,
    baseline: Baseline,
    observed_naive_wmape: float,
    observed_champion_wmape: float,
) -> None:
    """Assert that the champion still beats the naive baseline by a meaningful
    margin (≥ ``MIN_FVA_RATIO`` of the blessed improvement, and strictly > 0).

    This is the *business-value* gate: forecasting must add value over the
    dumbest baseline, not just converge.
    """
    observed_fva = observed_naive_wmape - observed_champion_wmape
    testcase.assertGreater(
        observed_fva,
        0.0,
        f"Champion ({observed_champion_wmape:.4f}) did not beat naive "
        f"({observed_naive_wmape:.4f}) — forecast value added is non-positive.",
    )
    if baseline.fva_vs_naive is None or baseline.fva_vs_naive <= 0:
        return  # no blessed FVA improvement to compare against
    required = baseline.fva_vs_naive * MIN_FVA_RATIO
    testcase.assertGreaterEqual(
        observed_fva,
        required,
        f"FVA regression: observed improvement over naive={observed_fva:.4f} "
        f"< {MIN_FVA_RATIO:.0%} of baseline improvement={baseline.fva_vs_naive:.4f} "
        f"(required ≥ {required:.4f}).",
    )


def assert_calibration_preserved(
    testcase: unittest.TestCase,
    baseline: Baseline,
    observed_p90_coverage: float,
) -> None:
    """Assert that P90 interval coverage is within tolerance of baseline.

    Skipped when the baseline does not record calibration.
    """
    if baseline.calibration_p90_coverage is None:
        testcase.skipTest("Baseline does not record P90 calibration coverage.")
    delta = abs(observed_p90_coverage - baseline.calibration_p90_coverage)
    testcase.assertLessEqual(
        delta,
        CALIBRATION_TOLERANCE_PP,
        f"P90 coverage drift: observed={observed_p90_coverage:.3f}, "
        f"baseline={baseline.calibration_p90_coverage:.3f}, "
        f"delta={delta:.3f} > tol={CALIBRATION_TOLERANCE_PP}.",
    )
