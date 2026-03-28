"""
Statistical testing utilities for the Forecasting Platform.

All functions return structured dicts with results, p-values, and
human-readable interpretations.  Designed for forecast evaluation,
A/B testing of model configurations, and segment comparison.

All functions accept numpy arrays or Polars Series (converted internally).

Usage::

    from src.stats import (
        two_sample_proportion_test, two_sample_mean_test,
        mann_whitney_test, confidence_interval, chi_squared_test,
        bootstrap_ci, adjust_pvalues, characterize_distribution,
        rank_dimensions, sample_size_proportion, sample_size_mean,
        detectable_effect, forecast_accuracy_test,
    )

    result = two_sample_mean_test(model_a_errors, model_b_errors)
    print(result["interpretation"])
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

try:
    import polars as pl
    _PL_AVAILABLE = True
except ImportError:
    _PL_AVAILABLE = False

try:
    from scipy import stats as sp_stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Internal coercion
# ---------------------------------------------------------------------------

def _to_numpy(series) -> np.ndarray:
    """Coerce Polars Series, list, or array-like to numpy float64."""
    if _PL_AVAILABLE and isinstance(series, pl.Series):
        return series.drop_nulls().cast(pl.Float64).to_numpy()
    return np.asarray(series, dtype=float)


# ---------------------------------------------------------------------------
# Human-readable formatting
# ---------------------------------------------------------------------------

def format_significance(p_value: float, alpha: float = 0.05) -> str:
    """Return a human-readable significance statement."""
    if p_value < 0.001:
        return "Highly significant (p<0.001)"
    elif p_value < alpha:
        return f"Statistically significant (p={p_value:.3f})"
    return f"Not statistically significant (p={p_value:.3f})"


def interpret_effect_size(d: float, test_type: str = "cohens_d") -> str:
    """Translate a numeric effect size into a plain-English label."""
    d_abs = abs(d)
    if test_type == "cohens_d":
        if d_abs < 0.2:
            label = "Small"
        elif d_abs <= 0.8:
            label = "Medium"
        else:
            label = "Large"
        return f"{label} effect (d={d_abs:.2f})"
    return f"Effect size = {d_abs:.2f}"


# ---------------------------------------------------------------------------
# Proportion test
# ---------------------------------------------------------------------------

def two_sample_proportion_test(
    successes_a: int, n_a: int,
    successes_b: int, n_b: int,
    alpha: float = 0.05,
) -> dict:
    """Z-test for comparing two proportions (conversion rates, CTR, etc.).

    Args:
        successes_a: Successes in group A.
        n_a: Total observations in group A.
        successes_b: Successes in group B.
        n_b: Total observations in group B.
        alpha: Significance threshold.

    Returns:
        dict with test, p_value, z_stat, significant, prop_a, prop_b,
        diff, ci_lower, ci_upper, interpretation.
    """
    prop_a = successes_a / n_a
    prop_b = successes_b / n_b
    diff = prop_b - prop_a

    pooled = (successes_a + successes_b) / (n_a + n_b)
    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z_stat = diff / se_pooled if se_pooled > 0 else 0.0

    if _SCIPY_AVAILABLE:
        p_value = 2 * (1 - sp_stats.norm.cdf(abs(z_stat)))
        z_crit = sp_stats.norm.ppf(1 - alpha / 2)
    else:
        p_value = _approx_two_tail_p(z_stat)
        z_crit = 1.96

    se_diff = math.sqrt(prop_a * (1 - prop_a) / n_a + prop_b * (1 - prop_b) / n_b)
    ci_lower = diff - z_crit * se_diff
    ci_upper = diff + z_crit * se_diff

    return {
        "test": "z-test proportions",
        "p_value": float(p_value),
        "z_stat": float(z_stat),
        "significant": bool(p_value < alpha),
        "prop_a": float(prop_a),
        "prop_b": float(prop_b),
        "diff": float(diff),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "interpretation": format_significance(p_value, alpha),
    }


# ---------------------------------------------------------------------------
# Mean comparison (Welch's t-test)
# ---------------------------------------------------------------------------

def two_sample_mean_test(series_a, series_b, alpha: float = 0.05) -> dict:
    """Welch's t-test for comparing means between two groups.

    Does NOT assume equal variance.

    Args:
        series_a: Values for group A (array-like or Polars Series).
        series_b: Values for group B.
        alpha: Significance threshold.

    Returns:
        dict with test, p_value, t_stat, significant, mean_a, mean_b,
        diff, effect_size, effect_label, interpretation.
    """
    a = _to_numpy(series_a)
    b = _to_numpy(series_b)

    if not _SCIPY_AVAILABLE:
        raise ImportError("scipy is required for two_sample_mean_test")

    t_stat, p_value = sp_stats.ttest_ind(a, b, equal_var=False)

    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    diff = mean_b - mean_a

    n_a, n_b = len(a), len(b)
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    pooled_std = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    cohens_d = diff / pooled_std if pooled_std > 0 else 0.0

    return {
        "test": "welch_t",
        "p_value": float(p_value),
        "t_stat": float(t_stat),
        "significant": bool(p_value < alpha),
        "mean_a": mean_a,
        "mean_b": mean_b,
        "diff": float(diff),
        "effect_size": float(cohens_d),
        "effect_label": interpret_effect_size(cohens_d),
        "interpretation": format_significance(p_value, alpha),
    }


# ---------------------------------------------------------------------------
# Non-parametric comparison
# ---------------------------------------------------------------------------

def mann_whitney_test(series_a, series_b, alpha: float = 0.05) -> dict:
    """Mann-Whitney U test for comparing distributions (skewed data).

    Args:
        series_a: Values for group A.
        series_b: Values for group B.
        alpha: Significance threshold.

    Returns:
        dict with test, p_value, u_stat, significant, median_a, median_b,
        rank_biserial, interpretation.
    """
    a = _to_numpy(series_a)
    b = _to_numpy(series_b)

    if not _SCIPY_AVAILABLE:
        raise ImportError("scipy is required for mann_whitney_test")

    u_stat, p_value = sp_stats.mannwhitneyu(a, b, alternative="two-sided")

    n_a, n_b = len(a), len(b)
    rank_biserial = 1 - (2 * u_stat) / (n_a * n_b) if (n_a * n_b) > 0 else 0.0

    return {
        "test": "mann_whitney_u",
        "p_value": float(p_value),
        "u_stat": float(u_stat),
        "significant": bool(p_value < alpha),
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
        "rank_biserial": float(rank_biserial),
        "interpretation": format_significance(p_value, alpha),
    }


# ---------------------------------------------------------------------------
# Confidence interval (single sample)
# ---------------------------------------------------------------------------

def confidence_interval(series, confidence: float = 0.95) -> dict:
    """Compute a confidence interval for the mean.

    Args:
        series: Numeric values.
        confidence: Confidence level (default 0.95).

    Returns:
        dict with mean, ci_lower, ci_upper, std, n, confidence.
    """
    a = _to_numpy(series)
    n = len(a)
    mean = float(np.mean(a))
    std = float(np.std(a, ddof=1))
    se = std / math.sqrt(n) if n > 0 else 0.0

    if _SCIPY_AVAILABLE:
        t_crit = sp_stats.t.ppf((1 + confidence) / 2, df=max(n - 1, 1))
    else:
        t_crit = 1.96
    margin = t_crit * se

    return {
        "mean": mean,
        "ci_lower": float(mean - margin),
        "ci_upper": float(mean + margin),
        "std": std,
        "n": n,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Chi-squared test
# ---------------------------------------------------------------------------

def chi_squared_test(observed_table, alpha: float = 0.05) -> dict:
    """Chi-squared test of independence for a contingency table.

    Args:
        observed_table: 2D array of observed counts.
        alpha: Significance threshold.

    Returns:
        dict with test, p_value, chi2_stat, significant, dof, interpretation.
    """
    if not _SCIPY_AVAILABLE:
        raise ImportError("scipy is required for chi_squared_test")

    observed = np.asarray(observed_table)
    chi2_stat, p_value, dof, expected = sp_stats.chi2_contingency(observed)

    return {
        "test": "chi_squared",
        "p_value": float(p_value),
        "chi2_stat": float(chi2_stat),
        "significant": bool(p_value < alpha),
        "dof": int(dof),
        "interpretation": format_significance(p_value, alpha),
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------

def bootstrap_ci(
    series,
    stat_func: Optional[Callable] = None,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> dict:
    """Non-parametric confidence interval via bootstrapping.

    Args:
        series: Numeric values.
        stat_func: Callable(array) → scalar. Defaults to np.mean.
        n_bootstrap: Number of resamples.
        confidence: Confidence level.

    Returns:
        dict with stat, ci_lower, ci_upper, n_bootstrap, confidence.
    """
    if stat_func is None:
        stat_func = np.mean

    a = _to_numpy(series)
    observed_stat = float(stat_func(a))

    rng = np.random.default_rng()
    boot_stats = np.array([
        stat_func(rng.choice(a, size=len(a), replace=True))
        for _ in range(n_bootstrap)
    ])

    alpha_half = (1 - confidence) / 2
    ci_lower = float(np.percentile(boot_stats, 100 * alpha_half))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha_half)))

    return {
        "stat": observed_stat,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Multiple-testing correction
# ---------------------------------------------------------------------------

def adjust_pvalues(
    pvalues: Sequence[float],
    method: str = "benjamini-hochberg",
) -> dict:
    """Adjust p-values for multiple comparisons.

    Args:
        pvalues: Raw p-values.
        method: 'benjamini-hochberg', 'bonferroni', or 'holm'.

    Returns:
        dict with adjusted, method, n_significant_raw,
        n_significant_adjusted, interpretation.
    """
    pvals = np.asarray(pvalues, dtype=float)
    n = len(pvals)

    if n == 0:
        return {
            "adjusted": [],
            "method": method,
            "n_significant_raw": 0,
            "n_significant_adjusted": 0,
            "interpretation": "No p-values provided.",
        }

    if method == "bonferroni":
        adjusted = np.minimum(pvals * n, 1.0)
    elif method == "holm":
        order = np.argsort(pvals)
        sorted_pvals = pvals[order]
        adjusted_sorted = np.array([
            sorted_pvals[i] * (n - i) for i in range(n)
        ])
        for i in range(1, n):
            adjusted_sorted[i] = max(adjusted_sorted[i], adjusted_sorted[i - 1])
        adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
        adjusted = np.zeros(n)
        adjusted[order] = adjusted_sorted
    elif method == "benjamini-hochberg":
        order = np.argsort(pvals)
        sorted_pvals = pvals[order]
        adjusted_sorted = np.array([
            sorted_pvals[i] * n / (i + 1) for i in range(n)
        ])
        for i in range(n - 2, -1, -1):
            adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])
        adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
        adjusted = np.zeros(n)
        adjusted[order] = adjusted_sorted
    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            "Choose 'benjamini-hochberg', 'bonferroni', or 'holm'."
        )

    n_sig_raw = int(np.sum(pvals < 0.05))
    n_sig_adj = int(np.sum(adjusted < 0.05))
    interpretation = (
        f"{n_sig_raw} of {n} tests significant before correction; "
        f"{n_sig_adj} after {method} correction."
    )
    if n_sig_raw > n_sig_adj:
        interpretation += (
            f" {n_sig_raw - n_sig_adj} result(s) were likely false positives."
        )

    return {
        "adjusted": [float(p) for p in adjusted],
        "method": method,
        "n_significant_raw": n_sig_raw,
        "n_significant_adjusted": n_sig_adj,
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# Distribution characterization
# ---------------------------------------------------------------------------

def characterize_distribution(series, name: Optional[str] = None) -> dict:
    """Profile a numeric series' distribution shape.

    Args:
        series: Numeric values (array-like or Polars Series).
        name: Optional label.

    Returns:
        dict with n, mean, median, std, skewness, kurtosis,
        normality_test, modality, shape_description.
    """
    a = _to_numpy(series)
    label = name or "series"
    n = len(a)

    if n < 3:
        return {
            "name": label, "n": n,
            "mean": float(np.mean(a)) if n > 0 else None,
            "median": float(np.median(a)) if n > 0 else None,
            "shape_description": "Too few values to characterize.",
        }

    mean_val = float(np.mean(a))
    median_val = float(np.median(a))
    std_val = float(np.std(a, ddof=1))

    # Normality test
    normality_test = None
    if _SCIPY_AVAILABLE:
        if n < 5000:
            stat_val, p_norm = sp_stats.shapiro(a)
        else:
            stat_val, p_norm = sp_stats.normaltest(a)
        normality_test = {
            "statistic": float(stat_val),
            "p_value": float(p_norm),
            "is_normal": bool(p_norm >= 0.05),
        }

    skewness = float(sp_stats.skew(a)) if _SCIPY_AVAILABLE else _simple_skewness(a)
    kurtosis = float(sp_stats.kurtosis(a)) if _SCIPY_AVAILABLE else _simple_kurtosis(a)

    modality = _estimate_modality(a)

    shape_parts = []
    if abs(skewness) < 0.5:
        shape_parts.append("approximately symmetric")
    elif skewness > 0:
        shape_parts.append("right-skewed")
    else:
        shape_parts.append("left-skewed")
    if kurtosis > 1:
        shape_parts.append("heavy-tailed")
    elif kurtosis < -1:
        shape_parts.append("light-tailed")
    if modality != "unimodal":
        shape_parts.append(modality)

    return {
        "name": label,
        "n": n,
        "mean": mean_val,
        "median": median_val,
        "std": std_val,
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "p5": float(np.percentile(a, 5)),
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
        "p95": float(np.percentile(a, 95)),
        "skewness": skewness,
        "kurtosis": kurtosis,
        "normality_test": normality_test,
        "modality": modality,
        "shape_description": ", ".join(shape_parts),
    }


# ---------------------------------------------------------------------------
# Dimension ranking (eta-squared / ANOVA)
# ---------------------------------------------------------------------------

def rank_dimensions(
    df,
    metric_col: str,
    dimension_cols: Sequence[str],
) -> list[dict]:
    """Rank categorical dimensions by explanatory power (eta-squared).

    Args:
        df: Polars DataFrame.
        metric_col: Numeric metric column.
        dimension_cols: Categorical dimension columns.

    Returns:
        list of dicts sorted by eta_squared desc with dimension,
        eta_squared, n_groups, f_statistic, p_value, rank, interpretation.
    """
    if not _SCIPY_AVAILABLE:
        raise ImportError("scipy is required for rank_dimensions")
    if not _PL_AVAILABLE:
        raise ImportError("polars is required for rank_dimensions")

    results = []
    data = df.drop_nulls(subset=[metric_col])

    for dim in dimension_cols:
        if dim not in data.columns:
            continue
        subset = data.drop_nulls(subset=[dim])

        groups = []
        for grp_val in subset[dim].unique().to_list():
            grp_data = subset.filter(pl.col(dim) == grp_val)[metric_col].to_numpy()
            if len(grp_data) >= 2:
                groups.append(grp_data.astype(float))

        if len(groups) < 2:
            results.append({
                "dimension": dim, "eta_squared": 0.0, "n_groups": len(groups),
                "f_statistic": 0.0, "p_value": 1.0, "rank": 0,
                "interpretation": f"'{dim}' has fewer than 2 valid groups.",
            })
            continue

        f_stat, p_value = sp_stats.f_oneway(*groups)

        all_vals = np.concatenate(groups)
        grand_mean = np.mean(all_vals)
        ss_total = float(np.sum((all_vals - grand_mean) ** 2))
        ss_between = sum(
            len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups
        )
        eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0

        if eta_sq < 0.01:
            effect_label = "negligible"
        elif eta_sq < 0.06:
            effect_label = "small"
        elif eta_sq < 0.14:
            effect_label = "medium"
        else:
            effect_label = "large"

        results.append({
            "dimension": dim,
            "eta_squared": round(eta_sq, 4),
            "n_groups": len(groups),
            "f_statistic": round(float(f_stat), 4),
            "p_value": float(p_value),
            "rank": 0,
            "interpretation": (
                f"'{dim}' explains {eta_sq:.1%} of variance in {metric_col} "
                f"({effect_label} effect). {format_significance(p_value)}"
            ),
        })

    results.sort(key=lambda x: x["eta_squared"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1
    return results


# ---------------------------------------------------------------------------
# Power analysis
# ---------------------------------------------------------------------------

def sample_size_proportion(
    baseline_rate: float, mde: float,
    alpha: float = 0.05, power: float = 0.80,
) -> dict:
    """Sample size per group for a proportion test.

    Args:
        baseline_rate: Current rate (e.g. 0.10).
        mde: Minimum detectable effect as relative change (0.05 = 5% lift).
        alpha: Significance level.
        power: Statistical power.

    Returns:
        dict with sample_size_per_group, total_sample_size, interpretation.
    """
    if not _SCIPY_AVAILABLE:
        raise ImportError("scipy is required for sample_size_proportion")

    p1 = baseline_rate
    p2 = p1 * (1 + mde)
    delta = abs(p2 - p1)

    if delta == 0:
        return {
            "sample_size_per_group": float("inf"),
            "total_sample_size": float("inf"),
            "interpretation": "MDE is zero — infinite sample required.",
        }

    z_alpha = sp_stats.norm.ppf(1 - alpha / 2)
    z_beta = sp_stats.norm.ppf(power)
    n = (z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / delta ** 2
    n_per_group = int(math.ceil(n))

    return {
        "sample_size_per_group": n_per_group,
        "total_sample_size": n_per_group * 2,
        "baseline_rate": float(p1),
        "expected_rate": float(p2),
        "absolute_difference": float(delta),
        "interpretation": (
            f"Need {n_per_group:,} per group ({n_per_group * 2:,} total) "
            f"to detect a {mde:.1%} lift from {p1:.2%} to {p2:.2%} "
            f"with {power:.0%} power at alpha={alpha}."
        ),
    }


def sample_size_mean(
    baseline_mean: float, baseline_std: float, mde: float,
    alpha: float = 0.05, power: float = 0.80,
) -> dict:
    """Sample size per group for a mean comparison test.

    Args:
        baseline_mean: Current mean.
        baseline_std: Standard deviation.
        mde: Minimum detectable absolute difference.
        alpha: Significance level.
        power: Statistical power.

    Returns:
        dict with sample_size_per_group, total_sample_size, interpretation.
    """
    if not _SCIPY_AVAILABLE:
        raise ImportError("scipy is required for sample_size_mean")

    if mde == 0:
        return {
            "sample_size_per_group": float("inf"),
            "total_sample_size": float("inf"),
            "interpretation": "MDE is zero — infinite sample required.",
        }

    z_alpha = sp_stats.norm.ppf(1 - alpha / 2)
    z_beta = sp_stats.norm.ppf(power)
    n = (z_alpha + z_beta) ** 2 * 2 * baseline_std ** 2 / mde ** 2
    n_per_group = int(math.ceil(n))
    effect_d = float(mde / baseline_std) if baseline_std > 0 else 0.0

    return {
        "sample_size_per_group": n_per_group,
        "total_sample_size": n_per_group * 2,
        "effect_size_d": effect_d,
        "interpretation": (
            f"Need {n_per_group:,} per group ({n_per_group * 2:,} total) "
            f"to detect a difference of {mde:,.2f} (d={effect_d:.2f}) "
            f"with {power:.0%} power at alpha={alpha}."
        ),
    }


def detectable_effect(
    n_per_group: int,
    baseline_rate: Optional[float] = None,
    baseline_std: Optional[float] = None,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict:
    """Given a fixed sample size, calculate minimum detectable effect.

    Provide either ``baseline_rate`` (proportion) or ``baseline_std`` (mean).

    Args:
        n_per_group: Available sample per group.
        baseline_rate: Current rate (proportion test).
        baseline_std: Metric std (mean test).
        alpha: Significance level.
        power: Statistical power.

    Returns:
        dict with mde_absolute, mde_relative (if proportion), interpretation.
    """
    if baseline_rate is None and baseline_std is None:
        raise ValueError("Provide either baseline_rate or baseline_std.")

    if not _SCIPY_AVAILABLE:
        raise ImportError("scipy is required for detectable_effect")

    z_alpha = sp_stats.norm.ppf(1 - alpha / 2)
    z_beta = sp_stats.norm.ppf(power)

    if baseline_rate is not None:
        p = baseline_rate
        mde_abs = (z_alpha + z_beta) * math.sqrt(2 * p * (1 - p) / n_per_group)
        mde_rel = float(mde_abs / p) if p > 0 else 0.0
        return {
            "mde_absolute": float(mde_abs),
            "mde_relative": mde_rel,
            "interpretation": (
                f"With {n_per_group:,} per group, smallest detectable change "
                f"is {mde_abs:.4f} ({mde_rel:.1%} relative) from {p:.2%} baseline."
            ),
        }

    mde_abs = (z_alpha + z_beta) * baseline_std * math.sqrt(2 / n_per_group)
    return {
        "mde_absolute": float(mde_abs),
        "interpretation": (
            f"With {n_per_group:,} per group, smallest detectable mean "
            f"difference is {mde_abs:,.2f}."
        ),
    }


# ---------------------------------------------------------------------------
# Forecast-specific: paired accuracy comparison
# ---------------------------------------------------------------------------

def forecast_accuracy_test(
    errors_a,
    errors_b,
    alpha: float = 0.05,
    metric_name: str = "error",
) -> dict:
    """Paired test for comparing two models' forecast accuracy.

    Uses the Diebold-Mariano-style paired t-test on error differences.
    Suitable when two models forecast the same set of series.

    Args:
        errors_a: Absolute errors for model A per series/period.
        errors_b: Absolute errors for model B per series/period.
        alpha: Significance threshold.
        metric_name: Label for the error metric.

    Returns:
        dict with test, p_value, t_stat, significant, mean_a, mean_b,
        diff, winner, interpretation.
    """
    a = _to_numpy(errors_a)
    b = _to_numpy(errors_b)

    if len(a) != len(b):
        raise ValueError(
            f"Paired test requires equal lengths: got {len(a)} vs {len(b)}"
        )

    if not _SCIPY_AVAILABLE:
        raise ImportError("scipy is required for forecast_accuracy_test")

    diffs = a - b
    t_stat, p_value = sp_stats.ttest_1samp(diffs, 0)

    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    mean_diff = float(np.mean(diffs))

    if p_value < alpha:
        winner = "Model A" if mean_a < mean_b else "Model B"
        interp = (
            f"{winner} has significantly lower {metric_name} "
            f"(diff={mean_diff:+.4f}). {format_significance(p_value, alpha)}"
        )
    else:
        winner = "No significant difference"
        interp = (
            f"No significant difference in {metric_name} between models "
            f"(diff={mean_diff:+.4f}). {format_significance(p_value, alpha)}"
        )

    return {
        "test": "paired_t (Diebold-Mariano style)",
        "p_value": float(p_value),
        "t_stat": float(t_stat),
        "significant": bool(p_value < alpha),
        "mean_a": mean_a,
        "mean_b": mean_b,
        "diff": mean_diff,
        "winner": winner,
        "interpretation": interp,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _estimate_modality(values: np.ndarray) -> str:
    """Histogram-based modality estimate."""
    n = len(values)
    n_bins = min(max(int(math.sqrt(n)), 10), 50)
    counts, _ = np.histogram(values, bins=n_bins)

    peaks = 0
    for i in range(1, len(counts) - 1):
        if counts[i] > counts[i - 1] and counts[i] > counts[i + 1]:
            peaks += 1
    if len(counts) >= 2:
        if counts[0] > counts[1]:
            peaks += 1
        if counts[-1] > counts[-2]:
            peaks += 1

    if peaks <= 1:
        return "unimodal"
    elif peaks == 2:
        return "bimodal"
    return "multimodal"


def _simple_skewness(arr: np.ndarray) -> float:
    """Compute sample skewness without scipy."""
    n = len(arr)
    if n < 3:
        return 0.0
    m = arr.mean()
    s = arr.std(ddof=1)
    if s == 0:
        return 0.0
    return float((n / ((n - 1) * (n - 2))) * np.sum(((arr - m) / s) ** 3))


def _simple_kurtosis(arr: np.ndarray) -> float:
    """Compute excess kurtosis without scipy."""
    n = len(arr)
    if n < 4:
        return 0.0
    m = arr.mean()
    s = arr.std(ddof=1)
    if s == 0:
        return 0.0
    return float(np.mean(((arr - m) / s) ** 4) - 3.0)


def _approx_two_tail_p(z: float) -> float:
    """Rough two-tailed p-value approximation (no scipy)."""
    # Abramowitz & Stegun approximation
    a = abs(z)
    t = 1.0 / (1.0 + 0.2316419 * a)
    d = 0.3989422804014327  # 1/sqrt(2*pi)
    p_one_tail = d * math.exp(-a * a / 2) * t * (
        0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
    )
    return 2 * p_one_tail
