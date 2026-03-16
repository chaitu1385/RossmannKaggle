"""
Forecastability signal computations.

Computes per-series statistical signals that indicate how forecastable a time
series is.  All algorithms use numpy only (no scipy dependency).

Signals
-------
- **CV** — Coefficient of variation (std / mean).
- **ApEn** — Approximate entropy; low = regular/predictable, high = chaotic.
- **Spectral entropy** — Normalised Shannon entropy of the power spectrum.
- **SNR** — Signal-to-noise ratio (variance of trend / variance of residual).
- **Trend strength** — Fraction of variance explained by linear trend.
- **Seasonal strength** — Fraction of variance explained by periodic component.
- **Forecastability score** — Weighted composite in [0, 1]; higher = easier.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl


# --------------------------------------------------------------------------- #
#  Result dataclasses
# --------------------------------------------------------------------------- #

@dataclass
class SeriesSignals:
    """Forecastability signals for a single series."""

    series_id: str
    cv: float
    apen: float
    spectral_entropy: float
    snr: float
    trend_strength: float
    seasonal_strength: float
    demand_class: str
    forecastability_score: float


@dataclass
class ForecastabilityReport:
    """Aggregate forecastability assessment across all series."""

    overall_score: float
    n_series: int
    score_distribution: Dict[str, int] = field(default_factory=dict)
    demand_class_distribution: Dict[str, int] = field(default_factory=dict)
    per_series: Optional[pl.DataFrame] = None
    warnings: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
#  Individual signal computations
# --------------------------------------------------------------------------- #

def compute_cv(values: np.ndarray) -> float:
    """Coefficient of variation: std / |mean|.  Returns 0 for constant series."""
    if len(values) < 2:
        return 0.0
    m = np.mean(values)
    if abs(m) < 1e-12:
        return 0.0
    return float(np.std(values, ddof=1) / abs(m))


def compute_approximate_entropy(
    values: np.ndarray, m: int = 2, r_factor: float = 0.2
) -> float:
    """Approximate entropy (ApEn).

    Measures the regularity of a time series.  Low ApEn indicates a
    predictable, regular series; high ApEn indicates irregular / chaotic.

    Parameters
    ----------
    values : array
        Univariate time series values.
    m : int
        Embedding dimension (template length).
    r_factor : float
        Tolerance multiplier — ``r = r_factor * std(values)``.

    Returns
    -------
    float
        ApEn value >= 0.  Returns 0.0 for constant or very short series.
    """
    n = len(values)
    if n < m + 2:
        return 0.0
    std = np.std(values, ddof=1)
    if std < 1e-12:
        return 0.0
    r = r_factor * std

    def _phi(dim: int) -> float:
        """Count template matches at given embedding dimension."""
        templates = np.array([values[i : i + dim] for i in range(n - dim + 1)])
        count = 0.0
        total = len(templates)
        for i in range(total):
            # Chebyshev (max-norm) distance
            dists = np.max(np.abs(templates - templates[i]), axis=1)
            count += np.sum(dists <= r) / total
        return np.log(count / total) if count > 0 else 0.0

    return max(0.0, float(_phi(m) - _phi(m + 1)))


def compute_spectral_entropy(values: np.ndarray) -> float:
    """Normalised Shannon entropy of the power spectrum.

    0 = single dominant frequency (very predictable),
    1 = white noise (unpredictable).

    Detrends via linear fit before FFT.
    """
    n = len(values)
    if n < 4:
        return 0.0
    std = np.std(values, ddof=1)
    if std < 1e-12:
        return 0.0

    # Detrend
    x = np.arange(n, dtype=float)
    coeffs = np.polyfit(x, values, 1)
    detrended = values - np.polyval(coeffs, x)

    # Power spectrum (one-sided)
    fft_vals = np.fft.rfft(detrended)
    power = np.abs(fft_vals) ** 2
    power = power[1:]  # drop DC component
    if power.sum() < 1e-12:
        return 0.0

    # Normalise to probability distribution
    p = power / power.sum()
    # Shannon entropy normalised by log(N)
    log_n = np.log(len(p))
    if log_n < 1e-12:
        return 0.0
    entropy = -np.sum(p * np.log(p + 1e-12)) / log_n
    return float(np.clip(entropy, 0.0, 1.0))


def compute_snr(values: np.ndarray, season_length: int = 52) -> float:
    """Signal-to-noise ratio.

    Signal = moving-average smoothed series (window = season_length).
    Noise = residual.  Returns var(signal) / var(noise).
    Uses 'valid' convolution to avoid boundary artefacts, then trims both
    arrays to the valid interior.
    """
    n = len(values)
    if n < 6:
        return 0.0
    win = min(season_length, max(3, n // 4))

    kernel = np.ones(win) / win
    signal = np.convolve(values, kernel, mode="valid")
    # Trim original to match valid convolution output length
    offset = (win - 1) // 2
    trimmed = values[offset : offset + len(signal)]
    noise = trimmed - signal
    var_noise = np.var(noise)
    if var_noise < 1e-12:
        return 100.0  # essentially noiseless
    return float(np.var(signal) / var_noise)


def compute_trend_strength(values: np.ndarray) -> float:
    """Trend strength in [0, 1].

    ``1 - var(residual) / var(values)`` where residual = values - linear trend.
    """
    n = len(values)
    if n < 3:
        return 0.0
    var_orig = np.var(values, ddof=1)
    if var_orig < 1e-12:
        return 0.0
    x = np.arange(n, dtype=float)
    coeffs = np.polyfit(x, values, 1)
    trend = np.polyval(coeffs, x)
    residual = values - trend
    strength = max(0.0, 1.0 - np.var(residual, ddof=1) / var_orig)
    return float(strength)


def compute_seasonal_strength(
    values: np.ndarray, season_length: int = 52
) -> float:
    """Seasonal strength in [0, 1].

    Computes a seasonal component by period-averaging (mean value for each
    position within the season), then measures ``1 - var(remainder) / var(deseasonalised)``.
    """
    n = len(values)
    if n < season_length + 1:
        return 0.0
    var_orig = np.var(values, ddof=1)
    if var_orig < 1e-12:
        return 0.0

    # Seasonal component via period-averaging
    seasonal = np.zeros(n)
    for pos in range(season_length):
        indices = np.arange(pos, n, season_length)
        seasonal[indices] = np.mean(values[indices])

    remainder = values - seasonal
    var_remainder = np.var(remainder, ddof=1)
    strength = max(0.0, 1.0 - var_remainder / var_orig)
    return float(strength)


def compute_forecastability_score(signals: SeriesSignals) -> float:
    """Composite forecastability score in [0, 1].  Higher = easier to forecast.

    Weights
    -------
    - 0.25 * (1 - normalised ApEn)       — regularity matters most
    - 0.20 * (1 - spectral_entropy)       — concentrated spectrum
    - 0.20 * seasonal_strength            — strong season
    - 0.15 * trend_strength               — detectable trend
    - 0.10 * (1 - normalised CV)          — stability
    - 0.10 * normalised SNR               — signal vs noise
    """
    # Normalise ApEn: typical range [0, 2.5], cap at 2.5
    norm_apen = min(signals.apen / 2.5, 1.0)
    # Normalise CV: typical range [0, 3], cap at 3
    norm_cv = min(signals.cv / 3.0, 1.0)
    # Normalise SNR: log scale, map [0, 100] → [0, 1]
    norm_snr = min(np.log1p(signals.snr) / np.log1p(100), 1.0)

    score = (
        0.25 * (1 - norm_apen)
        + 0.20 * (1 - signals.spectral_entropy)
        + 0.20 * signals.seasonal_strength
        + 0.15 * signals.trend_strength
        + 0.10 * (1 - norm_cv)
        + 0.10 * norm_snr
    )
    return float(np.clip(score, 0.0, 1.0))


# --------------------------------------------------------------------------- #
#  Analyzer class
# --------------------------------------------------------------------------- #

class ForecastabilityAnalyzer:
    """Compute forecastability signals for every series in a panel DataFrame.

    Parameters
    ----------
    season_length : int
        Assumed seasonal period (52 for weekly data).
    """

    def __init__(self, season_length: int = 52):
        self.season_length = season_length

    def analyze(
        self,
        df: pl.DataFrame,
        target_col: str,
        time_col: str,
        id_col: str,
    ) -> ForecastabilityReport:
        """Run all forecastability signals on each series.

        Parameters
        ----------
        df : pl.DataFrame
            Panel data with at least ``[id_col, time_col, target_col]``.
        target_col, time_col, id_col :
            Column names.

        Returns
        -------
        ForecastabilityReport
        """
        from ..series.sparse_detector import SparseDetector

        # Demand classification
        detector = SparseDetector()
        try:
            classified = detector.classify(df, target_col=target_col, id_col=id_col)
            class_map = {
                row[id_col]: row["demand_class"]
                for row in classified.iter_rows(named=True)
            }
        except Exception:
            class_map = {}

        # Per-series signal computation
        series_ids = df[id_col].unique().sort().to_list()
        all_signals: List[SeriesSignals] = []
        warnings: List[str] = []

        for sid in series_ids:
            mask = df.filter(pl.col(id_col) == sid).sort(time_col)
            vals = mask[target_col].to_numpy().astype(float)
            vals = vals[~np.isnan(vals)]

            if len(vals) < 4:
                warnings.append(f"Series {sid}: too short ({len(vals)} points), signals unreliable")

            cv = compute_cv(vals)
            apen = compute_approximate_entropy(vals)
            se = compute_spectral_entropy(vals)
            snr = compute_snr(vals, self.season_length)
            trend = compute_trend_strength(vals)
            seasonal = compute_seasonal_strength(vals, self.season_length)
            demand_class = class_map.get(sid, "unknown")

            sig = SeriesSignals(
                series_id=sid,
                cv=cv,
                apen=apen,
                spectral_entropy=se,
                snr=snr,
                trend_strength=trend,
                seasonal_strength=seasonal,
                demand_class=demand_class,
                forecastability_score=0.0,  # set below
            )
            sig.forecastability_score = compute_forecastability_score(sig)
            all_signals.append(sig)

        # Build per-series DataFrame
        per_series = pl.DataFrame({
            "series_id": [s.series_id for s in all_signals],
            "cv": [s.cv for s in all_signals],
            "apen": [s.apen for s in all_signals],
            "spectral_entropy": [s.spectral_entropy for s in all_signals],
            "snr": [s.snr for s in all_signals],
            "trend_strength": [s.trend_strength for s in all_signals],
            "seasonal_strength": [s.seasonal_strength for s in all_signals],
            "demand_class": [s.demand_class for s in all_signals],
            "forecastability_score": [s.forecastability_score for s in all_signals],
        })

        # Aggregate
        scores = [s.forecastability_score for s in all_signals]
        overall = float(np.mean(scores)) if scores else 0.0

        # Score distribution
        high = sum(1 for s in scores if s >= 0.6)
        medium = sum(1 for s in scores if 0.3 <= s < 0.6)
        low = sum(1 for s in scores if s < 0.3)
        score_dist = {"high": high, "medium": medium, "low": low}

        # Demand class distribution
        class_dist: Dict[str, int] = {}
        for s in all_signals:
            class_dist[s.demand_class] = class_dist.get(s.demand_class, 0) + 1

        return ForecastabilityReport(
            overall_score=overall,
            n_series=len(all_signals),
            score_distribution=score_dist,
            demand_class_distribution=class_dist,
            per_series=per_series,
            warnings=warnings,
        )
