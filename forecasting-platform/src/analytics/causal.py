"""
Causal / econometric analysis layer.

Provides three capabilities that complement the forecasting pipeline:

1. **Price Elasticity Estimation** — log-log regression to estimate the
   percentage change in demand for a 1% change in price, per series or
   aggregated across a group.

2. **Cannibalization Detection** — identifies pairs of products whose
   demand is negatively correlated after controlling for trend and
   seasonality, suggesting that one product's gain comes at the expense
   of another.

3. **Promotional Lift Estimation** — compares demand during promoted vs
   non-promoted periods (using a causal difference-in-differences style
   approach where possible, or simple lift ratios) to quantify the
   incremental volume driven by promotions.

All methods are implemented in pure NumPy/Polars (no external causal
inference library required).  The outputs are Polars DataFrames suitable
for BI export, SHAP overlay, or feeding back into the forecast pipeline
as informed priors.

Usage
-----
>>> from src.analytics.causal import CausalAnalyzer
>>> ca = CausalAnalyzer()
>>> elasticities = ca.estimate_price_elasticity(df, price_col="unit_price")
>>> cannib = ca.detect_cannibalization(df, group_col="category")
>>> lifts = ca.estimate_promo_lift(df, promo_col="promo_flag")
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


class CausalAnalyzer:
    """Causal and econometric analysis for retail demand data.

    Parameters
    ----------
    season_length:
        Number of periods in one seasonal cycle (52 for weekly).
    min_observations:
        Minimum number of observations required per series for
        elasticity or lift estimation.
    """

    def __init__(self, season_length: int = 52, min_observations: int = 20):
        self.season_length = season_length
        self.min_observations = min_observations

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Price Elasticity
    # ─────────────────────────────────────────────────────────────────────────

    def estimate_price_elasticity(
        self,
        df: pl.DataFrame,
        price_col: str = "unit_price",
        target_col: str = "quantity",
        id_col: str = "series_id",
        time_col: str = "week",
        control_cols: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        """Estimate own-price elasticity per series using log-log regression.

        Elasticity = d(ln Q) / d(ln P), estimated via OLS on:
            ln(Q) = β₀ + β₁·ln(P) + β₂·trend + Σβₖ·controls + ε

        Parameters
        ----------
        df:
            Panel data with price and quantity columns.
        price_col:
            Column containing unit price.
        target_col:
            Column containing demand/quantity.
        id_col:
            Series identifier column.
        control_cols:
            Optional additional control variables (e.g. promo flags).

        Returns
        -------
        DataFrame with columns:
          [id_col, "elasticity", "elasticity_se", "p_value",
           "r_squared", "n_obs", "is_significant", "interpretation"]
        """
        if price_col not in df.columns:
            raise ValueError(
                f"Price column {price_col!r} not found. "
                f"Available: {df.columns}"
            )

        control_cols = control_cols or []
        results = []

        for sid in df[id_col].unique().sort().to_list():
            series = df.filter(pl.col(id_col) == sid).sort(time_col)

            q = series[target_col].to_numpy().astype(float)
            p = series[price_col].to_numpy().astype(float)

            # Filter out zero/negative prices and quantities
            mask = (q > 0) & (p > 0)
            q = q[mask]
            p = p[mask]

            if len(q) < self.min_observations:
                results.append({
                    id_col: sid,
                    "elasticity": None,
                    "elasticity_se": None,
                    "p_value": None,
                    "r_squared": None,
                    "n_obs": int(len(q)),
                    "is_significant": False,
                    "interpretation": "Insufficient data",
                })
                continue

            ln_q = np.log(q)
            ln_p = np.log(p)

            # Build design matrix: [intercept, ln(price), trend, ...controls]
            n = len(ln_q)
            trend = np.arange(n, dtype=float) / n
            X = np.column_stack([np.ones(n), ln_p, trend])

            # Add control columns if available
            for cc in control_cols:
                if cc in series.columns:
                    ctrl = series[cc].to_numpy().astype(float)[mask]
                    if len(ctrl) == n:
                        X = np.column_stack([X, ctrl])

            # OLS: β = (X'X)^{-1} X'y
            try:
                XtX = X.T @ X
                XtX_inv = np.linalg.inv(XtX + 1e-10 * np.eye(X.shape[1]))
                beta = XtX_inv @ (X.T @ ln_q)

                # Residuals and standard errors
                residuals = ln_q - X @ beta
                sigma2 = np.sum(residuals**2) / max(n - X.shape[1], 1)
                se = np.sqrt(np.diag(sigma2 * XtX_inv))

                elasticity = float(beta[1])
                elasticity_se = float(se[1])

                # t-statistic and p-value (two-sided)
                t_stat = elasticity / max(elasticity_se, 1e-10)
                # Approximate p-value using normal distribution
                p_value = float(2 * (1 - _normal_cdf(abs(t_stat))))

                # R-squared
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((ln_q - np.mean(ln_q))**2)
                r_squared = float(1 - ss_res / max(ss_tot, 1e-10))

                # Interpretation
                if p_value > 0.05:
                    interp = "Not statistically significant"
                elif elasticity < -1:
                    interp = "Elastic — demand highly sensitive to price"
                elif elasticity < 0:
                    interp = "Inelastic — demand moderately sensitive to price"
                elif elasticity == 0:
                    interp = "Perfectly inelastic — price has no effect"
                else:
                    interp = "Positive elasticity — Giffen/Veblen good or data issue"

                results.append({
                    id_col: sid,
                    "elasticity": elasticity,
                    "elasticity_se": elasticity_se,
                    "p_value": p_value,
                    "r_squared": r_squared,
                    "n_obs": n,
                    "is_significant": p_value <= 0.05,
                    "interpretation": interp,
                })

            except np.linalg.LinAlgError:
                results.append({
                    id_col: sid,
                    "elasticity": None,
                    "elasticity_se": None,
                    "p_value": None,
                    "r_squared": None,
                    "n_obs": int(n),
                    "is_significant": False,
                    "interpretation": "Singular matrix — insufficient price variation",
                })

        return pl.DataFrame(results)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Cannibalization Detection
    # ─────────────────────────────────────────────────────────────────────────

    def detect_cannibalization(
        self,
        df: pl.DataFrame,
        target_col: str = "quantity",
        id_col: str = "series_id",
        time_col: str = "week",
        group_col: Optional[str] = None,
        correlation_threshold: float = -0.3,
    ) -> pl.DataFrame:
        """Detect cannibalization between product pairs.

        Computes pairwise correlations on de-trended, de-seasonalized
        demand residuals.  Strong negative correlations (below threshold)
        suggest cannibalization — one product's demand increases when
        another's decreases.

        Parameters
        ----------
        df:
            Panel data.
        group_col:
            Optional grouping column (e.g. category).  If provided,
            cannibalization is only checked within groups.
        correlation_threshold:
            Correlation below this value flags a pair as cannibalizing.

        Returns
        -------
        DataFrame with columns:
          ["series_a", "series_b", "group", "residual_correlation",
           "raw_correlation", "is_cannibalizing", "strength"]
        """
        results = []

        # Determine groups to analyze within
        if group_col and group_col in df.columns:
            groups = df[group_col].unique().sort().to_list()
        else:
            groups = [None]

        for group in groups:
            if group is not None:
                subset = df.filter(pl.col(group_col) == group)
            else:
                subset = df

            series_ids = sorted(subset[id_col].unique().to_list())
            if len(series_ids) < 2:
                continue

            # Build a time-aligned residual matrix
            residual_map: Dict[str, np.ndarray] = {}
            raw_map: Dict[str, np.ndarray] = {}
            common_times = None

            for sid in series_ids:
                s = subset.filter(pl.col(id_col) == sid).sort(time_col)
                times = set(s[time_col].to_list())
                if common_times is None:
                    common_times = times
                else:
                    common_times = common_times & times

            if not common_times or len(common_times) < self.min_observations:
                continue

            sorted_times = sorted(common_times)

            for sid in series_ids:
                s = (
                    subset.filter(pl.col(id_col) == sid)
                    .filter(pl.col(time_col).is_in(sorted_times))
                    .sort(time_col)
                )
                vals = s[target_col].to_numpy().astype(float)
                raw_map[sid] = vals
                residual_map[sid] = self._detrend_deseason(vals)

            # Pairwise correlations
            for i, sid_a in enumerate(series_ids):
                for sid_b in series_ids[i + 1:]:
                    if sid_a not in residual_map or sid_b not in residual_map:
                        continue

                    res_a = residual_map[sid_a]
                    res_b = residual_map[sid_b]
                    raw_a = raw_map[sid_a]
                    raw_b = raw_map[sid_b]

                    if len(res_a) != len(res_b) or len(res_a) < 3:
                        continue

                    resid_corr = float(np.corrcoef(res_a, res_b)[0, 1])
                    raw_corr = float(np.corrcoef(raw_a, raw_b)[0, 1])

                    is_cannib = resid_corr < correlation_threshold

                    if resid_corr < -0.6:
                        strength = "strong"
                    elif resid_corr < -0.3:
                        strength = "moderate"
                    elif resid_corr < -0.1:
                        strength = "weak"
                    else:
                        strength = "none"

                    results.append({
                        "series_a": sid_a,
                        "series_b": sid_b,
                        "group": group or "all",
                        "residual_correlation": round(resid_corr, 4),
                        "raw_correlation": round(raw_corr, 4),
                        "is_cannibalizing": is_cannib,
                        "strength": strength,
                    })

        if not results:
            return pl.DataFrame(schema={
                "series_a": pl.Utf8, "series_b": pl.Utf8,
                "group": pl.Utf8, "residual_correlation": pl.Float64,
                "raw_correlation": pl.Float64, "is_cannibalizing": pl.Boolean,
                "strength": pl.Utf8,
            })

        return pl.DataFrame(results).sort("residual_correlation")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Promotional Lift Estimation
    # ─────────────────────────────────────────────────────────────────────────

    def estimate_promo_lift(
        self,
        df: pl.DataFrame,
        promo_col: str = "promo_flag",
        target_col: str = "quantity",
        id_col: str = "series_id",
        time_col: str = "week",
        price_col: Optional[str] = None,
    ) -> pl.DataFrame:
        """Estimate promotional lift per series.

        Compares demand during promoted vs non-promoted periods.  When a
        price column is available, also estimates the price-adjusted lift
        (controlling for price changes that typically accompany promos).

        The lift ratio is: mean(demand | promo) / mean(demand | no promo).
        A lift of 1.5 means promos drive 50% more demand on average.

        For statistical rigor, a Welch t-test is performed comparing
        promo vs non-promo demand.

        Parameters
        ----------
        df:
            Panel data with a binary promo indicator.
        promo_col:
            Column containing promo flag (1/True = promoted, 0/False = not).
            Can also be a float (promo intensity).
        price_col:
            Optional price column for price-controlled lift.

        Returns
        -------
        DataFrame with columns:
          [id_col, "baseline_demand", "promo_demand", "lift_ratio",
           "lift_pct", "incremental_volume_per_week", "n_promo_weeks",
           "n_base_weeks", "t_statistic", "p_value", "is_significant",
           "price_adjusted_lift"]
        """
        if promo_col not in df.columns:
            raise ValueError(
                f"Promo column {promo_col!r} not found. "
                f"Available: {df.columns}"
            )

        results = []

        for sid in df[id_col].unique().sort().to_list():
            series = df.filter(pl.col(id_col) == sid).sort(time_col)

            demand = series[target_col].to_numpy().astype(float)
            promo = series[promo_col].to_numpy().astype(float)

            # Binary promo mask (threshold at 0.5 for intensity columns)
            promo_mask = promo > 0.5 if promo.max() > 1 else promo > 0
            no_promo_mask = ~promo_mask

            n_promo = int(promo_mask.sum())
            n_base = int(no_promo_mask.sum())

            if n_promo < 3 or n_base < 3:
                results.append({
                    id_col: sid,
                    "baseline_demand": float(np.mean(demand[no_promo_mask])) if n_base > 0 else None,
                    "promo_demand": float(np.mean(demand[promo_mask])) if n_promo > 0 else None,
                    "lift_ratio": None,
                    "lift_pct": None,
                    "incremental_volume_per_week": None,
                    "n_promo_weeks": n_promo,
                    "n_base_weeks": n_base,
                    "t_statistic": None,
                    "p_value": None,
                    "is_significant": False,
                    "price_adjusted_lift": None,
                })
                continue

            base_demand = demand[no_promo_mask]
            promo_demand = demand[promo_mask]
            mean_base = float(np.mean(base_demand))
            mean_promo = float(np.mean(promo_demand))

            lift_ratio = mean_promo / max(mean_base, 1e-10)
            lift_pct = (lift_ratio - 1.0) * 100
            incremental = mean_promo - mean_base

            # Welch t-test (unequal variance)
            t_stat, p_value = _welch_t_test(promo_demand, base_demand)

            # Price-adjusted lift
            price_adj_lift = None
            if price_col and price_col in series.columns:
                price = series[price_col].to_numpy().astype(float)
                price_promo = price[promo_mask]
                price_base = price[no_promo_mask]

                mean_price_promo = np.mean(price_promo) if len(price_promo) > 0 else 0
                mean_price_base = np.mean(price_base) if len(price_base) > 0 else 0

                if mean_price_base > 0 and mean_price_promo > 0:
                    # Revenue-per-unit adjusted lift
                    # Adjust for price difference to isolate volume effect
                    price_ratio = mean_price_promo / mean_price_base
                    price_adj_lift = float(lift_ratio / max(price_ratio, 1e-10))

            results.append({
                id_col: sid,
                "baseline_demand": round(mean_base, 2),
                "promo_demand": round(mean_promo, 2),
                "lift_ratio": round(lift_ratio, 4),
                "lift_pct": round(lift_pct, 1),
                "incremental_volume_per_week": round(incremental, 2),
                "n_promo_weeks": n_promo,
                "n_base_weeks": n_base,
                "t_statistic": round(t_stat, 4),
                "p_value": round(p_value, 6),
                "is_significant": p_value <= 0.05,
                "price_adjusted_lift": round(price_adj_lift, 4) if price_adj_lift is not None else None,
            })

        return pl.DataFrame(results)

    # ─────────────────────────────────────────────────────────────────────────
    # Summary / combined analysis
    # ─────────────────────────────────────────────────────────────────────────

    def full_causal_report(
        self,
        df: pl.DataFrame,
        target_col: str = "quantity",
        id_col: str = "series_id",
        time_col: str = "week",
        price_col: Optional[str] = None,
        promo_col: Optional[str] = None,
        group_col: Optional[str] = None,
    ) -> Dict[str, pl.DataFrame]:
        """Run all available causal analyses and return a dict of results.

        Parameters
        ----------
        df:
            Panel data.
        price_col:
            If present, runs price elasticity estimation.
        promo_col:
            If present, runs promotional lift estimation.
        group_col:
            If present, runs cannibalization detection within groups.

        Returns
        -------
        Dict with keys "elasticity", "cannibalization", "promo_lift"
        (only keys for which the required columns exist).
        """
        report: Dict[str, pl.DataFrame] = {}

        if price_col and price_col in df.columns:
            logger.info("Running price elasticity estimation...")
            report["elasticity"] = self.estimate_price_elasticity(
                df, price_col=price_col, target_col=target_col,
                id_col=id_col, time_col=time_col,
            )
            n_sig = report["elasticity"].filter(pl.col("is_significant")).height
            logger.info(
                "Price elasticity: %d/%d series have significant elasticity",
                n_sig, report["elasticity"].height,
            )

        report["cannibalization"] = self.detect_cannibalization(
            df, target_col=target_col, id_col=id_col,
            time_col=time_col, group_col=group_col,
        )
        n_cannib = report["cannibalization"].filter(pl.col("is_cannibalizing")).height
        logger.info(
            "Cannibalization: %d product pairs detected", n_cannib,
        )

        if promo_col and promo_col in df.columns:
            logger.info("Running promotional lift estimation...")
            report["promo_lift"] = self.estimate_promo_lift(
                df, promo_col=promo_col, target_col=target_col,
                id_col=id_col, time_col=time_col, price_col=price_col,
            )
            sig_lifts = report["promo_lift"].filter(pl.col("is_significant"))
            if not sig_lifts.is_empty():
                avg_lift = sig_lifts["lift_pct"].mean()
                logger.info(
                    "Promo lift: %d/%d series significant, avg lift %.1f%%",
                    sig_lifts.height, report["promo_lift"].height,
                    avg_lift if avg_lift is not None else 0,
                )

        return report

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _detrend_deseason(self, values: np.ndarray) -> np.ndarray:
        """Remove trend and seasonality to get residuals for correlation."""
        n = len(values)
        if n < 4:
            return values - np.mean(values)

        # Remove trend (linear)
        x = np.arange(n, dtype=float)
        coeffs = np.polyfit(x, values, 1)
        trend = np.polyval(coeffs, x)
        detrended = values - trend

        # Remove seasonality (average per seasonal position)
        sl = min(self.season_length, n // 2)
        if sl < 2:
            return detrended

        seasonal = np.zeros(n)
        for pos in range(sl):
            indices = np.arange(pos, n, sl)
            seasonal[indices] = np.mean(detrended[indices])

        residual = detrended - seasonal
        return residual


# ─────────────────────────────────────────────────────────────────────────────
# Pure-NumPy statistical helpers (no scipy dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF using Abramowitz & Stegun (7.1.26)."""
    # Good to ~1e-5 accuracy
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    t = 1.0 / (1.0 + 0.2316419 * x)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 +
           t * (-1.821255978 + t * 1.330274429))))
    cdf = 1.0 - poly * np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)
    return 0.5 + sign * (cdf - 0.5)


def _welch_t_test(
    a: np.ndarray, b: np.ndarray,
) -> Tuple[float, float]:
    """Welch's t-test for unequal variances (two-sided).

    Returns (t_statistic, p_value).  Uses normal approximation for
    large samples; exact for small samples would require scipy.
    """
    n_a, n_b = len(a), len(b)
    mean_a, mean_b = np.mean(a), np.mean(b)
    var_a = np.var(a, ddof=1) if n_a > 1 else 0.0
    var_b = np.var(b, ddof=1) if n_b > 1 else 0.0

    se = np.sqrt(var_a / max(n_a, 1) + var_b / max(n_b, 1))
    if se < 1e-10:
        return 0.0, 1.0

    t_stat = float((mean_a - mean_b) / se)
    # Normal approximation for p-value (accurate for df > 30)
    p_value = float(2 * (1 - _normal_cdf(abs(t_stat))))
    return t_stat, p_value
