"""
Neural forecasters via neuralforecast (Nixtla).

Wraps N-BEATS, NHITS, and TFT to provide the BaseForecaster interface.
These deep learning models excel at long-horizon forecasting and can
capture complex non-linear patterns that statistical models miss.

All three models require ``neuralforecast`` and ``pytorch-lightning``.
Install with::

    pip install neuralforecast

Usage
-----
>>> from src.forecasting.neural import NBEATSForecaster
>>> f = NBEATSForecaster(max_steps=500)
>>> f.fit(train_df)
>>> forecast = f.predict(13)
>>> intervals = f.predict_quantiles(13, [0.1, 0.5, 0.9])
"""

import logging
from typing import Any, Dict, List, Optional

import polars as pl

from .base import BaseForecaster
from .registry import registry

logger = logging.getLogger(__name__)

# Attempt to import neuralforecast; fall back gracefully
try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATS as _NBEATS
    from neuralforecast.models import NHITS as _NHITS
    from neuralforecast.models import TFT as _TFT
    _HAS_NEURALFORECAST = True
except ImportError:
    _HAS_NEURALFORECAST = False


class _NeuralforecastBase(BaseForecaster):
    """
    Shared logic for neuralforecast-backed models.

    Handles the Polars <-> pandas conversion that neuralforecast requires,
    and maps columns to the expected ``unique_id / ds / y`` schema.

    Subclasses only need to implement ``_get_model(horizon)`` which returns
    a configured neuralforecast model instance.
    """

    def __init__(
        self,
        max_steps: int = 500,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        val_check_steps: int = 50,
        random_seed: int = 42,
        accelerator: str = "cpu",
        enable_progress_bar: bool = False,
        quantiles: Optional[List[float]] = None,
    ):
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.val_check_steps = val_check_steps
        self.random_seed = random_seed
        self.accelerator = accelerator
        self.enable_progress_bar = enable_progress_bar
        self._quantiles = quantiles

        self._nf: Optional[Any] = None
        self._id_col: str = "series_id"
        self._time_col: str = "week"
        self._target_col: str = "quantity"
        self._horizon: Optional[int] = None

    def _get_model(self, horizon: int):
        """Return a configured neuralforecast model instance."""
        raise NotImplementedError

    def _get_quantile_model(self, horizon: int, loss_quantiles: List[float]):
        """Return a model configured for quantile regression."""
        raise NotImplementedError

    def fit(
        self,
        df: pl.DataFrame,
        target_col: str = "quantity",
        time_col: str = "week",
        id_col: str = "series_id",
    ) -> None:
        if not _HAS_NEURALFORECAST:
            raise ImportError(
                f"{self.name} requires 'neuralforecast'. "
                "Install with: pip install neuralforecast"
            )

        self._id_col = id_col
        self._time_col = time_col
        self._target_col = target_col

        # neuralforecast expects pandas with columns: unique_id, ds, y
        pdf = (
            df.select([id_col, time_col, target_col])
            .rename({id_col: "unique_id", time_col: "ds", target_col: "y"})
            .to_pandas()
        )
        pdf["ds"] = pdf["ds"].astype("datetime64[ns]")

        # neuralforecast needs the horizon at model construction time.
        # Infer a sensible default from the data: use 13 weeks (one quarter).
        # The actual horizon is set at predict() time — if it differs from the
        # trained horizon, we refit (neuralforecast requires h at init).
        self._horizon = 13
        model = self._get_model(self._horizon)
        self._nf = NeuralForecast(
            models=[model],
            freq="W",
        )

        logger.info(
            "%s: fitting on %d series (max_steps=%d)…",
            self.name, pdf["unique_id"].nunique(), self.max_steps,
        )
        self._nf.fit(df=pdf)
        logger.info("%s: fit complete.", self.name)

    def predict(
        self,
        horizon: int,
        id_col: str = "series_id",
        time_col: str = "week",
    ) -> pl.DataFrame:
        if self._nf is None:
            raise RuntimeError(f"{self.name}: call fit() before predict()")

        result_pdf = self._nf.predict()

        # Bring index columns back if needed
        if "unique_id" not in result_pdf.columns or "ds" not in result_pdf.columns:
            result_pdf = result_pdf.reset_index()

        result = pl.from_pandas(result_pdf)

        # neuralforecast names prediction column after model class
        pred_cols = [
            c for c in result.columns
            if c not in ("unique_id", "ds")
            and "-median" not in c
            and "-lo-" not in c
            and "-hi-" not in c
        ]
        if pred_cols:
            result = result.rename({pred_cols[0]: "forecast"})

        result = result.rename({"unique_id": id_col, "ds": time_col})
        result = result.select([id_col, time_col, "forecast"])

        if result[time_col].dtype != pl.Date:
            result = result.with_columns(pl.col(time_col).cast(pl.Date))

        # Trim to requested horizon (neuralforecast always produces h steps)
        result = self._trim_horizon(result, horizon, id_col, time_col)
        return result

    def predict_quantiles(
        self,
        horizon: int,
        quantiles: List[float],
        id_col: str = "series_id",
        time_col: str = "week",
    ) -> pl.DataFrame:
        """
        Quantile forecasts via neuralforecast prediction intervals.

        neuralforecast supports ``level`` parameter for prediction intervals,
        similar to statsforecast.  A ``level=80`` interval covers [P10, P90].
        """
        if self._nf is None:
            raise RuntimeError(f"{self.name}: call fit() before predict_quantiles()")

        # Determine coverage levels from lower quantiles
        lower_qs = sorted(q for q in quantiles if q < 0.5 - 1e-6)
        levels = sorted({int(round((1 - 2 * q) * 100)) for q in lower_qs}) if lower_qs else [80]

        result_pdf = self._nf.predict(level=levels)

        if "unique_id" not in result_pdf.columns or "ds" not in result_pdf.columns:
            result_pdf = result_pdf.reset_index()

        result = pl.from_pandas(result_pdf)
        result = result.rename({"unique_id": id_col, "ds": time_col})
        if result[time_col].dtype != pl.Date:
            result = result.with_columns(pl.col(time_col).cast(pl.Date))

        # Find point forecast column (no -lo- / -hi- suffix)
        point_col = next(
            (c for c in result.columns
             if c not in (id_col, time_col)
             and "-lo-" not in c and "-hi-" not in c
             and "-median" not in c),
            None,
        )

        output = result.select([id_col, time_col])
        for q in quantiles:
            col = f"forecast_p{int(round(q * 100))}"
            if abs(q - 0.5) < 1e-6:
                output = output.with_columns(result[point_col].alias(col))
            elif q < 0.5:
                level = int(round((1 - 2 * q) * 100))
                lo = next((c for c in result.columns if f"-lo-{level}" in c), None)
                src = result[lo] if lo else result[point_col]
                output = output.with_columns(src.alias(col))
            else:
                mirror_q = 1.0 - q
                level = int(round((1 - 2 * mirror_q) * 100))
                hi = next((c for c in result.columns if f"-hi-{level}" in c), None)
                src = result[hi] if hi else result[point_col]
                output = output.with_columns(src.alias(col))

        output = self._trim_horizon(output, horizon, id_col, time_col)
        return output

    def _trim_horizon(
        self, df: pl.DataFrame, horizon: int, id_col: str, time_col: str,
    ) -> pl.DataFrame:
        """Keep only the first ``horizon`` steps per series."""
        if self._horizon is not None and horizon < self._horizon:
            df = (
                df.sort([id_col, time_col])
                .with_columns(
                    pl.col(time_col).cum_count().over(id_col).alias("_step")
                )
                .filter(pl.col("_step") <= horizon)
                .drop("_step")
            )
        return df


# ── N-BEATS ──────────────────────────────────────────────────────────────────────


@registry.register("nbeats")
class NBEATSForecaster(_NeuralforecastBase):
    """
    N-BEATS (Neural Basis Expansion Analysis) via neuralforecast.

    Pure deep learning architecture using backward and forward residual links
    with basis expansion.  Strong on M3/M4/M5 benchmarks.  Works well without
    external features — learns directly from the time series.

    Parameters
    ----------
    max_steps:
        Maximum training iterations.  500–1000 is typical.
    learning_rate:
        Adam learning rate.
    input_size_multiplier:
        Input window = ``input_size_multiplier × horizon``.
    stack_types:
        N-BEATS stack configuration.  Default uses both trend and
        seasonality interpretable stacks.
    n_blocks:
        Blocks per stack.
    """

    name = "nbeats"

    def __init__(
        self,
        max_steps: int = 500,
        learning_rate: float = 1e-3,
        input_size_multiplier: int = 2,
        stack_types: Optional[List[str]] = None,
        n_blocks: Optional[List[int]] = None,
        batch_size: int = 32,
        accelerator: str = "cpu",
        random_seed: int = 42,
    ):
        super().__init__(
            max_steps=max_steps,
            learning_rate=learning_rate,
            batch_size=batch_size,
            accelerator=accelerator,
            random_seed=random_seed,
        )
        self.input_size_multiplier = input_size_multiplier
        self.stack_types = stack_types or ["trend", "seasonality"]
        self.n_blocks = n_blocks or [1, 1]

    def _get_model(self, horizon: int):
        return _NBEATS(
            h=horizon,
            input_size=self.input_size_multiplier * horizon,
            stack_types=self.stack_types,
            n_blocks=self.n_blocks,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            accelerator=self.accelerator,
            random_seed=self.random_seed,
            enable_progress_bar=self.enable_progress_bar,
        )

    def get_params(self) -> Dict[str, Any]:
        return {
            "model": "NBEATS",
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "input_size_multiplier": self.input_size_multiplier,
            "stack_types": self.stack_types,
        }


# ── N-HiTS ───────────────────────────────────────────────────────────────────────


@registry.register("nhits")
class NHITSForecaster(_NeuralforecastBase):
    """
    N-HiTS (Neural Hierarchical Interpolation for Time Series) via neuralforecast.

    Extension of N-BEATS with multi-rate signal sampling and hierarchical
    interpolation.  Significantly faster training and better long-horizon
    performance than N-BEATS.  State-of-the-art on M-competition benchmarks.

    Parameters
    ----------
    max_steps:
        Maximum training iterations.
    learning_rate:
        Adam learning rate.
    input_size_multiplier:
        Input window = ``input_size_multiplier × horizon``.
    n_blocks:
        Blocks per stack.
    """

    name = "nhits"

    def __init__(
        self,
        max_steps: int = 500,
        learning_rate: float = 1e-3,
        input_size_multiplier: int = 2,
        n_blocks: Optional[List[int]] = None,
        batch_size: int = 32,
        accelerator: str = "cpu",
        random_seed: int = 42,
    ):
        super().__init__(
            max_steps=max_steps,
            learning_rate=learning_rate,
            batch_size=batch_size,
            accelerator=accelerator,
            random_seed=random_seed,
        )
        self.input_size_multiplier = input_size_multiplier
        self.n_blocks = n_blocks or [1, 1, 1]

    def _get_model(self, horizon: int):
        return _NHITS(
            h=horizon,
            input_size=self.input_size_multiplier * horizon,
            n_blocks=self.n_blocks,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            accelerator=self.accelerator,
            random_seed=self.random_seed,
            enable_progress_bar=self.enable_progress_bar,
        )

    def get_params(self) -> Dict[str, Any]:
        return {
            "model": "NHITS",
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "input_size_multiplier": self.input_size_multiplier,
            "n_blocks": self.n_blocks,
        }


# ── TFT ──────────────────────────────────────────────────────────────────────────


@registry.register("tft")
class TFTForecaster(_NeuralforecastBase):
    """
    Temporal Fusion Transformer (TFT) via neuralforecast.

    Attention-based architecture that combines high-performance multi-horizon
    forecasting with interpretable temporal attention weights and variable
    importance scores.  Excels when external regressors (promotions, holidays,
    price) are available.

    Parameters
    ----------
    max_steps:
        Maximum training iterations.
    learning_rate:
        Adam learning rate.
    input_size_multiplier:
        Input window = ``input_size_multiplier × horizon``.
    hidden_size:
        Size of the hidden state in the LSTM encoder and GRN layers.
    n_head:
        Number of attention heads in the multi-head attention layer.
    """

    name = "tft"

    def __init__(
        self,
        max_steps: int = 500,
        learning_rate: float = 1e-3,
        input_size_multiplier: int = 2,
        hidden_size: int = 64,
        n_head: int = 4,
        batch_size: int = 32,
        accelerator: str = "cpu",
        random_seed: int = 42,
    ):
        super().__init__(
            max_steps=max_steps,
            learning_rate=learning_rate,
            batch_size=batch_size,
            accelerator=accelerator,
            random_seed=random_seed,
        )
        self.input_size_multiplier = input_size_multiplier
        self.hidden_size = hidden_size
        self.n_head = n_head

    def _get_model(self, horizon: int):
        return _TFT(
            h=horizon,
            input_size=self.input_size_multiplier * horizon,
            hidden_size=self.hidden_size,
            n_head=self.n_head,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            accelerator=self.accelerator,
            random_seed=self.random_seed,
            enable_progress_bar=self.enable_progress_bar,
        )

    def get_params(self) -> Dict[str, Any]:
        return {
            "model": "TFT",
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "hidden_size": self.hidden_size,
            "n_head": self.n_head,
        }
