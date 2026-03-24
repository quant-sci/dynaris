"""High-level DLM API — the user-friendly entry point for dynaris.

Example::

    from dynaris import LocalLevel, Seasonal, DLM

    model = LocalLevel() + Seasonal(period=12)
    dlm = DLM(model)
    dlm.fit(y)
    dlm.forecast(steps=12)
    dlm.plot()
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array

from dynaris.core.results import FilterResult, SmootherResult
from dynaris.core.state_space import StateSpaceModel
from dynaris.core.types import GaussianState
from dynaris.estimation.diagnostics import (
    ljung_box,
    standardized_residuals,
)
from dynaris.filters.kalman import kalman_filter
from dynaris.forecast.forecast import (
    ForecastResult,
    confidence_bands,
    fit_batch,
    forecast_from_filter,
    forecast_from_smoother,
)
from dynaris.smoothers.rts import rts_smooth


def _to_jax_2d(y: Any) -> tuple[Array, pd.DatetimeIndex | None]:
    """Convert input to a 2D JAX array (T, obs_dim) and extract index."""
    index: pd.DatetimeIndex | None = None

    if isinstance(y, (pd.DataFrame, pd.Series)):
        if isinstance(y.index, pd.DatetimeIndex):
            index = y.index
        y = y.values

    arr = jnp.asarray(np.asarray(y, dtype=np.float64))
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr, index


class DLM:
    """High-level Dynamic Linear Model interface.

    Wraps a :class:`StateSpaceModel` with convenient fit/forecast/plot methods.

    Args:
        model: A ``StateSpaceModel``, typically built by composing
            components like ``LocalLevel() + Seasonal(period=12)``.

    Example::

        from dynaris import LocalLevel, Seasonal, DLM

        model = LocalLevel() + Seasonal(period=12)
        dlm = DLM(model)
        dlm.fit(y)
        print(dlm.summary())
        forecast_df = dlm.forecast(steps=12)
        dlm.plot()
    """

    def __init__(self, model: StateSpaceModel) -> None:
        self._model = model
        self._filter_result: FilterResult | None = None
        self._smoother_result: SmootherResult | None = None
        self._forecast_result: ForecastResult | None = None
        self._observations: Array | None = None
        self._index: pd.DatetimeIndex | None = None
        self._is_fitted = False

    # --- Properties ---

    @property
    def model(self) -> StateSpaceModel:
        """The underlying state-space model."""
        return self._model

    @property
    def filter_result(self) -> FilterResult:
        """Filter result from the last ``fit()`` call."""
        if self._filter_result is None:
            msg = "Model not fitted. Call .fit() first."
            raise RuntimeError(msg)
        return self._filter_result

    @property
    def smoother_result(self) -> SmootherResult:
        """Smoother result from the last ``smooth()`` call."""
        if self._smoother_result is None:
            msg = "Model not smoothed. Call .smooth() first."
            raise RuntimeError(msg)
        return self._smoother_result

    # --- Core methods ---

    def fit(
        self,
        y: Any,
        initial_state: GaussianState | None = None,
    ) -> DLM:
        """Fit the model via Kalman filtering.

        Accepts ``jax.numpy``, ``numpy`` arrays, or ``pandas``
        DataFrames/Series. If a pandas object with a DatetimeIndex
        is provided, the index is preserved for forecast output.

        Args:
            y: Observations, shape (T,) or (T, obs_dim).
            initial_state: Optional custom initial state.

        Returns:
            self (for method chaining).
        """
        obs, self._index = _to_jax_2d(y)
        self._observations = obs
        self._filter_result = kalman_filter(
            self._model, obs, initial_state=initial_state
        )
        self._smoother_result = None
        self._forecast_result = None
        self._is_fitted = True
        return self

    def smooth(self) -> DLM:
        """Run RTS backward smoother on the filtered results.

        Returns:
            self (for method chaining).
        """
        fr = self.filter_result
        self._smoother_result = rts_smooth(self._model, fr)
        return self

    def forecast(self, steps: int = 1) -> pd.DataFrame:
        """Multi-step-ahead forecast from the last filtered state.

        Args:
            steps: Number of steps ahead.

        Returns:
            DataFrame with columns ``mean``, ``lower``, ``upper``
            (95% confidence). If the model was fit with a DatetimeIndex,
            the forecast index continues the series.
        """
        fr = self.filter_result
        fc = forecast_from_filter(self._model, fr, steps=steps)
        self._forecast_result = fc
        return self._forecast_to_df(fc, steps)

    def forecast_smoothed(self, steps: int = 1) -> pd.DataFrame:
        """Forecast from the last smoothed state.

        Args:
            steps: Number of steps ahead.

        Returns:
            DataFrame with ``mean``, ``lower``, ``upper``.
        """
        sr = self.smoother_result
        fc = forecast_from_smoother(self._model, sr, steps=steps)
        self._forecast_result = fc
        return self._forecast_to_df(fc, steps)

    def fit_batch(
        self, y_batch: Any
    ) -> FilterResult:
        """Fit multiple series in parallel via ``jax.vmap``.

        Args:
            y_batch: Batch of observations, shape (n_series, T, obs_dim).

        Returns:
            FilterResult with batch dimensions prepended.
        """
        if isinstance(y_batch, (pd.DataFrame, pd.Series)):
            y_batch = y_batch.values
        arr = jnp.asarray(np.asarray(y_batch, dtype=np.float64))
        if arr.ndim == 2:
            arr = arr[:, :, None]
        return fit_batch(self._model, arr)

    # --- Diagnostics ---

    def residuals(self) -> Array:
        """Standardized one-step-ahead prediction residuals."""
        return standardized_residuals(self.filter_result, self._model)

    def ljung_box(self, n_lags: int = 10) -> tuple[float, float]:
        """Ljung-Box test on standardized residuals.

        Returns:
            (test_statistic, p_value).
        """
        return ljung_box(self.residuals(), n_lags=n_lags)

    def summary(self) -> str:
        """Print a summary of the fitted model."""
        lines = [
            "DLM Summary",
            "=" * 40,
            f"State dim:       {self._model.state_dim}",
            f"Observation dim: {self._model.obs_dim}",
            f"Has input (B):   {self._model.input_matrix is not None}",
        ]
        if self._is_fitted:
            ll = float(self.filter_result.log_likelihood)
            n = int(self.filter_result.observations.shape[0])
            lines.append(f"Observations:    {n}")
            lines.append(f"Log-likelihood:  {ll:.4f}")
            lines.append(f"AIC:             {-2 * ll + 2 * self._model.state_dim:.4f}")

            resids = self.residuals()
            resids_np = np.asarray(resids)
            if resids_np.ndim > 1:
                resids_np = resids_np[:, 0]
            lines.append(f"Residual mean:   {float(np.mean(resids_np)):.4f}")
            lines.append(f"Residual std:    {float(np.std(resids_np)):.4f}")

            q_stat, p_val = self.ljung_box()
            lines.append(f"Ljung-Box(10):   Q={q_stat:.2f}, p={p_val:.4f}")
        else:
            lines.append("(not fitted)")
        lines.append("=" * 40)
        return "\n".join(lines)

    # --- Plotting ---

    def plot(self, kind: str = "filtered", **kwargs: Any) -> Any:
        """Plot model results.

        Args:
            kind: One of ``"filtered"``, ``"smoothed"``, ``"forecast"``,
                ``"diagnostics"``, ``"components"``, ``"panel"``.
            **kwargs: Passed to the underlying plot function.

        Returns:
            matplotlib Figure.
        """
        from dynaris.plotting.plots import (
            plot_components,
            plot_diagnostics,
            plot_filtered,
            plot_forecast,
            plot_panel,
            plot_smoothed,
        )

        if kind == "filtered":
            return plot_filtered(self.filter_result, self._model, **kwargs)
        if kind == "smoothed":
            return plot_smoothed(self.smoother_result, self._model, **kwargs)
        if kind == "forecast":
            if self._forecast_result is None:
                msg = "No forecast available. Call .forecast() first."
                raise RuntimeError(msg)
            return plot_forecast(
                self.filter_result, self._forecast_result, self._model,
                **kwargs,
            )
        if kind == "diagnostics":
            return plot_diagnostics(self.filter_result, self._model, **kwargs)
        if kind == "components":
            if self._smoother_result is None:
                self.smooth()
            return plot_components(self.smoother_result, **kwargs)
        if kind == "panel":
            return plot_panel(
                self.filter_result,
                self._smoother_result,
                self._forecast_result,
                self._model,
                **kwargs,
            )
        msg = (
            f"Unknown plot kind: {kind!r}. "
            "Use 'filtered', 'smoothed', 'forecast', 'diagnostics', "
            "'components', or 'panel'."
        )
        raise ValueError(msg)

    # --- Output formatting ---

    def filtered_states_df(self) -> pd.DataFrame:
        """Filtered states as a pandas DataFrame."""
        fr = self.filter_result
        states = np.asarray(fr.filtered_states)
        cols = [f"state_{i}" for i in range(states.shape[1])]
        index = self._index if self._index is not None else None
        return pd.DataFrame(states, columns=cols, index=index)

    def smoothed_states_df(self) -> pd.DataFrame:
        """Smoothed states as a pandas DataFrame."""
        sr = self.smoother_result
        states = np.asarray(sr.smoothed_states)
        cols = [f"state_{i}" for i in range(states.shape[1])]
        index = self._index if self._index is not None else None
        return pd.DataFrame(states, columns=cols, index=index)

    # --- Private helpers ---

    def _forecast_to_df(
        self, fc: ForecastResult, steps: int
    ) -> pd.DataFrame:
        """Convert ForecastResult to a pandas DataFrame."""
        mean = np.asarray(fc.mean)
        lower_arr, upper_arr = confidence_bands(fc.mean, fc.covariance)
        lower = np.asarray(lower_arr)
        upper = np.asarray(upper_arr)

        if mean.ndim == 2 and mean.shape[1] == 1:
            mean = mean[:, 0]
            lower = lower[:, 0]
            upper = upper[:, 0]

        index: pd.DatetimeIndex | pd.RangeIndex | None = None
        if self._index is not None and len(self._index) > 0:
            freq = self._index.freq or pd.infer_freq(self._index)
            if freq is not None:
                last = self._index[-1]
                index = pd.date_range(
                    start=last + pd.tseries.frequencies.to_offset(freq),
                    periods=steps,
                    freq=freq,
                )

        return pd.DataFrame(
            {"mean": mean, "lower_95": lower, "upper_95": upper},
            index=index,
        )

    # --- Repr ---

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"DLM(state_dim={self._model.state_dim}, "
            f"obs_dim={self._model.obs_dim}, {status})"
        )
