"""Unified state-space model interface with automatic filter selection.

Provides a high-level API for both linear and nonlinear state-space models,
automatically selecting the best filtering algorithm based on the model type.

Example::

    from dynaris import SSM, StochasticVolatility, LocalLevel

    # Nonlinear model — auto-selects UKF
    model = StochasticVolatility()
    ssm = SSM(model)
    ssm.fit(y_star)

    # Linear model — auto-selects Kalman filter
    ssm = SSM(LocalLevel())
    ssm.fit(y)
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array

from dynaris.core.nonlinear import NonlinearSSM
from dynaris.core.results import FilterResult, SwitchingFilterResult
from dynaris.core.state_space import StateSpaceModel
from dynaris.core.switching import MarkovSwitchingSSM
from dynaris.core.types import GaussianState
from dynaris.filters.ekf import ekf_filter
from dynaris.filters.hamilton import hamilton_filter
from dynaris.filters.kalman import kalman_filter
from dynaris.filters.particle import particle_filter
from dynaris.filters.ukf import ukf_filter

_VALID_FILTERS = {"auto", "kalman", "ekf", "ukf", "particle", "hamilton"}
_LINEAR_FILTERS = {"kalman"}
_NONLINEAR_FILTERS = {"ekf", "ukf", "particle"}
_SWITCHING_FILTERS = {"hamilton"}


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


class SSM:
    """Unified state-space model interface with automatic filter selection.

    Accepts either a linear :class:`StateSpaceModel` or a nonlinear
    :class:`NonlinearSSM` and automatically selects the best filtering
    algorithm. For linear models, uses the Kalman filter. For nonlinear
    models, defaults to the Unscented Kalman Filter (UKF).

    Args:
        model: A ``StateSpaceModel`` (linear) or ``NonlinearSSM`` (nonlinear).
        filter: Filter algorithm. ``"auto"`` selects based on model type.
            Options: ``"auto"``, ``"kalman"``, ``"ekf"``, ``"ukf"``,
            ``"particle"``.
        key: JAX PRNG key for the particle filter. If ``None`` and
            ``filter="particle"``, uses ``jax.random.PRNGKey(0)``.
        **filter_kwargs: Extra arguments passed to the selected filter.
            For UKF: ``alpha``, ``beta``, ``kappa``.
            For particle filter: ``n_particles``, ``resampling``.

    Example::

        from dynaris import SSM, LocalLevel, StochasticVolatility

        # Linear model — auto-selects Kalman filter
        ssm = SSM(LocalLevel())
        ssm.fit(y)

        # Nonlinear model — auto-selects UKF
        ssm = SSM(StochasticVolatility(), filter="ukf")
        ssm.fit(y_star)

        # Force particle filter
        ssm = SSM(model, filter="particle", n_particles=2000)
        ssm.fit(y, key=jax.random.PRNGKey(42))
    """

    def __init__(
        self,
        model: StateSpaceModel | NonlinearSSM | MarkovSwitchingSSM,
        filter: str = "auto",
        *,
        key: Array | None = None,
        **filter_kwargs: Any,
    ) -> None:
        if not isinstance(model, (StateSpaceModel, NonlinearSSM, MarkovSwitchingSSM)):
            msg = (
                f"model must be a StateSpaceModel, NonlinearSSM, or "
                f"MarkovSwitchingSSM, got {type(model).__name__}"
            )
            raise TypeError(msg)

        if filter not in _VALID_FILTERS:
            msg = f"Unknown filter: {filter!r}. Use one of {sorted(_VALID_FILTERS)}."
            raise ValueError(msg)

        # Resolve auto-selection
        is_linear = isinstance(model, StateSpaceModel)
        is_switching = isinstance(model, MarkovSwitchingSSM)
        if filter == "auto":
            if is_switching:
                filter_name = "hamilton"
            elif is_linear:
                filter_name = "kalman"
            else:
                filter_name = "ukf"
        else:
            filter_name = filter

        # Validate filter/model compatibility
        if filter_name in _LINEAR_FILTERS and not is_linear:
            msg = f"Filter {filter_name!r} requires a StateSpaceModel."
            raise ValueError(msg)
        if filter_name in _NONLINEAR_FILTERS and (is_linear or is_switching):
            msg = f"Filter {filter_name!r} requires a NonlinearSSM."
            raise ValueError(msg)
        if filter_name in _SWITCHING_FILTERS and not is_switching:
            msg = f"Filter {filter_name!r} requires a MarkovSwitchingSSM."
            raise ValueError(msg)

        self._model = model
        self._filter_name = filter_name
        self._filter_kwargs = filter_kwargs
        self._key = key
        self._filter_result: FilterResult | SwitchingFilterResult | None = None
        self._observations: Array | None = None
        self._index: pd.DatetimeIndex | None = None
        self._is_fitted = False

    # --- Properties ---

    @property
    def model(self) -> StateSpaceModel | NonlinearSSM | MarkovSwitchingSSM:
        """The underlying state-space model."""
        return self._model

    @property
    def filter_name(self) -> str:
        """Name of the selected filter algorithm."""
        return self._filter_name

    @property
    def filter_result(self) -> FilterResult | SwitchingFilterResult:
        """Filter result from the last ``fit()`` call."""
        if self._filter_result is None:
            msg = "Model not fitted. Call .fit() first."
            raise RuntimeError(msg)
        return self._filter_result

    # --- Core methods ---

    def fit(
        self,
        y: Any,
        initial_state: GaussianState | None = None,
    ) -> SSM:
        """Fit the model by running the selected filter.

        Accepts ``jax.numpy``, ``numpy`` arrays, or ``pandas``
        DataFrames/Series. If a pandas object with a DatetimeIndex
        is provided, the index is preserved for output.

        Args:
            y: Observations, shape (T,) or (T, obs_dim).
            initial_state: Optional custom initial state.

        Returns:
            self (for method chaining).
        """
        obs, self._index = _to_jax_2d(y)
        self._observations = obs

        if self._filter_name == "kalman":
            self._filter_result = kalman_filter(self._model, obs, initial_state=initial_state)
        elif self._filter_name == "ekf":
            self._filter_result = ekf_filter(self._model, obs, initial_state=initial_state)
        elif self._filter_name == "ukf":
            self._filter_result = ukf_filter(
                self._model,
                obs,
                initial_state=initial_state,
                **self._filter_kwargs,
            )
        elif self._filter_name == "particle":
            key = self._key if self._key is not None else jax.random.PRNGKey(0)
            self._filter_result = particle_filter(
                self._model,
                obs,
                key=key,
                initial_state=initial_state,
                **self._filter_kwargs,
            )
        elif self._filter_name == "hamilton":
            self._filter_result = hamilton_filter(self._model, obs, initial_state=initial_state)

        self._is_fitted = True
        return self

    def residuals(self) -> Array:
        """Compute innovation residuals: observed - predicted observations.

        For linear models, returns standardized residuals (scaled by
        innovation covariance). For nonlinear models, returns raw
        (unstandardized) residuals.

        Returns:
            Residual array, shape (T, obs_dim).
        """
        fr = self.filter_result

        if isinstance(self._model, StateSpaceModel):
            from dynaris.estimation.diagnostics import standardized_residuals

            return standardized_residuals(fr, self._model)

        # Nonlinear: compute y - h(predicted_state)
        predicted_obs = jax.vmap(self._model.h)(fr.predicted_states)
        return fr.observations - predicted_obs

    def filtered_states_df(self) -> pd.DataFrame:
        """Filtered states as a pandas DataFrame."""
        fr = self.filter_result
        states = np.asarray(fr.filtered_states)
        cols = [f"state_{i}" for i in range(states.shape[1])]
        index = self._index if self._index is not None else None
        return pd.DataFrame(states, columns=cols, index=index)

    def summary(self) -> str:
        """Print a summary of the fitted model."""
        model_type = type(self._model).__name__
        lines = [
            "SSM Summary",
            "=" * 40,
            f"Model type:      {model_type}",
            f"State dim:       {self._model.state_dim}",
            f"Observation dim: {self._model.obs_dim}",
            f"Filter:          {self._filter_name}",
        ]
        if self._is_fitted:
            ll = float(self.filter_result.log_likelihood)
            n = int(self.filter_result.observations.shape[0])
            lines.append(f"Observations:    {n}")
            lines.append(f"Log-likelihood:  {ll:.4f}")
        else:
            lines.append("(not fitted)")
        lines.append("=" * 40)
        return "\n".join(lines)

    def plot(self, kind: str = "filtered", **kwargs: Any) -> Any:
        """Plot model results.

        Args:
            kind: Plot type. ``"filtered"`` shows filtered states
                vs observations.
            **kwargs: Passed to the underlying plot function.

        Returns:
            matplotlib Figure.
        """
        if kind == "filtered":
            return self._plot_filtered(**kwargs)
        msg = f"Unknown plot kind: {kind!r}. Currently supported: 'filtered'."
        raise ValueError(msg)

    # --- Private helpers ---

    def _plot_filtered(self, **kwargs: Any) -> Any:
        """Plot filtered states vs observations."""
        fr = self.filter_result

        if isinstance(self._model, StateSpaceModel):
            from dynaris.plotting.plots import plot_filtered

            return plot_filtered(fr, self._model, **kwargs)

        # Nonlinear: compute observation-space predictions
        import matplotlib.pyplot as plt

        filtered_obs = jax.vmap(self._model.h)(fr.filtered_states)
        obs = np.asarray(fr.observations)
        filt = np.asarray(filtered_obs)

        n_obs = obs.shape[1]
        fig, axes = plt.subplots(n_obs, 1, figsize=(10, 3 * n_obs), squeeze=False)

        for i in range(n_obs):
            ax = axes[i, 0]
            ax.plot(obs[:, i], color="black", alpha=0.5, linewidth=0.8, label="Observed")
            ax.plot(filt[:, i], color="#E45A31", linewidth=1.2, label="Filtered")
            ax.set_ylabel(f"obs[{i}]")
            ax.legend(frameon=False)

        axes[-1, 0].set_xlabel("Time")
        fig.suptitle(kwargs.get("title", "SSM Filtered States"))
        fig.tight_layout()
        return fig

    # --- Repr ---

    def __repr__(self) -> str:
        model_type = type(self._model).__name__
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"SSM(model={model_type}, filter={self._filter_name!r}, "
            f"state_dim={self._model.state_dim}, obs_dim={self._model.obs_dim}, "
            f"{status})"
        )
