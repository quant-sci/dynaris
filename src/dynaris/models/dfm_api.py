"""High-level Dynamic Factor Model API.

Provides a user-friendly interface for fitting DFMs, extracting factors,
computing forecasts, and generating summaries.

Example::

    from dynaris.models import DFMModel

    dfm = DFMModel(n_factors=2)
    dfm.fit(panel_data)
    print(dfm.summary())
    factors = dfm.factor_states_df()
    forecast_df = dfm.forecast(steps=12)
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array

from dynaris.core.types import GaussianState
from dynaris.estimation.dfm import DFMResult, fit_dfm_em
from dynaris.forecast.forecast import forecast


class DFMModel:
    """High-level Dynamic Factor Model interface.

    Fits a DFM via EM estimation, extracting r latent factors from
    m observed variables. Provides methods for accessing factors,
    loadings, forecasts, and summaries.

    Args:
        n_factors: Number of latent factors.
        factor_order: AR order for factor dynamics (default 1 = random walk).
        max_iter: Maximum EM iterations.
        tol: Convergence tolerance.
        init_method: Initialization method (``"pca"`` or ``"random"``).

    Example::

        from dynaris.models import DFMModel

        dfm = DFMModel(n_factors=2)
        dfm.fit(panel_data)
        print(dfm.loadings_df())
        print(dfm.forecast(steps=6))
    """

    def __init__(
        self,
        n_factors: int,
        factor_order: int = 1,
        max_iter: int = 200,
        tol: float = 1e-6,
        init_method: str = "pca",
    ) -> None:
        self.n_factors = n_factors
        self.factor_order = factor_order
        self.max_iter = max_iter
        self.tol = tol
        self.init_method = init_method
        self._result: DFMResult | None = None
        self._observations: Array | None = None
        self._index: pd.DatetimeIndex | None = None
        self._columns: list[str] | None = None

    # --- Core methods ---

    def fit(self, y: Any) -> DFMModel:
        """Fit the DFM via EM estimation.

        Accepts ``jax.numpy``, ``numpy`` arrays, or ``pandas``
        DataFrames. If a DataFrame is provided, column names and
        DatetimeIndex are preserved.

        Args:
            y: Observations, shape (T, m).

        Returns:
            self (for method chaining).
        """
        if isinstance(y, pd.DataFrame):
            if isinstance(y.index, pd.DatetimeIndex):
                self._index = y.index
            self._columns = list(y.columns)
            y = y.values
        elif isinstance(y, pd.Series):
            y = y.values

        obs = jnp.asarray(np.asarray(y, dtype=np.float64))
        if obs.ndim == 1:
            obs = obs[:, None]
        self._observations = obs

        self._result = fit_dfm_em(
            obs,
            n_factors=self.n_factors,
            factor_order=self.factor_order,
            max_iter=self.max_iter,
            tol=self.tol,
            init_method=self.init_method,
        )
        return self

    @property
    def result(self) -> DFMResult:
        """The underlying DFMResult from EM estimation."""
        if self._result is None:
            msg = "Model not fitted. Call .fit() first."
            raise RuntimeError(msg)
        return self._result

    def forecast(self, steps: int = 1) -> pd.DataFrame:
        """Multi-step-ahead forecast from the last smoothed state.

        Returns observation-space forecasts with 95% confidence bands.

        Args:
            steps: Number of steps ahead.

        Returns:
            DataFrame with mean forecast per variable.
        """
        r = self.result
        last_state = GaussianState(
            mean=r.smoother_result.smoothed_states[-1],
            cov=r.smoother_result.smoothed_covariances[-1],
        )
        fc = forecast(r.model, last_state, steps)
        mean = np.asarray(fc.mean)

        m = mean.shape[1]
        cols = self._columns or [f"var_{i}" for i in range(m)]

        index = None
        if self._index is not None and len(self._index) > 0:
            freq = self._index.freq or pd.infer_freq(self._index)
            if freq is not None:
                last = self._index[-1]
                index = pd.date_range(
                    start=last + pd.tseries.frequencies.to_offset(freq),
                    periods=steps,
                    freq=freq,
                )

        return pd.DataFrame(mean, columns=cols, index=index)

    def loadings_df(self) -> pd.DataFrame:
        """Factor loadings as a pandas DataFrame.

        Returns:
            DataFrame with shape (m, r), rows = variables,
            columns = factor_0, factor_1, ...
        """
        loadings = np.asarray(self.result.loadings)
        rows = self._columns or [f"var_{i}" for i in range(loadings.shape[0])]
        cols = [f"factor_{j}" for j in range(loadings.shape[1])]
        return pd.DataFrame(loadings, index=rows, columns=cols)

    def factor_states_df(self) -> pd.DataFrame:
        """Smoothed factor estimates as a pandas DataFrame.

        Returns:
            DataFrame with shape (T, r).
        """
        states = np.asarray(self.result.factor_states)
        cols = [f"factor_{j}" for j in range(states.shape[1])]
        index = self._index if self._index is not None else None
        return pd.DataFrame(states, columns=cols, index=index)

    def explained_variance_df(self) -> pd.DataFrame:
        """Variance explained per factor.

        Returns:
            DataFrame with columns ``"factor"``, ``"variance"``,
            ``"proportion"``.
        """
        ev = np.asarray(self.result.explained_variance)
        total = float(np.sum(ev))
        return pd.DataFrame(
            {
                "factor": [f"factor_{j}" for j in range(len(ev))],
                "variance": ev,
                "proportion": ev / total if total > 0 else ev,
            }
        )

    def summary(self) -> str:
        """Print a summary of the fitted DFM."""
        lines = [
            "DFM Summary",
            "=" * 40,
            f"Factors:         {self.n_factors}",
            f"Variables:       {self.result.loadings.shape[0]}",
            f"Factor order:    {self.factor_order}",
        ]
        lines.append(f"Observations:    {self.result.factor_states.shape[0]}")
        lines.append(f"Log-likelihood:  {self.result.log_likelihood:.4f}")
        lines.append(f"EM iterations:   {self.result.n_iterations}")
        lines.append(f"Converged:       {self.result.converged}")
        ev = np.asarray(self.result.explained_variance)
        total = float(np.sum(ev))
        for j in range(len(ev)):
            pct = 100.0 * float(ev[j]) / total if total > 0 else 0.0
            lines.append(f"  Factor {j}: variance={float(ev[j]):.4f} ({pct:.1f}%)")
        lines.append("=" * 40)
        return "\n".join(lines)

    def __repr__(self) -> str:
        status = "fitted" if self._result is not None else "not fitted"
        return f"DFMModel(n_factors={self.n_factors}, {status})"
