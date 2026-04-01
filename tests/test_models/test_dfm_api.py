"""Tests for the DFMModel high-level API."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from dynaris.models.dfm_api import DFMModel


def _simulate_panel(n_obs: int = 100, n_vars: int = 6) -> jnp.ndarray:
    """Simulate a simple panel dataset."""
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    factors = jnp.cumsum(jax.random.normal(k1, (n_obs, 2)) * 0.5, axis=0)
    loadings = jax.random.normal(k2, (n_vars, 2))
    noise = jax.random.normal(k3, (n_obs, n_vars))
    return factors @ loadings.T + noise


def test_dfm_model_fit() -> None:
    data = _simulate_panel()
    dfm = DFMModel(n_factors=2, max_iter=10)
    result = dfm.fit(data)
    assert result is dfm  # method chaining


def test_dfm_model_repr() -> None:
    dfm = DFMModel(n_factors=2)
    assert "not fitted" in repr(dfm)
    dfm.fit(_simulate_panel())
    assert "fitted" in repr(dfm)


def test_dfm_model_summary() -> None:
    dfm = DFMModel(n_factors=2, max_iter=10).fit(_simulate_panel())
    s = dfm.summary()
    assert "Factors:" in s
    assert "Variables:" in s
    assert "Log-likelihood:" in s


def test_dfm_model_loadings_df() -> None:
    dfm = DFMModel(n_factors=2, max_iter=10).fit(_simulate_panel())
    df = dfm.loadings_df()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (6, 2)


def test_dfm_model_factor_states_df() -> None:
    dfm = DFMModel(n_factors=2, max_iter=10).fit(_simulate_panel())
    df = dfm.factor_states_df()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (100, 2)


def test_dfm_model_explained_variance_df() -> None:
    dfm = DFMModel(n_factors=2, max_iter=10).fit(_simulate_panel())
    df = dfm.explained_variance_df()
    assert "variance" in df.columns
    assert "proportion" in df.columns
    assert len(df) == 2


def test_dfm_model_forecast() -> None:
    dfm = DFMModel(n_factors=2, max_iter=10).fit(_simulate_panel())
    fc = dfm.forecast(steps=5)
    assert isinstance(fc, pd.DataFrame)
    assert fc.shape == (5, 6)


def test_dfm_model_with_pandas_input() -> None:
    data = np.asarray(_simulate_panel())
    cols = [f"series_{i}" for i in range(6)]
    df = pd.DataFrame(data, columns=cols)

    dfm = DFMModel(n_factors=2, max_iter=10).fit(df)
    loadings = dfm.loadings_df()
    assert list(loadings.index) == cols


def test_dfm_model_with_datetime_index() -> None:
    data = np.asarray(_simulate_panel())
    idx = pd.date_range("2020-01-01", periods=100, freq="MS")
    df = pd.DataFrame(data, index=idx, columns=[f"v{i}" for i in range(6)])

    dfm = DFMModel(n_factors=2, max_iter=10).fit(df)
    factors = dfm.factor_states_df()
    assert isinstance(factors.index, pd.DatetimeIndex)


def test_dfm_model_not_fitted_raises() -> None:
    dfm = DFMModel(n_factors=2)
    import pytest

    with pytest.raises(RuntimeError, match="not fitted"):
        _ = dfm.result
