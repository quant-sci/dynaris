"""Tests for the DLM high-level API."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import pandas as pd
import pytest

from dynaris import DLM, LocalLevel, LocalLinearTrend, Seasonal
from dynaris.datasets import load_nile_jax

matplotlib.use("Agg")

NILE = load_nile_jax()


# ===================================================================
# Construction and repr
# ===================================================================


def test_dlm_construction() -> None:
    model = LocalLevel()
    dlm = DLM(model)
    assert dlm.model.state_dim == 1
    assert "not fitted" in repr(dlm)


def test_dlm_repr_fitted() -> None:
    dlm = DLM(LocalLevel())
    dlm.fit(NILE)
    assert "fitted" in repr(dlm)
    assert "not fitted" not in repr(dlm)


# ===================================================================
# fit() — accepts various input types
# ===================================================================


def test_fit_jax_array() -> None:
    dlm = DLM(LocalLevel(sigma_level=40.0, sigma_obs=120.0))
    dlm.fit(NILE)
    assert dlm.filter_result.filtered_states.shape[0] == len(NILE)


def test_fit_numpy_array() -> None:
    dlm = DLM(LocalLevel(sigma_level=40.0, sigma_obs=120.0))
    dlm.fit(np.array(NILE))
    assert dlm.filter_result.filtered_states.shape[0] == len(NILE)


def test_fit_pandas_series() -> None:
    series = pd.Series(np.array(NILE))
    dlm = DLM(LocalLevel(sigma_level=40.0, sigma_obs=120.0))
    dlm.fit(series)
    assert dlm.filter_result.filtered_states.shape[0] == len(NILE)


def test_fit_pandas_dataframe() -> None:
    df = pd.DataFrame({"y": np.array(NILE)})
    dlm = DLM(LocalLevel(sigma_level=40.0, sigma_obs=120.0))
    dlm.fit(df)
    assert dlm.filter_result.filtered_states.shape[0] == len(NILE)


def test_fit_pandas_with_datetime_index() -> None:
    index = pd.date_range("1871", periods=len(NILE), freq="YE")
    series = pd.Series(np.array(NILE), index=index)
    dlm = DLM(LocalLevel(sigma_level=40.0, sigma_obs=120.0))
    dlm.fit(series)
    assert dlm._index is not None


def test_fit_returns_self() -> None:
    dlm = DLM(LocalLevel())
    result = dlm.fit(NILE)
    assert result is dlm


# ===================================================================
# smooth()
# ===================================================================


def test_smooth() -> None:
    dlm = DLM(LocalLevel(sigma_level=40.0, sigma_obs=120.0))
    dlm.fit(NILE).smooth()
    sr = dlm.smoother_result
    assert sr.smoothed_states.shape[0] == len(NILE)


def test_smooth_before_fit_raises() -> None:
    dlm = DLM(LocalLevel())
    with pytest.raises(RuntimeError, match="not fitted"):
        dlm.smooth()


# ===================================================================
# forecast()
# ===================================================================


def test_forecast_returns_dataframe() -> None:
    dlm = DLM(LocalLevel(sigma_level=40.0, sigma_obs=120.0))
    dlm.fit(NILE)
    fc = dlm.forecast(steps=10)
    assert isinstance(fc, pd.DataFrame)
    assert len(fc) == 10
    assert "mean" in fc.columns
    assert "lower_95" in fc.columns
    assert "upper_95" in fc.columns


def test_forecast_with_datetime_index() -> None:
    index = pd.date_range("1871", periods=len(NILE), freq="YE")
    series = pd.Series(np.array(NILE), index=index)
    dlm = DLM(LocalLevel(sigma_level=40.0, sigma_obs=120.0))
    dlm.fit(series)
    fc = dlm.forecast(steps=5)
    assert isinstance(fc.index, pd.DatetimeIndex)
    assert len(fc) == 5


def test_forecast_before_fit_raises() -> None:
    dlm = DLM(LocalLevel())
    with pytest.raises(RuntimeError, match="not fitted"):
        dlm.forecast(steps=5)


# ===================================================================
# summary()
# ===================================================================


def test_summary_fitted() -> None:
    dlm = DLM(LocalLevel(sigma_level=40.0, sigma_obs=120.0))
    dlm.fit(NILE)
    s = dlm.summary()
    assert "Log-likelihood" in s
    assert "Residual mean" in s
    assert "Ljung-Box" in s


def test_summary_not_fitted() -> None:
    dlm = DLM(LocalLevel())
    s = dlm.summary()
    assert "not fitted" in s


# ===================================================================
# residuals and diagnostics
# ===================================================================


def test_residuals() -> None:
    dlm = DLM(LocalLevel(sigma_level=40.0, sigma_obs=120.0))
    dlm.fit(NILE)
    resids = dlm.residuals()
    assert resids.shape == (len(NILE),)
    assert jnp.all(jnp.isfinite(resids))


def test_ljung_box() -> None:
    dlm = DLM(LocalLevel(sigma_level=40.0, sigma_obs=120.0))
    dlm.fit(NILE)
    q, p = dlm.ljung_box()
    assert q >= 0
    assert 0 <= p <= 1


# ===================================================================
# DataFrames output
# ===================================================================


def test_filtered_states_df() -> None:
    dlm = DLM(LocalLevel(sigma_level=40.0, sigma_obs=120.0))
    dlm.fit(NILE)
    df = dlm.filtered_states_df()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (len(NILE), 1)


def test_smoothed_states_df() -> None:
    dlm = DLM(LocalLevel(sigma_level=40.0, sigma_obs=120.0))
    dlm.fit(NILE).smooth()
    df = dlm.smoothed_states_df()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (len(NILE), 1)


def test_filtered_states_df_with_datetime() -> None:
    index = pd.date_range("1871", periods=len(NILE), freq="YE")
    series = pd.Series(np.array(NILE), index=index)
    dlm = DLM(LocalLevel(sigma_level=40.0, sigma_obs=120.0))
    dlm.fit(series)
    df = dlm.filtered_states_df()
    assert isinstance(df.index, pd.DatetimeIndex)


# ===================================================================
# Plotting
# ===================================================================


def test_plot_filtered() -> None:
    import matplotlib.pyplot as plt

    dlm = DLM(LocalLevel(sigma_level=40.0, sigma_obs=120.0))
    dlm.fit(NILE)
    fig = dlm.plot(kind="filtered")
    assert fig is not None
    plt.close(fig)


def test_plot_diagnostics() -> None:
    import matplotlib.pyplot as plt

    dlm = DLM(LocalLevel(sigma_level=40.0, sigma_obs=120.0))
    dlm.fit(NILE)
    fig = dlm.plot(kind="diagnostics")
    assert fig is not None
    plt.close(fig)


def test_plot_forecast() -> None:
    import matplotlib.pyplot as plt

    dlm = DLM(LocalLevel(sigma_level=40.0, sigma_obs=120.0))
    dlm.fit(NILE)
    dlm.forecast(steps=10)
    fig = dlm.plot(kind="forecast")
    assert fig is not None
    plt.close(fig)


def test_plot_invalid_kind() -> None:
    dlm = DLM(LocalLevel())
    dlm.fit(NILE)
    with pytest.raises(ValueError, match="Unknown plot kind"):
        dlm.plot(kind="invalid")


# ===================================================================
# Composed model
# ===================================================================


def test_composed_model_full_workflow() -> None:
    import matplotlib.pyplot as plt

    model = LocalLinearTrend(sigma_level=1.0, sigma_slope=0.1, sigma_obs=0.0) + Seasonal(
        period=12, sigma_seasonal=0.5, sigma_obs=2.0
    )
    t = jnp.arange(60, dtype=jnp.float32)
    y = (
        t * 0.3
        + 5.0 * jnp.sin(2.0 * jnp.pi * t / 12.0)
        + jax.random.normal(jax.random.PRNGKey(0), (60,)) * 2.0
    )

    dlm = DLM(model)
    dlm.fit(y).smooth()

    fc = dlm.forecast(steps=12)
    assert isinstance(fc, pd.DataFrame)
    assert len(fc) == 12

    s = dlm.summary()
    assert "State dim:       13" in s

    fig = dlm.plot(kind="forecast")
    plt.close(fig)


# ===================================================================
# fit_batch
# ===================================================================


def test_fit_batch() -> None:
    model = LocalLevel(sigma_level=40.0, sigma_obs=120.0)
    dlm = DLM(model)
    key = jax.random.PRNGKey(0)
    batch = jax.random.normal(key, (3, 30, 1)) * 100 + 800
    result = dlm.fit_batch(batch)
    assert result.filtered_states.shape == (3, 30, 1)
    assert result.log_likelihood.shape == (3,)
