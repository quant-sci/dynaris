"""Tests for the unified SSM class with automatic filter selection."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from dynaris.core.nonlinear import NonlinearSSM
from dynaris.core.results import FilterResult
from dynaris.core.ssm import SSM
from dynaris.core.types import GaussianState
from dynaris.datasets import load_nile_jax
from dynaris.dlm.api import DLM
from dynaris.dlm.components import LocalLevel
from dynaris.models.nonlinear import (
    BearingsTracking,
    LorenzAttractor,
    StochasticVolatility,
    transform_returns,
)

NILE = load_nile_jax()


# ---------------------------------------------------------------------------
# Helper: simple nonlinear model
# ---------------------------------------------------------------------------


def _simple_nonlinear() -> NonlinearSSM:
    return NonlinearSSM(
        transition_fn=lambda x: x,
        observation_fn=lambda x: x,
        transition_cov=jnp.array([[1.0]]),
        observation_cov=jnp.array([[1.0]]),
        state_dim=1,
        obs_dim=1,
    )


# ---------------------------------------------------------------------------
# Construction and auto-selection
# ---------------------------------------------------------------------------


def test_ssm_linear_auto_selects_kalman() -> None:
    ssm = SSM(LocalLevel())
    assert ssm.filter_name == "kalman"


def test_ssm_nonlinear_auto_selects_ukf() -> None:
    ssm = SSM(_simple_nonlinear())
    assert ssm.filter_name == "ukf"


def test_ssm_filter_override_ekf() -> None:
    ssm = SSM(_simple_nonlinear(), filter="ekf")
    assert ssm.filter_name == "ekf"


def test_ssm_filter_override_particle() -> None:
    ssm = SSM(_simple_nonlinear(), filter="particle")
    assert ssm.filter_name == "particle"


def test_ssm_invalid_filter_raises() -> None:
    with pytest.raises(ValueError, match="Unknown filter"):
        SSM(LocalLevel(), filter="invalid")


def test_ssm_kalman_with_nonlinear_raises() -> None:
    with pytest.raises(ValueError, match="requires a StateSpaceModel"):
        SSM(_simple_nonlinear(), filter="kalman")


def test_ssm_ekf_with_linear_raises() -> None:
    with pytest.raises(ValueError, match="requires a NonlinearSSM"):
        SSM(LocalLevel(), filter="ekf")


def test_ssm_invalid_model_type_raises() -> None:
    with pytest.raises(TypeError, match="must be a StateSpaceModel or NonlinearSSM"):
        SSM("not a model")  # type: ignore[arg-type]


def test_ssm_repr_not_fitted() -> None:
    ssm = SSM(LocalLevel())
    r = repr(ssm)
    assert "kalman" in r
    assert "not fitted" in r


def test_ssm_repr_fitted() -> None:
    ssm = SSM(LocalLevel()).fit(NILE)
    r = repr(ssm)
    assert "fitted" in r
    assert "not fitted" not in r


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------


def test_fit_linear_returns_self() -> None:
    ssm = SSM(LocalLevel())
    result = ssm.fit(NILE)
    assert result is ssm


def test_fit_linear_produces_filter_result() -> None:
    ssm = SSM(LocalLevel()).fit(NILE)
    fr = ssm.filter_result
    assert isinstance(fr, FilterResult)
    assert fr.filtered_states.shape == (100, 1)


def test_fit_linear_matches_dlm() -> None:
    """SSM with linear model should produce identical results to DLM."""
    model = LocalLevel(sigma_level=38.0, sigma_obs=123.0)

    ssm = SSM(model).fit(NILE)
    dlm = DLM(model).fit(NILE)

    np.testing.assert_allclose(
        ssm.filter_result.filtered_states,
        dlm.filter_result.filtered_states,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        ssm.filter_result.log_likelihood,
        dlm.filter_result.log_likelihood,
        atol=1e-5,
    )


def test_fit_nonlinear_ekf() -> None:
    ssm = SSM(_simple_nonlinear(), filter="ekf").fit(NILE)
    assert jnp.all(jnp.isfinite(ssm.filter_result.filtered_states))
    assert jnp.isfinite(ssm.filter_result.log_likelihood)


def test_fit_nonlinear_ukf() -> None:
    init = GaussianState(mean=jnp.array([1000.0]), cov=jnp.eye(1) * 1e4)
    ssm = SSM(_simple_nonlinear(), filter="ukf", alpha=1.0).fit(NILE, initial_state=init)
    assert jnp.all(jnp.isfinite(ssm.filter_result.filtered_states))


def test_fit_nonlinear_particle() -> None:
    ssm = SSM(
        _simple_nonlinear(),
        filter="particle",
        key=jax.random.PRNGKey(0),
        n_particles=200,
    ).fit(NILE)
    assert jnp.all(jnp.isfinite(ssm.filter_result.filtered_states))


def test_fit_accepts_numpy() -> None:
    y = np.asarray(NILE)
    ssm = SSM(LocalLevel()).fit(y)
    assert ssm.filter_result.filtered_states.shape == (100, 1)


def test_fit_accepts_pandas_series() -> None:
    y = pd.Series(np.asarray(NILE))
    ssm = SSM(LocalLevel()).fit(y)
    assert ssm.filter_result.filtered_states.shape == (100, 1)


def test_fit_accepts_pandas_with_datetime() -> None:
    idx = pd.date_range("1871", periods=100, freq="YS")
    y = pd.Series(np.asarray(NILE), index=idx)
    ssm = SSM(LocalLevel()).fit(y)
    df = ssm.filtered_states_df()
    assert isinstance(df.index, pd.DatetimeIndex)


def test_filter_result_before_fit_raises() -> None:
    ssm = SSM(LocalLevel())
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = ssm.filter_result


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------


def test_summary_linear() -> None:
    ssm = SSM(LocalLevel()).fit(NILE)
    s = ssm.summary()
    assert "kalman" in s
    assert "Log-likelihood" in s
    assert "StateSpaceModel" in s


def test_summary_nonlinear() -> None:
    init = GaussianState(mean=jnp.array([1000.0]), cov=jnp.eye(1) * 1e4)
    ssm = SSM(_simple_nonlinear(), alpha=1.0).fit(NILE, initial_state=init)
    s = ssm.summary()
    assert "ukf" in s
    assert "NonlinearSSM" in s


def test_summary_not_fitted() -> None:
    ssm = SSM(LocalLevel())
    s = ssm.summary()
    assert "not fitted" in s


# ---------------------------------------------------------------------------
# residuals()
# ---------------------------------------------------------------------------


def test_residuals_linear() -> None:
    ssm = SSM(LocalLevel(sigma_level=38.0, sigma_obs=123.0)).fit(NILE)
    resids = ssm.residuals()
    assert resids.shape[0] == 100
    assert jnp.all(jnp.isfinite(resids))


def test_residuals_nonlinear() -> None:
    init = GaussianState(mean=jnp.array([1000.0]), cov=jnp.eye(1) * 1e4)
    ssm = SSM(_simple_nonlinear(), alpha=1.0).fit(NILE, initial_state=init)
    resids = ssm.residuals()
    assert resids.shape == (100, 1)
    assert jnp.all(jnp.isfinite(resids))


# ---------------------------------------------------------------------------
# filtered_states_df()
# ---------------------------------------------------------------------------


def test_filtered_states_df() -> None:
    ssm = SSM(LocalLevel()).fit(NILE)
    df = ssm.filtered_states_df()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (100, 1)
    assert list(df.columns) == ["state_0"]


# ---------------------------------------------------------------------------
# Integration with built-in nonlinear models
# ---------------------------------------------------------------------------


def test_ssm_stochastic_volatility() -> None:
    model = StochasticVolatility()
    key = jax.random.PRNGKey(0)
    returns = jax.random.normal(key, (50,))
    y_star = transform_returns(returns)

    init = GaussianState(mean=jnp.array([0.0]), cov=jnp.array([[10.0]]))
    ssm = SSM(model, alpha=1.0).fit(y_star, initial_state=init)
    assert jnp.all(jnp.isfinite(ssm.filter_result.filtered_states))
    s = ssm.summary()
    assert "ukf" in s


def test_ssm_bearings_tracking() -> None:
    model = BearingsTracking(sigma_accel=0.5, sigma_bearing=0.1)
    key = jax.random.PRNGKey(10)

    # Simulate from a known trajectory
    true_state = jnp.array([10.0, 1.0, 10.0, 0.5])
    obs_list = []
    for _ in range(30):
        true_state = model.f(true_state)
        bearing = model.h(true_state)
        obs_list.append(bearing + jax.random.normal(key, (1,)) * 0.1)
        key, _ = jax.random.split(key)
    observations = jnp.stack(obs_list)

    init = GaussianState(mean=jnp.array([10.0, 1.0, 10.0, 0.5]), cov=jnp.eye(4) * 10.0)
    ssm = SSM(model, filter="ekf").fit(observations, initial_state=init)
    assert jnp.all(jnp.isfinite(ssm.filter_result.filtered_states))
    assert ssm.filter_result.filtered_states.shape == (30, 4)


def test_ssm_lorenz() -> None:
    model = LorenzAttractor(dt=0.01, process_noise=0.5, obs_noise=2.0)
    key = jax.random.PRNGKey(20)

    state = jnp.array([1.0, 1.0, 1.0])
    obs_list = []
    for _ in range(50):
        state = model.f(state) + jax.random.normal(key, (3,)) * 0.5
        obs_list.append(model.h(state) + jax.random.normal(key, (3,)) * 2.0)
        key, _ = jax.random.split(key)
    observations = jnp.stack(obs_list)

    init = GaussianState(mean=jnp.array([1.0, 1.0, 1.0]), cov=jnp.eye(3) * 10.0)
    ssm = SSM(model, alpha=1.0).fit(observations, initial_state=init)
    assert jnp.all(jnp.isfinite(ssm.filter_result.filtered_states))
    assert ssm.filter_result.filtered_states.shape == (50, 3)
