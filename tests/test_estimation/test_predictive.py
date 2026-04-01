"""Tests for prior and posterior predictive checks."""

from __future__ import annotations

import jax
import jax.numpy as jnp

numpyro = __import__("pytest").importorskip("numpyro")

from dynaris.datasets import load_nile_jax  # noqa: E402
from dynaris.estimation.bayesian import fit_bayesian  # noqa: E402
from dynaris.estimation.predictive import (  # noqa: E402
    posterior_predictive_check,
    posterior_predictive_forecast,
    prior_predictive,
)

NILE = load_nile_jax()[:30]


def _model_fn(params: jax.Array) -> object:
    from dynaris.dlm.components import LocalLevel

    return LocalLevel(sigma_level=jnp.exp(params[0]), sigma_obs=jnp.exp(params[1]))


def _prior_sample_fn(key: jax.Array) -> jax.Array:
    return jax.random.normal(key, (2,)) * 2.0 + jnp.array([3.0, 5.0])


def test_prior_predictive_shape() -> None:
    result = prior_predictive(
        _model_fn,
        _prior_sample_fn,
        n_steps=20,
        n_samples=10,
        key=jax.random.PRNGKey(0),
    )
    assert result.shape == (10, 20, 1)
    assert jnp.all(jnp.isfinite(result))


def test_prior_predictive_different_keys() -> None:
    r1 = prior_predictive(
        _model_fn, _prior_sample_fn, n_steps=10, n_samples=5, key=jax.random.PRNGKey(0)
    )
    r2 = prior_predictive(
        _model_fn, _prior_sample_fn, n_steps=10, n_samples=5, key=jax.random.PRNGKey(1)
    )
    assert not jnp.allclose(r1, r2)


def _get_bayesian_result():
    return fit_bayesian(
        _model_fn,
        NILE.reshape(-1, 1),
        jnp.array([3.0, 5.0]),
        n_warmup=50,
        n_samples=50,
        key=jax.random.PRNGKey(10),
    )


def test_posterior_predictive_forecast_shape() -> None:
    result = _get_bayesian_result()
    fc = posterior_predictive_forecast(
        result,
        _model_fn,
        NILE.reshape(-1, 1),
        steps=10,
        n_posterior_samples=20,
    )
    assert "mean" in fc
    assert "lower" in fc
    assert "upper" in fc
    assert "samples" in fc
    assert fc["mean"].shape == (10, 1)
    assert fc["samples"].shape[0] == 20
    assert fc["samples"].shape[1] == 10


def test_posterior_predictive_forecast_intervals() -> None:
    result = _get_bayesian_result()
    fc = posterior_predictive_forecast(
        result,
        _model_fn,
        NILE.reshape(-1, 1),
        steps=5,
        n_posterior_samples=20,
    )
    assert jnp.all(fc["lower"] <= fc["mean"])
    assert jnp.all(fc["mean"] <= fc["upper"])


def test_posterior_predictive_check_shape() -> None:
    result = _get_bayesian_result()
    sim = posterior_predictive_check(
        result,
        _model_fn,
        NILE.reshape(-1, 1),
        n_posterior_samples=10,
    )
    assert sim.shape == (10, 30, 1)
