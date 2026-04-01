"""Tests for Bayesian MCMC estimation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

numpyro = __import__("pytest").importorskip("numpyro")

from dynaris.datasets import load_nile_jax  # noqa: E402
from dynaris.estimation.bayesian import BayesianResult, fit_bayesian  # noqa: E402
from dynaris.estimation.priors import inverse_gamma_log_prior  # noqa: E402

NILE = load_nile_jax()[:30]  # Short for speed


def _model_fn(params: jax.Array) -> object:
    from dynaris.dlm.components import LocalLevel

    return LocalLevel(sigma_level=jnp.exp(params[0]), sigma_obs=jnp.exp(params[1]))


def test_fit_bayesian_returns_result() -> None:
    result = fit_bayesian(
        _model_fn,
        NILE.reshape(-1, 1),
        jnp.array([3.0, 5.0]),
        n_warmup=50,
        n_samples=100,
        key=jax.random.PRNGKey(0),
    )
    assert isinstance(result, BayesianResult)


def test_fit_bayesian_samples_shape() -> None:
    result = fit_bayesian(
        _model_fn,
        NILE.reshape(-1, 1),
        jnp.array([3.0, 5.0]),
        n_warmup=50,
        n_samples=100,
        key=jax.random.PRNGKey(1),
    )
    assert result.samples.shape == (100, 2)
    assert result.log_likelihood_samples.shape == (100,)


def test_fit_bayesian_finite_outputs() -> None:
    result = fit_bayesian(
        _model_fn,
        NILE.reshape(-1, 1),
        jnp.array([3.0, 5.0]),
        n_warmup=50,
        n_samples=100,
        key=jax.random.PRNGKey(2),
    )
    assert jnp.all(jnp.isfinite(result.samples))
    assert jnp.all(jnp.isfinite(result.log_likelihood_samples))
    assert jnp.isfinite(result.filter_result.log_likelihood)


def test_fit_bayesian_with_prior() -> None:
    prior = inverse_gamma_log_prior(shape=2.0, scale=1.0)
    result = fit_bayesian(
        _model_fn,
        NILE.reshape(-1, 1),
        jnp.array([3.0, 5.0]),
        log_prior_fn=prior,
        n_warmup=50,
        n_samples=100,
        key=jax.random.PRNGKey(3),
    )
    assert isinstance(result, BayesianResult)
    assert jnp.all(jnp.isfinite(result.samples))


def test_fit_bayesian_model_at_posterior_mean() -> None:
    result = fit_bayesian(
        _model_fn,
        NILE.reshape(-1, 1),
        jnp.array([3.0, 5.0]),
        n_warmup=50,
        n_samples=100,
        key=jax.random.PRNGKey(4),
    )
    assert result.model is not None
    assert float(result.filter_result.log_likelihood) < 0.0


def test_fit_bayesian_param_names() -> None:
    result = fit_bayesian(
        _model_fn,
        NILE.reshape(-1, 1),
        jnp.array([3.0, 5.0]),
        n_warmup=50,
        n_samples=50,
        key=jax.random.PRNGKey(5),
        param_names=("log_sigma_level", "log_sigma_obs"),
    )
    assert result.param_names == ("log_sigma_level", "log_sigma_obs")


def test_fit_bayesian_posterior_concentrates() -> None:
    """Posterior std should be smaller than a very wide prior."""
    result = fit_bayesian(
        _model_fn,
        NILE.reshape(-1, 1),
        jnp.array([3.0, 5.0]),
        n_warmup=100,
        n_samples=200,
        key=jax.random.PRNGKey(6),
    )
    posterior_std = float(jnp.std(result.samples[:, 0]))
    # Posterior std for log(sigma_level^2) should be << 100 (the prior scale)
    assert posterior_std < 80.0
