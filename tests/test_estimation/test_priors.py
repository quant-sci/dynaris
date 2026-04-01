"""Tests for built-in log-prior factories."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from dynaris.estimation.priors import (
    combine_priors,
    half_normal_log_prior,
    inverse_gamma_log_prior,
    normal_log_prior,
)


def test_normal_log_prior_value() -> None:
    prior = normal_log_prior(loc=0.0, scale=1.0)
    params = jnp.array([0.0])
    lp = prior(params)
    # log N(0; 0, 1) = -0.5 * log(2*pi)
    expected = -0.5 * jnp.log(2.0 * jnp.pi)
    np.testing.assert_allclose(lp, expected, atol=1e-5)


def test_normal_log_prior_nonzero() -> None:
    prior = normal_log_prior(loc=0.0, scale=1.0)
    params = jnp.array([1.0])
    lp = prior(params)
    expected = -0.5 * 1.0 - 0.5 * jnp.log(2.0 * jnp.pi)
    np.testing.assert_allclose(lp, expected, atol=1e-5)


def test_normal_log_prior_jit() -> None:
    prior = normal_log_prior()
    params = jnp.array([1.0, 2.0])
    lp = jax.jit(prior)(params)
    assert jnp.isfinite(lp)


def test_normal_log_prior_grad() -> None:
    prior = normal_log_prior()
    params = jnp.array([1.0, 2.0])
    g = jax.grad(prior)(params)
    assert jnp.all(jnp.isfinite(g))


def test_inverse_gamma_log_prior_jit() -> None:
    prior = inverse_gamma_log_prior(shape=2.0, scale=1.0)
    params = jnp.array([0.0, 1.0])
    lp = jax.jit(prior)(params)
    assert jnp.isfinite(lp)


def test_inverse_gamma_log_prior_grad() -> None:
    prior = inverse_gamma_log_prior(shape=2.0, scale=1.0)
    params = jnp.array([0.0, 1.0])
    g = jax.grad(prior)(params)
    assert jnp.all(jnp.isfinite(g))


def test_inverse_gamma_log_prior_with_indices() -> None:
    prior = inverse_gamma_log_prior(shape=2.0, scale=1.0, indices=[0])
    params = jnp.array([0.0, 100.0])
    lp = prior(params)
    assert jnp.isfinite(lp)


def test_half_normal_log_prior_jit() -> None:
    prior = half_normal_log_prior(scale=1.0)
    params = jnp.array([0.0])
    lp = jax.jit(prior)(params)
    assert jnp.isfinite(lp)


def test_half_normal_log_prior_grad() -> None:
    prior = half_normal_log_prior(scale=1.0)
    params = jnp.array([0.5])
    g = jax.grad(prior)(params)
    assert jnp.isfinite(g)


def test_half_normal_log_prior_with_indices() -> None:
    prior = half_normal_log_prior(scale=2.0, indices=[1])
    params = jnp.array([0.0, 0.5])
    lp = prior(params)
    assert jnp.isfinite(lp)


def test_combine_priors() -> None:
    p1 = normal_log_prior(loc=0.0, scale=1.0)
    p2 = inverse_gamma_log_prior(shape=2.0, scale=1.0, indices=[0])
    combined = combine_priors(p1, p2)

    params = jnp.array([0.0, 1.0])
    lp = combined(params)
    expected = p1(params) + p2(params)
    np.testing.assert_allclose(lp, expected, atol=1e-5)


def test_combine_priors_jit_grad() -> None:
    combined = combine_priors(
        normal_log_prior(),
        inverse_gamma_log_prior(shape=2.0, scale=1.0),
    )
    params = jnp.array([0.0, 1.0])
    lp = jax.jit(combined)(params)
    g = jax.grad(combined)(params)
    assert jnp.isfinite(lp)
    assert jnp.all(jnp.isfinite(g))
