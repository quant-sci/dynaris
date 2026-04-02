"""Prior and posterior predictive checks and forecasting.

Provides functions for simulating data from the prior and posterior
predictive distributions, and for generating forecast ensembles
from posterior parameter samples.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from dynaris.core.state_space import StateSpaceModel
from dynaris.core.types import GaussianState
from dynaris.estimation.bayesian import BayesianResult
from dynaris.filters.kalman import kalman_filter
from dynaris.forecast.forecast import forecast

ModelFactory = Any  # Callable[[Array], StateSpaceModel]


def prior_predictive(
    model_fn: ModelFactory,
    prior_sample_fn: Callable[[Array], Array],
    n_steps: int,
    n_samples: int = 100,
    key: Array | None = None,
) -> Array:
    """Simulate observations from the prior predictive distribution.

    For each sample, draws parameters from the prior, builds a model,
    and simulates a forward trajectory.

    Args:
        model_fn: Maps unconstrained parameter vector to StateSpaceModel.
        prior_sample_fn: Draws one parameter sample from the prior.
            Signature: ``(key) -> params_array``.
        n_steps: Number of time steps to simulate.
        n_samples: Number of prior draws.
        key: JAX PRNG key. Defaults to ``PRNGKey(0)``.

    Returns:
        Simulated observations, shape (n_samples, n_steps, obs_dim).
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    def _simulate_one(k: Array) -> Array:
        k_param, k_sim = jax.random.split(k)
        params = prior_sample_fn(k_param)
        model = model_fn(params)
        return _simulate_from_model(model, n_steps, k_sim)

    keys = jax.random.split(key, n_samples)
    return jax.vmap(_simulate_one)(keys)


def posterior_predictive_forecast(
    result: BayesianResult,
    model_fn: ModelFactory,
    observations: Array,
    steps: int,
    n_posterior_samples: int | None = None,
    key: Array | None = None,
) -> dict[str, Array]:
    """Forecast from the posterior predictive distribution.

    For each posterior sample, runs the Kalman filter and forecasts
    ``steps`` ahead. Aggregates into mean and credible intervals.

    Args:
        result: BayesianResult from :func:`fit_bayesian`.
        model_fn: Same model factory used for fitting.
        observations: Same observations used for fitting.
        steps: Number of forecast steps ahead.
        n_posterior_samples: Subsample size. If ``None``, use all.
        key: JAX PRNG key for subsampling.

    Returns:
        dict with keys ``"mean"``, ``"lower"``, ``"upper"``, ``"samples"``.
        Shapes: ``(steps, obs_dim)`` for mean/lower/upper,
        ``(n_used, steps, obs_dim)`` for samples.
    """
    observations = jnp.asarray(observations)
    samples = result.samples

    if n_posterior_samples is not None and n_posterior_samples < samples.shape[0]:
        if key is None:
            key = jax.random.PRNGKey(0)
        idx = jax.random.choice(key, samples.shape[0], shape=(n_posterior_samples,), replace=False)
        samples = samples[idx]

    def _forecast_one(params: Array) -> Array:
        model = model_fn(params)
        fr = kalman_filter(model, observations)
        last_state = GaussianState(
            mean=fr.filtered_states[-1],
            cov=fr.filtered_covariances[-1],
        )
        fc = forecast(model, last_state, steps)
        return fc.mean  # type: ignore[no-any-return]  # (steps, obs_dim)

    all_forecasts = jax.vmap(_forecast_one)(samples)  # (n, steps, obs_dim)

    return {
        "mean": jnp.mean(all_forecasts, axis=0),
        "samples": all_forecasts,
        "lower": jnp.percentile(all_forecasts, 2.5, axis=0),
        "upper": jnp.percentile(all_forecasts, 97.5, axis=0),
    }


def posterior_predictive_check(
    result: BayesianResult,
    model_fn: ModelFactory,
    observations: Array,
    n_posterior_samples: int | None = None,
    key: Array | None = None,
) -> Array:
    """Simulate in-sample data from the posterior predictive.

    For each posterior sample, simulates data of the same length
    as the observations. Used to assess model fit.

    Args:
        result: BayesianResult from :func:`fit_bayesian`.
        model_fn: Same model factory used for fitting.
        observations: Original observations (used for length).
        n_posterior_samples: Subsample size.
        key: JAX PRNG key.

    Returns:
        Simulated observations, shape (n_used, T, obs_dim).
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    observations = jnp.asarray(observations)
    n_steps = observations.shape[0]
    samples = result.samples

    if n_posterior_samples is not None and n_posterior_samples < samples.shape[0]:
        k_sub, key = jax.random.split(key)
        idx = jax.random.choice(
            k_sub, samples.shape[0], shape=(n_posterior_samples,), replace=False
        )
        samples = samples[idx]

    def _simulate_one(params: Array, k: Array) -> Array:
        model = model_fn(params)
        return _simulate_from_model(model, n_steps, k)

    keys = jax.random.split(key, samples.shape[0])
    return jax.vmap(_simulate_one)(samples, keys)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _simulate_from_model(model: StateSpaceModel, n_steps: int, key: Array) -> Array:
    """Simulate observations from a linear state-space model.

    Uses the model's default initial state (diffuse prior) to draw the
    starting state, ensuring simulations start at a reasonable scale.

    Args:
        model: Linear-Gaussian state-space model.
        n_steps: Number of time steps.
        key: JAX PRNG key.

    Returns:
        Observations, shape (n_steps, obs_dim).
    """
    n = model.state_dim
    m = model.obs_dim
    k_init, k_scan = jax.random.split(key)

    init_state = jax.random.multivariate_normal(k_init, jnp.zeros(n), model.W * 10.0)

    def _step(state: Array, k: Array) -> tuple[Array, Array]:
        k_state, k_obs = jax.random.split(k)
        new_state = model.G @ state + jax.random.multivariate_normal(k_state, jnp.zeros(n), model.W)
        obs = model.F @ new_state + jax.random.multivariate_normal(k_obs, jnp.zeros(m), model.V)
        return new_state, obs

    keys = jax.random.split(k_scan, n_steps)
    _, observations = jax.lax.scan(_step, init_state, keys)
    return observations
