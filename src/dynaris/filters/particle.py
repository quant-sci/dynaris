"""Bootstrap particle filter / Sequential Monte Carlo for nonlinear state-space models.

Represents the filtering posterior as a weighted set of particles. Supports
multinomial, systematic, and stratified resampling with JIT-compatible
implementations via ``jax.lax.scan``.

References:
    Doucet, A., de Freitas, N. and Gordon, N. (2001). "Sequential Monte Carlo
    Methods in Practice." Springer.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from dynaris.core.nonlinear import NonlinearSSM
from dynaris.core.results import FilterResult
from dynaris.core.types import GaussianState

# Type alias for resampling functions.
ResampleFn = Callable[[Array, Array, int], Array]

# ---------------------------------------------------------------------------
# Resampling strategies
# ---------------------------------------------------------------------------


def _multinomial_resample(key: Array, weights: Array, n_particles: int) -> Array:
    """Multinomial resampling: draw indices from the categorical distribution.

    Args:
        key: JAX PRNG key.
        weights: Normalized weights, shape (N,).
        n_particles: Number of indices to draw.

    Returns:
        Resampled ancestor indices, shape (N,).
    """
    return jax.random.categorical(key, jnp.log(weights), shape=(n_particles,))


def _systematic_resample(key: Array, weights: Array, n_particles: int) -> Array:
    """Systematic resampling: single random offset with evenly spaced points.

    Args:
        key: JAX PRNG key.
        weights: Normalized weights, shape (N,).
        n_particles: Number of indices to draw.

    Returns:
        Resampled ancestor indices, shape (N,).
    """
    cdf = jnp.cumsum(weights)
    u0 = jax.random.uniform(key, shape=()) / n_particles
    positions = u0 + jnp.arange(n_particles) / n_particles
    return jnp.searchsorted(cdf, positions)


def _stratified_resample(key: Array, weights: Array, n_particles: int) -> Array:
    """Stratified resampling: independent uniform draw within each stratum.

    Args:
        key: JAX PRNG key.
        weights: Normalized weights, shape (N,).
        n_particles: Number of indices to draw.

    Returns:
        Resampled ancestor indices, shape (N,).
    """
    cdf = jnp.cumsum(weights)
    u = jax.random.uniform(key, shape=(n_particles,)) / n_particles
    positions = u + jnp.arange(n_particles) / n_particles
    return jnp.searchsorted(cdf, positions)


_RESAMPLE_FNS: dict[str, ResampleFn] = {
    "multinomial": _multinomial_resample,
    "systematic": _systematic_resample,
    "stratified": _stratified_resample,
}

# ---------------------------------------------------------------------------
# Weighted statistics
# ---------------------------------------------------------------------------


def _weighted_mean(particles: Array, weights: Array) -> Array:
    """Compute the weighted mean of a particle cloud.

    Args:
        particles: Particle positions, shape (N, n).
        weights: Normalized weights, shape (N,).

    Returns:
        Weighted mean, shape (n,).
    """
    return jnp.sum(weights[:, None] * particles, axis=0)


def _weighted_covariance(particles: Array, weights: Array, mean: Array) -> Array:
    """Compute the weighted covariance of a particle cloud.

    Args:
        particles: Particle positions, shape (N, n).
        weights: Normalized weights, shape (N,).
        mean: Weighted mean, shape (n,).

    Returns:
        Weighted covariance matrix, shape (n, n).
    """
    diff = particles - mean[None, :]  # (N, n)
    return jnp.sum(
        weights[:, None, None] * (diff[:, :, None] * diff[:, None, :]),
        axis=0,
    )


def effective_sample_size(weights: Array) -> Array:
    """Compute the effective sample size (ESS) from normalized weights.

    ESS = 1 / sum(w_i^2). Ranges from 1 (degenerate) to N (uniform).

    Args:
        weights: Normalized weights, shape (N,).

    Returns:
        Effective sample size, scalar.
    """
    return 1.0 / jnp.sum(weights**2)


# ---------------------------------------------------------------------------
# Core particle operations
# ---------------------------------------------------------------------------


def predict_particles(particles: Array, model: NonlinearSSM, key: Array) -> Array:
    """Propagate particles through the transition function with noise.

    x_t^i = f(x_{t-1}^i) + omega_t^i,  omega_t^i ~ N(0, Q)

    Args:
        particles: Current particles, shape (N, n).
        model: Nonlinear state-space model.
        key: JAX PRNG key for noise sampling.

    Returns:
        Predicted particles, shape (N, n).
    """
    n_particles = particles.shape[0]
    propagated = jax.vmap(model.f)(particles)  # (N, n)
    noise = jax.random.multivariate_normal(
        key, jnp.zeros(model.state_dim), model.Q, shape=(n_particles,)
    )
    return propagated + noise


def compute_log_weights(particles: Array, observation: Array, model: NonlinearSSM) -> Array:
    """Compute unnormalized log-weights from the observation likelihood.

    log w_t^i = log N(y_t | h(x_t^i), R)

    Args:
        particles: Predicted particles, shape (N, n).
        observation: Observation vector, shape (m,).
        model: Nonlinear state-space model.

    Returns:
        Unnormalized log-weights, shape (N,).
    """
    predicted_obs = jax.vmap(model.h)(particles)  # (N, m)
    diff = observation[None, :] - predicted_obs  # (N, m)

    # log N(y | h(x_i), R) for each particle
    m = observation.shape[-1]
    log_det = jnp.linalg.slogdet(model.R)[1]
    r_inv_diff = jnp.linalg.solve(model.R, diff.T).T  # (N, m)
    mahal = jnp.sum(diff * r_inv_diff, axis=-1)  # (N,)
    return -0.5 * (m * jnp.log(2.0 * jnp.pi) + log_det + mahal)  # type: ignore[no-any-return]


def _normalize_log_weights(log_weights: Array) -> tuple[Array, Array]:
    """Normalize log-weights using the log-sum-exp trick.

    Args:
        log_weights: Unnormalized log-weights, shape (N,).

    Returns:
        Tuple of (normalized weights, log evidence estimate).
    """
    log_evidence = jax.nn.logsumexp(log_weights)
    weights = jnp.exp(log_weights - log_evidence)
    return weights, log_evidence


# ---------------------------------------------------------------------------
# Internal scan carry
# ---------------------------------------------------------------------------


class _ScanCarry(NamedTuple):
    particles: Array  # (N, n)
    weights: Array  # (N,)
    log_likelihood: Array  # ()
    key: Array  # PRNG key


class _ScanOutput(NamedTuple):
    predicted_mean: Array  # (n,)
    predicted_cov: Array  # (n, n)
    filtered_mean: Array  # (n,)
    filtered_cov: Array  # (n, n)
    ess: Array  # ()


# ---------------------------------------------------------------------------
# Full forward pass via lax.scan
# ---------------------------------------------------------------------------


class ParticleFilter:
    """Bootstrap particle filter for nonlinear state-space models.

    Represents the filtering distribution as a weighted set of particles,
    propagated through the transition function and reweighted by the
    observation likelihood at each time step.

    Args:
        n_particles: Number of particles (default 1000).
        resampling: Resampling strategy, one of ``"multinomial"``,
            ``"systematic"``, ``"stratified"`` (default ``"systematic"``).
    """

    def __init__(
        self,
        n_particles: int = 1000,
        resampling: str = "systematic",
    ) -> None:
        self.n_particles = n_particles
        self.resampling = resampling

    def scan(
        self,
        model: NonlinearSSM,
        observations: Array,
        *,
        key: Array,
        initial_state: GaussianState | None = None,
    ) -> FilterResult:
        """Run full forward particle filter via jax.lax.scan.

        Args:
            model: Nonlinear state-space model.
            observations: Observation sequence, shape (T, obs_dim).
            key: JAX PRNG key for stochastic operations.
            initial_state: Initial state belief. Defaults to diffuse prior.

        Returns:
            FilterResult with filtered/predicted states and log-likelihood.
        """
        return _particle_filter_impl(
            model,
            observations,
            self.n_particles,
            key=key,
            resampling=self.resampling,
            initial_state=initial_state,
        )


def particle_filter(
    model: NonlinearSSM,
    observations: Array,
    n_particles: int = 1000,
    *,
    key: Array,
    resampling: str = "systematic",
    initial_state: GaussianState | None = None,
) -> FilterResult:
    """Bootstrap particle filter forward pass.

    Represents the filtering posterior as a weighted set of particles,
    propagated through the nonlinear transition function and reweighted
    by the observation likelihood at each time step. Supports multinomial,
    systematic, and stratified resampling.

    Args:
        model: Nonlinear state-space model with callable f and h.
        observations: Observation sequence, shape (T, obs_dim).
        n_particles: Number of particles (default 1000).
        key: JAX PRNG key for stochastic operations.
        resampling: Resampling strategy, one of ``"multinomial"``,
            ``"systematic"``, ``"stratified"`` (default ``"systematic"``).
        initial_state: Initial state belief. Defaults to diffuse prior.

    Returns:
        FilterResult with filtered/predicted states and log-likelihood.
        The ``filtered_states`` and ``filtered_covariances`` are the
        weighted mean and covariance of the particle cloud at each step.

    Example::

        import jax
        import jax.numpy as jnp
        from dynaris.core.nonlinear import NonlinearSSM
        from dynaris.filters.particle import particle_filter

        model = NonlinearSSM(
            transition_fn=lambda x: x,
            observation_fn=lambda x: x,
            transition_cov=jnp.eye(1),
            observation_cov=jnp.eye(1),
            state_dim=1, obs_dim=1,
        )
        result = particle_filter(
            model, observations, n_particles=1000,
            key=jax.random.PRNGKey(0),
        )
    """
    return _particle_filter_impl(
        model,
        observations,
        n_particles,
        key=key,
        resampling=resampling,
        initial_state=initial_state,
    )


def _particle_filter_impl(
    model: NonlinearSSM,
    observations: Array,
    n_particles: int,
    *,
    key: Array,
    resampling: str,
    initial_state: GaussianState | None,
) -> FilterResult:
    """Internal implementation — resampling function selected before JIT."""
    if initial_state is None:
        initial_state = model.initial_state()

    resample_fn = _RESAMPLE_FNS[resampling]

    # Sample initial particles from the prior distribution.
    key, init_key = jax.random.split(key)
    initial_particles = jax.random.multivariate_normal(
        init_key, initial_state.mean, initial_state.cov, shape=(n_particles,)
    )
    initial_weights = jnp.ones(n_particles) / n_particles

    return _particle_scan(
        model,
        observations,
        initial_particles,
        initial_weights,
        key,
        n_particles,
        resample_fn,
    )


def _particle_scan(
    model: NonlinearSSM,
    observations: Array,
    initial_particles: Array,
    initial_weights: Array,
    key: Array,
    n_particles: int,
    resample_fn: ResampleFn,
) -> FilterResult:
    """JIT-compiled scan loop for the particle filter."""
    init_carry = _ScanCarry(
        particles=initial_particles,
        weights=initial_weights,
        log_likelihood=jnp.array(0.0),
        key=key,
    )
    uniform_w = jnp.ones(n_particles) / n_particles

    @jax.jit
    def _run(carry_init: _ScanCarry, obs_seq: Array) -> tuple[_ScanCarry, _ScanOutput]:
        def _scan_step(carry: _ScanCarry, obs: Array) -> tuple[_ScanCarry, _ScanOutput]:
            key, k_resample, k_predict = jax.random.split(carry.key, 3)

            # --- Resample ---
            indices = resample_fn(k_resample, carry.weights, n_particles)
            resampled = carry.particles[indices]

            # --- Predict (propagate through transition + noise) ---
            predicted = predict_particles(resampled, model, k_predict)

            # Predicted statistics (uniform weights after resampling)
            pred_mean = jnp.mean(predicted, axis=0)
            pred_cov = _weighted_covariance(predicted, uniform_w, pred_mean)

            # --- Update (reweight by observation likelihood) ---
            log_w = compute_log_weights(predicted, obs, model)

            # Handle missing observations
            obs_valid = ~jnp.any(jnp.isnan(obs))
            log_w = jnp.where(obs_valid, log_w, jnp.zeros_like(log_w))

            weights, log_evidence = _normalize_log_weights(log_w)
            # log p(y_t | y_{1:t-1}) ≈ logsumexp(log_w) - log(N)
            ll_contrib = jnp.where(obs_valid, log_evidence - jnp.log(jnp.float32(n_particles)), 0.0)

            # Filtered statistics
            filt_mean = _weighted_mean(predicted, weights)
            filt_cov = _weighted_covariance(predicted, weights, filt_mean)

            # When observation is missing, filtered = predicted
            filt_mean = jnp.where(obs_valid, filt_mean, pred_mean)
            filt_cov = jnp.where(obs_valid, filt_cov, pred_cov)
            weights = jnp.where(obs_valid, weights, uniform_w)

            new_carry = _ScanCarry(
                particles=predicted,
                weights=weights,
                log_likelihood=carry.log_likelihood + ll_contrib,
                key=key,
            )
            output = _ScanOutput(
                predicted_mean=pred_mean,
                predicted_cov=pred_cov,
                filtered_mean=filt_mean,
                filtered_cov=filt_cov,
                ess=effective_sample_size(weights),
            )
            return new_carry, output

        return jax.lax.scan(_scan_step, carry_init, obs_seq)

    final_carry, outputs = _run(init_carry, observations)

    return FilterResult(
        filtered_states=outputs.filtered_mean,
        filtered_covariances=outputs.filtered_cov,
        predicted_states=outputs.predicted_mean,
        predicted_covariances=outputs.predicted_cov,
        log_likelihood=final_carry.log_likelihood,
        observations=observations,
    )
