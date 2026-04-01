"""Memory-efficient filtering via JAX gradient checkpointing.

For very long time series (T > 10,000), the standard Kalman filter stores
all intermediate states in memory. ``jax.checkpoint`` trades memory for
compute by recomputing intermediates during the backward pass instead
of storing them.

Usage::

    from dynaris.utils.checkpoint import checkpointed_kalman_filter

    # Same API as kalman_filter, but uses less memory for long series
    result = checkpointed_kalman_filter(model, observations)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from dynaris.core.results import FilterResult
from dynaris.core.state_space import StateSpaceModel
from dynaris.core.types import GaussianState
from dynaris.filters.kalman import predict, update


class _ScanCarry:
    """Lightweight carry for checkpointed scan (avoid NamedTuple for checkpoint)."""


@jax.jit
def checkpointed_kalman_filter(
    model: StateSpaceModel,
    observations: Array,
    initial_state: GaussianState | None = None,
) -> FilterResult:
    """Memory-efficient Kalman filter using gradient checkpointing.

    Identical output to :func:`~dynaris.filters.kalman.kalman_filter`,
    but uses ``jax.checkpoint`` on each scan step to reduce peak memory
    for very long time series. The trade-off is ~2x compute during
    backward passes (autodiff).

    Args:
        model: Linear-Gaussian state-space model.
        observations: Observation sequence, shape (T, obs_dim).
        initial_state: Initial state belief. Defaults to diffuse prior.

    Returns:
        FilterResult with filtered/predicted states and log-likelihood.
    """
    if initial_state is None:
        initial_state = model.initial_state()

    init_carry = (initial_state.mean, initial_state.cov, jnp.array(0.0))

    @jax.checkpoint
    def _scan_step(
        carry: tuple[Array, Array, Array], obs: Array
    ) -> tuple[tuple[Array, Array, Array], tuple[Array, Array, Array, Array]]:
        mean, cov, ll_acc = carry
        state = GaussianState(mean=mean, cov=cov)

        predicted = predict(state, model)
        filtered, ll = update(predicted, obs, model)

        new_carry = (filtered.mean, filtered.cov, ll_acc + ll)
        output = (predicted.mean, predicted.cov, filtered.mean, filtered.cov)
        return new_carry, output

    (_, _, total_ll), (pred_m, pred_c, filt_m, filt_c) = jax.lax.scan(
        _scan_step, init_carry, observations
    )

    return FilterResult(
        filtered_states=filt_m,
        filtered_covariances=filt_c,
        predicted_states=pred_m,
        predicted_covariances=pred_c,
        log_likelihood=total_ll,
        observations=observations,
    )
