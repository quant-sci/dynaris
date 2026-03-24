"""Rauch-Tung-Striebel (RTS) smoother for linear-Gaussian models."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from dynaris.core.results import FilterResult, SmootherResult
from dynaris.core.state_space import StateSpaceModel
from dynaris.core.types import GaussianState

# ---------------------------------------------------------------------------
# Internal scan carry / output
# ---------------------------------------------------------------------------


class _RTSCarry(NamedTuple):
    smoothed: GaussianState


class _RTSScanInput(NamedTuple):
    filtered_mean: Array
    filtered_cov: Array
    predicted_mean: Array
    predicted_cov: Array


# ---------------------------------------------------------------------------
# RTS Smoother
# ---------------------------------------------------------------------------


class RTSSmoother:
    """Rauch-Tung-Striebel smoother implementing SmootherProtocol."""

    def smooth(
        self,
        model: StateSpaceModel,
        filter_result: FilterResult,
    ) -> SmootherResult:
        """Run backward RTS smoothing given forward filter results."""
        return rts_smooth(model, filter_result)  # type: ignore[no-any-return]


@jax.jit
def rts_smooth(
    model: StateSpaceModel,
    filter_result: FilterResult,
) -> SmootherResult:
    """JIT-compiled RTS backward smoother.

    Args:
        model: Linear-Gaussian state-space model.
        filter_result: Output from a forward Kalman filter pass.

    Returns:
        SmootherResult with smoothed state estimates.
    """
    # The last filtered state is also the last smoothed state
    last_filtered = GaussianState(
        mean=filter_result.filtered_states[-1],
        cov=filter_result.filtered_covariances[-1],
    )

    init_carry = _RTSCarry(smoothed=last_filtered)

    def _rts_step(
        carry: _RTSCarry, inputs: _RTSScanInput
    ) -> tuple[_RTSCarry, tuple[Array, Array]]:
        # Unpack filtered[t] and predicted[t+1]
        filt_mean = inputs.filtered_mean
        filt_cov = inputs.filtered_cov
        pred_mean = inputs.predicted_mean
        pred_cov = inputs.predicted_cov

        # Smoother gain: G_t = P_{t|t} @ F^T @ P_{t+1|t}^{-1}
        gain = jnp.linalg.solve(
            pred_cov.T, (filt_cov @ model.F.T).T
        ).T

        # Smoothed estimates
        smoothed_mean = (
            filt_mean + gain @ (carry.smoothed.mean - pred_mean)
        )
        smoothed_cov = (
            filt_cov + gain @ (carry.smoothed.cov - pred_cov) @ gain.T
        )

        smoothed = GaussianState(mean=smoothed_mean, cov=smoothed_cov)
        new_carry = _RTSCarry(smoothed=smoothed)
        return new_carry, (smoothed_mean, smoothed_cov)

    # Build reverse scan inputs:
    # For time t, we need filtered[t] and predicted[t+1]
    # filtered[0:T-1] paired with predicted[1:T]
    scan_inputs = _RTSScanInput(
        filtered_mean=filter_result.filtered_states[:-1],
        filtered_cov=filter_result.filtered_covariances[:-1],
        predicted_mean=filter_result.predicted_states[1:],
        predicted_cov=filter_result.predicted_covariances[1:],
    )

    _, (smoothed_means, smoothed_covs) = jax.lax.scan(
        _rts_step, init_carry, scan_inputs, reverse=True
    )

    # Prepend the results: scan gives T-1 results, append the last one
    all_smoothed_means = jnp.concatenate(
        [smoothed_means, filter_result.filtered_states[-1:]], axis=0
    )
    all_smoothed_covs = jnp.concatenate(
        [smoothed_covs, filter_result.filtered_covariances[-1:]], axis=0
    )

    return SmootherResult(
        smoothed_states=all_smoothed_means,
        smoothed_covariances=all_smoothed_covs,
        filtered_states=filter_result.filtered_states,
        filtered_covariances=filter_result.filtered_covariances,
        predicted_states=filter_result.predicted_states,
        predicted_covariances=filter_result.predicted_covariances,
        log_likelihood=filter_result.log_likelihood,
        observations=filter_result.observations,
    )
