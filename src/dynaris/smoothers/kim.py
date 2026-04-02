"""Kim smoother for Markov-switching state-space models.

Runs a backward pass over the Hamilton filter output to produce smoothed
state estimates and regime probabilities. Uses per-regime RTS-like
smoother steps with Kim's collapse approximation.

References:
    Kim, C.-J. (1994). "Dynamic Linear Models with Markov-Switching."
    Journal of Econometrics, 60(1-2), 1-22.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from dynaris.core.results import SwitchingFilterResult, SwitchingSmootherResult
from dynaris.core.switching import MarkovSwitchingSSM

# ---------------------------------------------------------------------------
# Internal scan types
# ---------------------------------------------------------------------------


class _KimCarry(NamedTuple):
    smoothed_means: Array  # (K, n)
    smoothed_covs: Array  # (K, n, n)
    smoothed_probs: Array  # (K,)


class _KimScanInput(NamedTuple):
    regime_filtered_means: Array  # (K, n)
    regime_filtered_covs: Array  # (K, n, n)
    regime_predicted_means: Array  # (K, n)
    regime_predicted_covs: Array  # (K, n, n)
    regime_filtered_probs: Array  # (K,)
    regime_predicted_probs: Array  # (K,)  -- at t+1


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


def _regime_rts_step(
    filt_mean: Array,
    filt_cov: Array,
    pred_mean: Array,
    pred_cov: Array,
    smoothed_mean: Array,
    smoothed_cov: Array,
    g: Array,
) -> tuple[Array, Array]:
    """Single-regime RTS smoother step.

    Args:
        filt_mean: Filtered mean at t, (n,).
        filt_cov: Filtered cov at t, (n, n).
        pred_mean: Predicted mean at t+1, (n,).
        pred_cov: Predicted cov at t+1, (n, n).
        smoothed_mean: Smoothed mean at t+1, (n,).
        smoothed_cov: Smoothed cov at t+1, (n, n).
        g: System matrix G, (n, n).

    Returns:
        (smoothed_mean_t, smoothed_cov_t).
    """
    gain = jnp.linalg.solve(pred_cov.T, (filt_cov @ g.T).T).T
    sm_mean = filt_mean + gain @ (smoothed_mean - pred_mean)
    sm_cov = filt_cov + gain @ (smoothed_cov - pred_cov) @ gain.T
    return sm_mean, sm_cov


def _mixture_collapse(regime_means: Array, regime_covs: Array, probs: Array) -> tuple[Array, Array]:
    """Collapse K regime Gaussians into a single mixture."""
    mean = probs @ regime_means  # (n,)
    delta = regime_means - mean[None, :]  # (K, n)
    spread = delta[:, :, None] * delta[:, None, :]  # (K, n, n)
    cov = jnp.einsum("k,kij->ij", probs, regime_covs + spread)
    return mean, cov


# ---------------------------------------------------------------------------
# Kim Smoother
# ---------------------------------------------------------------------------


class KimSmoother:
    """Kim smoother for Markov-switching state-space models."""

    def smooth(
        self,
        model: MarkovSwitchingSSM,
        filter_result: SwitchingFilterResult,
    ) -> SwitchingSmootherResult:
        """Run backward Kim smoothing given forward Hamilton filter results."""
        return kim_smooth(model, filter_result)


def kim_smooth(
    model: MarkovSwitchingSSM,
    filter_result: SwitchingFilterResult,
) -> SwitchingSmootherResult:
    """Kim backward smoother for Markov-switching models.

    Args:
        model: Markov-switching state-space model.
        filter_result: Output from a forward Hamilton filter pass.

    Returns:
        SwitchingSmootherResult with smoothed states and regime probabilities.

    Example::

        from dynaris.filters.hamilton import hamilton_filter
        from dynaris.smoothers.kim import kim_smooth

        filt = hamilton_filter(model, observations)
        smoothed = kim_smooth(model, filt)
    """
    return _kim_scan(model, filter_result)  # type: ignore[no-any-return]


@jax.jit
def _kim_scan(
    model: MarkovSwitchingSSM,
    fr: SwitchingFilterResult,
) -> SwitchingSmootherResult:
    """JIT-compiled Kim smoother scan."""
    # Initialize with the last filtered states
    last_means = fr.regime_filtered_states[-1]  # (K, n)
    last_covs = fr.regime_filtered_covs[-1]  # (K, n, n)
    last_probs = fr.regime_filtered_probs[-1]  # (K,)

    init_carry = _KimCarry(
        smoothed_means=last_means,
        smoothed_covs=last_covs,
        smoothed_probs=last_probs,
    )

    # Vectorized RTS step over K regimes
    _v_rts = jax.vmap(
        _regime_rts_step,
        in_axes=(0, 0, 0, 0, 0, 0, 0),
    )

    def _kim_step(
        carry: _KimCarry, inputs: _KimScanInput
    ) -> tuple[_KimCarry, tuple[Array, Array, Array, Array, Array]]:
        # 1. Smoothed regime probabilities
        # smoothed_probs[i, t] = filtered_probs[i, t] *
        #   sum_j [P[i,j] * smoothed_probs[j, t+1] / predicted_probs[j, t+1]]
        pred_probs_tp1 = inputs.regime_predicted_probs  # (K,) at t+1
        pred_probs_tp1 = jnp.maximum(pred_probs_tp1, 1e-10)
        ratio = carry.smoothed_probs / pred_probs_tp1  # (K,)
        smoothed_probs = inputs.regime_filtered_probs * (model.transition_matrix @ ratio)
        smoothed_probs = jnp.maximum(smoothed_probs, 1e-10)
        smoothed_probs = smoothed_probs / jnp.sum(smoothed_probs)

        # 2. Per-regime RTS smoother step (vmapped)
        sm_means, sm_covs = _v_rts(
            inputs.regime_filtered_means,
            inputs.regime_filtered_covs,
            inputs.regime_predicted_means,
            inputs.regime_predicted_covs,
            carry.smoothed_means,
            carry.smoothed_covs,
            model.G_stack,
        )

        # 3. Collapse to overall mixture
        overall_mean, overall_cov = _mixture_collapse(sm_means, sm_covs, smoothed_probs)

        new_carry = _KimCarry(
            smoothed_means=sm_means,
            smoothed_covs=sm_covs,
            smoothed_probs=smoothed_probs,
        )
        return new_carry, (overall_mean, overall_cov, smoothed_probs, sm_means, sm_covs)

    # Build scan inputs: for time t, need filtered[t] and predicted[t+1]
    scan_inputs = _KimScanInput(
        regime_filtered_means=fr.regime_filtered_states[:-1],
        regime_filtered_covs=fr.regime_filtered_covs[:-1],
        regime_predicted_means=fr.regime_predicted_states[1:],
        regime_predicted_covs=fr.regime_predicted_covs[1:],
        regime_filtered_probs=fr.regime_filtered_probs[:-1],
        regime_predicted_probs=fr.regime_predicted_probs[1:],
    )

    _, (sm_means_seq, sm_covs_seq, sm_probs_seq, regime_sm_means_seq, regime_sm_covs_seq) = (
        jax.lax.scan(_kim_step, init_carry, scan_inputs, reverse=True)
    )

    # Append the last time step (last filtered = last smoothed)
    last_overall_mean, last_overall_cov = _mixture_collapse(last_means, last_covs, last_probs)

    all_sm_means = jnp.concatenate([sm_means_seq, last_overall_mean[None, :]], axis=0)
    all_sm_covs = jnp.concatenate([sm_covs_seq, last_overall_cov[None, :, :]], axis=0)
    all_sm_probs = jnp.concatenate([sm_probs_seq, last_probs[None, :]], axis=0)
    all_regime_sm_means = jnp.concatenate([regime_sm_means_seq, last_means[None, :, :]], axis=0)
    all_regime_sm_covs = jnp.concatenate([regime_sm_covs_seq, last_covs[None, :, :, :]], axis=0)

    return SwitchingSmootherResult(
        smoothed_states=all_sm_means,
        smoothed_covariances=all_sm_covs,
        filtered_states=fr.filtered_states,
        filtered_covariances=fr.filtered_covariances,
        predicted_states=fr.predicted_states,
        predicted_covariances=fr.predicted_covariances,
        log_likelihood=fr.log_likelihood,
        observations=fr.observations,
        regime_smoothed_probs=all_sm_probs,
        regime_filtered_probs=fr.regime_filtered_probs,
        regime_smoothed_states=all_regime_sm_means,
        regime_smoothed_covs=all_regime_sm_covs,
    )
