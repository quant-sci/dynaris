"""Hamilton filter for Markov-switching state-space models.

Runs K parallel Kalman filters (one per regime) with Kim's (1994)
moment-matching collapse approximation to keep the state representation
tractable. Tracks filtered regime probabilities via Bayesian updating.

References:
    Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of
    Nonstationary Time Series and the Business Cycle." Econometrica, 57(2).

    Kim, C.-J. (1994). "Dynamic Linear Models with Markov-Switching."
    Journal of Econometrics, 60(1-2), 1-22.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from dynaris.core.results import SwitchingFilterResult
from dynaris.core.switching import MarkovSwitchingSSM
from dynaris.core.types import GaussianState

# ---------------------------------------------------------------------------
# Internal scan types
# ---------------------------------------------------------------------------


class _ScanCarry(NamedTuple):
    regime_means: Array  # (K, n)
    regime_covs: Array  # (K, n, n)
    regime_probs: Array  # (K,)
    log_likelihood: Array  # ()


class _ScanOutput(NamedTuple):
    predicted_mean: Array  # (n,)
    predicted_cov: Array  # (n, n)
    filtered_mean: Array  # (n,)
    filtered_cov: Array  # (n, n)
    regime_filtered_probs: Array  # (K,)
    regime_predicted_probs: Array  # (K,)
    regime_filtered_means: Array  # (K, n)
    regime_filtered_covs: Array  # (K, n, n)
    regime_predicted_means: Array  # (K, n)
    regime_predicted_covs: Array  # (K, n, n)


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


def _collapse_step(
    regime_means: Array,
    regime_covs: Array,
    regime_probs: Array,
    transition_matrix: Array,
) -> tuple[Array, Array, Array]:
    """Kim's collapse: moment-matching mixture into K target-regime states.

    For each target regime j, compute the probability-weighted mixture of K
    source-regime filtered states, conditioned on transitioning to j.

    Args:
        regime_means: Per-regime filtered means, (K, n).
        regime_covs: Per-regime filtered covariances, (K, n, n).
        regime_probs: Filtered regime probabilities, (K,).
        transition_matrix: Markov transition matrix P, (K, K).

    Returns:
        Tuple of (collapsed_means, collapsed_covs, predicted_probs).
        collapsed_means: (K, n), collapsed_covs: (K, n, n),
        predicted_probs: (K,).
    """
    # Predicted regime probabilities: P[i,j] * pi[i] summed over i
    predicted_probs = transition_matrix.T @ regime_probs  # (K,)
    predicted_probs = jnp.maximum(predicted_probs, 1e-10)

    # Conditional weights: w[i|j] = P[i,j] * pi[i] / predicted_probs[j]
    # Shape: (K_source, K_target)
    weights = (transition_matrix * regime_probs[:, None]) / predicted_probs[None, :]

    # Collapsed means: (K_target, n)
    collapsed_means = weights.T @ regime_means  # (K, n)

    # Collapsed covariances with spread-of-means
    # delta[i, j] = regime_means[i] - collapsed_means[j]
    delta = regime_means[:, None, :] - collapsed_means[None, :, :]  # (K, K, n)
    spread = delta[:, :, :, None] * delta[:, :, None, :]  # (K, K, n, n)

    # weighted sum of (cov + spread)
    collapsed_covs = jnp.einsum("ij,ikl->jkl", weights, regime_covs) + jnp.einsum(
        "ij,ijkl->jkl", weights, spread
    )

    return collapsed_means, collapsed_covs, predicted_probs


def _regime_predict(mean: Array, cov: Array, g: Array, w: Array) -> tuple[Array, Array]:
    """Single-regime Kalman predict step.

    Args:
        mean: State mean, (n,).
        cov: State covariance, (n, n).
        g: System matrix G, (n, n).
        w: Evolution covariance W, (n, n).

    Returns:
        Predicted (mean, cov).
    """
    pred_mean = g @ mean
    pred_cov = g @ cov @ g.T + w
    return pred_mean, pred_cov


def _regime_update(
    pred_mean: Array,
    pred_cov: Array,
    obs: Array,
    f: Array,
    v: Array,
) -> tuple[Array, Array, Array]:
    """Single-regime Kalman update step.

    Args:
        pred_mean: Predicted state mean, (n,).
        pred_cov: Predicted state covariance, (n, n).
        obs: Observation vector, (m,).
        f: Observation matrix F, (m, n).
        v: Observation covariance V, (m, m).

    Returns:
        Tuple of (filtered_mean, filtered_cov, log_likelihood).
    """
    e = obs - f @ pred_mean  # (m,)
    s = f @ pred_cov @ f.T + v  # (m, m)

    # Kalman gain
    k_gain = jnp.linalg.solve(s.T, (pred_cov @ f.T).T).T  # (n, m)

    filt_mean = pred_mean + k_gain @ e
    filt_cov = (jnp.eye(pred_mean.shape[0]) - k_gain @ f) @ pred_cov

    # Log-likelihood contribution
    m = obs.shape[-1]
    log_det = jnp.linalg.slogdet(s)[1]
    mahal = e @ jnp.linalg.solve(s, e)
    ll = -0.5 * (m * jnp.log(2.0 * jnp.pi) + log_det + mahal)

    return filt_mean, filt_cov, ll


def _mixture_collapse(regime_means: Array, regime_covs: Array, probs: Array) -> tuple[Array, Array]:
    """Collapse K regime-specific Gaussians into a single mixture.

    Args:
        regime_means: (K, n).
        regime_covs: (K, n, n).
        probs: (K,).

    Returns:
        (collapsed_mean, collapsed_cov).
    """
    mean = probs @ regime_means  # (n,)
    delta = regime_means - mean[None, :]  # (K, n)
    spread = delta[:, :, None] * delta[:, None, :]  # (K, n, n)
    cov = jnp.einsum("k,kij->ij", probs, regime_covs + spread)
    return mean, cov


# ---------------------------------------------------------------------------
# Full forward pass
# ---------------------------------------------------------------------------


class HamiltonFilter:
    """Hamilton filter for Markov-switching state-space models.

    Runs K parallel Kalman filters with Kim's collapse approximation
    to maintain a tractable representation.
    """

    def scan(
        self,
        model: MarkovSwitchingSSM,
        observations: Array,
        initial_state: GaussianState | None = None,
    ) -> SwitchingFilterResult:
        """Run full forward Hamilton filter via jax.lax.scan."""
        return hamilton_filter(model, observations, initial_state)


def hamilton_filter(
    model: MarkovSwitchingSSM,
    observations: Array,
    initial_state: GaussianState | None = None,
) -> SwitchingFilterResult:
    """Hamilton forward filter for Markov-switching state-space models.

    Runs K parallel Kalman filters with Kim's collapse approximation,
    updating filtered regime probabilities at each time step.

    Args:
        model: Markov-switching state-space model with K regimes.
        observations: Observation sequence, shape (T, obs_dim).
        initial_state: Initial state belief (shared across regimes).
            Defaults to diffuse prior.

    Returns:
        SwitchingFilterResult with filtered states, regime probabilities,
        and log-likelihood.

    Example::

        from dynaris import LocalLevel, MarkovSwitchingSSM
        from dynaris.filters.hamilton import hamilton_filter
        import jax.numpy as jnp

        model = MarkovSwitchingSSM(
            models=(LocalLevel(1, 5), LocalLevel(5, 20)),
            transition_matrix=jnp.array([[0.95, 0.05], [0.10, 0.90]]),
            initial_probs=jnp.array([0.5, 0.5]),
        )
        result = hamilton_filter(model, observations)
    """
    if initial_state is None:
        initial_state = model.initial_state()

    return _hamilton_scan(model, observations, initial_state)


@jax.jit
def _hamilton_scan(
    model: MarkovSwitchingSSM,
    observations: Array,
    initial_state: GaussianState,
) -> SwitchingFilterResult:
    """JIT-compiled Hamilton filter scan."""
    k = model.n_regimes

    # Initialize: all regimes start with the same state
    init_means = jnp.tile(initial_state.mean[None, :], (k, 1))  # (K, n)
    init_covs = jnp.tile(initial_state.cov[None, :, :], (k, 1, 1))  # (K, n, n)

    init_carry = _ScanCarry(
        regime_means=init_means,
        regime_covs=init_covs,
        regime_probs=model.initial_probs,
        log_likelihood=jnp.array(0.0),
    )

    # Vectorized predict and update over regimes
    _v_predict = jax.vmap(_regime_predict, in_axes=(0, 0, 0, 0))
    _v_update = jax.vmap(_regime_update, in_axes=(0, 0, None, 0, 0))

    def _scan_step(carry: _ScanCarry, obs: Array) -> tuple[_ScanCarry, _ScanOutput]:
        # 1. Collapse: moment-matching mixture for each target regime
        collapsed_means, collapsed_covs, predicted_probs = _collapse_step(
            carry.regime_means,
            carry.regime_covs,
            carry.regime_probs,
            model.transition_matrix,
        )

        # 2. Predict: vmap Kalman predict over K regimes
        pred_means, pred_covs = _v_predict(
            collapsed_means, collapsed_covs, model.G_stack, model.W_stack
        )

        # 3. Update: vmap Kalman update over K regimes
        filt_means, filt_covs, lls = _v_update(
            pred_means, pred_covs, obs, model.F_stack, model.V_stack
        )

        # Handle missing observations
        obs_valid = ~jnp.any(jnp.isnan(obs))
        lls = jnp.where(obs_valid, lls, jnp.zeros_like(lls))
        filt_means = jnp.where(obs_valid, filt_means, pred_means)
        filt_covs = jnp.where(obs_valid, filt_covs, pred_covs)

        # 4. Update regime probabilities (log-sum-exp for stability)
        max_ll = jnp.max(lls)
        joint_unnorm = jnp.exp(lls - max_ll) * predicted_probs
        evidence = jnp.sum(joint_unnorm)
        filtered_probs = joint_unnorm / evidence
        filtered_probs = jnp.maximum(filtered_probs, 1e-10)
        filtered_probs = filtered_probs / jnp.sum(filtered_probs)

        log_evidence = jnp.where(obs_valid, max_ll + jnp.log(evidence), 0.0)

        # When obs is missing, keep previous regime probs
        filtered_probs = jnp.where(obs_valid, filtered_probs, carry.regime_probs)

        # 5. Mixture collapse for overall estimates
        overall_filt_mean, overall_filt_cov = _mixture_collapse(
            filt_means, filt_covs, filtered_probs
        )
        overall_pred_mean, overall_pred_cov = _mixture_collapse(
            pred_means, pred_covs, predicted_probs
        )

        new_carry = _ScanCarry(
            regime_means=filt_means,
            regime_covs=filt_covs,
            regime_probs=filtered_probs,
            log_likelihood=carry.log_likelihood + log_evidence,
        )

        output = _ScanOutput(
            predicted_mean=overall_pred_mean,
            predicted_cov=overall_pred_cov,
            filtered_mean=overall_filt_mean,
            filtered_cov=overall_filt_cov,
            regime_filtered_probs=filtered_probs,
            regime_predicted_probs=predicted_probs,
            regime_filtered_means=filt_means,
            regime_filtered_covs=filt_covs,
            regime_predicted_means=pred_means,
            regime_predicted_covs=pred_covs,
        )
        return new_carry, output

    final_carry, outputs = jax.lax.scan(_scan_step, init_carry, observations)

    return SwitchingFilterResult(
        filtered_states=outputs.filtered_mean,
        filtered_covariances=outputs.filtered_cov,
        predicted_states=outputs.predicted_mean,
        predicted_covariances=outputs.predicted_cov,
        log_likelihood=final_carry.log_likelihood,
        observations=observations,
        regime_filtered_probs=outputs.regime_filtered_probs,
        regime_predicted_probs=outputs.regime_predicted_probs,
        regime_filtered_states=outputs.regime_filtered_means,
        regime_filtered_covs=outputs.regime_filtered_covs,
        regime_predicted_states=outputs.regime_predicted_means,
        regime_predicted_covs=outputs.regime_predicted_covs,
    )
