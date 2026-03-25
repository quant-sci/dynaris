"""Unscented Kalman Filter for nonlinear state-space models.

Propagates sigma points through nonlinear transition and observation functions
to capture the posterior mean and covariance without linearization. Uses the
scaled unscented transform with configurable alpha, beta, kappa parameters.

References:
    Julier, S.J. and Uhlmann, J.K. (2004). "Unscented Filtering and
    Nonlinear Estimation." Proceedings of the IEEE, 92(3), 401-422.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from dynaris.core.nonlinear import NonlinearSSM
from dynaris.core.results import FilterResult
from dynaris.core.types import GaussianState

# ---------------------------------------------------------------------------
# Sigma-point weights
# ---------------------------------------------------------------------------


class SigmaWeights(NamedTuple):
    """Weights for the unscented transform."""

    wm: Array  # mean weights, shape (2n+1,)
    wc: Array  # covariance weights, shape (2n+1,)
    lam: Array  # scaling parameter lambda (scalar array)


def compute_weights(
    n: int,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> SigmaWeights:
    """Compute sigma-point weights for the scaled unscented transform.

    Args:
        n: State dimension.
        alpha: Spread of sigma points around the mean (typically 1e-4 to 1).
        beta: Prior knowledge of distribution (2.0 is optimal for Gaussian).
        kappa: Secondary scaling parameter (typically 0 or 3-n).

    Returns:
        SigmaWeights with mean weights, covariance weights, and lambda.
    """
    lam = alpha**2 * (n + kappa) - n

    wm = jnp.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
    wm = wm.at[0].set(lam / (n + lam))

    wc = jnp.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
    wc = wc.at[0].set(lam / (n + lam) + (1.0 - alpha**2 + beta))

    return SigmaWeights(wm=wm, wc=wc, lam=jnp.array(lam))


# ---------------------------------------------------------------------------
# Sigma-point generation
# ---------------------------------------------------------------------------


def sigma_points(state: GaussianState, lam: Array) -> Array:
    """Generate 2n+1 sigma points from a Gaussian state.

    Args:
        state: Gaussian belief with mean (n,) and covariance (n, n).
        lam: Scaling parameter lambda (scalar).

    Returns:
        Sigma points, shape (2n+1, n).
    """
    n = state.mean.shape[0]
    scaled_cov = (n + lam) * state.cov
    L = jnp.linalg.cholesky(scaled_cov)  # (n, n)

    # Build sigma points: [mean, mean + L_i, mean - L_i]
    offsets = jnp.concatenate([
        jnp.zeros((1, n)),
        L,    # rows of L as positive offsets
        -L,   # rows of L as negative offsets
    ], axis=0)  # (2n+1, n)

    return state.mean[None, :] + offsets


# ---------------------------------------------------------------------------
# Internal scan carry
# ---------------------------------------------------------------------------


class _ScanCarry(NamedTuple):
    filtered: GaussianState
    log_likelihood: Array  # scalar


class _ScanOutput(NamedTuple):
    predicted_mean: Array
    predicted_cov: Array
    filtered_mean: Array
    filtered_cov: Array


# ---------------------------------------------------------------------------
# Pure-function predict and update steps
# ---------------------------------------------------------------------------


def predict(
    state: GaussianState,
    model: NonlinearSSM,
    weights: SigmaWeights,
) -> GaussianState:
    """UKF predict step (time update).

    Generates sigma points, propagates them through the transition function,
    and recovers the predicted mean and covariance.
    """
    pts = sigma_points(state, weights.lam)  # (2n+1, n)

    # Propagate sigma points through transition function
    pts_pred = jax.vmap(model.f)(pts)  # (2n+1, n)

    # Recover predicted mean
    mean = jnp.sum(weights.wm[:, None] * pts_pred, axis=0)  # (n,)

    # Recover predicted covariance
    diff = pts_pred - mean[None, :]  # (2n+1, n)
    cov = jnp.sum(weights.wc[:, None, None] * (diff[:, :, None] * diff[:, None, :]), axis=0)
    cov = cov + model.Q

    return GaussianState(mean=mean, cov=cov)


def update(
    predicted: GaussianState,
    observation: Array,
    model: NonlinearSSM,
    weights: SigmaWeights,
) -> tuple[GaussianState, Array]:
    """UKF update step (measurement update).

    Generates sigma points from the predicted state, propagates through the
    observation function, and computes the Kalman gain.

    Returns the filtered state and the log-likelihood contribution.
    Handles missing observations (NaN) by skipping the update.
    """
    y = observation
    pts = sigma_points(predicted, weights.lam)  # (2n+1, n)

    # Propagate through observation function
    pts_obs = jax.vmap(model.h)(pts)  # (2n+1, m)

    # Predicted observation mean
    y_pred = jnp.sum(weights.wm[:, None] * pts_obs, axis=0)  # (m,)

    # Innovation covariance S = sum wc * (y_diff)(y_diff)' + R
    y_diff = pts_obs - y_pred[None, :]  # (2n+1, m)
    S = jnp.sum(weights.wc[:, None, None] * (y_diff[:, :, None] * y_diff[:, None, :]), axis=0)
    S = S + model.R  # (m, m)

    # Cross-covariance P_xy = sum wc * (x_diff)(y_diff)'
    x_diff = pts - predicted.mean[None, :]  # (2n+1, n)
    P_xy = jnp.sum(weights.wc[:, None, None] * (x_diff[:, :, None] * y_diff[:, None, :]), axis=0)
    # (n, m)

    # Kalman gain K = P_xy @ S^{-1}
    K = jnp.linalg.solve(S.T, P_xy.T).T  # (n, m)

    # Innovation
    e = y - y_pred  # (m,)

    filtered_mean = predicted.mean + K @ e
    filtered_cov = predicted.cov - K @ S @ K.T

    # Log-likelihood: log N(e; 0, S)
    m = observation.shape[-1]
    log_det = jnp.linalg.slogdet(S)[1]
    mahal = e @ jnp.linalg.solve(S, e)
    ll = -0.5 * (m * jnp.log(2.0 * jnp.pi) + log_det + mahal)

    # Handle missing observations
    obs_valid = ~jnp.any(jnp.isnan(y))
    filtered_mean = jnp.where(obs_valid, filtered_mean, predicted.mean)
    filtered_cov = jnp.where(obs_valid, filtered_cov, predicted.cov)
    ll = jnp.where(obs_valid, ll, 0.0)

    filtered = GaussianState(mean=filtered_mean, cov=filtered_cov)
    return filtered, ll


# ---------------------------------------------------------------------------
# Full forward pass via lax.scan
# ---------------------------------------------------------------------------


class UnscentedKalmanFilter:
    """Unscented Kalman Filter for nonlinear state-space models.

    Uses the scaled unscented transform to propagate sigma points through
    nonlinear functions, avoiding the need for Jacobian computation.

    Args:
        alpha: Spread of sigma points (default 1e-3).
        beta: Prior distribution parameter (default 2.0, optimal for Gaussian).
        kappa: Secondary scaling parameter (default 0.0).
    """

    def __init__(
        self,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

    def predict(self, state: GaussianState, model: NonlinearSSM) -> GaussianState:
        """UKF predict step (time update)."""
        w = compute_weights(model.state_dim, self.alpha, self.beta, self.kappa)
        return predict(state, model, w)

    def update(
        self,
        predicted: GaussianState,
        observation: Array,
        model: NonlinearSSM,
    ) -> GaussianState:
        """UKF update step (measurement update)."""
        w = compute_weights(model.state_dim, self.alpha, self.beta, self.kappa)
        filtered, _ll = update(predicted, observation, model, w)
        return filtered

    def scan(
        self,
        model: NonlinearSSM,
        observations: Array,
        initial_state: GaussianState | None = None,
    ) -> FilterResult:
        """Run full forward UKF via jax.lax.scan."""
        return _ukf_filter_impl(
            model, observations, initial_state,
            self.alpha, self.beta, self.kappa,
        )


def ukf_filter(
    model: NonlinearSSM,
    observations: Array,
    initial_state: GaussianState | None = None,
    *,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> FilterResult:
    """Unscented Kalman Filter forward pass.

    Uses the scaled unscented transform with configurable parameters to
    propagate sigma points through nonlinear transition and observation
    functions.

    Args:
        model: Nonlinear state-space model with callable f and h.
        observations: Observation sequence, shape (T, obs_dim).
        initial_state: Initial state belief. Defaults to diffuse prior.
        alpha: Spread of sigma points around the mean (default 1e-3).
        beta: Prior distribution parameter (default 2.0, optimal for Gaussian).
        kappa: Secondary scaling parameter (default 0.0).

    Returns:
        FilterResult with filtered/predicted states and log-likelihood.

    Example::

        import jax.numpy as jnp
        from dynaris.core.nonlinear import NonlinearSSM
        from dynaris.filters.ukf import ukf_filter

        model = NonlinearSSM(
            transition_fn=lambda x: x,
            observation_fn=lambda x: x,
            transition_cov=jnp.eye(1),
            observation_cov=jnp.eye(1),
            state_dim=1, obs_dim=1,
        )
        result = ukf_filter(model, observations)
    """
    return _ukf_filter_impl(model, observations, initial_state, alpha, beta, kappa)


def _ukf_filter_impl(
    model: NonlinearSSM,
    observations: Array,
    initial_state: GaussianState | None,
    alpha: float,
    beta: float,
    kappa: float,
) -> FilterResult:
    """Internal implementation — weights computed before JIT boundary."""
    if initial_state is None:
        initial_state = model.initial_state()

    weights = compute_weights(model.state_dim, alpha, beta, kappa)
    return _ukf_scan(model, observations, initial_state, weights)


@jax.jit
def _ukf_scan(
    model: NonlinearSSM,
    observations: Array,
    initial_state: GaussianState,
    weights: SigmaWeights,
) -> FilterResult:
    """JIT-compiled scan loop for UKF."""
    init_carry = _ScanCarry(
        filtered=initial_state,
        log_likelihood=jnp.array(0.0),
    )

    def _scan_step(
        carry: _ScanCarry, obs: Array
    ) -> tuple[_ScanCarry, _ScanOutput]:
        predicted = predict(carry.filtered, model, weights)
        filtered, ll = update(predicted, obs, model, weights)
        new_carry = _ScanCarry(
            filtered=filtered,
            log_likelihood=carry.log_likelihood + ll,
        )
        output = _ScanOutput(
            predicted_mean=predicted.mean,
            predicted_cov=predicted.cov,
            filtered_mean=filtered.mean,
            filtered_cov=filtered.cov,
        )
        return new_carry, output

    final_carry, outputs = jax.lax.scan(_scan_step, init_carry, observations)

    return FilterResult(
        filtered_states=outputs.filtered_mean,
        filtered_covariances=outputs.filtered_cov,
        predicted_states=outputs.predicted_mean,
        predicted_covariances=outputs.predicted_cov,
        log_likelihood=final_carry.log_likelihood,
        observations=observations,
    )
