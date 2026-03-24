"""Extended Kalman Filter for nonlinear state-space models.

Linearizes the transition and observation functions at each time step using
automatic Jacobians via ``jax.jacfwd``, then applies the standard Kalman
predict/update equations to the linearized system.
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


def predict(state: GaussianState, model: NonlinearSSM) -> GaussianState:
    """EKF predict step (time update).

    Propagates the state through the nonlinear transition function and
    linearizes using the Jacobian F = df/dx evaluated at the current mean.

    a_t = f(m_{t-1})
    F_t = Jacobian of f at m_{t-1}
    R_t = F_t @ C_{t-1} @ F_t' + Q
    """
    mean = model.f(state.mean)
    F_jac = jax.jacfwd(model.f)(state.mean)  # (n, n)
    cov = F_jac @ state.cov @ F_jac.T + model.Q
    return GaussianState(mean=mean, cov=cov)


def update(
    predicted: GaussianState,
    observation: Array,
    model: NonlinearSSM,
) -> tuple[GaussianState, Array]:
    """EKF update step (measurement update).

    Linearizes the observation function at the predicted state mean and
    applies the standard Kalman update.

    Returns the filtered state and the log-likelihood contribution.
    Handles missing observations (NaN) by skipping the update.
    """
    y = observation
    y_pred = model.h(predicted.mean)  # (m,)
    H_jac = jax.jacfwd(model.h)(predicted.mean)  # (m, n)

    e = y - y_pred  # innovation (m,)
    S = H_jac @ predicted.cov @ H_jac.T + model.R  # innovation covariance (m, m)

    # Kalman gain: K = P @ H' @ S^{-1}
    K = jnp.linalg.solve(S.T, (predicted.cov @ H_jac.T).T).T  # (n, m)

    filtered_mean = predicted.mean + K @ e
    identity = jnp.eye(predicted.mean.shape[-1])
    filtered_cov = (identity - K @ H_jac) @ predicted.cov

    # Log-likelihood: log N(e; 0, S)
    m = observation.shape[-1]
    log_det = jnp.linalg.slogdet(S)[1]
    mahal = e @ jnp.linalg.solve(S, e)
    ll = -0.5 * (m * jnp.log(2.0 * jnp.pi) + log_det + mahal)

    # Handle missing observations: if any element is NaN, skip update
    obs_valid = ~jnp.any(jnp.isnan(y))
    filtered_mean = jnp.where(obs_valid, filtered_mean, predicted.mean)
    filtered_cov = jnp.where(obs_valid, filtered_cov, predicted.cov)
    ll = jnp.where(obs_valid, ll, 0.0)

    filtered = GaussianState(mean=filtered_mean, cov=filtered_cov)
    return filtered, ll


# ---------------------------------------------------------------------------
# Full forward pass via lax.scan
# ---------------------------------------------------------------------------


class ExtendedKalmanFilter:
    """Extended Kalman Filter implementing the same interface as KalmanFilter.

    Linearizes nonlinear transition and observation functions at each step
    using ``jax.jacfwd`` for automatic Jacobian computation.
    """

    def predict(self, state: GaussianState, model: NonlinearSSM) -> GaussianState:
        """EKF predict step (time update)."""
        return predict(state, model)

    def update(
        self,
        predicted: GaussianState,
        observation: Array,
        model: NonlinearSSM,
    ) -> GaussianState:
        """EKF update step (measurement update)."""
        filtered, _ll = update(predicted, observation, model)
        return filtered

    def scan(
        self,
        model: NonlinearSSM,
        observations: Array,
        initial_state: GaussianState | None = None,
    ) -> FilterResult:
        """Run full forward EKF via jax.lax.scan."""
        return ekf_filter(model, observations, initial_state)


@jax.jit
def ekf_filter(
    model: NonlinearSSM,
    observations: Array,
    initial_state: GaussianState | None = None,
) -> FilterResult:
    """JIT-compiled Extended Kalman Filter forward pass.

    Linearizes the nonlinear transition and observation functions at each
    time step using automatic Jacobians (``jax.jacfwd``), then applies the
    standard Kalman filter predict/update recursion.

    Args:
        model: Nonlinear state-space model with callable f and h.
        observations: Observation sequence, shape (T, obs_dim).
        initial_state: Initial state belief. Defaults to diffuse prior.

    Returns:
        FilterResult with filtered/predicted states and log-likelihood.

    Example::

        import jax.numpy as jnp
        from dynaris.core.nonlinear import NonlinearSSM
        from dynaris.filters.ekf import ekf_filter

        def f(x):
            return x  # random walk

        def h(x):
            return x  # direct observation

        model = NonlinearSSM(
            transition_fn=f, observation_fn=h,
            transition_cov=jnp.eye(1), observation_cov=jnp.eye(1),
            state_dim=1, obs_dim=1,
        )
        result = ekf_filter(model, observations)
    """
    if initial_state is None:
        initial_state = model.initial_state()

    init_carry = _ScanCarry(
        filtered=initial_state,
        log_likelihood=jnp.array(0.0),
    )

    def _scan_step(
        carry: _ScanCarry, obs: Array
    ) -> tuple[_ScanCarry, _ScanOutput]:
        predicted = predict(carry.filtered, model)
        filtered, ll = update(predicted, obs, model)
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
