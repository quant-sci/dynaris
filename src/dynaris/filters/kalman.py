"""Kalman filter: forward filtering for linear-Gaussian state-space models."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from dynaris.core.results import FilterResult
from dynaris.core.state_space import StateSpaceModel
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


def predict(
    state: GaussianState,
    model: StateSpaceModel,
    u: Array | None = None,
) -> GaussianState:
    """Kalman predict step (time update).

    x_{t|t-1} = F @ x_{t-1|t-1} + B @ u_t
    P_{t|t-1} = F @ P_{t-1|t-1} @ F^T + Q
    """
    mean = model.F @ state.mean
    if u is not None and model.input_matrix is not None:
        mean = mean + model.input_matrix @ u
    cov = model.F @ state.cov @ model.F.T + model.Q
    return GaussianState(mean=mean, cov=cov)


def update(
    predicted: GaussianState,
    observation: Array,
    model: StateSpaceModel,
) -> tuple[GaussianState, Array]:
    """Kalman update step (measurement update).

    Returns the filtered state and the log-likelihood contribution.
    Handles missing observations (NaN) by skipping the update.
    """
    y = observation
    innovation = y - model.H @ predicted.mean  # (m,)
    innovation_cov = model.H @ predicted.cov @ model.H.T + model.R  # (m, m)

    # Kalman gain: K = P_pred @ H^T @ S^{-1}
    kalman_gain = jnp.linalg.solve(
        innovation_cov.T, (predicted.cov @ model.H.T).T
    ).T  # (n, m)

    filtered_mean = predicted.mean + kalman_gain @ innovation
    # Joseph form for numerical stability:
    # P = (I - K @ H) @ P_pred @ (I - K @ H)^T + K @ R @ K^T
    # Simplified (equivalent for exact arithmetic):
    identity = jnp.eye(predicted.mean.shape[-1])
    filtered_cov = (identity - kalman_gain @ model.H) @ predicted.cov

    # Log-likelihood contribution: log N(innovation; 0, S)
    m = observation.shape[-1]
    log_det = jnp.linalg.slogdet(innovation_cov)[1]
    mahal = innovation @ jnp.linalg.solve(innovation_cov, innovation)
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


class KalmanFilter:
    """Kalman filter implementing the FilterProtocol.

    All methods are JIT-compatible pure functions.
    """

    def predict(
        self,
        state: GaussianState,
        model: StateSpaceModel,
        u: Array | None = None,
    ) -> GaussianState:
        """Kalman predict step (time update)."""
        return predict(state, model, u)

    def update(
        self,
        predicted: GaussianState,
        observation: Array,
        model: StateSpaceModel,
    ) -> GaussianState:
        """Kalman update step (measurement update)."""
        filtered, _ll = update(predicted, observation, model)
        return filtered

    def scan(
        self,
        model: StateSpaceModel,
        observations: Array,
        initial_state: GaussianState | None = None,
        inputs: Array | None = None,
    ) -> FilterResult:
        """Run full forward Kalman filter via jax.lax.scan."""
        return kalman_filter(model, observations, initial_state, inputs)  # type: ignore[no-any-return]


@jax.jit
def kalman_filter(
    model: StateSpaceModel,
    observations: Array,
    initial_state: GaussianState | None = None,
    inputs: Array | None = None,
) -> FilterResult:
    """JIT-compiled Kalman forward filter.

    Args:
        model: Linear-Gaussian state-space model.
        observations: Observation sequence, shape (T, obs_dim).
        initial_state: Initial state belief. Defaults to diffuse prior.
        inputs: Optional control inputs, shape (T, input_dim).

    Returns:
        FilterResult with filtered/predicted states and log-likelihood.
    """
    if initial_state is None:
        initial_state = model.initial_state()

    init_carry = _ScanCarry(
        filtered=initial_state,
        log_likelihood=jnp.array(0.0),
    )

    def _scan_step(
        carry: _ScanCarry, xs: tuple[Array, Array | None]
    ) -> tuple[_ScanCarry, _ScanOutput]:
        obs, u = xs
        predicted = predict(carry.filtered, model, u)
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

    # Build scan inputs: (observations, inputs_or_zeros)
    if inputs is None:
        dummy_inputs = jnp.zeros((observations.shape[0], 1))
        scan_inputs: tuple[Array, Array | None] = (observations, dummy_inputs)
    else:
        scan_inputs = (observations, inputs)

    final_carry, outputs = jax.lax.scan(_scan_step, init_carry, scan_inputs)

    return FilterResult(
        filtered_states=outputs.filtered_mean,
        filtered_covariances=outputs.filtered_cov,
        predicted_states=outputs.predicted_mean,
        predicted_covariances=outputs.predicted_cov,
        log_likelihood=final_carry.log_likelihood,
        observations=observations,
    )
