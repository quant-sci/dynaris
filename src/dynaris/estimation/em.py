"""EM algorithm for variance estimation in linear-Gaussian SSMs."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from dynaris.core.results import SmootherResult
from dynaris.core.state_space import StateSpaceModel
from dynaris.filters.kalman import kalman_filter
from dynaris.smoothers.rts import rts_smooth


@dataclass(frozen=True)
class EMResult:
    """Result of EM estimation.

    Attributes:
        model: Fitted StateSpaceModel at convergence.
        log_likelihood: Final log-likelihood value.
        n_iterations: Number of EM iterations performed.
        converged: Whether the algorithm converged.
        log_likelihood_history: Log-likelihood at each iteration.
    """

    model: StateSpaceModel
    log_likelihood: float
    n_iterations: int
    converged: bool
    log_likelihood_history: list[float]


def _e_step(
    model: StateSpaceModel, observations: Array
) -> tuple[SmootherResult, float]:
    """E-step: run Kalman filter + RTS smoother."""
    fr = kalman_filter(model, observations)
    sr = rts_smooth(model, fr)
    return sr, float(fr.log_likelihood)


def _m_step(
    sr: SmootherResult, model: StateSpaceModel
) -> StateSpaceModel:
    """M-step: update Q and R from smoothed sufficient statistics.

    For a general linear-Gaussian SSM:
        Q_new = (1/T) * sum_t [P_{t|T} + x_{t|T} x_{t|T}^T
                                - (P_{t,t-1|T} + x_{t|T} x_{t-1|T}^T) F^T
                                - F (P_{t,t-1|T} + x_{t|T} x_{t-1|T}^T)^T
                                + F (P_{t-1|T} + x_{t-1|T} x_{t-1|T}^T) F^T]

        R_new = (1/T) * sum_t [(y_t - H x_{t|T})(y_t - H x_{t|T})^T
                                + H P_{t|T} H^T]

    We use a simplified version that directly estimates the
    diagonal variances, which is standard for DLM applications.
    """
    obs = sr.observations  # (T, m)
    x_smooth = sr.smoothed_states  # (T, n)
    p_smooth = sr.smoothed_covariances  # (T, n, n)
    n_time = obs.shape[0]

    # --- Estimate R (observation noise covariance) ---
    # residual_t = y_t - H @ x_{t|T}
    residuals = obs - (x_smooth @ model.H.T)  # (T, m)
    # R = (1/T) * sum_t [r_t r_t^T + H P_{t|T} H^T]
    outer_sum = jnp.einsum("ti,tj->ij", residuals, residuals)  # (m, m)
    hp_ht_sum = jnp.sum(model.H @ p_smooth @ model.H.T, axis=0)  # sum over T -> (m, m)
    new_r = (outer_sum + hp_ht_sum) / n_time

    # --- Estimate Q (state noise covariance) ---
    # Using: Q = (1/T) sum_t [P_{t|T} + (x_t - F x_{t-1})(x_t - F x_{t-1})^T
    #                          - F P_{t-1,t|T}^T - P_{t-1,t|T} F^T + F P_{t-1|T} F^T]
    # Simplified: approximate cross-covariance P_{t,t-1|T} via smoother gain
    # For practical DLM usage, we use:
    #   state_resid_t = x_{t|T} - F @ x_{t-1|T}
    #   Q ~ (1/(T-1)) sum_t [state_resid_t state_resid_t^T + P_{t|T} + F P_{t-1|T} F^T]
    # But a cleaner standard approach for the diagonal case:
    x_pred = (x_smooth[:-1] @ model.F.T)  # F @ x_{t-1|T}, shape (T-1, n)
    state_resids = x_smooth[1:] - x_pred  # (T-1, n)
    outer_q = jnp.einsum("ti,tj->ij", state_resids, state_resids)  # (n, n)
    # Add smoothed covariance terms
    p_curr = jnp.sum(p_smooth[1:], axis=0)  # sum P_{t|T} for t=1..T-1
    fp_ft = jnp.sum(
        model.F @ p_smooth[:-1] @ model.F.T, axis=0
    )  # sum F P_{t-1|T} F^T
    new_q = (outer_q + p_curr + fp_ft) / (n_time - 1)
    # Ensure symmetry
    new_q = (new_q + new_q.T) / 2.0
    new_r = (new_r + new_r.T) / 2.0

    return StateSpaceModel(
        transition_matrix=model.transition_matrix,
        observation_matrix=model.observation_matrix,
        state_noise_cov=new_q,
        obs_noise_cov=new_r,
        input_matrix=model.input_matrix,
    )


def fit_em(
    observations: Array,
    initial_model: StateSpaceModel,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> EMResult:
    """Fit a state-space model via the EM algorithm.

    Iteratively updates Q (state noise) and R (observation noise)
    covariance matrices while keeping F, H, and B fixed.

    Args:
        observations: Observation sequence, shape (T, obs_dim).
        initial_model: Starting model with initial variance guesses.
        max_iter: Maximum number of EM iterations.
        tol: Convergence tolerance on log-likelihood change.

    Returns:
        EMResult with the fitted model and convergence details.
    """
    observations = jnp.asarray(observations)
    model = initial_model
    ll_history: list[float] = []
    converged = False

    for i in range(max_iter):
        sr, ll = _e_step(model, observations)
        ll_history.append(ll)

        if i > 0 and abs(ll - ll_history[-2]) < tol:
            converged = True
            break

        model = _m_step(sr, model)

    return EMResult(
        model=model,
        log_likelihood=ll_history[-1] if ll_history else float("-inf"),
        n_iterations=len(ll_history),
        converged=converged,
        log_likelihood_history=ll_history,
    )
