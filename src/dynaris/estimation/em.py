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

        V_new = (1/T) * sum_t [(Y_t - F' m_t^s)(Y_t - F' m_t^s)' + F' C_t^s F]

    We use a simplified version that directly estimates the
    diagonal variances, which is standard for DLM applications.
    """
    obs = sr.observations  # (T, m)
    x_smooth = sr.smoothed_states  # (T, n)
    p_smooth = sr.smoothed_covariances  # (T, n, n)
    n_time = obs.shape[0]

    # --- Estimate V (observational variance) ---
    # residual_t = Y_t - F' @ m_t^(s)
    residuals = obs - (x_smooth @ model.F.T)  # (T, m)
    # V = (1/T) * sum_t [r_t r_t' + F' C_t^(s) F]
    outer_sum = jnp.einsum("ti,tj->ij", residuals, residuals)  # (m, m)
    fc_f_sum = jnp.sum(model.F @ p_smooth @ model.F.T, axis=0)  # sum over T -> (m, m)
    new_v = (outer_sum + fc_f_sum) / n_time

    # --- Estimate W (evolution covariance) ---
    # state_resid_t = m_t^(s) - G @ m_{t-1}^(s)
    x_pred = (x_smooth[:-1] @ model.G.T)  # G @ m_{t-1}^(s), shape (T-1, n)
    state_resids = x_smooth[1:] - x_pred  # (T-1, n)
    outer_w = jnp.einsum("ti,tj->ij", state_resids, state_resids)  # (n, n)
    # Add smoothed covariance terms
    p_curr = jnp.sum(p_smooth[1:], axis=0)  # sum C_t^(s) for t=1..T-1
    gp_gt = jnp.sum(
        model.G @ p_smooth[:-1] @ model.G.T, axis=0
    )  # sum G C_{t-1}^(s) G'
    new_w = (outer_w + p_curr + gp_gt) / (n_time - 1)
    # Ensure symmetry
    new_w = (new_w + new_w.T) / 2.0
    new_v = (new_v + new_v.T) / 2.0

    return StateSpaceModel(
        observation_matrix=model.observation_matrix,
        system_matrix=model.system_matrix,
        obs_cov=new_v,
        evolution_cov=new_w,
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
