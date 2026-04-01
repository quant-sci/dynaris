"""EM algorithm for Dynamic Factor Model estimation.

Jointly estimates the factor loading matrix (Lambda), diagonal observation
noise (R), and extracts smoothed latent factor estimates using the Kalman
filter and RTS smoother.

References:
    Shumway, R.H. and Stoffer, D.S. (1982). "An Approach to Time Series
    Smoothing and Forecasting Using the EM Algorithm."

    Watson, M.W. and Engle, R.F. (1983). "Alternative Algorithms for the
    Estimation of Dynamic Factor, MIMIC, and Varying Coefficient Regression
    Models."
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from dynaris.core.results import SmootherResult
from dynaris.core.state_space import StateSpaceModel
from dynaris.filters.kalman import kalman_filter
from dynaris.models.factor import (
    DynamicFactorModel,
    apply_identification_constraints,
    initialize_loadings_pca,
)
from dynaris.smoothers.rts import rts_smooth


@dataclass(frozen=True)
class DFMResult:
    """Result of Dynamic Factor Model estimation.

    Attributes:
        model: Fitted StateSpaceModel at convergence.
        loadings: Estimated loading matrix, shape (m, r).
        factor_states: Smoothed factor estimates, shape (T, r).
        factor_covariances: Smoothed factor covariances, shape (T, r, r).
        explained_variance: Variance explained per factor, shape (r,).
        log_likelihood: Final log-likelihood value.
        n_iterations: Number of EM iterations performed.
        converged: Whether the algorithm converged.
        log_likelihood_history: Log-likelihood at each iteration.
        smoother_result: Full smoother output for downstream use.
    """

    model: StateSpaceModel
    loadings: Array
    factor_states: Array
    factor_covariances: Array
    explained_variance: Array
    log_likelihood: float
    n_iterations: int
    converged: bool
    log_likelihood_history: list[float]
    smoother_result: SmootherResult


def fit_dfm_em(
    observations: Array,
    n_factors: int,
    factor_order: int = 1,
    max_iter: int = 200,
    tol: float = 1e-6,
    init_method: str = "pca",
    loadings: Array | None = None,
) -> DFMResult:
    """Fit a Dynamic Factor Model via the EM algorithm.

    Jointly estimates the factor loading matrix (Lambda) and diagonal
    observation noise (R), while keeping factor dynamics (G, Q) fixed.

    Args:
        observations: Observation matrix, shape (T, m).
        n_factors: Number of latent factors (r).
        factor_order: AR order for factor dynamics. Default 1 (random walk).
        max_iter: Maximum number of EM iterations.
        tol: Convergence tolerance on log-likelihood change.
        init_method: Initialization method. ``"pca"`` (default) or
            ``"random"``.
        loadings: Custom initial loadings, shape (m, r). Overrides
            ``init_method`` if provided.

    Returns:
        DFMResult with estimated loadings, factors, and model.

    Example::

        from dynaris.estimation.dfm import fit_dfm_em
        import jax.numpy as jnp

        # 10 variables, 2 factors
        result = fit_dfm_em(panel_data, n_factors=2)
        print(result.loadings.shape)  # (10, 2)
        print(result.factor_states.shape)  # (T, 2)
    """
    observations = jnp.asarray(observations)
    _t, m = observations.shape
    r = n_factors

    # --- Initialize ---
    if loadings is not None:
        init_loadings = jnp.asarray(loadings)
    elif init_method == "pca":
        init_loadings, _ = initialize_loadings_pca(observations, r)
    else:
        init_loadings = jnp.ones((m, r)) * 0.1

    # Initial observation noise from PCA residuals
    model = DynamicFactorModel(
        n_factors=r,
        n_variables=m,
        factor_order=factor_order,
        loadings=init_loadings,
        obs_noise=1.0,
    )

    # Initial R estimate from residual variance
    fr_init = kalman_filter(model, observations)
    residuals_init = observations - fr_init.filtered_states @ model.F.T
    r_diag = jnp.maximum(jnp.var(residuals_init, axis=0), 1e-4)
    model = DynamicFactorModel(
        n_factors=r,
        n_variables=m,
        factor_order=factor_order,
        loadings=init_loadings,
        obs_noise=r_diag,
    )

    ll_history: list[float] = []
    converged = False
    current_loadings = init_loadings

    for i in range(max_iter):
        # E-step
        fr = kalman_filter(model, observations)
        sr = rts_smooth(model, fr)
        ll = float(fr.log_likelihood)
        ll_history.append(ll)

        if i > 0 and abs(ll - ll_history[-2]) < tol:
            converged = True
            break

        # M-step
        current_loadings, r_diag_new = _dfm_m_step(sr, observations, r, factor_order)

        # Apply identification constraints
        current_loadings = apply_identification_constraints(current_loadings)

        # Rebuild model with updated parameters
        model = DynamicFactorModel(
            n_factors=r,
            n_variables=m,
            factor_order=factor_order,
            loadings=current_loadings,
            obs_noise=r_diag_new,
        )

    # Final E-step for output
    fr = kalman_filter(model, observations)
    sr = rts_smooth(model, fr)
    if not ll_history or float(fr.log_likelihood) != ll_history[-1]:
        ll_history.append(float(fr.log_likelihood))

    # Extract factor states (first r components of state)
    factor_states = sr.smoothed_states[:, :r]
    factor_covs = sr.smoothed_covariances[:, :r, :r]

    # Explained variance per factor
    explained_var = jnp.sum(current_loadings**2, axis=0)

    return DFMResult(
        model=model,
        loadings=current_loadings,
        factor_states=factor_states,
        factor_covariances=factor_covs,
        explained_variance=explained_var,
        log_likelihood=ll_history[-1] if ll_history else float("-inf"),
        n_iterations=len(ll_history),
        converged=converged,
        log_likelihood_history=ll_history,
        smoother_result=sr,
    )


def _dfm_m_step(
    sr: SmootherResult,
    observations: Array,
    n_factors: int,
    factor_order: int,
) -> tuple[Array, Array]:
    """DFM M-step: update Lambda and R from smoothed sufficient statistics.

    Args:
        sr: SmootherResult from E-step.
        observations: (T, m) observation matrix.
        n_factors: Number of factors r.
        factor_order: AR order (state_dim may be r * factor_order).

    Returns:
        Tuple of (Lambda_new, R_diag_new).
        Lambda_new: (m, r) updated loading matrix.
        R_diag_new: (m,) updated diagonal observation noise.
    """
    r = n_factors
    m_s = sr.smoothed_states[:, :r]  # (T, r) — first r components
    c_s = sr.smoothed_covariances[:, :r, :r]  # (T, r, r)

    # Lambda update: Lambda = S_yf @ inv(S_ff)
    # S_yf = sum_t y_t @ m_t^s'  ->  (m, r)
    s_yf = jnp.einsum("ti,tj->ij", observations, m_s)

    # S_ff = sum_t (C_t^s + m_t^s @ m_t^s')  ->  (r, r)
    s_ff = jnp.sum(c_s, axis=0) + jnp.einsum("ti,tj->ij", m_s, m_s)

    # Solve for Lambda: Lambda = S_yf @ S_ff^{-1}
    # Use solve for numerical stability: S_ff @ Lambda' = S_yf'
    lambda_new = jnp.linalg.solve(s_ff, s_yf.T).T  # (m, r)

    # R update (diagonal)
    residuals = observations - m_s @ lambda_new.T  # (T, m)
    r_diag = jnp.mean(residuals**2, axis=0)  # (m,) — first term

    # Add smoothed uncertainty: Lambda C Lambda' contribution
    lcl = jnp.einsum("ij,tjk,ik->ti", lambda_new, c_s, lambda_new)  # (T, m)
    r_diag = r_diag + jnp.mean(lcl, axis=0)

    # Ensure positivity
    r_diag = jnp.maximum(r_diag, 1e-6)

    return lambda_new, r_diag
