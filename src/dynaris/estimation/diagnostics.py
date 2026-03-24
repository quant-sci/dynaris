"""Model diagnostics for fitted state-space models."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from scipy import stats

from dynaris.core.results import FilterResult
from dynaris.core.state_space import StateSpaceModel


def standardized_residuals(
    filter_result: FilterResult,
    model: StateSpaceModel,
) -> Array:
    """Compute standardized (one-step-ahead) prediction residuals.

    e_t = (y_t - H @ x_{t|t-1}) / sqrt(H @ P_{t|t-1} @ H^T + R)

    Returns:
        Standardized residuals, shape (T,) for univariate or (T, obs_dim).
    """
    obs = filter_result.observations  # (T, m)
    pred_states = filter_result.predicted_states  # (T, n)
    pred_covs = filter_result.predicted_covariances  # (T, n, n)

    # Innovation: y_t - H @ x_{t|t-1}
    innovations = obs - pred_states @ model.H.T  # (T, m)

    # Innovation covariance: H @ P_{t|t-1} @ H^T + R
    # Shape: (T, m, m)
    innovation_covs = jnp.einsum(
        "ij,tjk,lk->til", model.H, pred_covs, model.H
    ) + model.R[None, :, :]

    # For univariate case, standardize directly
    # For multivariate, use diagonal elements
    std_devs = jnp.sqrt(
        jnp.diagonal(innovation_covs, axis1=-2, axis2=-1)
    )  # (T, m)

    std_resids = innovations / std_devs

    # Squeeze if univariate
    if std_resids.shape[-1] == 1:
        return std_resids[:, 0]
    return std_resids


def acf(x: Array, n_lags: int = 20) -> Array:
    """Compute the sample autocorrelation function.

    Args:
        x: 1D array of residuals, shape (T,).
        n_lags: Number of lags to compute.

    Returns:
        Autocorrelations at lags 0, 1, ..., n_lags. Shape (n_lags + 1,).
    """
    x = jnp.asarray(x).ravel()
    n = x.shape[0]
    x_centered = x - jnp.mean(x)
    var = jnp.sum(x_centered**2) / n

    lags = jnp.arange(n_lags + 1)

    def _acf_at_lag(lag: Array) -> Array:
        # For lag 0, return 1.0
        shifted = jnp.roll(x_centered, lag)
        # Zero out the rolled-in values
        mask = jnp.arange(n) >= lag
        cov = jnp.sum(x_centered * shifted * mask) / n
        return jnp.where(lag == 0, 1.0, cov / var)

    return jnp.vectorize(_acf_at_lag)(lags)  # type: ignore[no-any-return]


def pacf(x: Array, n_lags: int = 20) -> Array:
    """Compute the sample partial autocorrelation function via Durbin-Levinson.

    Args:
        x: 1D array of residuals, shape (T,).
        n_lags: Number of lags to compute.

    Returns:
        Partial autocorrelations at lags 0, 1, ..., n_lags.
        Shape (n_lags + 1,). PACF at lag 0 is 1.0.
    """
    acf_vals = acf(x, n_lags)

    result = [1.0]  # lag 0

    # Durbin-Levinson algorithm
    phi = float(acf_vals[1])
    result.append(phi)

    phi_prev = [phi]

    for k in range(2, n_lags + 1):
        # phi_k,k = (r(k) - sum_{j=1}^{k-1} phi_{k-1,j} * r(k-j))
        #           / (1 - sum_{j=1}^{k-1} phi_{k-1,j} * r(j))
        numer = float(acf_vals[k])
        denom = 1.0
        for j in range(len(phi_prev)):
            numer -= phi_prev[j] * float(acf_vals[k - j - 1])
            denom -= phi_prev[j] * float(acf_vals[j + 1])

        if abs(denom) < 1e-12:
            result.append(0.0)
            phi_prev = [0.0] * k
            continue

        phi_kk = numer / denom
        result.append(phi_kk)

        # Update phi coefficients
        new_phi = []
        for j in range(len(phi_prev)):
            new_phi.append(phi_prev[j] - phi_kk * phi_prev[-(j + 1)])
        new_phi.append(phi_kk)
        phi_prev = new_phi

    return jnp.array(result)


def ljung_box(
    residuals: Array, n_lags: int = 10
) -> tuple[float, float]:
    """Ljung-Box test for autocorrelation in residuals.

    Tests H0: the residuals are independently distributed (no autocorrelation).

    Args:
        residuals: 1D array of (standardized) residuals, shape (T,).
        n_lags: Number of lags to include in the test.

    Returns:
        Tuple of (test_statistic, p_value).
    """
    residuals = jnp.asarray(residuals).ravel()
    n = residuals.shape[0]
    acf_vals = acf(residuals, n_lags)

    # Q = n(n+2) * sum_{k=1}^{h} r_k^2 / (n-k)
    q_stat = 0.0
    for k in range(1, n_lags + 1):
        rk = float(acf_vals[k])
        q_stat += rk**2 / (n - k)
    q_stat *= float(n * (n + 2))

    # Under H0, Q ~ chi-squared(n_lags)
    p_value = float(1.0 - stats.chi2.cdf(q_stat, df=n_lags))

    return q_stat, p_value
