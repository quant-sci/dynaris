"""Pure NumPy Kalman filter — no JAX dependency required.

A lightweight implementation for environments without GPU/TPU support
or where JAX installation is not feasible. Provides the same Kalman
filter algorithm but without JIT compilation or autodiff.

Usage::

    from dynaris.backends.numpy_backend import kalman_filter_numpy

    result = kalman_filter_numpy(F, G, V, W, observations)

Note:
    This backend does NOT support autodiff, JIT, vmap, or GPU.
    For production use with optimization, use the JAX-based filters.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class NumpyFilterResult(NamedTuple):
    """Result of the NumPy Kalman filter.

    Attributes:
        filtered_states: Filtered state means, shape (T, n).
        filtered_covariances: Filtered state covariances, shape (T, n, n).
        predicted_states: One-step-ahead predicted means, shape (T, n).
        predicted_covariances: One-step-ahead predicted covs, shape (T, n, n).
        log_likelihood: Total log-likelihood scalar.
        observations: Input observations, shape (T, m).
    """

    filtered_states: NDArray[np.float64]
    filtered_covariances: NDArray[np.float64]
    predicted_states: NDArray[np.float64]
    predicted_covariances: NDArray[np.float64]
    log_likelihood: float
    observations: NDArray[np.float64]


def kalman_filter_numpy(
    F: NDArray[np.float64],  # noqa: N803
    G: NDArray[np.float64],  # noqa: N803
    V: NDArray[np.float64],  # noqa: N803
    W: NDArray[np.float64],  # noqa: N803
    observations: NDArray[np.float64],
    initial_mean: NDArray[np.float64] | None = None,
    initial_cov: NDArray[np.float64] | None = None,
) -> NumpyFilterResult:
    """Pure NumPy Kalman filter (no JAX required).

    Uses West & Harrison notation:
    - System: theta_t = G @ theta_{t-1} + w_t,  w_t ~ N(0, W)
    - Obs:    Y_t = F @ theta_t + v_t,           v_t ~ N(0, V)

    Args:
        F: Observation matrix, shape (m, n).
        G: System/transition matrix, shape (n, n).
        V: Observation covariance, shape (m, m).
        W: Evolution covariance, shape (n, n).
        observations: Observation sequence, shape (T, m).
        initial_mean: Initial state mean, shape (n,). Defaults to zeros.
        initial_cov: Initial state covariance, shape (n, n).
            Defaults to 1e6 * I (diffuse prior).

    Returns:
        NumpyFilterResult with filtered/predicted states and log-likelihood.

    Example::

        import numpy as np
        from dynaris.backends.numpy_backend import kalman_filter_numpy

        F = np.array([[1.0]])
        G = np.array([[1.0]])
        V = np.array([[100.0]])
        W = np.array([[1.0]])
        y = np.random.randn(100, 1) * 10
        result = kalman_filter_numpy(F, G, V, W, y)
    """
    observations = np.asarray(observations, dtype=np.float64)
    t_len, m = observations.shape
    n = G.shape[0]

    if initial_mean is None:
        initial_mean = np.zeros(n)
    if initial_cov is None:
        initial_cov = np.eye(n) * 1e6

    # Pre-allocate output arrays
    filt_means = np.zeros((t_len, n))
    filt_covs = np.zeros((t_len, n, n))
    pred_means = np.zeros((t_len, n))
    pred_covs = np.zeros((t_len, n, n))
    total_ll = 0.0

    mean = initial_mean.copy()
    cov = initial_cov.copy()

    log_2pi = np.log(2.0 * np.pi)

    for t in range(t_len):
        # --- Predict ---
        pred_mean = G @ mean
        pred_cov = G @ cov @ G.T + W
        pred_means[t] = pred_mean
        pred_covs[t] = pred_cov

        y = observations[t]

        # Check for missing observations
        if np.any(np.isnan(y)):
            filt_means[t] = pred_mean
            filt_covs[t] = pred_cov
            mean = pred_mean
            cov = pred_cov
            continue

        # --- Update ---
        e = y - F @ pred_mean  # innovation
        s = F @ pred_cov @ F.T + V  # innovation covariance
        k = np.linalg.solve(s.T, (pred_cov @ F.T).T).T  # Kalman gain

        filt_mean = pred_mean + k @ e
        filt_cov = (np.eye(n) - k @ F) @ pred_cov

        # Log-likelihood contribution
        _sign, log_det = np.linalg.slogdet(s)
        mahal = e @ np.linalg.solve(s, e)
        ll = -0.5 * (m * log_2pi + log_det + mahal)
        total_ll += ll

        filt_means[t] = filt_mean
        filt_covs[t] = filt_cov
        mean = filt_mean
        cov = filt_cov

    return NumpyFilterResult(
        filtered_states=filt_means,
        filtered_covariances=filt_covs,
        predicted_states=pred_means,
        predicted_covariances=pred_covs,
        log_likelihood=float(total_ll),
        observations=observations,
    )
