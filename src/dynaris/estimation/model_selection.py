"""Model selection utilities for Markov-switching state-space models.

Provides information criteria (AIC, BIC) and automated regime count selection
for comparing switching models with different numbers of regimes.
"""

from __future__ import annotations

import jax.numpy as jnp


def _count_switching_params(n_regimes: int, state_dim: int, obs_dim: int) -> int:
    """Count free parameters in a Markov-switching model.

    Per regime: G (n*n) + F (m*n) + W (n*(n+1)/2) + V (m*(m+1)/2).
    Transition matrix: K*(K-1) free parameters.
    Initial probs: K-1 free parameters.
    """
    n, m, k = state_dim, obs_dim, n_regimes
    per_regime = n * n + m * n + n * (n + 1) // 2 + m * (m + 1) // 2
    return k * per_regime + k * (k - 1) + (k - 1)


def switching_aic(
    log_likelihood: float,
    n_regimes: int,
    state_dim: int,
    obs_dim: int,
) -> float:
    """Akaike Information Criterion for a Markov-switching model.

    AIC = -2 * log_likelihood + 2 * n_params

    Args:
        log_likelihood: Total log-likelihood from the Hamilton filter.
        n_regimes: Number of discrete regimes K.
        state_dim: Continuous state dimension n.
        obs_dim: Observation dimension m.

    Returns:
        AIC value (lower is better).
    """
    n_params = _count_switching_params(n_regimes, state_dim, obs_dim)
    return float(-2.0 * log_likelihood + 2.0 * n_params)


def switching_bic(
    log_likelihood: float,
    n_regimes: int,
    state_dim: int,
    obs_dim: int,
    n_obs: int,
) -> float:
    """Bayesian Information Criterion for a Markov-switching model.

    BIC = -2 * log_likelihood + log(n_obs) * n_params

    Args:
        log_likelihood: Total log-likelihood from the Hamilton filter.
        n_regimes: Number of discrete regimes K.
        state_dim: Continuous state dimension n.
        obs_dim: Observation dimension m.
        n_obs: Number of observations T.

    Returns:
        BIC value (lower is better).
    """
    n_params = _count_switching_params(n_regimes, state_dim, obs_dim)
    return float(-2.0 * log_likelihood + jnp.log(n_obs) * n_params)
