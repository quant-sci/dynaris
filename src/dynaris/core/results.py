"""Result containers for filtering and smoothing operations."""

from __future__ import annotations

from typing import NamedTuple

from jax import Array


class FilterResult(NamedTuple):
    """Output of a forward filtering pass.

    All array fields have a leading time dimension T (number of observations).

    Attributes:
        filtered_states: Filtered state means, shape (T, state_dim).
        filtered_covariances: Filtered state covariances, shape (T, state_dim, state_dim).
        predicted_states: One-step-ahead predicted means, shape (T, state_dim).
        predicted_covariances: One-step-ahead predicted covariances,
            shape (T, state_dim, state_dim).
        log_likelihood: Total log-likelihood scalar, shape ().
        observations: The input observations, shape (T, obs_dim).
    """

    filtered_states: Array  # (T, n)
    filtered_covariances: Array  # (T, n, n)
    predicted_states: Array  # (T, n)
    predicted_covariances: Array  # (T, n, n)
    log_likelihood: Array  # ()
    observations: Array  # (T, m)


class SmootherResult(NamedTuple):
    """Output of a backward smoothing pass.

    Extends filter output with smoothed estimates.

    Attributes:
        smoothed_states: Smoothed state means, shape (T, state_dim).
        smoothed_covariances: Smoothed state covariances,
            shape (T, state_dim, state_dim).
        filtered_states: Filtered state means, shape (T, state_dim).
        filtered_covariances: Filtered state covariances,
            shape (T, state_dim, state_dim).
        predicted_states: One-step-ahead predicted means,
            shape (T, state_dim).
        predicted_covariances: One-step-ahead predicted covariances,
            shape (T, state_dim, state_dim).
        log_likelihood: Total log-likelihood scalar, shape ().
        observations: The input observations, shape (T, obs_dim).
    """

    smoothed_states: Array  # (T, n)
    smoothed_covariances: Array  # (T, n, n)
    filtered_states: Array  # (T, n)
    filtered_covariances: Array  # (T, n, n)
    predicted_states: Array  # (T, n)
    predicted_covariances: Array  # (T, n, n)
    log_likelihood: Array  # ()
    observations: Array  # (T, m)


class SwitchingFilterResult(NamedTuple):
    """Output of a Hamilton forward filtering pass for switching models.

    Contains mixture-collapsed state estimates and per-regime quantities.

    Attributes:
        filtered_states: Mixture-collapsed filtered means, shape (T, n).
        filtered_covariances: Mixture-collapsed filtered covs, shape (T, n, n).
        predicted_states: Mixture-collapsed predicted means, shape (T, n).
        predicted_covariances: Mixture-collapsed predicted covs, shape (T, n, n).
        log_likelihood: Total log-likelihood scalar, shape ().
        observations: Input observations, shape (T, m).
        regime_filtered_probs: Filtered regime probabilities, shape (T, K).
        regime_predicted_probs: Predicted regime probabilities, shape (T, K).
        regime_filtered_states: Per-regime filtered means, shape (T, K, n).
        regime_filtered_covs: Per-regime filtered covs, shape (T, K, n, n).
        regime_predicted_states: Per-regime predicted means, shape (T, K, n).
        regime_predicted_covs: Per-regime predicted covs, shape (T, K, n, n).
    """

    filtered_states: Array  # (T, n)
    filtered_covariances: Array  # (T, n, n)
    predicted_states: Array  # (T, n)
    predicted_covariances: Array  # (T, n, n)
    log_likelihood: Array  # ()
    observations: Array  # (T, m)
    regime_filtered_probs: Array  # (T, K)
    regime_predicted_probs: Array  # (T, K)
    regime_filtered_states: Array  # (T, K, n)
    regime_filtered_covs: Array  # (T, K, n, n)
    regime_predicted_states: Array  # (T, K, n)
    regime_predicted_covs: Array  # (T, K, n, n)


class SwitchingSmootherResult(NamedTuple):
    """Output of a Kim backward smoothing pass for switching models.

    Attributes:
        smoothed_states: Mixture-collapsed smoothed means, shape (T, n).
        smoothed_covariances: Mixture-collapsed smoothed covs, shape (T, n, n).
        filtered_states: Mixture-collapsed filtered means, shape (T, n).
        filtered_covariances: Mixture-collapsed filtered covs, shape (T, n, n).
        predicted_states: Mixture-collapsed predicted means, shape (T, n).
        predicted_covariances: Mixture-collapsed predicted covs, shape (T, n, n).
        log_likelihood: Total log-likelihood scalar, shape ().
        observations: Input observations, shape (T, m).
        regime_smoothed_probs: Smoothed regime probabilities, shape (T, K).
        regime_filtered_probs: Filtered regime probabilities, shape (T, K).
        regime_smoothed_states: Per-regime smoothed means, shape (T, K, n).
        regime_smoothed_covs: Per-regime smoothed covs, shape (T, K, n, n).
    """

    smoothed_states: Array  # (T, n)
    smoothed_covariances: Array  # (T, n, n)
    filtered_states: Array  # (T, n)
    filtered_covariances: Array  # (T, n, n)
    predicted_states: Array  # (T, n)
    predicted_covariances: Array  # (T, n, n)
    log_likelihood: Array  # ()
    observations: Array  # (T, m)
    regime_smoothed_probs: Array  # (T, K)
    regime_filtered_probs: Array  # (T, K)
    regime_smoothed_states: Array  # (T, K, n)
    regime_smoothed_covs: Array  # (T, K, n, n)
