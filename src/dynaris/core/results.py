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
