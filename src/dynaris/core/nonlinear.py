"""Nonlinear state-space model representation for EKF/UKF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

from dynaris.core.types import GaussianState

# Type aliases for transition and observation functions.
# transition_fn: (state_vec,) -> predicted_state_vec
# observation_fn: (state_vec,) -> predicted_observation_vec
TransitionFn = Callable[[Array], Array]
ObservationFn = Callable[[Array], Array]


@dataclass(frozen=True)
class NonlinearSSM:
    """Nonlinear state-space model for use with the Extended Kalman Filter.

    State equation:     theta_t = f(theta_{t-1}) + omega_t,  omega_t ~ N(0, Q)
    Observation eq:     Y_t = h(theta_t) + nu_t,             nu_t ~ N(0, R)

    The Jacobians of f and h are computed automatically via ``jax.jacfwd``,
    so no manual derivation is required.

    Attributes:
        transition_fn: f, maps state (n,) -> state (n,).
        observation_fn: h, maps state (n,) -> observation (m,).
        transition_cov: Q, evolution noise covariance, shape (n, n).
        observation_cov: R, observation noise covariance, shape (m, m).
        state_dim: Dimension of the state vector.
        obs_dim: Dimension of the observation vector.
    """

    transition_fn: TransitionFn
    observation_fn: ObservationFn
    transition_cov: Array  # Q: (n, n)
    observation_cov: Array  # R: (m, m)
    state_dim: int
    obs_dim: int

    # --- Short aliases ---

    @property
    def Q(self) -> Array:  # noqa: N802
        """Evolution / transition noise covariance."""
        return self.transition_cov

    @property
    def R(self) -> Array:  # noqa: N802
        """Observation noise covariance."""
        return self.observation_cov

    @property
    def f(self) -> TransitionFn:
        """Transition function alias."""
        return self.transition_fn

    @property
    def h(self) -> ObservationFn:
        """Observation function alias."""
        return self.observation_fn

    # --- Factory methods ---

    def initial_state(
        self,
        mean: Array | None = None,
        cov: Array | None = None,
    ) -> GaussianState:
        """Create a default initial GaussianState for this model.

        Args:
            mean: Initial state mean. Defaults to zeros.
            cov: Initial state covariance. Defaults to 1e6 * I (diffuse prior).

        Returns:
            GaussianState with the specified or default initial conditions.
        """
        n = self.state_dim
        if mean is None:
            mean = jnp.zeros(n)
        if cov is None:
            cov = jnp.eye(n) * 1e6
        return GaussianState(mean=mean, cov=cov)

    def __repr__(self) -> str:
        return f"NonlinearSSM(state_dim={self.state_dim}, obs_dim={self.obs_dim})"

    # --- JAX pytree registration ---

    def tree_flatten(self) -> tuple[list[Array], dict[str, object]]:
        """Flatten into JAX pytree leaves and auxiliary data."""
        leaves = [self.transition_cov, self.observation_cov]
        aux = {
            "transition_fn": self.transition_fn,
            "observation_fn": self.observation_fn,
            "state_dim": self.state_dim,
            "obs_dim": self.obs_dim,
        }
        return leaves, aux

    @classmethod
    def tree_unflatten(
        cls, aux_data: dict[str, object], children: list[Array]
    ) -> NonlinearSSM:
        """Reconstruct from JAX pytree leaves."""
        return cls(
            transition_fn=aux_data["transition_fn"],  # type: ignore[arg-type]
            observation_fn=aux_data["observation_fn"],  # type: ignore[arg-type]
            transition_cov=children[0],
            observation_cov=children[1],
            state_dim=aux_data["state_dim"],  # type: ignore[arg-type]
            obs_dim=aux_data["obs_dim"],  # type: ignore[arg-type]
        )


jax.tree_util.register_pytree_node_class(NonlinearSSM)
