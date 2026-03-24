"""State-space model representation."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from dynaris.core.types import GaussianState


@dataclass(frozen=True)
class StateSpaceModel:
    """Linear-Gaussian state-space model.

    State equation:   x_t = F @ x_{t-1} + B @ u_t + w_t,  w_t ~ N(0, Q)
    Observation eq:   y_t = H @ x_t + v_t,                 v_t ~ N(0, R)

    Attributes:
        transition_matrix: F, shape (state_dim, state_dim).
        observation_matrix: H, shape (obs_dim, state_dim).
        state_noise_cov: Q, shape (state_dim, state_dim).
        obs_noise_cov: R, shape (obs_dim, obs_dim).
        input_matrix: B, shape (state_dim, input_dim) or None.
    """

    transition_matrix: Array  # F: (n, n)
    observation_matrix: Array  # H: (m, n)
    state_noise_cov: Array  # Q: (n, n)
    obs_noise_cov: Array  # R: (m, m)
    input_matrix: Array | None = None  # B: (n, p) or None

    # --- Dimension properties ---

    @property
    def state_dim(self) -> int:
        """Dimension of the latent state vector."""
        return int(self.transition_matrix.shape[-1])

    @property
    def obs_dim(self) -> int:
        """Dimension of the observation vector."""
        return int(self.observation_matrix.shape[-2])

    # --- Short aliases for math-style access ---

    @property
    def F(self) -> Array:  # noqa: N802
        """Alias for transition_matrix."""
        return self.transition_matrix

    @property
    def H(self) -> Array:  # noqa: N802
        """Alias for observation_matrix."""
        return self.observation_matrix

    @property
    def Q(self) -> Array:  # noqa: N802
        """Alias for state_noise_cov."""
        return self.state_noise_cov

    @property
    def R(self) -> Array:  # noqa: N802
        """Alias for obs_noise_cov."""
        return self.obs_noise_cov

    @property
    def B(self) -> Array | None:  # noqa: N802
        """Alias for input_matrix."""
        return self.input_matrix

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

    # --- Composition ---

    def __add__(self, other: StateSpaceModel) -> StateSpaceModel:
        """Compose two models via block-diagonal concatenation.

        The resulting model has:
        - F_new = block_diag(self.F, other.F)
        - H_new = [self.H, other.H]  (horizontal concatenation)
        - Q_new = block_diag(self.Q, other.Q)
        - R_new = self.R + other.R  (shared observation noise adds)
        """
        n1, n2 = self.state_dim, other.state_dim

        transition = jnp.block([
            [self.F, jnp.zeros((n1, n2))],
            [jnp.zeros((n2, n1)), other.F],
        ])
        observation = jnp.concatenate([self.H, other.H], axis=-1)
        state_noise = jnp.block([
            [self.Q, jnp.zeros((n1, n2))],
            [jnp.zeros((n2, n1)), other.Q],
        ])
        obs_noise = self.R + other.R

        input_mat: Array | None = None
        if self.input_matrix is not None and other.input_matrix is not None:
            p1 = self.input_matrix.shape[-1]
            p2 = other.input_matrix.shape[-1]
            input_mat = jnp.block([
                [self.input_matrix, jnp.zeros((n1, p2))],
                [jnp.zeros((n2, p1)), other.input_matrix],
            ])
        elif self.input_matrix is not None:
            input_mat = jnp.concatenate(
                [
                    self.input_matrix,
                    jnp.zeros((n2, self.input_matrix.shape[-1])),
                ],
                axis=-2,
            )
        elif other.input_matrix is not None:
            input_mat = jnp.concatenate(
                [
                    jnp.zeros((n1, other.input_matrix.shape[-1])),
                    other.input_matrix,
                ],
                axis=-2,
            )

        return StateSpaceModel(
            transition_matrix=transition,
            observation_matrix=observation,
            state_noise_cov=state_noise,
            obs_noise_cov=obs_noise,
            input_matrix=input_mat,
        )

    def __repr__(self) -> str:
        b_info = (
            f", input_dim={self.input_matrix.shape[-1]}"
            if self.input_matrix is not None
            else ""
        )
        return (
            f"StateSpaceModel(state_dim={self.state_dim}, "
            f"obs_dim={self.obs_dim}{b_info})"
        )

    # --- JAX pytree registration ---

    def tree_flatten(self) -> tuple[list[Array], dict[str, bool]]:
        """Flatten into JAX pytree leaves and auxiliary data."""
        has_input = self.input_matrix is not None
        leaves: list[Array] = [
            self.transition_matrix,
            self.observation_matrix,
            self.state_noise_cov,
            self.obs_noise_cov,
        ]
        if has_input:
            leaves.append(self.input_matrix)  # type: ignore[arg-type]
        return leaves, {"has_input": has_input}

    @classmethod
    def tree_unflatten(
        cls, aux_data: dict[str, bool], children: list[Array]
    ) -> StateSpaceModel:
        """Reconstruct from JAX pytree leaves."""
        if aux_data["has_input"]:
            return cls(
                transition_matrix=children[0],
                observation_matrix=children[1],
                state_noise_cov=children[2],
                obs_noise_cov=children[3],
                input_matrix=children[4],
            )
        return cls(
            transition_matrix=children[0],
            observation_matrix=children[1],
            state_noise_cov=children[2],
            obs_noise_cov=children[3],
        )


jax.tree_util.register_pytree_node_class(StateSpaceModel)
