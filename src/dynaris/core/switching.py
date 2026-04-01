"""Markov-switching state-space model representation.

Supports K discrete regimes, each with its own linear-Gaussian state-space
model, linked by a Markov transition matrix governing regime switches.

References:
    Kim, C.-J. (1994). "Dynamic Linear Models with Markov-Switching."
    Journal of Econometrics, 60(1-2), 1-22.

    Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of
    Nonstationary Time Series and the Business Cycle." Econometrica, 57(2).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from dynaris.core.state_space import StateSpaceModel
from dynaris.core.types import GaussianState


@dataclass(frozen=True)
class MarkovSwitchingSSM:
    """Markov-switching state-space model with K discrete regimes.

    Each regime has its own :class:`StateSpaceModel`, and the discrete regime
    switches according to a Markov chain with transition matrix P.

    All regime models must share the same ``state_dim`` and ``obs_dim``.

    Attributes:
        models: Tuple of K regime-specific ``StateSpaceModel`` instances.
        transition_matrix: Markov transition matrix P, shape (K, K).
            ``P[i, j] = Pr(S_t = j | S_{t-1} = i)``. Rows sum to 1.
        initial_probs: Initial regime probability vector, shape (K,).

    Example::

        from dynaris import LocalLevel, MarkovSwitchingSSM
        import jax.numpy as jnp

        calm = LocalLevel(sigma_level=1.0, sigma_obs=5.0)
        volatile = LocalLevel(sigma_level=5.0, sigma_obs=20.0)

        model = MarkovSwitchingSSM(
            models=(calm, volatile),
            transition_matrix=jnp.array([[0.95, 0.05],
                                          [0.10, 0.90]]),
            initial_probs=jnp.array([0.5, 0.5]),
        )
    """

    models: tuple[StateSpaceModel, ...]
    transition_matrix: Array  # (K, K)
    initial_probs: Array  # (K,)

    def __post_init__(self) -> None:
        k = len(self.models)
        if k < 1:
            msg = "At least one regime model is required."
            raise ValueError(msg)

        sd = self.models[0].state_dim
        od = self.models[0].obs_dim
        for i, m in enumerate(self.models):
            if m.state_dim != sd or m.obs_dim != od:
                msg = (
                    f"All regime models must have the same dimensions. "
                    f"Model 0 has state_dim={sd}, obs_dim={od}, but model {i} "
                    f"has state_dim={m.state_dim}, obs_dim={m.obs_dim}."
                )
                raise ValueError(msg)

        if self.transition_matrix.shape != (k, k):
            msg = f"transition_matrix must be ({k}, {k}), got {self.transition_matrix.shape}."
            raise ValueError(msg)

        if self.initial_probs.shape != (k,):
            msg = f"initial_probs must be ({k},), got {self.initial_probs.shape}."
            raise ValueError(msg)

        # Cache stacked arrays for vectorized operations
        object.__setattr__(
            self,
            "_G_stack",
            jnp.stack([m.G for m in self.models]),
        )
        object.__setattr__(
            self,
            "_F_stack",
            jnp.stack([m.F for m in self.models]),
        )
        object.__setattr__(
            self,
            "_W_stack",
            jnp.stack([m.W for m in self.models]),
        )
        object.__setattr__(
            self,
            "_V_stack",
            jnp.stack([m.V for m in self.models]),
        )

    # --- Properties ---

    @property
    def n_regimes(self) -> int:
        """Number of discrete regimes K."""
        return len(self.models)

    @property
    def state_dim(self) -> int:
        """Continuous state dimension (shared across regimes)."""
        return self.models[0].state_dim

    @property
    def obs_dim(self) -> int:
        """Observation dimension (shared across regimes)."""
        return self.models[0].obs_dim

    @property
    def G_stack(self) -> Array:  # noqa: N802
        """Stacked system matrices, shape (K, n, n)."""
        return self._G_stack  # type: ignore[attr-defined]

    @property
    def F_stack(self) -> Array:  # noqa: N802
        """Stacked observation matrices, shape (K, m, n)."""
        return self._F_stack  # type: ignore[attr-defined]

    @property
    def W_stack(self) -> Array:  # noqa: N802
        """Stacked evolution covariances, shape (K, n, n)."""
        return self._W_stack  # type: ignore[attr-defined]

    @property
    def V_stack(self) -> Array:  # noqa: N802
        """Stacked observation covariances, shape (K, m, m)."""
        return self._V_stack  # type: ignore[attr-defined]

    # --- Factory methods ---

    def initial_state(
        self,
        mean: Array | None = None,
        cov: Array | None = None,
    ) -> GaussianState:
        """Create a default initial GaussianState.

        Uses the probability-weighted mixture of the K regime priors.

        Args:
            mean: Initial state mean. Defaults to zeros.
            cov: Initial state covariance. Defaults to 1e6 * I.

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
        return (
            f"MarkovSwitchingSSM(n_regimes={self.n_regimes}, "
            f"state_dim={self.state_dim}, obs_dim={self.obs_dim})"
        )

    # --- JAX pytree registration ---

    def tree_flatten(self) -> tuple[list[Array], dict[str, object]]:
        """Flatten into JAX pytree leaves and auxiliary data."""
        leaves = [
            self._G_stack,  # type: ignore[attr-defined]
            self._F_stack,  # type: ignore[attr-defined]
            self._W_stack,  # type: ignore[attr-defined]
            self._V_stack,  # type: ignore[attr-defined]
            self.transition_matrix,
            self.initial_probs,
        ]
        aux = {
            "n_regimes": self.n_regimes,
            "state_dim": self.state_dim,
            "obs_dim": self.obs_dim,
        }
        return leaves, aux

    @classmethod
    def tree_unflatten(
        cls, aux_data: dict[str, object], children: list[Array]
    ) -> MarkovSwitchingSSM:
        """Reconstruct from JAX pytree leaves."""
        g_stack, f_stack, w_stack, v_stack, trans_mat, init_probs = children
        k: int = aux_data["n_regimes"]  # type: ignore[assignment]
        models = tuple(
            StateSpaceModel(
                observation_matrix=f_stack[i],
                system_matrix=g_stack[i],
                obs_cov=v_stack[i],
                evolution_cov=w_stack[i],
            )
            for i in range(k)
        )
        return cls(
            models=models,
            transition_matrix=trans_mat,
            initial_probs=init_probs,
        )


jax.tree_util.register_pytree_node_class(MarkovSwitchingSSM)
