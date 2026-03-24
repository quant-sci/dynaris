"""Fundamental types for dynaris state-space models."""

from __future__ import annotations

from dataclasses import dataclass

import jax
from jax import Array


@dataclass(frozen=True)
class GaussianState:
    """A Gaussian belief state: mean vector and covariance matrix.

    Attributes:
        mean: State mean vector, shape (state_dim,).
        cov: State covariance matrix, shape (state_dim, state_dim).
    """

    mean: Array  # (n,)
    cov: Array  # (n, n)

    @property
    def dim(self) -> int:
        """State dimension, inferred from mean vector."""
        return int(self.mean.shape[-1])

    def tree_flatten(self) -> tuple[tuple[Array, Array], None]:
        """Flatten into JAX pytree leaves."""
        return (self.mean, self.cov), None

    @classmethod
    def tree_unflatten(
        cls, aux_data: None, children: tuple[Array, Array]
    ) -> GaussianState:
        """Reconstruct from JAX pytree leaves."""
        return cls(mean=children[0], cov=children[1])


jax.tree_util.register_pytree_node_class(GaussianState)
