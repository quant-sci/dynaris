"""Transforms for constraining parameters to valid ranges."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def softplus(x: Array) -> Array:
    """Softplus transform: log(1 + exp(x)). Maps R -> R+."""
    return jnp.logaddexp(x, 0.0)


def inverse_softplus(y: Array) -> Array:
    """Inverse softplus: log(exp(y) - 1). Maps R+ -> R."""
    return jnp.log(jnp.expm1(y))
