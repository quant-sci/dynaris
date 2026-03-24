"""Tests for dynaris.core.types — GaussianState."""

import jax
import jax.numpy as jnp
from jax import tree_util

from dynaris.core.types import GaussianState


def test_gaussian_state_construction() -> None:
    gs = GaussianState(mean=jnp.zeros(3), cov=jnp.eye(3))
    assert gs.mean.shape == (3,)
    assert gs.cov.shape == (3, 3)
    assert gs.dim == 3


def test_pytree_roundtrip() -> None:
    gs = GaussianState(mean=jnp.ones(4), cov=jnp.eye(4) * 2.0)
    leaves, treedef = tree_util.tree_flatten(gs)
    assert len(leaves) == 2
    gs2 = tree_util.tree_unflatten(treedef, leaves)
    assert gs2.dim == 4
    assert float(gs2.mean[0]) == 1.0
    assert float(gs2.cov[0, 0]) == 2.0


def test_jit_compatibility() -> None:
    gs = GaussianState(mean=jnp.zeros(3), cov=jnp.eye(3))

    @jax.jit
    def get_trace(s: GaussianState) -> jax.Array:
        return jnp.trace(s.cov)

    assert float(get_trace(gs)) == 3.0


def test_tree_map() -> None:
    gs = GaussianState(mean=jnp.ones(2), cov=jnp.eye(2))
    gs2 = tree_util.tree_map(lambda x: x * 3.0, gs)
    assert float(gs2.mean[0]) == 3.0
    assert float(gs2.cov[0, 0]) == 3.0


def test_in_lax_scan() -> None:
    def step(carry: GaussianState, _: None) -> tuple[GaussianState, jax.Array]:
        new = GaussianState(mean=carry.mean + 1.0, cov=carry.cov)
        return new, carry.mean

    init = GaussianState(mean=jnp.zeros(2), cov=jnp.eye(2))
    final, history = jax.lax.scan(step, init, None, length=5)
    assert final.mean.shape == (2,)
    assert float(final.mean[0]) == 5.0
    assert history.shape == (5, 2)
