"""Tests for dynaris.core.results — FilterResult and SmootherResult."""

import jax
import jax.numpy as jnp
from jax import tree_util

from dynaris.core.results import FilterResult, SmootherResult


def _make_filter_result(t_len: int = 10, n: int = 2, m: int = 1) -> FilterResult:
    return FilterResult(
        filtered_states=jnp.zeros((t_len, n)),
        filtered_covariances=jnp.zeros((t_len, n, n)),
        predicted_states=jnp.zeros((t_len, n)),
        predicted_covariances=jnp.zeros((t_len, n, n)),
        log_likelihood=jnp.array(0.0),
        observations=jnp.zeros((t_len, m)),
    )


def _make_smoother_result(t_len: int = 10, n: int = 2, m: int = 1) -> SmootherResult:
    return SmootherResult(
        smoothed_states=jnp.ones((t_len, n)),
        smoothed_covariances=jnp.ones((t_len, n, n)),
        filtered_states=jnp.zeros((t_len, n)),
        filtered_covariances=jnp.zeros((t_len, n, n)),
        predicted_states=jnp.zeros((t_len, n)),
        predicted_covariances=jnp.zeros((t_len, n, n)),
        log_likelihood=jnp.array(-5.0),
        observations=jnp.zeros((t_len, m)),
    )


def test_filter_result_fields() -> None:
    fr = _make_filter_result(t_len=10, n=3, m=2)
    assert fr.filtered_states.shape == (10, 3)
    assert fr.predicted_covariances.shape == (10, 3, 3)
    assert fr.observations.shape == (10, 2)
    assert fr.log_likelihood.shape == ()


def test_filter_result_pytree_roundtrip() -> None:
    fr = _make_filter_result()
    leaves, treedef = tree_util.tree_flatten(fr)
    assert len(leaves) == 6
    fr2 = tree_util.tree_unflatten(treedef, leaves)
    assert fr2.filtered_states.shape == fr.filtered_states.shape


def test_filter_result_jit() -> None:
    fr = _make_filter_result()

    @jax.jit
    def get_ll(r: FilterResult) -> jax.Array:
        return r.log_likelihood

    assert float(get_ll(fr)) == 0.0


def test_smoother_result_fields() -> None:
    sr = _make_smoother_result(t_len=8, n=4, m=2)
    assert sr.smoothed_states.shape == (8, 4)
    assert sr.smoothed_covariances.shape == (8, 4, 4)
    assert sr.filtered_states.shape == (8, 4)
    assert float(sr.log_likelihood) == -5.0


def test_smoother_result_pytree_roundtrip() -> None:
    sr = _make_smoother_result()
    leaves, treedef = tree_util.tree_flatten(sr)
    assert len(leaves) == 8
    sr2 = tree_util.tree_unflatten(treedef, leaves)
    assert float(sr2.smoothed_states[0, 0]) == 1.0
