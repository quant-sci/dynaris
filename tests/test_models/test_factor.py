"""Tests for Dynamic Factor Model factory and utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from dynaris.core.state_space import StateSpaceModel
from dynaris.filters.kalman import kalman_filter
from dynaris.models.factor import (
    DynamicFactorModel,
    apply_identification_constraints,
    initialize_loadings_pca,
    rotate_loadings,
)


def test_dfm_shapes_basic() -> None:
    model = DynamicFactorModel(n_factors=2, n_variables=5)
    assert model.state_dim == 2
    assert model.obs_dim == 5
    assert model.F.shape == (5, 2)
    assert model.G.shape == (2, 2)
    assert model.W.shape == (2, 2)
    assert model.V.shape == (5, 5)


def test_dfm_factor_order_1() -> None:
    model = DynamicFactorModel(n_factors=3, n_variables=10)
    np.testing.assert_allclose(model.G, jnp.eye(3), atol=1e-6)
    np.testing.assert_allclose(model.W, jnp.eye(3), atol=1e-6)


def test_dfm_factor_order_2() -> None:
    model = DynamicFactorModel(n_factors=2, n_variables=5, factor_order=2)
    assert model.state_dim == 4  # 2 * 2
    assert model.F.shape == (5, 4)


def test_dfm_custom_loadings() -> None:
    loadings = jnp.ones((5, 2)) * 0.5
    model = DynamicFactorModel(n_factors=2, n_variables=5, loadings=loadings)
    np.testing.assert_allclose(model.F, loadings, atol=1e-6)


def test_dfm_scalar_obs_noise() -> None:
    model = DynamicFactorModel(n_factors=2, n_variables=3, obs_noise=2.0)
    np.testing.assert_allclose(model.V, jnp.eye(3) * 2.0, atol=1e-6)


def test_dfm_vector_obs_noise() -> None:
    noise = jnp.array([1.0, 2.0, 3.0])
    model = DynamicFactorModel(n_factors=2, n_variables=3, obs_noise=noise)
    np.testing.assert_allclose(jnp.diag(model.V), noise, atol=1e-6)


def test_dfm_kalman_filter_runs() -> None:
    model = DynamicFactorModel(n_factors=2, n_variables=5)
    key = jax.random.PRNGKey(0)
    obs = jax.random.normal(key, (50, 5))
    result = kalman_filter(model, obs)
    assert jnp.all(jnp.isfinite(result.filtered_states))
    assert result.filtered_states.shape == (50, 2)


def test_dfm_pytree_roundtrip() -> None:
    model = DynamicFactorModel(n_factors=2, n_variables=5)
    leaves, treedef = jax.tree_util.tree_flatten(model)
    reconstructed = treedef.unflatten(leaves)
    assert isinstance(reconstructed, StateSpaceModel)
    assert reconstructed.state_dim == 2
    assert reconstructed.obs_dim == 5


# ---------------------------------------------------------------------------
# PCA initialization
# ---------------------------------------------------------------------------


def test_initialize_loadings_pca_shape() -> None:
    key = jax.random.PRNGKey(0)
    obs = jax.random.normal(key, (100, 8))
    loadings, factors = initialize_loadings_pca(obs, n_factors=3)
    assert loadings.shape == (8, 3)
    assert factors.shape == (100, 3)


def test_initialize_loadings_pca_finite() -> None:
    key = jax.random.PRNGKey(1)
    obs = jax.random.normal(key, (50, 5))
    loadings, factors = initialize_loadings_pca(obs, n_factors=2)
    assert jnp.all(jnp.isfinite(loadings))
    assert jnp.all(jnp.isfinite(factors))


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------


def test_rotate_loadings_preserves_communalities() -> None:
    """Varimax rotation should preserve Lambda @ Lambda' (communalities)."""
    key = jax.random.PRNGKey(2)
    loadings = jax.random.normal(key, (6, 2))
    rotated, _ = rotate_loadings(loadings, method="varimax")

    original_comm = loadings @ loadings.T
    rotated_comm = rotated @ rotated.T
    np.testing.assert_allclose(rotated_comm, original_comm, atol=1e-4)


def test_rotate_loadings_shape() -> None:
    loadings = jnp.ones((5, 3))
    rotated, rotation_matrix = rotate_loadings(loadings)
    assert rotated.shape == (5, 3)
    assert rotation_matrix.shape == (3, 3)


# ---------------------------------------------------------------------------
# Identification constraints
# ---------------------------------------------------------------------------


def test_identification_constraints_lower_triangular() -> None:
    loadings = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    constrained = apply_identification_constraints(loadings)
    # First 2x2 block should be lower-triangular
    assert float(constrained[0, 1]) == 0.0
    # Diagonal should be positive
    assert float(constrained[0, 0]) > 0.0
    assert float(constrained[1, 1]) > 0.0
    # Remaining rows unchanged (up to sign flip)
    assert constrained.shape == (3, 2)


def test_identification_constraints_positive_diagonal() -> None:
    loadings = jnp.array([[-1.0, 0.0], [2.0, -3.0]])
    constrained = apply_identification_constraints(loadings)
    assert float(constrained[0, 0]) > 0.0
    assert float(constrained[1, 1]) > 0.0
