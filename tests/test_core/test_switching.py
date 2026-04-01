"""Tests for the MarkovSwitchingSSM model type."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dynaris.core.switching import MarkovSwitchingSSM
from dynaris.dlm.components import LocalLevel, LocalLinearTrend


def _two_regime_model() -> MarkovSwitchingSSM:
    """Simple 2-regime local-level model."""
    return MarkovSwitchingSSM(
        models=(
            LocalLevel(sigma_level=1.0, sigma_obs=5.0),
            LocalLevel(sigma_level=5.0, sigma_obs=20.0),
        ),
        transition_matrix=jnp.array([[0.95, 0.05], [0.10, 0.90]]),
        initial_probs=jnp.array([0.5, 0.5]),
    )


def test_switching_model_creation() -> None:
    model = _two_regime_model()
    assert model.n_regimes == 2
    assert model.state_dim == 1
    assert model.obs_dim == 1


def test_switching_model_properties() -> None:
    model = _two_regime_model()
    assert model.G_stack.shape == (2, 1, 1)
    assert model.F_stack.shape == (2, 1, 1)
    assert model.W_stack.shape == (2, 1, 1)
    assert model.V_stack.shape == (2, 1, 1)


def test_switching_model_stacked_values() -> None:
    model = _two_regime_model()
    # Regime 0: sigma_level=1 -> W = [[1]]
    np.testing.assert_allclose(model.W_stack[0], [[1.0]], atol=1e-6)
    # Regime 1: sigma_level=5 -> W = [[25]]
    np.testing.assert_allclose(model.W_stack[1], [[25.0]], atol=1e-6)


def test_switching_model_dim_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same dimensions"):
        MarkovSwitchingSSM(
            models=(LocalLevel(), LocalLinearTrend()),
            transition_matrix=jnp.eye(2),
            initial_probs=jnp.array([0.5, 0.5]),
        )


def test_switching_model_trans_matrix_shape_raises() -> None:
    with pytest.raises(ValueError, match="transition_matrix"):
        MarkovSwitchingSSM(
            models=(LocalLevel(), LocalLevel()),
            transition_matrix=jnp.eye(3),
            initial_probs=jnp.array([0.5, 0.5]),
        )


def test_switching_model_initial_probs_shape_raises() -> None:
    with pytest.raises(ValueError, match="initial_probs"):
        MarkovSwitchingSSM(
            models=(LocalLevel(), LocalLevel()),
            transition_matrix=jnp.eye(2),
            initial_probs=jnp.array([0.5, 0.3, 0.2]),
        )


def test_switching_model_initial_state() -> None:
    model = _two_regime_model()
    state = model.initial_state()
    assert state.mean.shape == (1,)
    assert state.cov.shape == (1, 1)


def test_switching_model_repr() -> None:
    model = _two_regime_model()
    r = repr(model)
    assert "n_regimes=2" in r
    assert "state_dim=1" in r


def test_switching_model_pytree_roundtrip() -> None:
    model = _two_regime_model()
    leaves, treedef = jax.tree_util.tree_flatten(model)
    reconstructed = treedef.unflatten(leaves)
    assert reconstructed.n_regimes == model.n_regimes
    assert reconstructed.state_dim == model.state_dim
    np.testing.assert_allclose(reconstructed.transition_matrix, model.transition_matrix)
    np.testing.assert_allclose(reconstructed.G_stack, model.G_stack)


def test_three_regime_model() -> None:
    model = MarkovSwitchingSSM(
        models=(LocalLevel(1, 1), LocalLevel(3, 3), LocalLevel(5, 5)),
        transition_matrix=jnp.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]),
        initial_probs=jnp.array([1 / 3, 1 / 3, 1 / 3]),
    )
    assert model.n_regimes == 3
    assert model.G_stack.shape == (3, 1, 1)
