"""Tests for dynaris.core.state_space — StateSpaceModel."""

import jax
import jax.numpy as jnp
from jax import tree_util

from dynaris.core.state_space import StateSpaceModel


def _make_ssm(
    state_dim: int = 2, obs_dim: int = 1, with_B: bool = False
) -> StateSpaceModel:
    kwargs: dict = dict(
        transition_matrix=jnp.eye(state_dim),
        observation_matrix=jnp.ones((obs_dim, state_dim)),
        state_noise_cov=jnp.eye(state_dim) * 0.1,
        obs_noise_cov=jnp.eye(obs_dim),
    )
    if with_B:
        kwargs["input_matrix"] = jnp.ones((state_dim, 1))
    return StateSpaceModel(**kwargs)


def test_dimensions() -> None:
    ssm = _make_ssm(3, 2)
    assert ssm.state_dim == 3
    assert ssm.obs_dim == 2


def test_aliases() -> None:
    ssm = _make_ssm()
    assert ssm.F is ssm.transition_matrix
    assert ssm.H is ssm.observation_matrix
    assert ssm.Q is ssm.state_noise_cov
    assert ssm.R is ssm.obs_noise_cov
    assert ssm.B is ssm.input_matrix


def test_pytree_roundtrip_no_B() -> None:
    ssm = _make_ssm()
    leaves, treedef = tree_util.tree_flatten(ssm)
    assert len(leaves) == 4
    ssm2 = tree_util.tree_unflatten(treedef, leaves)
    assert ssm2.state_dim == ssm.state_dim
    assert ssm2.B is None


def test_pytree_roundtrip_with_B() -> None:
    ssm = _make_ssm(with_B=True)
    leaves, treedef = tree_util.tree_flatten(ssm)
    assert len(leaves) == 5
    ssm2 = tree_util.tree_unflatten(treedef, leaves)
    assert ssm2.B is not None
    assert ssm2.B.shape == (2, 1)


def test_jit_compatibility() -> None:
    ssm = _make_ssm(3, 1)

    @jax.jit
    def trace_F(m: StateSpaceModel) -> jax.Array:
        return jnp.trace(m.F)

    assert float(trace_F(ssm)) == 3.0


def test_add_block_diagonal() -> None:
    s1 = _make_ssm(2, 1)
    s2 = _make_ssm(3, 1)
    s3 = s1 + s2

    assert s3.state_dim == 5
    assert s3.obs_dim == 1
    assert s3.F.shape == (5, 5)
    assert s3.H.shape == (1, 5)
    assert s3.Q.shape == (5, 5)
    assert s3.R.shape == (1, 1)

    # F should be block-diagonal identity
    assert float(jnp.trace(s3.F)) == 5.0
    assert float(s3.F[0, 2]) == 0.0  # off-diagonal block is zero

    # H should be horizontal concatenation
    assert float(s3.H[0, 0]) == 1.0
    assert float(s3.H[0, 4]) == 1.0

    # R should be additive
    assert float(s3.R[0, 0]) == 2.0

    # No B
    assert s3.B is None


def test_add_both_have_B() -> None:
    s1 = _make_ssm(2, 1, with_B=True)
    s2 = _make_ssm(3, 1, with_B=True)
    s3 = s1 + s2
    assert s3.B is not None
    assert s3.B.shape == (5, 2)  # block-diagonal: (2,1) and (3,1) -> (5,2)


def test_add_one_has_B() -> None:
    s1 = _make_ssm(2, 1, with_B=True)
    s2 = _make_ssm(3, 1)
    s3 = s1 + s2
    assert s3.B is not None
    assert s3.B.shape == (5, 1)
    # Bottom block should be zeros
    assert float(s3.B[3, 0]) == 0.0

    # Reverse order
    s4 = s2 + s1
    assert s4.B is not None
    assert s4.B.shape == (5, 1)
    # Top block should be zeros
    assert float(s4.B[0, 0]) == 0.0


def test_add_under_jit() -> None:
    s1 = _make_ssm(2, 1)
    s2 = _make_ssm(3, 1)

    @jax.jit
    def compose_trace(a: StateSpaceModel, b: StateSpaceModel) -> jax.Array:
        return jnp.trace((a + b).F)

    assert float(compose_trace(s1, s2)) == 5.0


def test_initial_state_defaults() -> None:
    ssm = _make_ssm(3, 1)
    init = ssm.initial_state()
    assert init.mean.shape == (3,)
    assert init.cov.shape == (3, 3)
    assert float(init.mean[0]) == 0.0
    assert float(init.cov[0, 0]) == 1e6


def test_initial_state_custom() -> None:
    ssm = _make_ssm(2, 1)
    init = ssm.initial_state(mean=jnp.ones(2), cov=jnp.eye(2) * 0.5)
    assert float(init.mean[0]) == 1.0
    assert float(init.cov[0, 0]) == 0.5


def test_repr() -> None:
    ssm = _make_ssm(3, 2)
    r = repr(ssm)
    assert "state_dim=3" in r
    assert "obs_dim=2" in r
    assert "input_dim" not in r

    ssm_b = _make_ssm(3, 2, with_B=True)
    r_b = repr(ssm_b)
    assert "input_dim=1" in r_b
