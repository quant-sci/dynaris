"""Tests for dynaris.core.protocols — FilterProtocol and SmootherProtocol."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from dynaris.core.protocols import FilterProtocol, SmootherProtocol
from dynaris.core.results import FilterResult, SmootherResult
from dynaris.core.state_space import StateSpaceModel
from dynaris.core.types import GaussianState


class DummyFilter:
    """A stub that satisfies FilterProtocol."""

    def predict(
        self,
        state: GaussianState,
        model: StateSpaceModel,
        u: Array | None = None,
    ) -> GaussianState:
        return state

    def update(
        self,
        predicted: GaussianState,
        observation: Array,
        model: StateSpaceModel,
    ) -> GaussianState:
        return predicted

    def scan(
        self,
        model: StateSpaceModel,
        observations: Array,
        initial_state: GaussianState | None = None,
        inputs: Array | None = None,
    ) -> FilterResult:
        t_len = observations.shape[0]
        n = model.state_dim
        return FilterResult(
            filtered_states=jnp.zeros((t_len, n)),
            filtered_covariances=jnp.zeros((t_len, n, n)),
            predicted_states=jnp.zeros((t_len, n)),
            predicted_covariances=jnp.zeros((t_len, n, n)),
            log_likelihood=jnp.array(0.0),
            observations=observations,
        )


class DummySmoother:
    """A stub that satisfies SmootherProtocol."""

    def smooth(
        self,
        model: StateSpaceModel,
        filter_result: FilterResult,
    ) -> SmootherResult:
        return SmootherResult(
            smoothed_states=filter_result.filtered_states,
            smoothed_covariances=filter_result.filtered_covariances,
            filtered_states=filter_result.filtered_states,
            filtered_covariances=filter_result.filtered_covariances,
            predicted_states=filter_result.predicted_states,
            predicted_covariances=filter_result.predicted_covariances,
            log_likelihood=filter_result.log_likelihood,
            observations=filter_result.observations,
        )


class NotAFilter:
    """A class that does NOT satisfy FilterProtocol."""

    def predict(self) -> None:  # wrong signature
        pass


def test_filter_protocol_isinstance() -> None:
    f = DummyFilter()
    assert isinstance(f, FilterProtocol)


def test_smoother_protocol_isinstance() -> None:
    s = DummySmoother()
    assert isinstance(s, SmootherProtocol)


def test_non_conforming_class() -> None:
    nf = NotAFilter()
    assert not isinstance(nf, FilterProtocol)


def test_dummy_filter_scan() -> None:
    f = DummyFilter()
    ssm = StateSpaceModel(
        system_matrix=jnp.eye(2),
        observation_matrix=jnp.ones((1, 2)),
        evolution_cov=jnp.eye(2) * 0.1,
        obs_cov=jnp.array([[1.0]]),
    )
    obs = jnp.ones((5, 1))
    result = f.scan(ssm, obs)
    assert result.filtered_states.shape == (5, 2)
    assert float(result.log_likelihood) == 0.0
