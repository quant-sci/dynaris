"""Protocol definitions for filter and smoother interfaces."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from jax import Array

from dynaris.core.results import FilterResult, SmootherResult
from dynaris.core.state_space import StateSpaceModel
from dynaris.core.types import GaussianState


@runtime_checkable
class FilterProtocol(Protocol):
    """Interface that all filtering algorithms must implement."""

    def predict(
        self,
        state: GaussianState,
        model: StateSpaceModel,
        u: Array | None = None,
    ) -> GaussianState:
        """Predict the next state (time update / prior).

        Args:
            state: Current filtered state belief.
            model: The state-space model defining dynamics.
            u: Optional control input vector, shape (input_dim,).

        Returns:
            Predicted (prior) GaussianState for the next time step.
        """
        ...

    def update(
        self,
        predicted: GaussianState,
        observation: Array,
        model: StateSpaceModel,
    ) -> GaussianState:
        """Incorporate an observation (measurement update / posterior).

        Args:
            predicted: Predicted state from the predict step.
            observation: Observation vector, shape (obs_dim,).
            model: The state-space model defining the observation equation.

        Returns:
            Filtered (posterior) GaussianState.
        """
        ...

    def scan(
        self,
        model: StateSpaceModel,
        observations: Array,
        initial_state: GaussianState | None = None,
        inputs: Array | None = None,
    ) -> FilterResult:
        """Run the full forward filtering pass over a sequence.

        Args:
            model: The state-space model.
            observations: Observation sequence, shape (T, obs_dim).
            initial_state: Initial state belief. If None, uses model defaults.
            inputs: Optional control inputs, shape (T, input_dim).

        Returns:
            FilterResult containing all filtered and predicted states.
        """
        ...


@runtime_checkable
class SmootherProtocol(Protocol):
    """Interface for backward smoothing algorithms."""

    def smooth(
        self,
        model: StateSpaceModel,
        filter_result: FilterResult,
    ) -> SmootherResult:
        """Run backward smoothing given forward filter results.

        Args:
            model: The state-space model.
            filter_result: Output from a forward filtering pass.

        Returns:
            SmootherResult with smoothed state estimates.
        """
        ...
