"""Core math engine: state-space representation, filter protocols, result types."""

from dynaris.core.nonlinear import NonlinearSSM
from dynaris.core.protocols import FilterProtocol, SmootherProtocol
from dynaris.core.results import (
    FilterResult,
    SmootherResult,
    SwitchingFilterResult,
    SwitchingSmootherResult,
)
from dynaris.core.ssm import SSM
from dynaris.core.state_space import StateSpaceModel
from dynaris.core.switching import MarkovSwitchingSSM
from dynaris.core.types import GaussianState

__all__ = [
    "SSM",
    "FilterProtocol",
    "FilterResult",
    "GaussianState",
    "MarkovSwitchingSSM",
    "NonlinearSSM",
    "SmootherProtocol",
    "SmootherResult",
    "StateSpaceModel",
    "SwitchingFilterResult",
    "SwitchingSmootherResult",
]
