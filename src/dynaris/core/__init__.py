"""Core math engine: state-space representation, filter protocols, result types."""

from dynaris.core.protocols import FilterProtocol, SmootherProtocol
from dynaris.core.results import FilterResult, SmootherResult
from dynaris.core.state_space import StateSpaceModel
from dynaris.core.types import GaussianState

__all__ = [
    "FilterProtocol",
    "FilterResult",
    "GaussianState",
    "SmootherProtocol",
    "SmootherResult",
    "StateSpaceModel",
]
