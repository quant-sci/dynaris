"""Maximum Likelihood Estimation for state-space models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy.optimize import minimize

from dynaris.core.results import FilterResult
from dynaris.core.state_space import StateSpaceModel
from dynaris.filters.kalman import kalman_filter

# Type alias for model factory
ModelFactory = Any  # Callable[[Array], StateSpaceModel]


@dataclass(frozen=True)
class MLEResult:
    """Result of maximum likelihood estimation.

    Attributes:
        model: Fitted StateSpaceModel at optimal parameters.
        log_likelihood: Maximized log-likelihood value.
        params: Optimized (unconstrained) parameter vector.
        n_iterations: Number of optimizer iterations.
        converged: Whether the optimizer converged.
        filter_result: FilterResult from the fitted model.
    """

    model: StateSpaceModel
    log_likelihood: float
    params: Array
    n_iterations: int
    converged: bool
    filter_result: FilterResult


def fit_mle(
    model_fn: ModelFactory,
    observations: Array,
    init_params: Array,
    method: str = "L-BFGS-B",
    max_iter: int = 500,
    tol: float = 1e-6,
) -> MLEResult:
    """Fit a state-space model via maximum likelihood estimation.

    Uses ``scipy.optimize.minimize`` with JAX-computed gradients.

    Args:
        model_fn: A callable mapping an unconstrained parameter vector
            to a :class:`StateSpaceModel`. Use log/softplus transforms
            inside this function to ensure positivity of variances.
        observations: Observation sequence, shape (T, obs_dim).
        init_params: Initial (unconstrained) parameter vector.
        method: Scipy optimizer method. Default ``"L-BFGS-B"``.
        max_iter: Maximum number of optimizer iterations.
        tol: Convergence tolerance for the optimizer.

    Returns:
        MLEResult with the fitted model and optimization details.

    Example::

        import jax.numpy as jnp
        from dynaris import LocalLevel
        from dynaris.estimation import fit_mle

        def model_fn(params):
            return LocalLevel(
                sigma_level=jnp.exp(params[0]),
                sigma_obs=jnp.exp(params[1]),
            )

        result = fit_mle(model_fn, observations, jnp.zeros(2))
        fitted_model = result.model
    """
    observations = jnp.asarray(observations)

    @jax.jit
    def neg_ll(params: Array) -> Array:
        model = model_fn(params)
        fr = kalman_filter(model, observations)
        return -fr.log_likelihood  # type: ignore[no-any-return]

    grad_fn = jax.jit(jax.grad(neg_ll))

    def _objective(params_np: np.ndarray[Any, Any]) -> tuple[float, np.ndarray[Any, Any]]:
        params_jax = jnp.array(params_np)
        loss = float(neg_ll(params_jax))
        grad = np.array(grad_fn(params_jax), dtype=np.float64)
        return loss, grad

    result = minimize(
        _objective,
        x0=np.array(init_params, dtype=np.float64),
        method=method,
        jac=True,
        options={"maxiter": max_iter, "ftol": tol, "gtol": tol},
    )

    optimal_params = jnp.array(result.x)
    fitted_model = model_fn(optimal_params)
    fr = kalman_filter(fitted_model, observations)

    return MLEResult(
        model=fitted_model,
        log_likelihood=float(fr.log_likelihood),
        params=optimal_params,
        n_iterations=int(result.nit),
        converged=bool(result.success),
        filter_result=fr,
    )
