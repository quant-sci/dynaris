"""Bayesian model comparison: WAIC and LOO-CV via arviz.

Requires the ``bayesian`` extra: ``pip install dynaris[bayesian]``
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from dynaris.core.state_space import StateSpaceModel
from dynaris.core.types import GaussianState
from dynaris.estimation.bayesian import BayesianResult
from dynaris.filters.kalman import predict, update

ModelFactory = Any  # Callable[[Array], StateSpaceModel]


def _require_arviz() -> Any:
    try:
        import arviz

        return arviz
    except ImportError:
        msg = "Model comparison requires arviz. Install with: pip install dynaris[bayesian]"
        raise ImportError(msg) from None


@jax.jit
def _pointwise_log_likelihood(model: StateSpaceModel, observations: Array) -> Array:
    """Run Kalman filter and return per-observation log-likelihood contributions.

    Args:
        model: Linear-Gaussian state-space model.
        observations: Observation sequence, shape (T, obs_dim).

    Returns:
        Per-step log-likelihoods, shape (T,).
    """
    initial_state = model.initial_state()

    def _scan_step(state: GaussianState, obs: Array) -> tuple[GaussianState, Array]:
        predicted = predict(state, model)
        filtered, ll = update(predicted, obs, model)
        return filtered, ll

    _, pointwise_lls = jax.lax.scan(_scan_step, initial_state, observations)
    return pointwise_lls


def compute_waic(
    result: BayesianResult,
    model_fn: ModelFactory,
    observations: Array,
) -> dict[str, float]:
    """Compute WAIC (Widely Applicable Information Criterion).

    Args:
        result: BayesianResult from :func:`fit_bayesian`.
        model_fn: Same model factory used for fitting.
        observations: Same observations used for fitting.

    Returns:
        dict with keys ``"elpd_waic"``, ``"p_waic"``, ``"se"``.
    """
    observations = jnp.asarray(observations)

    def _pw_ll(params: Array) -> Array:
        model = model_fn(params)
        return _pointwise_log_likelihood(model, observations)  # type: ignore[no-any-return]

    pw_lls = jax.vmap(_pw_ll)(result.samples)  # (n_samples, T)

    # WAIC computed directly from pointwise log-likelihoods
    # elpd_waic = sum_i [log(mean_s(exp(ll_si))) - var_s(ll_si)]
    pw_lls_jax = pw_lls  # (n_samples, T)
    lppd_i = jax.nn.logsumexp(pw_lls_jax, axis=0) - jnp.log(jnp.float32(pw_lls_jax.shape[0]))
    p_waic_i = jnp.var(pw_lls_jax, axis=0)
    elpd_i = lppd_i - p_waic_i

    return {
        "elpd_waic": float(jnp.sum(elpd_i)),
        "p_waic": float(jnp.sum(p_waic_i)),
        "se": float(jnp.sqrt(pw_lls_jax.shape[1] * jnp.var(elpd_i))),
    }


def compute_loo(
    result: BayesianResult,
    model_fn: ModelFactory,
    observations: Array,
) -> dict[str, Any]:
    """Compute LOO-CV via Pareto-smoothed importance sampling.

    Args:
        result: BayesianResult from :func:`fit_bayesian`.
        model_fn: Same model factory used for fitting.
        observations: Same observations used for fitting.

    Returns:
        dict with keys ``"elpd_loo"``, ``"p_loo"``, ``"se"``.
    """
    arviz = _require_arviz()
    observations = jnp.asarray(observations)

    def _pw_ll(params: Array) -> Array:
        model = model_fn(params)
        return _pointwise_log_likelihood(model, observations)  # type: ignore[no-any-return]

    pw_lls = jax.vmap(_pw_ll)(result.samples)  # (n_samples, T)
    pw_lls_np = np.asarray(pw_lls)[np.newaxis, :, :]  # (1, n_samples, T)

    # arviz.loo needs a posterior group for r_eff computation
    samples_np = np.asarray(result.samples)
    n_params = samples_np.shape[1]
    names = result.param_names or tuple(f"param_{i}" for i in range(n_params))
    posterior_dict = {name: samples_np[np.newaxis, :, i] for i, name in enumerate(names)}

    idata = arviz.from_dict(
        {
            "posterior": posterior_dict,
            "log_likelihood": {"obs": pw_lls_np},
        }
    )
    loo_result = arviz.loo(idata)

    return {
        "elpd_loo": float(loo_result.elpd),
        "p_loo": float(loo_result.p),
        "se": float(loo_result.se),
    }


def to_arviz(
    result: BayesianResult,
    model_fn: ModelFactory,
    observations: Array,
) -> Any:
    """Convert a BayesianResult to an arviz InferenceData object.

    Includes posterior samples and pointwise log-likelihoods for
    further analysis with arviz's plotting and diagnostic tools.

    Args:
        result: BayesianResult from :func:`fit_bayesian`.
        model_fn: Same model factory used for fitting.
        observations: Same observations used for fitting.

    Returns:
        arviz.InferenceData with posterior and log_likelihood groups.
    """
    arviz = _require_arviz()
    observations = jnp.asarray(observations)

    def _pw_ll(params: Array) -> Array:
        model = model_fn(params)
        return _pointwise_log_likelihood(model, observations)  # type: ignore[no-any-return]

    pw_lls = jax.vmap(_pw_ll)(result.samples)

    samples_np = np.asarray(result.samples)
    pw_lls_np = np.asarray(pw_lls)

    # Build posterior dict
    n_params = samples_np.shape[1]
    names = result.param_names or tuple(f"param_{i}" for i in range(n_params))
    posterior_dict = {name: samples_np[np.newaxis, :, i] for i, name in enumerate(names)}

    return arviz.from_dict(
        {
            "posterior": posterior_dict,
            "log_likelihood": {"obs": pw_lls_np[np.newaxis, :, :]},
        }
    )
