"""Bayesian estimation for state-space models via MCMC.

Uses NumPyro's NUTS sampler to draw posterior samples of model parameters,
with the Kalman filter log-likelihood as the data term.

Requires the ``bayesian`` extra: ``pip install dynaris[bayesian]``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from dynaris.core.results import FilterResult
from dynaris.core.state_space import StateSpaceModel
from dynaris.estimation.priors import LogPriorFn
from dynaris.filters.kalman import kalman_filter

ModelFactory = Any  # Callable[[Array], StateSpaceModel]


def _require_numpyro() -> Any:
    try:
        import numpyro

        return numpyro
    except ImportError:
        msg = "Bayesian estimation requires numpyro. Install with: pip install dynaris[bayesian]"
        raise ImportError(msg) from None


@dataclass(frozen=True)
class BayesianResult:
    """Result of Bayesian MCMC estimation.

    Attributes:
        samples: Posterior samples (unconstrained), shape (n_samples, n_params).
        log_likelihood_samples: Log-likelihood at each sample, shape (n_samples,).
        model: StateSpaceModel at the posterior mean parameters.
        filter_result: FilterResult from the posterior-mean model.
        param_names: Optional parameter labels.
        info: Sampler diagnostics (divergences, acceptance rate, etc.).
    """

    samples: Array
    log_likelihood_samples: Array
    model: StateSpaceModel
    filter_result: FilterResult
    param_names: tuple[str, ...] | None = None
    info: dict[str, Any] | None = None


def _flat_prior(params: Array) -> Array:
    """Flat (improper) prior: constant zero log-density."""
    return jnp.array(0.0)


def fit_bayesian(
    model_fn: ModelFactory,
    observations: Array,
    init_params: Array,
    log_prior_fn: LogPriorFn | None = None,
    n_warmup: int = 500,
    n_samples: int = 1000,
    key: Array | None = None,
    param_names: tuple[str, ...] | None = None,
) -> BayesianResult:
    """Fit a state-space model via Bayesian MCMC (NUTS).

    Uses NumPyro's NUTS sampler with automatic warmup adaptation.
    The log-posterior is the Kalman filter log-likelihood plus the
    log-prior.

    Args:
        model_fn: Maps unconstrained parameter vector to a
            :class:`StateSpaceModel`. Same pattern as :func:`fit_mle`.
        observations: Observation sequence, shape (T, obs_dim).
        init_params: Initial (unconstrained) parameter vector.
        log_prior_fn: Log-prior function. Defaults to flat prior.
        n_warmup: Number of NUTS warmup steps.
        n_samples: Number of posterior samples to draw.
        key: JAX PRNG key. Defaults to ``PRNGKey(0)``.
        param_names: Optional names for each parameter dimension.

    Returns:
        BayesianResult with posterior samples and fitted model.

    Example::

        import jax.numpy as jnp
        from dynaris import LocalLevel
        from dynaris.estimation import fit_bayesian
        from dynaris.estimation.priors import inverse_gamma_log_prior

        def model_fn(params):
            return LocalLevel(
                sigma_level=jnp.exp(params[0]),
                sigma_obs=jnp.exp(params[1]),
            )

        result = fit_bayesian(
            model_fn, observations, jnp.zeros(2),
            log_prior_fn=inverse_gamma_log_prior(shape=2.0, scale=1.0),
        )
    """
    numpyro = _require_numpyro()
    from numpyro.infer import MCMC, NUTS

    observations = jnp.asarray(observations)
    init_params = jnp.asarray(init_params)
    if key is None:
        key = jax.random.PRNGKey(0)
    if log_prior_fn is None:
        log_prior_fn = _flat_prior

    n_params = init_params.shape[0]

    # Define the log-density for the sampler
    @jax.jit
    def _log_density(params: Array) -> Array:
        model = model_fn(params)
        fr = kalman_filter(model, observations)
        return fr.log_likelihood + log_prior_fn(params)

    # NumPyro model: sample unconstrained params, factor by log-density
    def _numpyro_model() -> None:
        params = numpyro.sample(
            "params",
            numpyro.distributions.Normal(0.0, 100.0).expand([n_params]).to_event(1),
        )
        log_density = _log_density(params)
        numpyro.factor("log_density", log_density)

    # Run MCMC
    kernel = NUTS(
        _numpyro_model, init_strategy=numpyro.infer.init_to_value(values={"params": init_params})
    )
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples, progress_bar=False)
    mcmc.run(key)

    samples = mcmc.get_samples()["params"]  # (n_samples, n_params)

    # Compute per-sample log-likelihoods
    @jax.jit
    def _compute_ll(params: Array) -> Array:
        model = model_fn(params)
        return kalman_filter(model, observations).log_likelihood

    log_lls = jax.vmap(_compute_ll)(samples)

    # Build posterior-mean model
    mean_params = jnp.mean(samples, axis=0)
    mean_model = model_fn(mean_params)
    mean_fr = kalman_filter(mean_model, observations)

    # Diagnostics
    info: dict[str, Any] = {}
    extra_fields = mcmc.get_extra_fields()
    if "diverging" in extra_fields:
        info["n_divergences"] = int(np.sum(np.asarray(extra_fields["diverging"])))
    if "accept_prob" in extra_fields:
        info["mean_accept_prob"] = float(np.mean(np.asarray(extra_fields["accept_prob"])))

    return BayesianResult(
        samples=samples,
        log_likelihood_samples=log_lls,
        model=mean_model,
        filter_result=mean_fr,
        param_names=param_names,
        info=info,
    )
