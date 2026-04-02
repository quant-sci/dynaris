"""Parallel MCMC chains for Bayesian estimation.

Runs multiple independent NUTS chains in parallel using NumPyro's
built-in ``chain_method="parallel"`` (backed by ``jax.vmap``).

Requires the ``bayesian`` extra: ``pip install dynaris[bayesian]``
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from dynaris.estimation.bayesian import BayesianResult, _require_numpyro
from dynaris.estimation.priors import LogPriorFn
from dynaris.filters.kalman import kalman_filter

ModelFactory = Any


def _flat_prior(_params: Array) -> Array:
    return jnp.array(0.0)


def fit_bayesian_parallel(
    model_fn: ModelFactory,
    observations: Array,
    init_params: Array,
    log_prior_fn: LogPriorFn | None = None,
    n_warmup: int = 500,
    n_samples: int = 1000,
    n_chains: int = 4,
    key: Array | None = None,
    param_names: tuple[str, ...] | None = None,
) -> BayesianResult:
    """Fit a state-space model via parallel MCMC chains (NUTS).

    Runs ``n_chains`` independent NUTS chains in parallel using
    ``jax.vmap``, providing better posterior exploration and
    convergence diagnostics.

    Args:
        model_fn: Maps unconstrained parameter vector to a
            :class:`StateSpaceModel`.
        observations: Observation sequence, shape (T, obs_dim).
        init_params: Initial (unconstrained) parameter vector.
        log_prior_fn: Log-prior function. Defaults to flat prior.
        n_warmup: Number of NUTS warmup steps per chain.
        n_samples: Number of posterior samples per chain.
        n_chains: Number of parallel chains.
        key: JAX PRNG key. Defaults to ``PRNGKey(0)``.
        param_names: Optional names for each parameter dimension.

    Returns:
        BayesianResult with concatenated samples from all chains.
        The ``info`` dict includes ``"n_chains"`` and per-chain
        diagnostics.

    Example::

        from dynaris.estimation.parallel_mcmc import fit_bayesian_parallel

        result = fit_bayesian_parallel(
            model_fn, observations, init_params,
            n_chains=4, n_samples=500,
        )
        print(result.samples.shape)  # (2000, n_params) = 4 chains * 500
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

    @jax.jit
    def _log_density(params: Array) -> Array:
        model = model_fn(params)
        fr = kalman_filter(model, observations)
        return fr.log_likelihood + log_prior_fn(params)  # type: ignore[no-any-return]

    def _numpyro_model() -> None:
        params = numpyro.sample(
            "params",
            numpyro.distributions.Normal(0.0, 100.0).expand([n_params]).to_event(1),
        )
        log_density = _log_density(params)
        numpyro.factor("log_density", log_density)

    kernel = NUTS(
        _numpyro_model,
        init_strategy=numpyro.infer.init_to_value(values={"params": init_params}),
    )
    mcmc = MCMC(
        kernel,
        num_warmup=n_warmup,
        num_samples=n_samples,
        num_chains=n_chains,
        chain_method="parallel",
        progress_bar=False,
    )
    mcmc.run(key)

    # Concatenate all chains: (n_chains, n_samples, n_params) -> (total, n_params)
    all_samples = mcmc.get_samples()["params"]  # already concatenated by NumPyro
    if all_samples.ndim == 3:
        all_samples = all_samples.reshape(-1, n_params)

    # Log-likelihoods
    @jax.jit
    def _compute_ll(params: Array) -> Array:
        model = model_fn(params)
        return kalman_filter(model, observations).log_likelihood  # type: ignore[no-any-return]

    log_lls = jax.vmap(_compute_ll)(all_samples)

    # Posterior mean model
    mean_params = jnp.mean(all_samples, axis=0)
    mean_model = model_fn(mean_params)
    mean_fr = kalman_filter(mean_model, observations)

    # Diagnostics
    info: dict[str, Any] = {"n_chains": n_chains}
    extra_fields = mcmc.get_extra_fields()
    if "diverging" in extra_fields:
        info["n_divergences"] = int(np.sum(np.asarray(extra_fields["diverging"])))
    if "accept_prob" in extra_fields:
        info["mean_accept_prob"] = float(np.mean(np.asarray(extra_fields["accept_prob"])))

    return BayesianResult(
        samples=all_samples,
        log_likelihood_samples=log_lls,
        model=mean_model,
        filter_result=mean_fr,
        param_names=param_names,
        info=info,
    )
