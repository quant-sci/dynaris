"""Built-in log-prior factories for Bayesian estimation.

Each factory returns a ``Callable[[Array], Array]`` that computes the
log-prior density of an unconstrained parameter vector. All functions
are JIT- and autodiff-compatible.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from jax import Array

LogPriorFn = Callable[[Array], Array]


def normal_log_prior(loc: float = 0.0, scale: float = 10.0) -> LogPriorFn:
    """Normal log-prior for unconstrained parameters.

    Computes ``sum log N(params_i; loc, scale)`` over all elements.

    Args:
        loc: Prior mean.
        scale: Prior standard deviation.

    Returns:
        A function ``(params) -> scalar log-density``.
    """

    @jax.jit
    def _log_prior(params: Array) -> Array:
        n = params.shape[0]
        return (
            -0.5 * jnp.sum(((params - loc) / scale) ** 2)
            - n * jnp.log(scale)
            - 0.5 * n * jnp.log(2.0 * jnp.pi)
        )

    return _log_prior


def inverse_gamma_log_prior(
    shape: float = 2.0,
    scale: float = 1.0,
    indices: Sequence[int] | None = None,
) -> LogPriorFn:
    r"""Inverse-Gamma log-prior for variance parameters in log-space.

    If ``x = exp(p)`` and ``x ~ InverseGamma(shape, scale)``, then
    the log-prior on the unconstrained parameter ``p`` is:

    .. math::

        \log IG(\exp(p) \mid a, b) + p
        = a \log b - \log \Gamma(a) - a p - b \exp(-p)

    The ``+p`` term is the log-Jacobian of the ``exp`` transform.

    Args:
        shape: Shape parameter ``a`` (must be > 0).
        scale: Scale parameter ``b`` (must be > 0).
        indices: Which parameter indices to apply the prior to.
            If ``None``, applies to all parameters.

    Returns:
        A function ``(params) -> scalar log-density``.
    """
    a, b = shape, scale
    log_norm = a * jnp.log(b) - jax.lax.lgamma(jnp.array(a))

    if indices is not None:
        idx = jnp.array(indices)

        @jax.jit
        def _log_prior(params: Array) -> Array:
            p = params[idx]
            return jnp.sum(log_norm - a * p - b * jnp.exp(-p))

    else:

        @jax.jit
        def _log_prior(params: Array) -> Array:
            n = params.shape[0]
            return jnp.sum(n * log_norm - a * params - b * jnp.exp(-params))

    return _log_prior


def half_normal_log_prior(
    scale: float = 1.0,
    indices: Sequence[int] | None = None,
) -> LogPriorFn:
    r"""Half-Normal log-prior for standard deviation parameters in log-space.

    If ``sigma = exp(p)`` and ``sigma ~ HalfNormal(scale)``, the
    log-prior on ``p`` includes the Jacobian correction:

    .. math::

        \log \text{HalfNormal}(\exp(p) \mid s) + p
        = \log 2 - \log s - \frac{1}{2} \log(2\pi)
          - \frac{\exp(2p)}{2 s^2} + p

    Args:
        scale: Scale parameter of the half-normal.
        indices: Which parameter indices to apply the prior to.
            If ``None``, applies to all parameters.

    Returns:
        A function ``(params) -> scalar log-density``.
    """
    s = scale
    log_const = jnp.log(2.0) - jnp.log(s) - 0.5 * jnp.log(2.0 * jnp.pi)

    if indices is not None:
        idx = jnp.array(indices)

        @jax.jit
        def _log_prior(params: Array) -> Array:
            p = params[idx]
            return jnp.sum(log_const - jnp.exp(2.0 * p) / (2.0 * s**2) + p)

    else:

        @jax.jit
        def _log_prior(params: Array) -> Array:
            return jnp.sum(log_const - jnp.exp(2.0 * params) / (2.0 * s**2) + params)

    return _log_prior


def combine_priors(*prior_fns: LogPriorFn) -> LogPriorFn:
    """Combine multiple log-prior functions by summing.

    Args:
        *prior_fns: Log-prior functions to combine.

    Returns:
        A function that returns the sum of all log-priors.
    """

    @jax.jit
    def _combined(params: Array) -> Array:
        total = jnp.array(0.0)
        for fn in prior_fns:
            total = total + fn(params)
        return total

    return _combined
