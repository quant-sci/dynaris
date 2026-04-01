"""Tests for Bayesian model comparison (WAIC, LOO)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

numpyro = __import__("pytest").importorskip("numpyro")
arviz = __import__("pytest").importorskip("arviz")

from dynaris.datasets import load_nile_jax  # noqa: E402
from dynaris.estimation.bayesian import fit_bayesian  # noqa: E402
from dynaris.estimation.comparison import compute_loo, compute_waic, to_arviz  # noqa: E402

NILE = load_nile_jax()[:30]


def _model_fn(params: jax.Array) -> object:
    from dynaris.dlm.components import LocalLevel

    return LocalLevel(sigma_level=jnp.exp(params[0]), sigma_obs=jnp.exp(params[1]))


def _get_result():
    return fit_bayesian(
        _model_fn,
        NILE.reshape(-1, 1),
        jnp.array([3.0, 5.0]),
        n_warmup=50,
        n_samples=100,
        key=jax.random.PRNGKey(0),
    )


def test_compute_waic_returns_dict() -> None:
    result = _get_result()
    waic = compute_waic(result, _model_fn, NILE.reshape(-1, 1))
    assert "elpd_waic" in waic
    assert "p_waic" in waic
    assert "se" in waic
    assert isinstance(waic["elpd_waic"], float)


def test_compute_loo_returns_dict() -> None:
    result = _get_result()
    loo = compute_loo(result, _model_fn, NILE.reshape(-1, 1))
    assert "elpd_loo" in loo
    assert "p_loo" in loo
    assert "se" in loo


def test_to_arviz_returns_datatree() -> None:
    result = _get_result()
    dt = to_arviz(result, _model_fn, NILE.reshape(-1, 1))
    assert "posterior" in dt
    assert "log_likelihood" in dt
