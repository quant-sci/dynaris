"""Tests for Dynamic Factor Model EM estimation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from dynaris.estimation.dfm import DFMResult, fit_dfm_em


def _simulate_dfm_data(
    n_factors: int = 2,
    n_variables: int = 8,
    n_obs: int = 200,
    key: jax.Array | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Simulate data from a known DFM for testing."""
    if key is None:
        key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    r, m, t = n_factors, n_variables, n_obs

    # True loadings
    true_loadings = jax.random.normal(k1, (m, r)) * 0.5 + 0.5

    # True factors (random walk)
    factor_noise = jax.random.normal(k2, (t, r)) * 0.5
    true_factors = jnp.cumsum(factor_noise, axis=0)

    # Observations
    obs_noise = jax.random.normal(k3, (t, m)) * 1.0
    observations = true_factors @ true_loadings.T + obs_noise

    return observations, true_loadings, true_factors


def test_dfm_em_returns_result() -> None:
    obs, _, _ = _simulate_dfm_data(n_obs=100)
    result = fit_dfm_em(obs, n_factors=2, max_iter=10)
    assert isinstance(result, DFMResult)


def test_dfm_em_shapes() -> None:
    obs, _, _ = _simulate_dfm_data(n_factors=2, n_variables=8, n_obs=100)
    result = fit_dfm_em(obs, n_factors=2, max_iter=20)

    assert result.loadings.shape == (8, 2)
    assert result.factor_states.shape == (100, 2)
    assert result.factor_covariances.shape == (100, 2, 2)
    assert result.explained_variance.shape == (2,)


def test_dfm_em_finite() -> None:
    obs, _, _ = _simulate_dfm_data(n_obs=100)
    result = fit_dfm_em(obs, n_factors=2, max_iter=20)

    assert jnp.all(jnp.isfinite(result.loadings))
    assert jnp.all(jnp.isfinite(result.factor_states))
    assert jnp.isfinite(jnp.array(result.log_likelihood))


def test_dfm_em_likelihood_nondecreasing() -> None:
    """EM log-likelihood should be non-decreasing."""
    obs, _, _ = _simulate_dfm_data(n_obs=100)
    result = fit_dfm_em(obs, n_factors=2, max_iter=30)

    ll_hist = result.log_likelihood_history
    for i in range(1, len(ll_hist)):
        assert ll_hist[i] >= ll_hist[i - 1] - 1.0, (
            f"LL decreased at step {i}: {ll_hist[i]} < {ll_hist[i - 1]}"
        )


def test_dfm_em_recovers_factors() -> None:
    """Smoothed factors should correlate with true factors."""
    obs, _, true_factors = _simulate_dfm_data(n_obs=200)
    result = fit_dfm_em(obs, n_factors=2, max_iter=50)

    # Check correlation of each estimated factor with best-matching true factor
    for j in range(2):
        corrs = []
        for k in range(2):
            pair = jnp.stack([result.factor_states[:, j], true_factors[:, k]])
            c = float(jnp.corrcoef(pair)[0, 1])
            corrs.append(abs(c))
        best_corr = max(corrs)
        assert best_corr > 0.5, f"Factor {j} best correlation {best_corr} too low"


def test_dfm_em_r_is_diagonal() -> None:
    """R should remain diagonal after estimation."""
    obs, _, _ = _simulate_dfm_data(n_obs=100)
    result = fit_dfm_em(obs, n_factors=2, max_iter=10)

    v = result.model.V
    off_diag = v - jnp.diag(jnp.diag(v))
    np.testing.assert_allclose(off_diag, jnp.zeros_like(off_diag), atol=1e-6)


def test_dfm_em_single_factor() -> None:
    """Should work with a single factor."""
    obs, _, _ = _simulate_dfm_data(n_factors=1, n_variables=5, n_obs=100)
    result = fit_dfm_em(obs, n_factors=1, max_iter=20)
    assert result.loadings.shape == (5, 1)
    assert result.factor_states.shape == (100, 1)


def test_dfm_em_pca_init() -> None:
    obs, _, _ = _simulate_dfm_data(n_obs=100)
    result = fit_dfm_em(obs, n_factors=2, max_iter=10, init_method="pca")
    assert isinstance(result, DFMResult)
    assert jnp.all(jnp.isfinite(result.loadings))
