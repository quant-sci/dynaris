"""Tests for switching model selection utilities."""

from __future__ import annotations

from dynaris.estimation.model_selection import switching_aic, switching_bic


def test_switching_aic_formula() -> None:
    """AIC = -2*LL + 2*k."""
    aic = switching_aic(log_likelihood=-500.0, n_regimes=1, state_dim=1, obs_dim=1)
    assert aic > 0.0
    # With K=1: params = 1*(1+1+1+1) + 0 + 0 = 4
    # AIC = -2*(-500) + 2*4 = 1008
    assert abs(aic - 1008.0) < 1e-6


def test_switching_bic_formula() -> None:
    """BIC = -2*LL + log(n)*k."""
    bic = switching_bic(log_likelihood=-500.0, n_regimes=1, state_dim=1, obs_dim=1, n_obs=100)
    assert bic > 0.0


def test_bic_penalizes_more_regimes() -> None:
    """More regimes means more parameters, so BIC should be higher for same LL."""
    bic_1 = switching_bic(-500.0, n_regimes=1, state_dim=1, obs_dim=1, n_obs=100)
    bic_3 = switching_bic(-500.0, n_regimes=3, state_dim=1, obs_dim=1, n_obs=100)
    assert bic_3 > bic_1


def test_aic_penalizes_more_regimes() -> None:
    """Same logic for AIC."""
    aic_1 = switching_aic(-500.0, n_regimes=1, state_dim=1, obs_dim=1)
    aic_3 = switching_aic(-500.0, n_regimes=3, state_dim=1, obs_dim=1)
    assert aic_3 > aic_1


def test_better_fit_lowers_criterion() -> None:
    """Higher log-likelihood (closer to 0) should lower the criterion."""
    aic_bad = switching_aic(-500.0, n_regimes=2, state_dim=1, obs_dim=1)
    aic_good = switching_aic(-300.0, n_regimes=2, state_dim=1, obs_dim=1)
    assert aic_good < aic_bad
