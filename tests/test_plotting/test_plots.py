"""Tests for plotting functions — verify they run without error."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import pytest

from dynaris.datasets import load_nile_jax
from dynaris.dlm.components import LocalLevel, LocalLinearTrend, Seasonal
from dynaris.filters.kalman import kalman_filter
from dynaris.forecast.forecast import forecast_from_filter
from dynaris.plotting.plots import (
    plot_components,
    plot_diagnostics,
    plot_filtered,
    plot_forecast,
    plot_panel,
    plot_smoothed,
)
from dynaris.smoothers.rts import rts_smooth

# Use non-interactive backend for tests
matplotlib.use("Agg")

NILE = load_nile_jax()


@pytest.fixture
def nile_fit() -> tuple:
    model = LocalLevel(sigma_level=38.33, sigma_obs=122.9)
    obs = NILE.reshape(-1, 1)
    fr = kalman_filter(model, obs)
    sr = rts_smooth(model, fr)
    return model, fr, sr


# ===================================================================
# plot_filtered
# ===================================================================


def test_plot_filtered(nile_fit: tuple) -> None:
    model, fr, _ = nile_fit
    fig = plot_filtered(fr, model)
    assert fig is not None
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_plot_filtered_custom_title(nile_fit: tuple) -> None:
    model, fr, _ = nile_fit
    fig = plot_filtered(fr, model, title="Nile filtered")
    assert fig.axes[0].get_title() == "Nile filtered"
    plt.close(fig)


def test_plot_filtered_on_existing_axes(nile_fit: tuple) -> None:
    model, fr, _ = nile_fit
    _, ax = plt.subplots()
    fig = plot_filtered(fr, model, ax=ax)
    assert fig is not None
    plt.close(fig)


# ===================================================================
# plot_smoothed
# ===================================================================


def test_plot_smoothed(nile_fit: tuple) -> None:
    model, _, sr = nile_fit
    fig = plot_smoothed(sr, model)
    assert fig is not None
    plt.close(fig)


# ===================================================================
# plot_components
# ===================================================================


def test_plot_components() -> None:
    model = (
        LocalLinearTrend(sigma_level=1.0, sigma_slope=0.1, sigma_obs=0.0)
        + Seasonal(period=12, sigma_seasonal=0.5, sigma_obs=2.0)
    )
    t = jnp.arange(60, dtype=jnp.float32)
    obs = (
        t * 0.3
        + 5.0 * jnp.sin(2.0 * jnp.pi * t / 12.0)
        + jax.random.normal(jax.random.PRNGKey(0), (60,)) * 2.0
    ).reshape(-1, 1)

    fr = kalman_filter(model, obs)
    sr = rts_smooth(model, fr)

    fig = plot_components(sr, {
        "Level": 0,
        "Slope": 1,
        "Seasonal": 2,
    })
    assert fig is not None
    assert len(fig.axes) == 3
    plt.close(fig)


# ===================================================================
# plot_forecast
# ===================================================================


def test_plot_forecast(nile_fit: tuple) -> None:
    model, fr, _ = nile_fit
    fc = forecast_from_filter(model, fr, steps=24)
    fig = plot_forecast(fr, fc, model)
    assert fig is not None
    plt.close(fig)


def test_plot_forecast_with_history(nile_fit: tuple) -> None:
    model, fr, _ = nile_fit
    fc = forecast_from_filter(model, fr, steps=12)
    fig = plot_forecast(fr, fc, model, n_history=30)
    assert fig is not None
    plt.close(fig)


def test_plot_forecast_custom_levels(nile_fit: tuple) -> None:
    model, fr, _ = nile_fit
    fc = forecast_from_filter(model, fr, steps=12)
    fig = plot_forecast(fr, fc, model, levels=(0.50, 0.90, 0.99))
    assert fig is not None
    plt.close(fig)


# ===================================================================
# plot_diagnostics
# ===================================================================


def test_plot_diagnostics(nile_fit: tuple) -> None:
    model, fr, _ = nile_fit
    fig = plot_diagnostics(fr, model)
    assert fig is not None
    assert len(fig.axes) == 4
    plt.close(fig)


def test_plot_diagnostics_custom_lags(nile_fit: tuple) -> None:
    model, fr, _ = nile_fit
    fig = plot_diagnostics(fr, model, n_lags=10)
    assert fig is not None
    plt.close(fig)


# ===================================================================
# Composed model plot
# ===================================================================


def test_plot_composed_model_forecast() -> None:
    model = (
        LocalLinearTrend(sigma_level=1.0, sigma_slope=0.1, sigma_obs=0.0)
        + Seasonal(period=12, sigma_seasonal=0.5, sigma_obs=2.0)
    )
    t = jnp.arange(60, dtype=jnp.float32)
    obs = (
        t * 0.3
        + 5.0 * jnp.sin(2.0 * jnp.pi * t / 12.0)
        + jax.random.normal(jax.random.PRNGKey(1), (60,)) * 2.0
    ).reshape(-1, 1)

    fr = kalman_filter(model, obs)
    fc = forecast_from_filter(model, fr, steps=24)
    fig = plot_forecast(fr, fc, model, n_history=24)
    assert fig is not None
    plt.close(fig)


# ===================================================================
# plot_panel
# ===================================================================


def test_plot_panel(nile_fit: tuple) -> None:
    model, fr, sr = nile_fit
    fc = forecast_from_filter(model, fr, steps=10)
    fig = plot_panel(fr, sr, fc, model, title="Panel test")
    assert fig is not None
    assert len(fig.axes) == 6
    plt.close(fig)


def test_plot_panel_no_smoother_no_forecast(nile_fit: tuple) -> None:
    model, fr, _ = nile_fit
    fig = plot_panel(fr, None, None, model)
    assert fig is not None
    assert len(fig.axes) == 6
    plt.close(fig)
