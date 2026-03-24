"""Tests for plotting functions — verify they run without error."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import pytest

from dynaris.core.state_space import StateSpaceModel
from dynaris.dlm.components import LocalLevel, LocalLinearTrend, Seasonal
from dynaris.filters.kalman import kalman_filter
from dynaris.forecast.forecast import forecast_from_filter
from dynaris.plotting.plots import (
    plot_components,
    plot_diagnostics,
    plot_filtered,
    plot_forecast,
    plot_smoothed,
)
from dynaris.smoothers.rts import rts_smooth

# Use non-interactive backend for tests
matplotlib.use("Agg")

NILE = jnp.array([
    1120, 1160, 963, 1210, 1160, 1160, 813, 1230, 1370, 1140,
    995, 935, 1110, 994, 1020, 960, 1180, 799, 958, 1140,
    1100, 1210, 1150, 1250, 1260, 1220, 1030, 1100, 774, 840,
    874, 694, 940, 833, 701, 916, 692, 1020, 1050, 969,
    831, 726, 456, 824, 702, 1120, 1100, 832, 764, 821,
    768, 845, 864, 862, 698, 845, 744, 796, 1040, 759,
    781, 865, 845, 944, 984, 897, 822, 1010, 771, 676,
    649, 846, 812, 742, 801, 1040, 860, 874, 848, 890,
    744, 749, 838, 1050, 918, 986, 797, 923, 975, 815,
    1020, 906, 901, 1170, 912, 746, 919, 718, 714, 740,
], dtype=jnp.float32)


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
