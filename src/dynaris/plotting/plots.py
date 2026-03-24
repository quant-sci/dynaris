"""Plot functions for DLM results — minimalist, grays with blue CIs."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from dynaris.core.results import FilterResult, SmootherResult
from dynaris.core.state_space import StateSpaceModel
from dynaris.estimation.diagnostics import acf as compute_acf
from dynaris.estimation.diagnostics import standardized_residuals
from dynaris.forecast.forecast import ForecastResult, confidence_bands
from dynaris.plotting.style import COLORS, apply_style, create_figure

# ---------------------------------------------------------------------------
# Filtered vs observed
# ---------------------------------------------------------------------------


def plot_filtered(
    filter_result: FilterResult,
    model: StateSpaceModel,
    component: int = 0,
    level: float = 0.95,
    ax: Axes | None = None,
    title: str = "Filtered state",
) -> Figure:
    """Plot filtered state estimates vs observed data with confidence bands.

    Args:
        filter_result: Output of a Kalman filter pass.
        model: The state-space model.
        component: Which observation dimension to plot (default 0).
        level: Confidence level for bands.
        ax: Optional existing axes.
        title: Plot title.

    Returns:
        The matplotlib Figure.
    """
    obs = np.asarray(filter_result.observations[:, component])
    fitted = np.asarray(filter_result.filtered_states @ model.F.T)[:, component]
    lower, upper = confidence_bands(
        filter_result.filtered_states @ model.F.T,
        jnp.einsum("ij,tjk,lk->til", model.F, filter_result.filtered_covariances, model.F)
        + model.V[None, :, :],
        level=level,
    )
    lower = np.asarray(lower)[:, component]
    upper = np.asarray(upper)[:, component]
    t = np.arange(len(obs))

    if ax is None:
        fig, ax = create_figure()
    else:
        fig = ax.figure

    ax.scatter(t, obs, s=6, color=COLORS["observed"], alpha=0.6, zorder=2, label="Observed")
    ax.plot(t, fitted, linewidth=1.2, color=COLORS["secondary"], zorder=3, label="Filtered")
    ax.fill_between(t, lower, upper, alpha=0.18, color=COLORS["ci_fill"], zorder=1)
    ax.set_xlabel("Time", fontsize=8)
    ax.set_ylabel("Value", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="medium")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Smoothed states with confidence intervals
# ---------------------------------------------------------------------------


def plot_smoothed(
    smoother_result: SmootherResult,
    model: StateSpaceModel,
    component: int = 0,
    level: float = 0.95,
    ax: Axes | None = None,
    title: str = "Smoothed state",
) -> Figure:
    """Plot smoothed state estimates with confidence bands.

    Args:
        smoother_result: Output of an RTS smoother pass.
        model: The state-space model.
        component: Which observation dimension to plot.
        level: Confidence level.
        ax: Optional existing axes.
        title: Plot title.

    Returns:
        The matplotlib Figure.
    """
    obs = np.asarray(smoother_result.observations[:, component])
    smoothed = np.asarray(smoother_result.smoothed_states @ model.F.T)[:, component]
    lower, upper = confidence_bands(
        smoother_result.smoothed_states @ model.F.T,
        jnp.einsum("ij,tjk,lk->til", model.F, smoother_result.smoothed_covariances, model.F)
        + model.V[None, :, :],
        level=level,
    )
    lower = np.asarray(lower)[:, component]
    upper = np.asarray(upper)[:, component]
    t = np.arange(len(obs))

    if ax is None:
        fig, ax = create_figure()
    else:
        fig = ax.figure

    ax.scatter(t, obs, s=6, color=COLORS["observed"], alpha=0.6, zorder=2, label="Observed")
    ax.plot(t, smoothed, linewidth=1.2, color=COLORS["tertiary"], zorder=3, label="Smoothed")
    ax.fill_between(t, lower, upper, alpha=0.18, color=COLORS["ci_fill_alt"], zorder=1)
    ax.set_xlabel("Time", fontsize=8)
    ax.set_ylabel("Value", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="medium")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Individual components (trend, seasonal, etc.)
# ---------------------------------------------------------------------------


def plot_components(
    smoother_result: SmootherResult,
    component_dims: dict[str, slice | int],
    level: float = 0.95,
    title: str = "State components",
) -> Figure:
    """Plot individual state components from a composed model.

    Args:
        smoother_result: Output of an RTS smoother pass.
        component_dims: Mapping of component name to state dimension
            index or slice. E.g. ``{"Trend": slice(0, 2), "Seasonal": slice(2, 13)}``.
            Only the first dimension of each slice is plotted.
        level: Confidence level for bands.
        title: Overall figure title.

    Returns:
        The matplotlib Figure.
    """
    n_comps = len(component_dims)
    fig, axes = create_figure(nrows=n_comps, figsize=(8, 2.5 * n_comps))
    if n_comps == 1:
        axes = [axes]

    colors = [COLORS["secondary"], COLORS["tertiary"], COLORS["dark"], COLORS["primary"]]

    for i, (name, dim_spec) in enumerate(component_dims.items()):
        ax = axes[i]
        if isinstance(dim_spec, int):
            idx = dim_spec
        else:
            idx = dim_spec.start if dim_spec.start is not None else 0

        states = np.asarray(smoother_result.smoothed_states[:, idx])
        variances = np.asarray(smoother_result.smoothed_covariances[:, idx, idx])
        lower, upper = confidence_bands(jnp.array(states), jnp.array(variances), level=level)
        lower = np.asarray(lower)
        upper = np.asarray(upper)
        t = np.arange(len(states))
        color = colors[i % len(colors)]

        ax.plot(t, states, linewidth=1.0, color=color)
        ax.fill_between(t, lower, upper, alpha=0.15, color=color)
        ax.set_title(name, fontsize=8, fontweight="medium")
        ax.set_xlabel("Time", fontsize=7)

    fig.suptitle(title, fontsize=10, fontweight="medium", y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Forecast fan chart
# ---------------------------------------------------------------------------


def plot_forecast(
    filter_result: FilterResult,
    forecast_result: ForecastResult,
    model: StateSpaceModel,
    component: int = 0,
    levels: tuple[float, ...] = (0.50, 0.80, 0.95),
    n_history: int | None = None,
    ax: Axes | None = None,
    title: str = "Forecast",
) -> Figure:
    """Plot forecast fan chart with multiple confidence levels.

    Args:
        filter_result: Historical filter result.
        forecast_result: Forecast result.
        model: The state-space model.
        component: Which observation dimension to plot.
        levels: Confidence levels for the fan bands (inner to outer).
        n_history: Number of historical points to show. None = all.
        ax: Optional existing axes.
        title: Plot title.

    Returns:
        The matplotlib Figure.
    """
    obs = np.asarray(filter_result.observations[:, component])
    n_obs = len(obs)
    if n_history is not None:
        obs = obs[-n_history:]
        t_hist = np.arange(n_obs - n_history, n_obs)
    else:
        t_hist = np.arange(n_obs)

    fc_mean = np.asarray(forecast_result.mean[:, component])
    n_fc = len(fc_mean)
    t_fc = np.arange(n_obs, n_obs + n_fc)

    if ax is None:
        fig, ax = create_figure()
    else:
        fig = ax.figure

    # Historical data
    ax.scatter(t_hist, obs, s=6, color=COLORS["observed"], alpha=0.6, zorder=2, label="Observed")

    # Filtered line for history
    fitted = np.asarray(filter_result.filtered_states @ model.F.T)[:, component]
    if n_history is not None:
        fitted = fitted[-n_history:]
    ax.plot(t_hist, fitted, linewidth=1.0, color=COLORS["secondary"], alpha=0.7, zorder=3)

    # Forecast mean
    ax.plot(t_fc, fc_mean, linewidth=1.2, color=COLORS["secondary"], zorder=4, label="Forecast")

    # Fan bands — outer first so inner draws on top
    import matplotlib

    cmap = matplotlib.colormaps["Blues"]
    sorted_levels = sorted(levels, reverse=True)
    n_levels = len(sorted_levels)
    for i, lev in enumerate(sorted_levels):
        lower, upper = confidence_bands(
            forecast_result.mean[:, component],
            forecast_result.covariance[:, component, component],
            level=lev,
        )
        lower = np.asarray(lower)
        upper = np.asarray(upper)
        # Map level index to bone color (darker = wider band)
        frac = 0.25 + 0.5 * (i / max(n_levels - 1, 1))
        color = cmap(frac)
        ax.fill_between(
            t_fc,
            lower,
            upper,
            alpha=0.20,
            color=color,
            label=f"{int(lev * 100)}% CI",
        )

    # Vertical line at forecast origin
    ax.axvline(n_obs - 0.5, color="#AAAAAA", linewidth=0.5, linestyle="--", zorder=1)

    ax.set_xlabel("Time", fontsize=8)
    ax.set_ylabel("Value", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="medium")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Residual diagnostics
# ---------------------------------------------------------------------------


def plot_diagnostics(
    filter_result: FilterResult,
    model: StateSpaceModel,
    n_lags: int = 20,
    title: str = "Residual diagnostics",
) -> Figure:
    """Plot residual diagnostic panel: residuals, histogram, QQ-plot, ACF.

    Args:
        filter_result: Output of a Kalman filter pass.
        model: The state-space model.
        n_lags: Number of lags for ACF plot.
        title: Overall figure title.

    Returns:
        The matplotlib Figure with 4 subplots.
    """
    from scipy import stats

    resids = np.asarray(standardized_residuals(filter_result, model))
    if resids.ndim > 1:
        resids = resids[:, 0]
    t = np.arange(len(resids))

    fig, axes = create_figure(nrows=2, ncols=2, figsize=(8, 5.5))

    # --- 1. Standardized residuals over time ---
    ax = axes[0, 0]
    ax.scatter(t, resids, s=4, color=COLORS["secondary"], alpha=0.7)
    ax.axhline(0, color="#888888", linewidth=0.5)
    ax.axhline(2, color=COLORS["tertiary"], linewidth=0.4, linestyle="--", alpha=0.5)
    ax.axhline(-2, color=COLORS["tertiary"], linewidth=0.4, linestyle="--", alpha=0.5)
    ax.set_xlabel("Time", fontsize=7)
    ax.set_ylabel("Std. residual", fontsize=7)
    ax.set_title("Standardized residuals", fontsize=8, fontweight="medium")

    # --- 2. Histogram ---
    ax = axes[0, 1]
    ax.hist(
        resids,
        bins=25,
        density=True,
        color=COLORS["secondary"],
        alpha=0.6,
        edgecolor="white",
        linewidth=0.3,
    )
    x_grid = np.linspace(float(resids.min()) - 0.5, float(resids.max()) + 0.5, 200)
    ax.plot(x_grid, stats.norm.pdf(x_grid), linewidth=1.0, color=COLORS["tertiary"])
    ax.set_xlabel("Value", fontsize=7)
    ax.set_ylabel("Density", fontsize=7)
    ax.set_title("Histogram", fontsize=8, fontweight="medium")

    # --- 3. QQ-plot ---
    ax = axes[1, 0]
    sorted_resids = np.sort(resids)
    n = len(sorted_resids)
    theoretical_q = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    ax.scatter(theoretical_q, sorted_resids, s=6, color=COLORS["secondary"], alpha=0.7)
    lims = [
        min(theoretical_q.min(), sorted_resids.min()),
        max(theoretical_q.max(), sorted_resids.max()),
    ]
    ax.plot(lims, lims, linewidth=0.8, color=COLORS["tertiary"], linestyle="--")
    ax.set_xlabel("Theoretical quantiles", fontsize=7)
    ax.set_ylabel("Sample quantiles", fontsize=7)
    ax.set_title("Q-Q plot", fontsize=8, fontweight="medium")

    # --- 4. ACF ---
    ax = axes[1, 1]
    acf_vals = np.asarray(compute_acf(jnp.array(resids), n_lags=n_lags))
    lags = np.arange(len(acf_vals))
    ax.bar(lags[1:], acf_vals[1:], width=0.6, color=COLORS["secondary"], alpha=0.7)
    # Significance bounds (approximate 95%)
    sig = 1.96 / np.sqrt(len(resids))
    ax.axhline(sig, color=COLORS["tertiary"], linewidth=0.5, linestyle="--", alpha=0.6)
    ax.axhline(-sig, color=COLORS["tertiary"], linewidth=0.5, linestyle="--", alpha=0.6)
    ax.axhline(0, color="#888888", linewidth=0.4)
    ax.set_xlabel("Lag", fontsize=7)
    ax.set_ylabel("ACF", fontsize=7)
    ax.set_title("Autocorrelation", fontsize=8, fontweight="medium")

    fig.suptitle(title, fontsize=10, fontweight="medium")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Panel — combined overview
# ---------------------------------------------------------------------------


def plot_panel(
    filter_result: FilterResult,
    smoother_result: SmootherResult | None,
    forecast_result: ForecastResult | None,
    model: StateSpaceModel,
    component: int = 0,
    level: float = 0.95,
    forecast_levels: tuple[float, ...] = (0.50, 0.80, 0.95),
    n_history: int | None = None,
    n_lags: int = 20,
    title: str = "",
) -> Figure:
    """Unified panel with filtered, smoothed, forecast, and diagnostics.

    Layout (2 rows x 3 cols):

    - [0, 0] Filtered vs observed
    - [0, 1] Smoothed vs observed
    - [0, 2] Forecast fan chart
    - [1, 0] Standardized residuals
    - [1, 1] Q-Q plot
    - [1, 2] ACF

    Args:
        filter_result: Output of a Kalman filter pass.
        smoother_result: Output of an RTS smoother (optional).
        forecast_result: Output of a forecast (optional).
        model: The state-space model.
        component: Which observation dimension to plot.
        level: Confidence level for filtered/smoothed bands.
        forecast_levels: Confidence levels for forecast fan chart.
        n_history: Historical points to show in forecast panel.
        n_lags: Number of lags for ACF.
        title: Overall figure title.

    Returns:
        The matplotlib Figure.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from scipy import stats

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    fig.set_facecolor("white")
    for ax in axes.flat:
        apply_style(ax)

    obs = np.asarray(filter_result.observations[:, component])
    t = np.arange(len(obs))
    n_obs = len(obs)

    # --- [0, 0] Filtered ---
    ax = axes[0, 0]
    fitted = np.asarray(filter_result.filtered_states @ model.F.T)[:, component]
    lower_f, upper_f = confidence_bands(
        filter_result.filtered_states @ model.F.T,
        jnp.einsum("ij,tjk,lk->til", model.F, filter_result.filtered_covariances, model.F)
        + model.V[None, :, :],
        level=level,
    )
    lower_f = np.asarray(lower_f)[:, component]
    upper_f = np.asarray(upper_f)[:, component]

    ax.scatter(t, obs, s=4, color=COLORS["observed"], alpha=0.5, zorder=2)
    ax.plot(t, fitted, linewidth=1.0, color=COLORS["secondary"], zorder=3, label="Filtered")
    ax.fill_between(t, lower_f, upper_f, alpha=0.15, color=COLORS["ci_fill"], zorder=1)
    ax.set_title("Filtered", fontsize=8, fontweight="medium")
    ax.legend(fontsize=6, frameon=False)

    # --- [0, 1] Smoothed ---
    ax = axes[0, 1]
    if smoother_result is not None:
        smoothed = np.asarray(smoother_result.smoothed_states @ model.F.T)[:, component]
        lower_s, upper_s = confidence_bands(
            smoother_result.smoothed_states @ model.F.T,
            jnp.einsum("ij,tjk,lk->til", model.F, smoother_result.smoothed_covariances, model.F)
            + model.V[None, :, :],
            level=level,
        )
        lower_s = np.asarray(lower_s)[:, component]
        upper_s = np.asarray(upper_s)[:, component]
        ax.scatter(t, obs, s=4, color=COLORS["observed"], alpha=0.5, zorder=2)
        ax.plot(t, smoothed, linewidth=1.0, color=COLORS["secondary"], zorder=3, label="Smoothed")
        ax.fill_between(t, lower_s, upper_s, alpha=0.15, color=COLORS["ci_fill_alt"], zorder=1)
        ax.set_title("Smoothed", fontsize=8, fontweight="medium")
        ax.legend(fontsize=6, frameon=False)
    else:
        ax.scatter(t, obs, s=4, color=COLORS["observed"], alpha=0.5, zorder=2)
        ax.plot(t, fitted, linewidth=1.0, color=COLORS["secondary"], zorder=3)
        ax.set_title("Filtered (no smoother)", fontsize=8, fontweight="medium")

    # --- [0, 2] Forecast ---
    ax = axes[0, 2]
    if forecast_result is not None:
        fc_mean = np.asarray(forecast_result.mean[:, component])
        n_fc = len(fc_mean)
        t_fc = np.arange(n_obs, n_obs + n_fc)

        hist_obs = obs
        hist_t = t
        hist_fit = fitted
        if n_history is not None:
            hist_obs = obs[-n_history:]
            hist_t = t[-n_history:]
            hist_fit = fitted[-n_history:]

        ax.scatter(hist_t, hist_obs, s=4, color=COLORS["observed"], alpha=0.5, zorder=2)
        ax.plot(hist_t, hist_fit, linewidth=0.8, color=COLORS["secondary"], alpha=0.6, zorder=3)
        ax.plot(t_fc, fc_mean, linewidth=1.0, color=COLORS["secondary"], zorder=4, label="Forecast")

        cmap = matplotlib.colormaps["Blues"]
        sorted_levels = sorted(forecast_levels, reverse=True)
        n_levels = len(sorted_levels)
        for i, lev in enumerate(sorted_levels):
            lo, hi = confidence_bands(
                forecast_result.mean[:, component],
                forecast_result.covariance[:, component, component],
                level=lev,
            )
            frac = 0.25 + 0.5 * (i / max(n_levels - 1, 1))
            ax.fill_between(
                t_fc,
                np.asarray(lo),
                np.asarray(hi),
                alpha=0.20,
                color=cmap(frac),
                label=f"{int(lev * 100)}%",
            )
        ax.axvline(n_obs - 0.5, color="#AAAAAA", linewidth=0.4, linestyle="--", zorder=1)
        ax.set_title("Forecast", fontsize=8, fontweight="medium")
        ax.legend(fontsize=6, frameon=False, ncol=2)
    else:
        ax.scatter(t, obs, s=4, color=COLORS["observed"], alpha=0.5)
        ax.set_title("Forecast (none)", fontsize=8, fontweight="medium", color="#AAAAAA")

    # --- [1, 0] Standardized residuals ---
    ax = axes[1, 0]
    resids = np.asarray(standardized_residuals(filter_result, model))
    if resids.ndim > 1:
        resids = resids[:, 0]
    ax.scatter(t, resids, s=3, color=COLORS["secondary"], alpha=0.6)
    ax.axhline(0, color="#888888", linewidth=0.4)
    ax.axhline(2, color=COLORS["ci_fill_alt"], linewidth=0.4, linestyle="--", alpha=0.5)
    ax.axhline(-2, color=COLORS["ci_fill_alt"], linewidth=0.4, linestyle="--", alpha=0.5)
    ax.set_title("Residuals", fontsize=8, fontweight="medium")
    ax.set_ylabel("Std. resid.", fontsize=7)

    # --- [1, 1] QQ-plot ---
    ax = axes[1, 1]
    sorted_resids = np.sort(resids)
    n = len(sorted_resids)
    theoretical_q = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    ax.scatter(theoretical_q, sorted_resids, s=4, color=COLORS["secondary"], alpha=0.6)
    lims = [
        min(theoretical_q.min(), sorted_resids.min()),
        max(theoretical_q.max(), sorted_resids.max()),
    ]
    ax.plot(lims, lims, linewidth=0.7, color=COLORS["ci_fill_alt"], linestyle="--")
    ax.set_xlabel("Theoretical", fontsize=7)
    ax.set_ylabel("Sample", fontsize=7)
    ax.set_title("Q-Q plot", fontsize=8, fontweight="medium")

    # --- [1, 2] ACF ---
    ax = axes[1, 2]
    acf_vals = np.asarray(compute_acf(jnp.array(resids), n_lags=n_lags))
    lags = np.arange(len(acf_vals))
    ax.bar(lags[1:], acf_vals[1:], width=0.6, color=COLORS["secondary"], alpha=0.6)
    sig = 1.96 / np.sqrt(len(resids))
    ax.axhline(sig, color=COLORS["ci_fill_alt"], linewidth=0.4, linestyle="--", alpha=0.6)
    ax.axhline(-sig, color=COLORS["ci_fill_alt"], linewidth=0.4, linestyle="--", alpha=0.6)
    ax.axhline(0, color="#888888", linewidth=0.3)
    ax.set_xlabel("Lag", fontsize=7)
    ax.set_ylabel("ACF", fontsize=7)
    ax.set_title("Autocorrelation", fontsize=8, fontweight="medium")

    if title:
        fig.suptitle(title, fontsize=11, fontweight="medium", y=1.0)
    fig.tight_layout(h_pad=1.8, w_pad=1.5)
    return fig
