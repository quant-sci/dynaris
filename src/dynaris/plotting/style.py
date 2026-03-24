"""Shared plot styling for dynaris — minimalist, precise, clean."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Grays for lines/points, blues for confidence intervals
COLORS = {
    "primary": "#F0F0F0",  # light gray
    "secondary": "#4D4D4D",  # dark gray (main line)
    "tertiary": "#808080",  # mid gray
    "dark": "#2B2B2B",  # near-black
    "darkest": "#1A1A1A",  # darkest gray
    "observed": "#999999",  # neutral gray for data points
    "ci_fill": "#4A90C4",  # blue confidence band
    "ci_fill_alt": "#2166AC",  # deeper blue alternate band
}

CMAP = "Greys"


def apply_style(ax: Axes) -> None:
    """Apply minimalist styling to an axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(labelsize=8, width=0.6, length=3)
    ax.grid(True, alpha=0.15, linewidth=0.4, color="#888888")
    ax.set_axisbelow(True)


def create_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> tuple[Figure, Any]:
    """Create a styled figure with subplots."""
    if figsize is None:
        figsize = (8, 3.2 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    fig.set_facecolor("white")

    if isinstance(axes, np.ndarray):
        for ax in axes.flat:
            apply_style(ax)
    else:
        apply_style(axes)

    return fig, axes
