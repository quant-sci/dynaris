"""Benchmark datasets for DLM examples and testing."""

from __future__ import annotations

import jax.numpy as jnp
import pandas as pd
from jax import Array

# ===================================================================
# Nile river flow (hydrology / engineering)
# ===================================================================


def load_nile() -> pd.Series:
    """Nile river annual flow at Aswan, 1871--1970.

    A classic benchmark for the local-level DLM. 100 observations of
    annual flow volume (10^8 m^3).

    Source: Durbin & Koopman (2012). Originally from Cobb (1978).

    Returns:
        pandas Series with DatetimeIndex (annual frequency).
    """
    index = pd.date_range("1871", periods=len(_NILE), freq="YE")
    return pd.Series(_NILE, index=index, name="nile_flow", dtype="float64")


def load_nile_jax() -> Array:
    """Nile river data as a JAX array, shape (100,)."""
    return jnp.array(_NILE, dtype=jnp.float32)


# ===================================================================
# Airline passengers (economics / engineering)
# ===================================================================


def load_airline() -> pd.Series:
    """International airline passengers, monthly 1949--1960.

    144 monthly observations of thousands of passengers. The classic
    Box--Jenkins dataset. Shows upward trend with multiplicative
    seasonality (use log-transform for additive DLM).

    Source: Box, Jenkins & Reinsel (2015), *Time Series Analysis*.

    Returns:
        pandas Series with monthly DatetimeIndex.
    """
    index = pd.date_range("1949-01", periods=len(_AIRLINE), freq="ME")
    return pd.Series(_AIRLINE, index=index, name="passengers", dtype="float64")


def load_airline_jax() -> Array:
    """Airline passengers as a JAX array, shape (144,)."""
    return jnp.array(_AIRLINE, dtype=jnp.float32)


# ===================================================================
# Canadian lynx trappings (biology / ecology)
# ===================================================================


def load_lynx() -> pd.Series:
    """Canadian lynx trappings, 1821--1934.

    114 annual observations of the number of lynx trapped in the
    Mackenzie River district of northwest Canada. Exhibits a prominent
    ~10-year population cycle driven by predator-prey dynamics.

    Source: Elton & Nicholson (1942). Standard ecology benchmark.

    Returns:
        pandas Series with annual DatetimeIndex.
    """
    index = pd.date_range("1821", periods=len(_LYNX), freq="YE")
    return pd.Series(_LYNX, index=index, name="lynx_trappings", dtype="float64")


def load_lynx_jax() -> Array:
    """Canadian lynx data as a JAX array, shape (114,)."""
    return jnp.array(_LYNX, dtype=jnp.float32)


# ===================================================================
# Sunspot numbers (physics / astronomy)
# ===================================================================


def load_sunspots() -> pd.Series:
    """Annual mean sunspot numbers, 1700--1988.

    289 annual observations of the Zurich/international sunspot number.
    Exhibits an approximately 11-year solar cycle with varying amplitude.

    Source: WDC-SILSO, Royal Observatory of Belgium.

    Returns:
        pandas Series with annual DatetimeIndex.
    """
    index = pd.date_range("1700", periods=len(_SUNSPOTS), freq="YE")
    return pd.Series(_SUNSPOTS, index=index, name="sunspot_number", dtype="float64")


def load_sunspots_jax() -> Array:
    """Sunspot numbers as a JAX array, shape (289,)."""
    return jnp.array(_SUNSPOTS, dtype=jnp.float32)


# ===================================================================
# Global temperature anomaly (physics / climate)
# ===================================================================


def load_temperature() -> pd.Series:
    """Global mean temperature anomaly, annual 1880--2023.

    144 annual observations of global surface temperature anomaly
    (degrees C relative to 1951-1980 mean). Shows the warming trend.

    Source: NASA GISS Surface Temperature Analysis (GISTEMP v4).

    Returns:
        pandas Series with annual DatetimeIndex.
    """
    index = pd.date_range("1880", periods=len(_TEMPERATURE), freq="YE")
    return pd.Series(
        _TEMPERATURE, index=index, name="temp_anomaly", dtype="float64"
    )


def load_temperature_jax() -> Array:
    """Global temperature anomaly as a JAX array, shape (144,)."""
    return jnp.array(_TEMPERATURE, dtype=jnp.float32)


# ===================================================================
# US real GDP growth (economics / finance)
# ===================================================================


def load_gdp() -> pd.Series:
    """US real GDP quarterly annualized growth rate, 1947Q2--2023Q4.

    307 quarterly observations of annualized percent change in real GDP.
    Shows business cycle fluctuations around a long-run mean.

    Source: US Bureau of Economic Analysis (BEA), NIPA Table 1.1.1.

    Returns:
        pandas Series with quarterly DatetimeIndex.
    """
    index = pd.date_range("1947-04", periods=len(_GDP_GROWTH), freq="QE")
    return pd.Series(
        _GDP_GROWTH, index=index, name="gdp_growth", dtype="float64"
    )


def load_gdp_jax() -> Array:
    """US GDP growth as a JAX array, shape (307,)."""
    return jnp.array(_GDP_GROWTH, dtype=jnp.float32)


# ===================================================================
# Raw data arrays
# ===================================================================

# Nile river flow (10^8 m^3), 1871-1970, 100 obs
_NILE: list[float] = [
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
]

# International airline passengers (thousands), monthly 1949-1960, 144 obs
_AIRLINE: list[float] = [
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
    145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
    171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
    196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
    204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
    242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
    284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
    315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
    340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
    360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
    417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432,
]

# Canadian lynx trappings, annual 1821-1934, 114 obs
_LYNX: list[float] = [
    269, 321, 585, 871, 1475, 2821, 3928, 5943, 4950, 2577,
    523, 98, 184, 279, 409, 2285, 2685, 3409, 1824, 409,
    151, 45, 68, 213, 546, 1033, 2129, 2536, 957, 361,
    377, 225, 360, 731, 1638, 2725, 2871, 2119, 684, 299,
    236, 245, 552, 1623, 3311, 6721, 4254, 687, 255, 473,
    358, 784, 1594, 1676, 2251, 1426, 756, 299, 201, 229,
    469, 736, 2042, 2811, 4431, 2511, 389, 73, 39, 49,
    59, 188, 377, 1292, 4031, 3495, 587, 105, 153, 387,
    758, 1307, 3465, 6991, 6313, 3794, 1836, 345, 382, 808,
    1388, 2713, 3800, 3091, 2985, 3790, 674, 81, 80, 108,
    229, 399, 1132, 2432, 3574, 2935, 1537, 529, 485, 662,
    1000, 1590, 2657, 3396,
]

# Annual mean sunspot numbers (Zurich), 1700-1988, 289 obs
_SUNSPOTS: list[float] = [
    5, 11, 16, 23, 36, 58, 29, 20, 10, 8, 3, 0, 0, 2, 11, 27, 47, 63,
    60, 39, 28, 26, 22, 11, 21, 40, 78, 122, 103, 73, 47, 35, 11, 5, 16,
    34, 70, 81, 111, 101, 73, 40, 20, 16, 5, 11, 22, 40, 60, 80.9, 83.4,
    47.7, 47.8, 30.7, 12.2, 9.6, 10.2, 32.4, 47.6, 54, 62.9, 85.9, 61.2,
    45.1, 36.4, 20.9, 11.4, 37.8, 69.8, 106.1, 100.8, 81.6, 66.5, 34.8,
    30.6, 7, 19.8, 92.5, 154.4, 125.9, 84.8, 68.1, 38.5, 22.8, 10.2,
    24.1, 82.9, 132, 130.9, 118.1, 89.9, 66.6, 60, 46.9, 41, 21.3, 16,
    6.4, 4.1, 6.8, 14.5, 34, 45, 43.1, 47.5, 42.2, 28.1, 10.1, 8.1,
    2.5, 0, 1.4, 5, 12.2, 13.9, 35.4, 45.8, 41.1, 30.1, 23.9, 15.6,
    6.6, 4, 1.8, 8.5, 16.6, 36.3, 49.6, 64.2, 67, 70.9, 47.8, 27.5,
    8.5, 13.2, 56.9, 121.5, 138.3, 103.2, 85.7, 64.6, 36.7, 24.2, 10.7,
    15, 40.1, 61.5, 98.5, 124.7, 96.3, 66.6, 64.5, 54.1, 39, 20.6, 6.7,
    4.3, 22.7, 54.8, 93.8, 95.8, 77.2, 59.1, 44, 47, 30.5, 16.3, 7.3,
    37.6, 74, 139, 111.2, 101.6, 66.2, 44.7, 17, 11.3, 12.4, 3.4, 6,
    32.3, 54.3, 59.7, 63.7, 63.5, 52.2, 25.4, 13.1, 6.8, 6.3, 7.1, 35.6,
    73, 85.1, 78, 64, 41.8, 26.2, 26.7, 12.1, 9.5, 2.7, 5, 24.4, 42,
    63.5, 53.8, 62, 48.5, 43.9, 18.6, 5.7, 3.6, 1.4, 9.6, 47.4, 57.1,
    103.9, 80.6, 63.6, 37.6, 26.1, 14.2, 5.8, 16.7, 44.3, 63.9, 69, 77.8,
    64.9, 35.7, 21.2, 11.1, 5.7, 8.7, 36.1, 79.7, 114.4, 109.6, 88.8,
    67.8, 47.5, 30.6, 16.3, 9.6, 33.2, 92.6, 151.6, 136.3, 134.7, 83.9,
    69.4, 31.5, 13.9, 4.4, 38, 141.7, 190.2, 184.8, 159, 112.3, 53.9,
    37.5, 27.9, 10.2, 15.1, 47, 93.7, 105.9, 105.5, 104.5, 66.6, 68.9,
    38, 34.5, 15.5, 12.6, 27.5, 92.5, 155.4, 154.6, 140.4, 115.9, 66.6,
    45.9, 17.9, 13.4, 29.2,
]

# Global mean temperature anomaly (deg C vs 1951-1980), annual 1880-2023, 144 obs
# Source: NASA GISS
_TEMPERATURE: list[float] = [
    -0.17, -0.09, -0.11, -0.17, -0.28, -0.33, -0.31, -0.36, -0.17, -0.10,
    -0.35, -0.25, -0.30, -0.35, -0.31, -0.23, -0.11, -0.11, -0.27, -0.18,
    -0.08, -0.16, -0.28, -0.37, -0.47, -0.26, -0.22, -0.39, -0.43, -0.48,
    -0.43, -0.44, -0.36, -0.34, -0.15, -0.14, -0.36, -0.46, -0.30, -0.28,
    -0.27, -0.19, -0.28, -0.26, -0.24, -0.20, -0.11, -0.22, -0.20, -0.29,
    -0.18, -0.07, -0.14, -0.23, -0.17, -0.21, -0.09, 0.02, 0.06, 0.10,
    0.02, 0.07, 0.04, 0.08, -0.01, 0.09, -0.07, -0.01, -0.07, -0.08,
    -0.16, -0.07, -0.01, 0.00, -0.05, -0.11, -0.06, -0.01, 0.01, 0.08,
    0.02, 0.06, 0.05, 0.10, -0.20, -0.11, -0.06, -0.02, -0.08, 0.05,
    0.03, 0.06, 0.02, 0.08, 0.14, -0.12, -0.07, -0.02, 0.04, 0.07,
    -0.02, 0.06, 0.06, 0.20, 0.05, -0.07, -0.01, 0.18, 0.07, 0.17,
    0.27, 0.32, 0.14, 0.31, 0.16, 0.12, 0.18, 0.32, 0.39, 0.27,
    0.45, 0.41, 0.22, 0.23, 0.32, 0.45, 0.33, 0.46, 0.61, 0.39,
    0.40, 0.54, 0.63, 0.62, 0.54, 0.68, 0.64, 0.67, 0.54, 0.66,
    0.72, 0.61, 0.65, 0.68,
]

# US real GDP quarterly annualized growth (%), 1947Q2-2023Q4, 307 obs
# Source: BEA NIPA Table 1.1.1
_GDP_GROWTH: list[float] = [
    -0.8, 6.4, 8.2, 6.7, -5.4, -7.4, 16.2, 15.7, 9.7, 8.2, 0.8, 0.7,
    5.3, 3.2, 7.5, 5.5, 7.2, 3.6, -3.5, -1.2, 7.0, 0.5, 3.8, 5.4,
    -2.4, 2.2, 5.5, 12.0, 5.9, 4.6, -4.2, -6.6, -1.7, 2.0, 8.3, 11.9,
    6.3, 4.6, 0.5, -1.4, 2.7, 9.8, 7.0, 1.9, -4.1, -1.9, 3.1, 11.3,
    -0.5, 1.5, 9.3, 0.8, 9.8, -0.3, 2.6, 2.6, 5.0, 2.8, 5.2, 2.6,
    -0.3, 6.4, 2.2, -0.3, 4.7, 7.5, 3.3, 9.0, 4.2, 8.4, 1.6, -3.0,
    5.4, 10.6, 0.3, -1.9, -2.9, 5.3, -0.2, 3.4, 1.2, 3.9, 7.5, 5.6,
    -1.6, 0.2, 2.8, 1.5, 3.2, 3.0, 8.5, 8.0, 3.4, 1.2, -2.2, 0.3,
    -2.0, -7.9, 5.4, 1.6, 2.8, -2.9, 2.2, 5.3, -3.2, 0.4, 6.1, 2.7,
    -0.7, 6.0, 1.1, 5.6, 1.6, 1.1, -4.6, -8.0, -1.6, 2.7, 3.4, 8.6,
    -3.7, 6.7, 3.9, 3.5, 2.3, 4.5, 3.2, 9.3, 1.4, 8.0, -1.9, 2.0,
    0.0, 0.2, 4.3, 4.3, 2.0, 3.6, 5.4, 4.0, 3.6, 2.7, 3.5, 2.1,
    3.3, 4.3, 3.9, 2.3, 4.8, 5.3, 1.1, 0.2, 3.4, 4.6, 3.8, 3.3,
    2.2, 3.2, 4.0, 4.4, 0.3, 1.0, 3.7, 2.5, 3.5, 3.7, 2.6, 7.2,
    4.4, 1.2, 5.8, 6.3, 4.7, 4.3, 4.5, 2.9, 1.7, -0.2, 1.1, 2.3,
    3.6, 0.5, 0.6, 2.2, 3.6, 3.6, 0.2, 2.2, 3.8, 3.8, 7.5, 4.3,
    2.5, 2.3, 3.5, 3.3, 4.1, 2.6, 3.5, 4.0, 2.4, 5.0, 3.0, 1.5,
    3.7, 4.5, 4.3, 4.1, 4.7, 3.0, 0.6, -1.6, 1.0, 2.7, 3.5, -0.2,
    1.5, 2.6, 3.6, 2.2, 3.1, 0.2, 3.4, 0.6, 1.5, 2.2, 3.2, 1.4,
    2.7, 3.5, 0.4, 2.1, 1.4, 3.2, 3.2, 3.2, 0.4, 3.4, 0.1, 1.5,
    0.5, 1.3, 3.2, 3.2, 3.8, -0.5, 1.3, 3.2, 0.1, -2.7, 2.1, -6.0,
    -0.7, -0.7, -5.4, -6.4, -8.5, -0.5, 1.3, 3.9, 1.7, 2.5, 2.9, 2.4,
    0.5, 1.3, 3.2, 4.6, 0.4, 2.8, 1.3, -1.5, 2.7, 0.5, 2.0, 0.1,
    3.2, 2.2, 2.6, 2.6, 1.5, 3.2, 5.2, 1.0, 2.0, 2.3, 0.6, 3.9,
    2.0, 1.6, 2.7, 3.2, 1.2, 3.4, 3.5, 2.6, 1.5, 3.5, 2.3, 3.2,
    -5.1, -31.2, 33.8, 4.5, 6.3, 6.7, 7.0, 2.7, -1.6, -0.6, 2.7, 3.2,
    2.2, 2.1, 4.9, 3.3, 5.2, 3.2, 1.6,
]
