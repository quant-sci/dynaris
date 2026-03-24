"""Forecasting: multi-step-ahead predictions with uncertainty."""

from dynaris.forecast.forecast import (
    ForecastResult,
    confidence_bands,
    fit_batch,
    forecast,
    forecast_batch,
    forecast_from_filter,
    forecast_from_smoother,
)

__all__ = [
    "ForecastResult",
    "confidence_bands",
    "fit_batch",
    "forecast",
    "forecast_batch",
    "forecast_from_filter",
    "forecast_from_smoother",
]
