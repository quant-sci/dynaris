"""Parameter estimation: MLE, EM, Bayesian, and model diagnostics."""

from dynaris.estimation.bayesian import BayesianResult, fit_bayesian
from dynaris.estimation.comparison import compute_loo, compute_waic, to_arviz
from dynaris.estimation.dfm import DFMResult, fit_dfm_em
from dynaris.estimation.diagnostics import acf, ljung_box, pacf, standardized_residuals
from dynaris.estimation.em import EMResult, fit_em
from dynaris.estimation.mle import MLEResult, fit_mle
from dynaris.estimation.model_selection import switching_aic, switching_bic
from dynaris.estimation.predictive import (
    posterior_predictive_check,
    posterior_predictive_forecast,
    prior_predictive,
)
from dynaris.estimation.priors import (
    combine_priors,
    half_normal_log_prior,
    inverse_gamma_log_prior,
    normal_log_prior,
)
from dynaris.estimation.transforms import inverse_softplus, softplus

__all__ = [
    "BayesianResult",
    "DFMResult",
    "EMResult",
    "MLEResult",
    "acf",
    "combine_priors",
    "compute_loo",
    "compute_waic",
    "fit_bayesian",
    "fit_dfm_em",
    "fit_em",
    "fit_mle",
    "half_normal_log_prior",
    "inverse_gamma_log_prior",
    "inverse_softplus",
    "ljung_box",
    "normal_log_prior",
    "pacf",
    "posterior_predictive_check",
    "posterior_predictive_forecast",
    "prior_predictive",
    "softplus",
    "standardized_residuals",
    "switching_aic",
    "switching_bic",
    "to_arviz",
]
