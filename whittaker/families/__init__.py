"""Response distribution families for Whittaker GAMs."""

from __future__ import annotations

from whittaker.families.base import Family
from whittaker.families.beta import Beta
from whittaker.families.beta_ls import BetaLS
from whittaker.families.binomial import Binomial
from whittaker.families.gamlss_base import GAMLSSFamily
from whittaker.families.gamma import Gamma
from whittaker.families.gamma_ls import GammaLS
from whittaker.families.gaussian import Gaussian
from whittaker.families.gaussian_ls import GaussianLS
from whittaker.families.inverse_gaussian import InverseGaussian
from whittaker.families.negative_binomial import NegativeBinomial
from whittaker.families.ordered_categorical import OrderedCategorical
from whittaker.families.poisson import Poisson
from whittaker.families.quantile import QuantileFamily
from whittaker.families.tweedie import Tweedie
from whittaker.families.tweedie_estimated import TweedieEstimated, tw
from whittaker.families.zero_inflated import ZeroInflatedNegativeBinomial, ZeroInflatedPoisson

__all__ = [
    "Beta",
    "BetaLS",
    "Binomial",
    "Family",
    "GAMLSSFamily",
    "Gamma",
    "GammaLS",
    "Gaussian",
    "GaussianLS",
    "InverseGaussian",
    "NegativeBinomial",
    "OrderedCategorical",
    "Poisson",
    "QuantileFamily",
    "Tweedie",
    "TweedieEstimated",
    "tw",
    "ZeroInflatedNegativeBinomial",
    "ZeroInflatedPoisson",
]
