"""Response distribution families for Whittaker GAMs."""

from __future__ import annotations

from whittaker.families.base import Family
from whittaker.families.binomial import Binomial
from whittaker.families.gamma import Gamma
from whittaker.families.gaussian import Gaussian
from whittaker.families.negative_binomial import NegativeBinomial
from whittaker.families.poisson import Poisson

__all__ = [
    "Binomial",
    "Family",
    "Gamma",
    "Gaussian",
    "NegativeBinomial",
    "Poisson",
]
