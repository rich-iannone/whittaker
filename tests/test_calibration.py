"""Tests for sigma calibration in quantile GAMs."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.calibration import calibrate_sigma
from whittaker.families.quantile import QuantileFamily
from whittaker.gam import GAM

_GRID = [0.05, 0.15, 0.5]


@pytest.fixture()
def sinusoidal_data():
    rng = np.random.default_rng(23)
    n = 150
    x = np.linspace(0, 2 * np.pi, n)
    y = np.sin(x) + rng.normal(0, 0.3, n)
    return {"x": x, "y": y}


class TestCalibrateSigma:
    def test_returns_positive_float(self, sinusoidal_data):
        sigma = calibrate_sigma(
            "y ~ s(x)",
            sinusoidal_data,
            tau=0.5,
            n_folds=3,
            sigma_values=_GRID,
            seed=23,
        )
        assert isinstance(sigma, float)
        assert sigma > 0

    def test_median_coverage_near_half(self, sinusoidal_data):
        sigma = calibrate_sigma(
            "y ~ s(x)",
            sinusoidal_data,
            tau=0.5,
            n_folds=3,
            sigma_values=_GRID,
            seed=23,
        )
        model = GAM("y ~ s(x)", family=QuantileFamily(tau=0.5, sigma=sigma))
        model.fit(sinusoidal_data)
        pred = model.predict(sinusoidal_data).values
        coverage = np.mean(sinusoidal_data["y"] <= pred)
        assert 0.30 < coverage < 0.70

    def test_quantile_ordering(self, sinusoidal_data):
        preds = []
        for tau in [0.25, 0.75]:
            sigma = calibrate_sigma(
                "y ~ s(x)",
                sinusoidal_data,
                tau=tau,
                n_folds=3,
                sigma_values=_GRID,
                seed=23,
            )
            model = GAM("y ~ s(x)", family=QuantileFamily(tau=tau, sigma=sigma))
            model.fit(sinusoidal_data)
            preds.append(model.predict(sinusoidal_data).values)
        assert np.mean(preds[0]) < np.mean(preds[1])

    def test_custom_sigma_values(self, sinusoidal_data):
        grid = [0.05, 0.1, 0.5, 1.0]
        sigma = calibrate_sigma(
            "y ~ s(x)",
            sinusoidal_data,
            tau=0.5,
            sigma_values=grid,
            n_folds=3,
            seed=23,
        )
        assert sigma in grid

    def test_deterministic_with_seed(self, sinusoidal_data):
        s1 = calibrate_sigma(
            "y ~ s(x)",
            sinusoidal_data,
            tau=0.5,
            n_folds=3,
            sigma_values=_GRID,
            seed=23,
        )
        s2 = calibrate_sigma(
            "y ~ s(x)",
            sinusoidal_data,
            tau=0.5,
            n_folds=3,
            sigma_values=_GRID,
            seed=23,
        )
        assert s1 == s2
