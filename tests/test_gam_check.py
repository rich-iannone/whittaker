"""Tests for gam_check() diagnostic method."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.families import Gaussian, Poisson
from whittaker.gam import GAM, GamCheckResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fitted_gaussian():
    rng = np.random.default_rng(23)
    n = 200
    x = rng.uniform(0, 2 * np.pi, n)
    y = np.sin(x) + rng.normal(0, 0.3, n)
    data = {"x": x, "y": y}
    gam = GAM("y ~ s(x)", family=Gaussian())
    gam.fit(data, method="REML")
    return gam, data


@pytest.fixture()
def fitted_two_smooth():
    rng = np.random.default_rng(23)
    n = 300
    x1 = rng.uniform(0, 2 * np.pi, n)
    x2 = rng.uniform(0, 2 * np.pi, n)
    y = np.sin(x1) + 0.5 * np.cos(x2) + rng.normal(0, 0.3, n)
    data = {"x1": x1, "x2": x2, "y": y}
    gam = GAM("y ~ s(x1) + s(x2)", family=Gaussian())
    gam.fit(data, method="REML")
    return gam, data


# ---------------------------------------------------------------------------
# Basic gam_check
# ---------------------------------------------------------------------------


class TestGamCheck:
    def test_returns_result(self, fitted_gaussian):
        gam, _ = fitted_gaussian
        result = gam.gam_check(n_sim=50)
        assert isinstance(result, GamCheckResult)

    def test_deviance_residuals_shape(self, fitted_gaussian):
        gam, data = fitted_gaussian
        result = gam.gam_check(n_sim=50)
        assert result.deviance_residuals.shape == (len(data["y"]),)

    def test_fitted_values_shape(self, fitted_gaussian):
        gam, data = fitted_gaussian
        result = gam.gam_check(n_sim=50)
        assert result.fitted_values.shape == (len(data["y"]),)

    def test_response_shape(self, fitted_gaussian):
        gam, data = fitted_gaussian
        result = gam.gam_check(n_sim=50)
        assert result.response.shape == (len(data["y"]),)

    def test_k_check_results(self, fitted_gaussian):
        gam, _ = fitted_gaussian
        result = gam.gam_check(n_sim=50)
        assert len(result.k_check) == 1
        assert result.k_check[0].k_index > 0

    def test_scalar_fields(self, fitted_gaussian):
        gam, data = fitted_gaussian
        result = gam.gam_check(n_sim=50)
        assert 0 < result.deviance_explained <= 1
        assert result.scale > 0
        assert result.edf_total > 0
        assert result.n_obs == len(data["y"])

    def test_repr(self, fitted_gaussian):
        gam, _ = fitted_gaussian
        result = gam.gam_check(n_sim=50)
        text = repr(result)
        assert "GAM check results" in text
        assert "k_index" in text
        assert "dev.expl" in text


# ---------------------------------------------------------------------------
# Multiple smooths
# ---------------------------------------------------------------------------


class TestGamCheckMultipleSmooths:
    def test_k_check_count(self, fitted_two_smooth):
        gam, _ = fitted_two_smooth
        result = gam.gam_check(n_sim=50)
        assert len(result.k_check) == 2

    def test_repr_two_smooths(self, fitted_two_smooth):
        gam, _ = fitted_two_smooth
        result = gam.gam_check(n_sim=50)
        text = repr(result)
        assert text.count("k_index") == 2


# ---------------------------------------------------------------------------
# Poisson
# ---------------------------------------------------------------------------


class TestGamCheckPoisson:
    def test_poisson(self):
        rng = np.random.default_rng(23)
        n = 200
        x = rng.uniform(0, 2 * np.pi, n)
        y = rng.poisson(np.exp(0.5 * np.sin(x)))
        data = {"x": x, "y": y}
        gam = GAM("y ~ s(x)", family=Poisson())
        gam.fit(data, method="REML")
        result = gam.gam_check(n_sim=50)
        assert isinstance(result, GamCheckResult)
        assert result.scale == 1.0


# ---------------------------------------------------------------------------
# Not fitted
# ---------------------------------------------------------------------------


class TestGamCheckNotFitted:
    def test_raises_not_fitted(self):
        gam = GAM("y ~ s(x)", family=Gaussian())
        with pytest.raises(RuntimeError, match="not been fitted"):
            gam.gam_check()
