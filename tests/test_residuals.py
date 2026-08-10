"""Tests for residual types."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from whittaker.families.binomial import Binomial
from whittaker.families.gamma import Gamma
from whittaker.families.gaussian import Gaussian
from whittaker.families.poisson import Poisson
from whittaker.gam import GAM

# ---------------------------------------------------------------------------
# Response residuals
# ---------------------------------------------------------------------------


class TestResponseResiduals:
    def test_matches_property(self) -> None:
        rng = np.random.default_rng(23)
        n = 100
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.3, n)

        model = GAM("y ~ s(x, k=8)").fit({"y": y, "x": x})
        assert_allclose(model.get_residuals("response"), model.residuals)

    def test_sums_near_zero_gaussian(self) -> None:
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(0, 1, n)
        y = x + rng.normal(0, 0.1, n)

        model = GAM("y ~ s(x, k=8)").fit({"y": y, "x": x})
        r = model.get_residuals("response")
        assert abs(np.mean(r)) < 0.05

    def test_equals_y_minus_mu(self) -> None:
        rng = np.random.default_rng(23)
        n = 100
        x = np.linspace(0, 1, n)
        y = x + rng.normal(0, 0.1, n)

        model = GAM("y ~ s(x, k=6)").fit({"y": y, "x": x})
        expected = y - model.fitted_values
        assert_allclose(model.get_residuals("response"), expected)


# ---------------------------------------------------------------------------
# Pearson residuals
# ---------------------------------------------------------------------------


class TestPearsonResiduals:
    def test_gaussian_same_as_response(self) -> None:
        rng = np.random.default_rng(23)
        n = 100
        x = np.linspace(0, 1, n)
        y = x + rng.normal(0, 0.1, n)

        model = GAM("y ~ s(x, k=6)").fit({"y": y, "x": x})
        assert_allclose(
            model.get_residuals("pearson"),
            model.get_residuals("response"),
        )

    def test_poisson_different_from_response(self) -> None:
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(0, 2, n)
        mu = np.exp(1.0 + x)
        y = rng.poisson(mu).astype(float)

        model = GAM("y ~ s(x, k=8)", family=Poisson()).fit({"y": y, "x": x})
        pearson = model.get_residuals("pearson")
        response = model.get_residuals("response")
        assert not np.allclose(pearson, response)

    def test_poisson_formula(self) -> None:
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(0, 2, n)
        mu = np.exp(0.5 * x)
        y = rng.poisson(mu).astype(float)

        model = GAM("y ~ s(x, k=8)", family=Poisson()).fit({"y": y, "x": x})
        pearson = model.get_residuals("pearson")
        fitted = model.fitted_values
        expected = (y - fitted) / np.sqrt(fitted)
        assert_allclose(pearson, expected, rtol=1e-10)

    def test_binomial_formula(self) -> None:
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(-2, 2, n)
        p = 1.0 / (1.0 + np.exp(-x))
        y = rng.binomial(1, p, n).astype(float)

        model = GAM("y ~ s(x, k=6)", family=Binomial()).fit({"y": y, "x": x})
        pearson = model.get_residuals("pearson")
        fitted = model.fitted_values
        expected = (y - fitted) / np.sqrt(fitted * (1.0 - fitted))
        assert_allclose(pearson, expected, rtol=1e-10)

    def test_gamma_scales_by_mu(self) -> None:
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(0.1, 2, n)
        mu_true = np.exp(0.5 * x)
        y = rng.gamma(shape=5.0, scale=mu_true / 5.0, size=n)

        model = GAM("y ~ s(x, k=8)", family=Gamma()).fit({"y": y, "x": x})
        pearson = model.get_residuals("pearson")
        fitted = model.fitted_values
        expected = (y - fitted) / fitted
        assert_allclose(pearson, expected, rtol=1e-10)

    def test_variance_near_scale_for_poisson(self) -> None:
        rng = np.random.default_rng(23)
        n = 500
        x = np.linspace(0, 2, n)
        mu = np.exp(1.0 + 0.5 * x)
        y = rng.poisson(mu).astype(float)

        model = GAM("y ~ s(x, k=8)", family=Poisson()).fit({"y": y, "x": x})
        pearson = model.get_residuals("pearson")
        # For well-specified Poisson, Pearson residuals should have variance ~ 1
        assert 0.5 < np.var(pearson) < 2.0


# ---------------------------------------------------------------------------
# Deviance residuals
# ---------------------------------------------------------------------------


class TestDevianceResiduals:
    def test_gaussian_matches_response(self) -> None:
        rng = np.random.default_rng(23)
        n = 100
        x = np.linspace(0, 1, n)
        y = x + rng.normal(0, 0.1, n)

        model = GAM("y ~ s(x, k=6)").fit({"y": y, "x": x})
        dev_r = model.get_residuals("deviance")
        resp_r = model.get_residuals("response")
        assert_allclose(dev_r, resp_r)

    def test_sign_matches_response(self) -> None:
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(0, 2, n)
        mu = np.exp(0.5 * x)
        y = rng.poisson(mu).astype(float)

        model = GAM("y ~ s(x, k=8)", family=Poisson()).fit({"y": y, "x": x})
        dev_r = model.get_residuals("deviance")
        resp_r = model.get_residuals("response")
        nonzero = np.abs(resp_r) > 1e-10
        assert np.all(np.sign(dev_r[nonzero]) == np.sign(resp_r[nonzero]))

    def test_squared_sum_equals_deviance(self) -> None:
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(0, 2, n)
        mu = np.exp(0.5 * x)
        y = rng.poisson(mu).astype(float)

        model = GAM("y ~ s(x, k=8)", family=Poisson()).fit({"y": y, "x": x})
        dev_r = model.get_residuals("deviance")
        assert_allclose(np.sum(dev_r**2), model.deviance, rtol=1e-10)

    def test_binomial_deviance_residuals(self) -> None:
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(-2, 2, n)
        p = 1.0 / (1.0 + np.exp(-x))
        y = rng.binomial(1, p, n).astype(float)

        model = GAM("y ~ s(x, k=6)", family=Binomial()).fit({"y": y, "x": x})
        dev_r = model.get_residuals("deviance")
        assert np.all(np.isfinite(dev_r))
        assert_allclose(np.sum(dev_r**2), model.deviance, rtol=1e-10)

    def test_gamma_deviance_residuals(self) -> None:
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(0.1, 2, n)
        mu_true = np.exp(0.5 * x)
        y = rng.gamma(shape=5.0, scale=mu_true / 5.0, size=n)

        model = GAM("y ~ s(x, k=8)", family=Gamma()).fit({"y": y, "x": x})
        dev_r = model.get_residuals("deviance")
        assert np.all(np.isfinite(dev_r))
        assert_allclose(np.sum(dev_r**2), model.deviance, rtol=1e-10)

    def test_default_type_is_deviance(self) -> None:
        rng = np.random.default_rng(23)
        n = 100
        x = np.linspace(0, 1, n)
        y = x + rng.normal(0, 0.1, n)

        model = GAM("y ~ s(x, k=6)").fit({"y": y, "x": x})
        assert_allclose(model.get_residuals(), model.get_residuals("deviance"))


# ---------------------------------------------------------------------------
# Working residuals
# ---------------------------------------------------------------------------


class TestWorkingResiduals:
    def test_gaussian_same_as_response(self) -> None:
        rng = np.random.default_rng(23)
        n = 100
        x = np.linspace(0, 1, n)
        y = x + rng.normal(0, 0.1, n)

        model = GAM("y ~ s(x, k=6)").fit({"y": y, "x": x})
        assert_allclose(
            model.get_residuals("working"),
            model.get_residuals("response"),
        )

    def test_poisson_formula(self) -> None:
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(0, 2, n)
        mu = np.exp(0.5 * x)
        y = rng.poisson(mu).astype(float)

        model = GAM("y ~ s(x, k=8)", family=Poisson()).fit({"y": y, "x": x})
        working = model.get_residuals("working")
        fitted = model.fitted_values
        expected = (y - fitted) / fitted
        assert_allclose(working, expected, rtol=1e-10)

    def test_binomial_working_residuals(self) -> None:
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(-2, 2, n)
        p = 1.0 / (1.0 + np.exp(-x))
        y = rng.binomial(1, p, n).astype(float)

        model = GAM("y ~ s(x, k=6)", family=Binomial()).fit({"y": y, "x": x})
        working = model.get_residuals("working")
        assert np.all(np.isfinite(working))


# ---------------------------------------------------------------------------
# Unit deviance
# ---------------------------------------------------------------------------


class TestUnitDeviance:
    def test_gaussian_unit_deviance(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.5, 1.5, 3.5])
        fam = Gaussian()
        assert_allclose(fam.unit_deviance(y, mu), (y - mu) ** 2)

    def test_gaussian_total(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.5, 1.5, 3.5])
        fam = Gaussian()
        assert_allclose(np.sum(fam.unit_deviance(y, mu)), fam.deviance(y, mu))

    def test_poisson_total(self) -> None:
        y = np.array([0.0, 1.0, 3.0, 5.0])
        mu = np.array([0.5, 1.5, 2.0, 4.0])
        fam = Poisson()
        assert_allclose(np.sum(fam.unit_deviance(y, mu)), fam.deviance(y, mu))

    def test_binomial_total(self) -> None:
        y = np.array([0.0, 1.0, 1.0, 0.0])
        mu = np.array([0.3, 0.7, 0.9, 0.1])
        fam = Binomial()
        assert_allclose(np.sum(fam.unit_deviance(y, mu)), fam.deviance(y, mu))

    def test_gamma_total(self) -> None:
        y = np.array([1.0, 2.0, 3.0, 4.0])
        mu = np.array([1.5, 1.5, 3.5, 3.5])
        fam = Gamma()
        assert_allclose(np.sum(fam.unit_deviance(y, mu)), fam.deviance(y, mu))

    def test_poisson_nonnegative(self) -> None:
        y = np.array([0.0, 1.0, 5.0, 10.0])
        mu = np.array([1.0, 2.0, 3.0, 8.0])
        fam = Poisson()
        assert np.all(fam.unit_deviance(y, mu) >= -1e-15)

    def test_binomial_zero_at_truth(self) -> None:
        y = np.array([0.0, 1.0])
        mu = np.array([0.0 + 1e-15, 1.0 - 1e-15])
        fam = Binomial()
        ud = fam.unit_deviance(y, mu)
        assert_allclose(ud, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestResidualErrors:
    def test_unknown_type_raises(self) -> None:
        rng = np.random.default_rng(23)
        n = 50
        x = np.linspace(0, 1, n)
        y = x + rng.normal(0, 0.1, n)

        model = GAM("y ~ s(x, k=4)").fit({"y": y, "x": x})
        with pytest.raises(ValueError, match="Unknown residual type"):
            model.get_residuals("bad_type")

    def test_unfitted_raises(self) -> None:
        model = GAM("y ~ s(x)")
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.get_residuals("deviance")

    def test_case_insensitive(self) -> None:
        rng = np.random.default_rng(23)
        n = 50
        x = np.linspace(0, 1, n)
        y = x + rng.normal(0, 0.1, n)

        model = GAM("y ~ s(x, k=4)").fit({"y": y, "x": x})
        assert_allclose(
            model.get_residuals("Deviance"),
            model.get_residuals("deviance"),
        )
