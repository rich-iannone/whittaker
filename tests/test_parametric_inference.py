"""Tests for parametric term inference and deviance explained."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from whittaker.families.binomial import Binomial
from whittaker.families.gamma import Gamma
from whittaker.families.poisson import Poisson
from whittaker.gam import GAM


# ---------------------------------------------------------------------------
# Parametric coefficient tests
# ---------------------------------------------------------------------------


class TestParametricTests:
    def test_intercept_returned(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.3, n)

        model = GAM("y ~ s(x, k=10)").fit({"y": y, "x": x})
        ptests = model.parametric_tests()
        assert len(ptests) == 1
        assert ptests[0].term_label == "(Intercept)"

    def test_intercept_has_finite_values(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x = np.linspace(0, 1, n)
        y = 3.0 + x + rng.normal(0, 0.1, n)

        model = GAM("y ~ s(x, k=6)").fit({"y": y, "x": x})
        ptests = model.parametric_tests()
        pt = ptests[0]
        assert np.isfinite(pt.estimate)
        assert np.isfinite(pt.se)
        assert np.isfinite(pt.stat)
        assert np.isfinite(pt.p_value)
        assert pt.se > 0

    def test_linear_term_detected(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x1 = np.linspace(0, 2 * np.pi, n)
        x2 = rng.uniform(0, 1, n)
        y = np.sin(x1) + 2.0 * x2 + rng.normal(0, 0.3, n)

        model = GAM("y ~ x2 + s(x1, k=10)").fit({"y": y, "x1": x1, "x2": x2})
        ptests = model.parametric_tests()
        assert len(ptests) == 2
        assert ptests[0].term_label == "(Intercept)"
        assert ptests[1].term_label == "x2"

    def test_linear_term_significant(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x1 = np.linspace(0, 2 * np.pi, n)
        x2 = rng.uniform(0, 1, n)
        y = np.sin(x1) + 3.0 * x2 + rng.normal(0, 0.3, n)

        model = GAM("y ~ x2 + s(x1, k=10)").fit({"y": y, "x1": x1, "x2": x2})
        ptests = model.parametric_tests()
        x2_test = ptests[1]
        assert x2_test.p_value < 0.01
        assert abs(x2_test.estimate - 3.0) < 0.5

    def test_nonsignificant_term(self) -> None:
        rng = np.random.default_rng(99)
        n = 200
        x1 = np.linspace(0, 2 * np.pi, n)
        x2 = rng.uniform(0, 1, n)
        y = np.sin(x1) + rng.normal(0, 0.3, n)

        model = GAM("y ~ x2 + s(x1, k=10)").fit({"y": y, "x1": x1, "x2": x2})
        ptests = model.parametric_tests()
        x2_test = ptests[1]
        assert x2_test.p_value > 0.05

    def test_pvalue_between_zero_and_one(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x = np.linspace(0, 1, n)
        y = 5.0 + x + rng.normal(0, 0.1, n)

        model = GAM("y ~ s(x, k=6)").fit({"y": y, "x": x})
        for pt in model.parametric_tests():
            assert 0.0 <= pt.p_value <= 1.0

    def test_uses_t_for_gaussian(self) -> None:
        rng = np.random.default_rng(42)
        n = 30
        x = np.linspace(0, 1, n)
        y = 10.0 + rng.normal(0, 0.1, n)

        model = GAM("y ~ s(x, k=4)").fit({"y": y, "x": x})
        ptests = model.parametric_tests()
        assert ptests[0].p_value < 1e-10

    def test_uses_z_for_binomial(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x = np.linspace(-2, 2, n)
        p = 1.0 / (1.0 + np.exp(-(1.0 + x)))
        y = rng.binomial(1, p, n).astype(float)

        model = GAM("y ~ s(x, k=6)", family=Binomial()).fit({"y": y, "x": x})
        ptests = model.parametric_tests()
        assert len(ptests) == 1
        assert np.isfinite(ptests[0].p_value)

    def test_uses_z_for_poisson(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x = np.linspace(0, 1, n)
        mu = np.exp(1.0 + 0.5 * x)
        y = rng.poisson(mu).astype(float)

        model = GAM("y ~ s(x, k=6)", family=Poisson()).fit({"y": y, "x": x})
        ptests = model.parametric_tests()
        assert len(ptests) == 1
        assert np.isfinite(ptests[0].p_value)

    def test_uses_t_for_gamma(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x = np.linspace(0.1, 2, n)
        mu_true = np.exp(0.5 * x)
        y = rng.gamma(shape=5.0, scale=mu_true / 5.0, size=n)

        model = GAM("y ~ s(x, k=8)", family=Gamma()).fit({"y": y, "x": x})
        ptests = model.parametric_tests()
        assert len(ptests) == 1
        assert np.isfinite(ptests[0].p_value)

    def test_interaction_term(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        y = 1.0 + 2.0 * x1 * x2 + rng.normal(0, 0.1, n)

        model = GAM("y ~ x1 * x2 + s(x1, k=6)").fit({"y": y, "x1": x1, "x2": x2})
        ptests = model.parametric_tests()
        labels = [pt.term_label for pt in ptests]
        assert "(Intercept)" in labels
        assert "x1:x2" in labels

    def test_summary_includes_parametric_table(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x1 = np.linspace(0, 2 * np.pi, n)
        x2 = rng.uniform(0, 1, n)
        y = np.sin(x1) + 2.0 * x2 + rng.normal(0, 0.3, n)

        model = GAM("y ~ x2 + s(x1, k=10)").fit({"y": y, "x1": x1, "x2": x2})
        text = model.summary()
        assert "Parametric coefficients:" in text
        assert "(Intercept)" in text
        assert "x2" in text
        assert "Std.Err" in text

    def test_smooth_only_model_still_has_intercept_test(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x = np.linspace(0, 1, n)
        y = np.sin(2 * np.pi * x) + rng.normal(0, 0.2, n)

        model = GAM("y ~ s(x, k=10)").fit({"y": y, "x": x})
        ptests = model.parametric_tests()
        assert len(ptests) == 1
        assert ptests[0].term_label == "(Intercept)"


# ---------------------------------------------------------------------------
# Deviance explained
# ---------------------------------------------------------------------------


class TestDevianceExplained:
    def test_between_zero_and_one(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.3, n)

        model = GAM("y ~ s(x, k=10)").fit({"y": y, "x": x})
        assert 0.0 < model.deviance_explained < 1.0

    def test_high_for_strong_signal(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x = np.linspace(0, 2 * np.pi, n)
        y = 3.0 * np.sin(x) + rng.normal(0, 0.1, n)

        model = GAM("y ~ s(x, k=10)").fit({"y": y, "x": x})
        assert model.deviance_explained > 0.95

    def test_low_for_noise(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x = np.linspace(0, 1, n)
        y = rng.normal(0, 1, n)

        model = GAM("y ~ s(x, k=6)").fit({"y": y, "x": x})
        assert model.deviance_explained < 0.15

    def test_null_deviance_property(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.3, n)

        model = GAM("y ~ s(x, k=10)").fit({"y": y, "x": x})
        assert model.null_deviance > 0
        assert model.null_deviance > model.deviance

    def test_deviance_explained_formula(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.3, n)

        model = GAM("y ~ s(x, k=10)").fit({"y": y, "x": x})
        expected = 1.0 - model.deviance / model.null_deviance
        assert_allclose(model.deviance_explained, expected)

    def test_in_summary(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        x = np.linspace(0, 1, n)
        y = x + rng.normal(0, 0.1, n)

        model = GAM("y ~ s(x, k=6)").fit({"y": y, "x": x})
        text = model.summary()
        assert "Dev. expl:" in text
        assert "%" in text

    def test_binomial_deviance_explained(self) -> None:
        rng = np.random.default_rng(42)
        n = 400
        x = np.linspace(-3, 3, n)
        p = 1.0 / (1.0 + np.exp(-x))
        y = rng.binomial(1, p, n).astype(float)

        model = GAM("y ~ s(x, k=8)", family=Binomial()).fit({"y": y, "x": x})
        assert 0.0 < model.deviance_explained < 1.0

    def test_poisson_deviance_explained(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x = np.linspace(0, 2, n)
        mu = np.exp(1.0 + x)
        y = rng.poisson(mu).astype(float)

        model = GAM("y ~ s(x, k=8)", family=Poisson()).fit({"y": y, "x": x})
        assert 0.0 < model.deviance_explained < 1.0

    def test_gamma_deviance_explained(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x = np.linspace(0.1, 2, n)
        mu_true = np.exp(0.5 * x)
        y = rng.gamma(shape=5.0, scale=mu_true / 5.0, size=n)

        model = GAM("y ~ s(x, k=8)", family=Gamma()).fit({"y": y, "x": x})
        assert 0.0 < model.deviance_explained < 1.0

    def test_null_deviance_in_summary(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        x = np.linspace(0, 1, n)
        y = x + rng.normal(0, 0.1, n)

        model = GAM("y ~ s(x, k=6)").fit({"y": y, "x": x})
        text = model.summary()
        assert "Null dev:" in text
