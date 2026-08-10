"""Tests for AIC/BIC and Gamma family GAM fitting."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from whittaker.families.binomial import Binomial
from whittaker.families.gamma import Gamma
from whittaker.families.gaussian import Gaussian
from whittaker.families.poisson import Poisson
from whittaker.gam import GAM


class TestAICBIC:
    def test_gaussian_aic_bic_exist(self) -> None:
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.3, n)

        model = GAM("y ~ s(x, k=10)").fit({"y": y, "x": x})
        assert np.isfinite(model.aic)
        assert np.isfinite(model.bic)

    def test_aic_less_than_bic_for_large_n(self) -> None:
        rng = np.random.default_rng(23)
        n = 500
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.3, n)

        model = GAM("y ~ s(x, k=10)").fit({"y": y, "x": x})
        # BIC penalizes more than AIC when n > e^2 ~ 7.4
        assert model.aic < model.bic

    def test_aic_formula(self) -> None:
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.3, n)

        model = GAM("y ~ s(x, k=10)").fit({"y": y, "x": x})
        fam = Gaussian()
        ll = fam.log_likelihood(y, model.fitted_values, model.scale)
        expected_aic = -2.0 * ll + 2.0 * model.edf_total
        assert_allclose(model.aic, expected_aic, rtol=1e-10)

    def test_bic_formula(self) -> None:
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.3, n)

        model = GAM("y ~ s(x, k=10)").fit({"y": y, "x": x})
        fam = Gaussian()
        ll = fam.log_likelihood(y, model.fitted_values, model.scale)
        expected_bic = -2.0 * ll + np.log(n) * model.edf_total
        assert_allclose(model.bic, expected_bic, rtol=1e-10)

    def test_simpler_model_lower_aic(self) -> None:
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(0, 1, n)
        y = 2.0 * x + rng.normal(0, 0.1, n)

        model_simple = GAM("y ~ s(x, k=4)").fit({"y": y, "x": x})
        model_complex = GAM("y ~ s(x, k=20)").fit({"y": y, "x": x})
        # For a linear signal, the simpler model should have lower AIC
        assert model_simple.aic <= model_complex.aic + 5

    def test_aic_bic_in_summary(self) -> None:
        rng = np.random.default_rng(23)
        n = 100
        x = np.linspace(0, 1, n)
        y = x + rng.normal(0, 0.1, n)

        model = GAM("y ~ s(x, k=6)").fit({"y": y, "x": x})
        text = model.summary()
        assert "AIC:" in text
        assert "BIC:" in text

    def test_poisson_aic_bic(self) -> None:
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(0, 1, n)
        mu = np.exp(0.5 * x)
        y = rng.poisson(mu).astype(float)

        model = GAM("y ~ s(x, k=6)", family=Poisson()).fit({"y": y, "x": x})
        assert np.isfinite(model.aic)
        assert np.isfinite(model.bic)

    def test_binomial_aic_bic(self) -> None:
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(-2, 2, n)
        p = 1.0 / (1.0 + np.exp(-x))
        y = rng.binomial(1, p, n).astype(float)

        model = GAM("y ~ s(x, k=6)", family=Binomial()).fit({"y": y, "x": x})
        assert np.isfinite(model.aic)
        assert np.isfinite(model.bic)


class TestGammaGAM:
    def test_gamma_fit(self) -> None:
        rng = np.random.default_rng(23)
        n = 300
        x = np.linspace(0.1, 2, n)
        mu_true = np.exp(0.5 * x)
        y = rng.gamma(shape=5.0, scale=mu_true / 5.0, size=n)

        model = GAM("y ~ s(x, k=10)", family=Gamma()).fit({"y": y, "x": x})
        assert model.is_fitted
        assert model.deviance > 0

    def test_gamma_recovery(self) -> None:
        rng = np.random.default_rng(23)
        n = 500
        x = np.linspace(0.1, 2, n)
        mu_true = np.exp(0.5 * x)
        y = rng.gamma(shape=10.0, scale=mu_true / 10.0, size=n)

        model = GAM("y ~ s(x, k=10)", family=Gamma()).fit({"y": y, "x": x})
        rmse = np.sqrt(np.mean((model.fitted_values - mu_true) ** 2))
        assert rmse < 0.3

    def test_gamma_predict(self) -> None:
        rng = np.random.default_rng(23)
        n = 300
        x = np.linspace(0.1, 2, n)
        mu_true = np.exp(0.5 * x)
        y = rng.gamma(shape=5.0, scale=mu_true / 5.0, size=n)

        model = GAM("y ~ s(x, k=10)", family=Gamma()).fit({"y": y, "x": x})
        pred = model.predict({"x": np.array([1.0])}, se=True)
        assert pred.values[0] > 0
        assert pred.se is not None
        assert pred.se[0] > 0

    def test_gamma_aic_bic(self) -> None:
        rng = np.random.default_rng(23)
        n = 300
        x = np.linspace(0.1, 2, n)
        mu_true = np.exp(0.5 * x)
        y = rng.gamma(shape=5.0, scale=mu_true / 5.0, size=n)

        model = GAM("y ~ s(x, k=10)", family=Gamma()).fit({"y": y, "x": x})
        assert np.isfinite(model.aic)
        assert np.isfinite(model.bic)

    def test_gamma_scale_not_one(self) -> None:
        rng = np.random.default_rng(23)
        n = 300
        x = np.linspace(0.1, 2, n)
        mu_true = np.exp(0.5 * x)
        y = rng.gamma(shape=5.0, scale=mu_true / 5.0, size=n)

        model = GAM("y ~ s(x, k=10)", family=Gamma()).fit({"y": y, "x": x})
        assert model.scale != 1.0
        # shape=5 → scale~0.2
        assert 0.05 < model.scale < 0.6

    def test_gamma_with_reml(self) -> None:
        rng = np.random.default_rng(23)
        n = 300
        x = np.linspace(0.1, 2, n)
        mu_true = np.exp(0.5 * x)
        y = rng.gamma(shape=5.0, scale=mu_true / 5.0, size=n)

        model = GAM("y ~ s(x, k=10)", family=Gamma()).fit({"y": y, "x": x}, method="REML")
        assert model.is_fitted

    def test_gamma_two_smooths(self) -> None:
        rng = np.random.default_rng(23)
        n = 400
        x1 = np.linspace(0.1, 2, n)
        x2 = np.linspace(0, 1, n)
        mu_true = np.exp(0.5 * x1 + 0.3 * x2)
        y = rng.gamma(shape=5.0, scale=mu_true / 5.0, size=n)

        model = GAM("y ~ s(x1, k=8) + s(x2, k=8)", family=Gamma()).fit({"y": y, "x1": x1, "x2": x2})
        assert len(model.edf) == 2
        assert model.is_fitted

    def test_gamma_summary(self) -> None:
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(0.1, 2, n)
        mu_true = np.exp(0.5 * x)
        y = rng.gamma(shape=5.0, scale=mu_true / 5.0, size=n)

        model = GAM("y ~ s(x, k=8)", family=Gamma()).fit({"y": y, "x": x})
        text = model.summary()
        assert "Gamma" in text
        assert "AIC:" in text
        assert "BIC:" in text

    def test_gamma_repr(self) -> None:
        model = GAM("y ~ s(x)", family=Gamma())
        assert "Gamma" in repr(model)
