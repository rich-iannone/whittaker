"""Tests for Negative Binomial GAM fitting with θ estimation."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from whittaker.families.negative_binomial import NegativeBinomial
from whittaker.gam import GAM


def _simulate_nb(rng, n, x, mu, theta):
    """Simulate NB data: y ~ NB(mu, theta)."""
    p = theta / (mu + theta)
    return rng.negative_binomial(theta, p, size=n).astype(float)


class TestNBGAMFit:
    def test_nb_fit_converges(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x = np.linspace(0, 2, n)
        mu = np.exp(0.5 + 0.5 * x)
        y = _simulate_nb(rng, n, x, mu, theta=3.0)

        model = GAM("y ~ s(x, k=8)", family=NegativeBinomial(theta=1.0)).fit({"y": y, "x": x})
        assert model.is_fitted

    def test_theta_estimated(self) -> None:
        rng = np.random.default_rng(42)
        n = 500
        x = np.linspace(0, 2, n)
        mu = np.exp(0.5 + 0.5 * x)
        true_theta = 3.0
        y = _simulate_nb(rng, n, x, mu, true_theta)

        fam = NegativeBinomial(theta=1.0)
        model = GAM("y ~ s(x, k=10)", family=fam).fit({"y": y, "x": x})
        # θ should be estimated near the true value
        assert 1.0 < fam.theta < 10.0

    def test_theta_recovery_small(self) -> None:
        rng = np.random.default_rng(42)
        n = 1000
        x = np.linspace(0, 2, n)
        mu = np.exp(1.0 + 0.3 * x)
        true_theta = 1.5
        y = _simulate_nb(rng, n, x, mu, true_theta)

        fam = NegativeBinomial(theta=5.0)
        GAM("y ~ s(x, k=10)", family=fam).fit({"y": y, "x": x})
        assert abs(fam.theta - true_theta) / true_theta < 0.5

    def test_theta_recovery_large(self) -> None:
        rng = np.random.default_rng(42)
        n = 1000
        x = np.linspace(0, 2, n)
        mu = np.exp(1.0 + 0.3 * x)
        true_theta = 10.0
        y = _simulate_nb(rng, n, x, mu, true_theta)

        fam = NegativeBinomial(theta=1.0)
        GAM("y ~ s(x, k=10)", family=fam).fit({"y": y, "x": x})
        assert abs(fam.theta - true_theta) / true_theta < 0.5

    def test_mean_recovery(self) -> None:
        rng = np.random.default_rng(42)
        n = 500
        x = np.linspace(0, 2, n)
        mu_true = np.exp(0.5 + 0.5 * x)
        y = _simulate_nb(rng, n, x, mu_true, theta=5.0)

        model = GAM("y ~ s(x, k=10)", family=NegativeBinomial()).fit({"y": y, "x": x})
        rmse = np.sqrt(np.mean((model.fitted_values - mu_true) ** 2))
        assert rmse < 1.0

    def test_predict(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x = np.linspace(0, 2, n)
        mu = np.exp(0.5 * x)
        y = _simulate_nb(rng, n, x, mu, theta=3.0)

        model = GAM("y ~ s(x, k=8)", family=NegativeBinomial()).fit({"y": y, "x": x})
        pred = model.predict({"x": np.array([1.0])}, se=True)
        assert pred.values[0] > 0
        assert pred.se is not None
        assert pred.se[0] > 0

    def test_aic_bic_finite(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x = np.linspace(0, 2, n)
        mu = np.exp(0.5 * x)
        y = _simulate_nb(rng, n, x, mu, theta=3.0)

        model = GAM("y ~ s(x, k=8)", family=NegativeBinomial()).fit({"y": y, "x": x})
        assert np.isfinite(model.aic)
        assert np.isfinite(model.bic)

    def test_deviance_explained(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x = np.linspace(0, 2, n)
        mu = np.exp(1.0 + x)
        y = _simulate_nb(rng, n, x, mu, theta=5.0)

        model = GAM("y ~ s(x, k=8)", family=NegativeBinomial()).fit({"y": y, "x": x})
        assert 0.0 < model.deviance_explained < 1.0

    def test_residual_types(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x = np.linspace(0, 2, n)
        mu = np.exp(0.5 * x)
        y = _simulate_nb(rng, n, x, mu, theta=3.0)

        model = GAM("y ~ s(x, k=8)", family=NegativeBinomial()).fit({"y": y, "x": x})
        for rtype in ("response", "pearson", "deviance", "working"):
            r = model.get_residuals(rtype)
            assert np.all(np.isfinite(r))

    def test_deviance_residuals_sum(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x = np.linspace(0, 2, n)
        mu = np.exp(0.5 * x)
        y = _simulate_nb(rng, n, x, mu, theta=3.0)

        model = GAM("y ~ s(x, k=8)", family=NegativeBinomial()).fit({"y": y, "x": x})
        dev_r = model.get_residuals("deviance")
        assert_allclose(np.sum(dev_r**2), model.deviance, rtol=1e-10)

    def test_two_smooths(self) -> None:
        rng = np.random.default_rng(42)
        n = 400
        x1 = np.linspace(0, 2, n)
        x2 = np.linspace(0, 1, n)
        mu = np.exp(0.5 * x1 + 0.3 * x2)
        y = _simulate_nb(rng, n, x1, mu, theta=3.0)

        model = GAM("y ~ s(x1, k=8) + s(x2, k=6)", family=NegativeBinomial()).fit(
            {"y": y, "x1": x1, "x2": x2}
        )
        assert len(model.edf) == 2

    def test_with_reml(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x = np.linspace(0, 2, n)
        mu = np.exp(0.5 * x)
        y = _simulate_nb(rng, n, x, mu, theta=3.0)

        model = GAM("y ~ s(x, k=8)", family=NegativeBinomial()).fit({"y": y, "x": x}, method="REML")
        assert model.is_fitted

    def test_summary(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x = np.linspace(0, 2, n)
        mu = np.exp(0.5 * x)
        y = _simulate_nb(rng, n, x, mu, theta=3.0)

        model = GAM("y ~ s(x, k=8)", family=NegativeBinomial()).fit({"y": y, "x": x})
        text = model.summary()
        assert "NegativeBinomial" in text

    def test_smooth_tests(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x = np.linspace(0, 2, n)
        mu = np.exp(1.0 + x)
        y = _simulate_nb(rng, n, x, mu, theta=5.0)

        model = GAM("y ~ s(x, k=8)", family=NegativeBinomial()).fit({"y": y, "x": x})
        tests = model.smooth_tests()
        assert len(tests) == 1
        assert tests[0].p_value < 0.05

    def test_nb_better_than_poisson_for_overdispersed(self) -> None:
        rng = np.random.default_rng(42)
        n = 500
        x = np.linspace(0, 2, n)
        mu = np.exp(0.5 + 0.5 * x)
        y = _simulate_nb(rng, n, x, mu, theta=2.0)

        from whittaker.families.poisson import Poisson

        model_pois = GAM("y ~ s(x, k=8)", family=Poisson()).fit({"y": y, "x": x})
        model_nb = GAM("y ~ s(x, k=8)", family=NegativeBinomial()).fit({"y": y, "x": x})
        assert model_nb.aic < model_pois.aic
