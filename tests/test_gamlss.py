"""Tests for GAMLSS integration."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.families import BetaLS, GammaLS, GaussianLS
from whittaker.gamlss import GAMLSS, GAMLSSPrediction

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def heteroscedastic_data():
    rng = np.random.default_rng(23)
    n = 300
    x = rng.uniform(0, 2 * np.pi, n)
    noise_scale = 0.1 + 0.5 * (x / (2 * np.pi))
    y = np.sin(x) + rng.normal(0, noise_scale, n)
    return {"x": x, "y": y}


@pytest.fixture()
def homoscedastic_data():
    rng = np.random.default_rng(23)
    n = 300
    x = rng.uniform(0, 2 * np.pi, n)
    y = np.sin(x) + rng.normal(0, 0.3, n)
    return {"x": x, "y": y}


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


class TestGAMLSSFitting:
    def test_fit_reml(self, heteroscedastic_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "sigma": "y ~ s(x)"},
            family=GaussianLS(),
        )
        model.fit(heteroscedastic_data, method="REML")
        assert model.is_fitted

    def test_fit_gcv(self, heteroscedastic_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "sigma": "y ~ s(x)"},
            family=GaussianLS(),
        )
        model.fit(heteroscedastic_data)
        assert model.is_fitted

    def test_fit_ml(self, heteroscedastic_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "sigma": "y ~ s(x)"},
            family=GaussianLS(),
        )
        model.fit(heteroscedastic_data, method="ML")
        assert model.is_fitted

    def test_converges(self, heteroscedastic_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "sigma": "y ~ s(x)"},
            family=GaussianLS(),
        )
        model.fit(heteroscedastic_data, method="REML")
        assert model.converged

    def test_coefficients_finite(self, heteroscedastic_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "sigma": "y ~ s(x)"},
            family=GaussianLS(),
        )
        model.fit(heteroscedastic_data, method="REML")
        assert np.isfinite(model.coefficients("mu")).all()
        assert np.isfinite(model.coefficients("sigma")).all()

    def test_default_family(self, heteroscedastic_data):
        model = GAMLSS(formulas={"mu": "y ~ s(x)", "sigma": "y ~ s(x)"})
        model.fit(heteroscedastic_data, method="REML")
        assert model.is_fitted


# ---------------------------------------------------------------------------
# Heteroscedasticity detection
# ---------------------------------------------------------------------------


class TestGAMLSSHeteroscedasticity:
    def test_sigma_varies_with_x(self, heteroscedastic_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "sigma": "y ~ s(x)"},
            family=GaussianLS(),
        )
        model.fit(heteroscedastic_data, method="REML")
        x_grid = np.linspace(0, 2 * np.pi, 20)
        pred = model.predict({"x": x_grid})
        sigma_low = pred.values["sigma"][:5].mean()
        sigma_high = pred.values["sigma"][-5:].mean()
        assert sigma_high > sigma_low * 1.5


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


class TestGAMLSSPrediction:
    def test_predict_returns_all_params(self, heteroscedastic_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "sigma": "y ~ s(x)"},
            family=GaussianLS(),
        )
        model.fit(heteroscedastic_data, method="REML")
        pred = model.predict(heteroscedastic_data)
        assert isinstance(pred, GAMLSSPrediction)
        assert "mu" in pred.values
        assert "sigma" in pred.values

    def test_predict_single_param(self, heteroscedastic_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "sigma": "y ~ s(x)"},
            family=GaussianLS(),
        )
        model.fit(heteroscedastic_data, method="REML")
        mu = model.predict(heteroscedastic_data, parameter="mu")
        assert isinstance(mu, np.ndarray)
        assert mu.shape == (len(heteroscedastic_data["y"]),)

    def test_predict_new_data(self, heteroscedastic_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "sigma": "y ~ s(x)"},
            family=GaussianLS(),
        )
        model.fit(heteroscedastic_data, method="REML")
        new = {"x": np.linspace(0, 2 * np.pi, 50)}
        pred = model.predict(new)
        assert pred.values["mu"].shape == (50,)
        assert pred.values["sigma"].shape == (50,)
        assert np.isfinite(pred.values["mu"]).all()
        assert np.all(pred.values["sigma"] > 0)

    def test_predict_sigma_positive(self, heteroscedastic_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "sigma": "y ~ s(x)"},
            family=GaussianLS(),
        )
        model.fit(heteroscedastic_data, method="REML")
        pred = model.predict(heteroscedastic_data)
        assert np.all(pred.values["sigma"] > 0)

    def test_predict_se_returns_all_params(self, heteroscedastic_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "sigma": "y ~ s(x)"},
            family=GaussianLS(),
        )
        model.fit(heteroscedastic_data, method="REML")
        pred = model.predict(heteroscedastic_data, se=True)
        assert pred.se is not None
        assert "mu" in pred.se
        assert "sigma" in pred.se

    def test_predict_se_positive_finite(self, heteroscedastic_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "sigma": "y ~ s(x)"},
            family=GaussianLS(),
        )
        model.fit(heteroscedastic_data, method="REML")
        pred = model.predict(heteroscedastic_data, se=True)
        for name in ("mu", "sigma"):
            assert np.all(pred.se[name] > 0)
            assert np.all(np.isfinite(pred.se[name]))

    def test_predict_se_correct_shape(self, heteroscedastic_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "sigma": "y ~ s(x)"},
            family=GaussianLS(),
        )
        model.fit(heteroscedastic_data, method="REML")
        new = {"x": np.linspace(0, 2 * np.pi, 20)}
        pred = model.predict(new, se=True)
        assert pred.se["mu"].shape == (20,)
        assert pred.se["sigma"].shape == (20,)

    def test_predict_se_none_when_false(self, heteroscedastic_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "sigma": "y ~ s(x)"},
            family=GaussianLS(),
        )
        model.fit(heteroscedastic_data, method="REML")
        pred = model.predict(heteroscedastic_data)
        assert pred.se is None

    def test_predict_with_offset(self):
        rng = np.random.default_rng(23)
        n = 200
        x = rng.uniform(0, 2 * np.pi, n)
        off = rng.normal(0, 0.1, n)
        y = np.sin(x) + off + rng.normal(0, 0.3, n)
        data = {"x": x, "off": off, "y": y}

        model = GAMLSS(
            formulas={"mu": "y ~ s(x) + offset(off)", "sigma": "y ~ 1"},
            family=GaussianLS(),
        )
        model.fit(data, method="REML")
        pred = model.predict(data, parameter="mu")
        assert np.all(np.isfinite(pred))
        assert pred.shape == (n,)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestGAMLSSProperties:
    def test_aic_bic(self, heteroscedastic_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "sigma": "y ~ s(x)"},
            family=GaussianLS(),
        )
        model.fit(heteroscedastic_data, method="REML")
        assert np.isfinite(model.aic)
        assert np.isfinite(model.bic)

    def test_log_likelihood(self, heteroscedastic_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "sigma": "y ~ s(x)"},
            family=GaussianLS(),
        )
        model.fit(heteroscedastic_data, method="REML")
        assert np.isfinite(model.log_likelihood)

    def test_global_deviance(self, heteroscedastic_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "sigma": "y ~ s(x)"},
            family=GaussianLS(),
        )
        model.fit(heteroscedastic_data, method="REML")
        np.testing.assert_allclose(model.global_deviance, -2.0 * model.log_likelihood, rtol=1e-10)

    def test_edf(self, heteroscedastic_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "sigma": "y ~ s(x)"},
            family=GaussianLS(),
        )
        model.fit(heteroscedastic_data, method="REML")
        assert len(model.edf("mu")) == 1
        assert len(model.edf("sigma")) == 1
        assert model.edf("mu")[0] > 0
        assert model.edf("sigma")[0] > 0

    def test_smoothing_params(self, heteroscedastic_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "sigma": "y ~ s(x)"},
            family=GaussianLS(),
        )
        model.fit(heteroscedastic_data, method="REML")
        assert len(model.smoothing_params("mu")) > 0
        assert len(model.smoothing_params("sigma")) > 0

    def test_fitted_values_all(self, heteroscedastic_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "sigma": "y ~ s(x)"},
            family=GaussianLS(),
        )
        model.fit(heteroscedastic_data, method="REML")
        fv = model.fitted_values()
        assert isinstance(fv, dict)
        assert "mu" in fv and "sigma" in fv

    def test_fitted_values_single(self, heteroscedastic_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "sigma": "y ~ s(x)"},
            family=GaussianLS(),
        )
        model.fit(heteroscedastic_data, method="REML")
        mu_fv = model.fitted_values("mu")
        assert isinstance(mu_fv, np.ndarray)

    def test_family_property(self, heteroscedastic_data):
        family = GaussianLS()
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "sigma": "y ~ s(x)"},
            family=family,
        )
        model.fit(heteroscedastic_data, method="REML")
        assert model.family is family

    def test_n_iter_positive(self, heteroscedastic_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "sigma": "y ~ s(x)"},
            family=GaussianLS(),
        )
        model.fit(heteroscedastic_data, method="REML")
        assert model.n_iter >= 1


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestGAMLSSSummary:
    def test_summary(self, heteroscedastic_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "sigma": "y ~ s(x)"},
            family=GaussianLS(),
        )
        model.fit(heteroscedastic_data, method="REML")
        s = model.summary()
        assert "GAMLSS" in s
        assert "mu" in s
        assert "sigma" in s
        assert "AIC" in s


# ---------------------------------------------------------------------------
# Simulate
# ---------------------------------------------------------------------------


class TestGAMLSSSimulate:
    def test_simulate(self, heteroscedastic_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "sigma": "y ~ s(x)"},
            family=GaussianLS(),
        )
        model.fit(heteroscedastic_data, method="REML")
        sims = model.simulate(n_sim=10, seed=23)
        assert sims.shape == (len(heteroscedastic_data["y"]), 10)
        assert np.isfinite(sims).all()


# ---------------------------------------------------------------------------
# Multiple smooths per parameter
# ---------------------------------------------------------------------------


class TestGAMLSSMultipleSmooths:
    def test_two_smooths_mu(self):
        rng = np.random.default_rng(23)
        n = 300
        x1 = rng.uniform(0, 2 * np.pi, n)
        x2 = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x1) + 0.5 * np.cos(x2) + rng.normal(0, 0.3, n)
        data = {"x1": x1, "x2": x2, "y": y}
        model = GAMLSS(
            formulas={"mu": "y ~ s(x1) + s(x2)", "sigma": "y ~ 1"},
            family=GaussianLS(),
        )
        model.fit(data, method="REML")
        assert model.is_fitted
        assert len(model.edf("mu")) == 2


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestGAMLSSValidation:
    def test_not_fitted_raises(self):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "sigma": "y ~ s(x)"},
            family=GaussianLS(),
        )
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict({"x": np.array([1.0])})

    def test_missing_param_formula_raises(self):
        with pytest.raises(ValueError, match="Missing formula"):
            GAMLSS(formulas={"mu": "y ~ s(x)"}, family=GaussianLS())

    def test_mismatched_response_raises(self, heteroscedastic_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "sigma": "z ~ s(x)"},
            family=GaussianLS(),
        )
        with pytest.raises(ValueError, match="same response"):
            model.fit(heteroscedastic_data)


# ---------------------------------------------------------------------------
# GAMLSS vs GAM comparison
# ---------------------------------------------------------------------------


class TestGAMLSSvsGAM:
    def test_mu_similar_to_gam_homoscedastic(self, homoscedastic_data):
        from whittaker.families import Gaussian
        from whittaker.gam import GAM

        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(homoscedastic_data, method="REML")
        pred_gam = gam.predict(homoscedastic_data).values

        gamlss = GAMLSS(
            formulas={"mu": "y ~ s(x)", "sigma": "y ~ 1"},
            family=GaussianLS(),
        )
        gamlss.fit(homoscedastic_data, method="REML")
        pred_gamlss = gamlss.predict(homoscedastic_data, parameter="mu")

        np.testing.assert_allclose(pred_gam, pred_gamlss, atol=0.3)


# ---------------------------------------------------------------------------
# GammaLS integration
# ---------------------------------------------------------------------------


@pytest.fixture()
def gamma_data():
    rng = np.random.default_rng(23)
    n = 400
    x = np.linspace(0, 4, n)
    mu_true = np.exp(1.5 - 0.3 * x)
    sigma_true = np.full(n, 0.3)
    shape = 1.0 / sigma_true**2
    scale = mu_true * sigma_true**2
    y = rng.gamma(shape=shape, scale=scale, size=n)
    return {"x": x, "y": y, "mu_true": mu_true, "sigma_true": sigma_true}


class TestGammaLSFitting:
    def test_converges(self, gamma_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "sigma": "y ~ 1"},
            family=GammaLS(),
        )
        model.fit(gamma_data, method="GCV")
        assert model.converged

    def test_mu_recovery(self, gamma_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "sigma": "y ~ 1"},
            family=GammaLS(),
        )
        model.fit(gamma_data, method="GCV")
        pred = model.predict(gamma_data)
        np.testing.assert_allclose(pred.values["mu"], gamma_data["mu_true"], rtol=0.15)

    def test_sigma_recovery(self, gamma_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "sigma": "y ~ 1"},
            family=GammaLS(),
        )
        model.fit(gamma_data, method="GCV")
        pred = model.predict(gamma_data)
        sigma_hat = pred.values["sigma"].mean()
        np.testing.assert_allclose(sigma_hat, 0.3, atol=0.1)

    def test_mu_positive(self, gamma_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "sigma": "y ~ 1"},
            family=GammaLS(),
        )
        model.fit(gamma_data, method="GCV")
        pred = model.predict(gamma_data)
        assert np.all(pred.values["mu"] > 0)
        assert np.all(pred.values["sigma"] > 0)

    def test_fit_reml(self, gamma_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "sigma": "y ~ 1"},
            family=GammaLS(),
        )
        model.fit(gamma_data, method="REML")
        assert model.is_fitted

    def test_sigma_smooth(self):
        rng = np.random.default_rng(23)
        n = 400
        x = np.linspace(0, 4, n)
        mu_true = np.full(n, 3.0)
        sigma_true = 0.2 + 0.3 * x / 4
        shape = 1.0 / sigma_true**2
        scale = mu_true * sigma_true**2
        y = rng.gamma(shape=shape, scale=scale, size=n)
        data = {"x": x, "y": y}
        model = GAMLSS(
            formulas={"mu": "y ~ 1", "sigma": "y ~ s(x)"},
            family=GammaLS(),
        )
        model.fit(data, method="GCV")
        pred = model.predict(data)
        sigma_low = pred.values["sigma"][:50].mean()
        sigma_high = pred.values["sigma"][-50:].mean()
        assert sigma_high > sigma_low * 1.3

    def test_simulate(self, gamma_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "sigma": "y ~ 1"},
            family=GammaLS(),
        )
        model.fit(gamma_data, method="GCV")
        sims = model.simulate(n_sim=5, seed=23)
        assert sims.shape == (len(gamma_data["y"]), 5)
        assert np.all(sims > 0)

    def test_summary(self, gamma_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "sigma": "y ~ 1"},
            family=GammaLS(),
        )
        model.fit(gamma_data, method="GCV")
        s = model.summary()
        assert "GammaLS" in s
        assert "mu" in s
        assert "sigma" in s


# ---------------------------------------------------------------------------
# BetaLS integration
# ---------------------------------------------------------------------------


@pytest.fixture()
def beta_data():
    rng = np.random.default_rng(23)
    n = 400
    x = np.linspace(-2, 2, n)
    from scipy.special import expit

    mu_true = expit(0.5 + 0.8 * x)
    phi_true = np.full(n, 20.0)
    a = mu_true * phi_true
    b = (1 - mu_true) * phi_true
    y = rng.beta(a, b)
    return {"x": x, "y": y, "mu_true": mu_true, "phi_true": phi_true}


class TestBetaLSFitting:
    def test_converges(self, beta_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "phi": "y ~ 1"},
            family=BetaLS(),
        )
        model.fit(beta_data, method="GCV")
        assert model.converged

    def test_mu_recovery(self, beta_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "phi": "y ~ 1"},
            family=BetaLS(),
        )
        model.fit(beta_data, method="GCV")
        pred = model.predict(beta_data)
        np.testing.assert_allclose(pred.values["mu"], beta_data["mu_true"], atol=0.1)

    def test_phi_recovery(self, beta_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "phi": "y ~ 1"},
            family=BetaLS(),
        )
        model.fit(beta_data, method="GCV")
        pred = model.predict(beta_data)
        phi_hat = pred.values["phi"].mean()
        np.testing.assert_allclose(phi_hat, 20.0, rtol=0.3)

    def test_mu_in_unit_interval(self, beta_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "phi": "y ~ 1"},
            family=BetaLS(),
        )
        model.fit(beta_data, method="GCV")
        pred = model.predict(beta_data)
        assert np.all((pred.values["mu"] > 0) & (pred.values["mu"] < 1))
        assert np.all(pred.values["phi"] > 0)

    def test_fit_reml(self, beta_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "phi": "y ~ 1"},
            family=BetaLS(),
        )
        model.fit(beta_data, method="REML")
        assert model.is_fitted

    def test_simulate(self, beta_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "phi": "y ~ 1"},
            family=BetaLS(),
        )
        model.fit(beta_data, method="GCV")
        sims = model.simulate(n_sim=5, seed=23)
        assert sims.shape == (len(beta_data["y"]), 5)
        assert np.all((sims > 0) & (sims < 1))

    def test_summary(self, beta_data):
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "phi": "y ~ 1"},
            family=BetaLS(),
        )
        model.fit(beta_data, method="GCV")
        s = model.summary()
        assert "BetaLS" in s
        assert "mu" in s
        assert "phi" in s
