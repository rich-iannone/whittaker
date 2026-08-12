"""Tests for GAMLSS family classes."""

from __future__ import annotations

import numpy as np

from whittaker.families.beta_ls import BetaLS
from whittaker.families.gamlss_base import GAMLSSFamily
from whittaker.families.gamma_ls import GammaLS
from whittaker.families.gaussian_ls import GaussianLS


class TestGaussianLSInterface:
    def test_parameter_names(self):
        f = GaussianLS()
        assert f.parameter_names == ("mu", "sigma")

    def test_mu_identity_link(self):
        f = GaussianLS()
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(f.link("mu", x), x)
        np.testing.assert_array_equal(f.link_inverse("mu", x), x)
        np.testing.assert_array_equal(f.link_derivative("mu", x), np.ones(3))

    def test_sigma_log_link(self):
        f = GaussianLS()
        sigma = np.array([0.5, 1.0, 2.0])
        eta = f.link("sigma", sigma)
        np.testing.assert_allclose(eta, np.log(sigma))
        np.testing.assert_allclose(f.link_inverse("sigma", eta), sigma)
        np.testing.assert_allclose(f.link_derivative("sigma", sigma), 1.0 / sigma)


class TestGaussianLSDerivatives:
    def test_dl_dmu(self):
        f = GaussianLS()
        y = np.array([1.0, 2.0, 3.0])
        params = {"mu": np.array([1.1, 1.9, 3.2]), "sigma": np.ones(3)}
        dl = f.dl_dtheta("mu", y, params)
        expected = (y - params["mu"]) / params["sigma"] ** 2
        np.testing.assert_allclose(dl, expected)

    def test_dl_dsigma(self):
        f = GaussianLS()
        y = np.array([1.0, 2.0, 3.0])
        params = {"mu": np.array([1.0, 2.0, 3.0]), "sigma": np.ones(3) * 2.0}
        dl = f.dl_dtheta("sigma", y, params)
        sigma = params["sigma"]
        expected = -1.0 / sigma + (y - params["mu"]) ** 2 / sigma**3
        np.testing.assert_allclose(dl, expected)

    def test_d2l_dmu2_positive(self):
        f = GaussianLS()
        y = np.array([1.0, 2.0, 3.0])
        params = {"mu": np.array([1.0, 2.0, 3.0]), "sigma": np.ones(3)}
        d2l = f.d2l_dtheta2("mu", y, params)
        assert np.all(d2l > 0)

    def test_d2l_dsigma2_positive(self):
        f = GaussianLS()
        y = np.array([1.0, 2.0, 3.0])
        params = {"mu": np.array([1.0, 2.0, 3.0]), "sigma": np.ones(3)}
        d2l = f.d2l_dtheta2("sigma", y, params)
        assert np.all(d2l > 0)


class TestGaussianLSOther:
    def test_log_likelihood(self):
        f = GaussianLS()
        y = np.array([0.0])
        params = {"mu": np.array([0.0]), "sigma": np.array([1.0])}
        ll = f.log_likelihood(y, params)
        expected = -0.5 * np.log(2 * np.pi)
        np.testing.assert_allclose(ll, expected, atol=1e-10)

    def test_initialize(self):
        f = GaussianLS()
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        init = f.initialize(y)
        np.testing.assert_array_equal(init["mu"], y)
        np.testing.assert_allclose(init["sigma"], np.std(y, ddof=1) * np.ones(5))

    def test_simulate_shape(self):
        f = GaussianLS()
        rng = np.random.default_rng(23)
        params = {"mu": np.zeros(100), "sigma": np.ones(100)}
        sims = f.simulate(params, rng)
        assert sims.shape == (100,)
        assert np.isfinite(sims).all()

    def test_repr(self):
        f = GaussianLS()
        assert "GaussianLS" in repr(f)

    def test_is_gamlss_family(self):
        assert isinstance(GaussianLS(), GAMLSSFamily)


# --- GammaLS ---


class TestGammaLSInterface:
    def test_parameter_names(self):
        f = GammaLS()
        assert f.parameter_names == ("mu", "sigma")

    def test_mu_log_link(self):
        f = GammaLS()
        mu = np.array([0.5, 1.0, 2.0])
        eta = f.link("mu", mu)
        np.testing.assert_allclose(eta, np.log(mu))
        np.testing.assert_allclose(f.link_inverse("mu", eta), mu)
        np.testing.assert_allclose(f.link_derivative("mu", mu), 1.0 / mu)

    def test_sigma_log_link(self):
        f = GammaLS()
        sigma = np.array([0.2, 0.5, 1.0])
        eta = f.link("sigma", sigma)
        np.testing.assert_allclose(eta, np.log(sigma))
        np.testing.assert_allclose(f.link_inverse("sigma", eta), sigma)
        np.testing.assert_allclose(f.link_derivative("sigma", sigma), 1.0 / sigma)

    def test_is_gamlss_family(self):
        assert isinstance(GammaLS(), GAMLSSFamily)

    def test_repr(self):
        assert "GammaLS" in repr(GammaLS())


class TestGammaLSDerivatives:
    def _numerical_dl(self, f, param, y, params, eps=1e-6):
        p0 = {k: v.copy() for k, v in params.items()}
        p1 = {k: v.copy() for k, v in params.items()}
        p0[param] = params[param] - eps
        p1[param] = params[param] + eps
        ll0 = f.log_likelihood(y, p0)
        ll1 = f.log_likelihood(y, p1)
        return (ll1 - ll0) / (2 * eps)

    def test_dl_dmu_vs_numerical(self):
        f = GammaLS()
        y = np.array([2.0, 3.0, 4.0])
        params = {"mu": np.array([2.5, 3.5, 3.8]), "sigma": np.full(3, 0.3)}
        analytic = f.dl_dtheta("mu", y, params)
        numerical = self._numerical_dl(f, "mu", y, params)
        np.testing.assert_allclose(np.sum(analytic), numerical, rtol=1e-4)

    def test_dl_dsigma_vs_numerical(self):
        f = GammaLS()
        y = np.array([2.0, 3.0, 4.0])
        params = {"mu": np.array([2.5, 3.5, 3.8]), "sigma": np.full(3, 0.3)}
        analytic = f.dl_dtheta("sigma", y, params)
        numerical = self._numerical_dl(f, "sigma", y, params)
        np.testing.assert_allclose(np.sum(analytic), numerical, rtol=1e-4)

    def test_dl_dmu_positive(self):
        f = GammaLS()
        1.0 / 0.3**2
        params = {"mu": np.array([1.0]), "sigma": np.array([0.3])}
        y = np.array([2.0])
        dl = f.dl_dtheta("mu", y, params)
        assert dl[0] > 0

    def test_d2l_dmu2_positive(self):
        f = GammaLS()
        y = np.array([1.0, 2.0, 3.0])
        params = {"mu": np.array([1.5, 2.5, 2.8]), "sigma": np.full(3, 0.5)}
        d2l = f.d2l_dtheta2("mu", y, params)
        assert np.all(d2l > 0)

    def test_d2l_dsigma2_positive(self):
        f = GammaLS()
        y = np.array([1.0, 2.0, 3.0])
        params = {"mu": np.array([1.5, 2.5, 2.8]), "sigma": np.full(3, 0.5)}
        d2l = f.d2l_dtheta2("sigma", y, params)
        assert np.all(d2l > 0)

    def test_score_zero_at_mle(self):
        f = GammaLS()
        rng = np.random.default_rng(23)
        n = 10000
        mu_true = np.full(n, 3.0)
        sigma_true = np.full(n, 0.4)
        shape = 1.0 / sigma_true**2
        scale = mu_true * sigma_true**2
        y = rng.gamma(shape=shape, scale=scale, size=n)
        params = {"mu": mu_true, "sigma": sigma_true}
        dl_mu = np.mean(f.dl_dtheta("mu", y, params))
        dl_sigma = np.mean(f.dl_dtheta("sigma", y, params))
        assert abs(dl_mu) < 0.05
        assert abs(dl_sigma) < 0.05


class TestGammaLSOther:
    def test_log_likelihood(self):
        f = GammaLS()
        rng = np.random.default_rng(23)
        mu = np.array([2.0])
        sigma = np.array([0.5])
        shape = 1.0 / sigma**2
        scale = mu * sigma**2
        y = rng.gamma(shape=shape, scale=scale, size=1)
        ll = f.log_likelihood(y, {"mu": mu, "sigma": sigma})
        from scipy.stats import gamma as gamma_dist

        expected = gamma_dist.logpdf(y, a=shape, scale=scale).sum()
        np.testing.assert_allclose(ll, expected, atol=1e-10)

    def test_initialize(self):
        f = GammaLS()
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        init = f.initialize(y)
        np.testing.assert_array_equal(init["mu"], y)
        cv = np.std(y) / np.mean(y)
        np.testing.assert_allclose(init["sigma"], np.full(5, cv))

    def test_simulate_positive(self):
        f = GammaLS()
        rng = np.random.default_rng(23)
        params = {"mu": np.full(100, 2.0), "sigma": np.full(100, 0.3)}
        sims = f.simulate(params, rng)
        assert sims.shape == (100,)
        assert np.all(sims > 0)


# --- BetaLS ---


class TestBetaLSInterface:
    def test_parameter_names(self):
        f = BetaLS()
        assert f.parameter_names == ("mu", "phi")

    def test_mu_logit_link(self):
        f = BetaLS()
        from scipy.special import logit

        mu = np.array([0.2, 0.5, 0.8])
        eta = f.link("mu", mu)
        np.testing.assert_allclose(eta, logit(mu))
        np.testing.assert_allclose(f.link_inverse("mu", eta), mu)
        np.testing.assert_allclose(f.link_derivative("mu", mu), 1.0 / (mu * (1 - mu)))

    def test_phi_log_link(self):
        f = BetaLS()
        phi = np.array([5.0, 10.0, 50.0])
        eta = f.link("phi", phi)
        np.testing.assert_allclose(eta, np.log(phi))
        np.testing.assert_allclose(f.link_inverse("phi", eta), phi)
        np.testing.assert_allclose(f.link_derivative("phi", phi), 1.0 / phi)

    def test_is_gamlss_family(self):
        assert isinstance(BetaLS(), GAMLSSFamily)

    def test_repr(self):
        assert "BetaLS" in repr(BetaLS())


class TestBetaLSDerivatives:
    def _numerical_dl(self, f, param, y, params, eps=1e-6):
        p0 = {k: v.copy() for k, v in params.items()}
        p1 = {k: v.copy() for k, v in params.items()}
        p0[param] = params[param] - eps
        p1[param] = params[param] + eps
        ll0 = f.log_likelihood(y, p0)
        ll1 = f.log_likelihood(y, p1)
        return (ll1 - ll0) / (2 * eps)

    def test_dl_dmu_vs_numerical(self):
        f = BetaLS()
        y = np.array([0.3, 0.5, 0.7])
        params = {"mu": np.array([0.35, 0.45, 0.65]), "phi": np.full(3, 20.0)}
        analytic = f.dl_dtheta("mu", y, params)
        numerical = self._numerical_dl(f, "mu", y, params)
        np.testing.assert_allclose(np.sum(analytic), numerical, rtol=1e-4)

    def test_dl_dphi_vs_numerical(self):
        f = BetaLS()
        y = np.array([0.3, 0.5, 0.7])
        params = {"mu": np.array([0.35, 0.45, 0.65]), "phi": np.full(3, 20.0)}
        analytic = f.dl_dtheta("phi", y, params)
        numerical = self._numerical_dl(f, "phi", y, params)
        np.testing.assert_allclose(np.sum(analytic), numerical, rtol=1e-4)

    def test_d2l_dmu2_positive(self):
        f = BetaLS()
        y = np.array([0.3, 0.5, 0.7])
        params = {"mu": np.array([0.35, 0.45, 0.65]), "phi": np.full(3, 20.0)}
        d2l = f.d2l_dtheta2("mu", y, params)
        assert np.all(d2l > 0)

    def test_d2l_dphi2_positive(self):
        f = BetaLS()
        y = np.array([0.3, 0.5, 0.7])
        params = {"mu": np.array([0.35, 0.45, 0.65]), "phi": np.full(3, 20.0)}
        d2l = f.d2l_dtheta2("phi", y, params)
        assert np.all(d2l > 0)

    def test_score_zero_at_mle(self):
        f = BetaLS()
        rng = np.random.default_rng(23)
        n = 50000
        mu_true = np.full(n, 0.4)
        phi_true = np.full(n, 10.0)
        a = mu_true * phi_true
        b = (1 - mu_true) * phi_true
        y = rng.beta(a, b)
        params = {"mu": mu_true, "phi": phi_true}
        dl_mu = np.mean(f.dl_dtheta("mu", y, params))
        dl_phi = np.mean(f.dl_dtheta("phi", y, params))
        assert abs(dl_mu) < 0.1
        assert abs(dl_phi) < 0.05


class TestBetaLSOther:
    def test_log_likelihood(self):
        f = BetaLS()
        y = np.array([0.3])
        mu = np.array([0.4])
        phi = np.array([10.0])
        ll = f.log_likelihood(y, {"mu": mu, "phi": phi})
        from scipy.stats import beta as beta_dist

        a, b = mu * phi, (1 - mu) * phi
        expected = beta_dist.logpdf(y, a, b).sum()
        np.testing.assert_allclose(ll, expected, atol=1e-10)

    def test_initialize(self):
        f = BetaLS()
        y = np.array([0.2, 0.4, 0.6, 0.8])
        init = f.initialize(y)
        np.testing.assert_allclose(init["mu"], y)
        assert np.all(init["phi"] > 0)

    def test_simulate_in_unit_interval(self):
        f = BetaLS()
        rng = np.random.default_rng(23)
        params = {"mu": np.full(100, 0.5), "phi": np.full(100, 20.0)}
        sims = f.simulate(params, rng)
        assert sims.shape == (100,)
        assert np.all((sims > 0) & (sims < 1))
