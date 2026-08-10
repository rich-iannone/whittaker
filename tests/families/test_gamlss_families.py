"""Tests for GAMLSS family classes."""

from __future__ import annotations

import numpy as np

from whittaker.families.gamlss_base import GAMLSSFamily
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
