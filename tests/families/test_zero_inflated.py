"""Tests for zero-inflated Poisson (ZIP) and negative binomial (ZINB) families."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.families.zero_inflated import ZeroInflatedNegativeBinomial, ZeroInflatedPoisson
from whittaker.gamlss import GAMLSS


class TestZIPFamily:
    def test_parameter_names(self):
        fam = ZeroInflatedPoisson()
        assert fam.parameter_names == ("mu", "pi")

    def test_link_inverse_roundtrip_mu(self):
        fam = ZeroInflatedPoisson()
        mu = np.array([0.5, 1.0, 5.0])
        np.testing.assert_allclose(fam.link_inverse("mu", fam.link("mu", mu)), mu, rtol=1e-10)

    def test_link_inverse_roundtrip_pi(self):
        fam = ZeroInflatedPoisson()
        pi = np.array([0.1, 0.3, 0.5])
        np.testing.assert_allclose(fam.link_inverse("pi", fam.link("pi", pi)), pi, rtol=1e-10)

    def test_log_likelihood_finite(self):
        fam = ZeroInflatedPoisson()
        rng = np.random.default_rng(23)
        y = rng.poisson(3.0, size=100).astype(float)
        y[:20] = 0
        params = {"mu": np.full(100, 3.0), "pi": np.full(100, 0.2)}
        ll = fam.log_likelihood(y, params)
        assert np.isfinite(ll)

    def test_log_likelihood_increases_with_better_pi(self):
        fam = ZeroInflatedPoisson()
        rng = np.random.default_rng(23)
        n = 500
        y = np.zeros(n)
        counts = rng.poisson(3.0, size=n)
        structural_zero = rng.uniform(size=n) < 0.3
        y = np.where(structural_zero, 0, counts).astype(float)

        params_good = {"mu": np.full(n, 3.0), "pi": np.full(n, 0.3)}
        params_bad = {"mu": np.full(n, 3.0), "pi": np.full(n, 0.01)}
        assert fam.log_likelihood(y, params_good) > fam.log_likelihood(y, params_bad)

    def test_simulate(self):
        fam = ZeroInflatedPoisson()
        rng = np.random.default_rng(23)
        params = {"mu": np.full(1000, 3.0), "pi": np.full(1000, 0.3)}
        sim = fam.simulate(params, rng)
        assert sim.shape == (1000,)
        assert np.all(sim >= 0)
        zero_frac = np.mean(sim == 0)
        assert zero_frac > 0.3

    def test_initialize(self):
        fam = ZeroInflatedPoisson()
        y = np.array([0, 0, 0, 1, 2, 3, 5], dtype=float)
        init = fam.initialize(y)
        assert "mu" in init and "pi" in init
        assert np.all(init["mu"] > 0)
        assert np.all(init["pi"] > 0) and np.all(init["pi"] < 1)

    def test_dl_dtheta_finite(self):
        fam = ZeroInflatedPoisson()
        y = np.array([0, 0, 1, 3, 5], dtype=float)
        params = {"mu": np.full(5, 2.0), "pi": np.full(5, 0.2)}
        for p in ["mu", "pi"]:
            dl = fam.dl_dtheta(p, y, params)
            assert np.all(np.isfinite(dl))

    def test_d2l_dtheta2_positive(self):
        fam = ZeroInflatedPoisson()
        y = np.array([0, 0, 1, 3, 5], dtype=float)
        params = {"mu": np.full(5, 2.0), "pi": np.full(5, 0.2)}
        for p in ["mu", "pi"]:
            d2l = fam.d2l_dtheta2(p, y, params)
            assert np.all(d2l > 0)

    def test_repr(self):
        assert "ZIP" in repr(ZeroInflatedPoisson()) or "ZeroInflatedPoisson" in repr(
            ZeroInflatedPoisson()
        )


class TestZINBFamily:
    def test_parameter_names(self):
        fam = ZeroInflatedNegativeBinomial(theta=2.0)
        assert fam.parameter_names == ("mu", "pi")

    def test_theta_property(self):
        fam = ZeroInflatedNegativeBinomial(theta=5.0)
        assert fam.theta == 5.0

    def test_theta_must_be_positive(self):
        with pytest.raises(ValueError, match="positive"):
            ZeroInflatedNegativeBinomial(theta=-1.0)

    def test_link_inverse_roundtrip(self):
        fam = ZeroInflatedNegativeBinomial(theta=2.0)
        mu = np.array([0.5, 2.0, 10.0])
        np.testing.assert_allclose(fam.link_inverse("mu", fam.link("mu", mu)), mu, rtol=1e-10)

    def test_log_likelihood_finite(self):
        fam = ZeroInflatedNegativeBinomial(theta=2.0)
        rng = np.random.default_rng(23)
        y = rng.negative_binomial(2, 0.4, size=100).astype(float)
        y[:20] = 0
        params = {"mu": np.full(100, 3.0), "pi": np.full(100, 0.2)}
        ll = fam.log_likelihood(y, params)
        assert np.isfinite(ll)

    def test_simulate(self):
        fam = ZeroInflatedNegativeBinomial(theta=2.0)
        rng = np.random.default_rng(23)
        params = {"mu": np.full(1000, 3.0), "pi": np.full(1000, 0.3)}
        sim = fam.simulate(params, rng)
        assert sim.shape == (1000,)
        assert np.all(sim >= 0)

    def test_repr(self):
        r = repr(ZeroInflatedNegativeBinomial(theta=2.0))
        assert "theta=2" in r


class TestZIPGAMLSS:
    @pytest.fixture()
    def zip_data(self):
        rng = np.random.default_rng(23)
        n = 400
        x = np.linspace(0, 4, n)
        mu_true = np.exp(0.5 + 0.3 * x)
        pi_true = 0.3
        structural_zero = rng.uniform(size=n) < pi_true
        counts = rng.poisson(mu_true)
        y = np.where(structural_zero, 0, counts).astype(float)
        return {"x": x, "y": y}, mu_true, pi_true

    def test_converges(self, zip_data):
        data, _, _ = zip_data
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "pi": "y ~ 1"},
            family=ZeroInflatedPoisson(),
        )
        model.fit(data, method="GCV")
        assert model.converged

    def test_mu_recovery(self, zip_data):
        data, mu_true, _ = zip_data
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "pi": "y ~ 1"},
            family=ZeroInflatedPoisson(),
        )
        model.fit(data)
        pred = model.predict(data)
        assert np.corrcoef(mu_true, pred.values["mu"])[0, 1] > 0.8

    def test_pi_recovery(self, zip_data):
        data, _, pi_true = zip_data
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "pi": "y ~ 1"},
            family=ZeroInflatedPoisson(),
        )
        model.fit(data)
        pred = model.predict(data)
        pi_est = np.mean(pred.values["pi"])
        assert abs(pi_est - pi_true) < 0.15

    def test_simulate(self, zip_data):
        data, _, _ = zip_data
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "pi": "y ~ 1"},
            family=ZeroInflatedPoisson(),
        )
        model.fit(data)
        sims = model.simulate(n_sim=3, seed=23)
        assert sims.shape == (len(data["y"]), 3)
        assert np.all(sims >= 0)

    def test_summary(self, zip_data):
        data, _, _ = zip_data
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "pi": "y ~ 1"},
            family=ZeroInflatedPoisson(),
        )
        model.fit(data)
        s = model.summary()
        assert "ZeroInflatedPoisson" in s
        assert "mu" in s
        assert "pi" in s


class TestZINBGAMLSS:
    @pytest.fixture()
    def zinb_data(self):
        rng = np.random.default_rng(23)
        n = 400
        x = np.linspace(0, 4, n)
        mu_true = np.exp(0.5 + 0.3 * x)
        pi_true = 0.25
        theta = 2.0
        p = theta / (mu_true + theta)
        structural_zero = rng.uniform(size=n) < pi_true
        counts = rng.negative_binomial(theta, p)
        y = np.where(structural_zero, 0, counts).astype(float)
        return {"x": x, "y": y}, mu_true, pi_true

    def test_converges(self, zinb_data):
        data, _, _ = zinb_data
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "pi": "y ~ 1"},
            family=ZeroInflatedNegativeBinomial(theta=2.0),
        )
        model.fit(data, method="GCV")
        assert model.converged

    def test_mu_recovery(self, zinb_data):
        data, mu_true, _ = zinb_data
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "pi": "y ~ 1"},
            family=ZeroInflatedNegativeBinomial(theta=2.0),
        )
        model.fit(data)
        pred = model.predict(data)
        assert np.corrcoef(mu_true, pred.values["mu"])[0, 1] > 0.7

    def test_simulate(self, zinb_data):
        data, _, _ = zinb_data
        model = GAMLSS(
            formulas={"mu": "y ~ s(x)", "pi": "y ~ 1"},
            family=ZeroInflatedNegativeBinomial(theta=2.0),
        )
        model.fit(data)
        sims = model.simulate(n_sim=3, seed=23)
        assert sims.shape == (len(data["y"]), 3)
        assert np.all(sims >= 0)
