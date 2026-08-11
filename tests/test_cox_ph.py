"""Tests for Cox proportional hazards family."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.families.cox_ph import CoxPH
from whittaker.gam import GAM


def _make_cox_data(n=500, seed=23):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    eta_true = 0.5 * x1 - 0.3 * x2
    time = rng.exponential(1.0 / np.exp(eta_true))
    censor_time = rng.exponential(3.0, n)
    event = (time <= censor_time).astype(float)
    time = np.minimum(time, censor_time)
    return {"y": time, "x1": x1, "x2": x2, "event": event}


def _make_simple_cox_data(n=200, seed=23):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    eta_true = 0.8 * x
    time = rng.exponential(1.0 / np.exp(eta_true))
    censor_time = rng.exponential(5.0, n)
    event = (time <= censor_time).astype(float)
    time = np.minimum(time, censor_time)
    return {"y": time, "x": x, "event": event}


class TestCoxPHFamily:
    def test_init_default(self):
        fam = CoxPH()
        assert fam.ties == "breslow"

    def test_init_efron(self):
        fam = CoxPH(ties="efron")
        assert fam.ties == "efron"

    def test_invalid_ties(self):
        with pytest.raises(ValueError, match="ties must be"):
            CoxPH(ties="invalid")

    def test_scale_known(self):
        fam = CoxPH()
        assert fam.scale_known is True

    def test_link_identity(self):
        fam = CoxPH()
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(fam.link(x), x)
        np.testing.assert_array_equal(fam.link_inverse(x), x)
        np.testing.assert_array_equal(fam.link_derivative(x), np.ones(3))

    def test_initialize_returns_zeros(self):
        fam = CoxPH()
        y = np.array([1.0, 2.5, 0.5, 3.0])
        mu0 = fam.initialize(y)
        np.testing.assert_array_equal(mu0, np.zeros(4))

    def test_set_data(self):
        fam = CoxPH(status="event")
        data = {"y": np.array([1.0, 2.0]), "event": np.array([1.0, 0.0])}
        fam.set_data(data)
        np.testing.assert_array_equal(fam._event, [1.0, 0.0])

    def test_set_data_missing_column(self):
        fam = CoxPH(status="status")
        with pytest.raises(KeyError, match="status"):
            fam.set_data({"y": np.array([1.0]), "event": np.array([1.0])})

    def test_repr(self):
        fam = CoxPH()
        assert repr(fam) == "CoxPH(ties='breslow')"


class TestCoxIRLS:
    def test_irls_update_shapes(self):
        data = _make_simple_cox_data(n=100)
        fam = CoxPH()
        fam.set_data(data)
        y = data["y"]
        fam.initialize(y)
        eta = np.zeros(100)
        z, W = fam.irls_update(y, eta, eta)
        assert z.shape == (100,)
        assert W.shape == (100,)

    def test_working_weights_positive(self):
        data = _make_simple_cox_data(n=100)
        fam = CoxPH()
        fam.set_data(data)
        y = data["y"]
        fam.initialize(y)
        eta = np.zeros(100)
        z, W = fam.irls_update(y, eta, eta)
        assert np.all(W > 0)

    def test_partial_log_likelihood_negative(self):
        data = _make_simple_cox_data(n=100)
        fam = CoxPH()
        fam.set_data(data)
        fam.initialize(data["y"])
        pll = fam._partial_log_likelihood(np.zeros(100))
        assert pll < 0

    def test_deviance_positive(self):
        data = _make_simple_cox_data(n=100)
        fam = CoxPH()
        fam.set_data(data)
        y = data["y"]
        fam.initialize(y)
        dev = fam.deviance(y, np.zeros(100))
        assert dev > 0


class TestCoxPHGAM:
    def test_fit_linear(self):
        data = _make_cox_data(n=500)
        model = GAM("y ~ x1 + x2", family=CoxPH(status="event"))
        model.fit(data)
        assert model.is_fitted

    def test_fit_smooth(self):
        data = _make_simple_cox_data(n=300)
        model = GAM("y ~ s(x)", family=CoxPH(status="event"))
        model.fit(data)
        assert model.is_fitted

    def test_coefficient_recovery(self):
        data = _make_cox_data(n=2000, seed=0)
        model = GAM("y ~ x1 + x2", family=CoxPH(status="event"))
        model.fit(data)
        coefs = model.coefficients
        x1_coef = coefs[1]
        x2_coef = coefs[2]
        assert abs(x1_coef - 0.5) < 0.15
        assert abs(x2_coef - (-0.3)) < 0.15

    def test_predict(self):
        data = _make_simple_cox_data(n=200)
        model = GAM("y ~ s(x)", family=CoxPH(status="event"))
        model.fit(data)
        pred = model.predict(data)
        assert pred.values.shape == (200,)
        assert np.all(np.isfinite(pred.values))

    def test_deviance_decreases(self):
        data = _make_cox_data(n=500)
        model = GAM("y ~ x1 + x2", family=CoxPH(status="event"))
        model.fit(data)
        assert model.deviance < model.null_deviance

    def test_baseline_hazard(self):
        data = _make_simple_cox_data(n=200)
        model = GAM("y ~ s(x)", family=CoxPH(status="event"))
        model.fit(data)
        fam = model._family
        times, cumhaz = fam.baseline_hazard()
        assert len(times) > 0
        assert len(cumhaz) == len(times)
        assert np.all(cumhaz >= 0)
        assert np.all(np.diff(cumhaz) >= 0)

    def test_survival_function(self):
        data = _make_simple_cox_data(n=200)
        model = GAM("y ~ s(x)", family=CoxPH(status="event"))
        model.fit(data)
        fam = model._family
        eta = model._fit_result.linear_predictor
        surv = fam.survival_function(eta)
        assert surv.shape == (200,)
        assert np.all(surv >= 0)
        assert np.all(surv <= 1)

    def test_censoring_rates(self):
        rng = np.random.default_rng(23)
        n = 300
        x = rng.normal(0, 1, n)
        time = rng.exponential(1.0, n)
        censor_time = rng.exponential(0.5, n)
        event = (time <= censor_time).astype(float)
        time = np.minimum(time, censor_time)
        data = {"y": time, "x": x, "event": event}
        model = GAM("y ~ s(x)", family=CoxPH(status="event"))
        model.fit(data)
        assert model.is_fitted

    def test_tied_times_breslow(self):
        rng = np.random.default_rng(23)
        n = 200
        x = rng.normal(0, 1, n)
        time = np.round(rng.exponential(1.0, n), 1)
        event = rng.binomial(1, 0.7, n).astype(float)
        data = {"y": time, "x": x, "event": event}
        model = GAM("y ~ s(x)", family=CoxPH(status="event", ties="breslow"))
        model.fit(data)
        assert model.is_fitted

    def test_tied_times_efron(self):
        rng = np.random.default_rng(23)
        n = 200
        x = rng.normal(0, 1, n)
        time = np.round(rng.exponential(1.0, n), 1)
        event = rng.binomial(1, 0.7, n).astype(float)
        data = {"y": time, "x": x, "event": event}
        model = GAM("y ~ s(x)", family=CoxPH(status="event", ties="efron"))
        model.fit(data)
        assert model.is_fitted

    def test_summary(self):
        data = _make_simple_cox_data(n=200)
        model = GAM("y ~ s(x)", family=CoxPH(status="event"))
        model.fit(data)
        s = model.summary()
        assert "CoxPH" in s

    def test_reml(self):
        data = _make_simple_cox_data(n=200)
        model = GAM("y ~ s(x)", family=CoxPH(status="event"))
        model.fit(data, method="REML")
        assert model.is_fitted

    def test_simulate(self):
        data = _make_simple_cox_data(n=200)
        model = GAM("y ~ s(x)", family=CoxPH(status="event"))
        model.fit(data)
        sim = model.simulate(n_sim=1, seed=0, unconditional=True)
        assert sim.shape == (200, 1)
        assert np.all(sim > 0)
