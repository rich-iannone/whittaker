"""Tests for Tweedie with estimated power parameter."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.families.tweedie import Tweedie
from whittaker.families.tweedie_estimated import TweedieEstimated, tw
from whittaker.gam import GAM


def _tweedie_data(p: float = 1.5, n: int = 500, seed: int = 42):
    """Generate Tweedie data with known p for testing."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 4, n)
    mu = np.exp(0.5 + 0.3 * np.sin(x))

    scale = 1.0
    lam = mu ** (2 - p) / (scale * (2 - p))
    alpha = (2 - p) / (p - 1)
    gamma_scale = scale * (p - 1) * mu ** (p - 1)

    N = rng.poisson(lam)
    y = np.zeros(n)
    for i in range(n):
        if N[i] > 0:
            y[i] = np.sum(rng.gamma(alpha, gamma_scale[i], size=N[i]))

    return {"x": x, "y": y}


class TestTweedieEstimatedFamily:
    def test_tw_factory(self):
        family = tw()
        assert isinstance(family, TweedieEstimated)
        assert not family.p_estimated

    def test_tw_custom_range(self):
        family = tw(p_range=(1.2, 1.8), n_grid=10)
        assert family._p_range == (1.2, 1.8)
        assert family._n_grid == 10

    def test_repr_before_estimation(self):
        family = tw()
        r = repr(family)
        assert "p=?" in r

    def test_repr_after_estimation(self):
        family = tw()
        family._set_p(1.5)
        r = repr(family)
        assert "estimated=True" in r
        assert "1.5" in r

    def test_inherits_tweedie(self):
        family = tw()
        assert isinstance(family, Tweedie)
        mu = np.array([1.0, 2.0, 3.0])
        assert family.variance(mu).shape == (3,)


class TestTweedieEstimatedGAM:
    def test_converges(self):
        data = _tweedie_data(p=1.5, n=500)
        model = GAM("y ~ s(x)", family=tw())
        model.fit(data)
        assert model.is_fitted

    def test_p_is_estimated(self):
        data = _tweedie_data(p=1.5, n=500)
        family = tw()
        model = GAM("y ~ s(x)", family=family)
        model.fit(data)
        assert family.p_estimated
        assert 1.0 < family.p < 2.0

    def test_estimated_p_reasonable(self):
        data = _tweedie_data(p=1.5, n=800)
        family = tw(n_grid=30)
        model = GAM("y ~ s(x)", family=family)
        model.fit(data)
        assert abs(family.p - 1.5) < 0.5

    def test_predict(self):
        data = _tweedie_data(p=1.5, n=500)
        model = GAM("y ~ s(x)", family=tw())
        model.fit(data)
        pred = model.predict(data)
        assert pred.values.shape == (500,)
        assert np.all(pred.values >= 0)

    def test_summary(self):
        data = _tweedie_data(p=1.5, n=300)
        model = GAM("y ~ s(x)", family=tw())
        model.fit(data)
        s = model.summary()
        assert "Tweedie" in s

    def test_custom_range(self):
        data = _tweedie_data(p=1.7, n=500)
        family = tw(p_range=(1.5, 1.99))
        model = GAM("y ~ s(x)", family=family)
        model.fit(data)
        assert family.p_estimated
        assert 1.5 <= family.p <= 1.99

    def test_fixed_vs_estimated(self):
        data = _tweedie_data(p=1.5, n=500)
        fixed = GAM("y ~ s(x)", family=Tweedie(p=1.5))
        fixed.fit(data)
        estimated = GAM("y ~ s(x)", family=tw())
        estimated.fit(data)
        pred_fixed = fixed.predict(data).values
        pred_est = estimated.predict(data).values
        assert np.corrcoef(pred_fixed, pred_est)[0, 1] > 0.9

    def test_reml_method(self):
        data = _tweedie_data(p=1.5, n=300)
        model = GAM("y ~ s(x)", family=tw())
        model.fit(data, method="REML")
        assert model.is_fitted
