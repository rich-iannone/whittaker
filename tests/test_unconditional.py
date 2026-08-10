"""Tests for unconditional confidence intervals (Marra & Wood 2012)."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.families import Gamma, Gaussian, Poisson
from whittaker.gam import GAM

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_data():
    rng = np.random.default_rng(23)
    n = 200
    x = rng.uniform(0, 2 * np.pi, n)
    y = np.sin(x) + rng.normal(0, 0.3, n)
    return {"x": x, "y": y}


@pytest.fixture()
def small_data():
    rng = np.random.default_rng(23)
    n = 50
    x = rng.uniform(0, 2 * np.pi, n)
    y = np.sin(x) + rng.normal(0, 0.5, n)
    return {"x": x, "y": y}


@pytest.fixture()
def two_smooth_data():
    rng = np.random.default_rng(23)
    n = 300
    x1 = rng.uniform(0, 2 * np.pi, n)
    x2 = rng.uniform(0, 2 * np.pi, n)
    y = np.sin(x1) + 0.5 * np.cos(x2) + rng.normal(0, 0.3, n)
    return {"x1": x1, "x2": x2, "y": y}


# ---------------------------------------------------------------------------
# Basic unconditional SE
# ---------------------------------------------------------------------------


class TestUnconditionalSE:
    def test_unconditional_se_wider_than_conditional(self, simple_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(simple_data, method="REML")
        res_c = gam.predict(simple_data, se=True)
        res_u = gam.predict(simple_data, se=True, unconditional=True)
        assert np.all(res_u.se >= res_c.se - 1e-10)

    def test_unconditional_se_finite(self, simple_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(simple_data, method="REML")
        res = gam.predict(simple_data, se=True, unconditional=True)
        assert np.isfinite(res.se).all()
        assert np.all(res.se > 0)

    def test_unconditional_larger_for_small_n(self, small_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(small_data, method="REML")
        res_c = gam.predict(small_data, se=True)
        res_u = gam.predict(small_data, se=True, unconditional=True)
        ratio = np.mean(res_u.se) / np.mean(res_c.se)
        assert ratio > 1.005

    def test_predictions_unchanged(self, simple_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(simple_data, method="REML")
        res_c = gam.predict(simple_data, se=True)
        res_u = gam.predict(simple_data, se=True, unconditional=True)
        np.testing.assert_allclose(res_c.values, res_u.values)


# ---------------------------------------------------------------------------
# Unconditional confidence intervals
# ---------------------------------------------------------------------------


class TestUnconditionalIntervals:
    def test_unconditional_ci_wider(self, simple_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(simple_data, method="REML")
        ci_c = gam.predict(simple_data, interval="confidence")
        ci_u = gam.predict(simple_data, interval="confidence", unconditional=True)
        width_c = np.mean(ci_c.upper - ci_c.lower)
        width_u = np.mean(ci_u.upper - ci_u.lower)
        assert width_u >= width_c - 1e-10

    def test_unconditional_ci_bounds_finite(self, simple_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(simple_data, method="REML")
        ci = gam.predict(simple_data, interval="confidence", unconditional=True)
        assert np.isfinite(ci.lower).all()
        assert np.isfinite(ci.upper).all()
        assert np.all(ci.lower <= ci.upper)

    def test_unconditional_prediction_interval(self, simple_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(simple_data, method="REML")
        pi = gam.predict(simple_data, interval="prediction", unconditional=True)
        ci = gam.predict(simple_data, interval="confidence", unconditional=True)
        assert np.all((pi.upper - pi.lower) >= (ci.upper - ci.lower) - 1e-10)

    def test_link_scale_intervals(self, simple_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(simple_data, method="REML")
        ci = gam.predict(simple_data, type="link", interval="confidence", unconditional=True)
        assert ci.lower is not None
        assert np.all(ci.lower <= ci.upper)


# ---------------------------------------------------------------------------
# Method validation
# ---------------------------------------------------------------------------


class TestMethodValidation:
    def test_gcv_raises(self, simple_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(simple_data, method="GCV")
        with pytest.raises(ValueError, match="REML.*ML"):
            gam.predict(simple_data, se=True, unconditional=True)

    def test_reml_works(self, simple_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(simple_data, method="REML")
        res = gam.predict(simple_data, se=True, unconditional=True)
        assert res.se is not None

    def test_ml_works(self, simple_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(simple_data, method="ML")
        res = gam.predict(simple_data, se=True, unconditional=True)
        assert res.se is not None
        assert np.isfinite(res.se).all()


# ---------------------------------------------------------------------------
# Multiple smooths
# ---------------------------------------------------------------------------


class TestUnconditionalMultipleSmooths:
    def test_two_smooths_se(self, two_smooth_data):
        gam = GAM("y ~ s(x1) + s(x2)", family=Gaussian())
        gam.fit(two_smooth_data, method="REML")
        res_c = gam.predict(two_smooth_data, se=True)
        res_u = gam.predict(two_smooth_data, se=True, unconditional=True)
        assert np.all(res_u.se >= res_c.se - 1e-10)
        ratio = np.mean(res_u.se) / np.mean(res_c.se)
        assert ratio > 1.0

    def test_two_smooths_ci(self, two_smooth_data):
        gam = GAM("y ~ s(x1) + s(x2)", family=Gaussian())
        gam.fit(two_smooth_data, method="REML")
        ci_u = gam.predict(two_smooth_data, interval="confidence", unconditional=True)
        assert np.all(ci_u.lower <= ci_u.upper)


# ---------------------------------------------------------------------------
# Non-Gaussian families
# ---------------------------------------------------------------------------


class TestUnconditionalNonGaussian:
    def test_poisson(self):
        rng = np.random.default_rng(23)
        n = 200
        x = rng.uniform(0, 2 * np.pi, n)
        y = rng.poisson(np.exp(0.5 * np.sin(x)))
        data = {"x": x, "y": y}
        gam = GAM("y ~ s(x)", family=Poisson())
        gam.fit(data, method="REML")
        res_c = gam.predict(data, se=True)
        res_u = gam.predict(data, se=True, unconditional=True)
        assert np.all(res_u.se >= res_c.se - 1e-10)
        ratio = np.mean(res_u.se) / np.mean(res_c.se)
        assert ratio > 1.01

    def test_gamma(self):
        rng = np.random.default_rng(23)
        n = 200
        x = rng.uniform(0, 2 * np.pi, n)
        mu = np.exp(np.sin(x) + 1)
        y = rng.gamma(5, mu / 5)
        data = {"x": x, "y": y}
        gam = GAM("y ~ s(x)", family=Gamma())
        gam.fit(data, method="REML")
        res_c = gam.predict(data, se=True)
        res_u = gam.predict(data, se=True, unconditional=True)
        assert np.all(res_u.se >= res_c.se - 1e-10)


# ---------------------------------------------------------------------------
# Terms prediction with unconditional
# ---------------------------------------------------------------------------


class TestUnconditionalTerms:
    def test_terms_se_wider(self, two_smooth_data):
        gam = GAM("y ~ s(x1) + s(x2)", family=Gaussian())
        gam.fit(two_smooth_data, method="REML")
        terms_c = gam.predict(two_smooth_data, type="terms", se=True)
        terms_u = gam.predict(two_smooth_data, type="terms", se=True, unconditional=True)
        for label in terms_c.labels:
            se_c = terms_c.se[label]
            se_u = terms_u.se[label]
            assert np.all(se_u >= se_c - 1e-10)

    def test_terms_values_unchanged(self, two_smooth_data):
        gam = GAM("y ~ s(x1) + s(x2)", family=Gaussian())
        gam.fit(two_smooth_data, method="REML")
        terms_c = gam.predict(two_smooth_data, type="terms")
        terms_u = gam.predict(two_smooth_data, type="terms", unconditional=True)
        for label in terms_c.labels:
            np.testing.assert_allclose(terms_c.terms[label], terms_u.terms[label])


# ---------------------------------------------------------------------------
# With select, weights, offset
# ---------------------------------------------------------------------------


class TestUnconditionalCombinations:
    def test_with_select(self, simple_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(simple_data, method="REML", select=True)
        res = gam.predict(simple_data, se=True, unconditional=True)
        assert np.isfinite(res.se).all()

    def test_with_weights(self, simple_data):
        n = len(simple_data["y"])
        w = np.ones(n) * 2.0
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(simple_data, method="REML", weights=w)
        res = gam.predict(simple_data, se=True, unconditional=True)
        assert np.isfinite(res.se).all()

    def test_with_offset(self):
        rng = np.random.default_rng(23)
        n = 200
        x = rng.uniform(0, 2 * np.pi, n)
        log_exposure = rng.uniform(0, 1, n)
        y = rng.poisson(np.exp(0.5 * np.sin(x) + log_exposure))
        data = {"x": x, "y": y, "log_exposure": log_exposure}
        gam = GAM("y ~ s(x) + offset(log_exposure)", family=Poisson())
        gam.fit(data, method="REML")
        res_c = gam.predict(data, se=True)
        res_u = gam.predict(data, se=True, unconditional=True)
        assert np.all(res_u.se >= res_c.se - 1e-10)

    def test_new_data(self, simple_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(simple_data, method="REML")
        new = {"x": np.linspace(0, 2 * np.pi, 50)}
        res = gam.predict(new, se=True, unconditional=True)
        assert res.se.shape == (50,)
        assert np.isfinite(res.se).all()
