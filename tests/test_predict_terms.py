"""Tests for predict(type='terms') and predict(type='link')."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.families import Gaussian, Poisson
from whittaker.gam import GAM, PredictionResult, TermsPredictionResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def two_smooth_data():
    rng = np.random.default_rng(23)
    n = 200
    x1 = rng.uniform(0, 2 * np.pi, n)
    x2 = rng.uniform(0, 2 * np.pi, n)
    y = np.sin(x1) + 0.5 * np.cos(x2) + rng.normal(0, 0.3, n)
    return {"x1": x1, "x2": x2, "y": y}


@pytest.fixture()
def single_smooth_data():
    rng = np.random.default_rng(23)
    n = 150
    x = rng.uniform(0, 2 * np.pi, n)
    y = np.sin(x) + rng.normal(0, 0.3, n)
    return {"x": x, "y": y}


# ---------------------------------------------------------------------------
# type="terms"
# ---------------------------------------------------------------------------


class TestPredictTerms:
    def test_returns_terms_result(self, two_smooth_data):
        gam = GAM("y ~ s(x1) + s(x2)", family=Gaussian()).fit(two_smooth_data)
        result = gam.predict(two_smooth_data, type="terms")
        assert isinstance(result, TermsPredictionResult)

    def test_one_entry_per_smooth(self, two_smooth_data):
        gam = GAM("y ~ s(x1) + s(x2)", family=Gaussian()).fit(two_smooth_data)
        result = gam.predict(two_smooth_data, type="terms")
        assert len(result.terms) == 2
        assert len(result.labels) == 2

    def test_term_shapes(self, two_smooth_data):
        gam = GAM("y ~ s(x1) + s(x2)", family=Gaussian()).fit(two_smooth_data)
        result = gam.predict(two_smooth_data, type="terms")
        n = len(two_smooth_data["y"])
        for label, vals in result.terms.items():
            assert vals.shape == (n,)

    def test_terms_sum_to_linear_predictor(self, two_smooth_data):
        gam = GAM("y ~ s(x1) + s(x2)", family=Gaussian()).fit(two_smooth_data)
        pred_response = gam.predict(two_smooth_data)
        pred_terms = gam.predict(two_smooth_data, type="terms")
        intercept = gam._fit_result.coefficients[0]
        terms_sum = sum(pred_terms.terms.values()) + intercept
        np.testing.assert_allclose(terms_sum, pred_response.linear_predictor, atol=1e-10)

    def test_se_none_by_default(self, two_smooth_data):
        gam = GAM("y ~ s(x1) + s(x2)", family=Gaussian()).fit(two_smooth_data)
        result = gam.predict(two_smooth_data, type="terms")
        assert result.se is None

    def test_se_computed_when_requested(self, two_smooth_data):
        gam = GAM("y ~ s(x1) + s(x2)", family=Gaussian()).fit(two_smooth_data)
        result = gam.predict(two_smooth_data, type="terms", se=True)
        assert result.se is not None
        assert len(result.se) == 2
        for label, se_vals in result.se.items():
            assert se_vals.shape == (len(two_smooth_data["y"]),)
            assert np.all(se_vals >= 0)

    def test_se_finite(self, two_smooth_data):
        gam = GAM("y ~ s(x1) + s(x2)", family=Gaussian()).fit(two_smooth_data)
        result = gam.predict(two_smooth_data, type="terms", se=True)
        for se_vals in result.se.values():
            assert np.isfinite(se_vals).all()

    def test_single_smooth(self, single_smooth_data):
        gam = GAM("y ~ s(x)", family=Gaussian()).fit(single_smooth_data)
        result = gam.predict(single_smooth_data, type="terms")
        assert len(result.terms) == 1

    def test_labels_match_keys(self, two_smooth_data):
        gam = GAM("y ~ s(x1) + s(x2)", family=Gaussian()).fit(two_smooth_data)
        result = gam.predict(two_smooth_data, type="terms")
        assert list(result.terms.keys()) == result.labels

    def test_on_new_data(self, two_smooth_data):
        gam = GAM("y ~ s(x1) + s(x2)", family=Gaussian()).fit(two_smooth_data)
        rng = np.random.default_rng(99)
        new_data = {
            "x1": rng.uniform(0, 2 * np.pi, 50),
            "x2": rng.uniform(0, 2 * np.pi, 50),
        }
        result = gam.predict(new_data, type="terms")
        assert len(result.terms) == 2
        for vals in result.terms.values():
            assert vals.shape == (50,)
            assert np.isfinite(vals).all()

    def test_poisson_terms(self):
        rng = np.random.default_rng(23)
        n = 200
        x = rng.uniform(0, 2 * np.pi, n)
        mu = np.exp(0.5 * np.sin(x))
        y = rng.poisson(mu)
        data = {"x": x, "y": y}
        gam = GAM("y ~ s(x)", family=Poisson()).fit(data)
        result = gam.predict(data, type="terms")
        assert len(result.terms) == 1
        for vals in result.terms.values():
            assert np.isfinite(vals).all()

    def test_with_by_factor(self):
        rng = np.random.default_rng(23)
        n = 120
        x = rng.uniform(0, 2 * np.pi, n)
        group = np.repeat(["A", "B"], n // 2)
        y = np.where(group == "A", np.sin(x), np.cos(x)) + rng.normal(0, 0.3, n)
        data = {"x": x, "y": y, "group": group}
        gam = GAM("y ~ s(x, by=group)", family=Gaussian()).fit(data)
        result = gam.predict(data, type="terms")
        assert len(result.terms) == 2

    def test_with_random_effect(self):
        rng = np.random.default_rng(23)
        n = 100
        group = np.repeat(np.arange(5).astype(str), 20)
        x = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.5, n)
        data = {"x": x, "y": y, "group": group}
        gam = GAM("y ~ s(x) + s(group, bs='re')", family=Gaussian()).fit(data)
        result = gam.predict(data, type="terms")
        assert len(result.terms) == 2

    def test_terms_contributions_nonzero(self, two_smooth_data):
        gam = GAM("y ~ s(x1) + s(x2)", family=Gaussian()).fit(two_smooth_data)
        result = gam.predict(two_smooth_data, type="terms")
        for vals in result.terms.values():
            assert np.std(vals) > 0.01

    def test_unfitted_raises(self):
        gam = GAM("y ~ s(x)", family=Gaussian())
        with pytest.raises(RuntimeError, match="fitted"):
            gam.predict({"x": np.ones(10)}, type="terms")


# ---------------------------------------------------------------------------
# type="link"
# ---------------------------------------------------------------------------


class TestPredictLink:
    def test_returns_prediction_result(self, single_smooth_data):
        gam = GAM("y ~ s(x)", family=Gaussian()).fit(single_smooth_data)
        result = gam.predict(single_smooth_data, type="link")
        assert isinstance(result, PredictionResult)

    def test_link_equals_linear_predictor(self, single_smooth_data):
        gam = GAM("y ~ s(x)", family=Gaussian()).fit(single_smooth_data)
        result = gam.predict(single_smooth_data, type="link")
        np.testing.assert_array_equal(result.values, result.linear_predictor)

    def test_link_differs_from_response_for_poisson(self):
        rng = np.random.default_rng(23)
        n = 200
        x = rng.uniform(0, 2 * np.pi, n)
        y = rng.poisson(np.exp(0.5 * np.sin(x)))
        data = {"x": x, "y": y}
        gam = GAM("y ~ s(x)", family=Poisson()).fit(data)
        link = gam.predict(data, type="link")
        response = gam.predict(data, type="response")
        assert not np.allclose(link.values, response.values)
        np.testing.assert_allclose(np.exp(link.values), response.values, atol=1e-10)

    def test_link_with_se(self, single_smooth_data):
        gam = GAM("y ~ s(x)", family=Gaussian()).fit(single_smooth_data)
        result = gam.predict(single_smooth_data, type="link", se=True)
        assert result.se is not None
        assert np.all(result.se >= 0)

    def test_gaussian_link_equals_response(self, single_smooth_data):
        gam = GAM("y ~ s(x)", family=Gaussian()).fit(single_smooth_data)
        link = gam.predict(single_smooth_data, type="link")
        response = gam.predict(single_smooth_data, type="response")
        np.testing.assert_allclose(link.values, response.values, atol=1e-10)


# ---------------------------------------------------------------------------
# type validation
# ---------------------------------------------------------------------------


class TestPredictTypeValidation:
    def test_unknown_type_raises(self, single_smooth_data):
        gam = GAM("y ~ s(x)", family=Gaussian()).fit(single_smooth_data)
        with pytest.raises(ValueError, match="Unknown prediction type"):
            gam.predict(single_smooth_data, type="unknown")

    def test_default_is_response(self, single_smooth_data):
        gam = GAM("y ~ s(x)", family=Gaussian()).fit(single_smooth_data)
        default = gam.predict(single_smooth_data)
        explicit = gam.predict(single_smooth_data, type="response")
        np.testing.assert_array_equal(default.values, explicit.values)
