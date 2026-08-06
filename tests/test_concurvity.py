"""Tests for concurvity diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from whittaker.families.poisson import Poisson
from whittaker.fitting.inference import ConcurvityResult, concurvity
from whittaker.gam import GAM
from whittaker.model_matrix import build_model_matrix
from whittaker.fitting.pirls import pirls_fit
from whittaker.families.gaussian import Gaussian
from whittaker.formula.parser import parse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fit_two_smooth(rng, n=200, corr=0.0):
    """Fit a two-smooth GAM with optionally correlated covariates."""
    x1 = rng.uniform(size=n)
    if corr > 0:
        x2 = corr * x1 + (1 - corr) * rng.uniform(size=n)
    else:
        x2 = rng.uniform(size=n)
    y = np.sin(3 * x1) + np.cos(3 * x2) + 0.2 * rng.standard_normal(n)
    model = GAM("y ~ s(x1, k=6) + s(x2, k=6)").fit({"y": y, "x1": x1, "x2": x2})
    return model


# ---------------------------------------------------------------------------
# Full concurvity (overall)
# ---------------------------------------------------------------------------


class TestFullConcurvity:
    def test_returns_concurvity_result(self) -> None:
        model = _fit_two_smooth(np.random.default_rng(42))
        c = model.concurvity(full=True)
        assert isinstance(c, ConcurvityResult)
        assert c.full is True

    def test_shape_matches_n_smooths(self) -> None:
        model = _fit_two_smooth(np.random.default_rng(42))
        c = model.concurvity(full=True)
        assert c.worst.shape == (2,)
        assert c.observed.shape == (2,)
        assert c.estimate.shape == (2,)

    def test_labels_populated(self) -> None:
        model = _fit_two_smooth(np.random.default_rng(42))
        c = model.concurvity(full=True)
        assert len(c.labels) == 2
        assert "s(x1" in c.labels[0]
        assert "s(x2" in c.labels[1]

    def test_values_between_zero_and_one(self) -> None:
        model = _fit_two_smooth(np.random.default_rng(42))
        c = model.concurvity(full=True)
        for arr in (c.worst, c.observed, c.estimate):
            assert np.all(arr >= 0.0)
            assert np.all(arr <= 1.0)

    def test_low_concurvity_independent(self) -> None:
        model = _fit_two_smooth(np.random.default_rng(42), corr=0.0)
        c = model.concurvity(full=True)
        assert np.all(c.worst < 0.3)
        assert np.all(c.observed < 0.3)

    def test_high_concurvity_correlated(self) -> None:
        model = _fit_two_smooth(np.random.default_rng(42), corr=0.95)
        c = model.concurvity(full=True)
        assert np.all(c.worst > 0.8)

    def test_worst_geq_observed(self) -> None:
        model = _fit_two_smooth(np.random.default_rng(42))
        c = model.concurvity(full=True)
        assert np.all(c.worst >= c.observed - 1e-10)

    def test_three_smooths(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        x3 = rng.uniform(size=n)
        y = np.sin(x1) + np.cos(x2) + x3 + 0.2 * rng.standard_normal(n)

        model = GAM("y ~ s(x1, k=5) + s(x2, k=5) + s(x3, k=5)").fit(
            {"y": y, "x1": x1, "x2": x2, "x3": x3}
        )
        c = model.concurvity(full=True)
        assert c.worst.shape == (3,)
        assert len(c.labels) == 3


# ---------------------------------------------------------------------------
# Pairwise concurvity
# ---------------------------------------------------------------------------


class TestPairwiseConcurvity:
    def test_returns_pairwise_result(self) -> None:
        model = _fit_two_smooth(np.random.default_rng(42))
        c = model.concurvity(full=False)
        assert isinstance(c, ConcurvityResult)
        assert c.full is False

    def test_shape_is_square(self) -> None:
        model = _fit_two_smooth(np.random.default_rng(42))
        c = model.concurvity(full=False)
        assert c.worst.shape == (2, 2)
        assert c.observed.shape == (2, 2)
        assert c.estimate.shape == (2, 2)

    def test_diagonal_is_one(self) -> None:
        model = _fit_two_smooth(np.random.default_rng(42))
        c = model.concurvity(full=False)
        for arr in (c.worst, c.observed, c.estimate):
            assert_allclose(np.diag(arr), 1.0)

    def test_off_diagonal_bounded(self) -> None:
        model = _fit_two_smooth(np.random.default_rng(42))
        c = model.concurvity(full=False)
        for arr in (c.worst, c.observed, c.estimate):
            assert np.all(arr >= 0.0)
            assert np.all(arr <= 1.0)

    def test_pairwise_symmetric_worst(self) -> None:
        model = _fit_two_smooth(np.random.default_rng(42))
        c = model.concurvity(full=False)
        assert_allclose(c.worst[0, 1], c.worst[1, 0], atol=1e-10)

    def test_pairwise_high_when_correlated(self) -> None:
        model = _fit_two_smooth(np.random.default_rng(42), corr=0.95)
        c = model.concurvity(full=False)
        assert c.worst[0, 1] > 0.8
        assert c.worst[1, 0] > 0.8

    def test_three_smooths_pairwise(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        x3 = rng.uniform(size=n)
        y = np.sin(x1) + np.cos(x2) + x3 + 0.2 * rng.standard_normal(n)

        model = GAM("y ~ s(x1, k=5) + s(x2, k=5) + s(x3, k=5)").fit(
            {"y": y, "x1": x1, "x2": x2, "x3": x3}
        )
        c = model.concurvity(full=False)
        assert c.worst.shape == (3, 3)
        assert_allclose(np.diag(c.worst), 1.0)


# ---------------------------------------------------------------------------
# Edge cases and integration
# ---------------------------------------------------------------------------


class TestConcurvityIntegration:
    def test_single_smooth_full(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        x = rng.uniform(size=n)
        y = np.sin(3 * x) + 0.1 * rng.standard_normal(n)

        model = GAM("y ~ s(x, k=6)").fit({"y": y, "x": x})
        c = model.concurvity(full=True)
        assert c.worst.shape == (1,)

    def test_single_smooth_pairwise(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        x = rng.uniform(size=n)
        y = np.sin(3 * x) + 0.1 * rng.standard_normal(n)

        model = GAM("y ~ s(x, k=6)").fit({"y": y, "x": x})
        c = model.concurvity(full=False)
        assert c.worst.shape == (1, 1)
        assert_allclose(c.worst[0, 0], 1.0)

    def test_tensor_product_smooth(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = np.sin(x1) + x1 * x2 + 0.2 * rng.standard_normal(n)

        model = GAM("y ~ s(x1, k=5) + te(x1, x2, k=4)").fit(
            {"y": y, "x1": x1, "x2": x2}
        )
        c = model.concurvity(full=True)
        assert c.worst.shape == (2,)
        assert np.all(np.isfinite(c.worst))

    def test_ti_interaction_smooth(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = np.sin(x1) + np.cos(x2) + x1 * x2 + 0.2 * rng.standard_normal(n)

        model = GAM("y ~ s(x1, k=5) + s(x2, k=5) + ti(x1, x2, k=4)").fit(
            {"y": y, "x1": x1, "x2": x2}
        )
        c = model.concurvity(full=True)
        assert c.worst.shape == (3,)
        assert np.all(np.isfinite(c.worst))

    def test_poisson_family(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        mu = np.exp(0.5 + x1 + 0.5 * x2)
        y = rng.poisson(mu).astype(float)

        model = GAM("y ~ s(x1, k=5) + s(x2, k=5)", family=Poisson()).fit(
            {"y": y, "x1": x1, "x2": x2}
        )
        c = model.concurvity(full=True)
        assert c.worst.shape == (2,)
        for arr in (c.worst, c.observed, c.estimate):
            assert np.all(arr >= 0.0)
            assert np.all(arr <= 1.0)

    def test_with_parametric_terms(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        z = rng.standard_normal(n)
        y = np.sin(3 * x1) + 0.5 * z + 0.2 * rng.standard_normal(n)

        model = GAM("y ~ z + s(x1, k=6) + s(x2, k=6)").fit(
            {"y": y, "x1": x1, "x2": x2, "z": z}
        )
        c = model.concurvity(full=True)
        assert c.worst.shape == (2,)

    def test_unfitted_raises(self) -> None:
        model = GAM("y ~ s(x)")
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.concurvity()

    def test_reml_method(self) -> None:
        model = _fit_two_smooth(np.random.default_rng(42))
        c = model.concurvity(full=True)
        assert np.all(np.isfinite(c.worst))


# ---------------------------------------------------------------------------
# Low-level function tests
# ---------------------------------------------------------------------------


class TestConcurvityLowLevel:
    def test_via_fitting_api(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = np.sin(3 * x1) + np.cos(3 * x2) + 0.2 * rng.standard_normal(n)

        formula = parse("y ~ s(x1, k=6) + s(x2, k=6)")
        mm = build_model_matrix(formula, {"y": y, "x1": x1, "x2": x2})
        fit = pirls_fit(mm, Gaussian())
        c = concurvity(fit, mm, full=True)

        assert isinstance(c, ConcurvityResult)
        assert c.worst.shape == (2,)

    def test_observed_equals_estimate_for_centered(self) -> None:
        model = _fit_two_smooth(np.random.default_rng(42))
        c = model.concurvity(full=True)
        assert_allclose(c.observed, c.estimate, atol=1e-10)
