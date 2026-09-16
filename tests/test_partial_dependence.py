"""Tests for GAM.partial_dependence() and PartialDependenceResult."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from whittaker.families.poisson import Poisson
from whittaker.gam import GAM, PartialDependenceResult

rng = np.random.default_rng(23)


# ---------------------------------------------------------------------------
# Helpers: build fitted models
# ---------------------------------------------------------------------------


def _single_smooth_model() -> GAM:
    """Gaussian GAM with one smooth term: y ~ s(x)."""
    n = 200
    x = np.linspace(0, 2 * np.pi, n)
    y = np.sin(x) + rng.normal(0, 0.2, n)
    return GAM("y ~ s(x, k=10)").fit({"y": y, "x": x})


def _multi_smooth_model() -> GAM:
    """Gaussian GAM with two smooth terms: y ~ s(x1) + s(x2)."""
    n = 300
    x1 = np.linspace(0, 2 * np.pi, n)
    x2 = np.linspace(0, 1, n)
    y = np.sin(x1) + 0.5 * x2 + rng.normal(0, 0.2, n)
    return GAM("y ~ s(x1, k=10) + s(x2, k=10)").fit({"y": y, "x1": x1, "x2": x2})


def _te_model() -> GAM:
    """Gaussian GAM with a tensor product smooth: y ~ te(x1, x2)."""
    n = 300
    x1 = rng.uniform(size=n)
    x2 = rng.uniform(size=n)
    y = np.sin(3 * x1) * np.cos(3 * x2) + 0.2 * rng.standard_normal(n)
    return GAM("y ~ te(x1, x2, k=5)").fit({"y": y, "x1": x1, "x2": x2})


def _poisson_model() -> GAM:
    """Poisson GAM: y ~ s(x)."""
    n = 300
    x = np.linspace(0, 2 * np.pi, n)
    mu = np.exp(0.5 + 0.5 * np.sin(x))
    y = rng.poisson(mu).astype(float)
    return GAM("y ~ s(x, k=10)", family=Poisson()).fit({"y": y, "x": x})


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestSingleSmooth:
    """Tests for a model with a single 1-D smooth."""

    def test_returns_list_of_length_one(self) -> None:
        model = _single_smooth_model()
        results = model.partial_dependence()

        assert isinstance(results, list)
        assert len(results) == 1

    def test_result_type(self) -> None:
        model = _single_smooth_model()
        result = model.partial_dependence()[0]

        assert isinstance(result, PartialDependenceResult)

    def test_default_shapes(self) -> None:
        model = _single_smooth_model()
        result = model.partial_dependence()
        r = result[0]

        assert r.effect.shape == (200,)
        assert r.se.shape == (200,)
        assert r.lower.shape == (200,)
        assert r.upper.shape == (200,)

    def test_x_dict_has_one_key(self) -> None:
        model = _single_smooth_model()
        r = model.partial_dependence()[0]

        assert len(r.x) == 1
        assert "x" in r.x
        assert r.x["x"].shape == (200,)

    def test_n_grid_property(self) -> None:
        model = _single_smooth_model()
        r = model.partial_dependence()[0]

        assert r.n_grid == 200

    def test_term_label(self) -> None:
        model = _single_smooth_model()
        r = model.partial_dependence()[0]

        assert "x" in r.term
        assert r.term.startswith("s(")

    def test_level_stored(self) -> None:
        model = _single_smooth_model()
        r = model.partial_dependence()[0]

        assert r.level == 0.95

    def test_edf_positive(self) -> None:
        model = _single_smooth_model()
        r = model.partial_dependence()[0]

        assert r.edf > 0.0

    def test_lower_below_upper(self) -> None:
        model = _single_smooth_model()
        r = model.partial_dependence()[0]

        npt.assert_array_less(r.lower, r.upper)


class TestMultiTerm:
    """Tests for a model with two smooth terms."""

    def test_returns_two_results(self) -> None:
        model = _multi_smooth_model()
        results = model.partial_dependence()

        assert len(results) == 2

    def test_term_labels_distinct(self) -> None:
        model = _multi_smooth_model()
        results = model.partial_dependence()
        labels = [r.term for r in results]

        assert len(set(labels)) == 2

    def test_term_labels_contain_variable_names(self) -> None:
        model = _multi_smooth_model()
        results = model.partial_dependence()
        labels = [r.term for r in results]

        assert any("x1" in lab for lab in labels)
        assert any("x2" in lab for lab in labels)

    def test_each_result_has_correct_x_key(self) -> None:
        model = _multi_smooth_model()
        results = model.partial_dependence()
        for r in results:
            assert len(r.x) == 1

            # The key should match the variable in the term label
            var_name = list(r.x.keys())[0]

            assert var_name in r.term


class TestCustomNPoints:
    """Tests that custom n_points is reflected in output shapes."""

    def test_custom_n_points_shapes(self) -> None:
        model = _single_smooth_model()
        n = 50
        r = model.partial_dependence(n_points=n)[0]

        assert r.effect.shape == (n,)
        assert r.se.shape == (n,)
        assert r.lower.shape == (n,)
        assert r.upper.shape == (n,)
        assert r.n_grid == n

    def test_custom_n_points_x_grid(self) -> None:
        model = _single_smooth_model()
        n = 50
        r = model.partial_dependence(n_points=n)[0]

        assert r.x["x"].shape == (n,)

    def test_different_n_points_different_lengths(self) -> None:
        model = _single_smooth_model()
        r1 = model.partial_dependence(n_points=50)[0]
        r2 = model.partial_dependence(n_points=100)[0]

        assert r1.n_grid != r2.n_grid


class TestCustomLevel:
    """Tests for the confidence level parameter."""

    def test_level_stored_correctly(self) -> None:
        model = _single_smooth_model()
        r = model.partial_dependence(level=0.99)[0]

        assert r.level == 0.99

    def test_wider_bands_at_higher_level(self) -> None:
        model = _single_smooth_model()
        r95 = model.partial_dependence(level=0.95)[0]
        r99 = model.partial_dependence(level=0.99)[0]

        width_95 = r95.upper - r95.lower
        width_99 = r99.upper - r99.lower

        # 99% bands should be wider than 95% bands everywhere
        npt.assert_array_less(width_95, width_99)

    def test_effects_unchanged_across_levels(self) -> None:
        model = _single_smooth_model()
        r95 = model.partial_dependence(level=0.95)[0]
        r99 = model.partial_dependence(level=0.99)[0]

        npt.assert_array_equal(r95.effect, r99.effect)

    def test_se_unchanged_across_levels(self) -> None:
        model = _single_smooth_model()
        r95 = model.partial_dependence(level=0.95)[0]
        r99 = model.partial_dependence(level=0.99)[0]

        npt.assert_array_equal(r95.se, r99.se)


class TestTensorProduct2D:
    """Tests for 2-D tensor product smooths."""

    def test_x_dict_has_two_keys(self) -> None:
        model = _te_model()
        r = model.partial_dependence()[0]

        assert len(r.x) == 2
        assert "x1" in r.x
        assert "x2" in r.x

    def test_effect_length_matches_grid(self) -> None:
        model = _te_model()
        n_points = 100
        r = model.partial_dependence(n_points=n_points)[0]
        n_side = max(int(np.ceil(np.sqrt(n_points))), 15)

        # For 2-D smooths, the effect length should be n_side^2
        assert r.n_grid == n_side**2

    def test_marginal_grid_lengths(self) -> None:
        model = _te_model()
        n_points = 100
        r = model.partial_dependence(n_points=n_points)[0]
        n_side = max(int(np.ceil(np.sqrt(n_points))), 15)

        assert r.x["x1"].shape == (n_side,)
        assert r.x["x2"].shape == (n_side,)

    def test_shapes_consistent(self) -> None:
        model = _te_model()
        r = model.partial_dependence()[0]

        assert r.se.shape == r.effect.shape
        assert r.lower.shape == r.effect.shape
        assert r.upper.shape == r.effect.shape

    def test_returns_single_result(self) -> None:
        model = _te_model()
        results = model.partial_dependence()

        assert len(results) == 1


class TestUnfittedModel:
    """Tests that an unfitted model raises an error."""

    def test_raises_runtime_error(self) -> None:
        model = GAM("y ~ s(x, k=10)")
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.partial_dependence()


class TestRepr:
    """Tests for the __repr__ of PartialDependenceResult."""

    def test_repr_format(self) -> None:
        model = _single_smooth_model()
        r = model.partial_dependence()[0]
        text = repr(r)

        assert text.startswith("PartialDependenceResult(")
        assert "term=" in text
        assert "n_grid=" in text
        assert "edf=" in text
        assert "level=" in text

    def test_repr_shows_correct_n_grid(self) -> None:
        model = _single_smooth_model()
        r = model.partial_dependence(n_points=50)[0]
        text = repr(r)

        assert "n_grid=50" in text

    def test_repr_shows_level(self) -> None:
        model = _single_smooth_model()
        r = model.partial_dependence(level=0.99)[0]
        text = repr(r)

        assert "level=0.99" in text


class TestPoissonFamily:
    """Tests that partial_dependence works with Poisson (non-Gaussian) family."""

    def test_poisson_returns_results(self) -> None:
        model = _poisson_model()
        results = model.partial_dependence()

        assert len(results) == 1

    def test_poisson_finite_effects(self) -> None:
        model = _poisson_model()
        r = model.partial_dependence()[0]

        assert np.all(np.isfinite(r.effect))
        assert np.all(np.isfinite(r.se))
        assert np.all(np.isfinite(r.lower))
        assert np.all(np.isfinite(r.upper))

    def test_poisson_shapes(self) -> None:
        model = _poisson_model()
        r = model.partial_dependence(n_points=100)[0]

        assert r.effect.shape == (100,)
        assert r.se.shape == (100,)


class TestConsistencyWithPredictTerms:
    """Effects from partial_dependence should be consistent with predict(type='terms')."""

    def test_effects_loosely_match_predict_terms(self) -> None:
        n = 300
        local_rng = np.random.default_rng(23)
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + local_rng.normal(0, 0.2, n)
        model = GAM("y ~ s(x, k=10)").fit({"y": y, "x": x})

        # Get partial dependence evaluated at training points
        pd_results = model.partial_dependence(n_points=n)
        pd_effect = pd_results[0].effect

        # Get predict(type="terms") at the same training data
        terms_result = model.predict({"y": y, "x": x}, type="terms")
        term_key = list(terms_result.terms.keys())[0]
        predict_effect = terms_result.terms[term_key]

        # They represent the same smooth contribution but are evaluated on different
        # grids (evenly spaced vs. original data), so we compare their ranges
        pd_range = np.ptp(pd_effect)
        predict_range = np.ptp(predict_effect)

        npt.assert_allclose(pd_range, predict_range, rtol=0.15)


class TestSEPositive:
    """Standard errors should be strictly positive."""

    def test_se_all_positive_single(self) -> None:
        model = _single_smooth_model()
        r = model.partial_dependence()[0]

        assert np.all(r.se > 0)

    def test_se_all_positive_multi(self) -> None:
        model = _multi_smooth_model()
        results = model.partial_dependence()
        for r in results:
            assert np.all(r.se > 0)

    def test_se_all_positive_tensor(self) -> None:
        model = _te_model()
        r = model.partial_dependence()[0]

        assert np.all(r.se > 0)

    def test_se_all_positive_poisson(self) -> None:
        model = _poisson_model()
        r = model.partial_dependence()[0]

        assert np.all(r.se > 0)
