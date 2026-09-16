"""Tests for simultaneous confidence bands."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.gam import GAM, SimultaneousCIResult


def _sin_data(n=300, seed=23):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2 * np.pi, n)
    y = np.sin(x) + rng.normal(0, 0.2, n)
    return {"x": x, "y": y}


def _two_smooth_data(n=300, seed=23):
    rng = np.random.default_rng(seed)
    x1 = np.linspace(0, 2 * np.pi, n)
    x2 = rng.uniform(0, 1, n)
    y = np.sin(x1) + x2**2 + rng.normal(0, 0.2, n)
    return {"x1": x1, "x2": x2, "y": y}


class TestSimultaneousCI:
    def test_basic(self):
        data = _sin_data()
        model = GAM("y ~ s(x)")
        model.fit(data)
        result = model.simultaneous_ci(data)

        assert "estimate" in result
        assert "se" in result
        assert "lower" in result
        assert "upper" in result
        assert "crit_value" in result
        assert result["estimate"].shape == (300,)
        assert result["lower"].shape == (300,)
        assert result["upper"].shape == (300,)

    def test_bands_wider_than_pointwise(self):
        data = _sin_data()
        model = GAM("y ~ s(x)")
        model.fit(data)
        sim = model.simultaneous_ci(data, level=0.95, seed=0)
        pred = model.predict(data, se=True, interval="confidence", level=0.95)
        sim_width = sim["upper"] - sim["lower"]
        pw_width = pred.upper - pred.lower

        assert np.mean(sim_width) > np.mean(pw_width)

    def test_crit_value_increases_with_level(self):
        data = _sin_data()
        model = GAM("y ~ s(x)")
        model.fit(data)
        ci_90 = model.simultaneous_ci(data, level=0.90, seed=0)
        ci_99 = model.simultaneous_ci(data, level=0.99, seed=0)

        assert ci_99["crit_value"] > ci_90["crit_value"]

    def test_bands_contain_truth(self):
        data = _sin_data(n=500)
        model = GAM("y ~ s(x)")
        model.fit(data)
        result = model.simultaneous_ci(data, level=0.95, n_sim=5000, seed=0)
        truth = np.sin(data["x"])
        estimate = result["estimate"]
        deviation = truth - estimate
        np.all(
            (deviation >= result["lower"] - estimate) & (deviation <= result["upper"] - estimate)
        )

        # With 95% bands, the true function should usually be within
        # We check that at least most points are covered
        coverage = np.mean((truth >= result["lower"]) & (truth <= result["upper"]))

        assert coverage > 0.8

    def test_term_by_index(self):
        data = _two_smooth_data()
        model = GAM("y ~ s(x1) + s(x2)")
        model.fit(data)
        result = model.simultaneous_ci(data, term=0)

        assert "s(x1)" in result["term_label"]

    def test_term_by_name(self):
        data = _two_smooth_data()
        model = GAM("y ~ s(x1) + s(x2)")
        model.fit(data)
        result = model.simultaneous_ci(data, term="s(x2)")

        assert "x2" in result["term_label"]

    def test_term_required_for_multi_smooth(self):
        data = _two_smooth_data()
        model = GAM("y ~ s(x1) + s(x2)")
        model.fit(data)
        with pytest.raises(ValueError, match="specify which one"):
            model.simultaneous_ci(data)

    def test_invalid_term_raises(self):
        data = _sin_data()
        model = GAM("y ~ s(x)")
        model.fit(data)
        with pytest.raises(ValueError, match="No smooth term"):
            model.simultaneous_ci(data, term="s(z)")

    def test_se_shape(self):
        data = _sin_data()
        model = GAM("y ~ s(x)")
        model.fit(data)
        result = model.simultaneous_ci(data)

        assert np.all(result["se"] >= 0)
        assert result["se"].shape == (300,)


class TestSimultaneousCIResult:
    """Tests for the SimultaneousCIResult dataclass."""

    @pytest.fixture()
    def result(self):
        data = _sin_data()
        model = GAM("y ~ s(x)")
        model.fit(data)
        return model.simultaneous_ci(data)

    def test_return_type(self, result):
        assert isinstance(result, SimultaneousCIResult)

    def test_attribute_access(self, result):
        assert isinstance(result.estimate, np.ndarray)
        assert isinstance(result.se, np.ndarray)
        assert isinstance(result.lower, np.ndarray)
        assert isinstance(result.upper, np.ndarray)
        assert isinstance(result.term_label, str)
        assert isinstance(result.crit_value, float)

    def test_n_points_property(self, result):
        assert result.n_points == 300

    def test_repr_contains_term_label_and_crit_value(self, result):
        r = repr(result)

        assert result.term_label in r
        assert "crit_value" in r

    def test_dict_style_access(self, result):
        np.testing.assert_array_equal(result["estimate"], result.estimate)
        np.testing.assert_array_equal(result["se"], result.se)
        np.testing.assert_array_equal(result["lower"], result.lower)
        np.testing.assert_array_equal(result["upper"], result.upper)
        assert result["term_label"] == result.term_label
        assert result["crit_value"] == result.crit_value

    def test_contains(self, result):
        assert "estimate" in result
        assert "se" in result
        assert "lower" in result
        assert "upper" in result
        assert "term_label" in result
        assert "crit_value" in result
        assert "bogus" not in result

    def test_keys(self, result):
        expected = ("estimate", "se", "lower", "upper", "term_label", "crit_value")

        assert result.keys() == expected

    def test_invalid_key_raises(self, result):
        with pytest.raises(KeyError, match="bogus"):
            result["bogus"]


class TestSimultaneousPredict:
    def test_predict_simultaneous_interval(self):
        data = _sin_data()
        model = GAM("y ~ s(x)")
        model.fit(data)
        pred = model.predict(data, interval="simultaneous", level=0.95)

        assert pred.lower is not None
        assert pred.upper is not None
        assert pred.lower.shape == (300,)
        assert np.all(pred.upper >= pred.lower)

    def test_simultaneous_wider_than_confidence(self):
        data = _sin_data()
        model = GAM("y ~ s(x)")
        model.fit(data)
        sim = model.predict(data, interval="simultaneous", level=0.95)
        pw = model.predict(data, interval="confidence", level=0.95)
        sim_width = np.mean(sim.upper - sim.lower)
        pw_width = np.mean(pw.upper - pw.lower)

        assert sim_width > pw_width

    def test_link_scale(self):
        data = _sin_data()
        model = GAM("y ~ s(x)")
        model.fit(data)
        pred = model.predict(data, interval="simultaneous", type="link")

        assert pred.lower is not None
        assert np.all(pred.upper >= pred.lower)

    def test_invalid_interval_type(self):
        data = _sin_data()
        model = GAM("y ~ s(x)")
        model.fit(data)
        with pytest.raises(ValueError, match="Unknown interval"):
            model.predict(data, interval="bogus")
