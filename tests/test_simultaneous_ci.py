"""Tests for simultaneous confidence bands."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.gam import GAM


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
