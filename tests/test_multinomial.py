"""Tests for multinomial logistic family."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.families.multinomial import Multinomial, _softmax
from whittaker.gam import GAM


def _make_multinomial_data(n=500, K=3, seed=23):
    """Generate multinomial data with category probabilities depending on x."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-3, 3, n)
    logits = np.zeros((n, K))
    logits[:, 0] = -1.0 + 0.8 * x
    logits[:, 1] = 0.5 - 0.3 * x
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(np.arange(1, K + 1), p=probs[i]) for i in range(n)], dtype=float)
    return {"x": x, "y": y}


class TestSoftmax:
    def test_sums_to_one(self):
        logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
        p = _softmax(logits)
        np.testing.assert_allclose(p.sum(axis=1), 1.0)

    def test_equal_logits(self):
        logits = np.array([[1.0, 1.0, 1.0]])
        p = _softmax(logits)
        np.testing.assert_allclose(p, 1.0 / 3.0)

    def test_numerical_stability(self):
        logits = np.array([[1000.0, 1001.0, 999.0]])
        p = _softmax(logits)
        assert np.all(np.isfinite(p))
        np.testing.assert_allclose(p.sum(), 1.0)


class TestMultinomialFamily:
    def test_init(self):
        fam = Multinomial(n_categories=3)
        assert fam.n_categories == 3

    def test_invalid_k(self):
        with pytest.raises(ValueError, match="n_categories must be >= 2"):
            Multinomial(n_categories=1)

    def test_scale_known(self):
        fam = Multinomial(n_categories=3)
        assert fam.scale_known is True

    def test_initialize(self):
        fam = Multinomial(n_categories=3)
        y = np.array([1, 2, 3, 1, 2, 3], dtype=float)
        mu0 = fam.initialize(y)
        assert mu0.shape == (6,)
        assert fam.category_intercepts is not None
        assert fam.category_loadings is not None

    def test_repr(self):
        fam = Multinomial(n_categories=4)
        assert repr(fam) == "Multinomial(K=4)"


class TestMultinomialGAM:
    def test_converges_3_categories(self):
        data = _make_multinomial_data(n=500, K=3)
        model = GAM("y ~ s(x)", family=Multinomial(n_categories=3))
        model.fit(data)
        assert model.is_fitted

    def test_converges_4_categories(self):
        rng = np.random.default_rng(23)
        n = 400
        x = np.linspace(-2, 2, n)
        K = 4
        logits = np.zeros((n, K))
        logits[:, 0] = -0.5 + 0.5 * x
        logits[:, 1] = 0.3 - 0.2 * x
        logits[:, 2] = -0.1 + 0.4 * x**2
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        y = np.array([rng.choice(np.arange(1, K + 1), p=probs[i]) for i in range(n)], dtype=float)
        data = {"x": x, "y": y}
        model = GAM("y ~ s(x)", family=Multinomial(n_categories=K))
        model.fit(data)
        assert model.is_fitted

    def test_predict(self):
        data = _make_multinomial_data(n=400, K=3)
        model = GAM("y ~ s(x)", family=Multinomial(n_categories=3))
        model.fit(data)
        pred = model.predict(data)
        assert pred.values.shape == (400,)
        assert np.all(np.isfinite(pred.values))

    def test_summary(self):
        data = _make_multinomial_data(n=300, K=3)
        model = GAM("y ~ s(x)", family=Multinomial(n_categories=3))
        model.fit(data)
        s = model.summary()
        assert "Multinomial" in s

    def test_simulate(self):
        data = _make_multinomial_data(n=300, K=3)
        model = GAM("y ~ s(x)", family=Multinomial(n_categories=3))
        model.fit(data)
        sim = model.simulate(n_sim=1, seed=0, unconditional=True)
        assert sim.shape == (300, 1)
        assert np.all(np.isin(sim.ravel(), [1, 2, 3]))

    def test_binary_case(self):
        rng = np.random.default_rng(23)
        n = 300
        x = np.linspace(-3, 3, n)
        p = 1.0 / (1.0 + np.exp(-x))
        y = rng.binomial(1, p).astype(float) + 1.0
        data = {"x": x, "y": y}
        model = GAM("y ~ s(x)", family=Multinomial(n_categories=2))
        model.fit(data)
        assert model.is_fitted

    def test_deviance_decreases(self):
        data = _make_multinomial_data(n=400, K=3)
        model = GAM("y ~ s(x)", family=Multinomial(n_categories=3))
        model.fit(data)
        assert model.deviance < model.null_deviance
