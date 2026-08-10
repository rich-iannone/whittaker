"""Tests for ordered categorical (proportional odds) family."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.families.ordered_categorical import OrderedCategorical
from whittaker.gam import GAM


class TestOrderedCategoricalFamily:
    def test_n_categories(self):
        fam = OrderedCategorical(n_categories=4)
        assert fam.n_categories == 4

    def test_minimum_categories(self):
        with pytest.raises(ValueError, match="n_categories"):
            OrderedCategorical(n_categories=1)

    def test_init_cutpoints(self):
        fam = OrderedCategorical(n_categories=3)
        y = np.array([1, 1, 2, 2, 3, 3], dtype=float)
        alpha = fam._init_cutpoints(y)
        assert len(alpha) == 2
        assert alpha[0] < alpha[1]

    def test_category_probs_sum_to_one(self):
        fam = OrderedCategorical(n_categories=4)
        y = np.array([1, 2, 3, 4, 1, 2, 3, 4], dtype=float)
        fam._cutpoints = fam._init_cutpoints(y)
        eta = np.array([0.0, 0.5, -0.5, 1.0, -1.0, 0.0, 0.5, -0.5])
        probs = fam._category_probs(eta)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-10)

    def test_category_probs_positive(self):
        fam = OrderedCategorical(n_categories=3)
        y = np.array([1, 2, 3, 1, 2, 3], dtype=float)
        fam._cutpoints = fam._init_cutpoints(y)
        eta = np.linspace(-2, 2, 6)
        probs = fam._category_probs(eta)
        assert np.all(probs > 0)

    def test_higher_eta_shifts_to_higher_categories(self):
        fam = OrderedCategorical(n_categories=3)
        fam._cutpoints = np.array([0.0, 1.0])
        eta_low = np.array([-2.0])
        eta_high = np.array([2.0])
        probs_low = fam._category_probs(eta_low)
        probs_high = fam._category_probs(eta_high)
        assert probs_low[0, 0] > probs_high[0, 0]
        assert probs_low[0, 2] < probs_high[0, 2]

    def test_repr(self):
        fam = OrderedCategorical(n_categories=5)
        assert "K=5" in repr(fam)

    def test_scale_known(self):
        fam = OrderedCategorical(n_categories=3)
        assert fam.scale_known is True


class TestOrderedCategoricalGAM:
    @pytest.fixture()
    def ordinal_data(self):
        rng = np.random.default_rng(23)
        n = 400
        x = np.linspace(-3, 3, n)
        eta_true = 1.5 * x
        cutpoints = np.array([-1.0, 0.5, 2.0])
        K = 4

        y = np.empty(n, dtype=float)
        for i in range(n):
            probs = np.empty(K)
            cum_prev = 0.0
            for k in range(K - 1):
                cum_k = 1.0 / (1.0 + np.exp(-(cutpoints[k] - eta_true[i])))
                probs[k] = max(cum_k - cum_prev, 1e-10)
                cum_prev = cum_k
            probs[K - 1] = max(1.0 - cum_prev, 1e-10)
            probs /= probs.sum()
            y[i] = rng.choice(np.arange(1, K + 1), p=probs)

        return {"x": x, "y": y}, eta_true

    def test_converges(self, ordinal_data):
        data, _ = ordinal_data
        model = GAM("y ~ s(x)", family=OrderedCategorical(n_categories=4))
        model.fit(data)
        assert model.is_fitted

    def test_cutpoints_estimated(self, ordinal_data):
        data, _ = ordinal_data
        fam = OrderedCategorical(n_categories=4)
        model = GAM("y ~ s(x)", family=fam)
        model.fit(data)
        assert fam.cutpoints is not None
        assert len(fam.cutpoints) == 3
        assert np.all(np.diff(fam.cutpoints) > 0)

    def test_predicted_ordering(self, ordinal_data):
        data, eta_true = ordinal_data
        model = GAM("y ~ s(x)", family=OrderedCategorical(n_categories=4))
        model.fit(data)
        pred = model.predict(data)
        eta_pred = pred.values
        assert np.corrcoef(eta_true, eta_pred)[0, 1] > 0.8

    def test_simulate(self, ordinal_data):
        data, _ = ordinal_data
        model = GAM("y ~ s(x)", family=OrderedCategorical(n_categories=4))
        model.fit(data)
        sims = model.simulate(n_sim=3, seed=23, unconditional=True)
        assert sims.shape == (len(data["y"]), 3)
        assert np.all(np.isin(sims, [1, 2, 3, 4]))

    def test_three_categories(self):
        rng = np.random.default_rng(23)
        n = 300
        x = np.linspace(-2, 2, n)
        eta_true = x
        cutpoints = np.array([0.0, 1.5])

        y = np.empty(n, dtype=float)
        for i in range(n):
            p1 = 1.0 / (1.0 + np.exp(-(cutpoints[0] - eta_true[i])))
            p2 = 1.0 / (1.0 + np.exp(-(cutpoints[1] - eta_true[i]))) - p1
            p3 = 1.0 - p1 - p2
            probs = np.maximum([p1, p2, p3], 1e-10)
            probs /= probs.sum()
            y[i] = rng.choice([1, 2, 3], p=probs)

        data = {"x": x, "y": y}
        model = GAM("y ~ s(x)", family=OrderedCategorical(n_categories=3))
        model.fit(data)
        assert model.is_fitted

    def test_summary(self, ordinal_data):
        data, _ = ordinal_data
        model = GAM("y ~ s(x)", family=OrderedCategorical(n_categories=4))
        model.fit(data)
        s = model.summary()
        assert "OrderedCategorical" in s
