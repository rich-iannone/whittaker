"""Tests for the multinomial (baseline-category logit) family."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.families.multinomial import Multinomial
from whittaker.gam import GAM


class TestMultinomialFamily:
    def test_n_categories(self):
        fam = Multinomial(n_categories=3)
        assert fam.n_categories == 3

    def test_minimum_categories(self):
        with pytest.raises(ValueError, match="n_categories"):
            Multinomial(n_categories=1)

    def test_category_intercepts_none_before_fit(self):
        fam = Multinomial(n_categories=3)
        assert fam.category_intercepts is None
        assert fam.category_loadings is None

    def test_init_params_sets_intercepts_and_loadings(self):
        fam = Multinomial(n_categories=3)
        y = np.array([1, 1, 2, 2, 3, 3], dtype=float)
        fam._init_params(y)
        assert fam.category_intercepts is not None
        assert len(fam.category_intercepts) == 2
        assert len(fam.category_loadings) == 2
        np.testing.assert_allclose(fam.category_loadings, 1.0)

    def test_link_is_identity(self):
        fam = Multinomial(n_categories=3)
        mu = np.array([0.1, -0.2, 0.5])
        np.testing.assert_array_equal(fam.link(mu), mu)

    def test_link_inverse_is_identity(self):
        fam = Multinomial(n_categories=3)
        eta = np.array([0.1, -0.2, 0.5])
        np.testing.assert_array_equal(fam.link_inverse(eta), eta)

    def test_link_derivative_is_ones(self):
        fam = Multinomial(n_categories=3)
        mu = np.array([0.1, -0.2, 0.5])
        result = fam.link_derivative(mu)
        np.testing.assert_array_equal(result, np.ones_like(mu))

    def test_variance_is_ones(self):
        fam = Multinomial(n_categories=3)
        mu = np.array([0.1, -0.2, 0.5, 1.0])
        result = fam.variance(mu)
        np.testing.assert_array_equal(result, np.ones_like(mu))

    def test_scale_known(self):
        fam = Multinomial(n_categories=3)
        assert fam.scale_known is True

    def test_repr(self):
        fam = Multinomial(n_categories=4)
        assert "K=4" in repr(fam)

    def test_category_probs_sum_to_one(self):
        fam = Multinomial(n_categories=3)
        y = np.array([1, 2, 3, 1, 2, 3], dtype=float)
        fam._init_params(y)
        eta = np.linspace(-2, 2, 6)
        probs = fam._category_probs(eta)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-10)

    def test_irls_update_initializes_params_on_first_call(self):
        fam = Multinomial(n_categories=3)
        assert fam.category_intercepts is None
        y = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1], dtype=float)
        eta = np.zeros_like(y)
        z, w = fam.irls_update(y, None, eta)
        assert fam.category_intercepts is not None
        assert z.shape == eta.shape
        assert w.shape == eta.shape
        assert np.all(w > 0)

    def test_deviance_before_fit_returns_len_y(self):
        fam = Multinomial(n_categories=3)
        y = np.array([1, 2, 3, 1, 2], dtype=float)
        assert fam.deviance(y, y) == float(len(y))

    def test_deviance_after_fit(self):
        fam = Multinomial(n_categories=3)
        y = np.array([1, 2, 3, 1, 2, 3], dtype=float)
        fam._init_params(y)
        eta = np.zeros_like(y)
        dev = fam.deviance(y, eta)
        assert dev > 0

    def test_deviance_with_weights_argument(self):
        fam = Multinomial(n_categories=3)
        y = np.array([1, 2, 3, 1, 2, 3], dtype=float)
        fam._init_params(y)
        eta = np.zeros_like(y)
        weights = np.ones_like(y)
        dev_weighted = fam.deviance(y, eta, weights=weights)
        dev_unweighted = fam.deviance(y, eta, weights=None)
        assert dev_weighted == dev_unweighted

    def test_unit_deviance_before_fit_returns_ones(self):
        fam = Multinomial(n_categories=3)
        y = np.array([1, 2, 3, 1, 2], dtype=float)
        result = fam.unit_deviance(y, y)
        np.testing.assert_array_equal(result, np.ones_like(y))

    def test_unit_deviance_after_fit_sums_to_deviance(self):
        fam = Multinomial(n_categories=3)
        y = np.array([1, 2, 3, 1, 2, 3], dtype=float)
        fam._init_params(y)
        eta = np.zeros_like(y)
        unit_dev = fam.unit_deviance(y, eta)
        dev = fam.deviance(y, eta)
        assert unit_dev.shape == y.shape
        np.testing.assert_allclose(unit_dev.sum(), dev)

    def test_log_likelihood(self):
        fam = Multinomial(n_categories=3)
        y = np.array([1, 2, 3, 1, 2, 3], dtype=float)
        fam._init_params(y)
        eta = np.zeros_like(y)
        ll = fam.log_likelihood(y, eta, scale=1.0)
        dev = fam.deviance(y, eta)
        assert ll == -0.5 * dev

    def test_simulate_raises_before_fit(self):
        fam = Multinomial(n_categories=3)
        rng = np.random.default_rng(0)
        eta = np.zeros(5)
        with pytest.raises(RuntimeError, match="must be fitted"):
            fam.simulate(eta, scale=1.0, rng=rng)

    def test_simulate_after_fit(self):
        fam = Multinomial(n_categories=3)
        y = np.array([1, 2, 3, 1, 2, 3], dtype=float)
        fam._init_params(y)
        eta = np.zeros_like(y)
        rng = np.random.default_rng(0)
        sims = fam.simulate(eta, scale=1.0, rng=rng)
        assert sims.shape == eta.shape
        assert np.all(np.isin(sims, [1, 2, 3]))

    def test_initialize_returns_zeros_and_sets_params(self):
        fam = Multinomial(n_categories=3)
        y = np.array([1, 2, 3, 1, 2, 3], dtype=float)
        eta0 = fam.initialize(y)
        np.testing.assert_array_equal(eta0, np.zeros_like(y))
        assert fam.category_intercepts is not None


class TestMultinomialGAM:
    @pytest.fixture()
    def multinomial_data(self):
        rng = np.random.default_rng(7)
        n = 200
        x = np.linspace(-3, 3, n)
        eta = np.sin(x)

        alphas = np.array([0.0, 0.5])
        betas = np.array([1.0, -1.5])

        logits = np.column_stack(
            [alphas[0] + betas[0] * eta, alphas[1] + betas[1] * eta, np.zeros(n)]
        )
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        y = np.array([rng.choice([1, 2, 3], p=probs[i]) for i in range(n)], dtype=float)

        return {"x": x, "y": y}

    def test_converges(self, multinomial_data):
        model = GAM("y ~ s(x)", family=Multinomial(n_categories=3))
        model.fit(multinomial_data)
        assert model.is_fitted

    def test_simulate_from_gam(self, multinomial_data):
        model = GAM("y ~ s(x)", family=Multinomial(n_categories=3))
        model.fit(multinomial_data)
        sims = model.simulate(n_sim=2, seed=7, unconditional=True)
        assert sims.shape == (len(multinomial_data["y"]), 2)
        assert np.all(np.isin(sims, [1, 2, 3]))
