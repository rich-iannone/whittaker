"""Tests for the full diagnostic suite."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.families.poisson import Poisson
from whittaker.gam import GAM


@pytest.fixture
def gaussian_model():
    rng = np.random.default_rng(23)
    x = np.linspace(0, 2 * np.pi, 200)
    y = np.sin(x) + rng.normal(0, 0.2, 200)
    data = {"x": x, "y": y}
    model = GAM("y ~ s(x)")
    model.fit(data)
    return model, data


@pytest.fixture
def poisson_model():
    rng = np.random.default_rng(23)
    x = np.linspace(0, 3, 300)
    y = rng.poisson(np.exp(0.5 * x))
    data = {"x": x, "y": y.astype(float)}
    model = GAM("y ~ s(x)", family=Poisson())
    model.fit(data)
    return model, data


@pytest.fixture
def multi_param_model():
    rng = np.random.default_rng(23)
    n = 300
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    y = 2.0 * x1 - 1.0 * x2 + 0.5 * x3 + rng.normal(0, 0.5, n)
    data = {"x1": x1, "x2": x2, "x3": x3, "y": y}
    model = GAM("y ~ x1 + x2 + x3")
    model.fit(data)
    return model, data


class TestInfluence:
    def test_hat_values_shape(self, gaussian_model):
        model, _ = gaussian_model
        result = model.influence()
        assert result.hat_values.shape == (200,)

    def test_hat_values_range(self, gaussian_model):
        model, _ = gaussian_model
        result = model.influence()
        assert np.all(result.hat_values >= 0)
        assert np.all(result.hat_values <= 1)

    def test_hat_values_sum_edf(self, gaussian_model):
        model, _ = gaussian_model
        result = model.influence()
        hat_sum = np.sum(result.hat_values)
        np.testing.assert_allclose(hat_sum, model.edf_total, rtol=0.1)

    def test_cooks_distance_shape(self, gaussian_model):
        model, _ = gaussian_model
        result = model.influence()
        assert result.cooks_distance.shape == (200,)
        assert np.all(result.cooks_distance >= 0)

    def test_cooks_distance_finite(self, gaussian_model):
        model, _ = gaussian_model
        result = model.influence()
        assert np.all(np.isfinite(result.cooks_distance))

    def test_poisson_influence(self, poisson_model):
        model, _ = poisson_model
        result = model.influence()
        assert result.hat_values.shape == (300,)
        assert np.all(result.hat_values >= 0)


class TestQuantileResiduals:
    def test_gaussian_shape(self, gaussian_model):
        model, _ = gaussian_model
        qr = model.quantile_residuals(seed=23)
        assert qr.shape == (200,)

    def test_gaussian_approx_normal(self, gaussian_model):
        model, _ = gaussian_model
        qr = model.quantile_residuals(seed=23)
        assert abs(np.mean(qr)) < 0.3
        assert abs(np.std(qr) - 1.0) < 0.3

    def test_poisson_shape(self, poisson_model):
        model, _ = poisson_model
        qr = model.quantile_residuals(seed=23)
        assert qr.shape == (300,)

    def test_poisson_approx_normal(self, poisson_model):
        model, _ = poisson_model
        qr = model.quantile_residuals(seed=23)
        assert abs(np.mean(qr)) < 0.5
        assert np.std(qr) > 0.3

    def test_finite(self, gaussian_model):
        model, _ = gaussian_model
        qr = model.quantile_residuals(seed=23)
        assert np.all(np.isfinite(qr))


class TestDispersionTest:
    def test_poisson_dispersion(self, poisson_model):
        model, _ = poisson_model
        result = model.dispersion_test()
        assert result.dispersion > 0
        assert result.chi2_stat > 0

    def test_gaussian_dispersion(self, gaussian_model):
        model, _ = gaussian_model
        result = model.dispersion_test()
        assert result.dispersion > 0

    def test_dispersion_near_one_for_poisson(self):
        rng = np.random.default_rng(23)
        n = 1000
        x = np.linspace(0, 2, n)
        mu = np.exp(0.5 * x)
        y = rng.poisson(mu).astype(float)
        data = {"x": x, "y": y}
        model = GAM("y ~ s(x)", family=Poisson())
        model.fit(data)
        result = model.dispersion_test()
        assert 0.5 < result.dispersion < 2.0


class TestVIF:
    def test_returns_results(self, multi_param_model):
        model, _ = multi_param_model
        results = model.vif()
        assert len(results) == 3

    def test_term_names(self, multi_param_model):
        model, _ = multi_param_model
        results = model.vif()
        names = [r.term for r in results]
        assert "x1" in names
        assert "x2" in names
        assert "x3" in names

    def test_independent_vars_low_vif(self, multi_param_model):
        model, _ = multi_param_model
        results = model.vif()
        for r in results:
            assert r.vif < 5.0

    def test_collinear_vars_high_vif(self):
        rng = np.random.default_rng(23)
        n = 300
        x1 = rng.normal(0, 1, n)
        x2 = x1 + rng.normal(0, 0.01, n)
        y = x1 + rng.normal(0, 0.5, n)
        data = {"x1": x1, "x2": x2, "y": y}
        model = GAM("y ~ x1 + x2")
        model.fit(data)
        results = model.vif()
        assert any(r.vif > 100 for r in results)

    def test_single_param_returns_empty(self, gaussian_model):
        rng = np.random.default_rng(23)
        n = 200
        x = rng.normal(0, 1, n)
        y = x + rng.normal(0, 0.5, n)
        data = {"x": x, "y": y}
        model = GAM("y ~ x")
        model.fit(data)
        results = model.vif()
        assert results == []

    def test_smooth_only_returns_empty(self, gaussian_model):
        model, _ = gaussian_model
        results = model.vif()
        assert results == []


class TestExistingDiagnostics:
    def test_get_residuals_types(self, gaussian_model):
        model, _ = gaussian_model
        for rtype in ["response", "pearson", "deviance", "working"]:
            r = model.get_residuals(rtype)
            assert r.shape == (200,)
            assert np.all(np.isfinite(r))

    def test_gam_check(self, gaussian_model):
        model, _ = gaussian_model
        result = model.gam_check(n_sim=50)
        assert result.n_obs == 200
        assert result.deviance_explained > 0

    def test_concurvity_full(self):
        rng = np.random.default_rng(23)
        n = 200
        x1 = np.linspace(0, 2 * np.pi, n)
        x2 = rng.uniform(0, 1, n)
        y = np.sin(x1) + x2 + rng.normal(0, 0.2, n)
        data = {"x1": x1, "x2": x2, "y": y}
        model = GAM("y ~ s(x1) + s(x2)")
        model.fit(data)
        result = model.concurvity(full=True)
        assert hasattr(result, "worst")
        assert hasattr(result, "labels")

    def test_smooth_tests(self, gaussian_model):
        model, _ = gaussian_model
        results = model.smooth_tests()
        assert len(results) >= 1
        assert results[0].p_value >= 0

    def test_parametric_tests(self, multi_param_model):
        model, _ = multi_param_model
        results = model.parametric_tests()
        assert len(results) >= 1

    def test_k_check(self, gaussian_model):
        model, _ = gaussian_model
        results = model.k_check(n_sim=50)
        assert len(results) >= 1
        assert results[0].k_index > 0
