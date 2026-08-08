"""Tests for basis dimension adequacy checking (k_check)."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.families import Gaussian, Poisson
from whittaker.gam import GAM

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def adequate_k_data():
    """Data where default k=10 is more than enough for a simple sine."""
    rng = np.random.default_rng(23)
    n = 300
    x = rng.uniform(0, 2 * np.pi, n)
    y = np.sin(x) + rng.normal(0, 0.3, n)
    return {"x": x, "y": y}


@pytest.fixture()
def inadequate_k_data():
    """Highly wiggly function that k=4 cannot capture."""
    rng = np.random.default_rng(23)
    n = 500
    x = rng.uniform(0, 1, n)
    y = np.sin(20 * x) + rng.normal(0, 0.3, n)
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
# Basic functionality
# ---------------------------------------------------------------------------


class TestKCheckBasic:
    def test_returns_list(self, adequate_k_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(adequate_k_data)
        results = gam.k_check()
        assert isinstance(results, list)
        assert len(results) == 1

    def test_result_fields(self, adequate_k_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(adequate_k_data)
        r = gam.k_check()[0]
        assert hasattr(r, "term_label")
        assert hasattr(r, "k_prime")
        assert hasattr(r, "edf")
        assert hasattr(r, "k_index")
        assert hasattr(r, "p_value")

    def test_k_prime_matches_basis_dim(self, adequate_k_data):
        gam = GAM("y ~ s(x, k=15)", family=Gaussian())
        gam.fit(adequate_k_data)
        r = gam.k_check()[0]
        assert r.k_prime == 14  # k=15 minus 1 for identifiability constraint

    def test_edf_matches(self, adequate_k_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(adequate_k_data)
        r = gam.k_check()[0]
        assert r.edf == gam.edf[0]

    def test_k_index_positive(self, adequate_k_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(adequate_k_data)
        r = gam.k_check()[0]
        assert r.k_index > 0

    def test_p_value_in_range(self, adequate_k_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(adequate_k_data)
        r = gam.k_check()[0]
        assert 0 <= r.p_value <= 1

    def test_unfitted_raises(self):
        gam = GAM("y ~ s(x)", family=Gaussian())
        with pytest.raises(RuntimeError):
            gam.k_check()


# ---------------------------------------------------------------------------
# Adequate basis: k-index near 1, p-value not small
# ---------------------------------------------------------------------------


class TestAdequateBasis:
    def test_adequate_k_index_near_one(self, adequate_k_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(adequate_k_data)
        r = gam.k_check(n_sim=200)[0]
        assert r.k_index > 0.5

    def test_adequate_p_value_not_small(self, adequate_k_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(adequate_k_data)
        r = gam.k_check(n_sim=200)[0]
        assert r.p_value > 0.01


# ---------------------------------------------------------------------------
# Inadequate basis: k-index well below 1, p-value small
# ---------------------------------------------------------------------------


class TestInadequateBasis:
    def test_inadequate_k_index_low(self, inadequate_k_data):
        gam = GAM("y ~ s(x, k=4)", family=Gaussian())
        gam.fit(inadequate_k_data)
        r = gam.k_check(n_sim=200)[0]
        assert r.k_index < 0.6

    def test_inadequate_p_value_small(self, inadequate_k_data):
        gam = GAM("y ~ s(x, k=4)", family=Gaussian())
        gam.fit(inadequate_k_data)
        r = gam.k_check(n_sim=200)[0]
        assert r.p_value < 0.1


# ---------------------------------------------------------------------------
# Multiple smooths
# ---------------------------------------------------------------------------


class TestKCheckMultipleSmooths:
    def test_two_smooths(self, two_smooth_data):
        gam = GAM("y ~ s(x1) + s(x2)", family=Gaussian())
        gam.fit(two_smooth_data)
        results = gam.k_check(n_sim=100)
        assert len(results) == 2

    def test_labels_distinct(self, two_smooth_data):
        gam = GAM("y ~ s(x1) + s(x2)", family=Gaussian())
        gam.fit(two_smooth_data)
        results = gam.k_check(n_sim=100)
        labels = [r.term_label for r in results]
        assert len(set(labels)) == 2


# ---------------------------------------------------------------------------
# With different families
# ---------------------------------------------------------------------------


class TestKCheckPoisson:
    def test_poisson_k_check(self):
        rng = np.random.default_rng(23)
        n = 300
        x = rng.uniform(0, 2 * np.pi, n)
        y = rng.poisson(np.exp(0.5 * np.sin(x)))
        data = {"x": x, "y": y}
        gam = GAM("y ~ s(x)", family=Poisson())
        gam.fit(data)
        results = gam.k_check(n_sim=100)
        assert len(results) == 1
        assert results[0].k_index > 0


# ---------------------------------------------------------------------------
# With select, weights, offset
# ---------------------------------------------------------------------------


class TestKCheckCombinations:
    def test_with_select(self, adequate_k_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(adequate_k_data, select=True)
        results = gam.k_check(n_sim=100)
        assert len(results) == 1

    def test_with_weights(self, adequate_k_data):
        n = len(adequate_k_data["y"])
        w = np.ones(n) * 2.0
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(adequate_k_data, weights=w)
        results = gam.k_check(n_sim=100)
        assert len(results) == 1

    def test_with_offset(self):
        rng = np.random.default_rng(23)
        n = 200
        x = rng.uniform(0, 2 * np.pi, n)
        log_exposure = rng.uniform(0, 2, n)
        y = rng.poisson(np.exp(0.5 * np.sin(x) + log_exposure))
        data = {"x": x, "y": y, "log_exposure": log_exposure}
        gam = GAM("y ~ s(x) + offset(log_exposure)", family=Poisson())
        gam.fit(data)
        results = gam.k_check(n_sim=100)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Different basis types
# ---------------------------------------------------------------------------


class TestKCheckBasisTypes:
    def test_cr_basis(self, adequate_k_data):
        gam = GAM("y ~ s(x, bs='cr')", family=Gaussian())
        gam.fit(adequate_k_data)
        results = gam.k_check(n_sim=100)
        assert len(results) == 1
        assert results[0].k_index > 0

    def test_ps_basis(self, adequate_k_data):
        gam = GAM("y ~ s(x, bs='ps')", family=Gaussian())
        gam.fit(adequate_k_data)
        results = gam.k_check(n_sim=100)
        assert len(results) == 1
        assert results[0].k_index > 0


# ---------------------------------------------------------------------------
# n_sim parameter
# ---------------------------------------------------------------------------


class TestKCheckNSim:
    def test_custom_n_sim(self, adequate_k_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(adequate_k_data)
        r = gam.k_check(n_sim=50)[0]
        assert 0 <= r.p_value <= 1

    def test_p_value_resolution(self, adequate_k_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(adequate_k_data)
        r = gam.k_check(n_sim=100)[0]
        assert r.p_value * 100 == int(r.p_value * 100)


# ---------------------------------------------------------------------------
# 2D smooth (tensor)
# ---------------------------------------------------------------------------


class TestKCheck2D:
    def test_tensor_smooth(self):
        rng = np.random.default_rng(23)
        n = 200
        x1 = rng.uniform(0, 2 * np.pi, n)
        x2 = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x1) * np.cos(x2) + rng.normal(0, 0.3, n)
        data = {"x1": x1, "x2": x2, "y": y}
        gam = GAM("y ~ te(x1, x2, k=5)", family=Gaussian())
        gam.fit(data)
        results = gam.k_check(n_sim=50)
        assert len(results) == 1
        assert results[0].k_index > 0
