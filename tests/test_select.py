"""Tests for select=True (double penalty smooth selection)."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.families import Gaussian, Poisson
from whittaker.formula.parser import parse
from whittaker.gam import GAM
from whittaker.model_matrix import build_model_matrix

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def signal_and_noise_data():
    rng = np.random.default_rng(23)
    n = 300
    x1 = rng.uniform(0, 2 * np.pi, n)
    x2 = rng.uniform(0, 2 * np.pi, n)
    x3 = rng.uniform(0, 2 * np.pi, n)
    y = np.sin(x1) + rng.normal(0, 0.3, n)
    return {"x1": x1, "x2": x2, "x3": x3, "y": y}


@pytest.fixture()
def simple_data():
    rng = np.random.default_rng(23)
    n = 200
    x = rng.uniform(0, 2 * np.pi, n)
    y = np.sin(x) + rng.normal(0, 0.3, n)
    return {"x": x, "y": y}


# ---------------------------------------------------------------------------
# Model matrix level: penalty structure
# ---------------------------------------------------------------------------


class TestSelectPenaltyStructure:
    def test_extra_penalty_added(self):
        rng = np.random.default_rng(23)
        n = 100
        x = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.3, n)
        data = {"x": x, "y": y}

        formula = parse("y ~ s(x)")
        mm_no = build_model_matrix(formula, data, select=False)
        mm_sel = build_model_matrix(formula, data, select=True)

        assert len(mm_sel.penalties) == len(mm_no.penalties) + 1

    def test_null_space_dim_zero_with_select(self):
        rng = np.random.default_rng(23)
        n = 100
        x = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.3, n)
        data = {"x": x, "y": y}

        formula = parse("y ~ s(x)")
        mm = build_model_matrix(formula, data, select=True)
        for info in mm.smooths:
            assert info.null_space_dim == 0

    def test_penalty_indices_updated(self):
        rng = np.random.default_rng(23)
        n = 100
        x = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.3, n)
        data = {"x": x, "y": y}

        formula = parse("y ~ s(x)")
        mm = build_model_matrix(formula, data, select=True)
        assert len(mm.smooths[0].penalty_indices) == 2

    def test_two_smooths_get_extra_penalties(self):
        rng = np.random.default_rng(23)
        n = 100
        x1 = rng.uniform(0, 2 * np.pi, n)
        x2 = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x1) + rng.normal(0, 0.3, n)
        data = {"x1": x1, "x2": x2, "y": y}

        formula = parse("y ~ s(x1) + s(x2)")
        mm_no = build_model_matrix(formula, data, select=False)
        mm_sel = build_model_matrix(formula, data, select=True)

        assert len(mm_sel.penalties) == len(mm_no.penalties) + 2

    def test_shrinkage_basis_unaffected(self):
        rng = np.random.default_rng(23)
        n = 100
        x = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.3, n)
        data = {"x": x, "y": y}

        formula = parse("y ~ s(x, bs='ts')")
        mm_no = build_model_matrix(formula, data, select=False)
        mm_sel = build_model_matrix(formula, data, select=True)

        assert len(mm_sel.penalties) == len(mm_no.penalties)

    def test_re_basis_unaffected(self):
        rng = np.random.default_rng(23)
        n = 100
        group = np.repeat(np.arange(5).astype(str), 20)
        x = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.3, n)
        data = {"x": x, "y": y, "group": group}

        formula = parse("y ~ s(x) + s(group, bs='re')")
        mm_no = build_model_matrix(formula, data, select=False)
        mm_sel = build_model_matrix(formula, data, select=True)

        assert len(mm_sel.penalties) == len(mm_no.penalties) + 1

    def test_null_penalty_symmetric_psd(self):
        rng = np.random.default_rng(23)
        n = 100
        x = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.3, n)
        data = {"x": x, "y": y}

        formula = parse("y ~ s(x)")
        mm = build_model_matrix(formula, data, select=True)
        S_null = mm.penalties[-1]
        np.testing.assert_allclose(S_null, S_null.T, atol=1e-14)
        eigvals = np.linalg.eigvalsh(S_null)
        assert np.all(eigvals >= -1e-12)


# ---------------------------------------------------------------------------
# GAM fitting with select=True
# ---------------------------------------------------------------------------


class TestSelectFitting:
    def test_fit_with_select(self, simple_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(simple_data, select=True)
        assert gam.is_fitted

    def test_fit_reml_with_select(self, simple_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(simple_data, select=True, method="REML")
        assert gam.is_fitted
        assert np.isfinite(gam.coefficients).all()

    def test_predict_after_select(self, simple_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(simple_data, select=True)
        result = gam.predict(simple_data)
        assert np.isfinite(result.values).all()

    def test_predict_se_after_select(self, simple_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(simple_data, select=True)
        result = gam.predict(simple_data, se=True)
        assert result.se is not None
        assert np.isfinite(result.se).all()

    def test_predict_interval_after_select(self, simple_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(simple_data, select=True)
        result = gam.predict(simple_data, interval="confidence")
        assert result.lower is not None
        assert np.all(result.lower <= result.upper)

    def test_summary_after_select(self, simple_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(simple_data, select=True)
        summary = gam.summary()
        assert "GAM fit summary" in summary

    def test_smooth_tests_after_select(self, simple_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(simple_data, select=True)
        tests = gam.smooth_tests()
        assert len(tests) == 1
        assert tests[0].edf > 0

    def test_deviance_explained(self, simple_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(simple_data, select=True)
        assert gam.deviance_explained > 0.5

    def test_poisson_with_select(self):
        rng = np.random.default_rng(23)
        n = 200
        x = rng.uniform(0, 2 * np.pi, n)
        y = rng.poisson(np.exp(0.5 * np.sin(x)))
        data = {"x": x, "y": y}
        gam = GAM("y ~ s(x)", family=Poisson())
        gam.fit(data, select=True)
        assert gam.is_fitted
        result = gam.predict(data)
        assert np.all(result.values > 0)


# ---------------------------------------------------------------------------
# Smooth selection: noise terms get shrunk
# ---------------------------------------------------------------------------


class TestSmoothSelection:
    def test_noise_term_lower_edf(self, signal_and_noise_data):
        gam = GAM("y ~ s(x1) + s(x2) + s(x3)", family=Gaussian())
        gam.fit(signal_and_noise_data, select=True, method="REML")
        edf = gam.edf
        assert edf[0] > edf[1]
        assert edf[0] > edf[2]

    def test_noise_edf_near_zero(self, signal_and_noise_data):
        gam = GAM("y ~ s(x1) + s(x2) + s(x3)", family=Gaussian())
        gam.fit(signal_and_noise_data, select=True, method="REML")
        edf = gam.edf
        assert edf[1] < 2.0
        assert edf[2] < 2.0

    def test_signal_term_retains_edf(self, signal_and_noise_data):
        gam = GAM("y ~ s(x1) + s(x2) + s(x3)", family=Gaussian())
        gam.fit(signal_and_noise_data, select=True, method="REML")
        assert gam.edf[0] > 2.0

    def test_select_vs_no_select_similar_deviance_for_signal(self, simple_data):
        gam_no = GAM("y ~ s(x)", family=Gaussian())
        gam_no.fit(simple_data)
        gam_sel = GAM("y ~ s(x)", family=Gaussian())
        gam_sel.fit(simple_data, select=True)
        assert abs(gam_sel.deviance_explained - gam_no.deviance_explained) < 0.1

    def test_select_more_penalties_than_no_select(self, simple_data):
        gam_no = GAM("y ~ s(x)", family=Gaussian())
        gam_no.fit(simple_data)
        gam_sel = GAM("y ~ s(x)", family=Gaussian())
        gam_sel.fit(simple_data, select=True)
        assert len(gam_sel.smoothing_params) > len(gam_no.smoothing_params)


# ---------------------------------------------------------------------------
# select with weights
# ---------------------------------------------------------------------------


class TestSelectWithWeights:
    def test_select_with_weights(self, simple_data):
        n = len(simple_data["y"])
        w = np.ones(n) * 2.0
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(simple_data, select=True, weights=w)
        assert gam.is_fitted


# ---------------------------------------------------------------------------
# select with offset
# ---------------------------------------------------------------------------


class TestSelectWithOffset:
    def test_select_with_offset(self):
        rng = np.random.default_rng(23)
        n = 200
        x = rng.uniform(0, 2 * np.pi, n)
        log_exposure = rng.uniform(0, 2, n)
        y = rng.poisson(np.exp(0.5 * np.sin(x) + log_exposure))
        data = {"x": x, "y": y, "log_exposure": log_exposure}
        gam = GAM("y ~ s(x) + offset(log_exposure)", family=Poisson())
        gam.fit(data, select=True)
        assert gam.is_fitted
        result = gam.predict(data)
        assert np.all(result.values > 0)


# ---------------------------------------------------------------------------
# Multiple basis types with select
# ---------------------------------------------------------------------------


class TestSelectMixedBases:
    def test_cr_basis_with_select(self, simple_data):
        gam = GAM("y ~ s(x, bs='cr')", family=Gaussian())
        gam.fit(simple_data, select=True)
        assert gam.is_fitted

    def test_ps_basis_with_select(self, simple_data):
        gam = GAM("y ~ s(x, bs='ps')", family=Gaussian())
        gam.fit(simple_data, select=True)
        assert gam.is_fitted

    def test_mixed_basis_types(self):
        rng = np.random.default_rng(23)
        n = 200
        x1 = rng.uniform(0, 2 * np.pi, n)
        x2 = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x1) + 0.5 * np.cos(x2) + rng.normal(0, 0.3, n)
        data = {"x1": x1, "x2": x2, "y": y}
        gam = GAM("y ~ s(x1, bs='tp') + s(x2, bs='cr')", family=Gaussian())
        gam.fit(data, select=True)
        assert gam.is_fitted
        assert len(gam.smoothing_params) == 4
