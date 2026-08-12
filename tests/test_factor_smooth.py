"""Tests for factor-smooth interaction basis (bs="fs")."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.families import Gaussian, Poisson
from whittaker.formula.parser import parse
from whittaker.gam import GAM
from whittaker.model_matrix import build_model_matrix
from whittaker.smooths.factor_smooth import FactorSmoothBasis

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def panel_data():
    rng = np.random.default_rng(23)
    n_subjects = 4
    n_per = 30
    n = n_subjects * n_per
    subject = np.repeat(np.arange(n_subjects).astype(str), n_per)
    x = np.tile(np.linspace(0, 2 * np.pi, n_per), n_subjects)
    subject_intercepts = rng.normal(0, 1.0, n_subjects)
    subject_slopes = rng.normal(0, 0.3, n_subjects)
    subj_idx = np.repeat(np.arange(n_subjects), n_per)
    y = (
        np.sin(x)
        + subject_intercepts[subj_idx]
        + subject_slopes[subj_idx] * x
        + rng.normal(0, 0.3, n)
    )
    return {"x": x, "y": y, "subject": subject}


@pytest.fixture()
def small_panel():
    rng = np.random.default_rng(23)
    n_groups = 3
    n_per = 20
    n_groups * n_per
    group = np.repeat(["A", "B", "C"], n_per)
    x = np.tile(np.linspace(0, 1, n_per), n_groups)
    effects = {"A": 0.0, "B": 2.0, "C": -1.0}
    y = np.array([effects[g] + xi + rng.normal(0, 0.2) for g, xi in zip(group, x, strict=False)])
    return {"x": x, "y": y, "group": group}


# ---------------------------------------------------------------------------
# FactorSmoothBasis unit tests
# ---------------------------------------------------------------------------


class TestFactorSmoothBasis:
    def test_fit_discovers_levels(self):
        x = np.linspace(0, 1, 30)
        fac = np.repeat(["A", "B", "C"], 10)
        basis = FactorSmoothBasis(k=5).fit(x, fac)
        assert basis.n_levels == 3
        assert set(basis.levels) == {"A", "B", "C"}

    def test_n_basis_is_levels_times_k(self):
        x = np.linspace(0, 1, 30)
        fac = np.repeat(["A", "B", "C"], 10)
        basis = FactorSmoothBasis(k=5).fit(x, fac)
        assert basis.n_basis == 3 * 5

    def test_basis_matrix_shape(self):
        x = np.linspace(0, 1, 30)
        fac = np.repeat(["A", "B", "C"], 10)
        basis = FactorSmoothBasis(k=5).fit(x, fac)
        B = basis.basis_matrix(x, fac)
        assert B.shape == (30, 15)

    def test_basis_matrix_block_diagonal(self):
        x = np.linspace(0, 1, 30)
        fac = np.repeat(["A", "B", "C"], 10)
        basis = FactorSmoothBasis(k=5).fit(x, fac)
        B = basis.basis_matrix(x, fac)
        k = 5
        for i, level in enumerate(["A", "B", "C"]):
            mask = fac == level
            block = B[mask, i * k : (i + 1) * k]
            assert np.all(block != 0) or block.sum() > 0
            for j in range(3):
                if j != i:
                    off_block = B[mask, j * k : (j + 1) * k]
                    np.testing.assert_array_equal(off_block, 0.0)

    def test_penalty_matrices_count(self):
        x = np.linspace(0, 1, 30)
        fac = np.repeat(["A", "B", "C"], 10)
        basis = FactorSmoothBasis(k=5, xt="tp").fit(x, fac)
        pens = basis.penalty_matrices()
        marginal_nsd = basis.marginal_basis.null_space_dimension()
        assert len(pens) == 1 + marginal_nsd

    def test_wiggliness_penalty_is_replicated(self):
        x = np.linspace(0, 1, 30)
        fac = np.repeat(["A", "B", "C"], 10)
        basis = FactorSmoothBasis(k=5, xt="cr").fit(x, fac)
        pens = basis.penalty_matrices()
        S_wiggle = pens[0]
        S_marginal = basis.marginal_basis.penalty_matrix()
        k = 5
        for i in range(3):
            cs = i * k
            ce = cs + k
            np.testing.assert_allclose(S_wiggle[cs:ce, cs:ce], S_marginal, atol=1e-12)
            for j in range(3):
                if j != i:
                    np.testing.assert_array_equal(S_wiggle[cs:ce, j * k : (j + 1) * k], 0.0)

    def test_null_space_dimension_is_zero(self):
        x = np.linspace(0, 1, 30)
        fac = np.repeat(["A", "B", "C"], 10)
        basis = FactorSmoothBasis(k=5).fit(x, fac)
        assert basis.null_space_dimension() == 0

    def test_no_identifiability_constraints(self):
        x = np.linspace(0, 1, 30)
        fac = np.repeat(["A", "B", "C"], 10)
        basis = FactorSmoothBasis(k=5).fit(x, fac)
        assert basis.identifiability_constraints() is None

    def test_penalties_symmetric_psd(self):
        x = np.linspace(0, 1, 30)
        fac = np.repeat(["A", "B", "C"], 10)
        basis = FactorSmoothBasis(k=5).fit(x, fac)
        for S in basis.penalty_matrices():
            np.testing.assert_allclose(S, S.T, atol=1e-12)
            eigvals = np.linalg.eigvalsh(S)
            assert np.all(eigvals >= -1e-10)

    def test_cr_marginal(self):
        x = np.linspace(0, 1, 30)
        fac = np.repeat(["A", "B"], 15)
        basis = FactorSmoothBasis(k=6, xt="cr").fit(x, fac)
        assert basis.n_basis == 2 * 6
        pens = basis.penalty_matrices()
        assert len(pens) == 1 + 2  # cr has nsd=2

    def test_ps_marginal(self):
        x = np.linspace(0, 1, 30)
        fac = np.repeat(["A", "B"], 15)
        basis = FactorSmoothBasis(k=8, xt="ps").fit(x, fac)
        assert basis.n_basis == 2 * 8

    def test_unsupported_xt_raises(self):
        with pytest.raises(ValueError, match="Unknown marginal basis"):
            FactorSmoothBasis(k=5, xt="xx")

    def test_mismatched_lengths_raises(self):
        x = np.linspace(0, 1, 10)
        fac = np.repeat(["A", "B"], 4)  # length 8, mismatched with x (length 10)
        with pytest.raises(ValueError, match="same length"):
            FactorSmoothBasis(k=5).fit(x, fac)

    def test_single_level_raises(self):
        x = np.linspace(0, 1, 10)
        fac = np.array(["A"] * 10)
        with pytest.raises(ValueError, match="at least 2"):
            FactorSmoothBasis(k=5).fit(x, fac)

    def test_not_fitted_raises(self):
        basis = FactorSmoothBasis(k=5)
        with pytest.raises(RuntimeError, match="fitted"):
            basis.basis_matrix(np.array([1.0, 2.0]), np.array(["A", "B"]))

    def test_unseen_level_gets_zero_columns(self):
        x_train = np.linspace(0, 1, 30)
        fac_train = np.repeat(["A", "B", "C"], 10)
        basis = FactorSmoothBasis(k=5).fit(x_train, fac_train)
        x_new = np.array([0.5])
        fac_new = np.array(["D"])
        B = basis.basis_matrix(x_new, fac_new)
        np.testing.assert_array_equal(B, 0.0)

    def test_integer_factor(self):
        x = np.linspace(0, 1, 30)
        fac = np.repeat([1, 2, 3], 10)
        basis = FactorSmoothBasis(k=5).fit(x, fac)
        assert basis.n_levels == 3

    def test_penalty_matrix_returns_wiggliness_penalty(self):
        x = np.linspace(0, 1, 30)
        fac = np.repeat(["A", "B", "C"], 10)
        basis = FactorSmoothBasis(k=5).fit(x, fac)
        np.testing.assert_array_equal(basis.penalty_matrix(), basis.penalty_matrices()[0])

    def test_n_basis_before_fit_raises(self):
        basis = FactorSmoothBasis(k=5)
        with pytest.raises(RuntimeError, match="fit"):
            _ = basis.n_basis

    def test_k_property_before_and_after_fit(self):
        basis = FactorSmoothBasis(k=5)
        assert basis.k == 5  # requested value, before fit()
        x = np.linspace(0, 1, 30)
        fac = np.repeat(["A", "B", "C"], 10)
        basis.fit(x, fac)
        assert basis.k == 5  # fitted marginal dimension, after fit()

    def test_k_property_deferred_default_before_fit(self):
        basis = FactorSmoothBasis(k=-1)
        assert basis.k == -1  # deferred to marginal default, unresolved before fit()

    def test_x_numeric_accepts_column_vector(self):
        x = np.linspace(0, 1, 30).reshape(-1, 1)
        fac = np.repeat(["A", "B", "C"], 10)
        basis = FactorSmoothBasis(k=5).fit(x, fac)
        assert basis.n_basis == 15
        B = basis.basis_matrix(x, fac)
        assert B.shape == (30, 15)

    def test_x_numeric_rejects_multi_column_array(self):
        x = np.column_stack([np.linspace(0, 1, 30), np.linspace(0, 1, 30)])
        fac = np.repeat(["A", "B", "C"], 10)
        with pytest.raises(ValueError, match="Expected 1-D numeric array"):
            FactorSmoothBasis(k=5).fit(x, fac)

    def test_factor_accepts_column_vector(self):
        x = np.linspace(0, 1, 30)
        fac = np.repeat(["A", "B", "C"], 10).reshape(-1, 1)
        basis = FactorSmoothBasis(k=5).fit(x, fac)
        assert basis.n_levels == 3

    def test_factor_rejects_multi_column_array(self):
        x = np.linspace(0, 1, 30)
        fac = np.column_stack([np.repeat(["A", "B", "C"], 10), np.repeat(["X", "Y", "Z"], 10)])
        with pytest.raises(ValueError, match="Expected 1-D factor array"):
            FactorSmoothBasis(k=5).fit(x, fac)


# ---------------------------------------------------------------------------
# Model matrix integration
# ---------------------------------------------------------------------------


class TestFactorSmoothModelMatrix:
    def test_builds_with_fs(self, small_panel):
        formula = parse("y ~ s(x, group, bs='fs')")
        mm = build_model_matrix(formula, small_panel)
        assert mm.X.shape[0] == len(small_panel["y"])
        assert len(mm.smooths) == 1
        assert mm.smooths[0].term.bs == "fs"

    def test_multiple_penalties(self, small_panel):
        formula = parse("y ~ s(x, group, bs='fs')")
        mm = build_model_matrix(formula, small_panel)
        assert len(mm.penalties) >= 2

    def test_no_constraint_applied(self, small_panel):
        mm_c = build_model_matrix(
            parse("y ~ s(x, group, bs='fs')"), small_panel, apply_constraints=True
        )
        mm_nc = build_model_matrix(
            parse("y ~ s(x, group, bs='fs')"), small_panel, apply_constraints=False
        )
        assert mm_c.X.shape[1] == mm_nc.X.shape[1]

    def test_fs_with_other_smooth(self, panel_data):
        formula = parse("y ~ s(x) + s(x, subject, bs='fs', k=5)")
        mm = build_model_matrix(formula, panel_data)
        assert len(mm.smooths) == 2

    def test_fs_single_variable_raises(self):
        with pytest.raises(ValueError, match="at least 2 variables"):
            parse_and_build("y ~ s(x, bs='fs')", {"x": np.ones(10), "y": np.ones(10)})

    def test_penalty_shapes(self, small_panel):
        formula = parse("y ~ s(x, group, bs='fs', k=5)")
        mm = build_model_matrix(formula, small_panel)
        p = mm.X.shape[1]
        for S in mm.penalties:
            assert S.shape == (p, p)


def parse_and_build(formula_str, data):
    return build_model_matrix(parse(formula_str), data)


# ---------------------------------------------------------------------------
# GAM fitting tests
# ---------------------------------------------------------------------------


class TestFactorSmoothGAM:
    def test_fit_gaussian(self, panel_data):
        gam = GAM("y ~ s(x, subject, bs='fs', k=6)", family=Gaussian())
        gam.fit(panel_data)
        assert gam.is_fitted
        assert gam._fit_result.deviance > 0

    def test_deviance_explained(self, panel_data):
        gam = GAM("y ~ s(x, subject, bs='fs', k=6)", family=Gaussian())
        gam.fit(panel_data)
        residuals = gam.get_residuals("response")
        ss_res = np.sum(residuals**2)
        ss_total = np.sum((panel_data["y"] - panel_data["y"].mean()) ** 2)
        deviance_explained = 1 - ss_res / ss_total
        assert deviance_explained > 0.5

    def test_predict(self, panel_data):
        gam = GAM("y ~ s(x, subject, bs='fs', k=6)", family=Gaussian())
        gam.fit(panel_data)
        pred = gam.predict(panel_data)
        assert pred.values.shape == (len(panel_data["y"]),)
        assert np.isfinite(pred.values).all()

    def test_predict_with_se(self, panel_data):
        gam = GAM("y ~ s(x, subject, bs='fs', k=6)", family=Gaussian())
        gam.fit(panel_data)
        pred = gam.predict(panel_data, se=True)
        assert pred.se is not None
        assert np.all(pred.se >= 0)

    def test_summary(self, panel_data):
        gam = GAM("y ~ s(x, subject, bs='fs', k=6)", family=Gaussian())
        gam.fit(panel_data)
        s = gam.summary()
        assert "s(x, subject" in s

    def test_smooth_tests(self, panel_data):
        gam = GAM("y ~ s(x, subject, bs='fs', k=6)", family=Gaussian())
        gam.fit(panel_data)
        tests = gam.smooth_tests()
        assert len(tests) == 1
        assert 0.0 <= tests[0].p_value <= 1.0

    def test_edf(self, panel_data):
        gam = GAM("y ~ s(x, subject, bs='fs', k=6)", family=Gaussian())
        gam.fit(panel_data)
        edfs = gam.edf
        assert len(edfs) == 1
        assert edfs[0] > 1

    def test_reml(self, panel_data):
        gam = GAM("y ~ s(x, subject, bs='fs', k=6)", family=Gaussian())
        gam.fit(panel_data, method="REML")
        assert gam.is_fitted

    def test_residuals(self, panel_data):
        gam = GAM("y ~ s(x, subject, bs='fs', k=6)", family=Gaussian())
        gam.fit(panel_data)
        for rtype in ("response", "pearson", "deviance", "working"):
            r = gam.get_residuals(rtype)
            assert r.shape == (len(panel_data["y"]),)
            assert np.isfinite(r).all()

    def test_anova_fs_vs_plain(self, panel_data):
        g1 = GAM("y ~ s(x)", family=Gaussian()).fit(panel_data)
        g2 = GAM("y ~ s(x, subject, bs='fs', k=6)", family=Gaussian()).fit(panel_data)
        result = g1.anova(g2)
        assert len(result.rows) == 2
        assert result.rows[1].deviance > 0

    def test_fs_with_re(self, panel_data):
        gam = GAM(
            "y ~ s(x, subject, bs='fs', k=6) + s(subject, bs='re')",
            family=Gaussian(),
        )
        gam.fit(panel_data)
        assert gam.is_fitted
        assert len(gam.edf) == 2

    def test_poisson_fs(self):
        rng = np.random.default_rng(23)
        n_groups = 3
        n_per = 40
        n_groups * n_per
        group = np.repeat(np.arange(n_groups).astype(str), n_per)
        x = np.tile(np.linspace(0, 2 * np.pi, n_per), n_groups)
        group_effects = rng.normal(0, 0.2, n_groups)
        subj_idx = np.repeat(np.arange(n_groups), n_per)
        mu = np.exp(0.5 * np.sin(x) + group_effects[subj_idx])
        y = rng.poisson(mu)
        data = {"x": x, "y": y, "group": group}
        gam = GAM("y ~ s(x, group, bs='fs', k=5)", family=Poisson())
        gam.fit(data)
        assert gam.is_fitted

    def test_concurvity_with_fs(self, panel_data):
        gam = GAM("y ~ s(x) + s(x, subject, bs='fs', k=5)", family=Gaussian())
        gam.fit(panel_data)
        c = gam.concurvity()
        assert len(c.labels) == 2

    def test_aic_bic_finite(self, panel_data):
        gam = GAM("y ~ s(x, subject, bs='fs', k=6)", family=Gaussian())
        gam.fit(panel_data)
        assert np.isfinite(gam._fit_result.aic)
        assert np.isfinite(gam._fit_result.bic)

    def test_fs_with_xt_cr(self, panel_data):
        gam = GAM("y ~ s(x, subject, bs='fs', k=6, xt='cr')", family=Gaussian())
        gam.fit(panel_data)
        assert gam.is_fitted
        pred = gam.predict(panel_data)
        assert np.isfinite(pred.values).all()

    def test_per_level_curves_differ(self, panel_data):
        gam = GAM("y ~ s(x, subject, bs='fs', k=6)", family=Gaussian())
        gam.fit(panel_data)
        subjects = np.unique(panel_data["subject"])
        x_grid = np.linspace(0, 2 * np.pi, 50)
        preds_by_subject = []
        for subj in subjects[:3]:
            pred = gam.predict(
                {
                    "x": x_grid,
                    "subject": np.array([subj] * len(x_grid)),
                }
            )
            preds_by_subject.append(pred.values)
        assert not np.allclose(preds_by_subject[0], preds_by_subject[1], atol=0.01)

    def test_fs_better_than_global_smooth(self, panel_data):
        g_global = GAM("y ~ s(x)", family=Gaussian()).fit(panel_data)
        g_fs = GAM("y ~ s(x, subject, bs='fs', k=6)", family=Gaussian()).fit(panel_data)
        assert g_fs._fit_result.deviance < g_global._fit_result.deviance
