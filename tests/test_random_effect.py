"""Tests for random effect smooth basis (bs="re")."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.families import Gaussian, Poisson
from whittaker.formula.parser import parse
from whittaker.gam import GAM
from whittaker.model_matrix import build_model_matrix
from whittaker.smooths.random import RandomEffectBasis

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def group_data():
    rng = np.random.default_rng(23)
    n_groups = 10
    n_per = 30
    n = n_groups * n_per
    group = np.repeat(np.arange(n_groups).astype(str), n_per)
    group_effects = rng.normal(0, 1.5, n_groups)
    y = group_effects[np.repeat(np.arange(n_groups), n_per)] + rng.normal(0, 0.5, n)
    x = rng.uniform(0, 1, n)
    return {"x": x, "y": y, "group": group}


@pytest.fixture()
def simple_groups():
    rng = np.random.default_rng(23)
    group = np.array(["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"])
    y = np.array([1.0, 5.0, 3.0, 1.5, 4.5, 3.5, 1.2, 5.2, 2.8, 0.8])
    x = rng.uniform(0, 1, len(y))
    return {"x": x, "y": y, "group": group}


# ---------------------------------------------------------------------------
# RandomEffectBasis unit tests
# ---------------------------------------------------------------------------


class TestRandomEffectBasis:
    def test_fit_discovers_levels(self):
        x = np.array(["A", "B", "C", "A", "B"])
        basis = RandomEffectBasis().fit(x)
        assert basis.n_basis == 3
        assert set(basis.levels) == {"A", "B", "C"}

    def test_basis_matrix_is_one_hot(self):
        x = np.array(["cat", "dog", "cat", "bird", "dog"])
        basis = RandomEffectBasis().fit(x)
        B = basis.basis_matrix(x)
        assert B.shape == (5, 3)
        assert np.all(B.sum(axis=1) == 1.0)
        for row in B:
            assert np.sum(row == 1.0) == 1

    def test_penalty_is_identity(self):
        x = np.array(["A", "B", "C", "D"] * 5)
        basis = RandomEffectBasis().fit(x)
        S = basis.penalty_matrix()
        np.testing.assert_array_equal(S, np.eye(4))

    def test_null_space_dimension_is_zero(self):
        x = np.array(["A", "B", "C"] * 5)
        basis = RandomEffectBasis().fit(x)
        assert basis.null_space_dimension() == 0

    def test_identifiability_constraint(self):
        x = np.array(["A", "B", "C"] * 5)
        basis = RandomEffectBasis().fit(x)
        C = basis.identifiability_constraints()
        assert C is not None
        assert C.shape == (1, 3)
        np.testing.assert_allclose(C, np.ones((1, 3)) / 3)

    def test_integer_levels(self):
        x = np.array([1, 2, 3, 1, 2, 3])
        basis = RandomEffectBasis().fit(x)
        assert basis.n_basis == 3
        B = basis.basis_matrix(x)
        assert B.shape == (6, 3)
        assert np.all(B.sum(axis=1) == 1.0)

    def test_unseen_level_gets_zero_row(self):
        x_train = np.array(["A", "B", "C"] * 5)
        x_new = np.array(["A", "D", "B"])
        basis = RandomEffectBasis().fit(x_train)
        B = basis.basis_matrix(x_new)
        assert B.shape == (3, 3)
        assert B[0].sum() == 1.0
        assert B[1].sum() == 0.0
        assert B[2].sum() == 1.0

    def test_k_limits_levels(self):
        x = np.array(["A", "B", "C", "D", "E"] * 3)
        basis = RandomEffectBasis(k=3).fit(x)
        assert basis.n_basis == 3

    def test_k_minus_one_uses_all(self):
        x = np.array(["A", "B", "C", "D", "E"] * 3)
        basis = RandomEffectBasis(k=-1).fit(x)
        assert basis.n_basis == 5

    def test_too_few_levels_raises(self):
        x = np.array(["A", "A", "A"])
        with pytest.raises(ValueError, match="at least 2"):
            RandomEffectBasis().fit(x)

    def test_k_too_small_raises(self):
        x = np.array(["A", "B", "C"] * 3)
        with pytest.raises(ValueError, match="at least 2"):
            RandomEffectBasis(k=1).fit(x)

    def test_not_fitted_raises(self):
        basis = RandomEffectBasis()
        with pytest.raises(RuntimeError, match="fitted"):
            basis.basis_matrix(np.array(["A", "B"]))

    def test_n_basis_before_fit_raises(self):
        basis = RandomEffectBasis()
        with pytest.raises(RuntimeError, match="not available until fit"):
            basis.n_basis

    def test_k_property_before_and_after_fit(self):
        basis = RandomEffectBasis(k=2)
        assert basis.k == 2
        x = np.array(["A", "B", "C"] * 3)
        basis.fit(x)
        assert basis.k == basis.n_basis == 2

    def test_2d_column_vector(self):
        x = np.array(["A", "B", "C", "A"]).reshape(-1, 1)
        basis = RandomEffectBasis().fit(x)
        assert basis.n_basis == 3

    def test_multidim_raises(self):
        x = np.array([["A", "B"], ["C", "D"]])
        with pytest.raises(ValueError, match="1-D"):
            RandomEffectBasis().fit(x)

    def test_penalty_symmetric_psd(self):
        x = np.array(["A", "B", "C", "D"] * 5)
        basis = RandomEffectBasis().fit(x)
        S = basis.penalty_matrix()
        np.testing.assert_array_equal(S, S.T)
        eigvals = np.linalg.eigvalsh(S)
        assert np.all(eigvals >= -1e-12)

    def test_sorted_levels(self):
        x = np.array(["C", "A", "B", "C", "A", "B"])
        basis = RandomEffectBasis().fit(x)
        np.testing.assert_array_equal(basis.levels, ["A", "B", "C"])


# ---------------------------------------------------------------------------
# Model matrix integration tests
# ---------------------------------------------------------------------------


class TestRandomEffectModelMatrix:
    def test_builds_with_re(self, simple_groups):
        formula = parse("y ~ s(group, bs='re')")
        mm = build_model_matrix(formula, simple_groups)
        assert mm.X.shape[0] == len(simple_groups["y"])
        assert mm.X.shape[1] >= 3
        assert len(mm.penalties) == 1

    def test_re_with_smooth(self, group_data):
        formula = parse("y ~ s(x) + s(group, bs='re')")
        mm = build_model_matrix(formula, group_data)
        assert len(mm.smooths) == 2
        assert mm.smooths[1].term.bs == "re"
        assert len(mm.penalties) >= 2

    def test_penalty_is_identity_block(self, simple_groups):
        formula = parse("y ~ s(group, bs='re')")
        mm = build_model_matrix(formula, simple_groups, apply_constraints=False)
        re_info = mm.smooths[0]
        cs, ce = re_info.col_start, re_info.col_end
        S = mm.penalties[0]
        block = S[cs:ce, cs:ce]
        np.testing.assert_array_equal(block, np.eye(ce - cs))

    def test_constraint_reduces_columns(self, simple_groups):
        mm_nc = build_model_matrix(
            parse("y ~ s(group, bs='re')"), simple_groups, apply_constraints=False
        )
        mm_c = build_model_matrix(
            parse("y ~ s(group, bs='re')"), simple_groups, apply_constraints=True
        )
        assert mm_c.X.shape[1] == mm_nc.X.shape[1] - 1


# ---------------------------------------------------------------------------
# GAM fitting tests
# ---------------------------------------------------------------------------


class TestRandomEffectGAM:
    def test_fit_gaussian(self, group_data):
        gam = GAM("y ~ s(x) + s(group, bs='re')", family=Gaussian())
        gam.fit(group_data)
        assert gam.is_fitted
        assert gam._fit_result.deviance > 0

    def test_shrinks_group_effects(self, group_data):
        gam = GAM("y ~ s(x) + s(group, bs='re')", family=Gaussian())
        gam.fit(group_data)
        residuals = gam.get_residuals("response")
        ss_res = np.sum(residuals**2)
        ss_total = np.sum((group_data["y"] - group_data["y"].mean()) ** 2)
        deviance_explained = 1 - ss_res / ss_total
        assert deviance_explained > 0.5

    def test_predict(self, group_data):
        gam = GAM("y ~ s(x) + s(group, bs='re')", family=Gaussian())
        gam.fit(group_data)
        pred = gam.predict(group_data)
        assert pred.values.shape == (len(group_data["y"]),)
        assert np.isfinite(pred.values).all()

    def test_predict_with_se(self, group_data):
        gam = GAM("y ~ s(x) + s(group, bs='re')", family=Gaussian())
        gam.fit(group_data)
        pred = gam.predict(group_data, se=True)
        assert pred.se is not None
        assert pred.se.shape == pred.values.shape
        assert np.all(pred.se >= 0)

    def test_summary(self, group_data):
        gam = GAM("y ~ s(x) + s(group, bs='re')", family=Gaussian())
        gam.fit(group_data)
        s = gam.summary()
        assert "s(group" in s
        assert "s(x" in s

    def test_smooth_tests(self, group_data):
        gam = GAM("y ~ s(x) + s(group, bs='re')", family=Gaussian())
        gam.fit(group_data)
        tests = gam.smooth_tests()
        assert len(tests) == 2
        for t in tests:
            assert 0.0 <= t.p_value <= 1.0

    def test_edf(self, group_data):
        gam = GAM("y ~ s(x) + s(group, bs='re')", family=Gaussian())
        gam.fit(group_data)
        edfs = gam.edf
        assert len(edfs) == 2
        re_edf = edfs[1]
        assert re_edf > 0

    def test_re_only_model(self, group_data):
        gam = GAM("y ~ s(group, bs='re')", family=Gaussian())
        gam.fit(group_data)
        assert gam.is_fitted
        pred = gam.predict(group_data)
        assert np.isfinite(pred.values).all()

    def test_re_with_reml(self, group_data):
        gam = GAM("y ~ s(x) + s(group, bs='re')", family=Gaussian())
        gam.fit(group_data, method="REML")
        assert gam.is_fitted

    def test_residuals(self, group_data):
        gam = GAM("y ~ s(x) + s(group, bs='re')", family=Gaussian())
        gam.fit(group_data)
        for rtype in ("response", "pearson", "deviance", "working"):
            r = gam.get_residuals(rtype)
            assert r.shape == (len(group_data["y"]),)
            assert np.isfinite(r).all()

    def test_anova_with_re(self, group_data):
        g1 = GAM("y ~ s(x)", family=Gaussian()).fit(group_data)
        g2 = GAM("y ~ s(x) + s(group, bs='re')", family=Gaussian()).fit(group_data)
        result = g1.anova(g2)
        assert len(result.rows) == 2
        assert result.rows[1].p_value is not None

    def test_poisson_with_re(self):
        rng = np.random.default_rng(23)
        n_groups = 8
        n_per = 40
        n = n_groups * n_per
        group = np.repeat(np.arange(n_groups).astype(str), n_per)
        x = rng.uniform(0, 2 * np.pi, n)
        group_effects = rng.normal(0, 0.3, n_groups)
        mu = np.exp(0.5 * np.sin(x) + group_effects[np.repeat(np.arange(n_groups), n_per)])
        y = rng.poisson(mu)
        data = {"x": x, "y": y, "group": group}
        gam = GAM("y ~ s(x) + s(group, bs='re')", family=Poisson())
        gam.fit(data)
        assert gam.is_fitted
        pred = gam.predict(data)
        assert np.isfinite(pred.values).all()

    def test_concurvity(self, group_data):
        gam = GAM("y ~ s(x) + s(group, bs='re')", family=Gaussian())
        gam.fit(group_data)
        c = gam.concurvity()
        assert len(c.labels) == 2

    def test_integer_groups(self):
        rng = np.random.default_rng(23)
        n = 100
        group = rng.choice(5, n)
        y = group.astype(float) + rng.normal(0, 0.5, n)
        data = {"y": y, "group": group}
        gam = GAM("y ~ s(group, bs='re')", family=Gaussian())
        gam.fit(data)
        assert gam.is_fitted

    def test_aic_bic_finite(self, group_data):
        gam = GAM("y ~ s(x) + s(group, bs='re')", family=Gaussian())
        gam.fit(group_data)
        assert np.isfinite(gam._fit_result.aic)
        assert np.isfinite(gam._fit_result.bic)

    def test_plot(self, group_data):
        gam = GAM("y ~ s(x) + s(group, bs='re')", family=Gaussian())
        gam.fit(group_data)
        chart = gam.plot()
        assert chart is not None
