"""Tests for by= variable support in smooth terms."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from whittaker.families.binomial import Binomial
from whittaker.families.poisson import Poisson
from whittaker.formula.parser import parse
from whittaker.gam import GAM
from whittaker.model_matrix import build_model_matrix


# ---------------------------------------------------------------------------
# Factor-by smooths
# ---------------------------------------------------------------------------


class TestFactorBy:
    def test_creates_one_smooth_per_level(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x = np.linspace(0, 1, n)
        group = np.array(["A"] * 100 + ["B"] * 100)
        y = x + rng.normal(0, 0.1, n)

        formula = parse("y ~ s(x, k=6, by='group')")
        mm = build_model_matrix(formula, {"y": y, "x": x, "group": group})
        assert len(mm.smooths) == 2
        assert mm.smooths[0].by_level == "A"
        assert mm.smooths[1].by_level == "B"

    def test_by_levels_sorted(self) -> None:
        rng = np.random.default_rng(42)
        n = 150
        x = np.linspace(0, 1, n)
        group = np.array(["C"] * 50 + ["A"] * 50 + ["B"] * 50)
        y = x + rng.normal(0, 0.1, n)

        formula = parse("y ~ s(x, k=6, by='group')")
        mm = build_model_matrix(formula, {"y": y, "x": x, "group": group})
        levels = [s.by_level for s in mm.smooths]
        assert levels == ["A", "B", "C"]

    def test_factor_by_skips_constraint(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x = np.linspace(0, 1, n)
        group = np.array(["A"] * 100 + ["B"] * 100)
        y = x + rng.normal(0, 0.1, n)

        formula = parse("y ~ s(x, k=6, by='group')")
        mm = build_model_matrix(formula, {"y": y, "x": x, "group": group})
        # Without constraint: each smooth keeps all 6 basis functions
        # With constraint: each would have 5
        for info in mm.smooths:
            assert info.col_end - info.col_start == 6

    def test_factor_by_separate_penalties(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x = np.linspace(0, 1, n)
        group = np.array(["A"] * 100 + ["B"] * 100)
        y = x + rng.normal(0, 0.1, n)

        formula = parse("y ~ s(x, k=6, by='group')")
        mm = build_model_matrix(formula, {"y": y, "x": x, "group": group})
        assert len(mm.penalties) == 2

    def test_factor_by_gam_fit(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x = np.linspace(0, 2 * np.pi, n)
        group = np.array(["A"] * 150 + ["B"] * 150)
        y = np.where(group == "A", np.sin(x), np.cos(x)) + rng.normal(0, 0.2, n)

        model = GAM("y ~ s(x, k=8, by='group')").fit({"y": y, "x": x, "group": group})
        assert model.is_fitted
        assert len(model.edf) == 2
        assert len(model.smoothing_params) == 2

    def test_factor_by_recovery(self) -> None:
        rng = np.random.default_rng(42)
        n = 400
        x = np.linspace(0, 2 * np.pi, n)
        group = np.array(["A"] * 200 + ["B"] * 200)
        f_true = np.where(group == "A", np.sin(x), 0.5 * np.cos(x))
        y = f_true + rng.normal(0, 0.2, n)

        model = GAM("y ~ s(x, k=10, by='group')").fit({"y": y, "x": x, "group": group})
        rmse = np.sqrt(np.mean((model.fitted_values - f_true) ** 2))
        assert rmse < 0.2

    def test_factor_by_predict(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x = np.linspace(0, 2 * np.pi, n)
        group = np.array(["A"] * 150 + ["B"] * 150)
        y = np.where(group == "A", np.sin(x), -np.sin(x)) + rng.normal(0, 0.2, n)

        model = GAM("y ~ s(x, k=8, by='group')").fit({"y": y, "x": x, "group": group})

        pred_A = model.predict({"x": np.array([np.pi / 2]), "group": np.array(["A"])}, se=True)
        pred_B = model.predict({"x": np.array([np.pi / 2]), "group": np.array(["B"])}, se=True)
        # Group A should be positive, Group B negative
        assert pred_A.values[0] > 0.5
        assert pred_B.values[0] < -0.5
        assert pred_A.se is not None
        assert pred_B.se is not None

    def test_factor_by_three_levels(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x = np.linspace(0, 1, n)
        group = np.array(["X"] * 100 + ["Y"] * 100 + ["Z"] * 100)
        y = x + rng.normal(0, 0.1, n)

        model = GAM("y ~ s(x, k=6, by='group')").fit({"y": y, "x": x, "group": group})
        assert len(model.edf) == 3
        assert len(model.smoothing_params) == 3

    def test_factor_by_summary(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x = np.linspace(0, 1, n)
        group = np.array(["A"] * 100 + ["B"] * 100)
        y = x + rng.normal(0, 0.1, n)

        model = GAM("y ~ s(x, k=6, by='group')").fit({"y": y, "x": x, "group": group})
        text = model.summary()
        assert ":A" in text
        assert ":B" in text

    def test_factor_by_with_reml(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x = np.linspace(0, 2 * np.pi, n)
        group = np.array(["A"] * 150 + ["B"] * 150)
        y = np.where(group == "A", np.sin(x), np.cos(x)) + rng.normal(0, 0.3, n)

        model = GAM("y ~ s(x, k=8, by='group')").fit(
            {"y": y, "x": x, "group": group}, method="REML"
        )
        assert model.is_fitted
        assert len(model.edf) == 2


# ---------------------------------------------------------------------------
# Continuous-by smooths
# ---------------------------------------------------------------------------


class TestContinuousBy:
    def test_creates_one_smooth(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x = np.linspace(0, 1, n)
        z = rng.uniform(0, 2, n)
        y = x * z + rng.normal(0, 0.1, n)

        formula = parse("y ~ s(x, k=6, by='z')")
        mm = build_model_matrix(formula, {"y": y, "x": x, "z": z})
        assert len(mm.smooths) == 1
        assert mm.smooths[0].by_var == "z"
        assert mm.smooths[0].by_level is None

    def test_continuous_by_skips_constraint(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x = np.linspace(0, 1, n)
        z = rng.uniform(0, 2, n)
        y = x * z + rng.normal(0, 0.1, n)

        formula = parse("y ~ s(x, k=6, by='z')")
        mm = build_model_matrix(formula, {"y": y, "x": x, "z": z})
        assert mm.smooths[0].col_end - mm.smooths[0].col_start == 6

    def test_continuous_by_gam_fit(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x = np.linspace(0, 2 * np.pi, n)
        z = rng.uniform(0, 2, n)
        y = np.sin(x) * z + rng.normal(0, 0.3, n)

        model = GAM("y ~ s(x, k=10, by='z')").fit({"y": y, "x": x, "z": z})
        assert model.is_fitted
        assert len(model.edf) == 1

    def test_continuous_by_recovery(self) -> None:
        rng = np.random.default_rng(42)
        n = 500
        x = np.linspace(0, 2 * np.pi, n)
        z = rng.uniform(0.5, 1.5, n)
        y = np.sin(x) * z + rng.normal(0, 0.2, n)

        model = GAM("y ~ s(x, k=10, by='z')").fit({"y": y, "x": x, "z": z})

        # Predict at z=1 should recover sin(x)
        x_test = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        pred = model.predict({"x": x_test, "z": np.ones_like(x_test)})
        expected = np.sin(x_test)
        assert_allclose(pred.values, expected, atol=0.15)

    def test_continuous_by_predict(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x = np.linspace(0, 1, n)
        z = rng.uniform(0, 2, n)
        y = x * z + rng.normal(0, 0.1, n)

        model = GAM("y ~ s(x, k=8, by='z')").fit({"y": y, "x": x, "z": z})

        pred = model.predict({"x": np.array([0.5]), "z": np.array([2.0])}, se=True)
        assert pred.se is not None
        assert pred.se[0] > 0
        assert abs(pred.values[0] - 1.0) < 0.3

    def test_continuous_by_with_reml(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x = np.linspace(0, 2 * np.pi, n)
        z = rng.uniform(0, 2, n)
        y = np.sin(x) * z + rng.normal(0, 0.3, n)

        model = GAM("y ~ s(x, k=10, by='z')").fit({"y": y, "x": x, "z": z}, method="REML")
        assert model.is_fitted


# ---------------------------------------------------------------------------
# Mixed by + non-by smooths
# ---------------------------------------------------------------------------


class TestMixedBy:
    def test_by_with_regular_smooth(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x1 = np.linspace(0, 2 * np.pi, n)
        x2 = np.linspace(0, 1, n)
        group = np.array(["A"] * 150 + ["B"] * 150)
        y = np.where(group == "A", np.sin(x1), np.cos(x1)) + 0.5 * x2 + rng.normal(0, 0.2, n)

        model = GAM("y ~ s(x1, k=8, by='group') + s(x2, k=8)").fit(
            {"y": y, "x1": x1, "x2": x2, "group": group}
        )
        # 2 factor-by smooths + 1 regular smooth = 3 smooth infos
        assert len(model.edf) == 3
        # 2 factor-by penalties + 1 regular penalty = 3
        assert len(model.smoothing_params) == 3

    def test_continuous_by_with_regular_smooth(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x1 = np.linspace(0, 2 * np.pi, n)
        x2 = np.linspace(0, 1, n)
        z = rng.uniform(0, 2, n)
        y = np.sin(x1) * z + x2 + rng.normal(0, 0.2, n)

        model = GAM("y ~ s(x1, k=8, by='z') + s(x2, k=8)").fit({"y": y, "x1": x1, "x2": x2, "z": z})
        assert len(model.edf) == 2
        assert len(model.smoothing_params) == 2

    def test_factor_by_binomial(self) -> None:
        rng = np.random.default_rng(42)
        n = 400
        x = np.linspace(-2, 2, n)
        group = np.array(["A"] * 200 + ["B"] * 200)
        eta = np.where(group == "A", x, -x)
        p_true = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, p_true, n).astype(float)

        model = GAM("y ~ s(x, k=6, by='group')", family=Binomial()).fit(
            {"y": y, "x": x, "group": group}
        )
        assert model.is_fitted
        assert len(model.edf) == 2

    def test_factor_by_poisson(self) -> None:
        rng = np.random.default_rng(42)
        n = 400
        x = np.linspace(0, 1, n)
        group = np.array(["A"] * 200 + ["B"] * 200)
        mu = np.where(group == "A", np.exp(x), np.exp(0.5 * x))
        y = rng.poisson(mu).astype(float)

        model = GAM("y ~ s(x, k=6, by='group')", family=Poisson()).fit(
            {"y": y, "x": x, "group": group}
        )
        assert model.is_fitted
        assert len(model.edf) == 2

    def test_smooth_tests_with_by(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x = np.linspace(0, 2 * np.pi, n)
        group = np.array(["A"] * 150 + ["B"] * 150)
        y = np.where(group == "A", np.sin(x), 0) + rng.normal(0, 0.3, n)

        model = GAM("y ~ s(x, k=8, by='group')").fit({"y": y, "x": x, "group": group})
        tests = model.smooth_tests()
        assert len(tests) == 2
        # Group A has real signal
        assert tests[0].p_value < 0.01
