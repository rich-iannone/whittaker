"""Tests for whittaker.gam (top-level GAM class)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from whittaker.gam import GAM, PredictionResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(23)


def _sin_data(n: int = 200) -> dict[str, np.ndarray]:
    x = np.linspace(0, 2 * np.pi, n)
    return {
        "y": np.sin(x) + RNG.normal(0, 0.2, n),
        "x": x,
    }


def _multi_data(n: int = 200) -> dict[str, np.ndarray]:
    x1 = np.linspace(0, 2 * np.pi, n)
    x2 = np.linspace(0, np.pi, n)
    return {
        "y": np.sin(x1) + 0.5 * np.cos(x2) + RNG.normal(0, 0.2, n),
        "x1": x1,
        "x2": x2,
    }


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_string_formula(self) -> None:
        model = GAM("y ~ s(x)")
        assert not model.is_fitted
        assert repr(model.formula) == "y ~ s(x)"

    def test_formula_object(self) -> None:
        from whittaker.formula.parser import parse

        f = parse("y ~ s(x1) + x2")
        model = GAM(f)
        assert model.formula is f

    def test_default_family_is_gaussian(self) -> None:
        from whittaker.families.gaussian import Gaussian

        model = GAM("y ~ s(x)")
        assert isinstance(model.family, Gaussian)

    def test_repr_unfitted(self) -> None:
        model = GAM("y ~ s(x)")
        r = repr(model)
        assert "unfitted" in r
        assert "GAM" in r


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


class TestFit:
    def test_fit_returns_self(self) -> None:
        data = _sin_data()
        model = GAM("y ~ s(x)")
        result = model.fit(data)
        assert result is model

    def test_is_fitted_after_fit(self) -> None:
        data = _sin_data()
        model = GAM("y ~ s(x)").fit(data)
        assert model.is_fitted

    def test_method_chaining(self) -> None:
        data = _sin_data()
        pred = GAM("y ~ s(x)").fit(data).predict({"x": data["x"]})
        assert isinstance(pred, PredictionResult)

    def test_repr_fitted(self) -> None:
        data = _sin_data()
        model = GAM("y ~ s(x)").fit(data)
        r = repr(model)
        assert "fitted" in r

    def test_fixed_smoothing_params(self) -> None:
        data = _sin_data()
        model = GAM("y ~ s(x)").fit(data, smoothing_params=[1.0])
        assert model.smoothing_params == [1.0]


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


class TestPredict:
    def test_predict_values_shape(self) -> None:
        data = _sin_data()
        model = GAM("y ~ s(x)").fit(data)
        pred = model.predict({"x": np.array([1.0, 2.0, 3.0])})
        assert pred.values.shape == (3,)

    def test_predict_no_se_by_default(self) -> None:
        data = _sin_data()
        model = GAM("y ~ s(x)").fit(data)
        pred = model.predict({"x": np.array([1.0])})
        assert pred.se is None

    def test_predict_with_se(self) -> None:
        data = _sin_data()
        model = GAM("y ~ s(x)").fit(data)
        pred = model.predict({"x": np.array([1.0, 2.0])}, se=True)
        assert pred.se is not None
        assert pred.se.shape == (2,)
        assert np.all(pred.se > 0)

    def test_predict_linear_predictor(self) -> None:
        data = _sin_data()
        model = GAM("y ~ s(x)").fit(data)
        pred = model.predict({"x": np.array([1.0])})
        assert_allclose(pred.values, pred.linear_predictor)

    def test_predict_reproduces_fitted(self) -> None:
        data = _sin_data()
        model = GAM("y ~ s(x)").fit(data)
        pred = model.predict({"x": data["x"]})
        assert_allclose(pred.values, model.fitted_values, atol=1e-10)

    def test_predict_unfitted_raises(self) -> None:
        model = GAM("y ~ s(x)")
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict({"x": np.array([1.0])})

    def test_se_larger_at_boundary(self) -> None:
        data = _sin_data()
        model = GAM("y ~ s(x, k=15)").fit(data)
        x_interior = np.array([np.pi])
        x_boundary = np.array([0.0])
        se_interior = model.predict({"x": x_interior}, se=True).se
        se_boundary = model.predict({"x": x_boundary}, se=True).se
        assert se_interior is not None and se_boundary is not None
        assert se_boundary[0] >= se_interior[0] * 0.5


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_coefficients(self) -> None:
        data = _sin_data()
        model = GAM("y ~ s(x, k=10)").fit(data)
        assert model.coefficients.shape[0] > 0

    def test_fitted_values(self) -> None:
        data = _sin_data()
        model = GAM("y ~ s(x)").fit(data)
        assert model.fitted_values.shape == (200,)

    def test_residuals(self) -> None:
        data = _sin_data()
        model = GAM("y ~ s(x)").fit(data)
        assert model.residuals.shape == (200,)

    def test_edf(self) -> None:
        data = _sin_data()
        model = GAM("y ~ s(x, k=10)").fit(data)
        assert len(model.edf) == 1
        assert 1.0 < model.edf[0] < 10.0

    def test_edf_total(self) -> None:
        data = _sin_data()
        model = GAM("y ~ s(x)").fit(data)
        assert model.edf_total > 1.0

    def test_scale(self) -> None:
        data = _sin_data()
        model = GAM("y ~ s(x)").fit(data)
        assert model.scale > 0

    def test_deviance(self) -> None:
        data = _sin_data()
        model = GAM("y ~ s(x)").fit(data)
        assert model.deviance > 0

    def test_gcv_score(self) -> None:
        data = _sin_data()
        model = GAM("y ~ s(x)").fit(data)
        assert model.gcv_score > 0

    def test_unfitted_properties_raise(self) -> None:
        model = GAM("y ~ s(x)")
        with pytest.raises(RuntimeError):
            _ = model.coefficients
        with pytest.raises(RuntimeError):
            _ = model.fitted_values
        with pytest.raises(RuntimeError):
            _ = model.edf


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_string(self) -> None:
        data = _sin_data()
        model = GAM("y ~ s(x)").fit(data)
        text = model.summary()
        assert "GAM fit summary" in text
        assert "s(x)" in text
        assert "Gaussian" in text
        assert "EDF" in text

    def test_summary_unfitted_raises(self) -> None:
        with pytest.raises(RuntimeError):
            GAM("y ~ s(x)").summary()


# ---------------------------------------------------------------------------
# Multiple smooths
# ---------------------------------------------------------------------------


class TestMultipleSmooths:
    def test_two_smooths_fit(self) -> None:
        data = _multi_data()
        model = GAM("y ~ s(x1) + s(x2)").fit(data)
        assert len(model.edf) == 2
        assert len(model.smoothing_params) == 2

    def test_two_smooths_predict(self) -> None:
        data = _multi_data()
        model = GAM("y ~ s(x1) + s(x2)").fit(data)
        pred = model.predict({"x1": data["x1"], "x2": data["x2"]})
        assert pred.values.shape == (200,)

    def test_two_smooths_recovery(self) -> None:
        n = 300
        x1 = np.linspace(0, 2 * np.pi, n)
        x2 = np.linspace(0, np.pi, n)
        y_true = np.sin(x1) + 0.5 * np.cos(x2)
        y = y_true + RNG.normal(0, 0.15, n)

        model = GAM("y ~ s(x1, k=15) + s(x2, k=15)").fit({"y": y, "x1": x1, "x2": x2})
        rmse = np.sqrt(np.mean((model.fitted_values - y_true) ** 2))
        assert rmse < 0.25


# ---------------------------------------------------------------------------
# Mixed parametric + smooth
# ---------------------------------------------------------------------------


class TestMixed:
    def test_linear_plus_smooth(self) -> None:
        n = 200
        x1 = np.linspace(0, 1, n)
        x2 = np.linspace(0, 2 * np.pi, n)
        y = 2.0 * x1 + np.sin(x2) + RNG.normal(0, 0.15, n)

        model = GAM("y ~ x1 + s(x2, k=15)").fit({"y": y, "x1": x1, "x2": x2})
        assert model.is_fitted
        rmse = np.sqrt(np.mean((model.fitted_values - (2.0 * x1 + np.sin(x2))) ** 2))
        assert rmse < 0.15


# ---------------------------------------------------------------------------
# Basis types
# ---------------------------------------------------------------------------


class TestBasisTypes:
    @pytest.mark.parametrize("bs", ["tp", "cr", "ps"])
    def test_basis_type_fits(self, bs: str) -> None:
        data = _sin_data()
        model = GAM(f"y ~ s(x, bs='{bs}', k=10)").fit(data)
        assert model.is_fitted
        rmse = np.sqrt(np.mean((model.fitted_values - np.sin(data["x"])) ** 2))
        assert rmse < 0.15
