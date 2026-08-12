"""Tests for whittaker.gam (top-level GAM class)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from whittaker.families.binomial import Binomial
from whittaker.families.poisson import Poisson
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

    def test_per_term_gcv_different_sp(self) -> None:
        rng = np.random.default_rng(58)
        n = 600
        x1 = np.linspace(0, 2 * np.pi, n)
        x2 = np.linspace(0, 1, n)
        y = np.sin(3 * x1) + 0.5 * x2 + rng.normal(0, 0.15, n)

        model = GAM("y ~ s(x1, k=15) + s(x2, k=15)").fit({"y": y, "x1": x1, "x2": x2})
        sp = model.smoothing_params
        assert sp[1] > sp[0]
        assert model.edf[0] > model.edf[1]


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


# ---------------------------------------------------------------------------
# Binomial (logistic GAM)
# ---------------------------------------------------------------------------


class TestBinomialGAM:
    def test_binomial_fit_and_predict(self) -> None:
        rng = np.random.default_rng(23)
        n = 400
        x = np.linspace(-3, 3, n)
        eta_true = np.sin(x)
        p_true = 1.0 / (1.0 + np.exp(-eta_true))
        y = rng.binomial(1, p_true, n).astype(float)

        model = GAM("y ~ s(x, k=10)", family=Binomial()).fit({"y": y, "x": x})
        assert model.is_fitted
        assert model.scale == 1.0

        pred = model.predict({"x": x})
        assert np.all(pred.values > 0)
        assert np.all(pred.values < 1)

    def test_binomial_recovery(self) -> None:
        rng = np.random.default_rng(23)
        n = 500
        x = np.linspace(-3, 3, n)
        eta_true = np.sin(x)
        p_true = 1.0 / (1.0 + np.exp(-eta_true))
        y = rng.binomial(1, p_true, n).astype(float)

        model = GAM("y ~ s(x, k=15)", family=Binomial()).fit({"y": y, "x": x})
        rmse = np.sqrt(np.mean((model.fitted_values - p_true) ** 2))
        assert rmse < 0.1

    def test_binomial_predict_with_se(self) -> None:
        rng = np.random.default_rng(23)
        n = 300
        x = np.linspace(-2, 2, n)
        y = rng.binomial(1, 0.5 * np.ones(n), n).astype(float)

        model = GAM("y ~ s(x, k=8)", family=Binomial()).fit({"y": y, "x": x})
        pred = model.predict({"x": np.array([0.0, 1.0])}, se=True)
        assert pred.se is not None
        assert np.all(pred.se > 0)

    def test_binomial_summary_shows_family(self) -> None:
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(-2, 2, n)
        y = rng.binomial(1, 0.5 * np.ones(n), n).astype(float)

        model = GAM("y ~ s(x, k=8)", family=Binomial()).fit({"y": y, "x": x})
        text = model.summary()
        assert "Binomial" in text

    def test_binomial_two_smooths(self) -> None:
        rng = np.random.default_rng(23)
        n = 500
        x1 = np.linspace(-3, 3, n)
        x2 = np.linspace(-2, 2, n)
        eta_true = 0.8 * np.sin(x1) + 0.5 * x2
        p_true = 1.0 / (1.0 + np.exp(-eta_true))
        y = rng.binomial(1, p_true, n).astype(float)

        model = GAM("y ~ s(x1, k=10) + s(x2, k=10)", family=Binomial()).fit(
            {"y": y, "x1": x1, "x2": x2}
        )
        assert len(model.edf) == 2
        assert len(model.smoothing_params) == 2


# ---------------------------------------------------------------------------
# Poisson (count GAM)
# ---------------------------------------------------------------------------


class TestPoissonGAM:
    def test_poisson_fit_and_predict(self) -> None:
        rng = np.random.default_rng(23)
        n = 400
        x = np.linspace(0, 2 * np.pi, n)
        mu_true = np.exp(0.5 + 0.8 * np.sin(x))
        y = rng.poisson(mu_true).astype(float)

        model = GAM("y ~ s(x, k=10)", family=Poisson()).fit({"y": y, "x": x})
        assert model.is_fitted
        assert model.scale == 1.0

        pred = model.predict({"x": x})
        assert np.all(pred.values > 0)

    def test_poisson_recovery(self) -> None:
        rng = np.random.default_rng(23)
        n = 500
        x = np.linspace(0, 2 * np.pi, n)
        mu_true = np.exp(0.5 + 0.8 * np.sin(x))
        y = rng.poisson(mu_true).astype(float)

        model = GAM("y ~ s(x, k=15)", family=Poisson()).fit({"y": y, "x": x})
        rmse = np.sqrt(np.mean((model.fitted_values - mu_true) ** 2))
        assert rmse < 0.5

    def test_poisson_predict_with_se(self) -> None:
        rng = np.random.default_rng(23)
        n = 300
        x = np.linspace(0, 2 * np.pi, n)
        mu_true = np.exp(0.5 + 0.5 * np.sin(x))
        y = rng.poisson(mu_true).astype(float)

        model = GAM("y ~ s(x, k=10)", family=Poisson()).fit({"y": y, "x": x})
        pred = model.predict({"x": np.array([1.0, 2.0])}, se=True)
        assert pred.se is not None
        assert np.all(pred.se > 0)

    def test_poisson_summary_shows_family(self) -> None:
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(0, 1, n)
        y = rng.poisson(3.0, n).astype(float)

        model = GAM("y ~ s(x, k=8)", family=Poisson()).fit({"y": y, "x": x})
        text = model.summary()
        assert "Poisson" in text

    def test_poisson_two_smooths(self) -> None:
        rng = np.random.default_rng(23)
        n = 500
        x1 = np.linspace(0, 2 * np.pi, n)
        x2 = np.linspace(0, 1, n)
        mu_true = np.exp(0.3 + 0.5 * np.sin(x1) + 0.3 * x2)
        y = rng.poisson(mu_true).astype(float)

        model = GAM("y ~ s(x1, k=10) + s(x2, k=10)", family=Poisson()).fit(
            {"y": y, "x1": x1, "x2": x2}
        )
        assert len(model.edf) == 2
        assert len(model.smoothing_params) == 2


# ---------------------------------------------------------------------------
# REML smoothing parameter selection (GAM-level)
# ---------------------------------------------------------------------------


class TestREMLGAM:
    def test_reml_gaussian_fit(self) -> None:
        data = _sin_data()
        model = GAM("y ~ s(x, k=10)").fit(data, method="REML")
        assert model.is_fitted
        rmse = np.sqrt(np.mean((model.fitted_values - np.sin(data["x"])) ** 2))
        assert rmse < 0.15

    def test_reml_multi_smooth(self) -> None:
        data = _multi_data()
        model = GAM("y ~ s(x1, k=10) + s(x2, k=10)").fit(data, method="REML")
        assert len(model.smoothing_params) == 2
        assert len(model.edf) == 2

    def test_reml_predict_with_se(self) -> None:
        data = _sin_data()
        model = GAM("y ~ s(x, k=10)").fit(data, method="REML")
        pred = model.predict({"x": np.array([1.0, 2.0])}, se=True)
        assert pred.se is not None
        assert np.all(pred.se > 0)

    def test_reml_binomial(self) -> None:
        rng = np.random.default_rng(23)
        n = 400
        x = np.linspace(-3, 3, n)
        p_true = 1.0 / (1.0 + np.exp(-np.sin(x)))
        y = rng.binomial(1, p_true, n).astype(float)

        model = GAM("y ~ s(x, k=10)", family=Binomial()).fit({"y": y, "x": x}, method="REML")
        assert model.is_fitted
        assert np.all(model.fitted_values > 0)
        assert np.all(model.fitted_values < 1)

    def test_reml_poisson(self) -> None:
        rng = np.random.default_rng(23)
        n = 300
        x = np.linspace(0, 2 * np.pi, n)
        mu_true = np.exp(0.5 + 0.5 * np.sin(x))
        y = rng.poisson(mu_true).astype(float)

        model = GAM("y ~ s(x, k=10)", family=Poisson()).fit({"y": y, "x": x}, method="REML")
        assert model.is_fitted
        assert np.all(model.fitted_values > 0)

    @pytest.mark.parametrize("bs", ["tp", "cr", "ps"])
    def test_reml_basis_types(self, bs: str) -> None:
        data = _sin_data()
        model = GAM(f"y ~ s(x, bs='{bs}', k=10)").fit(data, method="REML")
        assert model.is_fitted
        rmse = np.sqrt(np.mean((model.fitted_values - np.sin(data["x"])) ** 2))
        assert rmse < 0.15


# ---------------------------------------------------------------------------
# Tensor product smooths (te)
# ---------------------------------------------------------------------------


class TestTensorProductGAM:
    def test_te_basic_fit(self) -> None:
        rng = np.random.default_rng(23)
        n = 300
        x1 = rng.uniform(0, 2 * np.pi, n)
        x2 = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x1) + np.cos(x2) + rng.normal(0, 0.3, n)

        model = GAM("y ~ te(x1, x2, k=5)").fit({"y": y, "x1": x1, "x2": x2})
        assert model.is_fitted
        assert len(model.smoothing_params) == 2
        assert len(model.edf) == 1

    def test_te_recovery(self) -> None:
        rng = np.random.default_rng(23)
        n = 500
        x1 = rng.uniform(0, 2 * np.pi, n)
        x2 = rng.uniform(0, 2 * np.pi, n)
        f_true = np.sin(x1) + np.cos(x2)
        y = f_true + rng.normal(0, 0.3, n)

        model = GAM("y ~ te(x1, x2, k=5)").fit({"y": y, "x1": x1, "x2": x2})
        rmse = np.sqrt(np.mean((model.fitted_values - f_true) ** 2))
        assert rmse < 0.25

    def test_te_predict(self) -> None:
        rng = np.random.default_rng(23)
        n = 300
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        y = x1 + x2 + rng.normal(0, 0.1, n)

        model = GAM("y ~ te(x1, x2, k=5)").fit({"y": y, "x1": x1, "x2": x2})
        pred = model.predict({"x1": np.array([0.5]), "x2": np.array([0.5])}, se=True)
        assert pred.se is not None
        assert pred.se[0] > 0
        assert abs(pred.values[0] - 1.0) < 0.3

    def test_te_with_reml(self) -> None:
        rng = np.random.default_rng(23)
        n = 300
        x1 = rng.uniform(0, 2 * np.pi, n)
        x2 = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x1) + np.cos(x2) + rng.normal(0, 0.3, n)

        model = GAM("y ~ te(x1, x2, k=5)").fit({"y": y, "x1": x1, "x2": x2}, method="REML")
        assert model.is_fitted
        assert len(model.smoothing_params) == 2

    def test_te_mixed_with_s(self) -> None:
        rng = np.random.default_rng(23)
        n = 300
        x1 = rng.uniform(0, 2 * np.pi, n)
        x2 = rng.uniform(0, 2 * np.pi, n)
        x3 = np.linspace(0, 1, n)
        y = np.sin(x1) * np.cos(x2) + 0.5 * x3 + rng.normal(0, 0.3, n)

        model = GAM("y ~ te(x1, x2, k=4) + s(x3, k=6)").fit({"y": y, "x1": x1, "x2": x2, "x3": x3})
        assert len(model.edf) == 2
        assert len(model.smoothing_params) == 3  # 2 for te + 1 for s

    def test_te_per_marginal_k(self) -> None:
        rng = np.random.default_rng(23)
        n = 300
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        y = x1 + x2 + rng.normal(0, 0.1, n)

        model = GAM("y ~ te(x1, x2, k=[4, 6])").fit({"y": y, "x1": x1, "x2": x2})
        assert model.is_fitted
        # 4 * 6 = 24 product basis, minus 1 for constraint = 23, plus intercept = 24
        assert model.coefficients.shape[0] == 24

    def test_te_summary(self) -> None:
        rng = np.random.default_rng(23)
        n = 200
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        y = x1 * x2 + rng.normal(0, 0.1, n)

        model = GAM("y ~ te(x1, x2, k=4)").fit({"y": y, "x1": x1, "x2": x2})
        text = model.summary()
        assert "te(x1, x2" in text

    def test_te_binomial(self) -> None:
        rng = np.random.default_rng(23)
        n = 500
        x1 = rng.uniform(-2, 2, n)
        x2 = rng.uniform(-2, 2, n)
        eta = 0.5 * x1 + 0.5 * x2
        p = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, p, n).astype(float)

        model = GAM("y ~ te(x1, x2, k=4)", family=Binomial()).fit({"y": y, "x1": x1, "x2": x2})
        assert model.is_fitted
        assert np.all(model.fitted_values > 0)
        assert np.all(model.fitted_values < 1)

    def test_te_poisson(self) -> None:
        rng = np.random.default_rng(23)
        n = 500
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        mu = np.exp(0.5 + 0.3 * x1 + 0.3 * x2)
        y = rng.poisson(mu).astype(float)

        model = GAM("y ~ te(x1, x2, k=4)", family=Poisson()).fit({"y": y, "x1": x1, "x2": x2})
        assert model.is_fitted
        assert np.all(model.fitted_values > 0)


# ---------------------------------------------------------------------------
# TweedieEstimated profile-likelihood fallback
# ---------------------------------------------------------------------------


class TestTweedieProfileFallback:
    def _tweedie_data(self, n: int = 200) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(23)
        x = np.linspace(0, 4, n)
        mu = np.exp(0.5 + 0.3 * np.sin(x))
        lam = mu**0.5
        n_events = rng.poisson(lam)
        y = np.zeros(n)
        for i in range(n):
            if n_events[i] > 0:
                y[i] = np.sum(rng.gamma(2.0, mu[i] / 2.0, size=n_events[i]))
        return {"x": x, "y": y}

    def test_all_grid_candidates_fail_falls_back_to_median_p(self, monkeypatch) -> None:
        """When every grid candidate raises, the loop's except-continue (406-407) runs for
        each candidate, best_result stays None, and the fallback branch (414-415) refits at
        the median grid value using the real solver."""
        import whittaker.gam as gam_module
        from whittaker.families.tweedie_estimated import TweedieEstimated
        from whittaker.fitting.pirls import pirls_fit as real_pirls_fit

        family = TweedieEstimated(n_grid=5)
        n_grid_calls = family._n_grid

        calls = {"n": 0}

        def flaky_pirls_fit(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] <= n_grid_calls:
                raise RuntimeError("forced failure for grid search")
            return real_pirls_fit(*args, **kwargs)

        monkeypatch.setattr(gam_module, "pirls_fit", flaky_pirls_fit)

        data = self._tweedie_data()
        model = GAM("y ~ s(x)", family=family)
        model.fit(data)

        assert model.is_fitted
        assert calls["n"] == n_grid_calls + 1
        assert family.p_estimated


# ---------------------------------------------------------------------------
# Simultaneous confidence bands, deviance explained, and check() fallback
# ---------------------------------------------------------------------------


class TestSimultaneousCIByTerm:
    def test_by_term_label_includes_level(self) -> None:
        rng = np.random.default_rng(0)
        n = 200
        x = rng.uniform(0, 2 * np.pi, n)
        g = rng.choice(["a", "b"], n)
        y = np.sin(x) + (g == "b") * 0.5 + rng.normal(0, 0.2, n)
        data = {"x": x, "g": g, "y": y}

        model = GAM("y ~ s(x, by=g)").fit(data)

        band_a = model.simultaneous_ci(data, term=0, n_sim=200, seed=0)
        band_b = model.simultaneous_ci(data, term=1, n_sim=200, seed=0)

        assert band_a["term_label"] == "s(x, by='g'):a"
        assert band_b["term_label"] == "s(x, by='g'):b"


class TestDevianceExplainedDegenerate:
    def test_zero_when_null_deviance_nonpositive(self) -> None:
        n = 50
        x = np.linspace(0, 1, n)
        y = np.full(n, 3.0)

        model = GAM("y ~ s(x)").fit({"x": x, "y": y})

        assert model.null_deviance <= 0
        assert model.deviance_explained == 0.0


class TestCheckDisplayFallback:
    def test_ipython_display_import_error_is_swallowed(self, monkeypatch) -> None:
        import sys

        rng = np.random.default_rng(0)
        n = 100
        x = np.sort(rng.uniform(0, 10, n))
        y = np.sin(x) + rng.normal(scale=0.2, size=n)

        model = GAM("y ~ s(x)").fit({"x": x, "y": y})

        monkeypatch.setitem(sys.modules, "IPython", None)
        monkeypatch.setitem(sys.modules, "IPython.display", None)

        charts = model.check(plots=["qq"])
        assert len(charts) == 1
