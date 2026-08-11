"""Tests for multi-response GAMs."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.gam import GAM
from whittaker.multi_response import (
    MultiResponseGAM,
    MultiResponseResult,
    ResidualCorrelation,
)


@pytest.fixture
def correlated_data():
    """Two correlated responses driven by same smooth + noise."""
    rng = np.random.default_rng(23)
    n = 300
    x = np.linspace(0, 2 * np.pi, n)
    z = rng.uniform(0, 1, n)
    e1 = rng.normal(0, 0.3, n)
    e2 = 0.7 * e1 + rng.normal(0, 0.2, n)
    y1 = np.sin(x) + 2 * z + e1
    y2 = 0.5 * np.sin(x) + z + e2
    return {"x": x, "z": z, "y1": y1, "y2": y2}


@pytest.fixture
def three_response_data():
    rng = np.random.default_rng(23)
    n = 200
    x = np.linspace(0, 2 * np.pi, n)
    y1 = np.sin(x) + rng.normal(0, 0.2, n)
    y2 = np.cos(x) + rng.normal(0, 0.2, n)
    y3 = 0.5 * x + rng.normal(0, 0.2, n)
    return {"x": x, "y1": y1, "y2": y2, "y3": y3}


class TestMultiResponseGAMInit:
    def test_basic_init(self):
        model = MultiResponseGAM(["y1", "y2"], "s(x)")
        assert model.responses == ["y1", "y2"]
        assert model.n_responses == 2
        assert model.correlation == "independent"
        assert not model.is_fitted

    def test_unstructured_correlation(self):
        model = MultiResponseGAM(["y1", "y2"], "s(x)", correlation="unstructured")
        assert model.correlation == "unstructured"

    def test_single_response_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            MultiResponseGAM(["y1"], "s(x)")

    def test_duplicate_responses_raises(self):
        with pytest.raises(ValueError, match="unique"):
            MultiResponseGAM(["y1", "y1"], "s(x)")

    def test_invalid_correlation_raises(self):
        with pytest.raises(ValueError, match="correlation must be"):
            MultiResponseGAM(["y1", "y2"], "s(x)", correlation="ar1")

    def test_repr_unfitted(self):
        model = MultiResponseGAM(["y1", "y2"], "s(x)")
        r = repr(model)
        assert "MultiResponseGAM" in r
        assert "unfitted" in r


class TestMultiResponseFit:
    def test_fit_independent(self, correlated_data):
        model = MultiResponseGAM(["y1", "y2"], "s(x) + s(z)")
        model.fit(correlated_data)
        assert model.is_fitted

    def test_fit_unstructured(self, correlated_data):
        model = MultiResponseGAM(["y1", "y2"], "s(x) + s(z)", correlation="unstructured")
        model.fit(correlated_data)
        assert model.is_fitted

    def test_fit_returns_self(self, correlated_data):
        model = MultiResponseGAM(["y1", "y2"], "s(x)")
        result = model.fit(correlated_data)
        assert result is model

    def test_missing_response_raises(self):
        data = {"x": np.array([1.0, 2.0]), "y1": np.array([1.0, 2.0])}
        model = MultiResponseGAM(["y1", "y2"], "s(x)")
        with pytest.raises(ValueError, match="not found"):
            model.fit(data)

    def test_unfitted_raises(self):
        model = MultiResponseGAM(["y1", "y2"], "s(x)")
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict({"x": np.array([1.0])})

    def test_three_responses(self, three_response_data):
        model = MultiResponseGAM(["y1", "y2", "y3"], "s(x)")
        model.fit(three_response_data)
        assert model.is_fitted
        assert model.n_responses == 3


class TestMultiResponsePredict:
    def test_predict_shape(self, correlated_data):
        model = MultiResponseGAM(["y1", "y2"], "s(x) + s(z)")
        model.fit(correlated_data)
        new_data = {"x": np.linspace(0, 2 * np.pi, 50), "z": np.full(50, 0.5)}
        result = model.predict(new_data)
        assert isinstance(result, MultiResponseResult)
        assert result["y1"].values.shape == (50,)
        assert result["y2"].values.shape == (50,)

    def test_predict_with_se(self, correlated_data):
        model = MultiResponseGAM(["y1", "y2"], "s(x)")
        model.fit(correlated_data)
        new_data = {"x": np.linspace(0, 2 * np.pi, 30)}
        result = model.predict(new_data, se=True)
        assert result["y1"].se is not None
        assert result["y2"].se is not None

    def test_predict_iteration(self, correlated_data):
        model = MultiResponseGAM(["y1", "y2"], "s(x)")
        model.fit(correlated_data)
        new_data = {"x": np.linspace(0, 2 * np.pi, 20)}
        result = model.predict(new_data)
        resps = list(result)
        assert resps == ["y1", "y2"]

    def test_predict_finite(self, correlated_data):
        model = MultiResponseGAM(["y1", "y2"], "s(x) + s(z)")
        model.fit(correlated_data)
        new_data = {"x": np.linspace(0, 2 * np.pi, 30), "z": np.full(30, 0.5)}
        result = model.predict(new_data)
        assert np.all(np.isfinite(result["y1"].values))
        assert np.all(np.isfinite(result["y2"].values))

    def test_captures_different_patterns(self, correlated_data):
        model = MultiResponseGAM(["y1", "y2"], "s(x)")
        model.fit(correlated_data)
        x_test = np.linspace(0.5, 2 * np.pi - 0.5, 50)
        result = model.predict({"x": x_test})
        y1_range = result["y1"].values.max() - result["y1"].values.min()
        y2_range = result["y2"].values.max() - result["y2"].values.min()
        assert y1_range > y2_range


class TestJointPredict:
    def test_joint_predict_shape(self, correlated_data):
        model = MultiResponseGAM(["y1", "y2"], "s(x)")
        model.fit(correlated_data)
        new_data = {"x": np.linspace(0, 2 * np.pi, 40)}
        preds, cov = model.joint_predict(new_data)
        assert preds.shape == (40, 2)
        assert cov is None

    def test_joint_predict_unstructured(self, correlated_data):
        model = MultiResponseGAM(["y1", "y2"], "s(x)", correlation="unstructured")
        model.fit(correlated_data)
        new_data = {"x": np.linspace(0, 2 * np.pi, 40)}
        preds, cov = model.joint_predict(new_data)
        assert preds.shape == (40, 2)
        assert cov is not None
        assert cov.shape == (2, 2)


class TestResidualCorrelationEstimation:
    def test_correlation_matrix(self, correlated_data):
        model = MultiResponseGAM(["y1", "y2"], "s(x) + s(z)", correlation="unstructured")
        model.fit(correlated_data)
        rc = model.residual_correlation()
        assert isinstance(rc, ResidualCorrelation)
        assert rc.correlation.shape == (2, 2)
        assert rc.covariance.shape == (2, 2)

    def test_diagonal_is_one(self, correlated_data):
        model = MultiResponseGAM(["y1", "y2"], "s(x) + s(z)", correlation="unstructured")
        model.fit(correlated_data)
        rc = model.residual_correlation()
        np.testing.assert_allclose(np.diag(rc.correlation), 1.0, atol=1e-10)

    def test_detects_positive_correlation(self, correlated_data):
        model = MultiResponseGAM(["y1", "y2"], "s(x) + s(z)", correlation="unstructured")
        model.fit(correlated_data)
        rc = model.residual_correlation()
        assert rc.correlation[0, 1] > 0.3

    def test_symmetric(self, correlated_data):
        model = MultiResponseGAM(["y1", "y2"], "s(x) + s(z)", correlation="unstructured")
        model.fit(correlated_data)
        rc = model.residual_correlation()
        np.testing.assert_allclose(rc.correlation, rc.correlation.T)
        np.testing.assert_allclose(rc.covariance, rc.covariance.T)

    def test_independent_raises(self, correlated_data):
        model = MultiResponseGAM(["y1", "y2"], "s(x)")
        model.fit(correlated_data)
        with pytest.raises(ValueError, match="not estimated"):
            model.residual_correlation()

    def test_repr(self, correlated_data):
        model = MultiResponseGAM(["y1", "y2"], "s(x)", correlation="unstructured")
        model.fit(correlated_data)
        rc = model.residual_correlation()
        r = repr(rc)
        assert "y1" in r
        assert "y2" in r

    def test_three_response_correlation(self, three_response_data):
        model = MultiResponseGAM(["y1", "y2", "y3"], "s(x)", correlation="unstructured")
        model.fit(three_response_data)
        rc = model.residual_correlation()
        assert rc.correlation.shape == (3, 3)
        np.testing.assert_allclose(np.diag(rc.correlation), 1.0, atol=1e-10)


class TestResponseModel:
    def test_get_response_model(self, correlated_data):
        model = MultiResponseGAM(["y1", "y2"], "s(x)")
        model.fit(correlated_data)
        gam = model.response_model("y1")
        assert isinstance(gam, GAM)
        assert gam.is_fitted

    def test_invalid_response_raises(self, correlated_data):
        model = MultiResponseGAM(["y1", "y2"], "s(x)")
        model.fit(correlated_data)
        with pytest.raises(ValueError, match="not found"):
            model.response_model("y3")


class TestEDFDeviance:
    def test_edf_per_response(self, correlated_data):
        model = MultiResponseGAM(["y1", "y2"], "s(x)")
        model.fit(correlated_data)
        edfs = model.edf()
        assert "y1" in edfs
        assert "y2" in edfs
        assert all(e > 0 for e in edfs.values())

    def test_deviance_per_response(self, correlated_data):
        model = MultiResponseGAM(["y1", "y2"], "s(x)")
        model.fit(correlated_data)
        devs = model.deviance()
        assert "y1" in devs
        assert "y2" in devs
        assert all(d > 0 for d in devs.values())


class TestResponseSpecificFormulas:
    def test_response_specific_terms(self, correlated_data):
        model = MultiResponseGAM(
            ["y1", "y2"],
            "s(x)",
            response_formulas={"y1": "s(z)"},
        )
        model.fit(correlated_data)
        assert model.is_fitted
        edf_y1 = model.edf()["y1"]
        edf_y2 = model.edf()["y2"]
        assert edf_y1 > edf_y2


class TestSummary:
    def test_summary_content(self, correlated_data):
        model = MultiResponseGAM(["y1", "y2"], "s(x) + s(z)", correlation="unstructured")
        model.fit(correlated_data)
        s = model.summary()
        assert "MultiResponseGAM" in s
        assert "y1" in s
        assert "y2" in s
        assert "corr(" in s

    def test_summary_independent(self, correlated_data):
        model = MultiResponseGAM(["y1", "y2"], "s(x)")
        model.fit(correlated_data)
        s = model.summary()
        assert "MultiResponseGAM" in s
        assert "corr(" not in s

    def test_repr_fitted(self, correlated_data):
        model = MultiResponseGAM(["y1", "y2"], "s(x)")
        model.fit(correlated_data)
        r = repr(model)
        assert "fitted" in r


class TestAgreementWithSeparateGAMs:
    def test_independent_matches_separate(self, correlated_data):
        model = MultiResponseGAM(["y1", "y2"], "s(x)")
        model.fit(correlated_data)

        gam1 = GAM("y1 ~ s(x)")
        gam1.fit(correlated_data)
        gam2 = GAM("y2 ~ s(x)")
        gam2.fit(correlated_data)

        x_test = np.linspace(0.5, 2 * np.pi - 0.5, 30)
        mr_pred = model.predict({"x": x_test})
        g1_pred = gam1.predict({"x": x_test})
        g2_pred = gam2.predict({"x": x_test})

        np.testing.assert_allclose(mr_pred["y1"].values, g1_pred.values, atol=0.05)
        np.testing.assert_allclose(mr_pred["y2"].values, g2_pred.values, atol=0.05)
