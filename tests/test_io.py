"""Tests for model import/export."""

from __future__ import annotations

import json

import numpy as np
import pytest

from whittaker.families.gaussian import Gaussian
from whittaker.families.negative_binomial import NegativeBinomial
from whittaker.families.poisson import Poisson
from whittaker.formula.terms import SmoothTerm
from whittaker.gam import GAM
from whittaker.io import (
    _family_from_dict,
    _family_to_dict,
    _formula_from_dict,
    _formula_to_dict,
    from_mgcv_dict,
    load_gam,
    save_gam,
    to_mgcv_dict,
)


@pytest.fixture
def sin_data():
    rng = np.random.default_rng(23)
    x = np.linspace(0, 2 * np.pi, 200)
    y = np.sin(x) + rng.normal(0, 0.2, 200)
    return {"x": x, "y": y}


@pytest.fixture
def fitted_gam(sin_data):
    model = GAM("y ~ s(x)")
    model.fit(sin_data)
    return model


@pytest.fixture
def two_smooth_data():
    rng = np.random.default_rng(23)
    n = 200
    x1 = np.linspace(0, 2 * np.pi, n)
    x2 = rng.uniform(0, 1, n)
    y = np.sin(x1) + x2**2 + rng.normal(0, 0.2, n)
    return {"x1": x1, "x2": x2, "y": y}


class TestFamilySerialization:
    def test_gaussian_roundtrip(self):
        fam = Gaussian()
        d = _family_to_dict(fam)
        assert d["class"] == "Gaussian"
        restored = _family_from_dict(d)
        assert isinstance(restored, Gaussian)

    def test_poisson_roundtrip(self):
        fam = Poisson()
        d = _family_to_dict(fam)
        restored = _family_from_dict(d)
        assert isinstance(restored, Poisson)

    def test_negative_binomial_roundtrip(self):
        fam = NegativeBinomial()
        fam.theta = 3.5
        d = _family_to_dict(fam)
        assert d["theta"] == 3.5
        restored = _family_from_dict(d)
        assert isinstance(restored, NegativeBinomial)
        assert restored.theta == 3.5

    def test_unknown_family_raises(self):
        with pytest.raises(ValueError, match="Unknown family"):
            _family_from_dict({"class": "FakeFamily"})


class TestFormulaSerialization:
    def test_simple_roundtrip(self):
        from whittaker.formula.parser import parse

        formula = parse("y ~ s(x1) + s(x2) + x3")
        d = _formula_to_dict(formula)
        restored = _formula_from_dict(d)
        assert restored.response == "y"
        assert restored.intercept == formula.intercept
        assert len(restored.terms) == len(formula.terms)

    def test_smooth_term_preserved(self):
        from whittaker.formula.parser import parse

        formula = parse("y ~ s(x, k=15, bs='cr')")
        d = _formula_to_dict(formula)
        restored = _formula_from_dict(d)
        smooth = [t for t in restored.terms if isinstance(t, SmoothTerm)][0]
        assert smooth.variables == ("x",)
        assert smooth.bs == "cr"
        assert smooth.k == 15


class TestSaveLoadGAM:
    def test_roundtrip(self, fitted_gam, sin_data, tmp_path):
        path = tmp_path / "model.npz"
        save_gam(fitted_gam, path)
        assert path.exists()

        loaded = load_gam(path)
        assert loaded.is_fitted
        assert loaded.coefficients.shape == fitted_gam.coefficients.shape
        np.testing.assert_allclose(loaded.coefficients, fitted_gam.coefficients)

    def test_predictions_match(self, fitted_gam, sin_data, tmp_path):
        path = tmp_path / "model.npz"
        save_gam(fitted_gam, path)
        loaded = load_gam(path)

        pred_orig = fitted_gam.predict(sin_data)
        pred_loaded = loaded.predict(sin_data)
        np.testing.assert_allclose(pred_loaded.values, pred_orig.values, atol=1e-10)

    def test_predictions_with_se(self, fitted_gam, sin_data, tmp_path):
        path = tmp_path / "model.npz"
        save_gam(fitted_gam, path)
        loaded = load_gam(path)

        pred_orig = fitted_gam.predict(sin_data, se=True)
        pred_loaded = loaded.predict(sin_data, se=True)
        np.testing.assert_allclose(pred_loaded.se, pred_orig.se, atol=1e-10)

    def test_summary_works(self, fitted_gam, tmp_path):
        path = tmp_path / "model.npz"
        save_gam(fitted_gam, path)
        loaded = load_gam(path)
        s = loaded.summary()
        assert "s(x)" in s

    def test_two_smooths(self, two_smooth_data, tmp_path):
        model = GAM("y ~ s(x1) + s(x2)")
        model.fit(two_smooth_data)

        path = tmp_path / "model2.npz"
        save_gam(model, path)
        loaded = load_gam(path)

        pred_orig = model.predict(two_smooth_data)
        pred_loaded = loaded.predict(two_smooth_data)
        np.testing.assert_allclose(pred_loaded.values, pred_orig.values, atol=1e-10)

    def test_poisson_roundtrip(self, tmp_path):
        rng = np.random.default_rng(23)
        x = np.linspace(0, 3, 200)
        y = rng.poisson(np.exp(0.5 * x))
        data = {"x": x, "y": y.astype(float)}

        model = GAM("y ~ s(x)", family=Poisson())
        model.fit(data)

        path = tmp_path / "poisson.npz"
        save_gam(model, path)
        loaded = load_gam(path)

        assert isinstance(loaded.family, Poisson)
        pred = loaded.predict(data)
        np.testing.assert_allclose(pred.values, model.predict(data).values, atol=1e-10)

    def test_cr_basis_roundtrip(self, sin_data, tmp_path):
        model = GAM("y ~ s(x, bs='cr')")
        model.fit(sin_data)

        path = tmp_path / "cr.npz"
        save_gam(model, path)
        loaded = load_gam(path)

        pred = loaded.predict(sin_data)
        np.testing.assert_allclose(pred.values, model.predict(sin_data).values, atol=1e-10)

    def test_ps_basis_roundtrip(self, sin_data, tmp_path):
        model = GAM("y ~ s(x, bs='ps')")
        model.fit(sin_data)

        path = tmp_path / "ps.npz"
        save_gam(model, path)
        loaded = load_gam(path)

        pred = loaded.predict(sin_data)
        np.testing.assert_allclose(pred.values, model.predict(sin_data).values, atol=1e-10)

    def test_unfitted_raises(self, tmp_path):
        model = GAM("y ~ s(x)")
        with pytest.raises(RuntimeError, match="unfitted"):
            save_gam(model, tmp_path / "nope.npz")

    def test_non_gam_raises(self, tmp_path):
        with pytest.raises(TypeError, match="GAM"):
            save_gam("not a model", tmp_path / "nope.npz")

    def test_fit_metadata_preserved(self, fitted_gam, tmp_path):
        path = tmp_path / "model.npz"
        save_gam(fitted_gam, path)
        loaded = load_gam(path)

        assert loaded.deviance == pytest.approx(fitted_gam.deviance, rel=1e-10)
        assert loaded.null_deviance == pytest.approx(fitted_gam.null_deviance, rel=1e-10)
        np.testing.assert_allclose(
            loaded._fit_result.smoothing_params,
            fitted_gam._fit_result.smoothing_params,
        )


class TestMgcvExport:
    def test_export_structure(self, fitted_gam):
        d = to_mgcv_dict(fitted_gam)
        assert "coefficients" in d
        assert "sp" in d
        assert "smooth" in d
        assert "family" in d
        assert d["converged"] is True
        assert d["n"] == 200

    def test_coefficients_list(self, fitted_gam):
        d = to_mgcv_dict(fitted_gam)
        assert isinstance(d["coefficients"], list)
        assert len(d["coefficients"]) == len(fitted_gam.coefficients)

    def test_smooth_info(self, fitted_gam):
        d = to_mgcv_dict(fitted_gam)
        assert len(d["smooth"]) == 1
        s = d["smooth"][0]
        assert "term" in s
        assert "bs" in s
        assert "S" in s

    def test_json_serializable(self, fitted_gam):
        d = to_mgcv_dict(fitted_gam)
        json_str = json.dumps(d)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["n"] == 200

    def test_unfitted_raises(self):
        model = GAM("y ~ s(x)")
        with pytest.raises(RuntimeError, match="unfitted"):
            to_mgcv_dict(model)


class TestMgcvImport:
    def test_import_with_data(self, sin_data):
        model = GAM("y ~ s(x)")
        model.fit(sin_data)
        mgcv_d = to_mgcv_dict(model)

        imported = from_mgcv_dict(mgcv_d, data=sin_data)
        assert imported.is_fitted
        pred = imported.predict(sin_data)
        assert pred.values.shape == (200,)
        assert np.all(np.isfinite(pred.values))

    def test_import_without_data(self):
        d = {
            "coefficients": [0.0, 0.1, 0.2],
            "sp": [1.0],
            "formula": "y ~ s(x)",
            "family": {"family": "gaussian"},
            "smooth": [{"term": ["x"], "bs": "tp", "df": 3}],
            "intercept": True,
        }
        imported = from_mgcv_dict(d)
        assert not imported.is_fitted

    def test_roundtrip_via_mgcv(self, sin_data):
        model = GAM("y ~ s(x)")
        model.fit(sin_data)
        mgcv_d = to_mgcv_dict(model)
        imported = from_mgcv_dict(mgcv_d, data=sin_data)

        pred_orig = model.predict(sin_data)
        pred_imported = imported.predict(sin_data)
        np.testing.assert_allclose(pred_imported.values, pred_orig.values, atol=0.1)
