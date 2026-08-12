"""Tests for model import/export."""

from __future__ import annotations

import json

import numpy as np
import pytest

from whittaker.families.cox_ph import CoxPH
from whittaker.families.gaussian import Gaussian
from whittaker.families.multinomial import Multinomial
from whittaker.families.negative_binomial import NegativeBinomial
from whittaker.families.ordered_categorical import OrderedCategorical
from whittaker.families.poisson import Poisson
from whittaker.families.quantile import QuantileFamily
from whittaker.families.tweedie import Tweedie
from whittaker.families.tweedie_estimated import TweedieEstimated
from whittaker.formula.terms import Formula, SmoothTerm
from whittaker.gam import GAM
from whittaker.io import (
    _basis_from_state,
    _basis_state,
    _family_from_dict,
    _family_to_dict,
    _formula_from_dict,
    _formula_to_dict,
    _term_from_dict,
    _term_to_dict,
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

    def test_tweedie_roundtrip(self):
        fam = Tweedie(p=1.7)
        d = _family_to_dict(fam)
        assert d["class"] == "Tweedie"
        assert d["p"] == pytest.approx(1.7)
        restored = _family_from_dict(d)
        assert isinstance(restored, Tweedie)
        assert restored._p == pytest.approx(1.7)

    def test_tweedie_estimated_roundtrip(self):
        fam = TweedieEstimated(p_range=(1.1, 1.9), n_grid=15)
        fam._set_p(1.42)
        d = _family_to_dict(fam)
        assert d["class"] == "TweedieEstimated"
        assert d["p"] == pytest.approx(1.42)
        assert d["p_range"] == [1.1, 1.9]
        assert d["n_grid"] == 15
        restored = _family_from_dict(d)
        assert isinstance(restored, TweedieEstimated)
        assert restored._p == pytest.approx(1.42)
        assert restored._p_range == (1.1, 1.9)
        assert restored._n_grid == 15
        assert restored.p_estimated is True

    def test_quantile_family_roundtrip(self):
        fam = QuantileFamily(tau=0.25)
        d = _family_to_dict(fam)
        assert d["class"] == "QuantileFamily"
        assert d["tau"] == pytest.approx(0.25)
        restored = _family_from_dict(d)
        assert isinstance(restored, QuantileFamily)
        assert restored.tau == pytest.approx(0.25)

    def test_ordered_categorical_roundtrip_no_cutpoints(self):
        fam = OrderedCategorical(n_categories=4)
        d = _family_to_dict(fam)
        assert d["class"] == "OrderedCategorical"
        assert d["n_categories"] == 4
        assert "cutpoints" not in d
        restored = _family_from_dict(d)
        assert isinstance(restored, OrderedCategorical)
        assert restored.n_categories == 4
        assert restored.cutpoints is None

    def test_ordered_categorical_roundtrip_with_cutpoints(self):
        fam = OrderedCategorical(n_categories=3)
        fam._cutpoints = np.array([-0.5, 0.5])
        d = _family_to_dict(fam)
        assert d["cutpoints"] == [-0.5, 0.5]
        restored = _family_from_dict(d)
        assert isinstance(restored, OrderedCategorical)
        np.testing.assert_allclose(restored.cutpoints, [-0.5, 0.5])

    def test_multinomial_roundtrip_no_params(self):
        fam = Multinomial(n_categories=3)
        d = _family_to_dict(fam)
        assert d["class"] == "Multinomial"
        assert d["n_categories"] == 3
        assert "alphas" not in d
        assert "betas" not in d
        restored = _family_from_dict(d)
        assert isinstance(restored, Multinomial)
        assert restored.n_categories == 3
        assert restored.category_intercepts is None
        assert restored.category_loadings is None

    def test_multinomial_roundtrip_with_params(self):
        fam = Multinomial(n_categories=3)
        fam._alphas = np.array([0.1, 0.2])
        fam._betas = np.array([1.0, 1.1])
        d = _family_to_dict(fam)
        assert d["alphas"] == [0.1, 0.2]
        assert d["betas"] == [1.0, 1.1]
        restored = _family_from_dict(d)
        np.testing.assert_allclose(restored.category_intercepts, [0.1, 0.2])
        np.testing.assert_allclose(restored.category_loadings, [1.0, 1.1])

    def test_cox_ph_roundtrip(self):
        fam = CoxPH(status="died", ties="efron")
        d = _family_to_dict(fam)
        assert d["class"] == "CoxPH"
        assert d["status"] == "died"
        assert d["ties"] == "efron"
        restored = _family_from_dict(d)
        assert isinstance(restored, CoxPH)
        assert restored.ties == "efron"

    def test_cox_ph_from_dict_defaults(self):
        restored = _family_from_dict({"class": "CoxPH"})
        assert isinstance(restored, CoxPH)
        assert restored.ties == "breslow"


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

    def test_by_variable_preserved(self):
        from whittaker.formula.parser import parse

        formula = parse("y ~ s(x, k=6, by='group')")
        d = _formula_to_dict(formula)
        smooth_dict = [t for t in d["terms"] if t["type"] == "smooth"][0]
        assert smooth_dict["by"] == "group"

        restored = _formula_from_dict(d)
        smooth = [t for t in restored.terms if isinstance(t, SmoothTerm)][0]
        assert smooth.by == "group"

    def test_extra_with_plain_ndarray(self):
        term = SmoothTerm(
            variables=("x",),
            extra={"weights": np.array([1.0, 2.0, 3.0])},
        )
        d = _term_to_dict(term)
        assert d["extra"]["weights"] == [1.0, 2.0, 3.0]
        restored = _term_from_dict(d)
        assert restored.extra["weights"] == [1.0, 2.0, 3.0]

    def test_extra_with_xt_ndarray_dict(self):
        boundary = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        knots = np.array([[0.25, 0.25], [0.75, 0.75]])
        term = SmoothTerm(
            variables=("x", "y2"),
            smooth_type="s",
            bs="so",
            extra={"xt": {"boundary": boundary, "knots": knots, "label": "region"}},
        )
        d = _term_to_dict(term)
        xt_dict = d["extra"]["xt"]
        assert xt_dict["boundary"] == boundary.tolist()
        assert xt_dict["knots"] == knots.tolist()
        assert xt_dict["label"] == "region"

        restored = _term_from_dict(d)
        np.testing.assert_allclose(restored.extra["xt"]["boundary"], boundary)
        np.testing.assert_allclose(restored.extra["xt"]["knots"], knots)
        assert restored.extra["xt"]["label"] == "region"

    def test_tensor_term_roundtrip(self):
        term = SmoothTerm(
            variables=("x1", "x2"),
            smooth_type="te",
            bs="tp",
            extra={"k": [5, 4]},
        )
        formula = Formula(response="y", terms=[term], intercept=True)
        d = _formula_to_dict(formula)
        restored = _formula_from_dict(d)
        smooth = restored.terms[0]
        assert smooth.smooth_type == "te"
        assert smooth.variables == ("x1", "x2")
        assert smooth.extra["k"] == [5, 4]

    def test_offset_term_roundtrip(self):
        from whittaker.formula.terms import OffsetTerm

        formula = Formula(
            response="y",
            terms=[OffsetTerm(expression="log(exposure)")],
            intercept=True,
        )
        d = _formula_to_dict(formula)
        restored = _formula_from_dict(d)
        assert restored.terms[0].expression == "log(exposure)"

    def test_interaction_term_roundtrip(self):
        from whittaker.formula.terms import InteractionTerm

        formula = Formula(
            response="y",
            terms=[InteractionTerm(left="x1", right="x2", full=False)],
            intercept=True,
        )
        d = _formula_to_dict(formula)
        restored = _formula_from_dict(d)
        term = restored.terms[0]
        assert term.left == "x1"
        assert term.right == "x2"
        assert term.full is False

    def test_unknown_term_type_raises(self):
        with pytest.raises(ValueError, match="Unknown term type"):
            _term_from_dict({"type": "bogus"})

    def test_unknown_term_object_raises(self):
        with pytest.raises(TypeError, match="Unknown term type"):
            _term_to_dict(object())


class TestBasisStateSerialization:
    def test_unknown_basis_class_raises(self):
        with pytest.raises(ValueError, match="Unknown smooth basis"):
            _basis_from_state({"class": "NotARealBasis"})

    def test_tensor_marginals_state_roundtrip(self):
        from whittaker.smooths.tensor import TensorProductBasis
        from whittaker.smooths.tprs import TPRS

        rng = np.random.default_rng(23)
        x1 = rng.uniform(0, 1, 50)
        x2 = rng.uniform(0, 1, 50)
        x = np.column_stack([x1, x2])

        basis = TensorProductBasis(marginals=[TPRS(k=6), TPRS(k=5)])
        basis.fit(x)

        state = _basis_state(basis)
        assert state["class"] == "TensorProductBasis"
        assert isinstance(state["_marginals"], list)
        assert all(m["class"] == "TPRS" for m in state["_marginals"])

        restored = _basis_from_state(state)
        assert isinstance(restored, TensorProductBasis)
        assert len(restored._marginals) == 2
        for m in restored._marginals:
            assert isinstance(m, TPRS)

        B_orig = basis.basis_matrix(x)
        B_restored = restored.basis_matrix(x)
        np.testing.assert_allclose(B_restored, B_orig)

    def test_plain_list_attribute_state(self):
        from whittaker.smooths.tensor import TensorInteractionBasis
        from whittaker.smooths.tprs import TPRS

        rng = np.random.default_rng(23)
        x1 = rng.uniform(0, 1, 50)
        x2 = rng.uniform(0, 1, 50)
        x = np.column_stack([x1, x2])

        basis = TensorInteractionBasis(marginals=[TPRS(k=6), TPRS(k=5)])
        basis.fit(x)

        state = _basis_state(basis)
        # _range_dims is a plain list of ints (not SmoothBasis instances), exercising the
        # "else" branch that stores non-basis lists as-is.
        assert isinstance(state["_range_dims"], list)
        assert all(isinstance(v, int) for v in state["_range_dims"])


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

    def test_weights_roundtrip(self, sin_data, tmp_path):
        rng = np.random.default_rng(23)
        weights = rng.uniform(0.5, 2.0, len(sin_data["x"]))

        model = GAM("y ~ s(x)")
        model.fit(sin_data, weights=weights)

        path = tmp_path / "weighted.npz"
        save_gam(model, path)
        loaded = load_gam(path)

        assert model._fit_result.prior_weights is not None
        np.testing.assert_allclose(
            loaded._fit_result.prior_weights, model._fit_result.prior_weights
        )
        pred = loaded.predict(sin_data)
        np.testing.assert_allclose(pred.values, model.predict(sin_data).values, atol=1e-10)

    def test_offset_roundtrip(self, tmp_path):
        rng = np.random.default_rng(23)
        n = 150
        x = np.linspace(0, 3, n)
        log_exposure = rng.uniform(0, 1, n)
        y = rng.poisson(np.exp(0.5 * x + log_exposure)).astype(float)
        data = {"x": x, "y": y, "log_exposure": log_exposure}

        model = GAM("y ~ s(x) + offset(log_exposure)", family=Poisson())
        model.fit(data)

        path = tmp_path / "offset.npz"
        save_gam(model, path)
        loaded = load_gam(path)

        assert loaded._model_matrix.offset is not None
        np.testing.assert_allclose(loaded._model_matrix.offset, model._model_matrix.offset)
        pred = loaded.predict(data)
        np.testing.assert_allclose(pred.values, model.predict(data).values, atol=1e-10)

    def test_pseudo_data_roundtrip_non_gaussian(self, tmp_path):
        rng = np.random.default_rng(23)
        x = np.linspace(0, 3, 150)
        y = rng.poisson(np.exp(0.5 * x)).astype(float)
        data = {"x": x, "y": y}

        model = GAM("y ~ s(x)", family=Poisson())
        model.fit(data)

        path = tmp_path / "pseudo.npz"
        save_gam(model, path)
        loaded = load_gam(path)

        assert model._fit_result.pseudo_data is not None
        np.testing.assert_allclose(loaded._fit_result.pseudo_data, model._fit_result.pseudo_data)

    def test_by_variable_roundtrip(self, tmp_path):
        rng = np.random.default_rng(23)
        n = 200
        x = rng.uniform(0, 1, n)
        group = rng.choice(["a", "b"], n)
        y = np.where(group == "a", np.sin(2 * np.pi * x), np.cos(2 * np.pi * x))
        y = y + rng.normal(0, 0.1, n)
        data = {"x": x, "y": y, "group": group}

        model = GAM("y ~ s(x, k=8, by='group')")
        model.fit(data)

        path = tmp_path / "by.npz"
        save_gam(model, path)
        loaded = load_gam(path)

        si = loaded._model_matrix.smooths[0]
        assert si.by_var == "group"
        assert si.by_level is not None

        pred = loaded.predict(data)
        np.testing.assert_allclose(pred.values, model.predict(data).values, atol=1e-10)

    def test_random_effect_basis_roundtrip(self, tmp_path):
        rng = np.random.default_rng(23)
        n_groups = 8
        n_per = 20
        group = np.repeat(np.arange(n_groups).astype(str), n_per)
        group_effects = rng.normal(0, 1.5, n_groups)
        y = group_effects[np.repeat(np.arange(n_groups), n_per)] + rng.normal(
            0, 0.5, n_groups * n_per
        )
        data = {"y": y, "group": group}

        model = GAM("y ~ s(group, bs='re')")
        model.fit(data)

        path = tmp_path / "re.npz"
        save_gam(model, path)
        loaded = load_gam(path)

        pred = loaded.predict(data)
        np.testing.assert_allclose(pred.values, model.predict(data).values, atol=1e-10)

    def test_tweedie_family_roundtrip(self, tmp_path):
        from whittaker.families.tweedie import Tweedie

        rng = np.random.default_rng(23)
        n = 150
        x = np.linspace(0, 2 * np.pi, n)
        mu = np.exp(1.0 + 0.3 * np.sin(x))
        y = rng.gamma(2.0, mu / 2.0)
        y[rng.uniform(size=n) < 0.2] = 0.0
        data = {"x": x, "y": y}

        model = GAM("y ~ s(x)", family=Tweedie(p=1.5))
        model.fit(data)

        path = tmp_path / "tweedie.npz"
        save_gam(model, path)
        loaded = load_gam(path)

        assert isinstance(loaded.family, Tweedie)
        assert loaded.family._p == pytest.approx(1.5)
        pred = loaded.predict(data)
        np.testing.assert_allclose(pred.values, model.predict(data).values, atol=1e-10)

    def test_factor_smooth_basis_roundtrip(self, tmp_path):
        rng = np.random.default_rng(23)
        n_subjects = 4
        n_per = 30
        n = n_subjects * n_per
        subject = np.repeat(np.arange(n_subjects).astype(str), n_per)
        x = np.tile(np.linspace(0, 2 * np.pi, n_per), n_subjects)
        subject_intercepts = rng.normal(0, 1.0, n_subjects)
        subj_idx = np.repeat(np.arange(n_subjects), n_per)
        y = np.sin(x) + subject_intercepts[subj_idx] + rng.normal(0, 0.3, n)
        data = {"x": x, "y": y, "subject": subject}

        model = GAM("y ~ s(x, subject, bs='fs', k=6)")
        model.fit(data)

        path = tmp_path / "fs.npz"
        save_gam(model, path)
        loaded = load_gam(path)

        pred = loaded.predict(data)
        np.testing.assert_allclose(pred.values, model.predict(data).values, atol=1e-10)

    def test_cox_ph_family_roundtrip(self, tmp_path):
        rng = np.random.default_rng(23)
        n = 200
        x = rng.normal(0, 1, n)
        eta_true = 0.8 * x
        time = rng.exponential(1.0 / np.exp(eta_true))
        censor_time = rng.exponential(5.0, n)
        event = (time <= censor_time).astype(float)
        time = np.minimum(time, censor_time)
        data = {"y": time, "x": x, "event": event}

        model = GAM("y ~ s(x)", family=CoxPH(status="event"))
        model.fit(data)

        path = tmp_path / "cox.npz"
        save_gam(model, path)
        loaded = load_gam(path)

        assert isinstance(loaded.family, CoxPH)
        assert loaded.family.ties == "breslow"


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

    def test_non_gam_raises(self):
        with pytest.raises(TypeError, match="GAM"):
            to_mgcv_dict("not a model")

    def test_by_variable_export(self):
        rng = np.random.default_rng(23)
        n = 200
        x = rng.uniform(0, 1, n)
        group = rng.choice(["a", "b"], n)
        y = np.where(group == "a", np.sin(2 * np.pi * x), np.cos(2 * np.pi * x))
        y = y + rng.normal(0, 0.1, n)
        model = GAM("y ~ s(x, k=8, by='group')")
        model.fit({"x": x, "y": y, "group": group})

        d = to_mgcv_dict(model)
        s = d["smooth"][0]
        assert s["by"] == "group"
        assert "by.level" in s

    def test_random_effect_export_levels(self):
        rng = np.random.default_rng(23)
        n_groups = 6
        n_per = 15
        group = np.repeat(np.arange(n_groups).astype(str), n_per)
        y = rng.normal(0, 1, n_groups * n_per)
        model = GAM("y ~ s(group, bs='re')")
        model.fit({"y": y, "group": group})

        d = to_mgcv_dict(model)
        s = d["smooth"][0]
        assert s["bs"] == "re"
        assert "levels" in s

    def test_cr_basis_export_knots(self, sin_data):
        model = GAM("y ~ s(x, bs='cr')")
        model.fit(sin_data)
        d = to_mgcv_dict(model)
        s = d["smooth"][0]
        assert s["bs"] == "cr"
        assert "knots" in s

    def test_tensor_smooth_export(self):
        rng = np.random.default_rng(23)
        n = 150
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        y = np.sin(2 * np.pi * x1) * x2 + rng.normal(0, 0.1, n)
        model = GAM("y ~ te(x1, x2, k=4)")
        model.fit({"x1": x1, "x2": x2, "y": y})

        d = to_mgcv_dict(model)
        assert len(d["smooth"]) == 1
        s = d["smooth"][0]
        assert set(s["term"]) == {"x1", "x2"}
        assert len(s["S"]) >= 1

    def test_tweedie_family_power_exported(self):
        rng = np.random.default_rng(23)
        n = 150
        x = np.linspace(0, 2 * np.pi, n)
        mu = np.exp(1.0 + 0.3 * np.sin(x))
        y = rng.gamma(2.0, mu / 2.0)
        y[rng.uniform(size=n) < 0.2] = 0.0
        model = GAM("y ~ s(x)", family=Tweedie(p=1.5))
        model.fit({"x": x, "y": y})

        d = to_mgcv_dict(model)
        assert d["family"]["power"] == pytest.approx(1.5)

    def test_negative_binomial_theta_exported(self):
        rng = np.random.default_rng(23)
        n = 150
        x = np.linspace(0, 3, n)
        y = rng.poisson(np.exp(0.5 * x)).astype(float)
        model = GAM("y ~ s(x)", family=NegativeBinomial())
        model.fit({"x": x, "y": y})

        d = to_mgcv_dict(model)
        assert "theta" in d["family"]


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

    def test_family_info_as_string(self):
        d = {
            "coefficients": [0.0, 0.1, 0.2],
            "sp": [1.0],
            "formula": "y ~ s(x)",
            "family": "poisson",
            "smooth": [{"term": ["x"], "bs": "tp", "df": 3}],
            "intercept": True,
        }
        imported = from_mgcv_dict(d)
        from whittaker.families.poisson import Poisson

        assert isinstance(imported.family, Poisson)

    def test_tweedie_power_mapped(self):
        d = {
            "coefficients": [0.0, 0.1],
            "sp": [1.0],
            "formula": "y ~ s(x)",
            "family": {"family": "Tweedie", "power": 1.4},
            "smooth": [{"term": ["x"], "bs": "tp", "df": 2}],
            "intercept": True,
        }
        imported = from_mgcv_dict(d)
        assert isinstance(imported.family, Tweedie)
        assert imported.family._p == pytest.approx(1.4)

    def test_negative_binomial_theta_mapped(self):
        d = {
            "coefficients": [0.0, 0.1],
            "sp": [1.0],
            "formula": "y ~ s(x)",
            "family": {"family": "nb", "theta": 2.5},
            "smooth": [{"term": ["x"], "bs": "tp", "df": 2}],
            "intercept": True,
        }
        imported = from_mgcv_dict(d)
        assert isinstance(imported.family, NegativeBinomial)
        assert imported.family.theta == pytest.approx(2.5)

    def test_cox_ph_family_mapped(self):
        d = {
            "coefficients": [0.1],
            "sp": [],
            "formula": "y ~ x",
            "family": {"family": "cox.ph"},
            "smooth": [],
            "intercept": False,
        }
        imported = from_mgcv_dict(d)
        assert isinstance(imported.family, CoxPH)

    def test_formula_without_tilde_builds_smooth_terms(self):
        d = {
            "coefficients": [0.0, 0.1, 0.2, 0.3],
            "sp": [1.0],
            "formula": "",
            "family": {"family": "gaussian"},
            "smooth": [{"term": ["x1", "x2"], "bs": "tp", "df": 3}],
            "intercept": True,
        }
        imported = from_mgcv_dict(d)
        assert not imported.is_fitted
        assert imported._formula.response == "y"
        smooth_term = imported._formula.terms[0]
        assert smooth_term.variables == ("x1", "x2")
        assert smooth_term.bs == "tp"
        assert smooth_term.k == 3

    def test_import_with_data_and_offset_family(self):
        rng = np.random.default_rng(23)
        n = 150
        x = np.linspace(0, 3, n)
        y = rng.poisson(np.exp(0.3 * x)).astype(float)
        data = {"x": x, "y": y}

        model = GAM("y ~ s(x)", family=Poisson())
        model.fit(data)
        mgcv_d = to_mgcv_dict(model)

        imported = from_mgcv_dict(mgcv_d, data=data)
        assert imported.is_fitted
        assert isinstance(imported.family, Poisson)
        pred = imported.predict(data)
        assert np.all(np.isfinite(pred.values))

    def test_import_with_data_and_offset_term(self):
        rng = np.random.default_rng(23)
        n = 150
        x = np.linspace(0, 3, n)
        log_exposure = rng.uniform(0, 1, n)
        y = rng.poisson(np.exp(0.3 * x + log_exposure)).astype(float)
        data = {"x": x, "y": y, "log_exposure": log_exposure}

        model = GAM("y ~ s(x) + offset(log_exposure)", family=Poisson())
        model.fit(data)
        mgcv_d = to_mgcv_dict(model)
        assert "offset(log_exposure)" in mgcv_d["formula"]

        imported = from_mgcv_dict(mgcv_d, data=data)
        assert imported.is_fitted
        assert imported._model_matrix.offset is not None
        pred = imported.predict(data)
        assert np.all(np.isfinite(pred.values))
