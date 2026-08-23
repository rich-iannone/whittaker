"""Model import/export for Whittaker GAMs.

This module provides two independent ways to move a fitted `~whittaker.gam.GAM` in and out of
process:

1. `save_gam` / `load_gam`: Serialize a fitted model to a single `.npz` archive and reconstruct it
   later without re-fitting. This is the round-trip path within Python — everything the model needs
   for `predict()`, `summary()`, and further inference is captured, including the fitted smooth
   basis state, so loading is exact and cheap (no basis refitting or optimization is repeated).
2. `to_mgcv_dict` / `from_mgcv_dict`: Convert to and from a plain dictionary whose keys and
   structure mirror an `mgcv::gam` object in R. This is the cross-language path — export a
   Python-fitted model for inspection or comparison in R, or import a model fitted with
   `mgcv::gam()` in R for use in Python. Because `mgcv` and `whittaker` do not expose identical
   internals, this conversion is necessarily lossy in places; see the `Notes` sections of
   `to_mgcv_dict` and `from_mgcv_dict` for specifics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from whittaker.families.base import Family
from whittaker.fitting.pirls import FitResult
from whittaker.formula.terms import (
    Formula,
    InteractionTerm,
    LinearTerm,
    OffsetTerm,
    SmoothTerm,
)
from whittaker.model_matrix import ModelMatrix, SmoothInfo
from whittaker.smooths.base import SmoothBasis


def _family_to_dict(family: Family) -> dict[str, Any]:
    from whittaker.families.beta import Beta
    from whittaker.families.binomial import Binomial
    from whittaker.families.cox_ph import CoxPH
    from whittaker.families.gamma import Gamma
    from whittaker.families.gaussian import Gaussian
    from whittaker.families.inverse_gaussian import InverseGaussian
    from whittaker.families.multinomial import Multinomial
    from whittaker.families.negative_binomial import NegativeBinomial
    from whittaker.families.ordered_categorical import OrderedCategorical
    from whittaker.families.poisson import Poisson
    from whittaker.families.quantile import QuantileFamily
    from whittaker.families.tweedie import Tweedie
    from whittaker.families.tweedie_estimated import TweedieEstimated

    d: dict[str, Any] = {"class": type(family).__name__}

    if isinstance(family, TweedieEstimated):
        d["p"] = float(family._p)
        d["p_range"] = list(family._p_range)
        d["n_grid"] = family._n_grid
    elif isinstance(family, Tweedie):
        d["p"] = float(family._p)
    elif isinstance(family, NegativeBinomial):
        d["theta"] = float(family.theta)
    elif isinstance(family, QuantileFamily):
        d["tau"] = float(family._tau)
    elif isinstance(family, OrderedCategorical):
        d["n_categories"] = family._K
        if family._cutpoints is not None:
            d["cutpoints"] = family._cutpoints.tolist()
    elif isinstance(family, Multinomial):
        d["n_categories"] = family._K
        if family._alphas is not None:
            d["alphas"] = family._alphas.tolist()
        if family._betas is not None:
            d["betas"] = family._betas.tolist()
    elif isinstance(family, CoxPH):
        d["status"] = family._status_col
        d["ties"] = family._ties
    elif isinstance(family, (Gaussian, Poisson, Binomial, Gamma, Beta, InverseGaussian)):
        pass

    return d


def _family_from_dict(d: dict[str, Any]) -> Family:
    from whittaker.families.beta import Beta
    from whittaker.families.binomial import Binomial
    from whittaker.families.cox_ph import CoxPH
    from whittaker.families.gamma import Gamma
    from whittaker.families.gaussian import Gaussian
    from whittaker.families.inverse_gaussian import InverseGaussian
    from whittaker.families.multinomial import Multinomial
    from whittaker.families.negative_binomial import NegativeBinomial
    from whittaker.families.ordered_categorical import OrderedCategorical
    from whittaker.families.poisson import Poisson
    from whittaker.families.quantile import QuantileFamily
    from whittaker.families.tweedie import Tweedie
    from whittaker.families.tweedie_estimated import TweedieEstimated

    cls_name = d["class"]
    registry: dict[str, type] = {
        "Gaussian": Gaussian,
        "Poisson": Poisson,
        "Binomial": Binomial,
        "Gamma": Gamma,
        "Beta": Beta,
        "InverseGaussian": InverseGaussian,
        "Tweedie": Tweedie,
        "TweedieEstimated": TweedieEstimated,
        "NegativeBinomial": NegativeBinomial,
        "QuantileFamily": QuantileFamily,
        "OrderedCategorical": OrderedCategorical,
        "Multinomial": Multinomial,
        "CoxPH": CoxPH,
    }

    if cls_name not in registry:
        raise ValueError(f"Unknown family class: {cls_name!r}")

    if cls_name == "TweedieEstimated":
        fam = TweedieEstimated(p_range=tuple(d["p_range"]), n_grid=d["n_grid"])
        fam._set_p(d["p"])
        return fam
    elif cls_name == "Tweedie":
        return Tweedie(p=d["p"])
    elif cls_name == "NegativeBinomial":
        fam = NegativeBinomial()
        fam.theta = d["theta"]
        return fam
    elif cls_name == "QuantileFamily":
        return QuantileFamily(tau=d["tau"])
    elif cls_name == "OrderedCategorical":
        fam = OrderedCategorical(n_categories=d["n_categories"])
        if "cutpoints" in d:
            fam._cutpoints = np.array(d["cutpoints"])
        return fam
    elif cls_name == "Multinomial":
        fam = Multinomial(n_categories=d["n_categories"])
        if "alphas" in d:
            fam._alphas = np.array(d["alphas"])
        if "betas" in d:
            fam._betas = np.array(d["betas"])
        return fam
    elif cls_name == "CoxPH":
        return CoxPH(status=d.get("status", "event"), ties=d.get("ties", "breslow"))
    else:
        return registry[cls_name]()


def _term_to_dict(term: Any) -> dict[str, Any]:
    if isinstance(term, SmoothTerm):
        d: dict[str, Any] = {
            "type": "smooth",
            "variables": list(term.variables),
            "smooth_type": term.smooth_type,
            "bs": term.bs,
            "k": term.k,
        }
        if term.by is not None:
            d["by"] = term.by
        extra_serializable = {}
        for k, v in term.extra.items():
            if k == "xt" and isinstance(v, dict):
                xt_clean = {}
                for xk, xv in v.items():
                    if isinstance(xv, np.ndarray):
                        xt_clean[xk] = xv.tolist()
                    else:
                        xt_clean[xk] = xv
                extra_serializable[k] = xt_clean
            elif isinstance(v, np.ndarray):
                extra_serializable[k] = v.tolist()
            else:
                extra_serializable[k] = v
        if extra_serializable:
            d["extra"] = extra_serializable
        return d
    elif isinstance(term, LinearTerm):
        return {"type": "linear", "variable": term.variable}
    elif isinstance(term, InteractionTerm):
        return {"type": "interaction", "left": term.left, "right": term.right, "full": term.full}
    elif isinstance(term, OffsetTerm):
        return {"type": "offset", "expression": term.expression}
    else:
        raise TypeError(f"Unknown term type: {type(term).__name__}")


def _term_from_dict(d: dict[str, Any]) -> Any:
    ttype = d["type"]
    if ttype == "smooth":
        extra = d.get("extra", {})
        if "xt" in extra and isinstance(extra["xt"], dict):
            xt = extra["xt"]
            for xk, xv in xt.items():
                if isinstance(xv, list) and xv and isinstance(xv[0], list):
                    xt[xk] = np.array(xv)
        return SmoothTerm(
            variables=tuple(d["variables"]),
            smooth_type=d.get("smooth_type", "s"),
            bs=d.get("bs", "tp"),
            k=d.get("k", -1),
            by=d.get("by"),
            extra=extra,
        )
    elif ttype == "linear":
        return LinearTerm(variable=d["variable"])
    elif ttype == "interaction":
        return InteractionTerm(left=d["left"], right=d["right"], full=d.get("full", False))
    elif ttype == "offset":
        return OffsetTerm(expression=d["expression"])
    else:
        raise ValueError(f"Unknown term type: {ttype!r}")


def _formula_to_dict(formula: Formula) -> dict[str, Any]:
    return {
        "response": formula.response,
        "intercept": formula.intercept,
        "terms": [_term_to_dict(t) for t in formula.terms],
    }


def _formula_from_dict(d: dict[str, Any]) -> Formula:
    terms = [_term_from_dict(t) for t in d["terms"]]
    return Formula(response=d["response"], terms=terms, intercept=d.get("intercept", True))


def _basis_state(basis: SmoothBasis) -> dict[str, Any]:
    state: dict[str, Any] = {"class": type(basis).__name__}
    for attr in sorted(vars(basis)):
        val = getattr(basis, attr)
        if isinstance(val, (np.ndarray, int, float, str, bool, type(None))):
            state[attr] = val
        elif isinstance(val, dict):
            state[attr] = {str(k): v for k, v in val.items()}
        elif isinstance(val, list):
            if val and isinstance(val[0], SmoothBasis):
                state[attr] = [_basis_state(b) for b in val]
            else:
                state[attr] = val
        elif isinstance(val, SmoothBasis):
            state[attr] = _basis_state(val)
    return state


def _basis_from_state(state: dict[str, Any]) -> SmoothBasis:
    from whittaker.smooths.adaptive import AdaptiveTPRS
    from whittaker.smooths.cubic import CRS
    from whittaker.smooths.cyclic import CyclicCRS, CyclicPSpline
    from whittaker.smooths.duchon import DuchonSpline
    from whittaker.smooths.factor_smooth import FactorSmoothBasis
    from whittaker.smooths.gp import GaussianProcess
    from whittaker.smooths.monotone import ConvexPSpline, MonotonePSpline
    from whittaker.smooths.mrf import MRFBasis
    from whittaker.smooths.pspline import PSpline
    from whittaker.smooths.random import RandomEffectBasis
    from whittaker.smooths.shrinkage import ShrinkageCRS, ShrinkageTPRS
    from whittaker.smooths.soap_film import SoapFilm
    from whittaker.smooths.tensor import (
        TensorInteractionBasis,
        TensorProductBasis,
        TensorProductBasisT2,
    )
    from whittaker.smooths.tprs import TPRS

    registry: dict[str, type] = {
        "TPRS": TPRS,
        "CRS": CRS,
        "PSpline": PSpline,
        "CyclicCRS": CyclicCRS,
        "CyclicPSpline": CyclicPSpline,
        "ShrinkageTPRS": ShrinkageTPRS,
        "ShrinkageCRS": ShrinkageCRS,
        "RandomEffectBasis": RandomEffectBasis,
        "MRFBasis": MRFBasis,
        "AdaptiveTPRS": AdaptiveTPRS,
        "GaussianProcess": GaussianProcess,
        "DuchonSpline": DuchonSpline,
        "SoapFilm": SoapFilm,
        "MonotonePSpline": MonotonePSpline,
        "ConvexPSpline": ConvexPSpline,
        "FactorSmoothBasis": FactorSmoothBasis,
        "TensorProductBasis": TensorProductBasis,
        "TensorInteractionBasis": TensorInteractionBasis,
        "TensorProductBasisT2": TensorProductBasisT2,
    }

    cls_name = state["class"]
    if cls_name not in registry:
        raise ValueError(f"Unknown smooth basis class: {cls_name!r}")

    basis = object.__new__(registry[cls_name])
    for attr, val in state.items():
        if attr == "class":
            continue
        if isinstance(val, np.ndarray):
            setattr(basis, attr, val)
        elif isinstance(val, dict) and "class" in val:
            setattr(basis, attr, _basis_from_state(val))
        elif isinstance(val, list) and val and isinstance(val[0], dict) and "class" in val[0]:
            setattr(basis, attr, [_basis_from_state(b) for b in val])
        elif isinstance(val, dict):
            setattr(basis, attr, val)
        else:
            setattr(basis, attr, val)
    return basis


def save_gam(model: Any, path: str | Path) -> None:
    r"""Save a fitted GAM to a `.npz` archive.

    Serializes everything needed to reconstruct a fitted `~whittaker.gam.GAM` for prediction and
    inference without recomputing basis fits or re-running P-IRLS. Use this to persist a model
    between sessions, ship a fitted model to another machine, or cache an expensive fit. The
    archive is a standard `numpy` `.npz` file (produced with `numpy.savez_compressed`) and can, in
    principle, be inspected with `numpy.load` alone, though `load_gam` is the supported way to
    read it back.

    Internally the archive stores two kinds of data under one file:

    - A single JSON-encoded metadata blob under the key `"__metadata__"`, containing the formula
      (response, intercept flag, and each term), the family (class name and any extra parameters
      such as a Tweedie power or negative-binomial dispersion), fit statistics (smoothing
      parameters, scale, GCV score, EDF, deviance, iteration count, convergence flag, AIC/BIC,
      etc.), model-matrix metadata (column names, intercept/parametric counts, offset expressions),
      and, per smooth term, its formula term, coefficient column range, null-space dimension,
      penalty indices, and basis state (attribute values of the fitted
      `~whittaker.smooths.base.SmoothBasis`).
    - Raw `numpy` arrays stored alongside the metadata: `coefficients`, `linear_predictor`,
      `fitted_values`, `residuals`, the training design matrix `X`, the `response` vector, and,
      when present, `weights`, `prior_weights`, `pseudo_data`, and `offset`. Each smooth's penalty
      matrix is stored as `penalty_{i}` (one array per penalty block, in the order the smooths
      contribute penalties). Any array-valued attribute of a smooth's fitted basis (e.g. knot
      locations, training covariate values) is stored under a key of the form
      `smooth_{idx}_basis_{attr}` (or `smooth_{idx}_basis_{attr}_{subkey}` for nested dict
      attributes), with a `{"__ndarray__": key}` pointer left in the metadata blob so `load_gam` can
      find it.

    Note that the archive has no explicit format-version field: there is currently no mechanism to
    detect or migrate across schema changes, so a saved archive is only guaranteed to load correctly
    with a `whittaker` version compatible with the one that wrote it.

    Parameters
    ----------
    model : GAM
        A fitted `~whittaker.gam.GAM` instance, i.e. one on which `fit()` has already been called.
    path : str or pathlib.Path
        Output file path. `numpy.savez_compressed` appends a `.npz` extension automatically if the
        given path does not already end in one.

    Raises
    ------
    TypeError
        If `model` is not a `GAM` instance.
    RuntimeError
        If `model` has not been fitted (`model.is_fitted` is `False`).

    Examples
    --------
    ```{python}
    import numpy as np
    import whittaker as wt
    from whittaker.io import save_gam, load_gam

    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 1, 200))
    y = np.sin(2 * np.pi * x) + rng.normal(scale=0.2, size=200)

    model = wt.GAM("y ~ s(x)").fit({"x": x, "y": y})

    save_gam(model, "gam_model.npz")
    reloaded = load_gam("gam_model.npz")

    new_x = np.linspace(0, 1, 5)
    np.allclose(
        model.predict({"x": new_x}).values,
        reloaded.predict({"x": new_x}).values,
    )
    ```
    """
    from whittaker.gam import GAM

    if not isinstance(model, GAM):
        raise TypeError(f"Expected a GAM instance, got {type(model).__name__}.")
    if not model.is_fitted:
        raise RuntimeError("Cannot save an unfitted model. Call fit() first.")
    from whittaker.fitting.mcmc import MCMCResult
    from whittaker.fitting.vi import VIResult

    if isinstance(model._fit_result, (VIResult, MCMCResult)):
        raise NotImplementedError(
            f"save_gam does not support {model._fit_result.method} fits."
        )

    path = Path(path)
    fr = model._fit_result
    mm = model._model_matrix

    metadata = {
        "formula": _formula_to_dict(model._formula),
        "family": _family_to_dict(model._family),
        "fit": {
            "smoothing_params": fr.smoothing_params,
            "scale": fr.scale,
            "gcv_score": fr.gcv_score,
            "edf": fr.edf,
            "edf_total": fr.edf_total,
            "deviance": fr.deviance,
            "n_iter": fr.n_iter,
            "converged": fr.converged,
            "hat_matrix_trace": fr.hat_matrix_trace,
            "null_deviance": fr.null_deviance,
            "aic": fr.aic,
            "bic": fr.bic,
            "method": fr.method,
        },
        "model_matrix": {
            "has_intercept": mm.has_intercept,
            "n_parametric": mm.n_parametric,
            "column_names": mm.column_names,
            "offset_expressions": mm.offset_expressions,
        },
        "smooths": [],
    }

    arrays: dict[str, NDArray] = {
        "coefficients": fr.coefficients,
        "linear_predictor": fr.linear_predictor,
        "fitted_values": fr.fitted_values,
        "residuals": fr.residuals,
        "X": mm.X,
        "response": mm.response,
    }

    if fr.weights is not None:
        arrays["weights"] = fr.weights
    if fr.prior_weights is not None:
        arrays["prior_weights"] = fr.prior_weights
    if fr.pseudo_data is not None:
        arrays["pseudo_data"] = fr.pseudo_data
    if mm.offset is not None:
        arrays["offset"] = mm.offset

    for i, pen in enumerate(mm.penalties):
        arrays[f"penalty_{i}"] = pen

    basis_arrays: dict[str, NDArray] = {}
    smooth_meta_list = []
    for si_idx, si in enumerate(mm.smooths):
        si_dict: dict[str, Any] = {
            "term": _term_to_dict(si.term),
            "col_start": si.col_start,
            "col_end": si.col_end,
            "null_space_dim": si.null_space_dim,
            "penalty_indices": si.penalty_indices,
            "by_var": si.by_var,
            "by_level": si.by_level,
        }

        basis_state = _basis_state(si.basis)
        basis_json: dict[str, Any] = {}
        for k, v in basis_state.items():
            if isinstance(v, np.ndarray):
                arr_key = f"smooth_{si_idx}_basis_{k}"
                basis_arrays[arr_key] = v
                basis_json[k] = {"__ndarray__": arr_key}
            elif isinstance(v, dict) and any(isinstance(vv, np.ndarray) for vv in v.values()):
                sub = {}
                for sk, sv in v.items():
                    if isinstance(sv, np.ndarray):
                        arr_key = f"smooth_{si_idx}_basis_{k}_{sk}"
                        basis_arrays[arr_key] = sv
                        sub[sk] = {"__ndarray__": arr_key}
                    else:
                        sub[sk] = sv
                basis_json[k] = sub
            else:
                basis_json[k] = v
        si_dict["basis_state"] = basis_json
        smooth_meta_list.append(si_dict)

    metadata["smooths"] = smooth_meta_list

    arrays["__metadata__"] = np.array([json.dumps(metadata)])
    arrays.update(basis_arrays)

    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]


def load_gam(path: str | Path) -> Any:
    r"""Load a fitted GAM from a `.npz` archive created by `save_gam`.

    Reads back every piece of state that `save_gam` wrote — the formula, family, fitted
    coefficients and fit statistics, training design matrix and penalties, and each smooth's basis
    state (restored via an internal `_basis_from_state` helper that reconstructs the original
    `~whittaker.smooths.base.SmoothBasis` subclass without calling its constructor) — and assembles
    them into a fully fitted `~whittaker.gam.GAM`. The returned model behaves exactly as it did
    before saving: `predict()`, `summary()`, `plot()`, and `check()` all work immediately, with no
    re-fitting or basis refitting performed.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the `.npz` file written by `save_gam`.

    Returns
    -------
    GAM
        A fitted `~whittaker.gam.GAM` ready for prediction and inference.

    Examples
    --------
    ```{python}
    import numpy as np
    import whittaker as wt
    from whittaker.io import save_gam, load_gam

    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(0, 1, 150))
    y = np.cos(3 * x) + rng.normal(scale=0.15, size=150)

    model = wt.GAM("y ~ s(x)").fit({"x": x, "y": y})
    save_gam(model, "gam_model.npz")

    reloaded = load_gam("gam_model.npz")
    reloaded.is_fitted
    ```
    """
    from whittaker.gam import GAM

    path = Path(path)
    data = np.load(path, allow_pickle=True)

    metadata = json.loads(str(data["__metadata__"][0]))

    formula = _formula_from_dict(metadata["formula"])
    family = _family_from_dict(metadata["family"])

    fit_meta = metadata["fit"]
    fr = FitResult(
        coefficients=data["coefficients"],
        linear_predictor=data["linear_predictor"],
        fitted_values=data["fitted_values"],
        smoothing_params=fit_meta["smoothing_params"],
        scale=fit_meta["scale"],
        gcv_score=fit_meta["gcv_score"],
        edf=fit_meta["edf"],
        edf_total=fit_meta["edf_total"],
        deviance=fit_meta["deviance"],
        n_iter=fit_meta["n_iter"],
        converged=fit_meta["converged"],
        hat_matrix_trace=fit_meta["hat_matrix_trace"],
        residuals=data["residuals"],
        weights=data.get("weights", None),
        prior_weights=data.get("prior_weights", None),
        null_deviance=fit_meta.get("null_deviance"),
        aic=fit_meta.get("aic"),
        bic=fit_meta.get("bic"),
        method=fit_meta.get("method", "GCV"),
        pseudo_data=data.get("pseudo_data", None),
    )

    mm_meta = metadata["model_matrix"]
    n_penalties = sum(len(s["penalty_indices"]) for s in metadata["smooths"])
    penalties = [data[f"penalty_{i}"] for i in range(n_penalties)]

    smooths = []
    for si_dict in metadata["smooths"]:
        basis_json = si_dict["basis_state"]
        basis_state: dict[str, Any] = {}
        for k, v in basis_json.items():
            if isinstance(v, dict) and "__ndarray__" in v:
                basis_state[k] = data[v["__ndarray__"]]
            elif isinstance(v, dict) and any(
                isinstance(sv, dict) and "__ndarray__" in sv for sv in v.values()
            ):
                sub = {}
                for sk, sv in v.items():
                    if isinstance(sv, dict) and "__ndarray__" in sv:
                        sub[sk] = data[sv["__ndarray__"]]
                    else:
                        sub[sk] = sv
                basis_state[k] = sub
            else:
                basis_state[k] = v

        basis = _basis_from_state(basis_state)
        term = _term_from_dict(si_dict["term"])

        smooths.append(
            SmoothInfo(
                term=term,
                basis=basis,
                col_start=si_dict["col_start"],
                col_end=si_dict["col_end"],
                null_space_dim=si_dict["null_space_dim"],
                penalty_indices=si_dict["penalty_indices"],
                by_var=si_dict.get("by_var"),
                by_level=si_dict.get("by_level"),
            )
        )

    mm = ModelMatrix(
        X=data["X"],
        penalties=penalties,
        smooths=smooths,
        column_names=mm_meta["column_names"],
        has_intercept=mm_meta["has_intercept"],
        n_parametric=mm_meta["n_parametric"],
        offset=data.get("offset", None),
        offset_expressions=mm_meta.get("offset_expressions", []),
        response=data["response"],
    )

    gam = GAM(formula, family=family)
    gam._model_matrix = mm
    gam._fit_result = fr
    gam._fitted = True

    return gam


def to_mgcv_dict(model: Any) -> dict[str, Any]:
    """Export a fitted GAM as an mgcv-compatible dictionary.

    Builds a plain `dict` whose keys and nested structure mirror the fields of a fitted `gam`
    object from R's `mgcv` package, rather than `whittaker`'s own internal representation. Use
    this when you need to hand a Python-fitted model to R code — typically by serializing the
    result with `json.dumps` and reading it in R with `jsonlite::fromJSON`, or comparing a
    `whittaker` fit against an equivalent `mgcv::gam()` fit term by term.

    Top-level keys include `coefficients` (the fitted coefficient vector), `sp` (smoothing
    parameters), `scale` and `scale.estimated`, `gcv.ubre`, `edf` and `edf.total`, `deviance` and
    `null.deviance`, `aic`, `n` (observation count) and `p` (coefficient count), `converged`,
    `iter`, `method`, `formula` (as a string), `family` (a nested dict with the family name and
    any extra parameter such as a Tweedie power or negative-binomial `theta`), `smooth` (a list,
    one entry per smooth term), `nsdf`, and `intercept`.

    Each entry in `smooth` describes one smooth term with mgcv-style field names: `term`
    (covariate names), `bs` (the two-letter mgcv basis-type code), `label`, `first.para`/
    `last.para` (1-based coefficient column range), `null.space.dim`, `df`, optionally
    `by`/`by.level` for `by`-variable smooths, and `S` (a list of penalty matrix blocks, sliced
    from the model's full penalty matrices down to just this term's coefficient columns). The
    `bs` code is produced by mapping the `whittaker` basis class name to mgcv's naming convention,
    e.g. `TPRS` maps to `"tp"`, `ShrinkageTPRS` to `"ts"`, `CRS` to `"cr"`, `PSpline` to `"ps"`,
    `CyclicPSpline` to `"cp"`, `RandomEffectBasis` to `"re"`, and so on; a basis with no known
    mgcv counterpart falls back to its `whittaker` class name unchanged.

    Parameters
    ----------
    model : GAM
        A fitted `~whittaker.gam.GAM` instance.

    Returns
    -------
    dict
        An mgcv-compatible dictionary with keys such as `coefficients`, `sp`, `family`, `smooth`,
        `edf`, and `deviance`, suitable for JSON serialization and import into R.

    Raises
    ------
    TypeError
        If `model` is not a `GAM` instance.
    RuntimeError
        If `model` has not been fitted (`model.is_fitted` is `False`).

    Notes
    -----
    This function is intended to interoperate with the R `mgcv` package's `gam` object structure
    so that a `whittaker` fit can be inspected or compared from R. Full fidelity is not
    guaranteed: not every `mgcv` field is populated (for example, no `Vp`/`Vc` covariance matrices
    are exported), and not every `whittaker` family or basis has a direct `mgcv` equivalent, in
    which case the original class name is used as-is rather than an invented mgcv code.

    Examples
    --------
    ```{python}
    import json
    import numpy as np
    import whittaker as wt
    from whittaker.io import to_mgcv_dict

    rng = np.random.default_rng(2)
    x = np.sort(rng.uniform(0, 1, 100))
    y = x**2 + rng.normal(scale=0.1, size=100)

    model = wt.GAM("y ~ s(x)").fit({"x": x, "y": y})
    mgcv_dict = to_mgcv_dict(model)

    sorted(mgcv_dict.keys())
    ```

    ```{python}
    # The dict is JSON-serializable and can be written out for R to read.
    payload = json.dumps(mgcv_dict)
    mgcv_dict["smooth"][0]["bs"]
    ```
    """
    from whittaker.gam import GAM

    if not isinstance(model, GAM):
        raise TypeError(f"Expected a GAM instance, got {type(model).__name__}.")
    if not model.is_fitted:
        raise RuntimeError("Cannot export an unfitted model.")
    from whittaker.fitting.mcmc import MCMCResult
    from whittaker.fitting.vi import VIResult

    if isinstance(model._fit_result, (VIResult, MCMCResult)):
        raise NotImplementedError(
            f"to_mgcv_dict does not support {model._fit_result.method} fits."
        )

    fr = model._fit_result
    mm = model._model_matrix

    _bs_map = {
        "TPRS": "tp",
        "ShrinkageTPRS": "ts",
        "CRS": "cr",
        "ShrinkageCRS": "cs",
        "PSpline": "ps",
        "CyclicCRS": "cc",
        "CyclicPSpline": "cp",
        "RandomEffectBasis": "re",
        "MRFBasis": "mrf",
        "GaussianProcess": "gp",
        "DuchonSpline": "ds",
        "SoapFilm": "so",
    }

    smooth_list = []
    for si in mm.smooths:
        bs_label = _bs_map.get(type(si.basis).__name__, type(si.basis).__name__)
        s: dict[str, Any] = {
            "term": list(si.term.variables),
            "bs": bs_label,
            "label": repr(si.term),
            "first.para": si.col_start + 1,
            "last.para": si.col_end,
            "null.space.dim": si.null_space_dim,
            "df": si.col_end - si.col_start,
        }
        if si.by_var:
            s["by"] = si.by_var
        if si.by_level:
            s["by.level"] = si.by_level

        if hasattr(si.basis, "_knots"):
            s["knots"] = si.basis._knots.tolist()  # type: ignore[attr-defined]
        if hasattr(si.basis, "_x_train"):
            s["X"] = si.basis._x_train.tolist()  # type: ignore[attr-defined]
        if hasattr(si.basis, "_levels"):
            s["levels"] = si.basis._levels.tolist()  # type: ignore[attr-defined]

        for pi in si.penalty_indices:
            pen = mm.penalties[pi]
            block = pen[si.col_start : si.col_end, si.col_start : si.col_end]
            if "S" not in s:
                s["S"] = [block.tolist()]
            else:
                s["S"].append(block.tolist())

        smooth_list.append(s)

    family_name = type(model._family).__name__
    family_dict: dict[str, Any] = {"family": family_name}
    if hasattr(model._family, "_p"):
        family_dict["power"] = float(model._family._p)  # type: ignore[attr-defined]
    if hasattr(model._family, "theta"):
        family_dict["theta"] = float(model._family.theta)  # type: ignore[attr-defined]

    result: dict[str, Any] = {
        "coefficients": fr.coefficients.tolist(),
        "sp": fr.smoothing_params,
        "scale": fr.scale,
        "scale.estimated": not model._family.scale_known,
        "gcv.ubre": fr.gcv_score,
        "edf": fr.edf,
        "edf.total": fr.edf_total,
        "deviance": fr.deviance,
        "null.deviance": fr.null_deviance,
        "aic": fr.aic,
        "n": mm.n_obs,
        "p": mm.n_coefs,
        "converged": fr.converged,
        "iter": fr.n_iter,
        "method": fr.method,
        "formula": str(model._formula),
        "family": family_dict,
        "smooth": smooth_list,
        "nsdf": mm.n_parametric + (1 if mm.has_intercept else 0),
        "intercept": mm.has_intercept,
    }

    return result


def from_mgcv_dict(
    d: dict[str, Any],
    data: dict[str, NDArray] | None = None,
) -> Any:
    """Import an mgcv `gam` object exported as a dictionary.

    The inverse of `to_mgcv_dict`: reconstructs a `~whittaker.gam.GAM` from a dictionary shaped like
    an R `mgcv::gam` object, typically produced in R with `jsonlite::toJSON(gam_model)` (or an
    equivalent hand-built dict) and passed into Python after parsing the JSON. Use this to bring a
    model fitted in R into `whittaker` for further prediction, plotting, or comparison against a
    Python fit.

    There are two modes, selected by whether `data` is supplied:

    - Without `data` (the default): only the formula, family, fitted coefficients, and smoothing
      parameters are restored onto the returned `GAM`. No model matrix or smooth basis is built,
      so the result is a lightweight container for inspecting the imported coefficients — it is
      *not* usable for `predict()`, since the smooth bases (knots, constraints, etc.) that the
      coefficients were fit against are not reconstructed.
    - With `data` (the original training data, as `{name: 1-D array}`): `~whittaker.model_matrix
      .build_model_matrix` is called on `data` to refit each smooth's basis and assemble the design
      matrix, the linear predictor and fitted values are recomputed from the imported coefficients,
      and a full `FitResult` (deviance, residuals, etc.) is attached. In this mode the returned
      model is fully usable for `predict()` on new data, since its smooth bases were rebuilt from
      the same training data mgcv used.

    The R family name in `d["family"]["family"]` is translated to the corresponding `whittaker`
    family class via an internal mapping (`_mgcv_family_map`), e.g. `"gaussian"` -> `Gaussian`,
    `"poisson"` -> `Poisson`, `"binomial"` -> `Binomial`, `"Gamma"` -> `Gamma`,
    `"inverse.gaussian"` -> `InverseGaussian`, `"Tweedie"` -> `Tweedie`, `"nb"` ->
    `NegativeBinomial`, `"cox.ph"` -> `CoxPH`, and `"betar"` -> `Beta`. A family name not in this
    table is passed through unchanged and will raise `ValueError` if it does not match a known
    `whittaker` family class.

    Parameters
    ----------
    d : dict
        An mgcv-compatible dictionary, e.g. parsed from `jsonlite::toJSON(gam_model)` in R, or
        produced by `to_mgcv_dict`. Must contain at least `"coefficients"`; `"formula"`, `"family"`,
        `"sp"`, and `"smooth"` are used when present to reconstruct the formula, family, and
        smoothing parameters as accurately as possible.
    data : dict[str, numpy.ndarray], optional
        The original training data used to fit the model in R, as `{name: 1-D array}`. When given,
        smooth bases are rebuilt from this data and the returned model supports `predict()`. When
        omitted (the default), only coefficients and smoothing parameters are restored and the model
        cannot be used for prediction.

    Returns
    -------
    GAM
        A `~whittaker.gam.GAM` instance. Fully fitted and prediction-ready when `data` is provided;
        otherwise a formula/family/coefficient container only.

    Notes
    -----
    This function is intended to interoperate with the R `mgcv` package's `gam` object structure.
    Full fidelity is not guaranteed: only the family names listed in `_mgcv_family_map` are
    recognized, and mgcv fields with no `whittaker` counterpart (e.g. certain smooth-specific `xt`
    options) are ignored rather than reconstructed.

    Examples
    --------
    ```{python}
    import numpy as np
    import whittaker as wt
    from whittaker.io import to_mgcv_dict, from_mgcv_dict

    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(0, 1, 120))
    y = np.sin(2 * x) + rng.normal(scale=0.1, size=120)
    data = {"x": x, "y": y}

    model = wt.GAM("y ~ s(x)").fit(data)
    mgcv_dict = to_mgcv_dict(model)

    # Round-trip through the mgcv-style dict, refitting bases from the training data.
    reimported = from_mgcv_dict(mgcv_dict, data=data)
    reimported.predict({"x": np.linspace(0, 1, 3)}).values
    ```
    """
    from whittaker.gam import GAM
    from whittaker.model_matrix import build_model_matrix

    _mgcv_family_map = {
        "gaussian": "Gaussian",
        "poisson": "Poisson",
        "binomial": "Binomial",
        "Gamma": "Gamma",
        "inverse.gaussian": "InverseGaussian",
        "Tweedie": "Tweedie",
        "nb": "NegativeBinomial",
        "cox.ph": "CoxPH",
        "betar": "Beta",
    }

    family_info = d.get("family", {})
    if isinstance(family_info, str):
        family_info = {"family": family_info}
    fam_name = family_info.get("family", "gaussian")
    whittaker_name = _mgcv_family_map.get(fam_name, fam_name)
    fam_dict: dict[str, Any] = {"class": whittaker_name}
    if "power" in family_info:
        fam_dict["p"] = family_info["power"]
    if "theta" in family_info:
        fam_dict["theta"] = family_info["theta"]
    family = _family_from_dict(fam_dict)

    formula_str = d.get("formula", "")
    if isinstance(formula_str, str) and "~" in formula_str:
        from whittaker.formula.parser import parse

        formula = parse(formula_str)
    else:
        terms = []
        for s in d.get("smooth", []):
            variables = tuple(s.get("term", []))
            bs = s.get("bs", "tp")
            k = s.get("df", -1)
            terms.append(SmoothTerm(variables=variables, bs=bs, k=k, extra={}))
        response = formula_str.split("~")[0].strip() if "~" in formula_str else "y"
        formula = Formula(
            response=response,
            terms=terms,
            intercept=d.get("intercept", True),
        )

    coefficients = np.array(d["coefficients"])
    sp = d.get("sp", [])

    if data is not None:
        mm = build_model_matrix(formula, data)

        eta = mm.X @ coefficients
        if mm.offset is not None:
            eta = eta + mm.offset
        mu = family.link_inverse(eta)
        y = mm.response
        dev = family.deviance(y, mu)

        fr = FitResult(
            coefficients=coefficients,
            linear_predictor=eta,
            fitted_values=mu,
            smoothing_params=sp,
            scale=d.get("scale", 1.0),
            gcv_score=d.get("gcv.ubre", 0.0),
            edf=d.get("edf", []),
            edf_total=d.get("edf.total", 0.0),
            deviance=dev,
            n_iter=d.get("iter", 0),
            converged=d.get("converged", True),
            hat_matrix_trace=d.get("edf.total", 0.0),
            residuals=y - mu,
            null_deviance=d.get("null.deviance"),
            aic=d.get("aic"),
            method=d.get("method", "GCV"),
        )

        gam = GAM(formula, family=family)
        gam._model_matrix = mm
        gam._fit_result = fr
        gam._fitted = True
        gam._data = data
        return gam

    gam = GAM(formula, family=family)
    return gam
