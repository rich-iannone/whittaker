"""GAMLSS: Generalized Additive Models for Location, Scale, and Shape."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from whittaker.families.gamlss_base import GAMLSSFamily
from whittaker.families.gaussian_ls import GaussianLS
from whittaker.fitting.gamlss_fit import GAMLSSFitResult, _compute_zw, gamlss_fit
from whittaker.formula.parser import parse
from whittaker.formula.terms import Formula
from whittaker.model_matrix import ModelMatrix, build_model_matrix, predict_matrix


@dataclass
class GAMLSSPrediction:
    """Result of `GAMLSS.predict()`.

    Attributes
    ----------
    values:
        Dict mapping parameter names to predicted values on the response scale.
    linear_predictors:
        Dict mapping parameter names to predicted linear predictors.
    se:
        Dict mapping parameter names to standard errors on the linear predictor scale, or `None` if
        `se=False`.
    """

    values: dict[str, NDArray]
    linear_predictors: dict[str, NDArray]
    se: dict[str, NDArray] | None = None


class GAMLSS:
    """Generalized Additive Model for Location, Scale, and Shape.

    Fits multiple distributional parameters simultaneously, each with its
    own additive predictor. Uses the RS algorithm (Rigby & Stasinopoulos 2005).

    Parameters
    ----------
    formulas:
        Dict mapping parameter names to formula strings. All formulas must share the same response
        variable. Example: `{"mu": "y ~ s(x1)", "sigma": "y ~ s(x2)"}`
    family:
        A `GAMLSSFamily` specifying the distributional model. Defaults to `GaussianLS()` (Gaussian
        location-scale).

    Examples
    --------
    >>> model = GAMLSS(
    ...     formulas={"mu": "y ~ s(x)", "sigma": "y ~ s(x)"},
    ...     family=GaussianLS(),
    ... )
    >>> model.fit(data, method="REML")
    >>> pred = model.predict(new_data)
    """

    def __init__(
        self,
        formulas: dict[str, str],
        family: GAMLSSFamily | None = None,
    ) -> None:
        if family is None:
            family = GaussianLS()
        self._family = family
        self._formula_strings = dict(formulas)
        self._formulas: dict[str, Formula] = {}
        self._models: dict[str, ModelMatrix] = {}
        self._fit_result: GAMLSSFitResult | None = None

        for param in family.parameter_names:
            if param not in formulas:
                raise ValueError(
                    f"Missing formula for parameter '{param}'. "
                    f"Required parameters: {family.parameter_names}"
                )

    @property
    def is_fitted(self) -> bool:
        return self._fit_result is not None

    @property
    def family(self) -> GAMLSSFamily:
        return self._family

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted. Call .fit() first.")

    @property
    def log_likelihood(self) -> float:
        self._check_fitted()
        return self._fit_result.log_likelihood

    @property
    def aic(self) -> float:
        self._check_fitted()
        return self._fit_result.aic

    @property
    def bic(self) -> float:
        self._check_fitted()
        return self._fit_result.bic

    @property
    def global_deviance(self) -> float:
        self._check_fitted()
        return self._fit_result.global_deviance

    @property
    def converged(self) -> bool:
        self._check_fitted()
        return self._fit_result.converged

    @property
    def n_iter(self) -> int:
        self._check_fitted()
        return self._fit_result.n_iter

    def coefficients(self, parameter: str) -> NDArray:
        self._check_fitted()
        return self._fit_result.params[parameter].coefficients

    def smoothing_params(self, parameter: str) -> list[float]:
        self._check_fitted()
        return self._fit_result.params[parameter].smoothing_params

    def edf(self, parameter: str) -> list[float]:
        self._check_fitted()
        return self._fit_result.params[parameter].edf

    def fitted_values(self, parameter: str | None = None) -> dict[str, NDArray] | NDArray:
        self._check_fitted()
        if parameter is not None:
            return self._fit_result.params[parameter].fitted_values
        return {name: pr.fitted_values for name, pr in self._fit_result.params.items()}

    def fit(
        self,
        data: dict[str, NDArray],
        *,
        method: str = "GCV",
        max_outer: int = 50,
        max_inner: int = 20,
        tol: float = 1e-6,
        select: bool = False,
    ) -> GAMLSS:
        """Fit the GAMLSS model.

        Parameters
        ----------
        data:
            Column-oriented data dict.
        method:
            Smoothing parameter selection: `"GCV"`, `"REML"`, or `"ML"`.
        max_outer:
            Maximum outer RS iterations.
        max_inner:
            Maximum inner iterations per parameter per outer step.
        tol:
            Convergence tolerance.
        select:
            Whether to add shrinkage penalties for smooth selection.

        Returns
        -------
        self
        """
        response_name = None
        for param, fstr in self._formula_strings.items():
            formula = parse(fstr)
            self._formulas[param] = formula
            if response_name is None:
                response_name = formula.response
            elif formula.response != response_name:
                raise ValueError(
                    f"All formulas must share the same response variable. "
                    f"Parameter '{param}' uses '{formula.response}' but expected '{response_name}'."
                )

        models: dict[str, ModelMatrix] = {}
        y = None
        for param in self._family.parameter_names:
            model = build_model_matrix(self._formulas[param], data, select=select)
            models[param] = model
            if y is None:
                y = model.response

        self._models = models
        self._fit_result = gamlss_fit(
            models,
            self._family,
            y,
            method=method,
            max_outer=max_outer,
            max_inner=max_inner,
            tol=tol,
        )
        return self

    def predict(
        self,
        new_data: dict[str, NDArray],
        *,
        parameter: str | None = None,
        se: bool = False,
    ) -> GAMLSSPrediction | NDArray:
        """Predict distributional parameters for new data.

        Parameters
        ----------
        new_data:
            Column-oriented new data dict.
        parameter:
            If given, return only this parameter's predicted values as an array. Otherwise return a
            `GAMLSSPrediction` with all parameters.
        se:
            If `True`, compute standard errors on the linear predictor scale for each parameter.

        Returns
        -------
        `GAMLSSPrediction` or `NDArray`
        """
        self._check_fitted()

        values: dict[str, NDArray] = {}
        linear_predictors: dict[str, NDArray] = {}
        se_dict: dict[str, NDArray] = {} if se else {}

        for name in self._family.parameter_names:
            model = self._models[name]
            X_new = predict_matrix(model, new_data)
            beta = self._fit_result.params[name].coefficients
            eta = X_new @ beta
            offset = model.offset
            if offset is not None:
                eta = eta + offset
            theta = self._family.link_inverse(name, eta)
            values[name] = theta
            linear_predictors[name] = eta

            if se:
                se_dict[name] = self._prediction_se(name, X_new)

        if parameter is not None:
            return values[parameter]

        return GAMLSSPrediction(
            values=values,
            linear_predictors=linear_predictors,
            se=se_dict if se else None,
        )

    def _prediction_se(self, param: str, X_new: NDArray) -> NDArray:
        from whittaker.fitting.inference import _bayesian_covariance

        fr = self._fit_result
        pr = fr.params[param]
        model = self._models[param]
        y = fr.response

        _, W_irls = _compute_zw(
            self._family,
            param,
            y,
            {name: r.fitted_values for name, r in fr.params.items()},
            pr.linear_predictor,
        )

        sp = pr.smoothing_params if pr.smoothing_params else [1.0] * len(model.penalties)
        V_beta = _bayesian_covariance(model.X, model.penalties, sp, scale=1.0, W=W_irls)
        var_diag = np.sum(X_new * (X_new @ V_beta), axis=1)
        return np.sqrt(np.maximum(var_diag, 0.0))

    def simulate(self, n_sim: int = 1, *, seed: int | None = None) -> NDArray:
        """Simulate responses from the fitted model.

        Returns
        -------
        NDArray
            Simulated values, shape `(n, n_sim)`.
        """
        self._check_fitted()
        rng = np.random.default_rng(seed)
        params = {name: pr.fitted_values for name, pr in self._fit_result.params.items()}
        sims = np.column_stack([self._family.simulate(params, rng) for _ in range(n_sim)])
        return sims

    def summary(self) -> str:
        """Return a text summary of the fitted model."""
        self._check_fitted()
        fr = self._fit_result
        lines = ["GAMLSS fit summary", "=" * 40]
        lines.append(f"Family: {self._family!r}")
        lines.append(f"N obs: {fr.n_obs}")
        lines.append(f"Global deviance: {fr.global_deviance:.4f}")
        lines.append(f"AIC: {fr.aic:.4f}")
        lines.append(f"BIC: {fr.bic:.4f}")
        lines.append(f"Log-likelihood: {fr.log_likelihood:.4f}")
        lines.append(f"Converged: {fr.converged} ({fr.n_iter} iterations)")
        lines.append("")

        for name in self._family.parameter_names:
            pr = fr.params[name]
            lines.append(f"--- {name} ---")
            lines.append(f"  EDF total: {pr.edf_total:.2f}")
            if pr.edf:
                for i, edf_val in enumerate(pr.edf):
                    lines.append(f"  Smooth {i + 1}: edf = {edf_val:.2f}")
            lines.append("")

        return "\n".join(lines)
