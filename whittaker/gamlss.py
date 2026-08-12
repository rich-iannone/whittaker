r"""GAMLSS: Generalized Additive Models for Location, Scale, and Shape."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from whittaker.data import InputData, prepare_data
from whittaker.families.gamlss_base import GAMLSSFamily
from whittaker.families.gaussian_ls import GaussianLS
from whittaker.fitting.gamlss_fit import GAMLSSFitResult, _compute_zw, gamlss_fit
from whittaker.formula.parser import parse
from whittaker.formula.terms import Formula
from whittaker.model_matrix import ModelMatrix, build_model_matrix, predict_matrix


@dataclass
class GAMLSSPrediction:
    r"""Result of `GAMLSS.predict()`.

    Holds, for every distributional parameter in the fitted `GAMLSSFamily`, the predicted value on
    the response scale (`values`), the corresponding linear predictor (`linear_predictors`), and
    optionally the standard error of that linear predictor (`se`). Because a GAMLSS estimates several
    parameters at once (for example `mu` and `sigma` of a location-scale family), the results are
    keyed by parameter name rather than returned as a single array.

    Attributes
    ----------
    values:
        Dict mapping parameter names to predicted values on the response scale, i.e.
        `theta = g^{-1}(eta)` for each parameter's link function `g`.
    linear_predictors:
        Dict mapping parameter names to predicted linear predictors `eta = X @ beta` (plus any
        offset).
    se:
        Dict mapping parameter names to standard errors on the linear predictor scale, or `None` if
        `se=False` was passed to `predict()`.
    """

    values: dict[str, NDArray]
    linear_predictors: dict[str, NDArray]
    se: dict[str, NDArray] | None = None


class GAMLSS:
    r"""Generalized Additive Model for Location, Scale, and Shape.

    A GAMLSS extends the ordinary GAM by allowing every parameter of the response distribution, not
    just its mean, to depend on covariates through its own smooth additive predictor. For a
    distribution with parameters `theta_1, ..., theta_K` (e.g. location `mu`, scale `sigma`, and
    possibly shape parameters `nu`, `tau`), each parameter has its own link function `g_k` and its own
    formula:

    $$g_k(\theta_k) = \eta_k = X_k \beta_k, \quad k = 1, \dots, K$$

    This makes it possible to model, for example, both the mean and the variance of `y` as smooth
    functions of `x`, which ordinary (mean-only) GAMs cannot do. Fitting uses the RS ("Rigby and
    Stasinopoulos") algorithm, which cycles through the parameters, holding all but one fixed,
    updating it via penalized IRLS, and repeating until the penalized log-likelihood converges.

    Use `GAMLSS` when the assumption of a fixed dispersion (constant variance, constant shape) is
    implausible, such as heteroscedastic regression, or regression with distributions like the
    negative binomial or beta that have separate location and shape parameters.

    Parameters
    ----------
    formulas:
        Dict mapping parameter names to formula strings. All formulas must share the same response
        variable. Example: `{"mu": "y ~ s(x1)", "sigma": "y ~ s(x2)"}`. The set of keys must match
        `family.parameter_names` exactly.
    family:
        A `GAMLSSFamily` specifying the distributional model, including the number and names of
        parameters, their link functions, and log-likelihood derivatives used by the RS algorithm.
        Defaults to `GaussianLS()` (Gaussian location-scale, i.e. `mu` and `sigma` both modeled).

    Notes
    -----
    Rigby & Stasinopoulos (2005) formulate GAMLSS fitting as penalized maximum likelihood. Within
    each outer RS iteration, and for each parameter `theta_k` in turn, a working response and weight
    are formed from the score and Fisher information of the log-likelihood with respect to `theta_k`:

    $$z_k = \eta_k + \frac{\partial \ell / \partial \theta_k}{\partial^2 \ell / \partial \theta_k^2}
    \cdot g_k'(\theta_k), \qquad w_k = -\frac{\partial^2 \ell}{\partial \theta_k^2}
    \Big/ g_k'(\theta_k)^2$$

    and a penalized weighted least squares problem is solved for `beta_k`, with all other parameters
    held at their current fitted values. Smoothing parameters for each parameter's smooth terms can be
    selected by GCV, REML, or ML at every inner iteration. The algorithm alternates over parameters
    until the global deviance `-2 * log_likelihood` stops improving.

    Examples
    --------
    ```{python}
    import numpy as np
    from whittaker.gamlss import GAMLSS
    from whittaker.families.gaussian_ls import GaussianLS

    rng = np.random.default_rng(0)
    n = 500
    x = rng.uniform(0, 1, n)
    mu = np.sin(2 * np.pi * x)
    sigma = np.exp(-1 + 2 * x)
    y = rng.normal(mu, sigma)

    model = GAMLSS(
        formulas={"mu": "y ~ s(x)", "sigma": "y ~ s(x)"},
        family=GaussianLS(),
    )
    model.fit({"x": x, "y": y}, method="REML")
    pred = model.predict({"x": x[:5]})
    print(pred.values)
    ```
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
        """Whether `fit()` has been called successfully.

        Returns
        -------
        bool
            `True` once a fit result is available, `False` otherwise. All other accessors
            (`coefficients()`, `fitted_values()`, `edf()`, etc.) raise `RuntimeError` when this
            is `False`.
        """
        return self._fit_result is not None

    @property
    def family(self) -> GAMLSSFamily:
        """The `GAMLSSFamily` used to build this model.

        Returns
        -------
        GAMLSSFamily
            The distributional family passed to (or defaulted in) `__init__`, giving the
            parameter names, link functions, and likelihood used by the RS fitting algorithm.
        """
        return self._family

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted. Call .fit() first.")

    @property
    def log_likelihood(self) -> float:
        """Log-likelihood of the fitted model at convergence.

        Returns
        -------
        float
            The (penalized) log-likelihood evaluated at the final RS iteration's coefficients
            and smoothing parameters, related to `global_deviance` by
            `global_deviance = -2 * log_likelihood`.
        """
        self._check_fitted()
        return self._fit_result.log_likelihood

    @property
    def aic(self) -> float:
        """Akaike information criterion of the fitted model.

        Computed from the global deviance and the total effective degrees of freedom summed
        across all distributional parameters' smooth and parametric terms, and can be used to
        compare GAMLSS models fit with different formulas or families on the same data.

        Returns
        -------
        float
            `AIC = global_deviance + 2 * edf_total`, lower values indicating a better
            deviance/complexity trade-off.
        """
        self._check_fitted()
        return self._fit_result.aic

    @property
    def bic(self) -> float:
        """Bayesian information criterion of the fitted model.

        Like `aic`, but penalizes model complexity more heavily (using `log(n)` instead of `2`
        as the multiplier on total effective degrees of freedom), so it tends to favor simpler
        models when comparing fits.

        Returns
        -------
        float
            `BIC = global_deviance + log(n) * edf_total`.
        """
        self._check_fitted()
        return self._fit_result.bic

    @property
    def global_deviance(self) -> float:
        """Global deviance of the fitted model.

        The RS algorithm minimizes this quantity at each outer iteration; it is defined as
        `-2 * log_likelihood` and is the primary measure of fit used to check convergence.

        Returns
        -------
        float
            The deviance at the final RS iteration.
        """
        self._check_fitted()
        return self._fit_result.global_deviance

    @property
    def converged(self) -> bool:
        """Whether the RS algorithm converged before hitting `max_outer` iterations.

        Returns
        -------
        bool
            `True` if the change in global deviance between successive outer iterations fell
            below `tol` before `max_outer` was reached, `False` otherwise.
        """
        self._check_fitted()
        return self._fit_result.converged

    @property
    def n_iter(self) -> int:
        """Number of outer RS iterations actually performed.

        Returns
        -------
        int
            Count of full passes over all distributional parameters carried out during fitting,
            at most `max_outer`.
        """
        self._check_fitted()
        return self._fit_result.n_iter

    def coefficients(self, parameter: str) -> NDArray:
        """Fitted basis coefficients for one distributional parameter.

        Parameters
        ----------
        parameter:
            Name of the distributional parameter (must be one of `family.parameter_names`), e.g.
            `"mu"` or `"sigma"`.

        Returns
        -------
        NDArray
            Coefficient vector `beta_k` for that parameter's linear predictor
            `eta_k = X_k @ beta_k`, in the order of its model matrix's columns.
        """
        self._check_fitted()
        return self._fit_result.params[parameter].coefficients

    def smoothing_params(self, parameter: str) -> list[float]:
        """Fitted smoothing parameters for one distributional parameter's smooth terms.

        Parameters
        ----------
        parameter:
            Name of the distributional parameter, e.g. `"mu"` or `"sigma"`.

        Returns
        -------
        list[float]
            One smoothing parameter (`lambda`) per smooth term in that parameter's formula, in
            the order the terms were declared. Empty if the formula has no smooth terms.
        """
        self._check_fitted()
        return self._fit_result.params[parameter].smoothing_params

    def edf(self, parameter: str) -> list[float]:
        """Effective degrees of freedom per smooth term for one distributional parameter.

        Parameters
        ----------
        parameter:
            Name of the distributional parameter, e.g. `"mu"` or `"sigma"`.

        Returns
        -------
        list[float]
            EDF of each smooth term in that parameter's formula, reflecting how much smoothing
            was applied (lower values indicate heavier penalization toward linearity).
        """
        self._check_fitted()
        return self._fit_result.params[parameter].edf

    def fitted_values(self, parameter: str | None = None) -> dict[str, NDArray] | NDArray:
        """Fitted values on the response scale for one or all distributional parameters.

        Parameters
        ----------
        parameter:
            If given, return only this parameter's fitted values as a single array. If `None`
            (default), return a dict of fitted values for every distributional parameter.

        Returns
        -------
        dict[str, NDArray] or NDArray
            `theta_k = g_k^{-1}(eta_k)` for the requested parameter(s), evaluated at the training
            covariate values used in `fit()`.
        """
        self._check_fitted()
        if parameter is not None:
            return self._fit_result.params[parameter].fitted_values
        return {name: pr.fitted_values for name, pr in self._fit_result.params.items()}

    def fit(
        self,
        data: InputData,
        *,
        method: str = "GCV",
        max_outer: int = 50,
        max_inner: int = 20,
        tol: float = 1e-6,
        select: bool = False,
    ) -> GAMLSS:
        r"""Fit the GAMLSS model via the RS algorithm.

        Parses each parameter's formula, builds its model matrix (basis functions, penalties, and
        any parametric terms), and then runs the outer Rigby & Stasinopoulos loop: for each
        parameter in turn, forms the working response and IRLS weights from the family's likelihood
        derivatives, selects smoothing parameters (if the formula includes smooth terms), and solves
        a penalized weighted least squares problem, holding the other parameters fixed. This repeats
        until the global deviance stabilizes or `max_outer` iterations are reached.

        Parameters
        ----------
        data:
            Column-oriented data dict (or any type accepted by `prepare_data`) containing the shared
            response column and all covariates referenced in `formulas`.
        method:
            Smoothing parameter selection method applied to every parameter's smooth terms:
            `"GCV"` (generalized cross-validation, the default), `"REML"` (restricted maximum
            likelihood), or `"ML"` (maximum likelihood).
        max_outer:
            Maximum number of outer RS iterations (full passes over all parameters).
        max_inner:
            Maximum inner iterations per parameter within a single outer step, used to converge the
            penalized IRLS fit for that parameter before moving to the next one.
        tol:
            Relative convergence tolerance, applied both to the coefficient update within a
            parameter's inner loop and to the change in overall log-likelihood between outer
            iterations.
        select:
            If `True`, add an extra shrinkage penalty to each smooth's null space so that terms can
            be shrunk essentially to zero, enabling automatic term selection.

        Returns
        -------
        GAMLSS
            Returns `self`, with `is_fitted` now `True` and per-parameter results accessible via
            `coefficients()`, `fitted_values()`, `edf()`, and related accessors.
        """
        data = prepare_data(data)
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
        new_data: InputData,
        *,
        parameter: str | None = None,
        se: bool = False,
    ) -> GAMLSSPrediction | NDArray:
        r"""Predict distributional parameters for new data.

        For each parameter `theta_k`, builds the prediction design matrix from the fitted basis
        (using the same knots/constraints as training), forms the linear predictor
        `eta_k = X_new @ beta_k` (plus any offset), and maps it back to the response scale via the
        parameter's inverse link, `theta_k = g_k^{-1}(eta_k)`. When `se=True`, the standard error of
        `eta_k` is also computed from the Bayesian posterior covariance of `beta_k`.

        Parameters
        ----------
        new_data:
            Column-oriented new data dict containing all covariates used in the fitted formulas.
        parameter:
            If given, return only this parameter's predicted values as an array on the response
            scale. Otherwise return a `GAMLSSPrediction` with all parameters.
        se:
            If `True`, compute standard errors on the linear predictor scale for each parameter,
            using the Bayesian covariance `V_beta_k = (X_k' W_k X_k + S_k)^{-1}` implied by the final
            IRLS weights and smoothing parameters for that parameter.

        Returns
        -------
        GAMLSSPrediction or NDArray
            A `GAMLSSPrediction` with all parameters' values, linear predictors, and (optionally)
            standard errors, or a single `NDArray` of predicted values if `parameter` is given.
        """
        self._check_fitted()
        new_data = prepare_data(new_data)

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

        Draws `n_sim` independent replicate response vectors from the fitted distribution, using
        each observation's fitted parameter values (`mu`, `sigma`, and any shape parameters) as the
        distribution's parameters. This is useful for posterior-predictive checks, e.g. comparing
        the distribution of simulated data against the observed response.

        Parameters
        ----------
        n_sim:
            Number of independent simulated replicates to draw per observation.
        seed:
            Random seed for reproducibility.

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
        """Return a text summary of the fitted model.

        Includes global fit statistics (global deviance, AIC, BIC, log-likelihood, convergence
        status and iteration count) followed by a per-parameter section listing the total
        effective degrees of freedom and, if present, the EDF of each individual smooth term.

        Returns
        -------
        str
            Multi-line, human-readable summary suitable for printing.
        """
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
