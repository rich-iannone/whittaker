r"""Top-level GAM class: the primary user-facing API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from whittaker.data import InputData, prepare_data
from whittaker.families.base import Family
from whittaker.families.gaussian import Gaussian
from whittaker.families.tweedie_estimated import TweedieEstimated
from whittaker.fitting.mcmc import MCMCResult, mcmc_fit
from whittaker.fitting.pirls import FitResult, pirls_fit
from whittaker.fitting.vi import VIResult, vi_fit
from whittaker.formula.parser import parse
from whittaker.formula.terms import Formula
from whittaker.model_matrix import ModelMatrix, build_model_matrix, predict_matrix, predict_offset


@dataclass
class PredictionResult:
    """Container returned by `GAM.predict()` for `type="response"` or `type="link"`.

    Bundles the point predictions together with their optional standard errors and interval
    bounds so that all quantities produced by a single `predict()` call travel together. Use
    `values` for the predictions themselves; the other attributes are populated only when the
    corresponding arguments (`se=True`, `interval=...`) were requested.

    Attributes
    ----------
    values : numpy.ndarray
        Predicted values, shape `(n,)`. On the response scale (`mu`) when `type="response"`, or on
        the linear predictor scale (`eta`) when `type="link"`.
    se : numpy.ndarray or None
        Standard errors of the linear predictor, shape `(n,)`. `None` unless `se=True` was passed
        to `predict()`.
    linear_predictor : numpy.ndarray
        Predictions on the linear predictor scale, shape `(n,)`. Always populated, regardless of
        `type`, so that the response-scale mean can be recovered via the link function.
    lower : numpy.ndarray or None
        Lower bound of the requested interval, on the same scale as `values`. `None` unless
        `interval` was set to `"confidence"`, `"prediction"`, or `"simultaneous"`.
    upper : numpy.ndarray or None
        Upper bound of the requested interval, on the same scale as `values`. `None` unless
        `interval` was set.
    """

    values: NDArray
    se: NDArray | None
    linear_predictor: NDArray
    lower: NDArray | None = None
    upper: NDArray | None = None


@dataclass
class TermsPredictionResult:
    """Container returned by `GAM.predict(type="terms")`.

    Instead of collapsing every smooth's effect into a single linear predictor, each smooth term's
    contribution is kept separate. This is useful for decomposing a fitted additive model into its
    constituent partial effects (e.g., to inspect how much of the prediction at a point comes from
    `s(x1)` versus `s(x2)`) without needing to build partial-effect plots.

    Attributes
    ----------
    terms : dict[str, numpy.ndarray]
        Maps each term label (e.g. `"s(x1)"`, `"te(x1, x2)"`, or `"s(x1):group_a"` for factor-`by`
        smooths) to that term's contribution to the linear predictor, each of shape `(n,)`.
        Contributions sum (plus the intercept and any parametric terms) to the full linear
        predictor.
    se : dict[str, numpy.ndarray] or None
        Maps each term label to its per-term standard error, each of shape `(n,)`. `None` unless
        `se=True` was passed to `predict()`.
    labels : list[str]
        Term labels in formula order, matching the keys of `terms` and `se`.
    """

    terms: dict[str, NDArray]
    se: dict[str, NDArray] | None
    labels: list[str] = field(default_factory=list)

    @property
    def values(self) -> NDArray:
        """Sum of all term contributions (overall linear predictor)."""
        arrays = list(self.terms.values())
        return np.sum(arrays, axis=0)  # type: ignore[return-value]


@dataclass
class GamCheckResult:
    """Container returned by `GAM.gam_check()`, bundling residual diagnostics with fit summary
    statistics and basis-dimension adequacy checks.

    This mirrors the console output of R mgcv's `gam.check()`: it lets you inspect whether the
    residuals look well-behaved and whether any smooth's basis dimension `k` was set too small (in
    which case the smooth may be under-fitting), all from a single object. Printing the result (or
    relying on its `__repr__`) gives a compact textual report; the individual attributes are also
    available for building custom diagnostic plots (see `GAM.check()`).

    Attributes
    ----------
    deviance_residuals : numpy.ndarray
        Deviance residuals, shape `(n,)`. Should look approximately normal and homoscedastic for a
        well-specified model.
    fitted_values : numpy.ndarray
        Fitted values `mu` on the response scale, shape `(n,)`.
    response : numpy.ndarray
        Observed response values `y` used for fitting, shape `(n,)`.
    k_check : list[KCheckResult]
        One basis-dimension check per smooth term. Each entry reports a k-index and a
        simulation-based p-value; low p-values (typically flagged with `*`) suggest the smooth's
        basis dimension `k` may be too small to capture the true function.
    deviance_explained : float
        Proportion of null deviance explained by the model, in `[0, 1]` (analogous to R-squared for
        non-Gaussian families).
    scale : float
        The estimated scale (dispersion) parameter `phi`.
    edf_total : float
        The total effective degrees of freedom across all model terms.
    n_obs : int
        The number of observations used in the fit.
    """

    deviance_residuals: NDArray
    fitted_values: NDArray
    response: NDArray
    k_check: list
    deviance_explained: float
    scale: float
    edf_total: float
    n_obs: int

    def __repr__(self) -> str:
        lines = [
            "GAM check results",
            f"  n = {self.n_obs}, edf = {self.edf_total:.1f}, "
            f"scale = {self.scale:.4f}, dev.expl = {self.deviance_explained:.1%}",
            "",
            "Basis dimension check:",
        ]
        for kc in self.k_check:
            star = " *" if kc.p_value < 0.05 else ""
            lines.append(
                f"  {kc.term_label}: k_index={kc.k_index:.3f}, "
                f"edf={kc.edf:.1f}/{kc.k_prime}, p={kc.p_value:.3f}{star}"
            )
        return "\n".join(lines)


class ModelSummary:
    """Rich-display wrapper returned by `GAM.summary()`.

    Renders as plain text in terminals and Quarto/Jupyter cells; calling
    `model.summary()` as the last expression in a cell produces readable output
    without needing `print()`.
    """

    def __init__(self, text: str) -> None:
        self._text = text

    def __repr__(self) -> str:
        return self._text

    def __str__(self) -> str:
        return self._text

    def __contains__(self, item: str) -> bool:  # type: ignore[override]
        return item in self._text


class GAM:
    r"""Generalized Additive Model with automatic smoothness selection.

    A GAM extends the generalized linear model by replacing some or all linear predictor terms
    with smooth, data-driven functions of the covariates:

    $$
    g(\mathbb{E}[y_i]) = \eta_i = \beta_0 + \sum_j \beta_j x_{ij} + \sum_k f_k(z_{ik})
    $$

    where $g$ is a link function, $\beta_j x_{ij}$ are ordinary parametric (linear) terms, and
    each $f_k$ is an unspecified smooth function represented by a spline basis (`s()`) or a
    tensor product of bases for multivariate smooths (`te()`, `ti()`, `t2()`). This lets the
    model capture nonlinear relationships without having to guess a parametric form ahead of
    time, while still supporting the full range of exponential-family response distributions
    (Gaussian, Binomial, Poisson, Gamma, Tweedie, and more) via a `Family` object.

    Use `GAM` when you suspect a covariate's effect on the response is nonlinear, when you want
    interaction surfaces between two or more continuous covariates, or when you want automatic,
    data-driven control of model complexity rather than manually choosing a polynomial degree or
    a fixed set of basis functions.

    A `GAM` is specified with a formula string in an R/mgcv-like syntax, e.g.
    `"y ~ s(x1) + s(x2, bs='cr', k=15) + te(x3, x4) + group"`, where `s()` denotes a univariate
    (or `by=`-varying) smooth, `te()`/`ti()`/`t2()` denote tensor-product smooths of two or more
    variables, and bare names denote ordinary parametric terms. See `Formula`, `SmoothTerm`,
    `LinearTerm`, `InteractionTerm`, and `OffsetTerm` for the term types this formula parses
    into.

    Fitting (`fit()`) proceeds by Penalized Iteratively Reweighted Least Squares (P-IRLS): each
    smooth's wiggliness is controlled by a quadratic penalty $\lambda_k \boldsymbol{\beta}_k^T
    \mathbf{S}_k \boldsymbol{\beta}_k$ on its coefficients, and the smoothing parameters
    $\lambda_k$ are themselves estimated from the data. By default this is via Generalized
    Cross-Validation (GCV), or via Restricted Maximum Likelihood (REML) or Marginal Likelihood
    (ML) when smooths are treated as correlated random effects. Larger $\lambda_k$ shrinks a smooth
    toward a simpler (e.g. linear or constant) shape; smaller $\lambda_k$ allows more flexibility.
    This automatic selection is what distinguishes a GAM from simply choosing a fixed spline basis:
    the *effective* complexity of each term (its effective degrees of freedom, or EDF) is learned
    rather than fixed in advance.

    Once fitted, a `GAM` supports prediction with standard errors and intervals (`predict()`),
    partial-effect plotting (`plot()`), residual and basis-dimension diagnostics (`check()`,
    `gam_check()`, `k_check()`), hypothesis tests for parametric and smooth terms
    (`parametric_tests()`, `smooth_tests()`), and a text summary (`summary()`) analogous to
    `summary.gam()` in R's mgcv.

    Parameters
    ----------
    formula : str or Formula
        Model formula, either as a string (e.g. `"y ~ s(x1) + s(x2) + x3"`) or an already-parsed
        `Formula` object. The left-hand side names the response column; the right-hand side lists
        smooth terms (`s()`, `te()`, `ti()`, `t2()`), parametric terms (bare column names),
        interactions (`x1 * x2`), and optionally an `offset(...)` term. Use `0 +` or `- 1` on the
        right-hand side to suppress the intercept.
    family : Family or None
        Response distribution and link function. Defaults to `Gaussian()` (identity link) if not
        given. Other options include `Binomial`, `Poisson`, `Gamma`, and `Tweedie`-family classes,
        each defining the variance function, deviance, and link used during P-IRLS.

    Examples
    --------
    ```{python}
    import numpy as np
    from whittaker import GAM

    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 10, 200))
    y = np.sin(x) + rng.normal(scale=0.2, size=200)

    gam = GAM("y ~ s(x)").fit({"x": x, "y": y})
    print(gam.summary())
    ```
    """

    def __init__(
        self,
        formula: str | Formula,
        family: Family | None = None,
    ) -> None:
        if isinstance(formula, str):
            self._formula = parse(formula)
        else:
            self._formula = formula

        self._family = family if family is not None else Gaussian()
        self._fitted = False

        self._model_matrix: ModelMatrix
        self._fit_result: FitResult | VIResult | MCMCResult

    @property
    def formula(self) -> Formula:
        """The parsed model formula.

        This is the `Formula` object produced by parsing the formula string passed to
        `GAM.__init__()` (or the `Formula` object passed directly). It lists the response
        column and the parsed smooth (`s()`, `te()`, `ti()`, `t2()`), parametric, interaction,
        and offset terms that make up the right-hand side.

        Returns
        -------
        Formula
            The parsed formula.
        """
        return self._formula

    @property
    def family(self) -> Family:
        """The response distribution family used to fit this model.

        Determines the variance function, deviance, and link function used during P-IRLS.
        Defaults to `Gaussian()` when no `family` argument was given to `GAM.__init__()`.

        Returns
        -------
        Family
            The family object supplied to (or defaulted by) the constructor.
        """
        return self._family

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted.

        `True` once `fit()` has completed successfully; `False` beforehand. Most other
        properties and methods (`coefficients`, `predict()`, `summary()`, etc.) require this to
        be `True` and raise a `RuntimeError` via `_check_fitted()` otherwise.

        Returns
        -------
        bool
            Fitted status of the model.
        """
        return self._fitted

    def fit(
        self,
        data: InputData,
        *,
        smoothing_params: list[float] | None = None,
        method: str = "GCV",
        weights: NDArray | None = None,
        select: bool = False,
        vi_options: dict | None = None,
        mcmc_options: dict | None = None,
    ) -> GAM:
        r"""Fit the GAM to data via Penalized Iteratively Reweighted Least Squares (P-IRLS).

        Builds the model matrix from the formula, then alternates between (a) an IRLS step that
        linearizes the exponential-family log-likelihood around the current fit and (b) a
        penalized weighted-least-squares solve that shrinks each smooth toward simplicity
        according to its smoothing parameter. When `smoothing_params` is not fixed, this inner
        P-IRLS loop is itself wrapped in an outer loop that re-estimates the smoothing parameters
        (by GCV, REML, or ML) at each iteration, until both the coefficients and the smoothing
        parameters converge.

        Parameters
        ----------
        data : dict[str, numpy.ndarray]
            Column-oriented data as `{name: 1-D array}`. All columns referenced by the formula
            (response, smooth covariates, parametric terms, and any `by=` factor) must be
            present.
        smoothing_params : list[float] or None
            Fixed smoothing parameters $\lambda_k$, one per penalty (a `te()`/`t2()` term
            contributes more than one). If `None` (default), smoothing parameters are selected
            automatically via `method`.
        method : str
            Criterion used to select smoothing parameters when `smoothing_params` is not fixed.
            One of:

            - `"GCV"` (default): minimizes the Generalized Cross-Validation score
              $\text{GCV} = n \cdot D / (n - \text{tr}(\mathbf{H}))^2$, where $D$ is the deviance
              and $\mathbf{H}$ is the influence (hat) matrix. Fast and does not require treating
              smooths as random effects, but can occasionally undersmooth.
            - `"REML"`: maximizes the Restricted Maximum Likelihood, treating each smooth's
              penalized coefficients as correlated Gaussian random effects and integrating out
              the fixed (unpenalized) effects. Generally the most reliable choice and the one
              recommended when using `select=True`.
            - `"ML"`: maximizes the Marginal Likelihood, similar to REML but without correcting
              for uncertainty in the fixed effects; tends to undersmooth slightly relative to
              REML.
        weights : numpy.ndarray or None
            Observation (prior) weights, shape `(n,)`. Must be positive. When provided, the model
            minimizes the weighted deviance $\sum_i w_i d_i$ and uses weighted IRLS throughout.
        select : bool
            If `True`, augment each smooth's wiggliness penalty with a second penalty on its null
            space (the component, such as a pure linear trend, that the ordinary penalty never
            shrinks). With both penalties free, GCV/REML/ML can drive a term's smoothing
            parameters high enough to remove it from the model entirely, giving automatic term
            selection analogous to the lasso (Marra & Wood, 2011). Recommended together with
            `method="REML"`. Has no additional effect on bases whose null space is already zero
            (e.g. `bs="re"`, `bs="fs"`, or the shrinkage bases `"ts"`/`"cs"`).
        vi_options : dict or None
            Extra keyword arguments forwarded to `~whittaker.fitting.vi.vi_fit` when
            `method="VI"`.  Accepted keys: `n_quad`, `lr`, `max_iter`, `tol`,
            `patience`, `seed`, `cov_structure`, `phi_inference`.  Ignored for
            all other methods.
        mcmc_options : dict or None
            Extra keyword arguments forwarded to `~whittaker.fitting.mcmc.mcmc_fit` when
            `method="MCMC"`.  Accepted keys: `n_samples`, `n_warmup`, `n_chains`,
            `leapfrog_steps`, `target_accept`, `seed`.  Ignored for all other methods.

        Returns
        -------
        GAM
            Returns `self` for method chaining, e.g. `GAM(formula).fit(data).predict(new_data)`.
        """
        data = prepare_data(data)
        self._data = data

        if hasattr(self._family, "set_data"):
            cast(Any, self._family).set_data(data)

        self._model_matrix = build_model_matrix(self._formula, data, select=select)

        pw = None
        if weights is not None:
            pw = np.asarray(weights, dtype=float)
            if pw.ndim != 1 or len(pw) != self._model_matrix.n_obs:
                raise ValueError(
                    f"weights must be a 1-D array of length {self._model_matrix.n_obs}, "
                    f"got shape {pw.shape}."
                )
            if np.any(pw <= 0):
                raise ValueError("All weights must be positive.")

        if method.upper() == "VI":
            self._fit_result = vi_fit(
                self._model_matrix,
                self._family,
                smoothing_params=smoothing_params,
                prior_weights=pw,
                **(vi_options or {}),
            )
        elif method.upper() == "MCMC":
            self._fit_result = mcmc_fit(
                self._model_matrix,
                self._family,
                smoothing_params=smoothing_params,
                prior_weights=pw,
                **(mcmc_options or {}),
            )
        elif isinstance(self._family, TweedieEstimated) and not self._family.p_estimated:
            self._fit_result = self._profile_tweedie_power(
                self._model_matrix,
                self._family,
                smoothing_params=smoothing_params,
                method=method,
                prior_weights=pw,
            )
        else:
            self._fit_result = pirls_fit(
                self._model_matrix,
                self._family,
                smoothing_params=smoothing_params,
                method=method,
                prior_weights=pw,
            )
        self._fitted = True
        return self

    @staticmethod
    def _profile_tweedie_power(
        model_matrix: ModelMatrix,
        family: TweedieEstimated,
        *,
        smoothing_params: list[float] | None,
        method: str,
        prior_weights: NDArray | None,
    ) -> FitResult:
        from whittaker.families.tweedie import Tweedie

        p_lo, p_hi = family._p_range
        grid = np.linspace(p_lo, p_hi, family._n_grid)

        best_aic = np.inf
        best_result: FitResult | None = None
        best_p = grid[len(grid) // 2]

        for p_val in grid:
            candidate = Tweedie(p=p_val)
            try:
                result = pirls_fit(
                    model_matrix,
                    candidate,
                    smoothing_params=smoothing_params,
                    method=method,
                    prior_weights=prior_weights,
                )
            except Exception:
                continue
            if result.aic is not None and result.aic < best_aic:
                best_aic = result.aic
                best_result = result
                best_p = p_val

        if best_result is None:
            candidate = Tweedie(p=best_p)
            best_result = pirls_fit(
                model_matrix,
                candidate,
                smoothing_params=smoothing_params,
                method=method,
                prior_weights=prior_weights,
            )

        family._set_p(best_p)
        return best_result

    def predict(
        self,
        new_data: InputData,
        *,
        se: bool = False,
        type: str = "response",
        interval: str | None = None,
        level: float = 0.95,
        unconditional: bool = False,
    ) -> PredictionResult | TermsPredictionResult:
        r"""Predict from the fitted model on new data.

        Applies the estimated coefficients to a model matrix built from `new_data`, using the
        same basis transformations (knots, factor levels, centering constraints) fitted on the
        training data. Supports predictions on the response or linear-predictor scale, per-term
        decompositions, and pointwise or simultaneous uncertainty intervals.

        Parameters
        ----------
        new_data : dict[str, numpy.ndarray]
            Column-oriented new data. Must contain all covariate columns referenced by the
            formula (the response column is not needed).
        se : bool
            If `True`, compute standard errors on the linear predictor scale (for
            `type="response"` and `type="link"`) or per-term standard errors (for
            `type="terms"`). Default `False`.
        type : str
            Prediction type. One of:

            - `"response"` (default): predictions on the response scale,
              $\mu = g^{-1}(\eta)$.
            - `"link"`: predictions on the linear predictor scale, $\eta = \mathbf{X}
              \boldsymbol{\beta}$.
            - `"terms"`: individual smooth term contributions to the linear predictor, returned
              separately rather than summed (see `TermsPredictionResult`).
        interval : str or None
            Interval type. `None` (default) returns no intervals. Ignored (and must be `None`)
            when `type="terms"`. Otherwise one of:

            - `"confidence"`: interval for the mean response, reflecting uncertainty in $\eta$
              only.
            - `"prediction"`: interval for a new individual observation, adding the
              response-distribution variance on top of the uncertainty in $\eta$.
            - `"simultaneous"`: a band with `level` coverage for the *entire* curve
              simultaneously (via posterior simulation), rather than pointwise coverage.

            All interval types are computed on the linear predictor scale and transformed to the
            response scale for `type="response"`.
        level : float
            Nominal coverage probability for the interval, e.g. `0.95` for a 95% interval
            (default).
        unconditional : bool
            If `True`, include smoothing-parameter uncertainty in standard errors and intervals
            (Marra & Wood, 2012), using the unconditional covariance matrix $V_c$ in place of the
            conditional $V_p$. This produces wider, more honest intervals that account for the
            fact that $\lambda$ was itself estimated from the data. Requires that the model was
            fitted with `method="REML"` or `method="ML"`.

        Returns
        -------
        PredictionResult or TermsPredictionResult
            For `type="response"` or `type="link"`, a `PredictionResult` with `values`, `se`,
            `linear_predictor`, `lower`, and `upper`. For `type="terms"`, a
            `TermsPredictionResult` with per-smooth contributions and standard errors.

        Examples
        --------
        ```{python}
        import numpy as np
        from whittaker import GAM

        rng = np.random.default_rng(0)
        x = np.sort(rng.uniform(0, 10, 200))
        y = np.sin(x) + rng.normal(scale=0.2, size=200)

        gam = GAM("y ~ s(x)").fit({"x": x, "y": y})
        new_x = np.linspace(0, 10, 5)
        result = gam.predict({"x": new_x}, se=True, interval="confidence")
        result.values, result.lower, result.upper
        ```
        """
        self._check_fitted()
        new_data = prepare_data(new_data)
        type_lower = type.lower()

        if type_lower == "terms":
            if interval is not None:
                raise ValueError("Intervals are not supported for type='terms'.")
            return self._predict_terms(new_data, se=se, unconditional=unconditional)

        if type_lower not in ("response", "link"):
            raise ValueError(
                f"Unknown prediction type {type!r}. Choose from 'response', 'link', or 'terms'."
            )

        if unconditional and self._fit_result.method not in ("REML", "ML"):
            raise ValueError(
                "unconditional=True requires method='REML' or method='ML', "
                f"but model was fitted with method='{self._fit_result.method}'."
            )

        if interval is not None:
            interval_lower = interval.lower()
            if interval_lower not in ("confidence", "prediction", "simultaneous"):
                raise ValueError(
                    f"Unknown interval type {interval!r}. "
                    "Choose from 'confidence', 'prediction', or 'simultaneous'."
                )
        else:
            interval_lower = None

        X_new = predict_matrix(self._model_matrix, new_data)
        eta = X_new @ self._fit_result.coefficients

        new_offset = predict_offset(self._model_matrix, new_data)
        if new_offset is not None:
            eta = eta + new_offset

        need_se = se or interval_lower is not None
        se_values = self._prediction_se(X_new, unconditional=unconditional) if need_se else None

        lower = None
        upper = None
        if interval_lower is not None and se_values is not None:
            if interval_lower == "simultaneous":
                crit = self._simultaneous_quantile(
                    X_new,
                    level=level,
                    unconditional=unconditional,
                )
                eta_lower = eta - crit * se_values
                eta_upper = eta + crit * se_values
                if type_lower == "link":
                    lower, upper = eta_lower, eta_upper
                else:
                    lower = self._family.link_inverse(eta_lower)
                    upper = self._family.link_inverse(eta_upper)
            else:
                lower, upper = self._compute_interval(
                    eta,
                    se_values,
                    X_new,
                    interval_lower,
                    level,
                    type_lower,
                )

        if type_lower == "link":
            return PredictionResult(
                values=eta,
                se=se_values if se else None,
                linear_predictor=eta,
                lower=lower,
                upper=upper,
            )

        mu = self._family.link_inverse(eta)
        return PredictionResult(
            values=mu,
            se=se_values if se else None,
            linear_predictor=eta,
            lower=lower,
            upper=upper,
        )

    def _predict_terms(
        self,
        new_data: dict[str, NDArray],
        *,
        se: bool = False,
        unconditional: bool = False,
    ) -> TermsPredictionResult:
        """Compute per-smooth-term contributions to the linear predictor."""
        X_new = predict_matrix(self._model_matrix, new_data)
        beta = self._fit_result.coefficients

        V_beta = None
        if se:
            V_beta = self._covariance_matrix(unconditional=unconditional)

        terms_dict: dict[str, NDArray] = {}
        se_dict: dict[str, NDArray] | None = {} if se else None
        labels: list[str] = []

        for info in self._model_matrix.smooths:
            cs, ce = info.col_start, info.col_end
            label = repr(info.term)
            if info.by_level is not None:
                label = f"{label}:{info.by_level}"

            X_j = X_new[:, cs:ce]
            beta_j = beta[cs:ce]
            terms_dict[label] = X_j @ beta_j
            labels.append(label)

            if se and V_beta is not None and se_dict is not None:
                V_j = V_beta[cs:ce, cs:ce]
                var_j = np.sum(X_j * (X_j @ V_j), axis=1)
                se_dict[label] = np.sqrt(np.maximum(var_j, 0.0))

        return TermsPredictionResult(terms=terms_dict, se=se_dict, labels=labels)

    def _prediction_se(self, X_new: NDArray, *, unconditional: bool = False) -> NDArray:
        """Compute standard errors for predictions at X_new.

        SE = sqrt(diag(X_new @ V_β @ X_new.T))

        where V_β is the Bayesian posterior covariance (conditional) or the unconditional covariance
        that includes smoothing parameter uncertainty (Marra & Wood 2012).
        """
        W = self._combined_weights()
        V_beta = self._covariance_matrix(unconditional=unconditional, W=W)
        var_diag = np.sum(X_new * (X_new @ V_beta), axis=1)
        return np.sqrt(np.maximum(var_diag, 0.0))

    def _covariance_matrix(
        self, *, unconditional: bool = False, W: NDArray | None = None
    ) -> NDArray:
        """Return the coefficient covariance matrix V_β.

        When `unconditional=True`, returns `V_c` (Marra & Wood 2012) which accounts for smoothing
        Otherwise returns the conditional Bayesian covariance `V_p`.
        """
        from whittaker.fitting.inference import _bayesian_covariance

        if W is None:
            W = self._combined_weights()

        if isinstance(self._fit_result, (VIResult, MCMCResult)):
            return self._fit_result.posterior_cov

        if unconditional and self._fit_result.method in ("REML", "ML"):
            from whittaker.fitting.inference import _unconditional_covariance

            n_unpenalized = (
                1 if self._model_matrix.has_intercept else 0
            ) + self._model_matrix.n_parametric
            for s_info in self._model_matrix.smooths:
                n_unpenalized += s_info.null_space_dim

            return _unconditional_covariance(
                self._model_matrix.X,
                self._model_matrix.penalties,
                self._fit_result.smoothing_params,
                self._fit_result.scale,
                self._fit_result.coefficients,
                self._fit_result.method,
                W=W,
                n_unpenalized=n_unpenalized,
                y=self._fit_result.pseudo_data,
                offset=self._model_matrix.offset,
            )

        return _bayesian_covariance(
            self._model_matrix.X,
            self._model_matrix.penalties,
            self._fit_result.smoothing_params,
            self._fit_result.scale,
            W=W,
        )

    def _combined_weights(self) -> NDArray | None:
        """Combine IRLS working weights and prior weights into a single weight vector."""
        W_irls = self._fit_result.weights
        pw = self._fit_result.prior_weights
        if W_irls is not None and pw is not None:
            return pw * W_irls
        if W_irls is not None:
            return W_irls
        if pw is not None:
            return pw
        return None

    def _compute_interval(
        self,
        eta: NDArray,
        se_eta: NDArray,
        X_new: NDArray,
        interval_type: str,
        level: float,
        pred_type: str,
    ) -> tuple[NDArray, NDArray]:
        """Compute confidence or prediction interval bounds on the response scale."""
        from scipy.stats import norm
        from scipy.stats import t as t_dist

        n_obs = self._model_matrix.X.shape[0]
        residual_df = n_obs - self._fit_result.edf_total

        if self._family.scale_known:
            q = norm.ppf(1.0 - (1.0 - level) / 2.0)
        else:
            q = t_dist.ppf(1.0 - (1.0 - level) / 2.0, df=max(residual_df, 1.0))

        if interval_type == "prediction":
            mu_for_var = self._family.link_inverse(eta)
            response_var = self._family.variance(mu_for_var) * self._fit_result.scale
            total_se = np.sqrt(se_eta**2 + response_var)
        else:
            total_se = se_eta

        eta_lower = eta - q * total_se
        eta_upper = eta + q * total_se

        if pred_type == "link":
            return eta_lower, eta_upper

        return self._family.link_inverse(eta_lower), self._family.link_inverse(eta_upper)

    def _simultaneous_quantile(
        self,
        X_new: NDArray,
        *,
        level: float = 0.95,
        n_sim: int = 10_000,
        unconditional: bool = False,
        seed: int = 0,
    ) -> float:
        """Compute the critical value for simultaneous confidence bands.

        Simulates from the posterior of β, computes max |deviation / se| across all prediction
        points, and returns the empirical quantile at the requested coverage level.
        """
        V_beta = self._covariance_matrix(unconditional=unconditional)
        se_values = np.sqrt(np.maximum(np.sum(X_new * (X_new @ V_beta), axis=1), 0.0))
        se_values = np.maximum(se_values, np.finfo(float).eps)

        rng = np.random.default_rng(seed)
        L = np.linalg.cholesky(V_beta + np.eye(V_beta.shape[0]) * 1e-10)

        max_devs = np.empty(n_sim)
        for i in range(n_sim):
            z = rng.standard_normal(V_beta.shape[0])
            beta_sim = L @ z
            f_sim = X_new @ beta_sim
            max_devs[i] = np.max(np.abs(f_sim) / se_values)

        return float(np.quantile(max_devs, level))

    def simultaneous_ci(
        self,
        new_data: InputData,
        *,
        term: int | str | None = None,
        level: float = 0.95,
        n_sim: int = 10_000,
        unconditional: bool = False,
        seed: int = 0,
    ) -> dict:
        """Compute simultaneous confidence bands for smooth terms.

        Unlike pointwise intervals, these bands have (approximate) `level=` coverage probability for
        the *entire* function simultaneously, not just at individual points.

        Parameters
        ----------
        new_data:
            Prediction data.
        term:
            Which smooth term to compute bands for. An integer index (0-based) or the term label
            string. If `None` and the model has exactly one smooth, that term is used.
        level:
            Nominal simultaneous coverage probability (default `0.95`).
        n_sim:
            Number of posterior simulations for the critical value (default `10_000`).
        unconditional:
            If `True`, include smoothing parameter uncertainty.
        seed:
            Random seed for reproducibility.

        Returns
        -------
        dict
            Keys: `"estimate"`, `"se"`, `"lower"`, `"upper"`, `"term_label"`, `"crit_value"`.
        """
        self._check_fitted()
        new_data = prepare_data(new_data)

        smooths = self._model_matrix.smooths
        if term is None:
            if len(smooths) != 1:
                raise ValueError(
                    f"Model has {len(smooths)} smooth terms; specify which one via 'term'."
                )
            idx = 0
        elif isinstance(term, int):
            idx = term
        else:
            idx = None
            for i, s in enumerate(smooths):
                if repr(s.term) == term or term in repr(s.term):
                    idx = i
                    break
            if idx is None:
                raise ValueError(f"No smooth term matching {term!r}.")

        info = smooths[idx]
        cs, ce = info.col_start, info.col_end

        X_new = predict_matrix(self._model_matrix, new_data)
        X_j = np.zeros_like(X_new)
        X_j[:, cs:ce] = X_new[:, cs:ce]

        beta = self._fit_result.coefficients
        estimate = X_j @ beta

        V_beta = self._covariance_matrix(unconditional=unconditional)
        se_values = np.sqrt(np.maximum(np.sum(X_j * (X_j @ V_beta), axis=1), 0.0))

        crit = self._simultaneous_quantile(
            X_j,
            level=level,
            n_sim=n_sim,
            unconditional=unconditional,
            seed=seed,
        )

        label = repr(info.term)
        if info.by_level is not None:
            label = f"{label}:{info.by_level}"

        return {
            "estimate": estimate,
            "se": se_values,
            "lower": estimate - crit * se_values,
            "upper": estimate + crit * se_values,
            "term_label": label,
            "crit_value": crit,
        }

    @property
    def coefficients(self) -> NDArray:
        r"""Estimated coefficients $\boldsymbol{\beta}$.

        A single flat vector holding the intercept, parametric term coefficients, and every
        smooth term's basis coefficients concatenated in formula order. Use
        `self._model_matrix.smooths` (or `predict(type="terms")`) to map sub-ranges of this
        vector back to individual terms.

        Returns
        -------
        numpy.ndarray
            Coefficient vector, shape `(n_coefs,)`. A copy, safe to mutate.
        """
        self._check_fitted()
        return self._fit_result.coefficients.copy()

    @property
    def fitted_values(self) -> NDArray:
        r"""Fitted values $\mu$ on the response scale.

        Equal to $g^{-1}(\eta)$ where $\eta = \mathbf{X}\boldsymbol{\beta}$ is the linear
        predictor evaluated on the training data used in `fit()`.

        Returns
        -------
        numpy.ndarray
            Fitted response values, shape `(n,)`. A copy, safe to mutate.
        """
        self._check_fitted()
        return self._fit_result.fitted_values.copy()

    @property
    def residuals(self) -> NDArray:
        r"""Response residuals ($y - \mu$) on the training data.

        These are the raw (unstandardized) residuals. For Pearson, deviance, or working residuals
        (or residuals on new data) use `get_residuals()` instead.

        Returns
        -------
        numpy.ndarray
            Residual vector, shape `(n,)`. A copy, safe to mutate.
        """
        self._check_fitted()
        return self._require_fit_result().residuals.copy()

    def get_residuals(self, type: str = "deviance") -> NDArray:
        """Compute residuals of the specified type.

        Parameters
        ----------
        type:
            One of `"response"`, `"pearson"`, `"deviance"`, or `"working"`.

        Returns
        -------
        NDArray
            Residual vector of shape `(n,)`.
        """
        self._check_fitted()
        y = self._model_matrix.response
        mu = self._fit_result.fitted_values
        type_lower = type.lower()

        if type_lower == "response":
            return y - mu

        if type_lower == "pearson":
            v = self._family.variance(mu)
            return (y - mu) / np.sqrt(v)

        if type_lower == "deviance":
            d = self._family.unit_deviance(y, mu)
            return np.sign(y - mu) * np.sqrt(np.maximum(d, 0.0))

        if type_lower == "working":
            dmu_deta = 1.0 / self._family.link_derivative(mu)
            return (y - mu) / dmu_deta

        raise ValueError(
            f"Unknown residual type {type!r}. "
            "Choose from 'response', 'pearson', 'deviance', or 'working'."
        )

    @property
    def smoothing_params(self) -> list[float]:
        r"""Selected or fixed smoothing parameters $\lambda_j$, one per penalty.

        If `fit()` was called with `smoothing_params=None` (the default), these are the values
        chosen automatically via GCV, REML, or ML; otherwise they are the fixed values that were
        passed in. A `te()`/`t2()` term contributes more than one entry (one per marginal
        penalty), so this list is generally longer than the number of smooth terms.

        Returns
        -------
        list[float]
            One smoothing parameter per penalty, in the order the penalties were built.
        """
        self._check_fitted()
        return list(self._fit_result.smoothing_params)

    @property
    def edf(self) -> list[float]:
        """Effective degrees of freedom (EDF) for each smooth term.

        Each value is the trace of the portion of the hat matrix attributable to that term,
        reflecting how much shrinkage its smoothing parameter applied: values near the term's
        basis dimension indicate little penalization, values near 1 indicate near-linear
        shrinkage.

        Returns
        -------
        list[float]
            One EDF value per smooth term, in formula order.
        """
        self._check_fitted()
        return list(self._fit_result.edf)

    @property
    def edf_total(self) -> float:
        """Total effective degrees of freedom across all model terms.

        The sum of the per-term EDF values (plus the intercept and parametric terms), i.e. the
        trace of the full hat (influence) matrix. Used in `summary()`, `gam_check()`, and in
        computing residual degrees of freedom for interval and test calculations.

        Returns
        -------
        float
            Total EDF of the fitted model.
        """
        self._check_fitted()
        return self._fit_result.edf_total

    @property
    def scale(self) -> float:
        r"""Estimated scale (dispersion) parameter $\phi$.

        For families with a known scale (Binomial, Poisson) this is fixed at `1.0`. For families
        with unknown scale (Gaussian, Gamma, Tweedie) it is estimated from the Pearson residuals
        and is used to scale coefficient standard errors and prediction intervals.

        Returns
        -------
        float
            Estimated (or fixed) dispersion parameter.
        """
        self._check_fitted()
        return self._fit_result.scale

    @property
    def deviance(self) -> float:
        """Model deviance at convergence.

        Twice the difference between the saturated log-likelihood and the fitted model's
        log-likelihood, evaluated at the final coefficients. Lower values indicate a better fit
        to the training data; compare against `null_deviance` via `deviance_explained`.

        Returns
        -------
        float
            Deviance of the fitted model.
        """
        self._check_fitted()
        if isinstance(self._fit_result, (VIResult, MCMCResult)):
            raise NotImplementedError("deviance is not computed for Bayesian fits.")
        return self._fit_result.deviance

    @property
    def gcv_score(self) -> float:
        r"""Generalized Cross-Validation score at the fitted smoothing parameters.

        Computed as $\text{GCV} = n \cdot D / (n - \text{tr}(\mathbf{H}))^2$, where $D$ is the
        deviance and $\mathbf{H}$ is the hat matrix. This is the criterion minimized when
        `fit(method="GCV")` selects smoothing parameters, and is reported even when a different
        `method` was used.

        Returns
        -------
        float
            GCV score of the fitted model.
        """
        self._check_fitted()
        if isinstance(self._fit_result, (VIResult, MCMCResult)):
            raise NotImplementedError("gcv_score is not computed for Bayesian fits.")
        return self._fit_result.gcv_score

    @property
    def null_deviance(self) -> float:
        """Deviance of the intercept-only (null) model.

        Fit on the same data and with the same family and weights, but with every covariate
        effect (smooth and parametric) removed. Serves as the baseline against which
        `deviance_explained` measures the reduction in deviance achieved by the fitted model.

        Returns
        -------
        float
            Deviance of the intercept-only model.
        """
        self._check_fitted()
        if isinstance(self._fit_result, (VIResult, MCMCResult)):
            raise NotImplementedError("null_deviance is not computed for Bayesian fits.")
        assert self._fit_result.null_deviance is not None
        return self._fit_result.null_deviance

    @property
    def deviance_explained(self) -> float:
        """Proportion of null deviance explained by the model (analogous to R²).

        Computed as `1 - deviance / null_deviance`. Ranges from `0` (no improvement over an
        intercept-only model) up to `1` (a perfect fit), and provides a family-agnostic measure
        of goodness of fit that generalizes R² beyond the Gaussian case.

        Returns
        -------
        float
            Proportion of deviance explained, in `[0, 1]` for a sensible fit. Returns `0.0` if
            the null deviance is non-positive.
        """
        self._check_fitted()
        if isinstance(self._fit_result, (VIResult, MCMCResult)):
            raise NotImplementedError("deviance_explained is not computed for Bayesian fits.")
        null_dev = self._fit_result.null_deviance
        if null_dev is None or null_dev <= 0:
            return 0.0
        return 1.0 - self._fit_result.deviance / null_dev

    @property
    def aic(self) -> float:
        """Akaike Information Criterion of the fitted model.

        Balances goodness of fit against model complexity (using the total effective degrees of
        freedom in place of the raw parameter count). Lower values indicate a preferable
        trade-off; use it to compare non-nested models fitted to the same data and family.

        Returns
        -------
        float
            AIC of the fitted model.
        """
        self._check_fitted()
        if isinstance(self._fit_result, (VIResult, MCMCResult)):
            raise NotImplementedError("AIC is not computed for Bayesian fits.")
        assert self._fit_result.aic is not None
        return self._fit_result.aic

    @property
    def bic(self) -> float:
        """Bayesian Information Criterion of the fitted model.

        Like `aic`, but penalizes model complexity more heavily as sample size grows (using
        `log(n)` in place of `2` as the per-degree-of-freedom penalty), which tends to favor
        simpler models than AIC for larger datasets.

        Returns
        -------
        float
            BIC of the fitted model.
        """
        self._check_fitted()
        if isinstance(self._fit_result, (VIResult, MCMCResult)):
            raise NotImplementedError("BIC is not computed for Bayesian fits.")
        assert self._fit_result.bic is not None
        return self._fit_result.bic

    @property
    def vi_result(self) -> VIResult | None:
        """The `VIResult` when the model was fitted with `method="VI"`, else `None`."""
        self._check_fitted()
        return self._fit_result if isinstance(self._fit_result, VIResult) else None

    @property
    def mcmc_result(self) -> MCMCResult | None:
        """The `MCMCResult` when the model was fitted with `method="MCMC"`, else `None`."""
        self._check_fitted()
        return self._fit_result if isinstance(self._fit_result, MCMCResult) else None

    def posterior_samples(self, n: int = 1000, *, seed: int | None = None) -> NDArray:
        """Draw coefficient vectors from the posterior.

        For VI fits, samples from the variational posterior `N(m, C)`.
        For Laplace fits (REML/GCV/ML), samples from `N(β̂, V_β)`.

        Parameters
        ----------
        n:
            Number of draws.
        seed:
            Random seed.

        Returns
        -------
        NDArray
            Shape `(p, n)`, where `p` is the number of model coefficients.
        """
        self._check_fitted()
        from whittaker.fitting.inference import _bayesian_covariance

        if isinstance(self._fit_result, (VIResult, MCMCResult)):
            return self._fit_result.draw(n, seed=seed)

        W = self._combined_weights()
        V_beta = _bayesian_covariance(
            self._model_matrix.X,
            self._model_matrix.penalties,
            self._fit_result.smoothing_params,
            self._fit_result.scale,
            W=W,
        )
        rng = np.random.default_rng(seed)
        V_beta = (V_beta + V_beta.T) * 0.5
        eigvals, eigvecs = np.linalg.eigh(V_beta)
        eigvals = np.maximum(eigvals, 0.0)
        L = eigvecs * np.sqrt(eigvals)[np.newaxis, :]
        z = rng.standard_normal((len(self._fit_result.coefficients), n))
        return self._fit_result.coefficients[:, np.newaxis] + L @ z

    def parametric_tests(self) -> list:
        """Compute Wald tests for parametric (non-smooth) coefficients.

        Uses the t-distribution for families with unknown scale (Gaussian, Gamma) and the
        z-distribution for known-scale families (Binomial, Poisson).

        Returns
        -------
        list[ParametricTestResult]
            One result per parametric coefficient (intercept + linear terms).
        """
        from whittaker.fitting.inference import parametric_tests

        self._check_fitted()
        return parametric_tests(
            self._require_fit_result(), self._model_matrix, self._family.scale_known
        )

    def smooth_tests(self) -> list:
        """Compute approximate p-values for all smooth terms.

        Uses the Wood (2013) approach: eigendecomposition of the Bayesian covariance block for each
        smooth, with a chi-squared reference distribution.

        Returns
        -------
        list[SmoothTestResult]
            One result per smooth term.
        """
        from whittaker.fitting.inference import smooth_tests

        self._check_fitted()
        return smooth_tests(self._require_fit_result(), self._model_matrix)

    def k_check(self, *, n_sim: int = 400) -> list:
        """Check basis dimension adequacy for each smooth term.

        For each smooth, computes a k-index based on the ratio of a neighbor-differencing variance
        estimate of the residuals (ordered by covariate) to the overall residual variance. A k-index
        well below 1 suggests that the basis dimension `k` may be too small. A simulation-based
        p-value is computed: low p-values indicate potential under-smoothing.

        Parameters
        ----------
        n_sim:
            Number of random permutations for the p-value simulation (default `400`).

        Returns
        -------
        list[KCheckResult]
            One result per smooth term.
        """
        from whittaker.fitting.inference import k_check

        self._check_fitted()
        resid = self.get_residuals("deviance")
        return k_check(
            self._require_fit_result(), self._model_matrix, self._data, resid, n_sim=n_sim
        )

    def gam_check(self, *, n_sim: int = 100) -> GamCheckResult:
        """Run all-in-one GAM diagnostics.

        Returns a `GamCheckResult` containing deviance residuals, fitted values, response values,
        basis dimension checks, and summary statistics. Print the result for a quick diagnostic
        summary.

        Parameters
        ----------
        n_sim:
            Number of permutations for the k-check p-values (default `100`).

        Returns
        -------
        GamCheckResult
            Diagnostic results with a readable `__repr__`.
        """
        self._check_fitted()
        return GamCheckResult(
            deviance_residuals=self.get_residuals("deviance"),
            fitted_values=self._fit_result.fitted_values.copy(),
            response=self._model_matrix.response.copy(),
            k_check=self.k_check(n_sim=n_sim),
            deviance_explained=self.deviance_explained,
            scale=self.scale,
            edf_total=self._fit_result.edf_total,
            n_obs=self._model_matrix.X.shape[0],
        )

    def simulate(
        self,
        new_data: InputData | None = None,
        *,
        n_sim: int = 1000,
        seed: int | None = None,
        unconditional: bool = False,
    ) -> NDArray:
        """Draw from the posterior distribution of the fitted model.

        Generates posterior simulations of the response by:

        1. Drawing coefficient vectors β* ~ MVN(β̂, V_β) from the Bayesian posterior.
        2. Computing η* = X @ β* (+ offset if present) for each draw.
        3. Transforming to the response scale: μ* = g⁻¹(η*).

        When `unconditional=True`, response noise is added by sampling from the family distribution
        at each μ*.

        Parameters
        ----------
        new_data:
            Column-oriented data for prediction. If `None`, uses the training data.
        n_sim:
            Number of posterior draws (default `1000`).
        seed:
            Random seed for reproducibility.
        unconditional:
            If `True`, add response-distribution noise on top of posterior uncertainty in the mean.
            This produces simulations of new observations rather than of the conditional mean.

        Returns
        -------
        NDArray
            Simulated values on the response scale, shape `(n, n_sim)` where `n` is the number of
            observations in the prediction data.
        """
        self._check_fitted()
        from whittaker.fitting.inference import _bayesian_covariance

        rng = np.random.default_rng(seed)

        if new_data is None:
            X_new = self._model_matrix.X
            offset = self._model_matrix.offset
        else:
            new_data = prepare_data(new_data)
            X_new = predict_matrix(self._model_matrix, new_data)
            offset = predict_offset(self._model_matrix, new_data)

        beta_hat = self._fit_result.coefficients

        if isinstance(self._fit_result, (VIResult, MCMCResult)):
            # Draw directly from the Bayesian posterior (variational or MCMC samples)
            beta_draws = self._fit_result.draw(n_sim, seed=seed)
        else:
            W = self._combined_weights()
            V_beta = _bayesian_covariance(
                self._model_matrix.X,
                self._model_matrix.penalties,
                self._fit_result.smoothing_params,
                self._fit_result.scale,
                W=W,
            )
            V_beta = (V_beta + V_beta.T) * 0.5

            eigvals, eigvecs = np.linalg.eigh(V_beta)
            eigvals = np.maximum(eigvals, 0.0)
            L_draws = eigvecs * np.sqrt(eigvals)[np.newaxis, :]
            z = rng.standard_normal((len(beta_hat), n_sim))
            beta_draws = beta_hat[:, np.newaxis] + L_draws @ z

        eta = X_new @ beta_draws
        if offset is not None:
            eta += offset[:, np.newaxis]

        mu = self._family.link_inverse(eta)

        if unconditional:
            result = np.empty_like(mu)
            for j in range(n_sim):
                result[:, j] = self._family.simulate(mu[:, j], self._fit_result.scale, rng)
            return result

        return mu

    def concurvity(self, *, full: bool = True) -> object:
        """Compute concurvity diagnostics for all smooth terms.

        Concurvity is the GAM analogue of collinearity. High values (> 0.8) indicate that a smooth's
        effect may be confounded with other model terms, making its estimate unstable.

        Parameters
        ----------
        full:
            If `True` (default), measure each smooth against all other model terms combined. If
            `False`, compute pairwise concurvity between each pair of smooths.

        Returns
        -------
        ConcurvityResult
            Object with `worst`, `observed`, and `estimate` arrays, plus `labels`.
        """
        from whittaker.fitting.inference import concurvity

        self._check_fitted()
        return concurvity(self._require_fit_result(), self._model_matrix, full=full)

    def anova(self, *others: GAM) -> object:
        """Compare this model with one or more other fitted GAMs via deviance-difference tests.

        All models must use the same family and be fitted to the same data. Models are automatically
        sorted by complexity (edf). For known-scale families (Poisson, Binomial) a chi-squared test
        is used. For unknown-scale families (Gaussian, Gamma) an F-test is used.

        Parameters
        ----------
        *others:
            One or more fitted `GAM` objects to compare against this model.

        Returns
        -------
        AnovaResult
            Sequential deviance-comparison table.
        """
        from whittaker.fitting.inference import anova_gam

        self._check_fitted()
        all_gams = [self, *others]
        for g in others:
            if not g._fitted:
                raise RuntimeError("All models must be fitted before calling anova().")
            if type(g._family) is not type(self._family):
                raise ValueError(
                    f"All models must use the same family. Got {self._family!r} and {g._family!r}."
                )

        model_pairs = tuple((g._fit_result, g._model_matrix) for g in all_gams)
        return anova_gam(*model_pairs, scale_known=self._family.scale_known)

    def summary(self) -> ModelSummary:
        """Return a text summary of the fitted model, analogous to `summary.gam()` in R's mgcv.

        The summary reports, in order: the formula and family; a table of parametric
        (non-smooth) coefficients with estimates, standard errors, test statistics (`t` for
        unknown-scale families such as Gaussian and Gamma, `z` for known-scale families such as
        Binomial and Poisson), and p-values; a table of approximate significance for each smooth
        term (effective degrees of freedom, reference degrees of freedom, a chi-squared-type
        statistic, and a p-value); and overall fit statistics (total EDF, deviance, null
        deviance, proportion of deviance explained, GCV score, estimated scale, AIC, and BIC).
        Use this for a quick, human-readable check of which terms are significant and how well
        the model fits, without extracting individual result objects via `parametric_tests()`
        and `smooth_tests()`.

        Returns
        -------
        ModelSummary
            Multi-line text summary. Displays cleanly as the last expression in a
            Jupyter or Quarto cell without needing `print()`.
        """
        self._check_fitted()
        r = self._fit_result
        mm = self._model_matrix

        if isinstance(r, VIResult):
            inference_label = "Variational Bayes"
        elif isinstance(r, MCMCResult):
            _sampler = "NUTS" if r.mean_tree_depth > 0.0 else "HMC"
            inference_label = f"MCMC ({_sampler}, {r.n_chains} chains × {r.n_samples} draws)"
        else:
            inference_label = r.method
        lines = [
            "GAM fit summary",
            "=" * 60,
            f"Formula:    {self._formula!r}",
            f"Family:     {self._family!r}",
            f"Inference:  {inference_label}",
            f"Observations: {mm.n_obs}",
            f"Coefficients: {mm.n_coefs}",
        ]

        if not isinstance(r, (VIResult, MCMCResult)):
            ptests = self.parametric_tests()
            if ptests:
                stat_label = "z value" if self._family.scale_known else "t value"
                lines.extend(
                    [
                        "",
                        "Parametric coefficients:",
                        f"  {'Term':<24} {'Estimate':>10} {'Std.Err':>10} "
                        f"{stat_label:>10} {'p-value':>10}",
                        f"  {'-' * 24} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}",
                    ]
                )
                for pt in ptests:
                    pval_str = f"{pt.p_value:.4g}" if pt.p_value >= 1e-16 else "< 1e-16"
                    lines.append(
                        f"  {pt.term_label:<24} {pt.estimate:>10.4f} {pt.se:>10.4f} "
                        f"{pt.stat:>10.3f} {pval_str:>10}"
                    )

            stests = self.smooth_tests()
            lines.extend(
                [
                    "",
                    "Approximate significance of smooth terms:",
                    f"  {'Term':<24} {'EDF':>6} {'Ref.df':>6} {'Chi.sq':>10} {'p-value':>10}",
                    f"  {'-' * 24} {'-' * 6} {'-' * 6} {'-' * 10} {'-' * 10}",
                ]
            )

            for test in stests:
                pval_str = f"{test.p_value:.4g}" if test.p_value >= 1e-16 else "< 1e-16"
                lines.append(
                    f"  {test.term_label:<24} {test.edf:>6.2f} {test.ref_df:>6.0f} "
                    f"{test.stat:>10.3f} {pval_str:>10}"
                )

        lines.append("")
        lines.append(f"Total EDF:  {r.edf_total:.2f}")
        lines.append(f"Scale est:  {r.scale:.6f}")
        if isinstance(r, VIResult):
            lines.append(f"ELBO:       {r.elbo:.4f}")
            lines.append(f"VI iters:   {r.n_iter}")
            lines.append(f"Converged:  {r.converged}")
        elif isinstance(r, MCMCResult):
            n_total = r.n_chains * r.n_samples
            mcmc_lines = [
                f"Draws:      {r.n_chains} chains × {r.n_samples} = {n_total}",
                f"Warmup:     {r.n_warmup} per chain",
                f"Acceptance: {r.acceptance_rate:.3f}",
                f"Step size:  {r.step_size:.5f}",
                f"Max R-hat:  {r.r_hat.max():.4f}",
                f"Min ESS:    {r.ess.min():.0f}",
            ]
            if r.mean_tree_depth > 0.0:
                mcmc_lines.append(f"Tree depth: {r.mean_tree_depth:.2f} (mean)")
            if r.n_divergent > 0:
                mcmc_lines.append(
                    f"Divergent:  {r.n_divergent} transitions"
                    " — raise target_accept or reparameterize"
                )
            lines.extend(mcmc_lines)
        else:
            dev_expl = self.deviance_explained
            lines.extend(
                [
                    f"Deviance:   {r.deviance:.4f}",
                    f"Null dev:   {r.null_deviance:.4f}",
                    f"Dev. expl:  {dev_expl:.1%}",
                    f"GCV score:  {r.gcv_score:.6f}",
                    f"AIC:        {r.aic:.2f}",
                    f"BIC:        {r.bic:.2f}",
                ]
            )

        text = "\n".join(lines)
        return ModelSummary(text)

    def plot(
        self,
        *,
        n_points: int = 200,
        level: float = 0.95,
    ) -> object:
        """Plot the estimated partial effect of each smooth term, with a confidence band.

        For every smooth in the formula, evaluates the term's contribution to the linear predictor
        over an evenly spaced grid spanning its covariate's observed range (holding other terms out,
        i.e., this shows the additive component `f_k(x)` itself, not the full fitted response),
        together with a pointwise confidence band derived from the model's coefficient covariance.
        This is the standard way to visually inspect the *shape* of each estimated smooth (e.g.,
        whether it is roughly linear, monotonic, or has a distinct peak) without needing to call
        `predict(type="terms")` and plot manually.

        Parameters
        ----------
        n_points : int
            Number of evenly spaced evaluation points per smooth (the default is `200`).
        level : float
            Confidence level for the bands, e.g. `0.95` for a 95% band (the default is `0.95`).

        Returns
        -------
        altair.VConcatChart or altair.Chart
            A vertically concatenated chart with one panel per smooth term (or a single `Chart` if
            the model has exactly one smooth).

        Examples
        --------
        ```{python}
        import numpy as np
        from whittaker import GAM

        rng = np.random.default_rng(0)
        x = np.sort(rng.uniform(0, 10, 200))
        y = np.sin(x) + rng.normal(scale=0.2, size=200)

        gam = GAM("y ~ s(x)").fit({"x": x, "y": y})
        gam.plot()
        ```
        """
        from whittaker.plotting import partial_effects

        return partial_effects(self, n_points=n_points, level=level)

    def influence(self) -> object:
        """Compute hat values and Cook's distance for each observation.

        Returns
        -------
        InfluenceResult
            Object with `hat_values` and `cooks_distance` arrays.
        """
        from whittaker.fitting.inference import influence

        self._check_fitted()
        return influence(self._require_fit_result(), self._model_matrix)

    def quantile_residuals(self, *, seed: int | None = None) -> NDArray:
        """Compute randomized quantile residuals (Dunn & Smyth 1996).

        For a correctly specified model, these should be approximately standard normal.

        Parameters
        ----------
        seed:
            Random seed for the jittering step (discrete families).

        Returns
        -------
        NDArray
            Quantile residuals, shape `(n,)`.
        """
        from whittaker.fitting.inference import quantile_residuals

        self._check_fitted()
        return quantile_residuals(
            self._require_fit_result(), self._model_matrix, self._family, seed=seed
        )

    def dispersion_test(self) -> object:
        """Test for overdispersion in Poisson or Binomial models.

        Returns
        -------
        DispersionTestResult
            Object with `dispersion`, `chi2_stat`, and `p_value`.
        """
        from whittaker.fitting.inference import dispersion_test

        self._check_fitted()
        return dispersion_test(self._require_fit_result(), self._model_matrix, self._family)

    def vif(self) -> list:
        """Compute variance inflation factors for parametric (linear) terms.

        Returns
        -------
        list[VIFResult]
            One result per parametric term. Empty if fewer than 2 parametric terms.
        """
        from whittaker.fitting.inference import vif

        self._check_fitted()
        return vif(self._model_matrix)

    def derivatives(
        self,
        variable: str,
        *,
        order: int = 1,
        n_points: int = 200,
        level: float = 0.95,
        eps: float | None = None,
        unconditional: bool = False,
    ) -> list:
        """Estimate derivatives of smooth terms with respect to a variable.

        Uses central finite differences on the basis matrix with delta-method standard errors.

        Parameters
        ----------
        variable:
            The covariate to differentiate with respect to.
        order:
            Derivative order: `1` for first derivative (rate of change), `2` for second derivative
            (curvature).
        n_points:
            Number of evaluation points along the variable's range.
        level:
            Confidence level for the bands.
        eps:
            Finite difference step size. If `None`, chosen automatically.
        unconditional:
            If `True`, use unconditional covariance (Marra & Wood 2012).

        Returns
        -------
        list[DerivativeResult]
            One result per smooth term involving the variable.
        """
        from whittaker.fitting.inference import smooth_derivatives

        self._check_fitted()
        V = self._covariance_matrix(unconditional=unconditional)
        return smooth_derivatives(
            self._require_fit_result(),
            self._model_matrix,
            variable,
            self._data,
            order=order,
            n_points=n_points,
            level=level,
            eps=eps,
            V_beta=V,
        )

    def marginal_effects(
        self,
        variable: str,
        *,
        at: dict | None = None,
        n_points: int = 200,
        level: float = 0.95,
        unconditional: bool = False,
    ) -> list:
        """Compute marginal (partial) effects of a variable.

        Evaluates the smooth term(s) involving *variable* over a grid while
        holding other variables at their means or at values specified via *at*.

        Parameters
        ----------
        variable:
            The focal covariate.
        at:
            Dict mapping other variable names to fixed values (or lists of
            values for a grid). Variables not listed are held at their mean.
        n_points:
            Number of evaluation points along the variable's range.
        level:
            Confidence level for the bands.
        unconditional:
            If `True`, use unconditional covariance.

        Returns
        -------
        list[MarginalEffectResult]
            One result per smooth term per `at` combination.
        """
        from whittaker.fitting.inference import marginal_effects

        self._check_fitted()
        V = self._covariance_matrix(unconditional=unconditional)
        return marginal_effects(
            self._require_fit_result(),
            self._model_matrix,
            variable,
            self._data,
            at=at,
            n_points=n_points,
            level=level,
            V_beta=V,
        )

    def pairwise_comparisons(
        self,
        variable: str,
        pairs: list[tuple[dict, dict]],
        *,
        n_points: int = 200,
        level: float = 0.95,
        unconditional: bool = False,
    ) -> list:
        """Compute pairwise contrasts between conditions.

        Each pair is `(condition1, condition2)` where each condition is a dict of covariate values.
        The contrast `f(x|cond1) - f(x|cond2)` is evaluated over a grid of the focal variable.

        Parameters
        ----------
        variable:
            The focal covariate (the x-axis for the contrast).
        pairs:
            List of `(cond1, cond2)` dicts specifying the two conditions.
        n_points:
            Number of evaluation points.
        level:
            Confidence level for the bands.
        unconditional:
            If `True`, use unconditional covariance.

        Returns
        -------
        list[ContrastResult]
            One result per smooth term per pair.
        """
        from whittaker.fitting.inference import pairwise_comparisons

        self._check_fitted()
        V = self._covariance_matrix(unconditional=unconditional)
        return pairwise_comparisons(
            self._require_fit_result(),
            self._model_matrix,
            variable,
            self._data,
            pairs,
            n_points=n_points,
            level=level,
            V_beta=V,
        )

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("This GAM has not been fitted yet. Call .fit(data) first.")

    def _require_fit_result(self) -> FitResult:
        """Return `_fit_result` narrowed to `FitResult`, raising for Bayesian fits."""
        if isinstance(self._fit_result, (VIResult, MCMCResult)):
            raise NotImplementedError(
                f"This method is not available for {self._fit_result.method} fits."
            )
        return self._fit_result

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return f"GAM({self._formula!r}, family={self._family!r}, {status})"
