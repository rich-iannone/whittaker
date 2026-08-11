"""Top-level GAM class: the primary user-facing API."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from whittaker.data import InputData, prepare_data
from whittaker.families.base import Family
from whittaker.families.gaussian import Gaussian
from whittaker.families.tweedie_estimated import TweedieEstimated
from whittaker.fitting.pirls import FitResult, pirls_fit
from whittaker.formula.parser import parse
from whittaker.formula.terms import Formula
from whittaker.model_matrix import ModelMatrix, build_model_matrix, predict_matrix, predict_offset


@dataclass
class PredictionResult:
    """Result of `GAM.predict()` with optional standard errors and intervals.

    Attributes
    ----------
    values:
        Predicted values on the response scale, shape `(n,)`.
    se:
        Standard errors of the predictions (on the linear predictor scale), shape `(n,)`, or `None`
        if `se=False`.
    linear_predictor:
        Predictions on the linear predictor scale, shape `(n,)`.
    lower:
        Lower bound of the interval on the response scale, or `None` if no interval requested.
    upper:
        Upper bound of the interval on the response scale, or `None` if no interval requested.
    """

    values: NDArray
    se: NDArray | None
    linear_predictor: NDArray
    lower: NDArray | None = None
    upper: NDArray | None = None


@dataclass
class TermsPredictionResult:
    """Result of `GAM.predict(type="terms")`.

    Each smooth term's contribution to the linear predictor is returned separately, along with an
    optional standard error per term.

    Attributes
    ----------
    terms:
        Dict mapping term labels to their contributions, each shape `(n,)`.
    se:
        Dict mapping term labels to standard errors, each shape `(n,)`. `None` if `se=False`.
    labels:
        Term labels in formula order.
    """

    terms: dict[str, NDArray]
    se: dict[str, NDArray] | None
    labels: list[str] = field(default_factory=list)


@dataclass
class GamCheckResult:
    """Result of `GAM.gam_check()` with diagnostic information.

    Attributes
    ----------
    deviance_residuals:
        Deviance residuals, shape `(n,)`.
    fitted_values:
        Fitted values μ on the response scale, shape `(n,)`.
    response:
        Response values y, shape `(n,)`.
    k_check:
        Basis dimension check results (list of `KCheckResult`).
    deviance_explained:
        Proportion of deviance explained.
    scale:
        Estimated scale parameter.
    edf_total:
        Total effective degrees of freedom.
    n_obs:
        Number of observations.
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


class GAM:
    """Generalized Additive Model.

    Parameters
    ----------
    formula:
        Model formula as a string (e.g. `"y ~ s(x1) + s(x2) + x3"`).
    family:
        Response distribution family. Defaults to `Gaussian()` (identity link).
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
        self._fit_result: FitResult

    @property
    def formula(self) -> Formula:
        """The parsed model formula."""
        return self._formula

    @property
    def family(self) -> Family:
        """The response distribution family."""
        return self._family

    @property
    def is_fitted(self) -> bool:
        """`True` after `fit()` has been called."""
        return self._fitted

    def fit(
        self,
        data: InputData,
        *,
        smoothing_params: list[float] | None = None,
        method: str = "GCV",
        weights: NDArray | None = None,
        select: bool = False,
    ) -> GAM:
        """Fit the GAM to data.

        Parameters
        ----------
        data:
            Column-oriented data as `{name: 1-D array}`. All columns referenced by the formula must
            be present.
        smoothing_params:
            Fixed smoothing parameters, one per smooth term. If `None`, smoothing parameters are
            selected automatically via *method*.
        method:
            Smoothing parameter selection: `"GCV"`, `"REML"`, or `"ML"`.
        weights:
            Observation (prior) weights, shape `(n,)`. Must be positive. When provided, the model
            minimizes the weighted deviance `sum(w_i * d_i)` and uses weighted IRLS.
        select:
            If `True`, add an extra penalty on each smooth's null space so that terms can be
            penalized to zero entirely (double penalty approach). This enables automatic smooth
            selection: irrelevant smooths are shrunk out of the model. Recommended with
            `method="REML"`.

        Returns
        -------
        GAM
            Returns `self` for method chaining.
        """
        data = prepare_data(data)
        self._data = data

        if hasattr(self._family, "set_data"):
            self._family.set_data(data)

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

        if isinstance(self._family, TweedieEstimated) and not self._family.p_estimated:
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
        """Predict on new data.

        Parameters
        ----------
        new_data:
            Column-oriented new data. Must contain all covariate columns referenced by the formula
            (response column is not needed).
        se:
            If `True`, compute standard errors on the linear predictor scale (for `type="response"`
            and `"link"`) or per-term standard errors (for `type="terms"`).
        type:
            Prediction type:

            - `"response"` (default): predictions on the response scale (mu = g^{-1}(eta)).
            - `"link"`: predictions on the linear predictor scale (eta = X beta).
            - `"terms"`: individual smooth term contributions to the linear predictor.
        interval:
            Interval type. `None` (default) returns no intervals. `"confidence"` computes
            intervals for the mean response (uncertainty in eta only). `"prediction"` computes
            intervals for a new observation (adds response-distribution variance). Intervals are
            computed on the linear predictor scale and transformed to the response scale. Not
            available for `type="terms"`.
        level:
            Nominal coverage probability for the interval (default `0.95`).
        unconditional:
            If `True`, include smoothing parameter uncertainty in standard errors and intervals
            (Marra & Wood 2012). This uses the unconditional covariance matrix V_c instead of the
            conditional V_p, producing wider and more honest intervals. Requires that the model was
            fitted with `method="REML"` or `method="ML"`.

        Returns
        -------
        PredictionResult | TermsPredictionResult
            For `type="response"` or `"link"`, a `PredictionResult`. For `type="terms"`, a
            `TermsPredictionResult` with per-smooth contributions.
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
        se_dict: dict[str, NDArray] = {} if se else None
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

            if se:
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
        """Estimated coefficients β."""
        self._check_fitted()
        return self._fit_result.coefficients.copy()

    @property
    def fitted_values(self) -> NDArray:
        """Fitted values μ on the response scale."""
        self._check_fitted()
        return self._fit_result.fitted_values.copy()

    @property
    def residuals(self) -> NDArray:
        """Response residuals (y − μ)."""
        self._check_fitted()
        return self._fit_result.residuals.copy()

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
            eta = self._fit_result.linear_predictor
            dmu_deta = 1.0 / self._family.link_derivative(mu)
            return (y - mu) / dmu_deta

        raise ValueError(
            f"Unknown residual type {type!r}. "
            "Choose from 'response', 'pearson', 'deviance', or 'working'."
        )

    @property
    def smoothing_params(self) -> list[float]:
        """Selected or fixed smoothing parameters λ_j."""
        self._check_fitted()
        return list(self._fit_result.smoothing_params)

    @property
    def edf(self) -> list[float]:
        """Effective degrees of freedom per smooth term."""
        self._check_fitted()
        return list(self._fit_result.edf)

    @property
    def edf_total(self) -> float:
        """Total model effective degrees of freedom."""
        self._check_fitted()
        return self._fit_result.edf_total

    @property
    def scale(self) -> float:
        """Estimated scale parameter φ."""
        self._check_fitted()
        return self._fit_result.scale

    @property
    def deviance(self) -> float:
        """Model deviance at convergence."""
        self._check_fitted()
        return self._fit_result.deviance

    @property
    def gcv_score(self) -> float:
        """GCV score at the fitted smoothing parameters."""
        self._check_fitted()
        return self._fit_result.gcv_score

    @property
    def null_deviance(self) -> float:
        """Null deviance (intercept-only model)."""
        self._check_fitted()
        return self._fit_result.null_deviance

    @property
    def deviance_explained(self) -> float:
        """Proportion of null deviance explained by the model (analogous to R²)."""
        self._check_fitted()
        null_dev = self._fit_result.null_deviance
        if null_dev <= 0:
            return 0.0
        return 1.0 - self._fit_result.deviance / null_dev

    @property
    def aic(self) -> float:
        """Akaike Information Criterion."""
        self._check_fitted()
        return self._fit_result.aic

    @property
    def bic(self) -> float:
        """Bayesian Information Criterion."""
        self._check_fitted()
        return self._fit_result.bic

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
        return parametric_tests(self._fit_result, self._model_matrix, self._family.scale_known)

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
        return smooth_tests(self._fit_result, self._model_matrix)

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
        return k_check(self._fit_result, self._model_matrix, self._data, resid, n_sim=n_sim)

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

        W = self._combined_weights()
        V_beta = _bayesian_covariance(
            self._model_matrix.X,
            self._model_matrix.penalties,
            self._fit_result.smoothing_params,
            self._fit_result.scale,
            W=W,
        )

        beta_hat = self._fit_result.coefficients
        V_beta = (V_beta + V_beta.T) * 0.5

        eigvals, eigvecs = np.linalg.eigh(V_beta)
        eigvals = np.maximum(eigvals, 0.0)
        L = eigvecs * np.sqrt(eigvals)[np.newaxis, :]

        z = rng.standard_normal((len(beta_hat), n_sim))
        beta_draws = beta_hat[:, np.newaxis] + L @ z

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
        return concurvity(self._fit_result, self._model_matrix, full=full)

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

    def summary(self) -> str:
        """Return a text summary of the fitted model."""
        self._check_fitted()
        r = self._fit_result
        mm = self._model_matrix

        lines = [
            "GAM fit summary",
            "=" * 60,
            f"Formula:    {self._formula!r}",
            f"Family:     {self._family!r}",
            f"Observations: {mm.n_obs}",
            f"Coefficients: {mm.n_coefs}",
        ]

        ptests = self.parametric_tests()
        if ptests:
            stat_label = "z value" if self._family.scale_known else "t value"
            lines.extend(
                [
                    "",
                    "Parametric coefficients:",
                    f"  {'Term':<24} {'Estimate':>10} {'Std.Err':>10} {stat_label:>10} {'p-value':>10}",
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

        dev_expl = self.deviance_explained
        lines.extend(
            [
                "",
                f"Total EDF:  {r.edf_total:.2f}",
                f"Deviance:   {r.deviance:.4f}",
                f"Null dev:   {r.null_deviance:.4f}",
                f"Dev. expl:  {dev_expl:.1%}",
                f"GCV score:  {r.gcv_score:.6f}",
                f"Scale est:  {r.scale:.6f}",
                f"AIC:        {r.aic:.2f}",
                f"BIC:        {r.bic:.2f}",
            ]
        )

        text = "\n".join(lines)
        return text

    def plot(
        self,
        *,
        n_points: int = 200,
        level: float = 0.95,
    ) -> object:
        """Plot partial effects with confidence bands for each smooth term.

        Parameters
        ----------
        n_points:
            Number of evenly spaced evaluation points per smooth.
        level:
            Confidence level for the bands (the default is `0.95`).

        Returns
        -------
        altair.VConcatChart or altair.Chart
            One panel per smooth term.
        """
        from whittaker.plotting import partial_effects

        return partial_effects(self, n_points=n_points, level=level)

    def check(self) -> object:
        """Produce GAM diagnostic plots (analogous to `mgcv::gam.check`).

        Returns a 2×2 panel: QQ plot of residuals, residuals vs fitted values, histogram of
        residuals, and response vs fitted values.

        Returns
        -------
        altair.VConcatChart
            A 2×2 diagnostic panel.
        """
        from whittaker.plotting import check as _check

        return _check(self)

    def influence(self) -> object:
        """Compute hat values and Cook's distance for each observation.

        Returns
        -------
        InfluenceResult
            Object with `hat_values` and `cooks_distance` arrays.
        """
        from whittaker.fitting.inference import influence

        self._check_fitted()
        return influence(self._fit_result, self._model_matrix)

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
        return quantile_residuals(self._fit_result, self._model_matrix, self._family, seed=seed)

    def dispersion_test(self) -> object:
        """Test for overdispersion in Poisson or Binomial models.

        Returns
        -------
        DispersionTestResult
            Object with `dispersion`, `chi2_stat`, and `p_value`.
        """
        from whittaker.fitting.inference import dispersion_test

        self._check_fitted()
        return dispersion_test(self._fit_result, self._model_matrix, self._family)

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
            self._fit_result,
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
            self._fit_result,
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
            self._fit_result,
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

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return f"GAM({self._formula!r}, family={self._family!r}, {status})"
