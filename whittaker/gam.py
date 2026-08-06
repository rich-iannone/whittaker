"""Top-level GAM class: the primary user-facing API."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from whittaker.families.base import Family
from whittaker.families.gaussian import Gaussian
from whittaker.fitting.pirls import FitResult, pirls_fit
from whittaker.formula.parser import parse
from whittaker.formula.terms import Formula
from whittaker.model_matrix import ModelMatrix, build_model_matrix, predict_matrix


@dataclass
class PredictionResult:
    """Result of `GAM.predict()` with optional standard errors.

    Attributes
    ----------
    values:
        Predicted values on the response scale, shape `(n,)`.
    se:
        Standard errors of the predictions (on the linear predictor scale), shape `(n,)`, or `None`
        if `se=False`.
    linear_predictor:
        Predictions on the linear predictor scale, shape `(n,)`.
    """

    values: NDArray
    se: NDArray | None
    linear_predictor: NDArray


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
        """``True`` after ``fit()`` has been called."""
        return self._fitted

    def fit(
        self,
        data: dict[str, NDArray],
        *,
        smoothing_params: list[float] | None = None,
        method: str = "GCV",
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
            Smoothing parameter selection: ``"GCV"`` or ``"REML"``.

        Returns
        -------
        GAM
            Returns ``self`` for method chaining.
        """
        self._model_matrix = build_model_matrix(self._formula, data)
        self._fit_result = pirls_fit(
            self._model_matrix,
            self._family,
            smoothing_params=smoothing_params,
            method=method,
        )
        self._fitted = True
        return self

    def predict(
        self,
        new_data: dict[str, NDArray],
        *,
        se: bool = False,
    ) -> PredictionResult:
        """Predict on new data.

        Parameters
        ----------
        new_data:
            Column-oriented new data. Must contain all covariate columns referenced by the formula
            (response column is not needed).
        se:
            If `True`, compute standard errors of the predictions on the linear predictor scale.

        Returns
        -------
        PredictionResult
            Predicted values (and optionally standard errors).
        """
        self._check_fitted()

        X_new = predict_matrix(self._model_matrix, new_data)
        eta = X_new @ self._fit_result.coefficients

        mu = self._family.link_inverse(eta)

        se_values = None
        if se:
            se_values = self._prediction_se(X_new)

        return PredictionResult(values=mu, se=se_values, linear_predictor=eta)

    def _prediction_se(self, X_new: NDArray) -> NDArray:
        """Compute standard errors for predictions at X_new.

        SE = sqrt(diag(X_new @ V_β @ X_new.T))

        where V_β = φ (X'X + Σλ_j S_j)⁻¹ is the Bayesian posterior covariance of the coefficients.
        """
        from scipy.linalg import cho_factor, cho_solve

        X = self._model_matrix.X
        sp = self._fit_result.smoothing_params
        penalties = self._model_matrix.penalties

        XtX = X.T @ X
        S_total = np.zeros_like(XtX)
        for lam, pen in zip(sp, penalties):
            S_total += lam * pen

        A = XtX + S_total
        A = (A + A.T) * 0.5

        cho, lower = cho_factor(A)

        # V_β = scale * A⁻¹
        # SE² = scale * diag(X_new @ A⁻¹ @ X_new.T)
        #      = scale * rowSums((X_new @ A⁻¹) * X_new)
        A_inv_Xt = cho_solve((cho, lower), X_new.T)  # (p, n_new)
        var_diag = np.sum(X_new * A_inv_Xt.T, axis=1) * self._fit_result.scale

        return np.sqrt(np.maximum(var_diag, 0.0))

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

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("This GAM has not been fitted yet. Call .fit(data) first.")

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return f"GAM({self._formula!r}, family={self._family!r}, {status})"
