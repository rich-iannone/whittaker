r"""Non-crossing quantile GAM.

Provides `QuantileGAM`, which fits multiple quantile levels simultaneously and enforces the
non-crossing constraint: for tau_1 < tau_2 < ... < tau_k, the fitted quantile curves satisfy
q_{tau_1}(x) <= q_{tau_2}(x) <= ... <= q_{tau_k}(x) at all observed covariate values.

The approach fits each quantile via the standard ELF-based PIRLS and then projects the fitted values
onto the monotone cone (isotonic regression across quantile levels at each observation) after each
IRLS iteration. This follows the "stepwise projection" strategy of Bondell, Reich, & Wang (2010).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from whittaker.data import InputData, prepare_data
from whittaker.families.quantile import QuantileFamily
from whittaker.fitting.pirls import FitResult
from whittaker.formula.parser import parse
from whittaker.formula.terms import Formula
from whittaker.gam import GAM, PredictionResult
from whittaker.model_matrix import ModelMatrix, build_model_matrix


@dataclass
class QuantileGAMResult:
    r"""Result container for a fitted QuantileGAM.

    Bundles the per-quantile fitted `GAM` objects and coefficient vectors produced by
    `QuantileGAM.fit()`, keyed by the quantile level `tau` they estimate.

    Attributes
    ----------
    quantiles:
        Sorted quantile levels, each in `(0, 1)`.
    models:
        Dict mapping `tau -> fitted GAM`, one GAM per quantile level, each fit with the
        expectile/quantile loss family for that `tau`.
    coefficients:
        Dict mapping `tau -> coefficient vector`, the fitted basis coefficients for each quantile's
        model.
    """

    quantiles: list[float]
    models: dict[float, GAM]
    coefficients: dict[float, NDArray]


def _isotonic_projection(values: NDArray) -> NDArray:
    """Project a 1-D array onto the monotone non-decreasing cone.

    Uses the pool-adjacent-violators algorithm (PAVA).
    """
    n = len(values)
    result = values.copy()
    blocks = [[i] for i in range(n)]

    i = 0
    while i < len(blocks) - 1:
        block_mean = lambda b: np.mean(result[b])
        if block_mean(blocks[i]) > block_mean(blocks[i + 1]):
            merged = blocks[i] + blocks[i + 1]
            m = np.mean(result[merged])
            for idx in merged:
                result[idx] = m
            blocks[i] = merged
            blocks.pop(i + 1)
            if i > 0:
                i -= 1
        else:
            i += 1

    return result


def _enforce_non_crossing(
    fitted_values: dict[float, NDArray],
    quantiles: list[float],
) -> dict[float, NDArray]:
    """Enforce non-crossing by isotonic projection at each observation.

    For each data point i, the vector [q_{tau_1}(x_i), ..., q_{tau_k}(x_i)]
    is projected onto the monotone non-decreasing cone via PAVA.
    """
    n = len(next(iter(fitted_values.values())))
    k = len(quantiles)

    corrected = {tau: fitted_values[tau].copy() for tau in quantiles}

    for i in range(n):
        vals = np.array([fitted_values[tau][i] for tau in quantiles])
        if np.all(np.diff(vals) >= 0):
            continue
        proj = _isotonic_projection(vals)
        for j, tau in enumerate(quantiles):
            corrected[tau][i] = proj[j]

    return corrected


class QuantileGAM:
    r"""Non-crossing quantile GAM.

    Fits a separate additive quantile regression model for each requested quantile level `tau`, and
    enforces the natural ordering constraint that quantile curves must not cross: for
    `tau_1 < tau_2`, the fitted curve `q_{tau_1}(x)` must lie at or below `q_{tau_2}(x)` at every
    observed covariate combination. Ordinary quantile GAMs, fit independently for each `tau`, provide
    no such guarantee and can produce curves that cross, especially in regions with sparse data or
    heavy smoothing.

    Use `QuantileGAM` whenever you need multiple quantiles of a conditional distribution (e.g. to
    build a prediction interval or characterize skewness/heteroscedasticity) and want the estimated
    quantiles to respect the required monotone ordering in `tau`.

    Parameters
    ----------
    formula:
        Model formula (e.g. `"y ~ s(x)"`), shared across all quantile levels; only the loss function
        differs between them.
    quantiles:
        Quantile levels to fit. Must be in `(0, 1)` and will be sorted. Defaults to
        `[0.1, 0.25, 0.5, 0.75, 0.9]`.
    sigma:
        Bandwidth of the smoothed pinball ("extended log-F", ELF) loss used to approximate the
        non-differentiable quantile check loss. Smaller `sigma` more closely approximates the true
        quantile loss but can slow IRLS convergence; use `calibrate_sigma()` to select it via
        cross-validation.
    non_crossing:
        If `True` (default), enforce the non-crossing constraint via iterative isotonic projection.
        If `False`, quantiles are fit completely independently and may cross.

    Notes
    -----
    Each quantile is fit by minimizing a smoothed pinball loss (the ELF loss of Fasiolo et al. 2021),
    which approximates the quantile check function

    $$\rho_\tau(u) = u \, (\tau - \mathbb{1}[u < 0])$$

    with a twice-differentiable surrogate suitable for IRLS. After each round of fitting, the vector
    of fitted quantiles at every observation, `[q_{\tau_1}(x_i), \dots, q_{\tau_k}(x_i)]`, is checked
    for monotonicity; if it is violated anywhere, the vector is projected onto the monotone
    non-decreasing cone via the pool-adjacent-violators algorithm (PAVA), following the "stepwise
    projection" strategy of Bondell, Reich, & Wang (2010). The projected fitted values are then used
    to re-derive coefficients (via a least-squares refit against the corrected working response), and
    the cycle repeats for up to `max_iter` rounds or until no crossings remain.

    Examples
    --------
    ```{python}
    import numpy as np
    from whittaker.quantile_gam import QuantileGAM

    rng = np.random.default_rng(0)
    n = 500
    x = rng.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x) + rng.normal(scale=0.2 + 0.3 * x, size=n)

    model = QuantileGAM("y ~ s(x)", quantiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    model.fit({"x": x, "y": y})
    preds = model.predict({"x": x[:5]})  # dict of tau -> PredictionResult
    print(model.crossing_fraction())
    ```
    """

    def __init__(
        self,
        formula: str | Formula,
        quantiles: list[float] | None = None,
        *,
        sigma: float = 0.1,
        non_crossing: bool = True,
    ) -> None:
        if isinstance(formula, str):
            self._formula = parse(formula)
        else:
            self._formula = formula

        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

        for tau in quantiles:
            if not 0 < tau < 1:
                raise ValueError(f"All quantiles must be in (0, 1), got {tau}")

        self._quantiles = sorted(quantiles)
        self._sigma = sigma
        self._non_crossing = non_crossing
        self._fitted = False
        self._models: dict[float, GAM] = {}
        self._data: dict[str, NDArray] | None = None
        self._model_matrix: ModelMatrix | None = None

    @property
    def formula(self) -> Formula:
        return self._formula

    @property
    def quantiles(self) -> list[float]:
        return list(self._quantiles)

    @property
    def sigma(self) -> float:
        return self._sigma

    @property
    def non_crossing(self) -> bool:
        return self._non_crossing

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        data: InputData,
        *,
        method: str = "REML",
        max_iter: int = 5,
        select: bool = False,
    ) -> QuantileGAM:
        r"""Fit the quantile GAM.

        Fits a separate ELF-based `GAM` for every quantile level in `self.quantiles`. When
        `non_crossing=True`, this is repeated for up to `max_iter` rounds: after each round, the
        fitted quantile curves are checked for crossing violations and, if found, corrected via
        isotonic projection (see class Notes); the corrected values are used to re-derive each
        quantile model's coefficients before the next round of refitting.

        Parameters
        ----------
        data:
            Column-oriented data containing the response and all covariates in `formula`.
        method:
            Smoothing parameter selection method applied to every quantile's `GAM` fit: `"GCV"`,
            `"REML"` (default), or `"ML"`.
        max_iter:
            Number of non-crossing projection iterations. Each iteration fits all quantiles and
            projects to enforce ordering. Ignored when `non_crossing=False` (in which case a single
            independent fit per quantile is performed).
        select:
            If `True`, enable double-penalty variable selection for each quantile's `GAM`.

        Returns
        -------
        QuantileGAM
            Returns `self` for method chaining.
        """
        self._data = prepare_data(data)

        mm = build_model_matrix(self._formula, self._data, select=select)
        self._model_matrix = mm

        if not self._non_crossing:
            max_iter = 1

        for iteration in range(max_iter):
            for tau in self._quantiles:
                family = QuantileFamily(tau=tau, sigma=self._sigma)
                model = GAM(self._formula, family=family)
                model.fit(self._data, method=method, select=select)
                self._models[tau] = model

            if not self._non_crossing:
                break

            fitted = {tau: self._models[tau].predict(self._data).values for tau in self._quantiles}

            if _check_non_crossing(fitted, self._quantiles):
                break

            corrected = _enforce_non_crossing(fitted, self._quantiles)

            for tau in self._quantiles:
                model = self._models[tau]
                X = model._model_matrix.X
                corrected_eta = model._family.link(corrected[tau])
                offset = model._model_matrix.offset
                if offset is not None:
                    corrected_eta = corrected_eta - offset
                beta_new, _ = np.linalg.lstsq(X, corrected_eta, rcond=None)[:2]
                model._fit_result = FitResult(
                    coefficients=beta_new,
                    linear_predictor=X @ beta_new + (offset if offset is not None else 0),
                    fitted_values=corrected[tau],
                    smoothing_params=model._fit_result.smoothing_params,
                    scale=model._fit_result.scale,
                    gcv_score=model._fit_result.gcv_score,
                    edf=model._fit_result.edf,
                    edf_total=model._fit_result.edf_total,
                    deviance=model._fit_result.deviance,
                    n_iter=model._fit_result.n_iter,
                    converged=model._fit_result.converged,
                    hat_matrix_trace=model._fit_result.hat_matrix_trace,
                    residuals=model._model_matrix.response - corrected[tau],
                    weights=model._fit_result.weights,
                    prior_weights=model._fit_result.prior_weights,
                    null_deviance=model._fit_result.null_deviance,
                    aic=model._fit_result.aic,
                    bic=model._fit_result.bic,
                    method=model._fit_result.method,
                    pseudo_data=model._fit_result.pseudo_data,
                )

        self._fitted = True
        return self

    def predict(
        self,
        new_data: InputData,
        *,
        se: bool = False,
    ) -> dict[float, PredictionResult]:
        r"""Predict quantiles on new data.

        Predicts each quantile level's `GAM` independently on `new_data`, then, if
        `non_crossing=True`, re-applies the isotonic projection across quantile levels at each new
        observation so the returned predictions never cross, even if the fitted models happen to
        disagree slightly on out-of-sample covariate combinations.

        Parameters
        ----------
        new_data:
            New covariate data.
        se:
            If `True`, include standard errors for each quantile's linear predictor.

        Returns
        -------
        dict[float, PredictionResult]
            Dict mapping quantile level to predictions.
        """
        self._check_fitted()
        new_data = prepare_data(new_data)

        results: dict[float, PredictionResult] = {}
        for tau in self._quantiles:
            results[tau] = self._models[tau].predict(new_data, se=se)

        if self._non_crossing:
            fitted = {tau: results[tau].values for tau in self._quantiles}
            if not _check_non_crossing(fitted, self._quantiles):
                corrected = _enforce_non_crossing(fitted, self._quantiles)
                for tau in self._quantiles:
                    old = results[tau]
                    results[tau] = PredictionResult(
                        values=corrected[tau],
                        se=old.se,
                        linear_predictor=old.linear_predictor,
                        lower=old.lower,
                        upper=old.upper,
                    )

        return results

    def predict_interval(
        self,
        new_data: InputData,
        lower_tau: float | None = None,
        upper_tau: float | None = None,
    ) -> tuple[NDArray, NDArray]:
        """Return prediction interval from the lowest and highest quantiles.

        Parameters
        ----------
        new_data:
            New covariate data.
        lower_tau:
            Lower quantile level. Defaults to the smallest fitted quantile.
        upper_tau:
            Upper quantile level. Defaults to the largest fitted quantile.

        Returns
        -------
        tuple[NDArray, NDArray]
            `(lower, upper)` prediction bounds.
        """
        self._check_fitted()
        preds = self.predict(new_data)

        if lower_tau is None:
            lower_tau = self._quantiles[0]
        if upper_tau is None:
            upper_tau = self._quantiles[-1]

        if lower_tau not in preds:
            raise ValueError(f"Quantile {lower_tau} not fitted. Available: {self._quantiles}")
        if upper_tau not in preds:
            raise ValueError(f"Quantile {upper_tau} not fitted. Available: {self._quantiles}")

        return preds[lower_tau].values, preds[upper_tau].values

    def coverage(self, data: InputData | None = None) -> float:
        """Compute empirical coverage of the outermost quantile interval.

        Parameters
        ----------
        data:
            Data to evaluate on. Defaults to the training data.

        Returns
        -------
        float
            Fraction of observations within `[q_low, q_high]`.
        """
        self._check_fitted()
        if data is None:
            data = self._data

        data = prepare_data(data)
        y = data[self._formula.response]
        lower, upper = self.predict_interval(data)
        return float(np.mean((y >= lower) & (y <= upper)))

    def crossing_fraction(self, data: InputData | None = None) -> float:
        """Fraction of observations where quantile curves cross.

        Parameters
        ----------
        data:
            Data to evaluate on. Defaults to the training data.

        Returns
        -------
        float
            Fraction of observations with at least one crossing (0.0 if
            non-crossing constraint is satisfied everywhere).
        """
        self._check_fitted()
        if data is None:
            data = self._data

        data = prepare_data(data)
        preds = {tau: self._models[tau].predict(data).values for tau in self._quantiles}
        n = len(next(iter(preds.values())))
        violations = 0
        for i in range(n):
            vals = [preds[tau][i] for tau in self._quantiles]
            if any(vals[j] > vals[j + 1] + 1e-10 for j in range(len(vals) - 1)):
                violations += 1
        return violations / n

    def summary(self) -> str:
        """Return a text summary of the fitted quantile GAM."""
        self._check_fitted()
        lines = [
            "QuantileGAM summary",
            "=" * 60,
            f"Formula:      {self._formula!r}",
            f"Quantiles:    {self._quantiles}",
            f"Non-crossing: {self._non_crossing}",
            f"Sigma:        {self._sigma}",
            "",
        ]
        for tau in self._quantiles:
            m = self._models[tau]
            lines.append(f"  tau={tau:.2f}: edf={m.edf_total:.1f}, dev={m.deviance:.1f}")
        return "\n".join(lines)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("This QuantileGAM has not been fitted yet. Call .fit(data) first.")

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        taus = ", ".join(f"{t:.2f}" for t in self._quantiles)
        return f"QuantileGAM({self._formula!r}, quantiles=[{taus}], {status})"


def _check_non_crossing(
    fitted: dict[float, NDArray],
    quantiles: list[float],
) -> bool:
    """Check if fitted values satisfy the non-crossing constraint."""
    for i in range(len(quantiles) - 1):
        tau_lo = quantiles[i]
        tau_hi = quantiles[i + 1]
        if np.any(fitted[tau_lo] > fitted[tau_hi] + 1e-10):
            return False
    return True
