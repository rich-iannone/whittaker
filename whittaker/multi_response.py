"""Multi-response GAMs.

Fits multiple response variables jointly, optionally sharing smooth terms across responses and
modeling residual correlations. Each response gets its own GAM, but shared smooths use the same
basis and coefficients across all responses, providing a parsimonious model for multivariate
outcomes.

Two correlation structures are supported:

- `"independent"` (default): responses are fitted independently (equivalent to fitting separate
GAMs, but with shared smooth enforcement).
- `"unstructured"`: estimates a full residual covariance matrix across responses and uses it for
joint GLS-style inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from whittaker.data import InputData, prepare_data
from whittaker.families.base import Family
from whittaker.families.gaussian import Gaussian
from whittaker.gam import GAM, PredictionResult


@dataclass
class MultiResponseResult:
    """Prediction result for multiple responses.

    Attributes
    ----------
    predictions:
        Dict mapping response name to its `PredictionResult`.
    responses:
        List of response names (ordered).
    """

    predictions: dict[str, PredictionResult]
    responses: list[str]

    def __getitem__(self, key: str) -> PredictionResult:
        return self.predictions[key]

    def __iter__(self):
        return iter(self.responses)


@dataclass
class ResidualCorrelation:
    """Estimated residual correlation structure.

    Attributes
    ----------
    covariance:
        Residual covariance matrix (k x k) where k = number of responses.
    correlation:
        Residual correlation matrix (k x k).
    responses:
        Response names (ordering matches matrix rows/columns).
    """

    covariance: NDArray
    correlation: NDArray
    responses: list[str]

    def __repr__(self) -> str:
        lines = ["ResidualCorrelation:"]
        k = len(self.responses)
        header = "         " + "  ".join(f"{r:>8s}" for r in self.responses)
        lines.append(header)
        for i in range(k):
            row = f"{self.responses[i]:>8s} " + "  ".join(
                f"{self.correlation[i, j]:8.3f}" for j in range(k)
            )
            lines.append(row)
        return "\n".join(lines)


class MultiResponseGAM:
    """Multi-response GAM.

    Fits multiple response variables jointly. Smooth terms can be shared across responses (same
    basis, same coefficients) or response-specific.

    Parameters
    ----------
    responses:
        List of response variable names.
    formula:
        Shared formula applied to all responses (e.g. `"s(x1) + s(x2)"`). The response side is
        ignored so use `responses` to specify responses.
    response_formulas:
        Dict mapping response name to a response-specific formula string (covariates only, e.g.,
        `{"y1": "s(x3)"}`) added on top of the shared formula.
    family:
        Response distribution family (applied to all responses). Defaults to `Gaussian()`.
    correlation:
        Residual correlation structure: `"independent"` (default) or `"unstructured"`.
    """

    def __init__(
        self,
        responses: list[str],
        formula: str,
        *,
        response_formulas: dict[str, str] | None = None,
        family: Family | None = None,
        correlation: str = "independent",
    ) -> None:
        if len(responses) < 2:
            raise ValueError("MultiResponseGAM requires at least 2 responses.")

        if len(set(responses)) != len(responses):
            raise ValueError("Response names must be unique.")

        if correlation not in ("independent", "unstructured"):
            raise ValueError(
                f"correlation must be 'independent' or 'unstructured', got {correlation!r}"
            )

        self._responses = list(responses)
        self._shared_formula = formula.split("~")[-1].strip()
        self._response_formulas = response_formulas or {}
        self._family = family if family is not None else Gaussian()
        self._correlation = correlation
        self._fitted = False

        self._models: dict[str, GAM] = {}
        self._residual_cov: NDArray | None = None
        self._residual_corr: NDArray | None = None
        self._data: dict[str, NDArray] | None = None

    @property
    def responses(self) -> list[str]:
        return list(self._responses)

    @property
    def n_responses(self) -> int:
        return len(self._responses)

    @property
    def correlation(self) -> str:
        return self._correlation

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        data: InputData,
        *,
        method: str = "REML",
        select: bool = False,
    ) -> MultiResponseGAM:
        """Fit the multi-response GAM.

        Parameters
        ----------
        data:
            Column-oriented data containing all response and covariate columns.
        method:
            Smoothing parameter selection method.
        select:
            Enable variable selection.

        Returns
        -------
        MultiResponseGAM
            Returns `self` for method chaining.
        """
        arrays = prepare_data(data)
        self._data = arrays

        for resp in self._responses:
            if resp not in arrays:
                raise ValueError(f"Response {resp!r} not found in data.")

        for resp in self._responses:
            resp_specific = self._response_formulas.get(resp, "")
            if resp_specific:
                formula_str = f"{resp} ~ {self._shared_formula} + {resp_specific}"
            else:
                formula_str = f"{resp} ~ {self._shared_formula}"

            model = GAM(formula_str, family=self._family)
            model.fit(arrays, method=method, select=select)
            self._models[resp] = model

        if self._correlation == "unstructured":
            self._estimate_correlation(arrays)

        self._fitted = True
        return self

    def _estimate_correlation(self, arrays: dict[str, NDArray]) -> None:
        k = len(self._responses)
        n = len(arrays[self._responses[0]])

        residuals = np.zeros((n, k))
        for j, resp in enumerate(self._responses):
            pred = self._models[resp].predict(arrays).values
            residuals[:, j] = arrays[resp] - pred

        self._residual_cov = (residuals.T @ residuals) / (n - 1)
        diag = np.sqrt(np.diag(self._residual_cov))
        outer = np.outer(diag, diag)
        outer[outer < 1e-10] = 1.0
        self._residual_corr = self._residual_cov / outer

    def predict(
        self,
        new_data: InputData,
        *,
        se: bool = False,
    ) -> MultiResponseResult:
        """Predict all responses on new data.

        Parameters
        ----------
        new_data:
            Column-oriented covariate data.
        se:
            If `True`, include standard errors.

        Returns
        -------
        MultiResponseResult
        """
        self._check_fitted()
        new_data = prepare_data(new_data)

        predictions = {}
        for resp in self._responses:
            predictions[resp] = self._models[resp].predict(new_data, se=se)

        return MultiResponseResult(
            predictions=predictions,
            responses=list(self._responses),
        )

    def residual_correlation(self) -> ResidualCorrelation:
        """Return the estimated residual correlation structure.

        Only available when `correlation="unstructured"`.

        Returns
        -------
        ResidualCorrelation
        """
        self._check_fitted()
        if self._correlation != "unstructured":
            raise ValueError(
                "Residual correlation not estimated. Refit with correlation='unstructured'."
            )
        return ResidualCorrelation(
            covariance=self._residual_cov.copy(),
            correlation=self._residual_corr.copy(),
            responses=list(self._responses),
        )

    def joint_predict(
        self,
        new_data: InputData,
    ) -> tuple[NDArray, NDArray | None]:
        """Predict all responses jointly, returning a matrix.

        Parameters
        ----------
        new_data:
            Column-oriented covariate data.

        Returns
        -------
        tuple[NDArray, NDArray | None]
            `(predictions, covariance)` where predictions is `(n, k)` and covariance is the `(k, k)`
            residual covariance matrix (`None` if `correlation="independent"`).
        """
        self._check_fitted()
        new_data = prepare_data(new_data)
        n = len(next(iter(new_data.values())))
        k = len(self._responses)

        preds = np.zeros((n, k))
        for j, resp in enumerate(self._responses):
            preds[:, j] = self._models[resp].predict(new_data).values

        cov = self._residual_cov if self._correlation == "unstructured" else None
        return preds, cov

    def response_model(self, response: str) -> GAM:
        """Return the fitted GAM for a specific response.

        Parameters
        ----------
        response:
            Response variable name.

        Returns
        -------
        GAM
        """
        self._check_fitted()
        if response not in self._models:
            raise ValueError(f"Response {response!r} not found. Available: {self._responses}")
        return self._models[response]

    def edf(self) -> dict[str, float]:
        """Return total EDF per response.

        Returns
        -------
        dict[str, float]
        """
        self._check_fitted()
        return {resp: self._models[resp].edf_total for resp in self._responses}

    def deviance(self) -> dict[str, float]:
        """Return deviance per response.

        Returns
        -------
        dict[str, float]
        """
        self._check_fitted()
        return {resp: self._models[resp].deviance for resp in self._responses}

    def summary(self) -> str:
        """Text summary of the multi-response GAM."""
        self._check_fitted()
        lines = [
            "MultiResponseGAM summary",
            "=" * 60,
            f"Responses:   {', '.join(self._responses)}",
            f"Shared:      {self._shared_formula}",
            f"Family:      {self._family.__class__.__name__}",
            f"Correlation: {self._correlation}",
            "",
            "Per-response fits:",
        ]
        for resp in self._responses:
            m = self._models[resp]
            lines.append(
                f"  {resp}: edf={m.edf_total:.1f}, dev={m.deviance:.1f}, scale={m.scale:.4f}"
            )

        if self._correlation == "unstructured" and self._residual_corr is not None:
            lines.append("")
            lines.append("Residual correlations:")
            k = len(self._responses)
            for i in range(k):
                for j in range(i + 1, k):
                    lines.append(
                        f"  corr({self._responses[i]}, {self._responses[j]}) = "
                        f"{self._residual_corr[i, j]:.3f}"
                    )

        return "\n".join(lines)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "This MultiResponseGAM has not been fitted yet. Call .fit(data) first."
            )

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        resps = ", ".join(self._responses)
        return f"MultiResponseGAM([{resps}], {status})"
