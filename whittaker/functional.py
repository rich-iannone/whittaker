"""Functional regression via GAMs.

Scalar-on-function regression where the response is scalar but predictors include functional
covariates (curves observed over a domain). Each functional term contributes a linear functional
effect: integral of X_i(t) * beta(t) dt, where beta(t) is a smooth coefficient function expanded in
a B-spline or Fourier basis.

The coefficient function beta(t) is penalized for roughness via integrated squared second
derivatives, giving a smooth estimate of how each point along the functional domain contributes to
the response.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from whittaker.data import InputData
from whittaker.families.base import Family
from whittaker.families.gaussian import Gaussian
from whittaker.fitting.pirls import FitResult, pirls_fit
from whittaker.formula.parser import parse
from whittaker.model_matrix import ModelMatrix, build_model_matrix, predict_matrix
from whittaker.smooths.pspline import _bspline_design, _bspline_knots, _diff_matrix


@dataclass
class _FunctionalSmoothInfo:
    """Lightweight stand-in for SmoothInfo compatible with pirls_fit."""

    col_start: int
    col_end: int
    null_space_dim: int = 0
    basis: object = None
    term: object = None
    penalty_indices: list = field(default_factory=list)
    by_var: str | None = None
    by_level: str | None = None


@dataclass
class FunctionalTerm:
    """Specification for a functional covariate.

    Attributes
    ----------
    name:
        Name of the functional covariate in the data dict. The corresponding data entry should be
        a 2-D array of shape `(n, T)` where `T` is the number of grid points.
    basis:
        Basis type for expanding beta(t): `"bspline"` (default) or `"fourier"`.
    domain:
        Tuple `(t_min, t_max)` specifying the domain of the functional argument. Grid points are
        assumed equally spaced over this domain.
    n_basis:
        Number of basis functions. Defaults to 15.
    penalty_order:
        Order of the difference penalty (for B-spline) or derivative penalty (for Fourier).
        Defaults to 2.
    """

    name: str
    basis: str = "bspline"
    domain: tuple[float, float] = (0.0, 1.0)
    n_basis: int = 15
    penalty_order: int = 2


@dataclass
class CoefficientFunction:
    """Estimated coefficient function beta(t) for a functional term.

    Attributes
    ----------
    grid:
        Evaluation grid on the functional domain, shape `(T,)`.
    values:
        Estimated beta(t) values at grid points, shape `(T,)`.
    se:
        Standard errors of beta(t), shape `(T,)`, or `None`.
    lower:
        Lower confidence bound, shape `(T,)`, or `None`.
    upper:
        Upper confidence bound, shape `(T,)`, or `None`.
    term_name:
        Name of the functional term.
    """

    grid: NDArray
    values: NDArray
    se: NDArray | None = None
    lower: NDArray | None = None
    upper: NDArray | None = None
    term_name: str = ""


def _fourier_basis(t: NDArray, n_basis: int, domain: tuple[float, float]) -> NDArray:
    """Evaluate Fourier basis functions on `t`.

    Returns an `(len(t), n_basis)` matrix. The first column is a constant (1/sqrt(L)),
    followed by sin/cos pairs at increasing frequencies.
    """
    t_min, t_max = domain
    L = t_max - t_min
    t_scaled = (t - t_min) / L

    B = np.zeros((len(t), n_basis))
    B[:, 0] = 1.0

    col = 1
    freq = 1
    while col < n_basis:
        B[:, col] = np.sin(2 * np.pi * freq * t_scaled)
        col += 1
        if col < n_basis:
            B[:, col] = np.cos(2 * np.pi * freq * t_scaled)
            col += 1
        freq += 1

    return B


def _fourier_penalty(n_basis: int, domain: tuple[float, float], order: int = 2) -> NDArray:
    """Penalty matrix for Fourier basis (diagonal, penalizing higher frequencies)."""
    L = domain[1] - domain[0]
    S = np.zeros((n_basis, n_basis))

    col = 1
    freq = 1
    while col < n_basis:
        omega = 2 * np.pi * freq / L
        pen = omega ** (2 * order)
        S[col, col] = pen
        col += 1
        if col < n_basis:
            S[col, col] = pen
            col += 1
        freq += 1

    return S


def _bspline_basis_and_penalty(
    t: NDArray,
    n_basis: int,
    domain: tuple[float, float],
    penalty_order: int = 2,
    degree: int = 3,
) -> tuple[NDArray, NDArray]:
    """Build B-spline basis matrix and difference penalty for the functional domain."""
    t_min, t_max = domain
    knots = _bspline_knots(t_min, t_max, n_basis, degree)
    B = _bspline_design(t, knots, degree)
    D = _diff_matrix(n_basis, penalty_order)
    S = D.T @ D
    return B, S


def _integration_weights(T: int, domain: tuple[float, float]) -> NDArray:
    """Trapezoidal quadrature weights for T equally-spaced points over domain."""
    dt = (domain[1] - domain[0]) / (T - 1)
    w = np.full(T, dt)
    w[0] = dt / 2
    w[-1] = dt / 2
    return w


def _build_functional_design(
    X_func: NDArray,
    basis_matrix: NDArray,
    weights: NDArray,
) -> NDArray:
    """Compute the functional design matrix via numerical integration.

    For each observation i and basis function k:
        J[i, k] = sum_t X_func[i, t] * basis_matrix[t, k] * weights[t]

    This approximates the integral of X_i(t) * phi_k(t) dt.
    """
    return (X_func * weights[np.newaxis, :]) @ basis_matrix


class FunctionalGAM:
    """Scalar-on-function GAM.

    Fits a model where the response is scalar and predictors include functional covariates
    (curves). Each functional covariate contributes a linear functional effect via numerical
    integration against a smooth coefficient function.

    Parameters
    ----------
    response:
        Name of the scalar response variable.
    functional_terms:
        List of `FunctionalTerm` specifications (or dicts with the same keys).
    scalar_terms:
        Optional formula string for scalar smooth/linear terms (e.g. `"s(temperature) + humidity"`).
    family:
        Response distribution family. Defaults to `Gaussian()`.
    """

    def __init__(
        self,
        response: str,
        functional_terms: list[FunctionalTerm | dict],
        *,
        scalar_terms: str | None = None,
        family: Family | None = None,
    ) -> None:
        if not functional_terms:
            raise ValueError("At least one functional term is required.")

        self._response = response
        self._functional_terms: list[FunctionalTerm] = []
        for ft in functional_terms:
            if isinstance(ft, dict):
                ft = FunctionalTerm(**ft)
            if ft.basis not in ("bspline", "fourier"):
                raise ValueError(
                    f"Unsupported basis {ft.basis!r} for term {ft.name!r}. "
                    f"Use 'bspline' or 'fourier'."
                )
            if ft.n_basis < 3:
                raise ValueError(f"n_basis must be >= 3, got {ft.n_basis} for term {ft.name!r}.")
            if ft.domain[0] >= ft.domain[1]:
                raise ValueError(
                    f"domain must have t_min < t_max, got {ft.domain} for term {ft.name!r}."
                )
            self._functional_terms.append(ft)

        self._scalar_terms = scalar_terms
        self._family = family if family is not None else Gaussian()
        self._fitted = False

        self._fit_result: FitResult | None = None
        self._basis_matrices: dict[str, NDArray] = {}
        self._penalty_matrices: dict[str, NDArray] = {}
        self._func_col_ranges: dict[str, tuple[int, int]] = {}
        self._total_cols: int = 0
        self._scalar_model_matrix: ModelMatrix | None = None
        self._n_intercept: int = 1
        self._func_domains: dict[str, NDArray] = {}

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def response(self) -> str:
        return self._response

    @property
    def functional_terms(self) -> list[FunctionalTerm]:
        return list(self._functional_terms)

    def fit(
        self,
        data: InputData,
        *,
        method: str = "REML",
        select: bool = False,
    ) -> FunctionalGAM:
        """Fit the functional GAM.

        Parameters
        ----------
        data:
            Column-oriented data. Scalar covariates and the response are 1-D arrays. Functional
            covariates are 2-D arrays of shape `(n, T)` where `T` is the number of grid points.
        method:
            Smoothing parameter selection method.
        select:
            Enable variable selection.

        Returns
        -------
        FunctionalGAM
            Returns `self` for method chaining.
        """
        arrays = self._prepare_functional_data(data)
        y = arrays[self._response]
        n = len(y)

        columns: list[NDArray] = [np.ones((n, 1))]
        col_names: list[str] = ["intercept"]
        penalties: list[NDArray] = []
        col_offset = 1

        for ft in self._functional_terms:
            X_func = arrays[ft.name]
            if X_func.ndim != 2:
                raise ValueError(
                    f"Functional covariate {ft.name!r} must be 2-D (n x T), "
                    f"got shape {X_func.shape}."
                )
            T = X_func.shape[1]
            if T < 3:
                raise ValueError(
                    f"Functional covariate {ft.name!r} must have at least 3 grid points, got {T}."
                )

            t_grid = np.linspace(ft.domain[0], ft.domain[1], T)
            self._func_domains[ft.name] = t_grid
            w = _integration_weights(T, ft.domain)

            if ft.basis == "bspline":
                B, S = _bspline_basis_and_penalty(t_grid, ft.n_basis, ft.domain, ft.penalty_order)
            else:
                B = _fourier_basis(t_grid, ft.n_basis, ft.domain)
                S = _fourier_penalty(ft.n_basis, ft.domain, ft.penalty_order)

            self._basis_matrices[ft.name] = B
            self._penalty_matrices[ft.name] = S

            J = _build_functional_design(X_func, B, w)

            J -= J.mean(axis=0, keepdims=True)

            k = ft.n_basis
            self._func_col_ranges[ft.name] = (col_offset, col_offset + k)
            columns.append(J)
            col_names.extend([f"{ft.name}_b{i}" for i in range(k)])

            S_full_placeholder = S
            penalties.append((col_offset, k, S_full_placeholder))
            col_offset += k

        scalar_offset = col_offset
        scalar_mm = None
        if self._scalar_terms is not None:
            scalar_formula = f"{self._response} ~ {self._scalar_terms}"
            scalar_data = {k: v for k, v in arrays.items() if v.ndim == 1}
            scalar_mm = build_model_matrix(parse(scalar_formula), scalar_data, select=select)
            self._scalar_model_matrix = scalar_mm

            scalar_X = scalar_mm.X[:, 1:] if scalar_mm.has_intercept else scalar_mm.X
            n_scalar_cols = scalar_X.shape[1]
            columns.append(scalar_X)

            scalar_names = (
                scalar_mm.column_names[1:] if scalar_mm.has_intercept else scalar_mm.column_names
            )
            col_names.extend(scalar_names)

            for pen in scalar_mm.penalties:
                if scalar_mm.has_intercept:
                    pen_block = pen[1:, 1:]
                else:
                    pen_block = pen
                penalties.append((scalar_offset, pen_block.shape[0], pen_block))

            col_offset += n_scalar_cols

        self._total_cols = col_offset

        X = np.column_stack(columns)
        p = X.shape[1]

        full_penalties = []
        for start, k, S_block in penalties:
            S_full = np.zeros((p, p))
            S_full[start : start + k, start : start + k] = S_block
            full_penalties.append(S_full)

        smooth_infos = []
        for ft in self._functional_terms:
            start, end = self._func_col_ranges[ft.name]
            smooth_infos.append(_FunctionalSmoothInfo(col_start=start, col_end=end))

        if scalar_mm is not None:
            for si in scalar_mm.smooths:
                offset_adj = scalar_offset - (1 if scalar_mm.has_intercept else 0)
                smooth_infos.append(
                    _FunctionalSmoothInfo(
                        col_start=si.col_start + offset_adj,
                        col_end=si.col_end + offset_adj,
                    )
                )

        mm = ModelMatrix(
            X=X,
            penalties=full_penalties,
            smooths=smooth_infos,
            column_names=col_names,
            has_intercept=True,
            n_parametric=0,
            offset=None,
            response=y,
        )

        self._fit_result = pirls_fit(
            mm,
            self._family,
            smoothing_params=None,
            method=method,
        )
        self._fitted = True
        self._data = arrays
        return self

    def predict(
        self,
        new_data: InputData,
        *,
        se: bool = False,
    ) -> NDArray:
        """Predict on new data.

        Parameters
        ----------
        new_data:
            Data dict with the same functional and scalar covariates as training data.
        se:
            If `True`, return `(predictions, standard_errors)` instead of just predictions.

        Returns
        -------
        NDArray or tuple[NDArray, NDArray]
        """
        self._check_fitted()
        arrays = self._prepare_functional_data(new_data)
        n = len(next(v for v in arrays.values() if v.ndim == 1))

        X_new = self._build_predict_matrix(arrays, n)

        beta = self._fit_result.coefficients
        eta = X_new @ beta
        mu = self._family.link_inverse(eta)

        if se:
            scale = self._fit_result.scale
            p = X_new.shape[1]
            XtX = self._fit_result_XtX(X_new)
            cov_beta = scale * np.linalg.inv(XtX)
            se_eta = np.sqrt(np.sum(X_new @ cov_beta * X_new, axis=1))
            return mu, se_eta

        return mu

    def _fit_result_XtX(self, X_new: NDArray) -> NDArray:
        """Reconstruct (X'WX + S) from the training data for variance computation."""
        beta = self._fit_result.coefficients
        sp = self._fit_result.smoothing_params

        X_train = self._build_full_training_matrix()
        n = X_train.shape[0]
        p = X_train.shape[1]

        mu_train = self._family.link_inverse(X_train @ beta)
        g_prime = self._family.link_derivative(mu_train)
        dmu_deta = 1.0 / np.maximum(np.abs(g_prime), 1e-10)
        variance = self._family.variance(mu_train)
        W = dmu_deta**2 / np.maximum(variance, 1e-10)

        sqrtW = np.sqrt(W)
        Xw = X_train * sqrtW[:, np.newaxis]
        XtWX = Xw.T @ Xw

        mm_penalties = self._get_full_penalties(p)
        S_total = np.zeros((p, p))
        for lam, pen in zip(sp, mm_penalties):
            S_total += lam * pen

        return XtWX + S_total

    def _build_full_training_matrix(self) -> NDArray:
        """Rebuild the full training design matrix."""
        return self._build_predict_matrix(self._data, len(self._data[self._response]))

    def _build_predict_matrix(self, arrays: dict[str, NDArray], n: int) -> NDArray:
        """Build prediction design matrix from data arrays."""
        columns: list[NDArray] = [np.ones((n, 1))]

        for ft in self._functional_terms:
            X_func = arrays[ft.name]
            B = self._basis_matrices[ft.name]
            w = _integration_weights(X_func.shape[1], ft.domain)
            J = _build_functional_design(X_func, B, w)
            J -= J.mean(axis=0, keepdims=True)
            columns.append(J)

        if self._scalar_model_matrix is not None:
            scalar_data = {k: v for k, v in arrays.items() if v.ndim == 1}
            scalar_mm = predict_matrix(self._scalar_model_matrix, scalar_data)
            scalar_X = scalar_mm[:, 1:] if self._scalar_model_matrix.has_intercept else scalar_mm
            columns.append(scalar_X)

        return np.column_stack(columns)

    def _get_full_penalties(self, p: int) -> list[NDArray]:
        """Reconstruct full penalty matrices."""
        penalties = []
        for ft in self._functional_terms:
            start, end = self._func_col_ranges[ft.name]
            k = end - start
            S = self._penalty_matrices[ft.name]
            S_full = np.zeros((p, p))
            S_full[start:end, start:end] = S
            penalties.append(S_full)

        if self._scalar_model_matrix is not None:
            scalar_offset = max(end for _, end in self._func_col_ranges.values())
            for pen in self._scalar_model_matrix.penalties:
                if self._scalar_model_matrix.has_intercept:
                    pen_block = pen[1:, 1:]
                else:
                    pen_block = pen
                S_full = np.zeros((p, p))
                k = pen_block.shape[0]
                S_full[scalar_offset : scalar_offset + k, scalar_offset : scalar_offset + k] = (
                    pen_block
                )
                penalties.append(S_full)

        return penalties

    def coefficient_function(
        self,
        term_name: str,
        *,
        n_grid: int = 200,
        level: float = 0.95,
    ) -> CoefficientFunction:
        """Extract the estimated coefficient function beta(t) for a functional term.

        Parameters
        ----------
        term_name:
            Name of the functional term.
        n_grid:
            Number of grid points for evaluation.
        level:
            Confidence level for pointwise intervals.

        Returns
        -------
        CoefficientFunction
        """
        self._check_fitted()

        ft = None
        for f in self._functional_terms:
            if f.name == term_name:
                ft = f
                break
        if ft is None:
            names = [f.name for f in self._functional_terms]
            raise ValueError(f"Term {term_name!r} not found. Available: {names}")

        t_eval = np.linspace(ft.domain[0], ft.domain[1], n_grid)

        if ft.basis == "bspline":
            B_eval, _ = _bspline_basis_and_penalty(t_eval, ft.n_basis, ft.domain, ft.penalty_order)
        else:
            B_eval = _fourier_basis(t_eval, ft.n_basis, ft.domain)

        start, end = self._func_col_ranges[ft.name]
        beta_func = self._fit_result.coefficients[start:end]

        values = B_eval @ beta_func

        from scipy.stats import norm

        z = norm.ppf(1 - (1 - level) / 2)

        scale = self._fit_result.scale
        p = self._total_cols
        X_train = self._build_full_training_matrix()
        A = self._fit_result_XtX(X_train)
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            A_inv = np.linalg.pinv(A)

        cov_block = scale * A_inv[start:end, start:end]

        se = np.sqrt(np.maximum(np.diag(B_eval @ cov_block @ B_eval.T), 0.0))
        lower = values - z * se
        upper = values + z * se

        return CoefficientFunction(
            grid=t_eval,
            values=values,
            se=se,
            lower=lower,
            upper=upper,
            term_name=term_name,
        )

    def edf(self) -> dict[str, float]:
        """Effective degrees of freedom per functional term.

        Returns
        -------
        dict[str, float]
        """
        self._check_fitted()
        result = {}
        edf_list = self._fit_result.edf
        for i, ft in enumerate(self._functional_terms):
            if i < len(edf_list):
                result[ft.name] = edf_list[i]
            else:
                result[ft.name] = float("nan")
        return result

    @property
    def scale(self) -> float:
        self._check_fitted()
        return self._fit_result.scale

    @property
    def deviance(self) -> float:
        self._check_fitted()
        return self._fit_result.deviance

    @property
    def edf_total(self) -> float:
        self._check_fitted()
        return self._fit_result.edf_total

    @property
    def coefficients(self) -> NDArray:
        self._check_fitted()
        return self._fit_result.coefficients.copy()

    def summary(self) -> str:
        """Text summary of the fitted functional GAM."""
        self._check_fitted()
        lines = [
            "FunctionalGAM summary",
            "=" * 60,
            f"Response:    {self._response}",
            f"Family:      {self._family.__class__.__name__}",
            f"N obs:       {len(self._data[self._response])}",
            f"EDF total:   {self._fit_result.edf_total:.1f}",
            f"Deviance:    {self._fit_result.deviance:.2f}",
            f"Scale:       {self._fit_result.scale:.4f}",
            "",
            "Functional terms:",
        ]
        for i, ft in enumerate(self._functional_terms):
            edf_val = self._fit_result.edf[i] if i < len(self._fit_result.edf) else float("nan")
            lines.append(
                f"  {ft.name}: basis={ft.basis}, k={ft.n_basis}, "
                f"domain={ft.domain}, edf={edf_val:.1f}"
            )

        if self._scalar_terms is not None:
            lines.append("")
            lines.append(f"Scalar terms: {self._scalar_terms}")

        return "\n".join(lines)

    def _prepare_functional_data(self, data: InputData) -> dict[str, NDArray]:
        """Convert input data, preserving 2-D arrays for functional covariates."""
        if isinstance(data, dict):
            result = {}
            func_names = {ft.name for ft in self._functional_terms}
            for k, v in data.items():
                arr = np.asarray(v, dtype=float)
                if k in func_names:
                    if arr.ndim != 2:
                        raise ValueError(
                            f"Functional covariate {k!r} must be 2-D (n x T), "
                            f"got shape {arr.shape}."
                        )
                    result[k] = arr
                else:
                    result[k] = arr.ravel()
            return result
        raise TypeError(f"Data must be a dict, got {type(data).__name__}.")

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("This FunctionalGAM has not been fitted yet. Call .fit(data) first.")

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        terms = ", ".join(ft.name for ft in self._functional_terms)
        return f"FunctionalGAM(response={self._response!r}, terms=[{terms}], {status})"
