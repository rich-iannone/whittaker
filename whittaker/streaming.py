r"""Streaming / online GAMs.

Provides `StreamingGAM` for incremental fitting of GAMs on data that arrives in batches. Instead of
storing the full dataset, sufficient statistics (X'WX, X'Wz, and observation count) are accumulated
across batches and periodically solved to update model coefficients.

Two update modes are supported:

- **Accumulate mode** (default): each `partial_fit` call adds the batch's sufficient statistics to a
running total. Call `solve()` to recompute coefficients from the accumulated statistics.
- **Sliding window mode**: maintains a fixed-size window by discounting old statistics with an
exponential decay factor.

Smoothing parameters can be fixed or re-estimated periodically during `solve()` calls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve

from whittaker.data import InputData, prepare_data
from whittaker.families.base import Family
from whittaker.families.gaussian import Gaussian
from whittaker.formula.parser import parse
from whittaker.formula.terms import Formula
from whittaker.gam import GAM, PredictionResult
from whittaker.model_matrix import ModelMatrix, build_model_matrix, predict_matrix


@dataclass
class StreamingSnapshot:
    r"""A snapshot of streaming GAM state at a point in time.

    Appended to `StreamingGAM`'s history every time `solve()` is called, letting you track how
    coefficients, smoothing parameters, and fit diagnostics evolve as more batches of data arrive.
    Retrieved via `StreamingGAM.smoothing_history()`.

    Attributes
    ----------
    n_obs:
        Total (possibly decayed, under sliding-window mode) observation count at the time of this
        `solve()` call.
    n_batches:
        Number of batches processed so far.
    coefficients:
        Coefficient estimates at this solve.
    smoothing_params:
        Smoothing parameters used for this solve.
    edf_total:
        Total effective degrees of freedom at this solve.
    scale:
        Estimated scale (dispersion) parameter at this solve.
    deviance:
        Accumulated (possibly decayed) deviance at this solve.
    """

    n_obs: int
    n_batches: int
    coefficients: NDArray
    smoothing_params: list[float]
    edf_total: float
    scale: float
    deviance: float


class StreamingGAM:
    r"""Streaming / online GAM.

    Fits a GAM incrementally by accumulating weighted sufficient statistics (`X'WX` and `X'Wz`)
    across data batches, rather than storing and re-fitting on the full dataset each time. This
    makes it suitable for data arriving continuously or in a stream too large to hold in memory at
    once: the memory footprint depends only on the number of basis coefficients `p` (via a
    `p x p` matrix), not on the number of observations seen. The model structure (formula, basis
    dimensions, penalties) is fixed at initialisation from a small pilot batch, and subsequent
    `partial_fit()` calls add data without storing the raw observations.

    Two update modes are supported: **accumulate mode** (`decay=1.0`, the default), where every
    batch contributes equally regardless of when it arrived, and **sliding-window mode**
    (`decay < 1.0`), where older batches' contributions to the accumulated statistics are
    exponentially downweighted, allowing the model to adapt to a slowly drifting data-generating
    process.

    Use `StreamingGAM` for large or continuously-arriving datasets where holding the full data in
    memory (as ordinary `GAM` does) is impractical, or where the underlying relationship may drift
    over time and old data should be forgotten.

    Parameters
    ----------
    formula:
        Model formula (e.g. `"y ~ s(x1) + s(x2)"`).
    family:
        Response distribution family. Defaults to `Gaussian()`.
    decay:
        Exponential decay factor for sliding window. `1.0` (default) means no decay (accumulate all
        data equally). Values less than `1.0` downweight older batches' sufficient statistics by a
        factor of `decay` every time a new batch arrives.
    smoothing_params:
        Fixed smoothing parameters. If `None`, estimated from the pilot batch (via a one-off
        ordinary `GAM` fit with REML) and optionally re-estimated later via
        `solve(reestimate_smoothing=True)`.

    Notes
    -----
    Each `partial_fit()` call treats the incoming batch as one step of iteratively reweighted least
    squares: given the current coefficients, it computes working responses `z` and IRLS weights `W`
    for the batch, forms the batch's contribution to the weighted normal equations,

    $$X_{\text{batch}}' W X_{\text{batch}}, \qquad X_{\text{batch}}' W z_{\text{batch}},$$

    and adds these to the running totals (after applying the decay factor, if any, to the existing
    totals). Calling `solve()` then solves the penalized normal equations

    $$\left(\sum_{\text{batches}} X'WX + S_\lambda\right) \beta = \sum_{\text{batches}} X'Wz$$

    via a Cholesky factorization, where `S_\lambda = \sum_j \lambda_j S_j` is the weighted sum of
    penalty matrices. Because only the accumulated `p x p` matrix `X'WX` and length-`p` vector
    `X'Wz` are retained, this scales to arbitrarily many observations at fixed memory cost in `p`.

    Examples
    --------
    ```{python}
    import numpy as np
    from whittaker.streaming import StreamingGAM

    rng = np.random.default_rng(0)
    model = StreamingGAM("y ~ s(x)")

    for _ in range(5):
        x = rng.uniform(0, 1, 200)
        y = np.sin(2 * np.pi * x) + rng.normal(scale=0.2, size=200)
        model.partial_fit({"x": x, "y": y})

    model.solve()
    print(model.summary())
    ```
    """

    def __init__(
        self,
        formula: str | Formula,
        *,
        family: Family | None = None,
        decay: float = 1.0,
        smoothing_params: list[float] | None = None,
    ) -> None:
        if isinstance(formula, str):
            self._formula = parse(formula)
        else:
            self._formula = formula

        self._family = family if family is not None else Gaussian()

        if not 0 < decay <= 1.0:
            raise ValueError(f"decay must be in (0, 1], got {decay}")
        self._decay = decay

        self._fixed_sp = smoothing_params
        self._initialised = False
        self._solved = False

        self._model_matrix: ModelMatrix | None = None
        self._p: int = 0
        self._XtWX: NDArray | None = None
        self._XtWz: NDArray | None = None
        self._n_obs: int = 0
        self._n_batches: int = 0
        self._sum_deviance: float = 0.0
        self._sum_y: float = 0.0
        self._sum_y2: float = 0.0

        self._coefficients: NDArray | None = None
        self._smoothing_params: list[float] = []
        self._edf: list[float] = []
        self._edf_total: float = 0.0
        self._scale: float = 1.0
        self._hat_trace: float = 0.0

        self._history: list[StreamingSnapshot] = []

    @property
    def formula(self) -> Formula:
        """Parsed model formula this streaming GAM was constructed with.

        The formula (response, smooth and parametric terms) is fixed for the lifetime of
        the model; it determines the basis structure built from the first `partial_fit()`
        batch.

        Returns
        -------
        Formula
        """
        return self._formula

    @property
    def family(self) -> Family:
        """Response distribution family used for the link function and variance model.

        Returns
        -------
        Family
        """
        return self._family

    @property
    def is_initialised(self) -> bool:
        """Whether the model structure has been built from a pilot batch.

        Becomes `True` after the first `partial_fit()` call, once the model matrix,
        basis dimensions, and penalties have been established.

        Returns
        -------
        bool
        """
        return self._initialised

    @property
    def is_solved(self) -> bool:
        """Whether coefficients are available from an initial fit or a `solve()` call.

        `predict()`, `coefficients`, and other solved-state accessors raise `RuntimeError`
        when this is `False`.

        Returns
        -------
        bool
        """
        return self._solved

    @property
    def n_obs(self) -> int:
        """Accumulated observation count across all ingested batches.

        Under sliding-window mode (`decay < 1.0`), this count is itself decayed at each
        `partial_fit()` call, so it reflects an effective rather than a raw cumulative
        count.

        Returns
        -------
        int
        """
        return self._n_obs

    @property
    def n_batches(self) -> int:
        """Total number of batches passed to `partial_fit()` so far.

        Unlike `n_obs`, this count is never decayed.

        Returns
        -------
        int
        """
        return self._n_batches

    @property
    def coefficients(self) -> NDArray:
        """Current coefficient vector from the most recent `solve()` (or the pilot fit).

        Returns a copy, so mutating the returned array does not affect the model.

        Returns
        -------
        NDArray
            Length-`p` array of basis coefficients.
        """
        self._check_solved()
        assert self._coefficients is not None
        return self._coefficients.copy()

    @property
    def edf_total(self) -> float:
        """Total effective degrees of freedom from the most recent `solve()`.

        Computed as the trace of the hat matrix implied by the accumulated `X'WX` and the
        current smoothing parameters.

        Returns
        -------
        float
        """
        self._check_solved()
        return self._edf_total

    @property
    def scale(self) -> float:
        """Estimated dispersion (scale) parameter from the most recent `solve()`.

        For families with a known scale (e.g. Poisson, binomial), this is fixed at `1.0`;
        otherwise it is estimated from the accumulated deviance and `edf_total`.

        Returns
        -------
        float
        """
        self._check_solved()
        return self._scale

    @property
    def smoothing_params(self) -> list[float]:
        """Smoothing parameters currently in use, one per penalized smooth term.

        These come from the constructor's `smoothing_params` argument if fixed, or from the
        pilot fit / most recent `solve(reestimate_smoothing=True)` call otherwise. Returns a
        copy, so mutating the returned list does not affect the model.

        Returns
        -------
        list[float]
        """
        self._check_solved()
        return list(self._smoothing_params)

    def partial_fit(self, data: InputData) -> StreamingGAM:
        r"""Ingest a batch of data.

        On the first call, builds the model matrix structure (basis dimensions, penalties, knot
        locations) from this batch via a pilot `GAM` fit, and initializes the coefficients and
        smoothing parameters from that fit (unless fixed smoothing parameters were supplied).
        Subsequent calls reuse this fixed structure: the batch's working response and IRLS weights
        are computed using the *current* coefficients (from the most recent `solve()`, or the pilot
        fit), and its contribution to the running `X'WX` / `X'Wz` sufficient statistics is added
        (after applying the decay factor to existing totals, if `decay < 1.0`).

        Parameters
        ----------
        data:
            Column-oriented batch data containing the response and all covariates in `formula`.

        Returns
        -------
        StreamingGAM
            Returns `self` for method chaining.
        """
        arrays = prepare_data(data)
        response_name = self._formula.response
        y = arrays[response_name]
        n = len(y)

        if not self._initialised:
            self._initialise(arrays)

        assert self._model_matrix is not None
        assert self._XtWX is not None
        assert self._XtWz is not None

        X = predict_matrix(self._model_matrix, arrays)

        mu = self._family.link_inverse(X @ self._coefficients) if self._solved else y.copy()
        mu = np.clip(mu, 1e-10, 1e10)

        g_prime = self._family.link_derivative(mu)
        dmu_deta = 1.0 / np.maximum(np.abs(g_prime), 1e-10)
        variance = self._family.variance(mu)
        W = dmu_deta**2 / np.maximum(variance, 1e-10)

        z = self._family.link(mu) + (y - mu) * g_prime

        sqrtW = np.sqrt(W)
        Xw = X * sqrtW[:, np.newaxis]
        zw = z * sqrtW

        batch_XtWX = Xw.T @ Xw
        batch_XtWz = Xw.T @ zw

        if self._decay < 1.0:
            self._XtWX *= self._decay
            self._XtWz *= self._decay
            self._sum_deviance *= self._decay
            self._n_obs = int(self._n_obs * self._decay)

        self._XtWX += batch_XtWX
        self._XtWz += batch_XtWz
        self._n_obs += n
        self._n_batches += 1

        batch_dev = self._family.deviance(y, mu)
        self._sum_deviance += batch_dev
        self._sum_y += float(np.sum(y))
        self._sum_y2 += float(np.sum(y**2))

        return self

    def _initialise(self, arrays: dict[str, NDArray]) -> None:
        mm = build_model_matrix(self._formula, arrays)
        self._model_matrix = mm
        self._p = mm.X.shape[1]

        self._XtWX = np.zeros((self._p, self._p))
        self._XtWz = np.zeros(self._p)

        if self._fixed_sp is not None:
            self._smoothing_params = list(self._fixed_sp)
        else:
            pilot_gam = GAM(self._formula, family=self._family)
            pilot_gam.fit(arrays, method="REML")
            self._smoothing_params = list(pilot_gam._fit_result.smoothing_params)
            self._coefficients = pilot_gam._fit_result.coefficients.copy()
            self._edf_total = pilot_gam._fit_result.edf_total
            self._scale = pilot_gam._fit_result.scale
            self._solved = True

        if self._coefficients is None:
            self._coefficients = np.zeros(self._p)

        self._initialised = True

    def solve(self, *, reestimate_smoothing: bool = False) -> StreamingGAM:
        r"""Solve for coefficients from accumulated sufficient statistics.

        Forms the penalized normal equations `(X'WX + S_lambda) beta = X'Wz` from the currently
        accumulated statistics and the current smoothing parameters, and solves them via a
        Cholesky factorization (with a small ridge added for numerical stability). Also updates
        effective degrees of freedom, the scale estimate, and appends a `StreamingSnapshot` to the
        fit history.

        Parameters
        ----------
        reestimate_smoothing:
            If `True`, re-estimate smoothing parameters from the current sufficient statistics using
            a GCV line search over each smoothing parameter in turn (coordinate-wise). Default
            `False` uses the current (pilot or fixed) smoothing parameters. Ignored (has no effect)
            if the model was constructed with fixed `smoothing_params`.

        Returns
        -------
        StreamingGAM
            Returns `self` for method chaining.
        """
        if not self._initialised:
            raise RuntimeError("No data has been ingested yet. Call partial_fit() first.")

        assert self._model_matrix is not None
        assert self._XtWX is not None
        assert self._XtWz is not None

        if reestimate_smoothing and self._fixed_sp is None:
            self._smoothing_params = self._estimate_smoothing_gcv()

        S_total = np.zeros((self._p, self._p))
        for lam, pen in zip(self._smoothing_params, self._model_matrix.penalties, strict=False):
            S_total += lam * pen

        A = self._XtWX + S_total
        A = (A + A.T) * 0.5
        ridge = np.finfo(float).eps * max(np.trace(A) / self._p, 1.0)
        A[np.diag_indices_from(A)] += ridge

        cho, lower = cho_factor(A)
        self._coefficients = cho_solve((cho, lower), self._XtWz)

        A_inv_XtWX = cho_solve((cho, lower), self._XtWX)
        self._hat_trace = float(np.trace(A_inv_XtWX))

        self._edf = []
        for info in self._model_matrix.smooths:
            cs, ce = info.col_start, info.col_end
            self._edf.append(float(np.trace(A_inv_XtWX[cs:ce, cs:ce])))
        self._edf_total = self._hat_trace

        n_eff = self._n_obs
        if not self._family.scale_known and n_eff > self._edf_total:
            self._scale = self._sum_deviance / (n_eff - self._edf_total)
        else:
            self._scale = 1.0

        assert self._coefficients is not None
        self._history.append(
            StreamingSnapshot(
                n_obs=self._n_obs,
                n_batches=self._n_batches,
                coefficients=self._coefficients.copy(),
                smoothing_params=list(self._smoothing_params),
                edf_total=self._edf_total,
                scale=self._scale,
                deviance=self._sum_deviance,
            )
        )

        self._solved = True
        return self

    def _estimate_smoothing_gcv(self) -> list[float]:
        from scipy.optimize import minimize_scalar

        assert self._model_matrix is not None
        assert self._XtWX is not None
        assert self._XtWz is not None

        n_sp = len(self._model_matrix.penalties)
        if n_sp == 0:
            return []

        best_sp = list(self._smoothing_params)

        for j in range(n_sp):

            def gcv_for_j(log_lam: float, j: int = j) -> float:
                assert self._model_matrix is not None
                assert self._XtWX is not None
                assert self._XtWz is not None

                sp_trial = list(best_sp)
                sp_trial[j] = np.exp(log_lam)

                S_total = np.zeros((self._p, self._p))
                for lam, pen in zip(sp_trial, self._model_matrix.penalties, strict=False):
                    S_total += lam * pen

                A = self._XtWX + S_total
                A = (A + A.T) * 0.5
                ridge = np.finfo(float).eps * max(np.trace(A) / self._p, 1.0)
                A[np.diag_indices_from(A)] += ridge

                try:
                    cho_l, lo = cho_factor(A)
                    beta = cho_solve((cho_l, lo), self._XtWz)
                    A_inv_XtWX = cho_solve((cho_l, lo), self._XtWX)
                    hat_tr = float(np.trace(A_inv_XtWX))
                except np.linalg.LinAlgError:
                    return 1e20

                rss = float(self._sum_y2 - 2 * beta @ self._XtWz + beta @ self._XtWX @ beta)
                rss = max(rss, 1e-10)
                n = self._n_obs
                denom = (1.0 - hat_tr / n) ** 2
                if denom < 1e-10:
                    return 1e20
                return rss / (n * denom)

            current_log_lam = np.log(max(best_sp[j], 1e-10))
            result = minimize_scalar(
                gcv_for_j,
                bounds=(current_log_lam - 5, current_log_lam + 5),
                method="bounded",
            )
            best_sp[j] = float(np.exp(result.x))

        return best_sp

    def predict(self, new_data: InputData, *, se: bool = False) -> PredictionResult:
        r"""Predict on new data.

        Builds the prediction design matrix using the fixed basis structure established at
        initialisation, and forms the linear predictor and (via the family's inverse link) fitted
        values from the current coefficients. Requires that `solve()` has been called at least once.

        Parameters
        ----------
        new_data:
            Column-oriented covariate data.
        se:
            If `True`, compute standard errors on the linear predictor scale from the Bayesian
            posterior covariance implied by the current accumulated `X'WX` and smoothing parameters.

        Returns
        -------
        PredictionResult
        """
        self._check_solved()
        assert self._model_matrix is not None
        assert self._coefficients is not None
        assert self._XtWX is not None

        new_data = prepare_data(new_data)
        X_new = predict_matrix(self._model_matrix, new_data)
        eta = X_new @ self._coefficients
        mu = self._family.link_inverse(eta)

        se_values = None
        if se:
            S_total = np.zeros((self._p, self._p))
            for lam, pen in zip(self._smoothing_params, self._model_matrix.penalties, strict=False):
                S_total += lam * pen
            A = self._XtWX + S_total
            A = (A + A.T) * 0.5
            ridge = np.finfo(float).eps * max(np.trace(A) / self._p, 1.0)
            A[np.diag_indices_from(A)] += ridge
            cho, lower = cho_factor(A)
            V_beta = self._scale * cho_solve((cho, lower), np.eye(self._p))
            var_diag = np.sum(X_new * (X_new @ V_beta), axis=1)
            se_values = np.sqrt(np.maximum(var_diag, 0.0))

        return PredictionResult(
            values=mu,
            se=se_values,
            linear_predictor=eta,
        )

    def should_refit(self, *, min_batches: int = 10) -> bool:
        """Check if a refit (solve) is recommended.

        Returns `True` when enough new data has accumulated since the last solve to justify
        re-estimating coefficients.

        Parameters
        ----------
        min_batches:
            Minimum batches since last solve before recommending refit.
        """
        if not self._initialised:
            return False
        last_solved = self._history[-1].n_batches if self._history else 0
        return (self._n_batches - last_solved) >= min_batches

    def smoothing_history(self) -> list[StreamingSnapshot]:
        """Return history of snapshots taken at each `solve()` call.

        Returns
        -------
        list[StreamingSnapshot]
        """
        return list(self._history)

    def summary(self) -> str:
        """Build a human-readable text summary of the streaming GAM's current state.

        Reports the formula, family, decay factor, accumulated observation and batch
        counts, total EDF, scale, and accumulated deviance, followed by a per-smooth-term
        EDF breakdown and the number of `solve()` snapshots recorded so far.

        Returns
        -------
        str
            Multi-line summary text.
        """
        self._check_solved()
        assert self._model_matrix is not None

        lines = [
            "StreamingGAM summary",
            "=" * 60,
            f"Formula:     {self._formula!r}",
            f"Family:      {self._family.__class__.__name__}",
            f"Decay:       {self._decay}",
            f"N obs:       {self._n_obs}",
            f"N batches:   {self._n_batches}",
            f"EDF total:   {self._edf_total:.1f}",
            f"Scale:       {self._scale:.4f}",
            f"Deviance:    {self._sum_deviance:.1f}",
            "",
            "Smooth terms:",
        ]
        for info, edf_val in zip(self._model_matrix.smooths, self._edf, strict=False):
            lines.append(f"  {info.term!r}: edf={edf_val:.1f}")

        lines.append("")
        lines.append(f"Snapshots:   {len(self._history)}")
        return "\n".join(lines)

    def reset(self) -> StreamingGAM:
        """Reset accumulated statistics (keep model structure).

        Returns
        -------
        StreamingGAM
        """
        if not self._initialised:
            raise RuntimeError("Cannot reset before initialisation.")

        self._XtWX = np.zeros((self._p, self._p))
        self._XtWz = np.zeros(self._p)
        self._n_obs = 0
        self._n_batches = 0
        self._sum_deviance = 0.0
        self._sum_y = 0.0
        self._sum_y2 = 0.0
        self._solved = False
        self._history = []
        return self

    def _check_solved(self) -> None:
        if not self._solved:
            raise RuntimeError(
                "Model has not been solved yet. "
                "Call partial_fit() then solve(), or use partial_fit() "
                "which auto-solves on the first batch."
            )

    def __repr__(self) -> str:
        if self._solved:
            status = f"fitted, n={self._n_obs}, batches={self._n_batches}"
        elif self._initialised:
            status = f"initialised, n={self._n_obs}, batches={self._n_batches}"
        else:
            status = "unfitted"
        return f"StreamingGAM({self._formula!r}, {status})"
