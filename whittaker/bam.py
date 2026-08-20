"""BigGAM: large-scale GAM fitting via discretized covariates.

Provides `BigGAM`, a drop-in replacement for `~whittaker.gam.GAM` that avoids materializing the
full `n x p` design matrix. Instead, each covariate is rounded (independently, then combined) to
a grid of at most `n_discrete` representative values, and the smooth basis is evaluated only
once per unique combination of discretized values rather than once per observation. Each
observation is mapped back to its bin through an index array, and the `X'WX` cross-product
needed by P-IRLS is accumulated
directly from the per-bin basis rows and bin membership counts (see `_compute_XtWX` and
`build_discretized_model_matrix`). This makes memory usage roughly `O(d p)` instead of `O(n p)`
(where `d` is the number of unique discretized rows and `d << n` for large datasets), and speeds up
the `X'WX` accumulation proportionally, since the cost scales with the number of unique bins rather
than the number of observations.

Use `BigGAM` when the standard `GAM` runs out of memory or is too slow because `n` is large (roughly
`n > 1_000_000`).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from whittaker.data import InputData, prepare_data
from whittaker.families.base import Family
from whittaker.fitting.bam import (
    DiscretizedBlock,
    DiscretizedModelMatrix,
    _compute_XtWX,
    _discretize_nd,
    bam_fit,
)
from whittaker.formula import Formula
from whittaker.formula.terms import InteractionTerm, LinearTerm, OffsetTerm, SmoothTerm
from whittaker.gam import GAM
from whittaker.model_matrix import (
    ModelMatrix,
    SmoothInfo,
    _apply_constraint,
    _apply_constraint_to_penalty,
    _extract_by_column,
    _extract_column,
    _is_factor,
    _null_space_penalty,
    _resolve_basis,
    _resolve_t2_basis,
    _resolve_tensor_basis,
    _resolve_tensor_interaction_basis,
)
from whittaker.smooths.factor_smooth import FactorSmoothBasis
from whittaker.smooths.random import RandomEffectBasis


def build_discretized_model_matrix(
    formula: Formula,
    data: dict[str, NDArray],
    *,
    n_discrete: int = 200,
    apply_constraints: bool = True,
    select: bool = False,
) -> DiscretizedModelMatrix:
    """Build a compressed model matrix by discretizing covariates.

    Follows the same logic as `~whittaker.model_matrix.build_model_matrix` but evaluates each
    smooth's basis only at unique discretized covariate values, producing a
    `~whittaker.fitting.bam.DiscretizedModelMatrix` that never stores the full `n x p` design
    matrix. Parametric (linear, interaction, offset) terms are still expanded to full columns, since
    they are cheap to store; only smooth terms are discretized and bin-compressed.

    Parameters
    ----------
    formula : Formula
        Parsed model formula describing the response, smooth terms, linear terms, and any
        `offset()` terms.
    data : dict[str, numpy.ndarray]
        Column-oriented data with one entry per variable referenced by `formula`. All arrays must
        have equal length.
    n_discrete : int
        Maximum number of unique representative values per covariate (or per combination of
        covariates, for multi-dimensional smooths). Each covariate is rounded onto an
        equally-spaced grid of this size before the basis is evaluated. Defaults to `200`.
    apply_constraints : bool
        If `True` (the default), apply each smooth's identifiability constraints (e.g. sum-to-zero)
        to the discretized basis and penalty matrices, matching the behavior of
        `~whittaker.model_matrix.build_model_matrix`. Terms with a `by=` variable are left
        unconstrained, as in the non-discretized path.
    select : bool
        If `True`, augment each smooth's penalty with an additional null-space penalty so the term
        can be shrunk to exactly zero (double-penalty selection), matching the `select` argument of
        `~whittaker.gam.GAM.fit`.

    Returns
    -------
    DiscretizedModelMatrix
        A compressed model matrix holding, for each smooth term, a `DiscretizedBlock` with the
        unique basis rows, the per-observation bin index array, and the term's column range, plus
        the full parametric columns, penalty matrices, and bookkeeping (column names, response,
        offset) needed to fit and predict from the model without ever forming the dense `n x p`
        design matrix.
    """
    lengths = {name: len(arr) for name, arr in data.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) > 1:
        detail = ", ".join(f"{k}: {v}" for k, v in lengths.items())
        raise ValueError(f"All data columns must have the same length. Got: {detail}.")
    n = next(iter(lengths.values())) if lengths else 0

    response = _extract_column(data, formula.response)

    param_blocks: list[NDArray] = []
    col_names: list[str] = []

    if formula.intercept:
        param_blocks.append(np.ones((n, 1)))
        col_names.append("(Intercept)")

    n_parametric = 0
    offset: NDArray | None = None
    offset_expressions: list[str] = []

    for term in formula.terms:
        if isinstance(term, LinearTerm):
            col = _extract_column(data, term.variable)
            param_blocks.append(col[:, np.newaxis])
            col_names.append(term.variable)
            n_parametric += 1
        elif isinstance(term, InteractionTerm):
            left = _extract_column(data, term.left)
            right = _extract_column(data, term.right)
            if term.full:
                param_blocks.append(left[:, np.newaxis])
                col_names.append(term.left)
                param_blocks.append(right[:, np.newaxis])
                col_names.append(term.right)
                n_parametric += 2
            param_blocks.append((left * right)[:, np.newaxis])
            col_names.append(f"{term.left}:{term.right}")
            n_parametric += 1
        elif isinstance(term, OffsetTerm):
            offset_col = _extract_column(data, term.expression)
            offset_expressions.append(term.expression)
            offset = offset_col.copy() if offset is None else offset + offset_col

    parametric_cols = np.column_stack(param_blocks) if param_blocks else None
    n_param_cols = parametric_cols.shape[1] if parametric_cols is not None else 0
    col_offset = n_param_cols

    disc_blocks: list[DiscretizedBlock] = []
    smooth_infos: list[SmoothInfo] = []
    penalty_blocks: list[tuple[int, int, NDArray]] = []

    for term in formula.terms:
        if not isinstance(term, SmoothTerm):
            continue

        if term.smooth_type == "te":
            basis = _resolve_tensor_basis(term)
        elif term.smooth_type == "ti":
            basis = _resolve_tensor_interaction_basis(term)
        elif term.smooth_type == "t2":
            basis = _resolve_t2_basis(term)
        elif term.smooth_type == "s":
            basis = _resolve_basis(term)
        else:
            raise NotImplementedError(f"Smooth type {term.smooth_type!r} not supported in BigGAM.")

        if isinstance(basis, FactorSmoothBasis):
            raise NotImplementedError(
                "Factor smooth interactions (bs='fs') are not yet supported in BigGAM. "
                "Use s(x, by=group) instead."
            )

        if isinstance(basis, RandomEffectBasis):
            x = _extract_by_column(data, term.variables[0])
            basis.fit(x)
            full_basis = basis.basis_matrix(x)
            nsd = basis.null_space_dimension()
            pen_mats = [basis.penalty_matrix()]

            has_by = term.by is not None
            if apply_constraints and not has_by:
                constraint = basis.identifiability_constraints()
                if constraint is not None:
                    full_basis = _apply_constraint(full_basis, constraint)
                    pen_mats = [_apply_constraint_to_penalty(pm, constraint) for pm in pen_mats]
                    nsd = max(nsd - constraint.shape[0], 0)
            if select and nsd > 0:
                pen_mats.append(_null_space_penalty(pen_mats[0]))
                nsd = 0

            k_eff = full_basis.shape[1]
            col_start = col_offset
            col_end = col_start + k_eff
            col_offset = col_end

            disc_blocks.append(
                DiscretizedBlock(
                    unique_basis=full_basis,
                    indices=np.arange(n),
                    col_start=col_start,
                    col_end=col_end,
                )
            )
            for j in range(k_eff):
                col_names.append(f"{term!r}[{j}]")
            pen_start = len(penalty_blocks)
            for pm in pen_mats:
                penalty_blocks.append((col_start, col_end, pm))
            smooth_infos.append(
                SmoothInfo(
                    term=term,
                    basis=basis,
                    col_start=col_start,
                    col_end=col_end,
                    null_space_dim=nsd,
                    penalty_indices=list(range(pen_start, pen_start + len(pen_mats))),
                )
            )
            continue

        if len(term.variables) == 1:
            x = _extract_column(data, term.variables[0])
        else:
            x = np.column_stack([_extract_column(data, v) for v in term.variables])

        unique_x, disc_indices = _discretize_nd(x, n_discrete)
        basis.fit(unique_x)
        nsd = basis.null_space_dimension()

        unique_basis_mat = basis.basis_matrix(unique_x)

        if hasattr(basis, "penalty_matrices"):
            pen_mats = basis.penalty_matrices()  # type: ignore[attr-defined]
        else:
            pen_mats = [basis.penalty_matrix()]

        has_by = term.by is not None
        if apply_constraints and not has_by:
            constraint = basis.identifiability_constraints()
            if constraint is not None:
                unique_basis_mat = _apply_constraint(unique_basis_mat, constraint)
                pen_mats = [_apply_constraint_to_penalty(pm, constraint) for pm in pen_mats]
                nsd = max(nsd - constraint.shape[0], 0)

        if select and nsd > 0:
            pen_mats.append(_null_space_penalty(pen_mats[0]))
            nsd = 0

        if has_by:
            assert term.by is not None
            by_col = _extract_by_column(data, term.by)

            if _is_factor(by_col):
                levels = sorted(np.unique(by_col))
                for level in levels:
                    indicator = (by_col == level).astype(float)
                    k_eff = unique_basis_mat.shape[1]
                    col_start = col_offset
                    col_end = col_start + k_eff
                    col_offset = col_end

                    disc_blocks.append(
                        DiscretizedBlock(
                            unique_basis=unique_basis_mat,
                            indices=disc_indices,
                            col_start=col_start,
                            col_end=col_end,
                            by_weights=indicator,
                        )
                    )
                    for j in range(k_eff):
                        col_names.append(f"{term!r}:{level}[{j}]")
                    pen_start = len(penalty_blocks)
                    for pm in pen_mats:
                        penalty_blocks.append((col_start, col_end, pm))
                    smooth_infos.append(
                        SmoothInfo(
                            term=term,
                            basis=basis,
                            col_start=col_start,
                            col_end=col_end,
                            null_space_dim=nsd,
                            penalty_indices=list(range(pen_start, pen_start + len(pen_mats))),
                            by_var=term.by,
                            by_level=str(level),
                        )
                    )
            else:
                by_vals = np.asarray(by_col, dtype=float)
                k_eff = unique_basis_mat.shape[1]
                col_start = col_offset
                col_end = col_start + k_eff
                col_offset = col_end

                disc_blocks.append(
                    DiscretizedBlock(
                        unique_basis=unique_basis_mat,
                        indices=disc_indices,
                        col_start=col_start,
                        col_end=col_end,
                        by_weights=by_vals,
                    )
                )
                for j in range(k_eff):
                    col_names.append(f"{term!r}[{j}]")
                pen_start = len(penalty_blocks)
                for pm in pen_mats:
                    penalty_blocks.append((col_start, col_end, pm))
                smooth_infos.append(
                    SmoothInfo(
                        term=term,
                        basis=basis,
                        col_start=col_start,
                        col_end=col_end,
                        null_space_dim=nsd,
                        penalty_indices=list(range(pen_start, pen_start + len(pen_mats))),
                        by_var=term.by,
                    )
                )
        else:
            k_eff = unique_basis_mat.shape[1]
            col_start = col_offset
            col_end = col_start + k_eff
            col_offset = col_end

            disc_blocks.append(
                DiscretizedBlock(
                    unique_basis=unique_basis_mat,
                    indices=disc_indices,
                    col_start=col_start,
                    col_end=col_end,
                )
            )
            for j in range(k_eff):
                col_names.append(f"{term!r}[{j}]")
            pen_start = len(penalty_blocks)
            for pm in pen_mats:
                penalty_blocks.append((col_start, col_end, pm))
            smooth_infos.append(
                SmoothInfo(
                    term=term,
                    basis=basis,
                    col_start=col_start,
                    col_end=col_end,
                    null_space_dim=nsd,
                    penalty_indices=list(range(pen_start, pen_start + len(pen_mats))),
                )
            )

    p_total = col_offset
    penalties: list[NDArray] = []
    for cs, ce, S_block in penalty_blocks:
        S_full = np.zeros((p_total, p_total))
        S_full[cs:ce, cs:ce] = S_block
        penalties.append(S_full)

    return DiscretizedModelMatrix(
        blocks=disc_blocks,
        parametric_cols=parametric_cols,
        n_param_cols=n_param_cols,
        penalties=penalties,
        smooth_infos=smooth_infos,
        n_obs=n,
        n_cols=p_total,
        response=response,
        offset=offset,
        has_intercept=formula.intercept,
        n_parametric=n_parametric,
        column_names=col_names,
        offset_expressions=offset_expressions,
    )


class BigGAM(GAM):
    r"""GAM for large datasets using discretized fitting.

    `BigGAM` is a drop-in subclass of `~whittaker.gam.GAM` for datasets too large to fit
    comfortably with the standard dense design matrix (roughly `n > 1_000_000`). It uses the `bam`
    approach of Wood, Li, & Shaddick (2017): each covariate is rounded onto a grid of at most
    `n_discrete` representative values, and the smooth basis is evaluated only once per unique
    (combination of) discretized value(s) rather than once per observation. An index array records,
    for every observation, which unique bin it fell into (see `DiscretizedBlock.indices` in
    `build_discretized_model_matrix`).

    Fitting still runs penalized iteratively reweighted least squares (P-IRLS), exactly as in
    `~whittaker.gam.GAM.fit`, alternating an inner coefficient update with an outer smoothing
    parameter selection. The difference is purely computational: instead of forming the full
    `n x p` design matrix `X` and computing `X'WX` and `X'Wz` directly, `bam_fit` (in
    `whittaker.fitting.bam`) accumulates these quantities per bin — for each smooth's discretized
    block, observation weights are aggregated into per-bin totals via `numpy.bincount` over the bin
    indices, and the resulting `d x d` cross-products of unique basis rows (`d` = number of unique
    bins) are scattered into the correct `p x p` block of `X'WX` (see `_compute_XtWX`). Because
    `d` can be orders of magnitude smaller than `n`, this reduces the memory needed for the
    cross-product step from `O(n p)` to `O(d p)`, and the resulting fit closely approximates the
    exact (non-discretized) GAM fit on the same data.

    `BigGAM` is a drop-in subclass of `GAM`: `predict()`, `summary()`, `plot()`, and `check()` all
    work the same way as for `GAM`. The one internal difference is that `self._model_matrix.X` is
    an empty array (the dense design matrix is never materialized) so operations that would
    otherwise reconstruct per-term columns (e.g. `smooth_tests()`) instead re-expand each smooth's
    columns on demand from its `DiscretizedBlock` via `_expand_block_columns`.

    Parameters
    ----------
    formula
        Model formula as a string (e.g. `"y ~ s(x1) + s(x2) + x3"`), or an already-parsed `Formula`
        object. Same syntax as `~whittaker.gam.GAM`.
    family
        Response distribution family, e.g. `Gaussian()`, `Binomial()`, `Poisson()`, `Gamma()`, or
        `Tweedie()`. Defaults to `Gaussian()`.
    n_discrete
        Maximum number of unique representative values per covariate (or per combination of
        covariates, for multi-dimensional smooths). Defaults to `200`.

    Notes
    -----
    `n_discrete` controls the accuracy/memory tradeoff directly. Larger values give a discretized
    grid that more finely resolves each covariate's range, so the fit approaches the exact
    (non-discretized) GAM fit at the cost of more unique bins `d` and therefore more memory and
    computation in the `X'WX` accumulation step. Smaller values reduce memory and speed up fitting,
    but coarsen the covariate resolution: because all observations within a bin share the same
    basis row, this can slightly bias smooths with high curvature or steep local features, since
    fine-scale variation within a bin is averaged away. The default of `200` is usually more than
    enough resolution for typical smooth terms; it rarely needs to be increased unless a covariate
    has an unusually large number of important local features. The benefit of discretization
    (versus plain `GAM`) is only realized once `n` is much larger than `n_discrete`, i.e. for large
    datasets — for small or moderate `n`, `GAM` is simpler and just as fast.

    Examples
    --------
    ```{python}
    import numpy as np
    import whittaker as wt
    from whittaker.bam import BigGAM

    rng = np.random.default_rng(0)
    n = 5_000
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x1) + x2**2 + rng.normal(scale=0.2, size=n)

    model = BigGAM("y ~ s(x1) + s(x2)", n_discrete=100).fit({"x1": x1, "x2": x2, "y": y})
    print(model.summary())
    ```

    This example uses a modest `n` for speed, but `BigGAM`'s memory and speed advantage over plain
    `GAM` really shows up once `n` reaches into the millions, where materializing the full design
    matrix would be impractical.
    """

    def __init__(
        self,
        formula: str | Formula,
        family: Family | None = None,
        *,
        n_discrete: int = 200,
    ) -> None:
        super().__init__(formula, family)
        self._n_discrete = n_discrete
        self._disc_model: DiscretizedModelMatrix | None = None

    @property
    def n_discrete(self) -> int:
        """Number of discretization grid points per covariate.

        This is the `n_discrete` value passed to `__init__`: the maximum number of unique
        representative values each covariate (or combination of covariates, for multi-dimensional
        smooths) is rounded onto before the smooth basis is evaluated. It bounds the number of
        unique rows `d` used when accumulating `X'WX` during fitting (see the class docstring),
        and therefore controls the tradeoff between fit accuracy and memory/speed.

        Returns
        -------
        int
            The discretization grid size configured for this model.
        """
        return self._n_discrete

    def fit(
        self,
        data: InputData,
        *,
        smoothing_params: list[float] | None = None,
        method: str = "fREML",
        weights: NDArray | None = None,
        select: bool = False,
    ) -> BigGAM:
        """Fit the BigGAM using discretized P-IRLS.

        Builds a `DiscretizedModelMatrix` via `build_discretized_model_matrix` and fits it with
        `bam_fit`, which accumulates `X'WX` and `X'Wz` from per-bin basis blocks instead of the
        full design matrix (see the class docstring for how the discretization and cross-product
        accumulation work).

        Parameters
        ----------
        data : dict[str, numpy.ndarray] or InputData
            Column-oriented data as `{name: 1-D array}` (or any `InputData`-compatible object).
            All columns referenced by the formula must be present and of equal length.
        smoothing_params : list of float, optional
            Fixed smoothing parameters `lambda_j`, one per smooth term, in formula order. If
            `None` (the default), smoothing parameters are selected automatically according to
            `method`.
        method : str
            Criterion used to select smoothing parameters when `smoothing_params` is `None`. One
            of `"fREML"` (default, fast discretized REML), `"REML"`, `"ML"`, or `"GCV"`. See
            `~whittaker.gam.GAM.fit` for a description of each criterion.
        weights : numpy.ndarray, optional
            Observation (prior) weights, shape `(n,)`. Must be strictly positive.
        select : bool
            If `True`, add an extra penalty on each smooth's null space so the term can be shrunk
            to exactly zero (double-penalty selection). Defaults to `False`.

        Returns
        -------
        BigGAM
            Returns `self` for method chaining, e.g. `model = BigGAM("y ~ s(x)").fit(data)`.
        """
        data = prepare_data(data)
        self._data = data

        self._disc_model = build_discretized_model_matrix(
            self._formula, data, n_discrete=self._n_discrete, select=select
        )

        pw = None
        if weights is not None:
            pw = np.asarray(weights, dtype=float)
            if pw.ndim != 1 or len(pw) != self._disc_model.n_obs:
                raise ValueError(
                    f"weights must be a 1-D array of length {self._disc_model.n_obs}, "
                    f"got shape {pw.shape}."
                )
            if np.any(pw <= 0):
                raise ValueError("All weights must be positive.")

        self._fit_result = bam_fit(
            self._disc_model,
            self._family,
            smoothing_params=smoothing_params,
            method=method,
            prior_weights=pw,
        )

        self._model_matrix = ModelMatrix(
            X=np.empty((0, self._disc_model.n_cols)),
            penalties=self._disc_model.penalties,
            smooths=self._disc_model.smooth_infos,
            column_names=self._disc_model.column_names,
            has_intercept=self._disc_model.has_intercept,
            n_parametric=self._disc_model.n_parametric,
            offset=self._disc_model.offset,
            offset_expressions=self._disc_model.offset_expressions,
            response=self._disc_model.response,
        )

        self._fitted = True
        return self

    def _expand_block_columns(self, block: DiscretizedBlock) -> NDArray:
        """Reconstruct the n x k column block from a discretized block."""
        cols = block.unique_basis[block.indices]
        if block.by_weights is not None:
            cols = cols * block.by_weights[:, None]
        return cols

    def smooth_tests(self):
        """Approximate significance tests for each smooth term.

        This is the `BigGAM` counterpart to `~whittaker.gam.GAM.smooth_tests`, adapted to the
        discretized fitting path: because `self._model_matrix.X` is never materialized, the
        per-term design columns needed for the test are reconstructed on demand from each
        smooth's `DiscretizedBlock` via `_expand_block_columns` (or read directly from
        `parametric_cols` for non-smooth columns) rather than sliced out of a dense `X`. For each
        smooth term, the coefficient sub-vector, its covariance sub-block, and its expanded
        design columns are passed to `~whittaker.fitting.inference._smooth_test` to obtain an
        approximate chi-squared test of whether the term is uniformly zero.

        Returns
        -------
        list of SmoothTestResult
            One result per smooth term (or per `by=` level), each with `term_label`, `stat`,
            `edf`, `ref_df`, and `p_value` attributes.
        """
        from whittaker.fitting.inference import SmoothTestResult, _smooth_test

        self._check_fitted()
        assert self._disc_model is not None
        dm = self._disc_model
        fit = self._fit_result

        w = fit.weights
        if fit.prior_weights is not None and w is None:
            w = fit.prior_weights
        wt = w if w is not None else np.ones(dm.n_obs)

        XtWX = _compute_XtWX(dm, wt)
        p = dm.n_cols
        S_total = np.zeros((p, p))
        for lam, pen in zip(fit.smoothing_params, dm.penalties, strict=False):
            S_total += lam * pen
        A = XtWX + S_total
        A = (A + A.T) * 0.5
        eigvals, eigvecs = np.linalg.eigh(A)
        tol = np.max(eigvals) * p * np.finfo(float).eps
        eigvals_inv = np.zeros_like(eigvals)
        keep = eigvals > tol
        eigvals_inv[keep] = 1.0 / eigvals[keep]
        V_beta = fit.scale * (eigvecs * eigvals_inv) @ eigvecs.T

        results = []
        block_map = {(b.col_start, b.col_end): b for b in dm.blocks}
        for idx, info in enumerate(dm.smooth_infos):
            cs, ce = info.col_start, info.col_end
            beta_j = fit.coefficients[cs:ce]
            V_j = V_beta[cs:ce, cs:ce]
            edf_j = fit.edf[idx]

            block = block_map.get((cs, ce))
            if block is not None:
                X_j = self._expand_block_columns(block)
            elif dm.parametric_cols is not None and ce <= dm.n_param_cols:  # pragma: no cover
                # Defensive fallback: `dm.blocks` and `dm.smooth_infos` are always built as a 1:1
                # pair with matching (col_start, col_end) ranges in
                # `build_discretized_model_matrix`, so every smooth's columns are always found via
                # `block_map` above. This branch guards against a hypothetical future change that
                # breaks that invariant; it is unreachable through the current public API.
                X_j = dm.parametric_cols[:, cs:ce]
            else:  # pragma: no cover
                X_j = np.zeros((dm.n_obs, ce - cs))

            stat, ref_df, pval = _smooth_test(beta_j, V_j, X_j, edf_j)
            label = repr(info.term)
            if info.by_level is not None:
                label = f"{label}:{info.by_level}"
            results.append(
                SmoothTestResult(
                    term_label=label, stat=stat, edf=edf_j, ref_df=ref_df, p_value=pval
                )
            )
        return results
