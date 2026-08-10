"""BigGAM: large-scale GAM fitting via discretized covariates.

Provides `BigGAM`, a drop-in replacement for `~whittaker.gam.GAM` that avoids materializing the full
*n x p* design matrix. Instead, each covariate is rounded to a grid of *n_discrete* representative
values and the basis functions are evaluated only at the unique grid points. This makes memory usage
roughly *O(d p)* instead of *O(n p)* (where *d << n*) and speeds up the X'WX accumulation
proportionally.

Use BigGAM when the standard GAM runs out of memory or is too slow because *n* is large (>1M).
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

    Follows the same logic as :func:`~whittaker.model_matrix.build_model_matrix` but evaluates each
    smooth's basis only at unique discretized covariate values, producing a
    `~whittaker.fitting.bam.DiscretizedModelMatrix` that never stores the full *n x p* design
    matrix.
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

        basis.fit(x)
        nsd = basis.null_space_dimension()

        unique_x, disc_indices = _discretize_nd(x, n_discrete)
        unique_basis_mat = basis.basis_matrix(unique_x)

        if hasattr(basis, "penalty_matrices"):
            pen_mats = basis.penalty_matrices()
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
    """GAM for large datasets using discretized fitting.

    Uses the bam approach (Wood, Li, & Shaddick, 2017): discretizes covariates to a grid of
    *n_discrete* points and evaluates basis functions only at the unique grid values. The full
    *n x p* design matrix is never materialized.

    Parameters
    ----------
    formula:
        Model formula (same as `~whittaker.gam.GAM`).
    family:
        Response distribution family.
    n_discrete:
        Number of grid points per covariate for discretization. Higher values give results closer
        to the exact GAM at the cost of more memory and computation.
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
        """Number of discretization grid points per covariate."""
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

        Parameters
        ----------
        data:
            Column-oriented data (dict, DataFrame, etc.).
        smoothing_params:
            Fixed smoothing parameters. If `None`, selected automatically.
        method:
            Smoothing selection method: `"fREML"` (default), `"REML"`, `"ML"`, or `"GCV"``.
        weights:
            Observation weights, shape `(n,)`.
        select:
            If `True`, enable double-penalty variable selection.

        Returns
        -------
        BigGAM
            Returns `self` for method chaining.
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
        """Approximate significance tests for smooth terms (discretized version)."""
        from whittaker.fitting.inference import SmoothTestResult, _smooth_test

        self._check_fitted()
        dm = self._disc_model
        fit = self._fit_result

        w = fit.weights
        if fit.prior_weights is not None and w is None:
            w = fit.prior_weights
        wt = w if w is not None else np.ones(dm.n_obs)

        XtWX = _compute_XtWX(dm, wt)
        p = dm.n_cols
        S_total = np.zeros((p, p))
        for lam, pen in zip(fit.smoothing_params, dm.penalties):
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
            elif dm.parametric_cols is not None and ce <= dm.n_param_cols:
                X_j = dm.parametric_cols[:, cs:ce]
            else:
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
