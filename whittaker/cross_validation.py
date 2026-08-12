"""K-fold cross-validation for GAMs.

This module provides an out-of-sample alternative to the fit-time smoothing-parameter selection
criteria (GCV, REML, ML). `cross_validate()` repeatedly refits a `~whittaker.gam.GAM` on training
folds and scores each fit on the held-out fold, returning per-fold and aggregate loss values in a
`CVResult`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from whittaker.data import InputData, prepare_data
from whittaker.families.base import Family
from whittaker.families.gaussian import Gaussian
from whittaker.gam import GAM


@dataclass
class CVResult:
    """Result of `cross_validate()`.

    Holds the per-fold and aggregate out-of-sample loss values produced by k-fold
    cross-validation of a `~whittaker.gam.GAM` specification, along with the number of folds used
    to obtain them. Use `cv_score` as a single summary number for model comparison, and `cv_scores`
    together with `cv_se` to gauge how much that summary varies across folds.

    Parameters
    ----------
    cv_score : float
        Mean out-of-sample loss across all folds, i.e. `numpy.mean(cv_scores)`. This is on the
        scale of whichever `metric` was requested from `cross_validate()` — mean deviance per test
        observation for `metric="deviance"`, or mean squared error on the response scale for
        `metric="mse"`. Lower values indicate better out-of-sample predictive performance; use this
        value to compare competing formulas, families, or fitting methods evaluated on the same
        data and folds.
    cv_scores : numpy.ndarray
        Per-fold out-of-sample loss values, shape `(n_folds,)`. Element `i` is the loss computed by
        fitting the GAM on every fold except `i` and scoring it on fold `i`. Inspect this array
        directly to check whether the CV estimate is driven by a small number of unusual folds.
    cv_se : float
        Standard error of the mean CV score across folds, computed as the sample standard
        deviation of `cv_scores` (with Bessel's correction, `ddof=1`) divided by
        `sqrt(n_folds)`. Provides a rough measure of the uncertainty in `cv_score` due to the
        particular random fold assignment; useful for judging whether a difference in `cv_score`
        between two models is likely to be meaningful.
    n_folds : int
        Number of folds actually requested when this result was produced. Matches the `n_folds`
        argument passed to `cross_validate()`.
    """

    cv_score: float
    cv_scores: NDArray
    cv_se: float
    n_folds: int


def cross_validate(
    formula: str,
    data: InputData,
    *,
    family: Family | None = None,
    n_folds: int = 10,
    method: str = "GCV",
    metric: str = "deviance",
    select: bool = False,
    seed: int | None = None,
) -> CVResult:
    r"""K-fold cross-validation for a GAM specification.

    Estimates the out-of-sample predictive performance of a GAM by repeatedly refitting it on
    `n_folds - 1` folds of the data and scoring the fit on the remaining held-out fold, then
    aggregating the resulting losses across folds. Because each fold is scored on data that was
    not used to fit that particular model, `cross_validate()` gives a more honest estimate of
    generalization error than simply scoring a single fit on the data it was trained on, which is
    optimistic (the fit has already adapted to that data's noise). Use it to compare candidate
    formulas, families, fitting methods, or basis choices on equal footing, or as a sanity check
    that a model selected by GCV/REML/ML also performs well out of sample.

    Two loss metrics are available via `metric`:

    - `"deviance"` (default): for each fold, the family's `deviance()` between the held-out
      responses and the predictions from a model fit on the training folds, divided by the number
      of test observations in that fold. This matches the deviance-based loss the family itself
      uses during fitting, so it is comparable across `method` choices for the same `family`.
    - `"mse"`: the mean squared error, `mean((y_test - pred) ** 2)`, on the response scale. This is
      family-agnostic and directly interpretable in the response's original units, but does not
      account for family-specific variance structure the way deviance does.

    Folds are constructed by randomly permuting the row indices with `rng.permutation(n)` (where
    `rng` is seeded from `seed`) and then assigning fold id `i * n_folds // n` to the row that ends
    up in position `i` of the permutation. This produces a non-stratified partition into `n_folds`
    contiguous-in-permutation-order, roughly (but not exactly, when `n` is not a multiple of
    `n_folds`) equal-size groups; no attempt is made to balance the distribution of the response or
    any covariate across folds.

    Parameters
    ----------
    formula : str
        GAM formula string, e.g. `"y ~ s(x1) + s(x2) + x3"`. The response named on the left-hand
        side is looked up in `data` to build the fold assignment and to compute the loss; the
        right-hand side is passed unchanged to `~whittaker.gam.GAM` for every fold.
    data : dict[str, numpy.ndarray] or InputData
        Column-oriented data as `{name: 1-D array}` (or any `InputData`-compatible object, such as
        a `pandas.DataFrame` or `polars.DataFrame`). Must contain every column referenced by
        `formula`, all of equal length.
    family : Family, optional
        Response distribution family, e.g. `Gaussian()`, `Binomial()`, `Poisson()`, `Gamma()`, or
        `Tweedie()`. Used both to fit each fold's `~whittaker.gam.GAM` and, when
        `metric="deviance"`, to compute each fold's loss via `family.deviance()`. Defaults to
        `Gaussian()`.
    n_folds : int
        Number of folds to split the data into. Must be at least 2 and, for every fold to receive
        at least one test observation, should not exceed the number of rows in `data`. Defaults to
        `10`. See the Notes section below for guidance on choosing this value.
    method : str
        Smoothing-parameter selection method passed through to `GAM.fit()` for every fold. One of
        `"GCV"` (default), `"REML"`, or `"ML"`; see `GAM.fit()` for what each criterion optimizes.
    metric : str
        Loss metric to compute on each held-out fold: `"deviance"` (default) or `"mse"`. See the
        discussion above for exactly how each is computed.
    select : bool
        Whether to add shrinkage penalties for automatic smooth-term selection, forwarded to
        `GAM.fit(select=...)` for every fold. Defaults to `False`.
    seed : int, optional
        Seed for the `numpy.random.default_rng()` random number generator used to build the fold
        assignment. Pass a fixed integer to make the fold split (and hence the resulting
        `CVResult`) reproducible across calls; `None` (the default) uses a fresh, non-reproducible
        seed.

    Returns
    -------
    CVResult
        Cross-validation result holding the mean out-of-sample loss (`cv_score`), the per-fold
        losses (`cv_scores`), their standard error (`cv_se`), and the number of folds used
        (`n_folds`).

    Notes
    -----
    The number of folds controls a bias-variance tradeoff in the CV estimate itself. With a small
    `n_folds` (e.g. `3`-`5`), each training fold omits a large fraction of the data, so the fitted
    model is somewhat different from (typically smoother/less flexible than) a model fit on the
    full dataset; the resulting `cv_score` tends to be pessimistically biased, but because there
    are only a few, relatively large folds, `cv_scores` tends to have lower variance across
    repeated runs. With a large `n_folds` (up to the leave-one-out limit, `n_folds = n`), each
    training fold is nearly the full dataset, so bias shrinks toward the true generalization error
    of the full-data fit — but the individual test folds are tiny (a single point at
    `n_folds = n`), so `cv_scores` becomes noisier (higher variance), and fitting cost grows
    linearly with `n_folds` since a full `GAM.fit()` is performed once per fold. In practice,
    `n_folds = 5` or `n_folds = 10` are common compromises between these effects. Leave-one-out
    cross-validation is rarely used directly for GAMs because of its cost; `method="GCV"` in
    `GAM.fit()` already computes an efficient analytical approximation to the leave-one-out error
    from a single fit, without refitting the model `n` times.

    Examples
    --------
    ```{python}
    import numpy as np
    import whittaker as wt

    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 1, 200))
    y = np.sin(2 * np.pi * x) + rng.normal(scale=0.2, size=200)

    result = wt.cross_validate("y ~ s(x)", {"x": x, "y": y}, n_folds=5, seed=0)
    print(result.cv_score, result.cv_se)
    ```

    ```{python}
    # Compare two candidate formulas on the same folds using the seed.
    linear_result = wt.cross_validate("y ~ x", {"x": x, "y": y}, n_folds=5, seed=0)
    smooth_result = wt.cross_validate("y ~ s(x)", {"x": x, "y": y}, n_folds=5, seed=0)
    linear_result.cv_score, smooth_result.cv_score
    ```
    """
    if family is None:
        family = Gaussian()

    arrays = prepare_data(data)
    rng = np.random.default_rng(seed)
    response = formula.split("~")[0].strip()
    y = arrays[response]
    n = len(y)

    perm = rng.permutation(n)
    fold_ids = np.zeros(n, dtype=int)
    for i in range(n):
        fold_ids[perm[i]] = i * n_folds // n

    fold_losses = np.zeros(n_folds)

    for fold in range(n_folds):
        train = fold_ids != fold
        test = fold_ids == fold
        n_test = int(np.sum(test))
        if n_test == 0:
            continue

        train_data = {k: v[train] for k, v in arrays.items()}
        test_data = {k: v[test] for k, v in arrays.items()}

        model = GAM(formula, family=family)
        model.fit(train_data, method=method, select=select)
        pred = model.predict(test_data).values

        if metric == "mse":
            fold_losses[fold] = float(np.mean((y[test] - pred) ** 2))
        else:
            fold_losses[fold] = family.deviance(y[test], pred) / n_test

    cv_score = float(np.mean(fold_losses))
    cv_se = float(np.std(fold_losses, ddof=1) / np.sqrt(n_folds))

    return CVResult(
        cv_score=cv_score,
        cv_scores=fold_losses,
        cv_se=cv_se,
        n_folds=n_folds,
    )
