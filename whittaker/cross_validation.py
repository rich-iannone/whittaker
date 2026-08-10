"""K-fold cross-validation for GAMs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from whittaker.families.base import Family
from whittaker.families.gaussian import Gaussian
from whittaker.gam import GAM


@dataclass
class CVResult:
    """Cross-validation result.

    Attributes
    ----------
    cv_score:
        Mean out-of-sample loss (deviance or MSE, depending on the family).
    cv_scores:
        Per-fold out-of-sample loss values.
    cv_se:
        Standard error of the mean CV score across folds.
    n_folds:
        Number of folds used.
    """

    cv_score: float
    cv_scores: NDArray
    cv_se: float
    n_folds: int


def cross_validate(
    formula: str,
    data: dict[str, NDArray],
    *,
    family: Family | None = None,
    n_folds: int = 10,
    method: str = "GCV",
    metric: str = "deviance",
    select: bool = False,
    seed: int | None = None,
) -> CVResult:
    """K-fold cross-validation for a GAM specification.

    Parameters
    ----------
    formula:
        GAM formula string.
    data:
        Column-oriented data dict.
    family:
        Response distribution family. Defaults to `Gaussian()`.
    n_folds:
        Number of CV folds.
    method:
        Smoothing parameter selection method used when fitting each fold.
    metric:
        Loss metric: `"deviance"` (default) or `"mse"`.
    select:
        Whether to add shrinkage penalties for smooth selection.
    seed:
        Random seed for fold assignment.

    Returns
    -------
    CVResult
    """
    if family is None:
        family = Gaussian()

    rng = np.random.default_rng(seed)
    response = formula.split("~")[0].strip()
    y = np.asarray(data[response], dtype=float)
    n = len(y)

    perm = rng.permutation(n)
    fold_ids = np.zeros(n, dtype=int)
    for i in range(n):
        fold_ids[perm[i]] = i * n_folds // n

    arrays = {k: np.asarray(data[k], dtype=float) for k in data}

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
