r"""Sigma calibration for quantile GAMs (Fasiolo et al. 2021)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from whittaker.data import InputData, prepare_data
from whittaker.families.quantile import QuantileFamily, _elf_loss
from whittaker.gam import GAM


def calibrate_sigma(
    formula: str,
    data: InputData,
    tau: float = 0.5,
    *,
    n_folds: int = 5,
    sigma_values: NDArray | list[float] | None = None,
    method: str = "GCV",
    seed: int | None = None,
) -> float:
    r"""Find the ELF bandwidth sigma that minimises out-of-sample ELF loss.

    The quantile family used by `QuantileGAM` and `GAM` with quantile loss replaces the
    non-differentiable check function with a smooth "extended log-F" (ELF) surrogate controlled by a
    bandwidth `sigma`. Too large a `sigma` over-smooths the check loss and biases the fitted
    quantile; too small a `sigma` makes the surrogate nearly non-differentiable again and can
    destabilize IRLS. `calibrate_sigma` selects `sigma` empirically by K-fold cross-validation: for
    each candidate value, the model is fit on `K - 1` folds, predictions are made on the held-out
    fold, and the true (non-smoothed) pinball loss

    $$\rho_\tau(y - \hat q_\tau) = (y - \hat q_\tau)\,(\tau - \mathbb{1}[y < \hat q_\tau])$$

    is accumulated across folds. The sigma minimizing total out-of-sample pinball loss is refined
    with a second, finer grid search around the best value from the coarse grid.

    Parameters
    ----------
    formula:
        GAM formula string, e.g. `"y ~ s(x)"`.
    data:
        Column-oriented data dict.
    tau:
        Target quantile level in `(0, 1)`.
    n_folds:
        Number of CV folds.
    sigma_values:
        Candidate sigma values to evaluate. If `None`, a log-spaced grid of 10 values from
        `0.01 * sd(y)` to `2 * sd(y)` is used, followed by a refinement grid around the best value.
        If an explicit grid is passed, no refinement step is performed.
    method:
        Smoothing parameter selection method used when fitting each candidate model (`"GCV"`,
        `"REML"`, `"ML"`).
    seed:
        Random seed for fold assignment.

    Returns
    -------
    float
        Calibrated sigma value (the one minimising CV pinball loss).

    Examples
    --------
    ```{python}
    import numpy as np
    from whittaker.calibration import calibrate_sigma

    rng = np.random.default_rng(0)
    n = 400
    x = rng.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x) + rng.normal(scale=0.2 + 0.3 * x, size=n)

    best_sigma = calibrate_sigma("y ~ s(x)", {"x": x, "y": y}, tau=0.9, n_folds=5, seed=0)
    print(round(best_sigma, 4))
    ```
    """
    arrays = prepare_data(data)
    rng = np.random.default_rng(seed)
    response = formula.split("~")[0].strip()
    y = arrays[response]
    n = len(y)

    fold_ids = rng.integers(0, n_folds, size=n)

    custom_grid = sigma_values is not None
    if custom_grid:
        sigma_values = np.asarray(sigma_values, dtype=float)
    else:
        sd_y = max(np.std(y), 1e-4)
        sigma_values = np.exp(np.linspace(np.log(0.01 * sd_y), np.log(2.0 * sd_y), 10))

    def _cv_loss(sigma: float) -> float:
        family = QuantileFamily(tau=tau, sigma=sigma)
        total_loss = 0.0
        for fold in range(n_folds):
            train = fold_ids != fold
            test = fold_ids == fold
            if not np.any(test):
                continue
            train_data = {k: v[train] for k, v in arrays.items()}
            test_data = {k: v[test] for k, v in arrays.items()}
            model = GAM(formula, family=family)
            try:
                model.fit(train_data, method=method)
                pred = model.predict(test_data).values
            except Exception:
                return float("inf")
            total_loss += float(np.sum(_elf_loss(y[test] - pred, tau, sigma)))
        return total_loss

    losses = np.array([_cv_loss(s) for s in sigma_values])
    best_idx = int(np.argmin(losses))

    if custom_grid or len(sigma_values) < 3:
        return float(sigma_values[best_idx])

    lo_idx = max(0, best_idx - 1)
    hi_idx = min(len(sigma_values) - 1, best_idx + 1)
    fine_grid = np.exp(np.linspace(np.log(sigma_values[lo_idx]), np.log(sigma_values[hi_idx]), 5))
    fine_losses = np.array([_cv_loss(s) for s in fine_grid])
    return float(fine_grid[int(np.argmin(fine_losses))])
