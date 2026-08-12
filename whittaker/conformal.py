r"""Conformal prediction intervals for GAMs.

Provides distribution-free prediction intervals with finite-sample coverage guarantees. Three
methods are available:

- **Split conformal**: fits on a training split, calibrates residual quantiles on the calibration
split, produces intervals on new data.
- **CV+ (cross-validation+)**: uses leave-one-out or K-fold residuals from the full dataset for
tighter intervals (Barber et al. 2021).
- **Jackknife+**: uses leave-one-out refits for valid marginal coverage (Barber et al. 2021).

All methods wrap around the existing `GAM` fitting infrastructure and return `ConformalResult`
objects that integrate with `PredictionResult`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from whittaker.data import InputData, prepare_data
from whittaker.families.base import Family
from whittaker.families.gaussian import Gaussian
from whittaker.gam import GAM, PredictionResult


class ConformalMethod(Enum):
    SPLIT = "split"
    CV_PLUS = "cv+"
    JACKKNIFE_PLUS = "jackknife+"


@dataclass
class ConformalResult:
    r"""Result of conformal prediction.

    Returned by `ConformalPredictor.predict()`. Holds point predictions together with
    distribution-free prediction intervals whose coverage is guaranteed (under exchangeability) to
    be at least the nominal `level`, regardless of whether the underlying `GAM` is correctly
    specified.

    Attributes
    ----------
    values:
        Point predictions (response scale); for `"cv+"` and `"jackknife+"` this is the average of
        the fold/leave-one-out models' predictions.
    lower:
        Lower prediction bounds.
    upper:
        Upper prediction bounds.
    level:
        Nominal coverage level (e.g. `0.95`).
    method:
        Conformal method used (`"split"`, `"cv+"`, or `"jackknife+"`).
    calibration_scores:
        Conformity scores (absolute residuals) from the calibration step.
    quantile:
        The calibration quantile used for interval width (only meaningful for the `"split"` method,
        where the interval is `values +/- quantile`; for `"cv+"`/`"jackknife+"` interval bounds vary
        per observation and are not simply `values +/- quantile`).
    """

    values: NDArray
    lower: NDArray
    upper: NDArray
    level: float
    method: str
    calibration_scores: NDArray
    quantile: float


@dataclass
class ConformalPredictor:
    r"""A calibrated conformal predictor ready to produce intervals.

    Created by `conformal_fit()`. Wraps a fitted `GAM` (or, for `"cv+"`/`"jackknife+"`, an ensemble
    of fold/leave-one-out GAMs) together with the conformity scores from calibration, so that
    `predict()` on new data returns intervals with a finite-sample marginal coverage guarantee that
    does not rely on the GAM's error distribution being correctly specified — only on the
    calibration and test data being exchangeable.

    Attributes
    ----------
    model:
        The GAM used for point predictions: fit on the training split for `"split"`, or on the full
        data for `"cv+"`/`"jackknife+"` (in which case predictions are instead ensembled from the
        per-fold/per-observation models).
    calibration_scores:
        Absolute residuals from the calibration step (calibration split, K-fold, or leave-one-out,
        depending on `method`).
    quantile:
        The calibration quantile of `calibration_scores` used to set interval half-width (`"split"`
        only).
    level:
        Nominal coverage level.
    method:
        Conformal method: `"split"`, `"cv+"`, or `"jackknife+"`.
    """

    model: GAM
    calibration_scores: NDArray
    quantile: float
    level: float
    method: str
    _models: list[GAM] | None = None
    _fold_preds: NDArray | None = None
    _fold_ids: NDArray | None = None

    def predict(self, new_data: InputData) -> ConformalResult:
        r"""Produce conformal prediction intervals on new data.

        Dispatches to the interval construction appropriate for `self.method`: simple
        `values +/- quantile` bands for `"split"`, or the min/max-based `"cv+"` and `"jackknife+"`
        constructions that combine every fold/leave-one-out model's prediction with every
        calibration residual.

        Parameters
        ----------
        new_data:
            Column-oriented covariate data.

        Returns
        -------
        ConformalResult
        """
        if self.method == ConformalMethod.CV_PLUS.value and self._models is not None:
            return self._predict_cv_plus(new_data)
        if self.method == ConformalMethod.JACKKNIFE_PLUS.value and self._models is not None:
            return self._predict_jackknife_plus(new_data)
        return self._predict_split(new_data)

    def _predict_split(self, new_data: InputData) -> ConformalResult:
        new_data = prepare_data(new_data)
        pred = self.model.predict(new_data)
        assert isinstance(pred, PredictionResult)
        mu = pred.values
        return ConformalResult(
            values=mu,
            lower=mu - self.quantile,
            upper=mu + self.quantile,
            level=self.level,
            method=self.method,
            calibration_scores=self.calibration_scores,
            quantile=self.quantile,
        )

    def _predict_cv_plus(self, new_data: InputData) -> ConformalResult:
        new_data = prepare_data(new_data)
        assert self._models is not None
        len(self.calibration_scores)
        n_folds = len(self._models)
        n_new = len(next(iter(new_data.values())))

        fold_preds = np.zeros((n_folds, n_new))
        for k, m in enumerate(self._models):
            p = m.predict(new_data)
            assert isinstance(p, PredictionResult)
            fold_preds[k] = p.values

        mu_ensemble = np.mean(fold_preds, axis=0)

        lower = np.zeros(n_new)
        upper = np.zeros(n_new)
        alpha = 1.0 - self.level
        q_lo = alpha / 2
        q_hi = 1.0 - alpha / 2

        for j in range(n_new):
            intervals_lo = fold_preds[:, j].min() - self.calibration_scores
            intervals_hi = fold_preds[:, j].max() + self.calibration_scores
            lower[j] = np.quantile(intervals_lo, q_lo)
            upper[j] = np.quantile(intervals_hi, q_hi)

        return ConformalResult(
            values=mu_ensemble,
            lower=lower,
            upper=upper,
            level=self.level,
            method=self.method,
            calibration_scores=self.calibration_scores,
            quantile=self.quantile,
        )

    def _predict_jackknife_plus(self, new_data: InputData) -> ConformalResult:
        new_data = prepare_data(new_data)
        assert self._models is not None
        n_loo = len(self._models)
        n_new = len(next(iter(new_data.values())))

        loo_preds = np.zeros((n_loo, n_new))
        for i, m in enumerate(self._models):
            p = m.predict(new_data)
            assert isinstance(p, PredictionResult)
            loo_preds[i] = p.values

        mu_ensemble = np.mean(loo_preds, axis=0)
        residuals = self.calibration_scores

        lower = np.zeros(n_new)
        upper = np.zeros(n_new)
        q_level = np.ceil((1.0 - (1.0 - self.level) / 2) * (n_loo + 1)) / n_loo
        q_level = min(q_level, 1.0)

        for j in range(n_new):
            lo_vals = loo_preds[:, j] - residuals
            hi_vals = loo_preds[:, j] + residuals
            lower[j] = np.quantile(lo_vals, 1.0 - q_level)
            upper[j] = np.quantile(hi_vals, q_level)

        return ConformalResult(
            values=mu_ensemble,
            lower=lower,
            upper=upper,
            level=self.level,
            method=self.method,
            calibration_scores=self.calibration_scores,
            quantile=self.quantile,
        )


def conformal_fit(
    formula: str,
    data: InputData,
    *,
    method: str = "split",
    family: Family | None = None,
    level: float = 0.95,
    cal_fraction: float = 0.25,
    n_folds: int = 10,
    fit_method: str = "REML",
    select: bool = False,
    seed: int | None = None,
) -> ConformalPredictor:
    r"""Fit a GAM with conformal calibration.

    Fits a `GAM` and calibrates it so that `ConformalPredictor.predict()` returns prediction
    intervals with a distribution-free, finite-sample marginal coverage guarantee, using one of
    three conformal methods:

    - **Split conformal**: the data is randomly split into a training set (fit the GAM) and a
      calibration set (compute absolute residuals). The interval half-width is the
      `ceil((n_cal+1) * level) / n_cal` empirical quantile of the calibration residuals, giving
      intervals of constant width `values +/- quantile`. Simple and fast, but "wastes" data on the
      calibration split and can be less efficient than the alternatives.
    - **CV+**: the data is split into `n_folds` folds; each fold is used to compute out-of-fold
      residuals from a model trained on the rest. At prediction time, all fold models' predictions
      are combined with all fold residuals via a min/max construction (Barber et al. 2021), giving
      tighter, per-observation intervals without a dedicated calibration split.
    - **Jackknife+**: the leave-one-out analogue of CV+, using `n` individual leave-one-out refits.
      Provides the tightest intervals of the three but is the most computationally expensive since
      it requires `n` refits.

    Parameters
    ----------
    formula:
        GAM formula string.
    data:
        Column-oriented data dict.
    method:
        Conformal method: `"split"` (default), `"cv+"`, or `"jackknife+"`.
    family:
        Response distribution family. Defaults to `Gaussian()`.
    level:
        Nominal coverage probability (default `0.95`).
    cal_fraction:
        Fraction of data held out for calibration in the split method (default `0.25`). Ignored for
        `"cv+"` and `"jackknife+"`.
    n_folds:
        Number of folds for the `"cv+"` method (default `10`). Ignored for `"split"` and
        `"jackknife+"`.
    fit_method:
        Smoothing parameter selection method for the GAM (default `"REML"`).
    select:
        If `True`, enable double-penalty variable selection.
    seed:
        Random seed for data splitting.

    Notes
    -----
    Split conformal computes the calibration quantile as

    $$\hat q = \left\lceil (n_{\text{cal}} + 1) \cdot \text{level} \right\rceil \big/ n_{\text{cal}}
    \quad \text{quantile of} \quad \{|y_i - \hat\mu(x_i)| : i \in \text{calibration set}\},$$

    which, under exchangeability of calibration and test points, guarantees
    `P(y \in [\hat\mu(x) - \hat q, \hat\mu(x) + \hat q]) \ge \text{level}` marginally over new
    draws. CV+ and jackknife+ replace this single quantile with, for each test point, the
    appropriate quantile of the `n` (or `n_folds`) values
    `{fold/LOO prediction +/- that fold's residual}`,
    trading extra computation for tighter, locally-adapted intervals while retaining the same
    finite-sample coverage guarantee.

    Returns
    -------
    ConformalPredictor
        A calibrated predictor that can produce intervals on new data.

    Examples
    --------
    ```{python}
    import numpy as np
    from whittaker.conformal import conformal_fit, conformal_coverage

    rng = np.random.default_rng(0)
    n = 500
    x = rng.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x) + rng.normal(scale=0.3, size=n)

    predictor = conformal_fit("y ~ s(x)", {"x": x, "y": y}, method="split", level=0.9, seed=0)
    result = predictor.predict({"x": x[:5]})
    print(result.lower, result.upper)
    print(conformal_coverage(predictor, {"x": x, "y": y}, response="y"))
    ```
    """
    method_lower = method.lower()
    if method_lower not in ("split", "cv+", "jackknife+"):
        raise ValueError(
            f"Unknown conformal method {method!r}. Choose from 'split', 'cv+', or 'jackknife+'."
        )

    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1), got {level}")

    if family is None:
        family = Gaussian()

    arrays = prepare_data(data)
    rng = np.random.default_rng(seed)
    response = formula.split("~")[0].strip()
    y = arrays[response]
    n = len(y)

    if method_lower == "split":
        return _split_conformal(
            formula, arrays, y, n, family, level, cal_fraction, fit_method, select, rng
        )
    elif method_lower == "cv+":
        return _cv_plus_conformal(
            formula, arrays, y, n, family, level, n_folds, fit_method, select, rng
        )
    else:
        return _jackknife_plus_conformal(
            formula, arrays, y, n, family, level, fit_method, select, rng
        )


def _split_conformal(
    formula: str,
    arrays: dict[str, NDArray],
    y: NDArray,
    n: int,
    family: Family,
    level: float,
    cal_fraction: float,
    fit_method: str,
    select: bool,
    rng: np.random.Generator,
) -> ConformalPredictor:
    perm = rng.permutation(n)
    n_cal = max(1, int(n * cal_fraction))
    cal_idx = perm[:n_cal]
    train_idx = perm[n_cal:]

    train_data = {k: v[train_idx] for k, v in arrays.items()}
    cal_data = {k: v[cal_idx] for k, v in arrays.items()}

    model = GAM(formula, family=family)
    model.fit(train_data, method=fit_method, select=select)

    cal_pred_result = model.predict(cal_data)
    assert isinstance(cal_pred_result, PredictionResult)
    cal_pred = cal_pred_result.values
    cal_residuals = np.abs(y[cal_idx] - cal_pred)

    q = np.ceil((n_cal + 1) * level) / n_cal
    q = min(q, 1.0)
    quantile_val = float(np.quantile(cal_residuals, q))

    return ConformalPredictor(
        model=model,
        calibration_scores=cal_residuals,
        quantile=quantile_val,
        level=level,
        method=ConformalMethod.SPLIT.value,
    )


def _cv_plus_conformal(
    formula: str,
    arrays: dict[str, NDArray],
    y: NDArray,
    n: int,
    family: Family,
    level: float,
    n_folds: int,
    fit_method: str,
    select: bool,
    rng: np.random.Generator,
) -> ConformalPredictor:
    perm = rng.permutation(n)
    fold_ids = np.zeros(n, dtype=int)
    for i in range(n):
        fold_ids[perm[i]] = i * n_folds // n

    loo_residuals = np.zeros(n)
    fold_models: list[GAM] = []

    for fold in range(n_folds):
        train = fold_ids != fold
        test = fold_ids == fold

        train_data = {k: v[train] for k, v in arrays.items()}
        test_data = {k: v[test] for k, v in arrays.items()}

        model = GAM(formula, family=family)
        model.fit(train_data, method=fit_method, select=select)
        fold_models.append(model)

        fold_pred = model.predict(test_data)
        assert isinstance(fold_pred, PredictionResult)
        loo_residuals[test] = np.abs(y[test] - fold_pred.values)

    full_model = GAM(formula, family=family)
    full_model.fit(arrays, method=fit_method, select=select)

    q = np.ceil((n + 1) * level) / n
    q = min(q, 1.0)
    quantile_val = float(np.quantile(loo_residuals, q))

    return ConformalPredictor(
        model=full_model,
        calibration_scores=loo_residuals,
        quantile=quantile_val,
        level=level,
        method=ConformalMethod.CV_PLUS.value,
        _models=fold_models,
        _fold_ids=fold_ids,
    )


def _jackknife_plus_conformal(
    formula: str,
    arrays: dict[str, NDArray],
    y: NDArray,
    n: int,
    family: Family,
    level: float,
    fit_method: str,
    select: bool,
    rng: np.random.Generator,
) -> ConformalPredictor:
    loo_residuals = np.zeros(n)
    loo_models: list[GAM] = []

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        train_data = {k: v[mask] for k, v in arrays.items()}

        model = GAM(formula, family=family)
        model.fit(train_data, method=fit_method, select=select)
        loo_models.append(model)

        single = {k: v[i : i + 1] for k, v in arrays.items()}
        single_pred = model.predict(single)
        assert isinstance(single_pred, PredictionResult)
        loo_residuals[i] = abs(y[i] - single_pred.values[0])

    full_model = GAM(formula, family=family)
    full_model.fit(arrays, method=fit_method, select=select)

    q = np.ceil((n + 1) * level) / n
    q = min(q, 1.0)
    quantile_val = float(np.quantile(loo_residuals, q))

    return ConformalPredictor(
        model=full_model,
        calibration_scores=loo_residuals,
        quantile=quantile_val,
        level=level,
        method=ConformalMethod.JACKKNIFE_PLUS.value,
        _models=loo_models,
    )


def conformal_coverage(
    predictor: ConformalPredictor,
    data: InputData,
    response: str,
) -> float:
    r"""Compute empirical coverage of conformal intervals on held-out data.

    A useful sanity check that the realized coverage on a given dataset is close to (at least) the
    nominal `predictor.level`; systematic under-coverage may indicate a violation of the
    exchangeability assumption underlying conformal prediction (e.g. distribution shift between
    calibration and test data).

    Parameters
    ----------
    predictor:
        A fitted `ConformalPredictor`.
    data:
        Data containing both covariates and the response.
    response:
        Name of the response variable.

    Returns
    -------
    float
        Fraction of observations falling within the conformal interval.
    """
    arrays = prepare_data(data)
    y = arrays[response]
    result = predictor.predict(data)
    return float(np.mean((y >= result.lower) & (y <= result.upper)))
