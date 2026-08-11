"""Conformal prediction intervals for GAMs.

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
from whittaker.gam import GAM


class ConformalMethod(Enum):
    SPLIT = "split"
    CV_PLUS = "cv+"
    JACKKNIFE_PLUS = "jackknife+"


@dataclass
class ConformalResult:
    """Result of conformal prediction.

    Attributes
    ----------
    values:
        Point predictions (response scale).
    lower:
        Lower prediction bounds.
    upper:
        Upper prediction bounds.
    level:
        Nominal coverage level.
    method:
        Conformal method used.
    calibration_scores:
        Conformity scores from the calibration step.
    quantile:
        The calibration quantile used for interval width.
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
    """A calibrated conformal predictor ready to produce intervals.

    Created by `conformal_fit()`. Call `predict()` on new data to get distribution-free prediction
    intervals.
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
        """Produce conformal prediction intervals on new data.

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
        n_cal = len(self.calibration_scores)
        n_folds = len(self._models)
        n_new = len(next(iter(new_data.values())))

        fold_preds = np.zeros((n_folds, n_new))
        for k, m in enumerate(self._models):
            fold_preds[k] = m.predict(new_data).values

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
        n_loo = len(self._models)
        n_new = len(next(iter(new_data.values())))

        loo_preds = np.zeros((n_loo, n_new))
        for i, m in enumerate(self._models):
            loo_preds[i] = m.predict(new_data).values

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
    """Fit a GAM with conformal calibration.

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

    Returns
    -------
    ConformalPredictor
        A calibrated predictor that can produce intervals on new data.
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

    cal_pred = model.predict(cal_data).values
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

        pred = model.predict(test_data).values
        loo_residuals[test] = np.abs(y[test] - pred)

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
        pred = model.predict(single).values[0]
        loo_residuals[i] = abs(y[i] - pred)

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
    """Compute empirical coverage of conformal intervals on held-out data.

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
