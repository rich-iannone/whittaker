"""Scikit-learn compatible estimator wrappers for Whittaker GAMs."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from whittaker.families.base import Family
from whittaker.families.binomial import Binomial
from whittaker.families.gaussian import Gaussian
from whittaker.gam import GAM

try:
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
    from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def _check_sklearn() -> None:
    if not _HAS_SKLEARN:
        raise ImportError(
            "scikit-learn is required for the sklearn wrapper. "
            "Install it with: pip install scikit-learn"
        )


def _array_to_data(
    X: NDArray, y: NDArray | None = None, *, response: str, feature_names: list[str]
) -> dict[str, NDArray]:
    data: dict[str, NDArray] = {}
    for j, name in enumerate(feature_names):
        data[name] = X[:, j]
    if y is not None:
        data[response] = y
    return data


def _make_feature_names(n_features: int) -> list[str]:
    return [f"x{i}" for i in range(n_features)]


def _build_formula(response: str, feature_names: list[str], formula: str | None) -> str:
    if formula is not None:
        if "~" in formula:
            return formula
        return f"{response} ~ {formula}"
    terms = " + ".join(f"s({name})" for name in feature_names)
    return f"{response} ~ {terms}"


class GAMRegressor(BaseEstimator, RegressorMixin):  # type: ignore[misc]
    """Scikit-learn compatible GAM regressor.

    Parameters
    ----------
    formula:
        GAM formula string for the right-hand side (e.g. `"s(x0) + s(x1)"`). If it contains `"~"`,
        the full formula is used as-is; otherwise `"y ~ "` is prepended. If `None`, a default
        formula with one smooth per feature is generated.
    family:
        Response distribution family. Defaults to `Gaussian()`.
    method:
        Smoothing parameter selection: `"GCV"`, `"REML"`, or `"ML"`.
    select:
        If `True`, enable double-penalty smooth selection.

    Examples
    --------
    >>> from whittaker.sklearn import GAMRegressor
    >>> reg = GAMRegressor(formula="s(x0) + s(x1)")
    >>> reg.fit(X, y)
    >>> reg.predict(X_new)

    Works with scikit-learn pipelines and cross-validation:

    >>> from sklearn.model_selection import cross_val_score
    >>> scores = cross_val_score(reg, X, y, cv=5)
    """

    def __init__(
        self,
        formula: str | None = None,
        *,
        family: Family | None = None,
        method: str = "GCV",
        select: bool = False,
    ) -> None:
        _check_sklearn()
        self.formula = formula
        self.family = family
        self.method = method
        self.select = select

    def fit(self, X: NDArray, y: NDArray, **fit_params: Any) -> GAMRegressor:
        """Fit the GAM regressor.

        Parameters
        ----------
        X:
            Feature matrix, shape `(n_samples, n_features)`.
        y:
            Target values, shape `(n_samples,)`.

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y, dtype="float64")
        self.n_features_in_ = X.shape[1]
        self.feature_names_ = _make_feature_names(self.n_features_in_)

        family = self.family if self.family is not None else Gaussian()
        formula_str = _build_formula("y", self.feature_names_, self.formula)

        self.gam_ = GAM(formula_str, family=family)
        data = _array_to_data(X, y, response="y", feature_names=self.feature_names_)
        self.gam_.fit(data, method=self.method, select=self.select)
        return self

    def predict(self, X: NDArray) -> NDArray:
        """Predict target values for X.

        Parameters
        ----------
        X:
            Feature matrix, shape `(n_samples, n_features)`.

        Returns
        -------
        NDArray
            Predicted values, shape `(n_samples,)`.
        """
        check_is_fitted(self, "gam_")
        X = check_array(X, dtype="float64")
        data = _array_to_data(X, response="y", feature_names=self.feature_names_)
        return self.gam_.predict(data).values


class GAMClassifier(BaseEstimator, ClassifierMixin):  # type: ignore[misc]
    """Scikit-learn compatible GAM classifier (binary).

    Uses a `Binomial(link="logit")` family internally. The formula interface is the same as
    `GAMRegressor`.

    Parameters
    ----------
    formula:
        GAM formula string for the right-hand side. If `None`, a default formula with one smooth per
        feature is generated.
    method:
        Smoothing parameter selection: `"GCV"`, `"REML"`, or `"ML"`.
    select:
        If `True`, enable double-penalty smooth selection.

    Examples
    --------
    >>> from whittaker.sklearn import GAMClassifier
    >>> clf = GAMClassifier()
    >>> clf.fit(X, y)
    >>> clf.predict_proba(X_new)
    """

    def __init__(
        self,
        formula: str | None = None,
        *,
        method: str = "GCV",
        select: bool = False,
    ) -> None:
        _check_sklearn()
        self.formula = formula
        self.method = method
        self.select = select

    def fit(self, X: NDArray, y: NDArray, **fit_params: Any) -> GAMClassifier:
        """Fit the GAM classifier.

        Parameters
        ----------
        X:
            Feature matrix, shape `(n_samples, n_features)`.
        y:
            Binary target values (0 or 1), shape `(n_samples,)`.

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y, dtype="float64")
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(
                f"GAMClassifier supports binary classification only. "
                f"Got {len(self.classes_)} classes: {self.classes_}"
            )
        self.n_features_in_ = X.shape[1]
        self.feature_names_ = _make_feature_names(self.n_features_in_)

        family = Binomial()
        formula_str = _build_formula("y", self.feature_names_, self.formula)

        self.gam_ = GAM(formula_str, family=family)
        data = _array_to_data(X, y, response="y", feature_names=self.feature_names_)
        self.gam_.fit(data, method=self.method, select=self.select)
        return self

    def predict_proba(self, X: NDArray) -> NDArray:
        """Predict class probabilities for X.

        Parameters
        ----------
        X:
            Feature matrix, shape `(n_samples, n_features)`.

        Returns
        -------
        NDArray
            Predicted probabilities, shape `(n_samples, 2)`. Columns correspond to `self.classes_`.
        """
        check_is_fitted(self, "gam_")
        X = check_array(X, dtype="float64")
        data = _array_to_data(X, response="y", feature_names=self.feature_names_)
        p1 = self.gam_.predict(data).values
        return np.column_stack([1 - p1, p1])

    def predict(self, X: NDArray) -> NDArray:
        """Predict class labels for X.

        Parameters
        ----------
        X:
            Feature matrix, shape `(n_samples, n_features)`.

        Returns
        -------
        NDArray
            Predicted class labels, shape `(n_samples,)`.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
