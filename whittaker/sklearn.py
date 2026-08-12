"""Scikit-learn compatible estimator wrappers for Whittaker GAMs.

This module adapts `whittaker.gam.GAM` to the scikit-learn estimator protocol so a GAM can be
dropped into scikit-learn workflows without hand-writing a formula or a data dictionary. Two
wrappers are provided:

- `GAMRegressor`: a `sklearn.base.RegressorMixin` for continuous responses (`Gaussian` family).
- `GAMClassifier`: a `sklearn.base.ClassifierMixin` for binary responses (`Binomial(link="logit")`
  family).

Both wrappers take plain numpy arrays — `X` of shape `(n_samples, n_features)` and `y` of shape
`(n_samples,)` — rather than the named, column-oriented data dictionaries that `GAM` expects
directly. Internally, each wrapper auto-names the feature columns (`x0`, `x1`, ...; see
`_make_feature_names`), builds a `y ~ s(x0) + s(x1) + ...` formula unless an explicit `formula` is
supplied (see `_build_formula`), converts the array into the `{name: column}` dictionary a `GAM`
needs (see `_array_to_data`), and delegates fitting and prediction to an underlying `GAM` instance
stored on the fitted estimator (`self.gam_`). This lets a `GAM` participate in `Pipeline`,
`GridSearchCV`, `cross_val_score`, and similar scikit-learn tooling that expects `fit`/`predict`
estimators, while the actual smoothing, penalization, and inference machinery remains the same as
using `GAM` directly.
"""

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
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False

    class BaseEstimator:  # type: ignore[no-redef]
        pass

    class RegressorMixin:  # type: ignore[no-redef]
        pass

    class ClassifierMixin:  # type: ignore[no-redef]
        pass


def _check_sklearn() -> None:
    if not _HAS_SKLEARN:  # pragma: no cover
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
    r"""Scikit-learn compatible GAM regressor.

    `GAMRegressor` wraps `whittaker.gam.GAM` behind the scikit-learn `BaseEstimator` /
    `RegressorMixin` interface, exposing the familiar `fit(X, y)`, `predict(X)`,
    `get_params()`/`set_params()` methods instead of `GAM`'s formula-and-data-dictionary API. This
    makes it a drop-in estimator anywhere scikit-learn expects one: inside a `Pipeline` (e.g.
    chained after a `StandardScaler` or a `ColumnTransformer`), as the estimator tuned by
    `GridSearchCV` or `RandomizedSearchCV` (searching over `formula`, `method`, or `select`),
    scored with `cross_val_score` or `cross_validate`, or combined with other regressors inside a
    `VotingRegressor` or a stacking ensemble.

    Because raw numpy feature columns carry no names, `GAMRegressor` assigns synthetic names `x0`,
    `x1`, ..., `x{n_features - 1}` to the columns of `X` in order (see `_make_feature_names`).
    Unless an explicit `formula` is supplied, a default additive formula with one smooth `s(xi)`
    per feature is built automatically (see `_build_formula`), giving

    $$
    \eta = \beta_0 + \sum_{i=0}^{p-1} f_i(x_i),
    $$

    connected to the mean response through the link implied by `family`. Supplying `formula`
    overrides this default; it accepts either a bare right-hand side such as `"s(x0) + s(x1)"` —
    in which case the response name `"y"` is prepended automatically to form `"y ~ s(x0) + s(x1)"`
    — or a complete formula already containing `"~"`, which is used as-is. This makes it possible
    to mix smooth and linear terms, use interactions (`x0:x1`), or omit features, exactly as with
    `GAM` directly.

    Parameters
    ----------
    formula : str, optional
        GAM formula for the right-hand side (e.g. `"s(x0) + s(x1)"`), or a complete formula
        containing `"~"` (e.g. `"y ~ s(x0) + x1"`). If it contains `"~"` it is used verbatim;
        otherwise the response `"y"` is prepended. If `None` (the default), a formula with one
        `s(xi)` smooth per input feature is generated automatically.
    family : Family, optional
        Response distribution family passed through to the underlying `GAM`. Defaults to
        `Gaussian()` (identity link), i.e. ordinary least-squares-style additive regression.
    method : str
        Smoothing parameter selection criterion forwarded to `GAM.fit`: `"GCV"` (default),
        `"REML"`, or `"ML"`. See `whittaker.gam.GAM.fit` for the meaning of each option.
    select : bool
        If `True`, enable double-penalty smooth selection (an extra penalty that can shrink an
        entire smooth to zero), forwarded to `GAM.fit`. Defaults to `False`.

    Notes
    -----
    The hyperparameters exposed to `GridSearchCV`/`RandomizedSearchCV` via `get_params()` are
    exactly the constructor arguments — `formula`, `family`, `method`, and `select` — because
    scikit-learn's `get_params` introspects the `__init__` signature. The smoothing parameters
    `\lambda_j` themselves are never tunable hyperparameters of `GAMRegressor`: they are always
    chosen internally, for the given `method`, during `fit()`. To tune smoothing behavior via
    cross-validation, search over `method` and `select` (which change how `\lambda_j` are
    selected) rather than trying to pass `\lambda_j` values directly.

    Examples
    --------
    ```{python}
    import numpy as np
    from whittaker.sklearn import GAMRegressor

    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(200, 2))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 + rng.normal(scale=0.2, size=200)

    reg = GAMRegressor(formula="s(x0) + s(x1)")
    reg.fit(X, y)
    reg.predict(X[:5])
    ```

    `GAMRegressor` works inside scikit-learn model-selection tooling, such as
    `sklearn.model_selection.cross_val_score` (requires `pip install scikit-learn`):

    ```{python}
    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(GAMRegressor(), X, y, cv=5)
    scores
    ```
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

        Validates `X` and `y` with scikit-learn's `check_X_y`, assigns synthetic feature names,
        builds the model formula (from `formula` if given, otherwise one `s(xi)` smooth per
        feature), and fits an internal `whittaker.gam.GAM` to the resulting data dictionary using
        `method` and `select`.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix, shape `(n_samples, n_features)`. Coerced to `float64` and checked by
            `sklearn.utils.validation.check_X_y`, which rejects non-finite values, non-2-D input,
            and samples with zero features.
        y : numpy.ndarray
            Target values, shape `(n_samples,)`. Coerced to `float64` alongside `X` by
            `check_X_y`; must have the same number of samples as `X`.
        **fit_params : Any
            Accepted for compatibility with the scikit-learn `fit` signature but currently unused.

        Returns
        -------
        GAMRegressor
            Returns `self`, with fitted attributes `n_features_in_` (number of input features),
            `feature_names_` (the synthetic `x0`, `x1`, ... names used internally), and `gam_`
            (the underlying fitted `whittaker.gam.GAM` instance), so that scikit-learn's
            `fit(...).predict(...)` chaining works as usual.
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

        Requires `fit()` to have been called first (checked via `sklearn.utils.validation.
        check_is_fitted`). Converts `X` into the internal data dictionary using the feature names
        recorded during `fit()`, and delegates to the fitted `GAM`'s `predict()`.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix, shape `(n_samples, n_features_in_)`, where `n_features_in_` matches
            the number of features seen during `fit()`. Coerced to `float64` and validated by
            `sklearn.utils.validation.check_array`.

        Returns
        -------
        NDArray
            Predicted mean response values, shape `(n_samples,)`, in the original (non-linear
            predictor) scale.
        """
        check_is_fitted(self, "gam_")
        X = check_array(X, dtype="float64")
        data = _array_to_data(X, response="y", feature_names=self.feature_names_)
        return self.gam_.predict(data).values


class GAMClassifier(BaseEstimator, ClassifierMixin):  # type: ignore[misc]
    r"""Scikit-learn compatible GAM classifier (binary).

    `GAMClassifier` wraps `whittaker.gam.GAM` behind the scikit-learn `BaseEstimator` /
    `ClassifierMixin` interface for binary classification. It always fits a
    `whittaker.families.binomial.Binomial(link="logit")` family internally — i.e. a logistic GAM
    — so the linear predictor is related to the class-1 probability `\mu` by the logit link,
    `\eta = \log(\mu / (1 - \mu))`. Only two-class problems are supported: `fit()` raises a
    `ValueError` if `y` does not contain exactly two distinct labels. The formula-building
    behavior (synthetic feature names, default `s(xi)`-per-feature formula, or an explicit
    `formula`) is identical to `GAMRegressor`; see that class for details.

    Fitted attributes follow the scikit-learn classifier convention: `self.classes_` holds the
    two observed labels sorted ascending (as returned by `numpy.unique`), and `predict_proba`
    returns probability columns ordered to match `self.classes_` (column 0 is `P(y = classes_[0])`,
    column 1 is `P(y = classes_[1])`). This makes `GAMClassifier` compatible with `Pipeline`,
    `GridSearchCV`/`RandomizedSearchCV` (scored via `"accuracy"`, `"roc_auc"`, or a custom scorer),
    and any tooling that consumes `predict_proba`, such as `sklearn.calibration.CalibratedClassifierCV`
    or manual decision-threshold tuning on the predicted probabilities.

    Parameters
    ----------
    formula : str, optional
        GAM formula for the right-hand side (e.g. `"s(x0) + s(x1)"`), or a complete formula
        containing `"~"`. If it contains `"~"` it is used verbatim; otherwise the response `"y"`
        is prepended. If `None` (the default), a formula with one `s(xi)` smooth per input
        feature is generated automatically.
    method : str
        Smoothing parameter selection criterion forwarded to `GAM.fit`: `"GCV"` (default),
        `"REML"`, or `"ML"`. See `whittaker.gam.GAM.fit` for the meaning of each option.
    select : bool
        If `True`, enable double-penalty smooth selection, forwarded to `GAM.fit`. Defaults to
        `False`.

    Notes
    -----
    As with `GAMRegressor`, the hyperparameters exposed to `GridSearchCV`/`RandomizedSearchCV` via
    `get_params()` are the constructor arguments — `formula`, `method`, and `select` — since
    scikit-learn's `get_params` introspects `__init__`. There is no `family` parameter here
    because the family is fixed to `Binomial(link="logit")`. The smoothing parameters
    `\lambda_j` are always selected internally by `method` during `fit()` rather than being
    directly tunable.

    Examples
    --------
    ```{python}
    import numpy as np
    from whittaker.sklearn import GAMClassifier

    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(300, 2))
    logit = 1.5 * np.sin(X[:, 0]) - X[:, 1]
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p)

    clf = GAMClassifier(formula="s(x0) + s(x1)")
    clf.fit(X, y)
    clf.predict_proba(X[:5])
    ```

    ```{python}
    clf.predict(X[:5])
    ```
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

        Validates `X` and `y` with `check_X_y`, records the two observed class labels in
        `self.classes_`, assigns synthetic feature names, builds the model formula, and fits an
        internal `Binomial(link="logit")` `whittaker.gam.GAM` to the resulting data dictionary
        using `method` and `select`.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix, shape `(n_samples, n_features)`. Coerced to `float64` and validated
            by `sklearn.utils.validation.check_X_y`.
        y : numpy.ndarray
            Class labels, shape `(n_samples,)`. Must contain exactly two distinct values (e.g.
            `0`/`1`, or any two comparable labels); coerced to `float64` alongside `X`.

        Returns
        -------
        GAMClassifier
            Returns `self`, with fitted attributes `classes_` (the two sorted class labels seen
            in `y`), `n_features_in_`, `feature_names_`, and `gam_` (the underlying fitted
            `Binomial`-family `whittaker.gam.GAM`).

        Raises
        ------
        ValueError
            If `y` contains fewer or more than two distinct values, since `GAMClassifier`
            supports binary classification only.
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

        Requires `fit()` to have been called first. Converts `X` into the internal data
        dictionary and calls the fitted `Binomial`-family `GAM`'s `predict()` to obtain the
        probability of the positive class (`self.classes_[1]`), then derives the probability of
        the negative class as its complement.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix, shape `(n_samples, n_features_in_)`. Coerced to `float64` and
            validated by `sklearn.utils.validation.check_array`.

        Returns
        -------
        NDArray
            Predicted probabilities, shape `(n_samples, 2)`. Column `0` is
            `P(y == self.classes_[0])` and column `1` is `P(y == self.classes_[1])`, matching the
            scikit-learn convention that `predict_proba` columns are ordered by `self.classes_`.
        """
        check_is_fitted(self, "gam_")
        X = check_array(X, dtype="float64")
        data = _array_to_data(X, response="y", feature_names=self.feature_names_)
        p1 = self.gam_.predict(data).values
        return np.column_stack([1 - p1, p1])

    def predict(self, X: NDArray) -> NDArray:
        """Predict class labels for X.

        Calls `predict_proba()` and, for each sample, returns the label in `self.classes_`
        corresponding to the higher predicted probability (i.e. thresholding at `0.5` on the
        positive-class probability).

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix, shape `(n_samples, n_features_in_)`.

        Returns
        -------
        NDArray
            Predicted class labels, shape `(n_samples,)`, drawn from `self.classes_`.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
