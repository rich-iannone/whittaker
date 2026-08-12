"""Tests for scikit-learn compatible estimator wrappers."""

from __future__ import annotations

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")

from sklearn.base import clone  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from whittaker.sklearn import GAMClassifier, GAMRegressor  # noqa: E402


@pytest.fixture()
def regression_data():
    rng = np.random.default_rng(23)
    n = 200
    X = rng.uniform(0, 2 * np.pi, (n, 2))
    y = np.sin(X[:, 0]) + 0.5 * np.cos(X[:, 1]) + rng.normal(0, 0.3, n)
    return X, y


@pytest.fixture()
def classification_data():
    rng = np.random.default_rng(23)
    n = 300
    X = rng.uniform(-3, 3, (n, 2))
    eta = 0.8 * X[:, 0] - 0.5 * X[:, 1]
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float)
    return X, y


class TestGAMRegressor:
    def test_fit_predict(self, regression_data):
        X, y = regression_data
        reg = GAMRegressor()
        reg.fit(X, y)
        pred = reg.predict(X)
        assert pred.shape == (len(y),)
        assert np.corrcoef(y, pred)[0, 1] > 0.7

    def test_custom_formula(self, regression_data):
        X, y = regression_data
        reg = GAMRegressor(formula="s(x0) + s(x1)")
        reg.fit(X, y)
        pred = reg.predict(X)
        assert pred.shape == (len(y),)

    def test_full_formula(self, regression_data):
        X, y = regression_data
        reg = GAMRegressor(formula="y ~ s(x0) + s(x1)")
        reg.fit(X, y)
        pred = reg.predict(X)
        assert pred.shape == (len(y),)

    def test_n_features_in(self, regression_data):
        X, y = regression_data
        reg = GAMRegressor()
        reg.fit(X, y)
        assert reg.n_features_in_ == 2

    def test_score(self, regression_data):
        X, y = regression_data
        reg = GAMRegressor()
        reg.fit(X, y)
        r2 = reg.score(X, y)
        assert r2 > 0.5

    def test_cross_val_score(self, regression_data):
        X, y = regression_data
        reg = GAMRegressor()
        scores = cross_val_score(reg, X, y, cv=3)
        assert len(scores) == 3
        assert all(np.isfinite(scores))

    def test_pipeline(self, regression_data):
        X, y = regression_data
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("gam", GAMRegressor()),
            ]
        )
        pipe.fit(X, y)
        pred = pipe.predict(X)
        assert pred.shape == (len(y),)

    def test_clone(self, regression_data):
        X, y = regression_data
        reg = GAMRegressor(method="REML", select=True)
        reg_clone = clone(reg)
        assert reg_clone.method == "REML"
        assert reg_clone.select is True
        reg_clone.fit(X, y)
        pred = reg_clone.predict(X)
        assert pred.shape == (len(y),)

    def test_get_params(self):
        reg = GAMRegressor(method="REML", select=True)
        params = reg.get_params()
        assert params["method"] == "REML"
        assert params["select"] is True

    def test_set_params(self):
        reg = GAMRegressor()
        reg.set_params(method="REML")
        assert reg.method == "REML"

    def test_reml_method(self, regression_data):
        X, y = regression_data
        reg = GAMRegressor(method="REML")
        reg.fit(X, y)
        pred = reg.predict(X)
        assert np.all(np.isfinite(pred))


class TestGAMClassifier:
    def test_fit_predict(self, classification_data):
        X, y = classification_data
        clf = GAMClassifier()
        clf.fit(X, y)
        pred = clf.predict(X)
        assert pred.shape == (len(y),)
        assert set(np.unique(pred)).issubset({0.0, 1.0})

    def test_predict_proba(self, classification_data):
        X, y = classification_data
        clf = GAMClassifier()
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(y), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_classes(self, classification_data):
        X, y = classification_data
        clf = GAMClassifier()
        clf.fit(X, y)
        np.testing.assert_array_equal(clf.classes_, [0.0, 1.0])

    def test_accuracy_above_chance(self, classification_data):
        X, y = classification_data
        clf = GAMClassifier()
        clf.fit(X, y)
        accuracy = clf.score(X, y)
        assert accuracy > 0.55

    def test_cross_val_score(self, classification_data):
        X, y = classification_data
        clf = GAMClassifier()
        scores = cross_val_score(clf, X, y, cv=3)
        assert len(scores) == 3
        assert all(np.isfinite(scores))

    def test_pipeline(self, classification_data):
        X, y = classification_data
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("gam", GAMClassifier()),
            ]
        )
        pipe.fit(X, y)
        pred = pipe.predict(X)
        assert pred.shape == (len(y),)

    def test_multiclass_raises(self):
        rng = np.random.default_rng(23)
        X = rng.uniform(size=(100, 2))
        y = rng.choice([0, 1, 2], size=100).astype(float)
        clf = GAMClassifier()
        with pytest.raises(ValueError, match="binary classification only"):
            clf.fit(X, y)

    def test_clone(self, classification_data):
        X, y = classification_data
        clf = GAMClassifier(method="REML")
        clf_clone = clone(clf)
        assert clf_clone.method == "REML"

    def test_get_set_params(self):
        clf = GAMClassifier(method="REML", select=True)
        params = clf.get_params()
        assert params["method"] == "REML"
        assert params["select"] is True
        clf.set_params(method="GCV")
        assert clf.method == "GCV"
