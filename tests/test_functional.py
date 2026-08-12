"""Tests for functional regression GAMs."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.functional import (
    CoefficientFunction,
    FunctionalGAM,
    FunctionalTerm,
    _build_functional_design,
    _fourier_basis,
    _fourier_penalty,
    _integration_weights,
)


@pytest.fixture
def scalar_on_function_data():
    """Scalar-on-function data: y = integral of X(t)*beta(t) dt + noise."""
    rng = np.random.default_rng(23)
    n = 200
    T = 50
    t_grid = np.linspace(0, 1, T)
    beta_true = np.sin(2 * np.pi * t_grid)

    X_func = np.zeros((n, T))
    for i in range(n):
        X_func[i, :] = rng.normal(0, 1, T).cumsum() / np.sqrt(T)

    dt = 1.0 / (T - 1)
    w = np.full(T, dt)
    w[0] = dt / 2
    w[-1] = dt / 2
    y = X_func @ (beta_true * w) + rng.normal(0, 0.3, n)

    return {"X_func": X_func, "y": y, "beta_true": beta_true, "t_grid": t_grid}


@pytest.fixture
def two_functional_data():
    """Data with two functional covariates."""
    rng = np.random.default_rng(23)
    n = 200
    T1, T2 = 40, 30
    t1 = np.linspace(0, 1, T1)
    t2 = np.linspace(0, 2, T2)

    beta1 = np.sin(2 * np.pi * t1)
    beta2 = t2**2 - t2

    X1 = rng.normal(0, 1, (n, T1)).cumsum(axis=1) / np.sqrt(T1)
    X2 = rng.normal(0, 1, (n, T2)).cumsum(axis=1) / np.sqrt(T2)

    dt1 = 1.0 / (T1 - 1)
    w1 = np.full(T1, dt1)
    w1[0] = w1[-1] = dt1 / 2
    dt2 = 2.0 / (T2 - 1)
    w2 = np.full(T2, dt2)
    w2[0] = w2[-1] = dt2 / 2

    y = X1 @ (beta1 * w1) + X2 @ (beta2 * w2) + rng.normal(0, 0.3, n)

    return {"f1": X1, "f2": X2, "y": y}


@pytest.fixture
def mixed_data():
    """Data with one functional covariate and one scalar covariate."""
    rng = np.random.default_rng(23)
    n = 200
    T = 40
    t_grid = np.linspace(0, 1, T)
    beta_true = np.sin(2 * np.pi * t_grid)

    X_func = rng.normal(0, 1, (n, T)).cumsum(axis=1) / np.sqrt(T)
    x_scalar = np.linspace(0, 2 * np.pi, n)

    dt = 1.0 / (T - 1)
    w = np.full(T, dt)
    w[0] = w[-1] = dt / 2
    y = X_func @ (beta_true * w) + np.sin(x_scalar) + rng.normal(0, 0.3, n)

    return {"spectrum": X_func, "temp": x_scalar, "y": y}


class TestFunctionalTerm:
    def test_default_values(self):
        ft = FunctionalTerm(name="x")
        assert ft.basis == "bspline"
        assert ft.domain == (0.0, 1.0)
        assert ft.n_basis == 15
        assert ft.penalty_order == 2

    def test_custom_values(self):
        ft = FunctionalTerm(name="spec", basis="fourier", domain=(400, 700), n_basis=20)
        assert ft.name == "spec"
        assert ft.basis == "fourier"
        assert ft.domain == (400, 700)
        assert ft.n_basis == 20


class TestHelpers:
    def test_integration_weights_sum(self):
        w = _integration_weights(100, (0.0, 1.0))
        assert abs(w.sum() - 1.0) < 1e-10

    def test_integration_weights_domain(self):
        w = _integration_weights(50, (2.0, 5.0))
        assert abs(w.sum() - 3.0) < 1e-10

    def test_fourier_basis_shape(self):
        t = np.linspace(0, 1, 100)
        B = _fourier_basis(t, 11, (0, 1))
        assert B.shape == (100, 11)

    def test_fourier_basis_orthogonality(self):
        t = np.linspace(0, 1, 1000)
        B = _fourier_basis(t, 5, (0, 1))
        dt = 1.0 / 999
        G = B.T @ B * dt
        off_diag = G - np.diag(np.diag(G))
        assert np.max(np.abs(off_diag)) < 0.05

    def test_fourier_penalty_shape(self):
        S = _fourier_penalty(11, (0, 1))
        assert S.shape == (11, 11)
        assert S[0, 0] == 0.0

    def test_fourier_penalty_increasing(self):
        S = _fourier_penalty(7, (0, 1))
        diag = np.diag(S)
        nonzero = diag[diag > 0]
        assert len(nonzero) > 0
        for i in range(len(nonzero) - 1):
            assert nonzero[i + 1] >= nonzero[i]

    def test_functional_design_shape(self):
        rng = np.random.default_rng(23)
        X = rng.normal(0, 1, (50, 100))
        B = _fourier_basis(np.linspace(0, 1, 100), 10, (0, 1))
        w = _integration_weights(100, (0, 1))
        J = _build_functional_design(X, B, w)
        assert J.shape == (50, 10)


class TestFunctionalGAMInit:
    def test_basic_init(self):
        model = FunctionalGAM(
            response="y",
            functional_terms=[FunctionalTerm(name="x")],
        )
        assert model.response == "y"
        assert not model.is_fitted
        assert len(model.functional_terms) == 1

    def test_dict_terms(self):
        model = FunctionalGAM(
            response="y",
            functional_terms=[{"name": "x", "basis": "fourier", "domain": (0, 1)}],
        )
        assert model.functional_terms[0].basis == "fourier"

    def test_no_terms_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            FunctionalGAM(response="y", functional_terms=[])

    def test_invalid_basis_raises(self):
        with pytest.raises(ValueError, match="Unsupported basis"):
            FunctionalGAM(
                response="y",
                functional_terms=[FunctionalTerm(name="x", basis="wavelet")],
            )

    def test_too_few_basis_raises(self):
        with pytest.raises(ValueError, match="n_basis must be >= 3"):
            FunctionalGAM(
                response="y",
                functional_terms=[FunctionalTerm(name="x", n_basis=2)],
            )

    def test_invalid_domain_raises(self):
        with pytest.raises(ValueError, match="t_min < t_max"):
            FunctionalGAM(
                response="y",
                functional_terms=[FunctionalTerm(name="x", domain=(1.0, 0.0))],
            )

    def test_repr_unfitted(self):
        model = FunctionalGAM(
            response="y",
            functional_terms=[FunctionalTerm(name="spec")],
        )
        r = repr(model)
        assert "FunctionalGAM" in r
        assert "unfitted" in r
        assert "spec" in r


class TestFunctionalGAMFit:
    def test_fit_bspline(self, scalar_on_function_data):
        data = scalar_on_function_data
        model = FunctionalGAM(
            response="y",
            functional_terms=[FunctionalTerm(name="X_func", domain=(0, 1))],
        )
        model.fit({"X_func": data["X_func"], "y": data["y"]})
        assert model.is_fitted

    def test_fit_fourier(self, scalar_on_function_data):
        data = scalar_on_function_data
        model = FunctionalGAM(
            response="y",
            functional_terms=[FunctionalTerm(name="X_func", basis="fourier", domain=(0, 1))],
        )
        model.fit({"X_func": data["X_func"], "y": data["y"]})
        assert model.is_fitted

    def test_fit_returns_self(self, scalar_on_function_data):
        data = scalar_on_function_data
        model = FunctionalGAM(
            response="y",
            functional_terms=[FunctionalTerm(name="X_func", domain=(0, 1))],
        )
        result = model.fit({"X_func": data["X_func"], "y": data["y"]})
        assert result is model

    def test_fit_1d_func_raises(self):
        model = FunctionalGAM(
            response="y",
            functional_terms=[FunctionalTerm(name="x")],
        )
        with pytest.raises(ValueError, match="2-D"):
            model.fit({"x": np.ones(10), "y": np.ones(10)})

    def test_too_few_grid_points_raises(self):
        rng = np.random.default_rng(23)
        n = 50
        X_func = rng.normal(size=(n, 2))
        y = rng.normal(size=n)
        model = FunctionalGAM(
            response="y",
            functional_terms=[FunctionalTerm(name="X_func", n_basis=3)],
        )
        with pytest.raises(ValueError, match="at least 3 grid points"):
            model.fit({"X_func": X_func, "y": y})

    def test_fit_non_dict_data_raises(self):
        model = FunctionalGAM(
            response="y",
            functional_terms=[FunctionalTerm(name="x")],
        )
        with pytest.raises(TypeError, match="Data must be a dict"):
            model.fit([1, 2, 3])

    def test_unfitted_predict_raises(self):
        model = FunctionalGAM(
            response="y",
            functional_terms=[FunctionalTerm(name="x")],
        )
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict({"x": np.ones((5, 10))})

    def test_two_functional_terms(self, two_functional_data):
        data = two_functional_data
        model = FunctionalGAM(
            response="y",
            functional_terms=[
                FunctionalTerm(name="f1", domain=(0, 1)),
                FunctionalTerm(name="f2", domain=(0, 2)),
            ],
        )
        model.fit(data)
        assert model.is_fitted


class TestFunctionalGAMPredict:
    def test_predict_shape(self, scalar_on_function_data):
        data = scalar_on_function_data
        model = FunctionalGAM(
            response="y",
            functional_terms=[FunctionalTerm(name="X_func", domain=(0, 1))],
        )
        model.fit({"X_func": data["X_func"], "y": data["y"]})
        pred = model.predict({"X_func": data["X_func"][:20], "y": data["y"][:20]})
        assert pred.shape == (20,)

    def test_predict_finite(self, scalar_on_function_data):
        data = scalar_on_function_data
        model = FunctionalGAM(
            response="y",
            functional_terms=[FunctionalTerm(name="X_func", domain=(0, 1))],
        )
        model.fit({"X_func": data["X_func"], "y": data["y"]})
        pred = model.predict({"X_func": data["X_func"], "y": data["y"]})
        assert np.all(np.isfinite(pred))

    def test_predict_with_se(self, scalar_on_function_data):
        data = scalar_on_function_data
        model = FunctionalGAM(
            response="y",
            functional_terms=[FunctionalTerm(name="X_func", domain=(0, 1))],
        )
        model.fit({"X_func": data["X_func"], "y": data["y"]})
        mu, se = model.predict({"X_func": data["X_func"][:10], "y": data["y"][:10]}, se=True)
        assert mu.shape == (10,)
        assert se.shape == (10,)
        assert np.all(se >= 0)

    def test_predict_accuracy(self, scalar_on_function_data):
        data = scalar_on_function_data
        model = FunctionalGAM(
            response="y",
            functional_terms=[FunctionalTerm(name="X_func", domain=(0, 1))],
        )
        model.fit({"X_func": data["X_func"], "y": data["y"]})
        pred = model.predict({"X_func": data["X_func"], "y": data["y"]})
        corr = np.corrcoef(pred, data["y"])[0, 1]
        assert corr > 0.5


class TestCoefficientFunction:
    def test_coefficient_function_shape(self, scalar_on_function_data):
        data = scalar_on_function_data
        model = FunctionalGAM(
            response="y",
            functional_terms=[FunctionalTerm(name="X_func", domain=(0, 1))],
        )
        model.fit({"X_func": data["X_func"], "y": data["y"]})
        cf = model.coefficient_function("X_func", n_grid=100)
        assert isinstance(cf, CoefficientFunction)
        assert cf.grid.shape == (100,)
        assert cf.values.shape == (100,)
        assert cf.se.shape == (100,)
        assert cf.lower.shape == (100,)
        assert cf.upper.shape == (100,)

    def test_coefficient_function_interval_contains_values(self, scalar_on_function_data):
        data = scalar_on_function_data
        model = FunctionalGAM(
            response="y",
            functional_terms=[FunctionalTerm(name="X_func", domain=(0, 1))],
        )
        model.fit({"X_func": data["X_func"], "y": data["y"]})
        cf = model.coefficient_function("X_func")
        assert np.all(cf.lower <= cf.values)
        assert np.all(cf.values <= cf.upper)

    def test_coefficient_function_captures_shape(self, scalar_on_function_data):
        data = scalar_on_function_data
        model = FunctionalGAM(
            response="y",
            functional_terms=[FunctionalTerm(name="X_func", domain=(0, 1), n_basis=20)],
        )
        model.fit({"X_func": data["X_func"], "y": data["y"]})
        cf = model.coefficient_function("X_func", n_grid=200)
        corr = np.corrcoef(cf.values, np.sin(2 * np.pi * cf.grid))[0, 1]
        assert corr > 0.5

    def test_invalid_term_raises(self, scalar_on_function_data):
        data = scalar_on_function_data
        model = FunctionalGAM(
            response="y",
            functional_terms=[FunctionalTerm(name="X_func", domain=(0, 1))],
        )
        model.fit({"X_func": data["X_func"], "y": data["y"]})
        with pytest.raises(ValueError, match="not found"):
            model.coefficient_function("nonexistent")

    def test_coefficient_function_with_intercept_scalar_terms(self, mixed_data):
        """Scalar terms with the default intercept exercise the has-intercept branch of the
        scalar penalty reconstruction used by `coefficient_function()`
        (`_get_full_penalties`).
        """
        model = FunctionalGAM(
            response="y",
            functional_terms=[FunctionalTerm(name="spectrum", domain=(0, 1))],
            scalar_terms="s(temp)",
        )
        model.fit(mixed_data)
        cf = model.coefficient_function("spectrum")
        assert np.all(np.isfinite(cf.values))
        assert np.all(np.isfinite(cf.se))

    def test_coefficient_function_with_no_intercept_scalar_terms(self, mixed_data):
        """Scalar terms without an intercept (`scalar_terms="0 + ..."`) exercise the
        no-intercept branch of the scalar penalty reconstruction used by
        `coefficient_function()` (`_get_full_penalties`).
        """
        model = FunctionalGAM(
            response="y",
            functional_terms=[FunctionalTerm(name="spectrum", domain=(0, 1))],
            scalar_terms="0 + s(temp)",
        )
        model.fit(mixed_data)
        cf = model.coefficient_function("spectrum")
        assert np.all(np.isfinite(cf.values))
        assert np.all(np.isfinite(cf.se))

    def test_coefficient_function_singular_matrix_falls_back_to_pinv(
        self, scalar_on_function_data, monkeypatch
    ):
        """If the reconstructed information matrix is singular, `coefficient_function()`
        falls back from `np.linalg.inv` to `np.linalg.pinv`.
        """
        data = scalar_on_function_data
        model = FunctionalGAM(
            response="y",
            functional_terms=[FunctionalTerm(name="X_func", domain=(0, 1))],
        )
        model.fit({"X_func": data["X_func"], "y": data["y"]})

        def _raise_linalg_error(a):
            raise np.linalg.LinAlgError("forced singular matrix for test")

        monkeypatch.setattr(np.linalg, "inv", _raise_linalg_error)
        cf = model.coefficient_function("X_func")
        assert np.all(np.isfinite(cf.values))
        assert np.all(np.isfinite(cf.se))

    def test_fourier_coefficient_function(self, scalar_on_function_data):
        data = scalar_on_function_data
        model = FunctionalGAM(
            response="y",
            functional_terms=[
                FunctionalTerm(name="X_func", basis="fourier", domain=(0, 1), n_basis=15)
            ],
        )
        model.fit({"X_func": data["X_func"], "y": data["y"]})
        cf = model.coefficient_function("X_func")
        assert cf.se is not None
        assert np.all(np.isfinite(cf.values))


class TestMixedModel:
    def test_fit_with_scalar_terms(self, mixed_data):
        model = FunctionalGAM(
            response="y",
            functional_terms=[FunctionalTerm(name="spectrum", domain=(0, 1))],
            scalar_terms="s(temp)",
        )
        model.fit(mixed_data)
        assert model.is_fitted

    def test_predict_with_scalar_terms(self, mixed_data):
        model = FunctionalGAM(
            response="y",
            functional_terms=[FunctionalTerm(name="spectrum", domain=(0, 1))],
            scalar_terms="s(temp)",
        )
        model.fit(mixed_data)
        pred = model.predict(mixed_data)
        assert pred.shape == (200,)
        assert np.all(np.isfinite(pred))

    def test_mixed_better_than_functional_only(self, mixed_data):
        model_func = FunctionalGAM(
            response="y",
            functional_terms=[FunctionalTerm(name="spectrum", domain=(0, 1))],
        )
        model_func.fit(mixed_data)

        model_mixed = FunctionalGAM(
            response="y",
            functional_terms=[FunctionalTerm(name="spectrum", domain=(0, 1))],
            scalar_terms="s(temp)",
        )
        model_mixed.fit(mixed_data)

        pred_func = model_func.predict(mixed_data)
        pred_mixed = model_mixed.predict(mixed_data)
        y = mixed_data["y"]

        mse_func = np.mean((pred_func - y) ** 2)
        mse_mixed = np.mean((pred_mixed - y) ** 2)
        assert mse_mixed < mse_func


class TestEDFAndSummary:
    def test_edf(self, scalar_on_function_data):
        data = scalar_on_function_data
        model = FunctionalGAM(
            response="y",
            functional_terms=[FunctionalTerm(name="X_func", domain=(0, 1))],
        )
        model.fit({"X_func": data["X_func"], "y": data["y"]})
        edfs = model.edf()
        assert "X_func" in edfs
        assert edfs["X_func"] > 0

    def test_edf_total(self, scalar_on_function_data):
        data = scalar_on_function_data
        model = FunctionalGAM(
            response="y",
            functional_terms=[FunctionalTerm(name="X_func", domain=(0, 1))],
        )
        model.fit({"X_func": data["X_func"], "y": data["y"]})
        assert model.edf_total > 1.0

    def test_scale_positive(self, scalar_on_function_data):
        data = scalar_on_function_data
        model = FunctionalGAM(
            response="y",
            functional_terms=[FunctionalTerm(name="X_func", domain=(0, 1))],
        )
        model.fit({"X_func": data["X_func"], "y": data["y"]})
        assert model.scale > 0

    def test_deviance_positive(self, scalar_on_function_data):
        data = scalar_on_function_data
        model = FunctionalGAM(
            response="y",
            functional_terms=[FunctionalTerm(name="X_func", domain=(0, 1))],
        )
        model.fit({"X_func": data["X_func"], "y": data["y"]})
        assert model.deviance > 0

    def test_summary(self, scalar_on_function_data):
        data = scalar_on_function_data
        model = FunctionalGAM(
            response="y",
            functional_terms=[FunctionalTerm(name="X_func", domain=(0, 1))],
        )
        model.fit({"X_func": data["X_func"], "y": data["y"]})
        s = model.summary()
        assert "FunctionalGAM" in s
        assert "X_func" in s
        assert "bspline" in s

    def test_summary_with_scalar(self, mixed_data):
        model = FunctionalGAM(
            response="y",
            functional_terms=[FunctionalTerm(name="spectrum", domain=(0, 1))],
            scalar_terms="s(temp)",
        )
        model.fit(mixed_data)
        s = model.summary()
        assert "Scalar terms" in s

    def test_repr_fitted(self, scalar_on_function_data):
        data = scalar_on_function_data
        model = FunctionalGAM(
            response="y",
            functional_terms=[FunctionalTerm(name="X_func", domain=(0, 1))],
        )
        model.fit({"X_func": data["X_func"], "y": data["y"]})
        r = repr(model)
        assert "fitted" in r

    def test_coefficients(self, scalar_on_function_data):
        data = scalar_on_function_data
        model = FunctionalGAM(
            response="y",
            functional_terms=[FunctionalTerm(name="X_func", domain=(0, 1))],
        )
        model.fit({"X_func": data["X_func"], "y": data["y"]})
        coefs = model.coefficients
        assert len(coefs) > 0
        assert np.all(np.isfinite(coefs))
