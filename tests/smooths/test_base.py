"""Tests for whittaker.smooths.base (SmoothBasis abstract base class)."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.smooths.base import SmoothBasis


class _MinimalBasis(SmoothBasis):
    """Concrete subclass implementing only the required abstract methods.

    Used to exercise the default (non-overridden) behavior defined on
    `SmoothBasis` itself, such as `identifiability_constraints()` and the
    `_as_2d()` helper.
    """

    def fit(self, x):
        self._fitted = True
        return self

    def basis_matrix(self, x):
        return self._as_2d(x)

    def penalty_matrix(self):
        return np.zeros((self.n_basis, self.n_basis))

    def null_space_dimension(self):
        return 0

    @property
    def n_basis(self):
        return 1


class TestIdentifiabilityConstraintsDefault:
    def test_default_returns_none(self) -> None:
        basis = _MinimalBasis().fit(np.linspace(0.0, 1.0, 20))
        assert basis.identifiability_constraints() is None


class TestAs2d:
    def test_1d_input_reshaped(self) -> None:
        x = np.linspace(0.0, 1.0, 10)
        out = SmoothBasis._as_2d(x)
        assert out.shape == (10, 1)

    def test_2d_input_passthrough(self) -> None:
        x = np.ones((10, 3))
        out = SmoothBasis._as_2d(x)
        assert out.shape == (10, 3)

    def test_3d_input_raises(self) -> None:
        x = np.ones((10, 3, 2))
        with pytest.raises(ValueError, match="1-D or 2-D"):
            SmoothBasis._as_2d(x)


class TestIsFitted:
    def test_unfitted_is_false(self) -> None:
        basis = _MinimalBasis()
        assert basis.is_fitted is False

    def test_fitted_is_true(self) -> None:
        basis = _MinimalBasis().fit(np.linspace(0.0, 1.0, 20))
        assert basis.is_fitted is True

    def test_check_fitted_raises_when_not_fitted(self) -> None:
        basis = _MinimalBasis()
        with pytest.raises(RuntimeError, match="must be fitted"):
            basis._check_fitted()
