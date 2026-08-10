"""Tests for whittaker.smooths.tensor (tensor product basis)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from whittaker.smooths.cubic import CRS
from whittaker.smooths.pspline import PSpline
from whittaker.smooths.tensor import TensorProductBasis, _row_tensor_product

RNG = np.random.default_rng(23)


class TestRowTensorProduct:
    def test_shape(self) -> None:
        B1 = RNG.standard_normal((10, 3))
        B2 = RNG.standard_normal((10, 4))
        result = _row_tensor_product(B1, B2)
        assert result.shape == (10, 12)

    def test_values(self) -> None:
        B1 = np.array([[1, 2], [3, 4]])
        B2 = np.array([[5, 6, 7], [8, 9, 10]])
        result = _row_tensor_product(B1, B2)
        expected = np.array(
            [
                [1 * 5, 1 * 6, 1 * 7, 2 * 5, 2 * 6, 2 * 7],
                [3 * 8, 3 * 9, 3 * 10, 4 * 8, 4 * 9, 4 * 10],
            ]
        )
        assert_allclose(result, expected)

    def test_identity_product(self) -> None:
        B = RNG.standard_normal((5, 3))
        ones = np.ones((5, 1))
        result = _row_tensor_product(ones, B)
        assert_allclose(result, B)


class TestTensorProductBasis:
    def test_requires_at_least_two_marginals(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            TensorProductBasis([CRS(k=5)])

    def test_fit_and_basis_matrix_shape(self) -> None:
        m1, m2 = CRS(k=5), CRS(k=6)
        tp = TensorProductBasis([m1, m2])
        x = np.column_stack(
            [
                np.linspace(0, 1, 50),
                np.linspace(0, 1, 50),
            ]
        )
        tp.fit(x)
        B = tp.basis_matrix(x)
        assert B.shape == (50, 30)  # 5 * 6

    def test_n_basis(self) -> None:
        tp = TensorProductBasis([CRS(k=4), CRS(k=5)])
        x = np.column_stack([np.linspace(0, 1, 30)] * 2)
        tp.fit(x)
        assert tp.n_basis == 20

    def test_penalty_matrices_count(self) -> None:
        tp = TensorProductBasis([CRS(k=5), CRS(k=5)])
        x = np.column_stack([np.linspace(0, 1, 30)] * 2)
        tp.fit(x)
        pens = tp.penalty_matrices()
        assert len(pens) == 2

    def test_penalty_matrix_shapes(self) -> None:
        k1, k2 = 4, 5
        tp = TensorProductBasis([CRS(k=k1), CRS(k=k2)])
        x = np.column_stack([np.linspace(0, 1, 30)] * 2)
        tp.fit(x)
        pens = tp.penalty_matrices()
        for pen in pens:
            assert pen.shape == (20, 20)  # k1*k2

    def test_penalty_matrices_symmetric(self) -> None:
        tp = TensorProductBasis([CRS(k=5), CRS(k=5)])
        x = np.column_stack([np.linspace(0, 1, 30)] * 2)
        tp.fit(x)
        for pen in tp.penalty_matrices():
            assert_allclose(pen, pen.T, atol=1e-14)

    def test_penalty_matrices_psd(self) -> None:
        tp = TensorProductBasis([CRS(k=5), CRS(k=5)])
        x = np.column_stack([np.linspace(0, 1, 30)] * 2)
        tp.fit(x)
        for pen in tp.penalty_matrices():
            eigvals = np.linalg.eigvalsh(pen)
            assert np.all(eigvals >= -1e-10)

    def test_penalty_matrix_is_sum_of_penalty_matrices(self) -> None:
        tp = TensorProductBasis([CRS(k=5), CRS(k=5)])
        x = np.column_stack([np.linspace(0, 1, 30)] * 2)
        tp.fit(x)
        pens = tp.penalty_matrices()
        assert_allclose(tp.penalty_matrix(), sum(pens), atol=1e-14)

    def test_null_space_dimension(self) -> None:
        tp = TensorProductBasis([CRS(k=5), CRS(k=5)])
        x = np.column_stack([np.linspace(0, 1, 30)] * 2)
        tp.fit(x)
        # CRS null space dim = 2 each, product = 4
        assert tp.null_space_dimension() == 4

    def test_identifiability_constraints(self) -> None:
        tp = TensorProductBasis([CRS(k=5), CRS(k=5)])
        x = np.column_stack([np.linspace(0, 1, 30)] * 2)
        tp.fit(x)
        C = tp.identifiability_constraints()
        assert C is not None
        assert C.shape == (1, 25)

    def test_unfitted_raises(self) -> None:
        tp = TensorProductBasis([CRS(k=5), CRS(k=5)])
        with pytest.raises(RuntimeError, match="fitted"):
            tp.basis_matrix(np.zeros((10, 2)))

    def test_wrong_columns_raises(self) -> None:
        tp = TensorProductBasis([CRS(k=5), CRS(k=5)])
        with pytest.raises(ValueError, match="Expected 2 columns"):
            tp.fit(np.zeros((10, 3)))

    def test_three_way_tensor(self) -> None:
        tp = TensorProductBasis([CRS(k=3), CRS(k=4), CRS(k=5)])
        x = np.column_stack([np.linspace(0, 1, 30)] * 3)
        tp.fit(x)
        assert tp.n_basis == 60  # 3 * 4 * 5
        B = tp.basis_matrix(x)
        assert B.shape == (30, 60)
        pens = tp.penalty_matrices()
        assert len(pens) == 3
        for pen in pens:
            assert pen.shape == (60, 60)

    def test_mixed_basis_types(self) -> None:
        tp = TensorProductBasis([CRS(k=5), PSpline(k=6)])
        x = np.column_stack([np.linspace(0, 1, 50)] * 2)
        tp.fit(x)
        B = tp.basis_matrix(x)
        assert B.shape == (50, 30)

    def test_prediction_at_new_points(self) -> None:
        tp = TensorProductBasis([CRS(k=5), CRS(k=5)])
        x_train = np.column_stack(
            [
                np.linspace(0, 1, 50),
                np.linspace(0, 1, 50),
            ]
        )
        tp.fit(x_train)
        x_new = np.column_stack(
            [
                np.array([0.25, 0.5, 0.75]),
                np.array([0.25, 0.5, 0.75]),
            ]
        )
        B_new = tp.basis_matrix(x_new)
        assert B_new.shape == (3, 25)
        assert np.all(np.isfinite(B_new))
