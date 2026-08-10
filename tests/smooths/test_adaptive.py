"""Tests for adaptive TPRS basis (bs='ad')."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.smooths.adaptive import AdaptiveTPRS


class TestAdaptiveTPRSBasis:
    def test_penalty_count_default(self):
        basis = AdaptiveTPRS(k=10, m=2)
        x = np.linspace(0, 1, 50)
        basis.fit(x)
        pens = basis.penalty_matrices()
        M = basis.null_space_dimension()
        assert len(pens) == 10 - M

    def test_penalty_count_custom(self):
        basis = AdaptiveTPRS(k=10, m=2, n_penalties=3)
        x = np.linspace(0, 1, 50)
        basis.fit(x)
        pens = basis.penalty_matrices()
        assert len(pens) == 3

    def test_penalty_shapes(self):
        basis = AdaptiveTPRS(k=10, m=2)
        x = np.linspace(0, 1, 50)
        basis.fit(x)
        for S in basis.penalty_matrices():
            assert S.shape == (10, 10)

    def test_penalties_symmetric(self):
        basis = AdaptiveTPRS(k=10, m=2)
        x = np.linspace(0, 1, 50)
        basis.fit(x)
        for S in basis.penalty_matrices():
            np.testing.assert_allclose(S, S.T, atol=1e-15)

    def test_penalties_positive_semidefinite(self):
        basis = AdaptiveTPRS(k=10, m=2)
        x = np.linspace(0, 1, 50)
        basis.fit(x)
        for S in basis.penalty_matrices():
            eigvals = np.linalg.eigvalsh(S)
            assert np.all(eigvals >= -1e-12)

    def test_null_space_dimension(self):
        basis = AdaptiveTPRS(k=10, m=2)
        x = np.linspace(0, 1, 50)
        basis.fit(x)
        assert basis.null_space_dimension() == 2

    def test_basis_matrix_same_as_tprs(self):
        from whittaker.smooths.tprs import TPRS

        x = np.linspace(0, 1, 50)
        ad = AdaptiveTPRS(k=10, m=2)
        ad.fit(x)
        tp = TPRS(k=10, m=2)
        tp.fit(x)
        np.testing.assert_allclose(ad.basis_matrix(x), tp.basis_matrix(x), atol=1e-12)

    def test_sum_of_penalties_diagonal(self):
        x = np.linspace(0, 1, 50)
        ad = AdaptiveTPRS(k=10, m=2)
        ad.fit(x)
        S_sum = sum(ad.penalty_matrices())
        M = ad.null_space_dimension()
        assert np.allclose(S_sum[:M, :], 0)
        assert np.allclose(S_sum[:, :M], 0)
        diag = np.diag(S_sum)[M:]
        assert np.all(diag > 0)

    def test_not_fitted_raises(self):
        basis = AdaptiveTPRS(k=10, m=2)
        with pytest.raises(RuntimeError):
            basis.penalty_matrices()

    def test_n_penalties_exceeds_rank(self):
        basis = AdaptiveTPRS(k=10, m=2, n_penalties=100)
        x = np.linspace(0, 1, 50)
        basis.fit(x)
        pens = basis.penalty_matrices()
        M = basis.null_space_dimension()
        assert len(pens) == 10 - M
