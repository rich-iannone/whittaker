"""Tests for the Gamma family."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from whittaker.families.gamma import Gamma


class TestGammaFamily:
    def setup_method(self) -> None:
        self.fam = Gamma()

    def test_link(self) -> None:
        mu = np.array([1.0, 2.0, np.e])
        assert_allclose(self.fam.link(mu), np.log(mu))

    def test_link_inverse(self) -> None:
        eta = np.array([0.0, 1.0, -1.0])
        assert_allclose(self.fam.link_inverse(eta), np.exp(eta))

    def test_link_derivative(self) -> None:
        mu = np.array([1.0, 2.0, 5.0])
        assert_allclose(self.fam.link_derivative(mu), 1.0 / mu)

    def test_link_link_inverse_roundtrip(self) -> None:
        mu = np.array([0.5, 1.0, 3.0, 10.0])
        assert_allclose(self.fam.link_inverse(self.fam.link(mu)), mu)

    def test_variance(self) -> None:
        mu = np.array([1.0, 2.0, 3.0])
        assert_allclose(self.fam.variance(mu), mu**2)

    def test_deviance_at_perfect_fit(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        assert_allclose(self.fam.deviance(y, y), 0.0, atol=1e-14)

    def test_deviance_positive(self) -> None:
        y = np.array([1.0, 2.0, 3.0, 4.0])
        mu = np.array([1.5, 1.5, 3.5, 3.5])
        assert self.fam.deviance(y, mu) > 0

    def test_deviance_formula(self) -> None:
        y = np.array([1.0, 3.0, 5.0])
        mu = np.array([2.0, 2.0, 4.0])
        expected = 2.0 * np.sum(-np.log(y / mu) + (y - mu) / mu)
        assert_allclose(self.fam.deviance(y, mu), expected)

    def test_scale_known_false(self) -> None:
        assert self.fam.scale_known is False

    def test_initialize(self) -> None:
        y = np.array([0.0, 1.0, 5.0])
        mu0 = self.fam.initialize(y)
        assert np.all(mu0 > 0)

    def test_log_likelihood_finite(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.5, 2.5, 2.8])
        ll = self.fam.log_likelihood(y, mu, scale=0.5)
        assert np.isfinite(ll)

    def test_log_likelihood_better_at_truth(self) -> None:
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ll_good = self.fam.log_likelihood(y, y, scale=0.1)
        ll_bad = self.fam.log_likelihood(y, np.full_like(y, 3.0), scale=0.1)
        assert ll_good > ll_bad

    def test_repr(self) -> None:
        assert repr(self.fam) == "Gamma(link='log')"
