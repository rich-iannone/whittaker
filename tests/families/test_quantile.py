"""Tests for QuantileFamily (ELF loss)."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.families.quantile import QuantileFamily, _elf_d1, _elf_d2, _elf_loss


class TestELFLoss:
    def test_symmetric_at_half(self):
        r = np.linspace(-5, 5, 101)
        loss = _elf_loss(r, tau=0.5, sigma=1.0)
        np.testing.assert_allclose(loss, loss[::-1], atol=1e-10)

    def test_positive(self):
        r = np.linspace(-10, 10, 200)
        for tau in [0.1, 0.5, 0.9]:
            loss = _elf_loss(r, tau=tau, sigma=1.0)
            assert np.all(loss >= -1e-12)

    def test_minimum_at_correct_quantile(self):
        r = np.linspace(-10, 10, 10000)
        for tau in [0.1, 0.3, 0.5, 0.7, 0.9]:
            loss = _elf_loss(r, tau=tau, sigma=0.01)
            idx_min = np.argmin(loss)
            assert abs(r[idx_min]) < 0.1


class TestELFDerivatives:
    def test_d1_zero_at_solution(self):
        for tau in [0.1, 0.5, 0.9]:
            r_sol = 0.0
            d1 = _elf_d1(np.array([r_sol]), tau, sigma=1.0)
            expected = tau - 0.5
            np.testing.assert_allclose(d1, expected, atol=1e-10)

    def test_d2_positive(self):
        r = np.linspace(-10, 10, 200)
        d2 = _elf_d2(r, sigma=1.0)
        assert np.all(d2 > 0)

    def test_d2_peak_at_zero(self):
        r = np.linspace(-5, 5, 1001)
        d2 = _elf_d2(r, sigma=1.0)
        assert np.argmax(d2) == 500


class TestQuantileFamilyInit:
    def test_valid_init(self):
        qf = QuantileFamily(tau=0.5, sigma=1.0)
        assert qf.tau == 0.5
        assert qf.sigma == 1.0

    def test_tau_zero_raises(self):
        with pytest.raises(ValueError, match="tau"):
            QuantileFamily(tau=0.0)

    def test_tau_one_raises(self):
        with pytest.raises(ValueError, match="tau"):
            QuantileFamily(tau=1.0)

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            QuantileFamily(tau=0.5, sigma=-1.0)

    def test_sigma_setter(self):
        qf = QuantileFamily(tau=0.5, sigma=1.0)
        qf.sigma = 0.5
        assert qf.sigma == 0.5

    def test_sigma_setter_invalid(self):
        qf = QuantileFamily(tau=0.5, sigma=1.0)
        with pytest.raises(ValueError):
            qf.sigma = -1.0


class TestQuantileFamilyLink:
    def test_identity_link(self):
        qf = QuantileFamily()
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(qf.link(x), x)
        np.testing.assert_array_equal(qf.link_inverse(x), x)
        np.testing.assert_array_equal(qf.link_derivative(x), np.ones(3))

    def test_variance_is_ones(self):
        qf = QuantileFamily()
        mu = np.array([1.0, 5.0, -3.0])
        np.testing.assert_array_equal(qf.variance(mu), np.ones(3))


class TestQuantileFamilyDeviance:
    def test_deviance_positive(self):
        qf = QuantileFamily(tau=0.5, sigma=1.0)
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.1, 1.9, 3.2])
        assert qf.deviance(y, mu) > 0

    def test_deviance_weighted(self):
        qf = QuantileFamily(tau=0.5, sigma=1.0)
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.1, 1.9, 3.2])
        w = np.array([2.0, 2.0, 2.0])
        np.testing.assert_allclose(
            qf.deviance(y, mu, weights=w), 2.0 * qf.deviance(y, mu), rtol=1e-10
        )

    def test_unit_deviance_sums_to_deviance(self):
        qf = QuantileFamily(tau=0.3, sigma=0.5)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        mu = np.array([1.1, 1.9, 3.2, 3.9])
        np.testing.assert_allclose(np.sum(qf.unit_deviance(y, mu)), qf.deviance(y, mu), rtol=1e-10)

    def test_unit_deviance_matches_elf_loss(self):
        qf = QuantileFamily(tau=0.7, sigma=1.0)
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.1, 1.9, 3.2])
        expected = 2.0 * _elf_loss(y - mu, tau=0.7, sigma=1.0)
        np.testing.assert_allclose(qf.unit_deviance(y, mu), expected)


class TestQuantileFamilyIRLS:
    def test_irls_update_returns_tuple(self):
        qf = QuantileFamily(tau=0.5, sigma=1.0)
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.1, 1.9, 3.2])
        eta = mu.copy()
        result = qf.irls_update(y, mu, eta)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_irls_weights_positive(self):
        qf = QuantileFamily(tau=0.5, sigma=1.0)
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.1, 1.9, 3.2])
        _, W = qf.irls_update(y, mu, mu)
        assert np.all(W > 0)

    def test_irls_pseudo_response_finite(self):
        qf = QuantileFamily(tau=0.9, sigma=0.1)
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.0, 2.0, 3.0])
        z, W = qf.irls_update(y, mu, mu)
        assert np.isfinite(z).all()
        assert np.isfinite(W).all()


class TestQuantileFamilyOther:
    def test_scale_known(self):
        assert QuantileFamily().scale_known is True

    def test_repr(self):
        qf = QuantileFamily(tau=0.75, sigma=0.5)
        assert "0.75" in repr(qf)
        assert "0.5" in repr(qf)

    def test_simulate_shape(self):
        qf = QuantileFamily(tau=0.5, sigma=1.0)
        rng = np.random.default_rng(23)
        mu = np.ones(100)
        y = qf.simulate(mu, 1.0, rng)
        assert y.shape == (100,)
        assert np.isfinite(y).all()

    def test_log_likelihood_weighted(self):
        qf = QuantileFamily(tau=0.4, sigma=1.0)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        mu = np.array([1.1, 1.9, 3.2, 3.9])
        w = np.array([2.0, 2.0, 2.0, 2.0])
        ll_unweighted = qf.log_likelihood(y, mu, scale=1.0)
        ll_weighted = qf.log_likelihood(y, mu, scale=1.0, weights=w)
        np.testing.assert_allclose(ll_weighted, 2.0 * ll_unweighted, rtol=1e-10)

    def test_initialize_copies_y(self):
        qf = QuantileFamily()
        y = np.array([1.0, 2.0, 3.0])
        mu0 = qf.initialize(y)
        np.testing.assert_array_equal(mu0, y)
        assert mu0 is not y
