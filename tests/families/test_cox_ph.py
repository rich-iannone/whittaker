"""Tests for the CoxPH family."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from whittaker.families.cox_ph import CoxPH


def _make_data(n: int = 30, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    time = rng.exponential(1.0, n)
    event = (rng.uniform(size=n) < 0.7).astype(float)
    return {"time": time, "event": event}


class TestCoxPHFamily:
    def setup_method(self) -> None:
        self.fam = CoxPH()

    def test_init_invalid_ties_raises(self) -> None:
        with pytest.raises(ValueError, match="ties must be"):
            CoxPH(ties="invalid")

    def test_ties_property(self) -> None:
        assert self.fam.ties == "breslow"
        assert CoxPH(ties="efron").ties == "efron"

    def test_link_is_identity(self) -> None:
        mu = np.array([0.0, 1.0, -1.0])
        assert_allclose(self.fam.link(mu), mu)

    def test_link_inverse_is_identity(self) -> None:
        eta = np.array([0.0, 1.0, -1.0])
        assert_allclose(self.fam.link_inverse(eta), eta)

    def test_link_derivative_is_ones(self) -> None:
        mu = np.array([1.0, 2.0, 3.0])
        assert_allclose(self.fam.link_derivative(mu), np.ones_like(mu))

    def test_variance_is_ones(self) -> None:
        mu = np.array([1.0, 2.0, 3.0])
        assert_allclose(self.fam.variance(mu), np.ones_like(mu))

    def test_scale_known_true(self) -> None:
        assert self.fam.scale_known is True

    def test_set_data_missing_status_column_raises(self) -> None:
        with pytest.raises(KeyError, match="Status column"):
            self.fam.set_data({"other_col": np.array([1.0, 0.0])})

    def test_set_data_custom_status_col(self) -> None:
        fam = CoxPH(status="died")
        fam.set_data({"died": np.array([1.0, 0.0, 1.0])})
        assert_allclose(fam._event, [1.0, 0.0, 1.0])

    def test_initialize_returns_zeros(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        mu0 = self.fam.initialize(y)
        assert_allclose(mu0, np.zeros_like(y))

    def test_partial_log_likelihood_before_set_data_is_zero(self) -> None:
        # initialize() sets _sort_idx, but without set_data() _event is still None,
        # so _partial_log_likelihood should short-circuit and return 0.0.
        y = np.array([1.0, 2.0, 3.0])
        self.fam.initialize(y)
        eta = np.zeros_like(y)
        assert self.fam.log_likelihood(y, eta, scale=1.0) == 0.0

    def test_partial_log_likelihood_zero_before_initialize(self) -> None:
        # Fresh family: _sort_idx is None too.
        eta = np.array([0.0, 1.0])
        assert self.fam._partial_log_likelihood(eta) == 0.0

    def test_irls_update_without_set_data_raises(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        eta = self.fam.initialize(y)
        with pytest.raises(RuntimeError, match="requires set_data"):
            self.fam.irls_update(y, eta, eta)

    def test_unit_deviance_is_ones(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([0.1, 0.2, 0.3])
        assert_allclose(self.fam.unit_deviance(y, mu), np.ones_like(y))

    def test_simulate_before_fit_raises(self) -> None:
        eta = np.array([0.0, 1.0])
        rng = np.random.default_rng(0)
        with pytest.raises(RuntimeError, match="must be fitted before simulation"):
            self.fam.simulate(eta, scale=1.0, rng=rng)

    def test_compute_baseline_hazard_noop_without_data(self) -> None:
        # Calling irls_update indirectly triggers _compute_baseline_hazard, but here
        # we test the guard clause directly: without set_data(), it should return
        # without raising and without setting baseline attributes.
        eta = np.array([0.0, 1.0, 2.0])
        self.fam._compute_baseline_hazard(eta)
        assert self.fam._baseline_cumhaz is None
        assert self.fam._baseline_times is None

    def test_baseline_hazard_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="Baseline hazard not computed"):
            self.fam.baseline_hazard()

    def test_survival_function_before_fit_raises(self) -> None:
        eta = np.array([0.0, 1.0])
        with pytest.raises(RuntimeError, match="Baseline hazard not computed"):
            self.fam.survival_function(eta)

    def test_repr(self) -> None:
        assert repr(self.fam) == "CoxPH(ties='breslow')"
        assert repr(CoxPH(ties="efron")) == "CoxPH(ties='efron')"

    @pytest.mark.parametrize("ties", ["breslow", "efron"])
    def test_full_fit_cycle(self, ties: str) -> None:
        data = _make_data(n=30, seed=1)
        fam = CoxPH(ties=ties)
        fam.set_data(data)
        eta = fam.initialize(data["time"])

        for _ in range(10):
            z, W = fam.irls_update(data["time"], eta, eta)
            assert np.all(np.isfinite(z))
            assert np.all(W > 0)
            eta = z

        dev = fam.deviance(data["time"], eta)
        ll = fam.log_likelihood(data["time"], eta, scale=1.0)
        assert_allclose(dev, -2.0 * ll)

        times, cumhaz = fam.baseline_hazard()
        assert len(times) == len(cumhaz)
        assert np.all(np.diff(times) >= 0)
        assert np.all(np.diff(cumhaz) >= 0)

        surv = fam.survival_function(eta)
        assert np.all(surv >= 0) and np.all(surv <= 1)

        rng = np.random.default_rng(2)
        sim_times = fam.simulate(eta, scale=1.0, rng=rng)
        assert sim_times.shape == eta.shape
        assert np.all(sim_times > 0)

    def test_efron_with_ties(self) -> None:
        # Force tied event times to exercise the Efron tie-correction branches.
        time = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 4.0])
        event = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        data = {"time": time, "event": event}

        fam = CoxPH(ties="efron")
        fam.set_data(data)
        eta = fam.initialize(time)
        eta = np.array([0.1, -0.2, 0.3, 0.0, 0.5, -0.1, 0.2])

        z, W = fam.irls_update(time, eta, eta)
        assert np.all(np.isfinite(z))
        assert np.all(W > 0)

        ll = fam.log_likelihood(time, eta, scale=1.0)
        assert np.isfinite(ll)
