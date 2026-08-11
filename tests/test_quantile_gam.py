"""Tests for non-crossing QuantileGAM."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.quantile_gam import (
    QuantileGAM,
    _check_non_crossing,
    _enforce_non_crossing,
    _isotonic_projection,
)


@pytest.fixture
def symmetric_data():
    rng = np.random.default_rng(23)
    n = 300
    x = np.linspace(0, 2 * np.pi, n)
    y = np.sin(x) + rng.normal(0, 0.3, n)
    return {"x": x, "y": y}


@pytest.fixture
def heteroscedastic_data():
    rng = np.random.default_rng(23)
    n = 400
    x = np.linspace(0, 2 * np.pi, n)
    noise_scale = 0.1 + 0.5 * (x / (2 * np.pi))
    y = np.sin(x) + rng.normal(0, noise_scale, n)
    return {"x": x, "y": y}


class TestIsotonicProjection:
    def test_already_sorted(self):
        x = np.array([1.0, 2.0, 3.0])
        result = _isotonic_projection(x)
        np.testing.assert_allclose(result, x)

    def test_reversed(self):
        x = np.array([3.0, 2.0, 1.0])
        result = _isotonic_projection(x)
        assert np.all(np.diff(result) >= -1e-10)
        np.testing.assert_allclose(result, [2.0, 2.0, 2.0])

    def test_single_violation(self):
        x = np.array([1.0, 3.0, 2.0, 4.0])
        result = _isotonic_projection(x)
        assert np.all(np.diff(result) >= -1e-10)
        np.testing.assert_allclose(result, [1.0, 2.5, 2.5, 4.0])

    def test_preserves_mean(self):
        x = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
        result = _isotonic_projection(x)
        np.testing.assert_allclose(np.mean(result), np.mean(x))

    def test_single_element(self):
        x = np.array([42.0])
        result = _isotonic_projection(x)
        np.testing.assert_allclose(result, [42.0])


class TestEnforceNonCrossing:
    def test_already_ordered(self):
        taus = [0.1, 0.5, 0.9]
        fitted = {
            0.1: np.array([1.0, 2.0]),
            0.5: np.array([2.0, 3.0]),
            0.9: np.array([3.0, 4.0]),
        }
        corrected = _enforce_non_crossing(fitted, taus)
        for tau in taus:
            np.testing.assert_allclose(corrected[tau], fitted[tau])

    def test_fixes_crossing(self):
        taus = [0.1, 0.5, 0.9]
        fitted = {
            0.1: np.array([1.0, 3.0]),
            0.5: np.array([2.0, 2.0]),
            0.9: np.array([3.0, 1.0]),
        }
        corrected = _enforce_non_crossing(fitted, taus)
        for i in range(2):
            vals = [corrected[tau][i] for tau in taus]
            assert all(vals[j] <= vals[j + 1] + 1e-10 for j in range(len(vals) - 1))


class TestCheckNonCrossing:
    def test_ordered_passes(self):
        taus = [0.1, 0.5, 0.9]
        fitted = {
            0.1: np.array([1.0, 2.0]),
            0.5: np.array([2.0, 3.0]),
            0.9: np.array([3.0, 4.0]),
        }
        assert _check_non_crossing(fitted, taus) is True

    def test_crossing_fails(self):
        taus = [0.1, 0.5, 0.9]
        fitted = {
            0.1: np.array([1.0, 3.0]),
            0.5: np.array([2.0, 2.0]),
            0.9: np.array([3.0, 1.0]),
        }
        assert _check_non_crossing(fitted, taus) is False


class TestQuantileGAMInit:
    def test_default_quantiles(self):
        model = QuantileGAM("y ~ s(x)")
        assert model.quantiles == [0.1, 0.25, 0.5, 0.75, 0.9]

    def test_custom_quantiles(self):
        model = QuantileGAM("y ~ s(x)", quantiles=[0.25, 0.75])
        assert model.quantiles == [0.25, 0.75]

    def test_quantiles_sorted(self):
        model = QuantileGAM("y ~ s(x)", quantiles=[0.9, 0.1, 0.5])
        assert model.quantiles == [0.1, 0.5, 0.9]

    def test_invalid_quantile_raises(self):
        with pytest.raises(ValueError, match="must be in"):
            QuantileGAM("y ~ s(x)", quantiles=[0.0, 0.5])

    def test_repr(self):
        model = QuantileGAM("y ~ s(x)", quantiles=[0.1, 0.9])
        assert "QuantileGAM" in repr(model)
        assert "unfitted" in repr(model)


class TestQuantileGAMFit:
    def test_fit_basic(self, symmetric_data):
        model = QuantileGAM("y ~ s(x)", quantiles=[0.1, 0.5, 0.9])
        model.fit(symmetric_data)
        assert model.is_fitted

    def test_fit_no_crossing(self, symmetric_data):
        model = QuantileGAM("y ~ s(x)", quantiles=[0.1, 0.5, 0.9], non_crossing=True)
        model.fit(symmetric_data)
        assert model.is_fitted

    def test_fit_without_crossing_constraint(self, symmetric_data):
        model = QuantileGAM("y ~ s(x)", quantiles=[0.1, 0.5, 0.9], non_crossing=False)
        model.fit(symmetric_data)
        assert model.is_fitted

    def test_method_chaining(self, symmetric_data):
        result = QuantileGAM("y ~ s(x)", quantiles=[0.5]).fit(symmetric_data)
        assert isinstance(result, QuantileGAM)

    def test_unfitted_raises(self):
        model = QuantileGAM("y ~ s(x)")
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict({"x": np.array([1.0])})


class TestQuantileGAMPredict:
    def test_predict_returns_dict(self, symmetric_data):
        model = QuantileGAM("y ~ s(x)", quantiles=[0.1, 0.5, 0.9])
        model.fit(symmetric_data)
        preds = model.predict(symmetric_data)
        assert isinstance(preds, dict)
        assert set(preds.keys()) == {0.1, 0.5, 0.9}

    def test_predict_shapes(self, symmetric_data):
        model = QuantileGAM("y ~ s(x)", quantiles=[0.1, 0.5, 0.9])
        model.fit(symmetric_data)
        preds = model.predict(symmetric_data)
        for tau in [0.1, 0.5, 0.9]:
            assert preds[tau].values.shape == (300,)

    def test_predict_with_se(self, symmetric_data):
        model = QuantileGAM("y ~ s(x)", quantiles=[0.5])
        model.fit(symmetric_data)
        preds = model.predict(symmetric_data, se=True)
        assert preds[0.5].se is not None

    def test_predict_new_data(self, symmetric_data):
        model = QuantileGAM("y ~ s(x)", quantiles=[0.1, 0.5, 0.9])
        model.fit(symmetric_data)
        new_data = {"x": np.linspace(0, 2 * np.pi, 50)}
        preds = model.predict(new_data)
        assert preds[0.5].values.shape == (50,)

    def test_non_crossing_in_predict(self, symmetric_data):
        model = QuantileGAM("y ~ s(x)", quantiles=[0.1, 0.5, 0.9], non_crossing=True)
        model.fit(symmetric_data)
        preds = model.predict(symmetric_data)
        vals_10 = preds[0.1].values
        vals_50 = preds[0.5].values
        vals_90 = preds[0.9].values
        assert np.all(vals_10 <= vals_50 + 1e-10)
        assert np.all(vals_50 <= vals_90 + 1e-10)

    def test_non_crossing_on_new_data(self, symmetric_data):
        model = QuantileGAM("y ~ s(x)", quantiles=[0.1, 0.25, 0.5, 0.75, 0.9], non_crossing=True)
        model.fit(symmetric_data)
        new_data = {"x": np.linspace(0, 2 * np.pi, 100)}
        preds = model.predict(new_data)
        for i in range(len(model.quantiles) - 1):
            tau_lo = model.quantiles[i]
            tau_hi = model.quantiles[i + 1]
            assert np.all(preds[tau_lo].values <= preds[tau_hi].values + 1e-10)


class TestQuantileGAMOrdering:
    def test_median_between_extremes(self, symmetric_data):
        model = QuantileGAM("y ~ s(x)", quantiles=[0.1, 0.5, 0.9])
        model.fit(symmetric_data)
        preds = model.predict(symmetric_data)
        assert np.all(preds[0.1].values <= preds[0.5].values + 1e-10)
        assert np.all(preds[0.5].values <= preds[0.9].values + 1e-10)

    def test_wider_intervals_for_wider_quantiles(self, symmetric_data):
        model = QuantileGAM("y ~ s(x)", quantiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        model.fit(symmetric_data)
        preds = model.predict(symmetric_data)
        width_outer = np.mean(preds[0.9].values - preds[0.1].values)
        width_inner = np.mean(preds[0.75].values - preds[0.25].values)
        assert width_outer > width_inner


class TestPredictInterval:
    def test_basic(self, symmetric_data):
        model = QuantileGAM("y ~ s(x)", quantiles=[0.1, 0.5, 0.9])
        model.fit(symmetric_data)
        lower, upper = model.predict_interval(symmetric_data)
        assert lower.shape == (300,)
        assert upper.shape == (300,)
        assert np.all(lower <= upper + 1e-10)

    def test_custom_taus(self, symmetric_data):
        model = QuantileGAM("y ~ s(x)", quantiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        model.fit(symmetric_data)
        lower, upper = model.predict_interval(symmetric_data, lower_tau=0.25, upper_tau=0.75)
        lower_full, upper_full = model.predict_interval(symmetric_data)
        assert np.mean(upper - lower) < np.mean(upper_full - lower_full)

    def test_invalid_tau_raises(self, symmetric_data):
        model = QuantileGAM("y ~ s(x)", quantiles=[0.1, 0.9])
        model.fit(symmetric_data)
        with pytest.raises(ValueError, match="not fitted"):
            model.predict_interval(symmetric_data, lower_tau=0.25)


class TestCoverage:
    def test_reasonable_coverage(self, symmetric_data):
        model = QuantileGAM("y ~ s(x)", quantiles=[0.1, 0.9])
        model.fit(symmetric_data)
        cov = model.coverage()
        assert 0.6 < cov < 0.95

    def test_wider_quantiles_more_coverage(self, symmetric_data):
        model_narrow = QuantileGAM("y ~ s(x)", quantiles=[0.25, 0.75])
        model_narrow.fit(symmetric_data)

        model_wide = QuantileGAM("y ~ s(x)", quantiles=[0.1, 0.9])
        model_wide.fit(symmetric_data)

        assert model_wide.coverage() > model_narrow.coverage()


class TestCrossingFraction:
    def test_non_crossing_zero(self, symmetric_data):
        model = QuantileGAM("y ~ s(x)", quantiles=[0.1, 0.5, 0.9], non_crossing=True)
        model.fit(symmetric_data)
        assert model.crossing_fraction() < 0.05

    def test_without_constraint_may_cross(self, heteroscedastic_data):
        model = QuantileGAM("y ~ s(x)", quantiles=[0.1, 0.5, 0.9], non_crossing=False)
        model.fit(heteroscedastic_data)
        frac = model.crossing_fraction()
        assert isinstance(frac, float)
        assert 0.0 <= frac <= 1.0


class TestSummary:
    def test_summary(self, symmetric_data):
        model = QuantileGAM("y ~ s(x)", quantiles=[0.1, 0.5, 0.9])
        model.fit(symmetric_data)
        s = model.summary()
        assert "QuantileGAM" in s
        assert "tau=0.10" in s
        assert "tau=0.50" in s
        assert "tau=0.90" in s


class TestHeteroscedastic:
    def test_wider_intervals_at_high_x(self, heteroscedastic_data):
        data = heteroscedastic_data
        model = QuantileGAM("y ~ s(x)", quantiles=[0.1, 0.9], non_crossing=True)
        model.fit(data)
        preds = model.predict(data)
        width = preds[0.9].values - preds[0.1].values
        x = data["x"]
        low_x = width[x < np.pi]
        high_x = width[x > np.pi]
        assert np.mean(high_x) > np.mean(low_x)
