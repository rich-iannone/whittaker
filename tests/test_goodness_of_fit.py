"""Tests for GAM.goodness_of_fit() and GoodnessOfFit."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker import GAM, GoodnessOfFit
from whittaker.families.poisson import Poisson

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gaussian_data():
    rng = np.random.default_rng(0)
    n = 100
    x = np.linspace(0, 5, n)
    y = np.sin(x) + rng.normal(scale=0.3, size=n)
    return {"x": x, "y": y}


@pytest.fixture(scope="module")
def reml_model(gaussian_data):
    return GAM("y ~ s(x)").fit(gaussian_data, method="REML")


@pytest.fixture(scope="module")
def vi_model(gaussian_data):
    return GAM("y ~ s(x)").fit(gaussian_data, method="VI")


@pytest.fixture(scope="module")
def mcmc_model(gaussian_data):
    return GAM("y ~ s(x)").fit(
        gaussian_data,
        method="MCMC",
        mcmc_options={"n_chains": 2, "n_samples": 200, "n_warmup": 100, "seed": 23},
    )


# ---------------------------------------------------------------------------
# GoodnessOfFit structure
# ---------------------------------------------------------------------------


class TestGoodnessOfFitStructure:
    """Verify the dataclass fields and types."""

    def test_returns_correct_type(self, reml_model):
        gof = reml_model.goodness_of_fit()

        assert isinstance(gof, GoodnessOfFit)

    def test_all_fields_present(self, reml_model):
        gof = reml_model.goodness_of_fit()

        assert hasattr(gof, "deviance")
        assert hasattr(gof, "null_deviance")
        assert hasattr(gof, "deviance_explained")
        assert hasattr(gof, "r_squared_adj")
        assert hasattr(gof, "aic")
        assert hasattr(gof, "bic")
        assert hasattr(gof, "gcv_score")
        assert hasattr(gof, "scale")
        assert hasattr(gof, "edf_total")
        assert hasattr(gof, "n_obs")

    def test_n_obs(self, reml_model, gaussian_data):
        gof = reml_model.goodness_of_fit()

        assert gof.n_obs == len(gaussian_data["y"])

    def test_repr(self, reml_model):
        gof = reml_model.goodness_of_fit()
        r = repr(gof)

        assert "GoodnessOfFit" in r
        assert "Deviance" in r
        assert "AIC" in r
        assert "BIC" in r
        assert "Adj. R-squared" in r
        assert "GCV" in r


# ---------------------------------------------------------------------------
# Consistency with individual properties
# ---------------------------------------------------------------------------


class TestConsistency:
    """Verify goodness_of_fit() matches the individual properties."""

    def test_deviance_matches(self, reml_model):
        gof = reml_model.goodness_of_fit()

        assert gof.deviance == reml_model.deviance

    def test_null_deviance_matches(self, reml_model):
        gof = reml_model.goodness_of_fit()

        assert gof.null_deviance == reml_model.null_deviance

    def test_deviance_explained_matches(self, reml_model):
        gof = reml_model.goodness_of_fit()

        assert gof.deviance_explained == reml_model.deviance_explained

    def test_aic_matches(self, reml_model):
        gof = reml_model.goodness_of_fit()

        assert gof.aic == reml_model.aic

    def test_bic_matches(self, reml_model):
        gof = reml_model.goodness_of_fit()

        assert gof.bic == reml_model.bic

    def test_gcv_matches(self, reml_model):
        gof = reml_model.goodness_of_fit()

        assert gof.gcv_score == reml_model.gcv_score

    def test_scale_matches(self, reml_model):
        gof = reml_model.goodness_of_fit()

        assert gof.scale == reml_model.scale

    def test_edf_total_matches(self, reml_model):
        gof = reml_model.goodness_of_fit()

        assert gof.edf_total == reml_model.edf_total


# ---------------------------------------------------------------------------
# Adjusted R-squared
# ---------------------------------------------------------------------------


class TestAdjustedRSquared:
    """Verify the adjusted R-squared calculation."""

    def test_adj_r_squared_finite(self, reml_model):
        gof = reml_model.goodness_of_fit()

        assert np.isfinite(gof.r_squared_adj)

    def test_adj_r_squared_less_than_deviance_explained(self, reml_model):
        gof = reml_model.goodness_of_fit()

        assert gof.r_squared_adj <= gof.deviance_explained

    def test_adj_r_squared_formula(self, reml_model, gaussian_data):
        gof = reml_model.goodness_of_fit()
        n = len(gaussian_data["y"])
        expected = 1.0 - (1.0 - gof.deviance_explained) * (n - 1.0) / (n - gof.edf_total - 1.0)

        np.testing.assert_allclose(gof.r_squared_adj, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Bayesian fits
# ---------------------------------------------------------------------------


class TestBayesianFits:
    """Verify goodness_of_fit works with VI and MCMC."""

    def test_vi_gcv_is_none(self, vi_model):
        gof = vi_model.goodness_of_fit()

        assert gof.gcv_score is None

    def test_vi_has_all_other_fields(self, vi_model):
        gof = vi_model.goodness_of_fit()

        assert np.isfinite(gof.deviance)
        assert np.isfinite(gof.aic)
        assert np.isfinite(gof.bic)
        assert np.isfinite(gof.deviance_explained)
        assert np.isfinite(gof.r_squared_adj)

    def test_vi_repr_no_gcv(self, vi_model):
        gof = vi_model.goodness_of_fit()

        assert "GCV" not in repr(gof)

    def test_mcmc_gcv_is_none(self, mcmc_model):
        gof = mcmc_model.goodness_of_fit()

        assert gof.gcv_score is None

    def test_mcmc_has_all_other_fields(self, mcmc_model):
        gof = mcmc_model.goodness_of_fit()

        assert np.isfinite(gof.deviance)
        assert np.isfinite(gof.aic)
        assert np.isfinite(gof.bic)
        assert np.isfinite(gof.deviance_explained)


# ---------------------------------------------------------------------------
# Non-Gaussian family
# ---------------------------------------------------------------------------


class TestPoissonGoodnessOfFit:
    def test_poisson_gof(self):
        rng = np.random.default_rng(23)
        n = 100
        x = np.linspace(0, 3, n)
        y = rng.poisson(np.exp(0.5 * np.sin(x))).astype(float)
        data = {"x": x, "y": y}

        model = GAM("y ~ s(x)", family=Poisson()).fit(data)
        gof = model.goodness_of_fit()

        assert gof.scale == 1.0
        assert np.isfinite(gof.deviance)
        assert np.isfinite(gof.aic)
        assert gof.deviance_explained >= 0.0


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TestErrors:
    def test_unfitted_raises(self):
        gam = GAM("y ~ s(x)")
        with pytest.raises(RuntimeError, match="not been fitted"):
            gam.goodness_of_fit()
