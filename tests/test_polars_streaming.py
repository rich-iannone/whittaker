"""Tests for Polars streaming GAM fitting."""

from __future__ import annotations

import numpy as np
import pytest

pl = pytest.importorskip("polars")

from whittaker.families.poisson import Poisson
from whittaker.polars_streaming import PolarsGAM, _count_lazy, _to_lazy


@pytest.fixture
def sin_df():
    """Polars DataFrame with sin(x) + noise data."""
    rng = np.random.default_rng(23)
    n = 500
    x = np.linspace(0, 2 * np.pi, n)
    y = np.sin(x) + rng.normal(0, 0.2, n)
    return pl.DataFrame({"x": x, "y": y})


@pytest.fixture
def sin_lazy(sin_df):
    """Polars LazyFrame with sin data."""
    return sin_df.lazy()


@pytest.fixture
def multi_df():
    """Polars DataFrame with multi-predictor data."""
    rng = np.random.default_rng(23)
    n = 400
    x1 = np.linspace(0, 2 * np.pi, n)
    x2 = rng.uniform(0, 1, n)
    y = np.sin(x1) + x2**2 + rng.normal(0, 0.2, n)
    return pl.DataFrame({"x1": x1, "x2": x2, "y": y})


@pytest.fixture
def poisson_df():
    """Polars DataFrame with count data."""
    rng = np.random.default_rng(23)
    n = 400
    x = np.linspace(0, 3, n)
    y = rng.poisson(np.exp(0.5 * x)).astype(float)
    return pl.DataFrame({"x": x, "y": y})


class TestHelpers:
    def test_to_lazy_from_lazy(self, sin_lazy):
        lf = _to_lazy(sin_lazy)
        assert isinstance(lf, pl.LazyFrame)

    def test_to_lazy_from_df(self, sin_df):
        lf = _to_lazy(sin_df)
        assert isinstance(lf, pl.LazyFrame)

    def test_to_lazy_from_parquet(self, sin_df, tmp_path):
        path = tmp_path / "data.parquet"
        sin_df.write_parquet(str(path))
        lf = _to_lazy(str(path))
        assert isinstance(lf, pl.LazyFrame)

    def test_to_lazy_from_csv(self, sin_df, tmp_path):
        path = tmp_path / "data.csv"
        sin_df.write_csv(str(path))
        lf = _to_lazy(str(path))
        assert isinstance(lf, pl.LazyFrame)

    def test_to_lazy_bad_extension(self):
        with pytest.raises(ValueError, match="Cannot infer"):
            _to_lazy("data.xlsx")

    def test_to_lazy_bad_type(self):
        with pytest.raises(TypeError, match="Unsupported"):
            _to_lazy(23)

    def test_count_lazy(self, sin_lazy):
        n = _count_lazy(sin_lazy)
        assert n == 500


class TestPolarsGAMBasic:
    def test_fit_from_lazy(self, sin_lazy):
        model = PolarsGAM("y ~ s(x)")
        model.fit(sin_lazy)
        assert model.is_fitted
        assert model.n_rows == 500

    def test_fit_from_df(self, sin_df):
        model = PolarsGAM("y ~ s(x)")
        model.fit(sin_df)
        assert model.is_fitted

    def test_fit_from_parquet(self, sin_df, tmp_path):
        path = tmp_path / "data.parquet"
        sin_df.write_parquet(str(path))
        model = PolarsGAM("y ~ s(x)")
        model.fit(str(path))
        assert model.is_fitted
        assert model.n_rows == 500

    def test_properties(self):
        model = PolarsGAM("y ~ s(x)", n_discrete=150, chunk_size=50_000)
        assert model.n_discrete == 150
        assert model.chunk_size == 50_000


class TestPolarsGAMPredictions:
    def test_predict(self, sin_lazy):
        model = PolarsGAM("y ~ s(x)")
        model.fit(sin_lazy)

        new_data = {"x": np.linspace(0, 2 * np.pi, 50)}
        pred = model.predict(new_data)
        assert pred.values.shape == (50,)
        assert np.all(np.isfinite(pred.values))

    def test_predict_with_se(self, sin_lazy):
        model = PolarsGAM("y ~ s(x)")
        model.fit(sin_lazy)

        new_data = {"x": np.linspace(0, 2 * np.pi, 50)}
        pred = model.predict(new_data, se=True)
        assert pred.se is not None
        assert pred.se.shape == (50,)

    def test_predict_captures_signal(self, sin_lazy):
        model = PolarsGAM("y ~ s(x)")
        model.fit(sin_lazy)

        x_test = np.array([np.pi / 2, np.pi, 3 * np.pi / 2])
        pred = model.predict({"x": x_test})
        np.testing.assert_allclose(pred.values[0], 1.0, atol=0.3)
        np.testing.assert_allclose(pred.values[1], 0.0, atol=0.3)
        np.testing.assert_allclose(pred.values[2], -1.0, atol=0.3)

    def test_multi_smooth(self, multi_df):
        model = PolarsGAM("y ~ s(x1) + s(x2)")
        model.fit(multi_df.lazy())

        new_data = {
            "x1": np.linspace(0, 2 * np.pi, 30),
            "x2": np.random.default_rng(23).uniform(0, 1, 30),
        }
        pred = model.predict(new_data)
        assert pred.values.shape == (30,)
        assert np.all(np.isfinite(pred.values))

    def test_poisson_family(self, poisson_df):
        model = PolarsGAM("y ~ s(x)", family=Poisson())
        model.fit(poisson_df.lazy())
        assert model.is_fitted

        pred = model.predict({"x": np.array([0.0, 1.5, 3.0])})
        assert np.all(pred.values > 0)


class TestPolarsGAMSummary:
    def test_summary(self, sin_lazy):
        model = PolarsGAM("y ~ s(x)")
        model.fit(sin_lazy)
        s = model.summary()
        assert "s(x)" in s

    def test_deviance_explained(self, sin_lazy):
        model = PolarsGAM("y ~ s(x)")
        model.fit(sin_lazy)
        assert model.deviance_explained > 0.5


class TestPolarsGAMAgreement:
    def test_agrees_with_biggam(self, sin_df):
        """PolarsGAM should produce similar results to BigGAM on same data."""
        from whittaker.bam import BigGAM

        dict_data = {col: sin_df[col].to_numpy() for col in sin_df.columns}

        big_model = BigGAM("y ~ s(x)", n_discrete=200)
        big_model.fit(dict_data, method="fREML")

        polars_model = PolarsGAM("y ~ s(x)", n_discrete=200)
        polars_model.fit(sin_df)

        x_test = np.linspace(0, 2 * np.pi, 20)
        pred_big = big_model.predict({"x": x_test})
        pred_polars = polars_model.predict({"x": x_test})

        np.testing.assert_allclose(pred_polars.values, pred_big.values, atol=1e-10)


class TestPolarsGAMFiltered:
    def test_filtered_lazy(self, sin_df):
        """Test that filtering a LazyFrame before fit works."""
        lf = sin_df.lazy().filter(pl.col("x") > 1.0)
        model = PolarsGAM("y ~ s(x)")
        model.fit(lf)
        assert model.is_fitted
        assert model.n_rows < 500
