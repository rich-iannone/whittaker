"""Tests for the Narwhals data input layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from whittaker.data import prepare_data
from whittaker.gam import GAM


@pytest.fixture()
def sinusoidal_arrays():
    rng = np.random.default_rng(23)
    n = 100
    x = np.linspace(0, 2 * np.pi, n)
    y = np.sin(x) + rng.normal(0, 0.2, n)
    return x, y


class TestPrepareData:
    def test_dict_passthrough(self):
        data = {"x": np.array([1.0, 2.0, 3.0]), "y": np.array([4.0, 5.0, 6.0])}
        result = prepare_data(data)
        assert isinstance(result, dict)
        np.testing.assert_array_equal(result["x"], data["x"])

    def test_dict_casts_to_float(self):
        data = {"x": [1, 2, 3], "y": [4, 5, 6]}
        result = prepare_data(data)
        assert result["x"].dtype == np.float64

    def test_pandas_dataframe(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        result = prepare_data(df)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"x", "y"}
        np.testing.assert_array_equal(result["x"], [1.0, 2.0, 3.0])

    def test_polars_dataframe(self):
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        result = prepare_data(df)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"x", "y"}
        np.testing.assert_array_equal(result["x"], [1.0, 2.0, 3.0])

    def test_polars_lazyframe_rejected(self):
        lf = pl.LazyFrame({"x": [1.0, 2.0, 3.0]})
        with pytest.raises(TypeError):
            prepare_data(lf)

    def test_unsupported_type(self):
        with pytest.raises(TypeError, match="Unsupported data type"):
            prepare_data("not a dataframe")

    def test_pandas_integer_columns(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        result = prepare_data(df)
        assert result["x"].dtype == np.float64


class TestGAMWithDataFrames:
    def test_fit_predict_pandas(self, sinusoidal_arrays):
        x, y = sinusoidal_arrays
        df = pd.DataFrame({"x": x, "y": y})
        model = GAM("y ~ s(x)")
        model.fit(df)
        pred = model.predict(df)
        assert pred.values.shape == (len(x),)
        assert np.corrcoef(y, pred.values)[0, 1] > 0.9

    def test_fit_predict_polars(self, sinusoidal_arrays):
        x, y = sinusoidal_arrays
        df = pl.DataFrame({"x": x, "y": y})
        model = GAM("y ~ s(x)")
        model.fit(df)
        pred = model.predict(df)
        assert pred.values.shape == (len(x),)
        assert np.corrcoef(y, pred.values)[0, 1] > 0.9

    def test_fit_dict_predict_pandas(self, sinusoidal_arrays):
        x, y = sinusoidal_arrays
        data_dict = {"x": x, "y": y}
        model = GAM("y ~ s(x)")
        model.fit(data_dict)
        pred_df = model.predict(pd.DataFrame({"x": x}))
        pred_dict = model.predict({"x": x})
        np.testing.assert_array_almost_equal(pred_df.values, pred_dict.values)

    def test_fit_pandas_predict_polars(self, sinusoidal_arrays):
        x, y = sinusoidal_arrays
        model = GAM("y ~ s(x)")
        model.fit(pd.DataFrame({"x": x, "y": y}))
        pred = model.predict(pl.DataFrame({"x": x}))
        assert pred.values.shape == (len(x),)

    def test_fit_polars_predict_dict(self, sinusoidal_arrays):
        x, y = sinusoidal_arrays
        model = GAM("y ~ s(x)")
        model.fit(pl.DataFrame({"x": x, "y": y}))
        pred = model.predict({"x": x})
        assert pred.values.shape == (len(x),)

    def test_results_match_across_formats(self, sinusoidal_arrays):
        x, y = sinusoidal_arrays
        data_dict = {"x": x, "y": y}
        data_pd = pd.DataFrame(data_dict)
        data_pl = pl.DataFrame(data_dict)

        models = []
        for data in [data_dict, data_pd, data_pl]:
            m = GAM("y ~ s(x)")
            m.fit(data)
            models.append(m)

        pred0 = models[0].predict(data_dict).values
        pred1 = models[1].predict(data_pd).values
        pred2 = models[2].predict(data_pl).values

        np.testing.assert_array_almost_equal(pred0, pred1)
        np.testing.assert_array_almost_equal(pred0, pred2)

    def test_pandas_with_se(self, sinusoidal_arrays):
        x, y = sinusoidal_arrays
        df = pd.DataFrame({"x": x, "y": y})
        model = GAM("y ~ s(x)")
        model.fit(df)
        pred = model.predict(df, se=True)
        assert pred.se is not None
        assert pred.se.shape == (len(x),)

    def test_polars_with_interval(self, sinusoidal_arrays):
        x, y = sinusoidal_arrays
        df = pl.DataFrame({"x": x, "y": y})
        model = GAM("y ~ s(x)")
        model.fit(df)
        pred = model.predict(df, interval="confidence")
        assert pred.lower is not None
        assert pred.upper is not None


class TestCrossValidateWithDataFrames:
    def test_cv_pandas(self, sinusoidal_arrays):
        from whittaker.cross_validation import cross_validate

        x, y = sinusoidal_arrays
        df = pd.DataFrame({"x": x, "y": y})
        result = cross_validate("y ~ s(x)", df, n_folds=3, seed=23)
        assert result.cv_score > 0

    def test_cv_polars(self, sinusoidal_arrays):
        from whittaker.cross_validation import cross_validate

        x, y = sinusoidal_arrays
        df = pl.DataFrame({"x": x, "y": y})
        result = cross_validate("y ~ s(x)", df, n_folds=3, seed=23)
        assert result.cv_score > 0
