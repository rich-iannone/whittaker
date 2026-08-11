"""Tests for DuckDB out-of-core GAM fitting."""

from __future__ import annotations

import numpy as np
import pytest

duckdb = pytest.importorskip("duckdb")

from whittaker.duckdb import DuckDBGAM, _count_rows, _normalize_source, _stream_as_dict
from whittaker.families.poisson import Poisson


@pytest.fixture
def conn_with_sin_data():
    """DuckDB connection with a table of sin(x) + noise data."""
    rng = np.random.default_rng(42)
    n = 500
    x = np.linspace(0, 2 * np.pi, n)
    y = np.sin(x) + rng.normal(0, 0.2, n)

    conn = duckdb.connect()
    conn.execute("CREATE TABLE sin_data (x DOUBLE, y DOUBLE)")
    for xi, yi in zip(x, y):
        conn.execute("INSERT INTO sin_data VALUES (?, ?)", [float(xi), float(yi)])
    return conn


@pytest.fixture
def conn_with_multi_data():
    """DuckDB connection with multi-predictor data."""
    rng = np.random.default_rng(42)
    n = 400
    x1 = np.linspace(0, 2 * np.pi, n)
    x2 = rng.uniform(0, 1, n)
    y = np.sin(x1) + x2**2 + rng.normal(0, 0.2, n)

    conn = duckdb.connect()
    conn.execute("CREATE TABLE multi_data (x1 DOUBLE, x2 DOUBLE, y DOUBLE)")
    for x1i, x2i, yi in zip(x1, x2, y):
        conn.execute(
            "INSERT INTO multi_data VALUES (?, ?, ?)",
            [float(x1i), float(x2i), float(yi)],
        )
    return conn


@pytest.fixture
def conn_with_poisson_data():
    """DuckDB connection with count data."""
    rng = np.random.default_rng(42)
    n = 400
    x = np.linspace(0, 3, n)
    y = rng.poisson(np.exp(0.5 * x))

    conn = duckdb.connect()
    conn.execute("CREATE TABLE count_data (x DOUBLE, y DOUBLE)")
    for xi, yi in zip(x, y):
        conn.execute("INSERT INTO count_data VALUES (?, ?)", [float(xi), float(yi)])
    return conn


class TestHelpers:
    def test_normalize_source_table(self):
        assert _normalize_source("my_table") == "SELECT * FROM my_table"

    def test_normalize_source_query(self):
        q = "SELECT * FROM foo WHERE x > 0"
        assert _normalize_source(q) == q

    def test_count_rows(self, conn_with_sin_data):
        n = _count_rows(conn_with_sin_data, "SELECT * FROM sin_data")
        assert n == 500

    def test_stream_as_dict(self, conn_with_sin_data):
        data = _stream_as_dict(conn_with_sin_data, "SELECT * FROM sin_data", chunk_size=100)
        assert "x" in data
        assert "y" in data
        assert len(data["x"]) == 500
        assert len(data["y"]) == 500


class TestDuckDBGAMBasic:
    def test_fit_from_table(self, conn_with_sin_data):
        model = DuckDBGAM("y ~ s(x)")
        model.fit("sin_data", conn_with_sin_data)
        assert model.is_fitted
        assert model.n_rows == 500

    def test_fit_from_query(self, conn_with_sin_data):
        model = DuckDBGAM("y ~ s(x)")
        model.fit("SELECT * FROM sin_data WHERE x > 1.0", conn_with_sin_data)
        assert model.is_fitted
        assert model.n_rows < 500

    def test_fit_query_method(self, conn_with_sin_data):
        model = DuckDBGAM("y ~ s(x)")
        model.fit_query("SELECT * FROM sin_data", conn_with_sin_data)
        assert model.is_fitted

    def test_properties(self, conn_with_sin_data):
        model = DuckDBGAM("y ~ s(x)", n_discrete=150, chunk_size=50_000)
        assert model.n_discrete == 150
        assert model.chunk_size == 50_000


class TestDuckDBGAMPredictions:
    def test_predict(self, conn_with_sin_data):
        model = DuckDBGAM("y ~ s(x)")
        model.fit("sin_data", conn_with_sin_data)

        new_data = {"x": np.linspace(0, 2 * np.pi, 50)}
        pred = model.predict(new_data)
        assert pred.values.shape == (50,)
        assert np.all(np.isfinite(pred.values))

    def test_predict_with_se(self, conn_with_sin_data):
        model = DuckDBGAM("y ~ s(x)")
        model.fit("sin_data", conn_with_sin_data)

        new_data = {"x": np.linspace(0, 2 * np.pi, 50)}
        pred = model.predict(new_data, se=True)
        assert pred.se is not None
        assert pred.se.shape == (50,)

    def test_predict_captures_signal(self, conn_with_sin_data):
        model = DuckDBGAM("y ~ s(x)")
        model.fit("sin_data", conn_with_sin_data)

        x_test = np.array([np.pi / 2, np.pi, 3 * np.pi / 2])
        pred = model.predict({"x": x_test})
        np.testing.assert_allclose(pred.values[0], 1.0, atol=0.3)
        np.testing.assert_allclose(pred.values[1], 0.0, atol=0.3)
        np.testing.assert_allclose(pred.values[2], -1.0, atol=0.3)

    def test_multi_smooth(self, conn_with_multi_data):
        model = DuckDBGAM("y ~ s(x1) + s(x2)")
        model.fit("multi_data", conn_with_multi_data)

        new_data = {
            "x1": np.linspace(0, 2 * np.pi, 30),
            "x2": np.random.default_rng(42).uniform(0, 1, 30),
        }
        pred = model.predict(new_data)
        assert pred.values.shape == (30,)
        assert np.all(np.isfinite(pred.values))

    def test_poisson_family(self, conn_with_poisson_data):
        model = DuckDBGAM("y ~ s(x)", family=Poisson())
        model.fit("count_data", conn_with_poisson_data)
        assert model.is_fitted

        pred = model.predict({"x": np.array([0.0, 1.5, 3.0])})
        assert np.all(pred.values > 0)


class TestDuckDBGAMSummary:
    def test_summary(self, conn_with_sin_data):
        model = DuckDBGAM("y ~ s(x)")
        model.fit("sin_data", conn_with_sin_data)
        s = model.summary()
        assert "s(x)" in s

    def test_deviance_explained(self, conn_with_sin_data):
        model = DuckDBGAM("y ~ s(x)")
        model.fit("sin_data", conn_with_sin_data)
        assert model.deviance_explained > 0.5


class TestDuckDBGAMAgreement:
    def test_agrees_with_biggam(self, conn_with_sin_data):
        """DuckDBGAM should produce similar results to BigGAM on same data."""
        from whittaker.bam import BigGAM

        rng = np.random.default_rng(42)
        n = 500
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.2, n)
        dict_data = {"x": x, "y": y}

        big_model = BigGAM("y ~ s(x)", n_discrete=200)
        big_model.fit(dict_data, method="fREML")

        duck_model = DuckDBGAM("y ~ s(x)", n_discrete=200)
        duck_model.fit("sin_data", conn_with_sin_data, method="fREML")

        x_test = np.linspace(0, 2 * np.pi, 20)
        pred_big = big_model.predict({"x": x_test})
        pred_duck = duck_model.predict({"x": x_test})

        np.testing.assert_allclose(pred_duck.values, pred_big.values, atol=0.1)


class TestDuckDBGAMParquet:
    def test_fit_from_parquet(self, tmp_path):
        """Test reading directly from a Parquet file via DuckDB."""
        rng = np.random.default_rng(42)
        n = 300
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.2, n)

        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            pytest.skip("pyarrow not installed")

        table = pa.table({"x": x, "y": y})
        parquet_path = tmp_path / "data.parquet"
        pq.write_table(table, str(parquet_path))

        conn = duckdb.connect()
        model = DuckDBGAM("y ~ s(x)")
        model.fit(f"SELECT * FROM '{parquet_path}'", conn)
        assert model.is_fitted
        assert model.n_rows == 300
