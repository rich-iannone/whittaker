"""DuckDB out-of-core GAM fitting.

Provides `DuckDBGAM`, a GAM variant that reads data from a DuckDB table or query.
Data is fetched via DuckDB's Arrow streaming interface and the model is fit using
discretized basis evaluation (same approach as `BigGAM`), so the full *n x p*
design matrix is never materialized.

The key memory savings: raw data is *n x d* (d covariates), while the design matrix
is *n x p* (p basis functions, often p >> d). By streaming the raw data and
discretizing, memory usage stays roughly *O(n d + grid_size * p)* instead of
*O(n p)*.

Requires the `duckdb` package (install via `pip install whittaker[duckdb]`).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from whittaker.bam import BigGAM, build_discretized_model_matrix
from whittaker.data import InternalData, _to_array
from whittaker.families.base import Family
from whittaker.fitting.bam import bam_fit
from whittaker.formula.terms import Formula
from whittaker.model_matrix import ModelMatrix


def _import_duckdb():
    try:
        import duckdb

        return duckdb
    except ImportError:
        raise ImportError(
            "DuckDB is required for DuckDBGAM. Install it with: pip install whittaker[duckdb]"
        ) from None


def _normalize_source(source: str) -> str:
    """Wrap a bare table name as a SELECT query."""
    stripped = source.strip()
    if stripped.upper().startswith("SELECT"):
        return stripped
    return f"SELECT * FROM {stripped}"


def _count_rows(conn, source_query: str) -> int:
    result = conn.sql(f"SELECT COUNT(*) FROM ({source_query})").fetchone()
    return int(result[0])


def _fetch_as_dict(conn, source_query: str) -> InternalData:
    """Fetch all rows from a DuckDB query as dict[str, NDArray]."""
    arrow_table = conn.sql(source_query).arrow()
    return {
        col: _to_array(arrow_table.column(col).to_numpy(zero_copy_only=False))
        for col in arrow_table.column_names
    }


def _stream_as_dict(conn, source_query: str, chunk_size: int = 100_000) -> InternalData:
    """Stream a DuckDB query into dict[str, NDArray] via Arrow batches.

    Builds the result incrementally so that only one batch is in memory at a
    time (plus the accumulating output arrays).
    """
    reader = conn.sql(source_query).fetch_arrow_reader(batch_size=chunk_size)
    columns: dict[str, list[NDArray]] = {}
    for batch in reader:
        for col in batch.schema.names:
            arr = _to_array(batch.column(col).to_numpy(zero_copy_only=False))
            columns.setdefault(col, []).append(arr)
    return {col: np.concatenate(parts) for col, parts in columns.items()}


class DuckDBGAM(BigGAM):
    """GAM that reads data from a DuckDB table or query.

    Data is fetched via DuckDB's Arrow interface and the model is fit using
    discretized basis evaluation (same as `BigGAM`), so the full design matrix
    is never materialized. This is useful when data lives in Parquet files,
    remote tables, or is produced by complex SQL transformations.

    Parameters
    ----------
    formula:
        Model formula (same as `~whittaker.gam.GAM`).
    family:
        Response distribution family.
    n_discrete:
        Number of discretization grid points per covariate.
    chunk_size:
        Number of rows per Arrow batch when streaming from DuckDB.

    Examples
    --------
    >>> import duckdb
    >>> conn = duckdb.connect()
    >>> conn.sql("CREATE TABLE data AS SELECT * FROM 'large_dataset.parquet'")
    >>> model = DuckDBGAM("y ~ s(x1) + s(x2)")
    >>> model.fit("data", conn)
    >>> predictions = model.predict(new_data)
    """

    def __init__(
        self,
        formula: str | Formula,
        family: Family | None = None,
        *,
        n_discrete: int = 200,
        chunk_size: int = 100_000,
    ) -> None:
        super().__init__(formula, family, n_discrete=n_discrete)
        self._chunk_size = chunk_size
        self._source: str | None = None
        self._n_rows: int = 0

    @property
    def chunk_size(self) -> int:
        """Arrow batch size when streaming from DuckDB."""
        return self._chunk_size

    @property
    def n_rows(self) -> int:
        """Total number of rows in the DuckDB source."""
        return self._n_rows

    def fit(
        self,
        source: str,
        conn,
        *,
        smoothing_params: list[float] | None = None,
        method: str = "fREML",
        select: bool = False,
    ) -> DuckDBGAM:
        """Fit the GAM by reading data from DuckDB.

        Data is streamed in Arrow batches of `chunk_size` rows, then discretized and fit using the
        BigGAM approach.

        Parameters
        ----------
        source:
            DuckDB table name or SQL query (e.g. `"my_table"` or
            `"SELECT * FROM my_table WHERE year > 2020"`).
        conn:
            DuckDB connection object (`duckdb.DuckDBPyConnection`).
        smoothing_params:
            Fixed smoothing parameters. If `None`, selected automatically.
        method:
            Smoothing selection method: `"fREML"` (default), `"REML"`, `"ML"`, or `"GCV"`.
        select:
            If `True`, enable double-penalty variable selection.

        Returns
        -------
        DuckDBGAM
            Returns `self` for method chaining.
        """
        _import_duckdb()
        self._source = source
        source_query = _normalize_source(source)

        self._n_rows = _count_rows(conn, source_query)

        data = _stream_as_dict(conn, source_query, self._chunk_size)
        self._data = data

        if hasattr(self._family, "set_data"):
            self._family.set_data(data)

        self._disc_model = build_discretized_model_matrix(
            self._formula, data, n_discrete=self._n_discrete, select=select
        )

        self._fit_result = bam_fit(
            self._disc_model,
            self._family,
            smoothing_params=smoothing_params,
            method=method,
        )

        self._model_matrix = ModelMatrix(
            X=np.empty((0, self._disc_model.n_cols)),
            penalties=self._disc_model.penalties,
            smooths=self._disc_model.smooth_infos,
            column_names=self._disc_model.column_names,
            has_intercept=self._disc_model.has_intercept,
            n_parametric=self._disc_model.n_parametric,
            offset=self._disc_model.offset,
            offset_expressions=self._disc_model.offset_expressions,
            response=self._disc_model.response,
        )

        self._fitted = True
        return self

    def fit_query(
        self,
        query: str,
        conn,
        **kwargs,
    ) -> DuckDBGAM:
        """Convenience method: fit from an explicit SQL query.

        Equivalent to `fit(query, conn, ...)` but makes the intent clearer
        when the source is a query rather than a table name.
        """
        return self.fit(query, conn, **kwargs)
