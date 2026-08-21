"""DuckDB out-of-core GAM fitting.

Provides `DuckDBGAM`, a GAM variant that reads data from a DuckDB table, view, or
arbitrary SQL query. Data is fetched via DuckDB's Arrow streaming interface, in
`chunk_size`-row batches, and the model is fit using discretized basis evaluation
(the same approach as `BigGAM`), so the full *n x p* design matrix is never
materialized.

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
    except ImportError:  # pragma: no cover - exercised only when duckdb is not installed
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
    arrow_table = conn.sql(source_query).fetch_arrow_table()
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
    r"""GAM that reads data directly from DuckDB via SQL.

    `DuckDBGAM` is a SQL-native variant of `BigGAM` for fitting a GAM without first
    loading the source data into pandas or polars. `fit()` accepts either a bare
    table/view name or an arbitrary `SELECT` query — anything expressible in SQL,
    including joins, filters, aggregations, and window functions — and DuckDB does
    the work of producing the resulting rows. Internally, `_stream_as_dict` reads
    those rows through DuckDB's Arrow batch interface
    (`conn.sql(query).fetch_arrow_reader(batch_size=chunk_size)`), concatenating
    `chunk_size`-row Arrow batches into the column-oriented dict that
    `build_discretized_model_matrix` and `bam_fit` (the same discretized-basis
    machinery used by `BigGAM`) consume to fit the model. Use `DuckDBGAM` whenever
    the training data already lives in DuckDB, in Parquet/CSV files DuckDB can scan
    directly, or in a view/query that would be expensive to materialize by hand
    before fitting.

    Parameters
    ----------
    formula : str or Formula
        Model formula (same syntax as `~whittaker.gam.GAM`), e.g.
        `"y ~ s(x1) + s(x2)"`. The left-hand side names the response column that
        must be selectable from `source`; the right-hand side lists smooth terms,
        linear terms, and interactions in `mgcv`-style syntax.
    family : Family, optional
        Response distribution family, e.g. `Gaussian()`, `Binomial()`, `Poisson()`,
        or `Gamma()`. Defaults to `Gaussian()` (identity link).
    n_discrete : int
        Number of discretization grid points per covariate used when building the
        discretized model matrix (see `build_discretized_model_matrix`). Larger
        values give a more accurate approximation to the exact basis evaluation at
        the cost of a larger `grid_size * p` term in memory and compute. Defaults
        to `200`, which is adequate for most smooths.
    chunk_size : int
        Number of rows per Arrow batch fetched from DuckDB while streaming
        (see `Notes` below). Larger values reduce Python-level batch-processing
        overhead but increase peak memory per batch; smaller values do the
        opposite. Defaults to `100_000`.

    Notes
    -----
    Memory usage during fitting is bounded by *O(n d + grid_size * p)* rather than
    *O(n p)*, where *n* is the row count, *d* the number of raw covariates, *p* the
    number of basis functions, and *grid_size* the discretization resolution
    (`n_discrete`). The `n d` term comes from streaming the raw columns rather than
    an *n x p* design matrix; the `grid_size * p` term comes from evaluating the
    basis only at the discretization grid. `_stream_as_dict` is what keeps the raw
    data at `O(n d)` rather than materializing a full *n x p* matrix twice as
    `conn.sql(...).df()` would: it pulls one `chunk_size`-row Arrow batch at a time
    from `fetch_arrow_reader` and appends each batch's columns to a list, so DuckDB
    never has to build (and Python never has to hold) more than one batch's worth
    of Arrow data plus the columns accumulated so far.

    Examples
    --------
    ```{python}
    import duckdb
    import numpy as np

    import whittaker as wt
    from whittaker.duckdb import DuckDBGAM

    conn = duckdb.connect()
    rng = np.random.default_rng(0)
    conn.execute(
        "CREATE TABLE data AS "
        "SELECT i AS id, (i / 100.0) AS x, "
        "sin(2 * pi() * i / 100.0) + ? * random() AS y "
        "FROM range(200) AS t(i)",
        [0.2],
    )

    model = DuckDBGAM("y ~ s(x)").fit("data", conn)
    print(model.summary())
    ```

    A query can be used directly in place of a table name, e.g. to filter or join
    before fitting:

    ```{python}
    model2 = DuckDBGAM("y ~ s(x)").fit_query(
        "SELECT x, y FROM data WHERE x < 0.8", conn
    )
    print(model2.n_rows)
    ```
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
        """Arrow batch size used when streaming from DuckDB.

        This is the `chunk_size` value passed to `__init__`: the number of rows per Arrow batch
        that `_stream_as_dict` requests from `conn.sql(query).fetch_arrow_reader(batch_size=...)`
        while reading `fit()`'s data source.

        Returns
        -------
        int
            The configured Arrow batch size, in rows.
        """
        return self._chunk_size

    @property
    def n_rows(self) -> int:
        """Total number of rows in the DuckDB source used by the most recent fit.

        Populated by `fit()` via `_count_rows`, which runs a `SELECT COUNT(*)` against the
        normalized source query before streaming the data. Remains `0` until `fit()` has been
        called at least once.

        Returns
        -------
        int
            Row count of the table, view, or query passed to `fit()`.
        """
        return self._n_rows

    def fit(  # type: ignore[override]
        self,
        source: str,
        conn,
        *,
        smoothing_params: list[float] | None = None,
        method: str = "fREML",
        select: bool = False,
    ) -> DuckDBGAM:
        """Fit the GAM by reading data from DuckDB.

        Data is streamed in Arrow batches of `chunk_size` rows (see `_stream_as_dict`), then
        discretized and fit using the `BigGAM` approach (`build_discretized_model_matrix` +
        `bam_fit`). The full design matrix is never materialized.

        Parameters
        ----------
        source : str
            DuckDB table or view name, or a full `SELECT` query. A bare name such as
            `"my_table"` is normalized by `_normalize_source` into
            `"SELECT * FROM my_table"`; a string that already starts with `SELECT`
            (case-insensitive) is used as-is, e.g.
            `"SELECT * FROM my_table WHERE year > 2020"`. Any query DuckDB can execute
            is accepted, including joins, aggregations, and window functions, as long
            as its output columns cover every variable referenced by `formula`.
        conn : duckdb.DuckDBPyConnection
            A live DuckDB connection (as returned by `duckdb.connect()`) against which
            `data` is resolved. The connection must remain open for the duration of
            `fit()`; it is not closed or modified by this method beyond issuing read
            queries.
        smoothing_params : list of float, optional
            Fixed smoothing parameters. If `None`, selected automatically.
        method : str
            Smoothing selection method: `"fREML"` (default), `"REML"`, `"ML"`, or `"GCV"`.
        select : bool
            If `True`, enable double-penalty variable selection.

        Returns
        -------
        DuckDBGAM
            Returns `self` for method chaining.

        Notes
        -----
        When `data` is unambiguously a SQL query rather than a table/view name,
        `fit_query()` is a thin, more explicit alias for this method.
        """
        _import_duckdb()
        self._source = source
        source_query = _normalize_source(source)

        self._n_rows = _count_rows(conn, source_query)

        col_data = _stream_as_dict(conn, source_query, self._chunk_size)
        self._data = col_data

        _set_data = getattr(self._family, "set_data", None)
        if _set_data is not None:
            _set_data(col_data)

        self._disc_model = build_discretized_model_matrix(
            self._formula, col_data, n_discrete=self._n_discrete, select=select
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
        """Fit the GAM from an explicit SQL query.

        Equivalent to calling `fit(query, conn, ...)` directly; this method exists purely to
        document intent at the call site when `source` is unambiguously a SQL query (e.g. one
        with a `WHERE`, `JOIN`, or aggregation) rather than a bare table or view name.

        Parameters
        ----------
        query : str
            A full `SELECT` query, e.g. `"SELECT * FROM my_table WHERE year > 2020"`. Its
            output columns must cover every variable referenced by `formula`.
        conn : duckdb.DuckDBPyConnection
            A live DuckDB connection against which `query` is executed.
        **kwargs
            Additional keyword arguments forwarded to `fit()`: `smoothing_params`, `method`,
            and `select`.

        Returns
        -------
        DuckDBGAM
            Returns `self` for method chaining.
        """
        return self.fit(query, conn, **kwargs)
