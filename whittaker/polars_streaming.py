"""Polars streaming GAM fitting.

Provides `PolarsGAM`, a GAM variant that reads data from a Polars LazyFrame or file path
(CSV/Parquet) and fits the model using discretized basis evaluation (same approach as `BigGAM`).
Data is processed in chunks via Polars' streaming engine, so the full dataset is never fully
materialized.

Suitable for datasets in the 1M--100M row range. For even larger datasets backed by DuckDB, see
`~whittaker.duckdb.DuckDBGAM`.

Requires the `polars` package (install via `pip install whittaker[polars]`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from whittaker.bam import BigGAM, build_discretized_model_matrix
from whittaker.data import InternalData, _to_array
from whittaker.families.base import Family
from whittaker.fitting.bam import bam_fit
from whittaker.formula.terms import Formula
from whittaker.model_matrix import ModelMatrix


def _import_polars():
    try:
        import polars as pl

        return pl
    except ImportError:  # pragma: no cover - exercised only when polars is not installed
        raise ImportError(
            "Polars is required for PolarsGAM. Install it with: pip install whittaker[polars]"
        ) from None


def _to_lazy(source) -> Any:
    """Convert a source to a Polars LazyFrame."""
    pl = _import_polars()

    if isinstance(source, pl.LazyFrame):
        return source
    if isinstance(source, pl.DataFrame):
        return source.lazy()
    if isinstance(source, (str, Path)):
        path = str(source)
        if path.endswith(".parquet"):
            return pl.scan_parquet(path)
        elif path.endswith(".csv"):
            return pl.scan_csv(path)
        elif path.endswith(".ipc") or path.endswith(".arrow"):
            return pl.scan_ipc(path)
        elif path.endswith(".ndjson") or path.endswith(".jsonl"):
            return pl.scan_ndjson(path)
        else:
            raise ValueError(
                f"Cannot infer file format from extension: {path!r}. "
                "Supported: .parquet, .csv, .ipc, .arrow, .ndjson, .jsonl"
            )
    raise TypeError(
        f"Unsupported source type: {type(source).__name__}. "
        "Pass a Polars LazyFrame, DataFrame, or a file path."
    )


def _lazyframe_to_dict(lf, chunk_size: int) -> InternalData:
    """Collect a LazyFrame into dict[str, NDArray] in chunks."""
    _import_polars()

    df = lf.collect(streaming=True)
    columns: dict[str, list[NDArray]] = {}
    for chunk in df.iter_slices(chunk_size):
        for col in chunk.columns:
            arr = _to_array(chunk[col].to_numpy(allow_copy=True))
            columns.setdefault(col, []).append(arr)
    return {col: np.concatenate(parts) for col, parts in columns.items()}


def _count_lazy(lf) -> int:
    """Count rows in a LazyFrame."""
    pl = _import_polars()
    return lf.select(pl.len()).collect(streaming=True).item()


class PolarsGAM(BigGAM):
    r"""GAM that reads data from Polars LazyFrames, DataFrames, or files.

    `PolarsGAM` extends `~whittaker.bam.BigGAM` with a data-loading layer built on Polars, so it
    can source data from an in-memory Polars `DataFrame`/`LazyFrame` or directly from a file on
    disk (CSV, Parquet, IPC/Arrow, or NDJSON), without requiring the caller to materialize the
    whole dataset into a `dict` of NumPy arrays first. File paths are opened lazily (`pl.scan_*`),
    and the resulting `LazyFrame` is collected using Polars' streaming query engine
    (`collect(streaming=True)`), which evaluates the query plan incrementally rather than loading
    the entire source at once. The collected frame is then walked in `chunk_size`-row slices
    (`iter_slices`) and each column is converted to a NumPy array and concatenated, producing the
    same `dict[str, numpy.ndarray]` that `~whittaker.gam.GAM.fit` and `~whittaker.bam.BigGAM.fit`
    expect. Fitting itself then proceeds exactly as in `BigGAM`: covariates are discretized to at
    most `n_discrete` unique values per smooth, and the design matrix is never materialized.

    Use `PolarsGAM` when the data already lives in Polars, or on disk in a Polars-readable format,
    and you want to avoid a manual load-then-convert step — particularly for datasets in the
    1M-100M row range where Polars' streaming engine keeps peak memory bounded during the read.
    For SQL-native sources or datasets that are more naturally expressed as a database query
    (joins, filters, aggregations), see `~whittaker.duckdb.DuckDBGAM` instead.

    Requires the `polars` package (install via `pip install whittaker[polars]`).

    Parameters
    ----------
    formula : str or Formula
        Model formula as a string (e.g. `"y ~ s(x1) + s(x2) + x3"`), or an already-parsed `Formula`
        object. Same syntax as `~whittaker.gam.GAM`.
    family : Family, optional
        Response distribution family, e.g. `Gaussian()`, `Binomial()`, `Poisson()`, `Gamma()`, or
        `Tweedie()`. Defaults to `Gaussian()`.
    n_discrete : int
        Maximum number of unique representative values per covariate used when discretizing
        smooth terms (see `~whittaker.bam.BigGAM`). Defaults to `200`.
    chunk_size : int
        Number of rows collected per slice when converting the source into NumPy arrays. Smaller
        values reduce peak memory during the Polars-to-NumPy conversion step at the cost of more
        Python-level overhead; larger values reduce overhead but require more memory per slice.
        Defaults to `100_000`.

    Examples
    --------
    ```{python}
    import numpy as np
    import polars as pl
    from whittaker.polars_streaming import PolarsGAM

    rng = np.random.default_rng(0)
    n = 5_000
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x1) + x2**2 + rng.normal(scale=0.2, size=n)

    df = pl.DataFrame({"x1": x1, "x2": x2, "y": y})

    model = PolarsGAM("y ~ s(x1) + s(x2)", n_discrete=100, chunk_size=1_000)
    model.fit(df.lazy())
    print(model.summary())
    ```

    A file path can be passed directly instead of a `DataFrame`/`LazyFrame` — `PolarsGAM` infers
    the format from the extension and scans it lazily:

    ```{python}
    #| eval: false
    model = PolarsGAM("y ~ s(x)")
    model.fit("large_dataset.parquet")
    ```

    Fitting from an actual multi-gigabyte file requires `pip install whittaker[polars]` and enough
    disk I/O bandwidth to stream the file; the in-memory example above is kept small so it runs
    quickly, but the same code path scales to files with tens of millions of rows.
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
        self._n_rows: int = 0

    @property
    def chunk_size(self) -> int:
        """Chunk size used when converting the Polars source to NumPy arrays.

        This is the `chunk_size` value passed to `__init__`: the number of rows per slice that
        `_lazyframe_to_dict` requests from `LazyFrame.iter_slices` while converting the collected
        frame into the `dict[str, numpy.ndarray]` consumed by fitting.

        Returns
        -------
        int
            The configured slice size, in rows.
        """
        return self._chunk_size

    @property
    def n_rows(self) -> int:
        """Total number of rows in the source used by the most recent fit.

        Populated by `fit()` via `_count_lazy`, which evaluates `lf.select(pl.len())` with
        Polars' streaming engine before collecting the data. Remains `0` until `fit()` has been
        called at least once.

        Returns
        -------
        int
            Row count of the `LazyFrame`, `DataFrame`, or file passed to `fit()`.
        """
        return self._n_rows

    def fit(  # type: ignore[override]
        self,
        data,
        *,
        smoothing_params: list[float] | None = None,
        method: str = "fREML",
        select: bool = False,
        **kwargs,
    ) -> PolarsGAM:
        """Fit the GAM from a Polars source.

        Converts `data` to a Polars `LazyFrame`, collects it via Polars' streaming engine in
        `chunk_size`-row slices, and combines the result into `dict[str, numpy.ndarray]` before
        delegating to `~whittaker.bam.BigGAM`'s discretized fitting (see the class docstring for
        details on the discretization and cross-product accumulation).

        Parameters
        ----------
        data : polars.LazyFrame or polars.DataFrame or str or pathlib.Path
            The data source. A Polars `LazyFrame` is used as-is; a `DataFrame` is converted with
            `.lazy()`; a file path (string or `Path`) is scanned lazily based on its extension —
            `.parquet` (`scan_parquet`), `.csv` (`scan_csv`), `.ipc`/`.arrow` (`scan_ipc`), or
            `.ndjson`/`.jsonl` (`scan_ndjson`). Any other type or unrecognized extension raises
            `TypeError` or `ValueError` respectively.
        weights : numpy.ndarray, optional
            Observation (prior) weights, shape `(n,)`. Must be strictly positive.
        smoothing_params : list of float, optional
            Fixed smoothing parameters `lambda_j`, one per smooth term, in formula order. If
            `None` (the default), smoothing parameters are selected automatically according to
            `method`.
        method : str
            Criterion used to select smoothing parameters when `smoothing_params` is `None`. One
            of `"fREML"` (default), `"REML"`, `"ML"`, or `"GCV"`. See `~whittaker.gam.GAM.fit` for
            a description of each criterion.
        select : bool
            If `True`, add an extra penalty on each smooth's null space so the term can be shrunk
            to exactly zero (double-penalty selection). Defaults to `False`.
        **kwargs
            Reserved for future keyword arguments; currently unused.

        Returns
        -------
        PolarsGAM
            Returns `self` for method chaining, e.g. `model = PolarsGAM("y ~ s(x)").fit(data)`.
        """
        lf = _to_lazy(data)
        self._n_rows = _count_lazy(lf)

        col_data = _lazyframe_to_dict(lf, self._chunk_size)
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
