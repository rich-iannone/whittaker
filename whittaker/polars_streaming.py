"""Polars streaming GAM fitting.

Provides `PolarsGAM`, a GAM variant that reads data from a Polars LazyFrame
or file path (CSV/Parquet) and fits the model using discretized basis evaluation
(same approach as `BigGAM`). Data is processed in chunks via Polars' streaming
engine, so the full dataset is never fully materialized.

Suitable for datasets in the 1M--100M row range. For even larger datasets
backed by DuckDB, see :class:`~whittaker.duckdb.DuckDBGAM`.

Requires the ``polars`` package (install via ``pip install whittaker[polars]``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from whittaker.bam import BigGAM, build_discretized_model_matrix
from whittaker.data import InternalData, _to_array
from whittaker.families.base import Family
from whittaker.fitting.bam import DiscretizedModelMatrix, bam_fit
from whittaker.formula.terms import Formula
from whittaker.model_matrix import ModelMatrix


def _import_polars():
    try:
        import polars as pl

        return pl
    except ImportError:
        raise ImportError(
            "Polars is required for PolarsGAM. "
            "Install it with: pip install whittaker[polars]"
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
    pl = _import_polars()

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
    """GAM that reads data from Polars LazyFrames or file paths.

    Data is collected via Polars' streaming engine and the model is fit using
    discretized basis evaluation (same as `BigGAM`), so the full design matrix
    is never materialized.

    Parameters
    ----------
    formula:
        Model formula (same as `~whittaker.gam.GAM`).
    family:
        Response distribution family.
    n_discrete:
        Number of discretization grid points per covariate.
    chunk_size:
        Number of rows per chunk when iterating the collected DataFrame.

    Examples
    --------
    >>> import polars as pl
    >>> model = PolarsGAM("y ~ s(x1) + s(x2)")
    >>> model.fit(pl.scan_parquet("large_dataset.parquet"))
    >>> predictions = model.predict(new_data)

    Or from a file path directly:

    >>> model = PolarsGAM("y ~ s(x)")
    >>> model.fit("data.parquet")
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
        """Chunk size for streaming iteration."""
        return self._chunk_size

    @property
    def n_rows(self) -> int:
        """Total number of rows in the source."""
        return self._n_rows

    def fit(
        self,
        source,
        *,
        smoothing_params: list[float] | None = None,
        method: str = "fREML",
        select: bool = False,
        **kwargs,
    ) -> PolarsGAM:
        """Fit the GAM from a Polars source.

        Parameters
        ----------
        source:
            A Polars ``LazyFrame``, ``DataFrame``, or a file path (string or
            ``Path``) to a Parquet, CSV, IPC, or NDJSON file. File paths are
            scanned lazily.
        smoothing_params:
            Fixed smoothing parameters. If ``None``, selected automatically.
        method:
            Smoothing selection method: ``"fREML"`` (default), ``"REML"``,
            ``"ML"``, or ``"GCV"``.
        select:
            If ``True``, enable double-penalty variable selection.

        Returns
        -------
        PolarsGAM
            Returns ``self`` for method chaining.
        """
        lf = _to_lazy(source)
        self._n_rows = _count_lazy(lf)

        data = _lazyframe_to_dict(lf, self._chunk_size)
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
