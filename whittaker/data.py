"""Data input layer: convert user-supplied data to internal dict representation."""

from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np
from numpy.typing import NDArray

InternalData = dict[str, NDArray]
InputData = Any


def prepare_data(data: InputData) -> InternalData:
    """Convert user-supplied data to `dict[str, NDArray]`.

    Accepts:

    - `dict[str, array-like]`: passed through with each value cast to a float64 array.
    - Any DataFrame supported by Narwhals (Polars, Pandas, cuDF, Modin, PyArrow): columns are
    extracted as NumPy arrays.

    Parameters
    ----------
    data:
        Column-oriented data. Either a `dict` mapping column names to array-like values, or a
        DataFrame supported by Narwhals.

    Returns
    -------
    dict[str, NDArray]
        Internal column-oriented representation with float64 arrays.
    """
    if isinstance(data, dict):
        return {k: _to_array(v) for k, v in data.items()}

    try:
        df = nw.from_native(data)
    except TypeError:
        raise TypeError(
            f"Unsupported data type: {type(data).__name__}. "
            "Pass a dict[str, array], Polars DataFrame, Pandas DataFrame, "
            "or any other Narwhals-supported DataFrame."
        ) from None

    return {col: _to_array(df[col].to_numpy()) for col in df.columns}


def _to_array(v: Any) -> NDArray:
    arr = np.asarray(v)
    if arr.dtype.kind in ("U", "S", "O"):
        return arr
    return arr.astype(float)
