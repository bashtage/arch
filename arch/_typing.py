from collections.abc import Callable, Hashable
import datetime as dt
from typing import Any, Literal, Optional, TypeVar, Union

import numpy as np
from pandas import DataFrame, Series, Timestamp

__all__ = [
    "AnyArray",
    "AnyArray1D",
    "AnyPandas",
    "ArrayLike",
    "ArrayLike1D",
    "ArrayLike2D",
    "BoolArray",
    "BootstrapIndexT",
    "DateLike",
    "Float64Array",
    "Float64Array1D",
    "Float64Array2D",
    "FloatOrArray",
    "ForecastingMethod",
    "Int32Array",
    "Int64Array",
    "Int64Array2D",
    "IntArray",
    "Label",
    "Literal",
    "NDArray",
    "NDArrayOrFrame",
    "RNGType",
    "RandomStateState",
    "Uint32Array",
    "UnitRootTrend",
]

NDArray = Union[np.ndarray]
Float64Array = np.ndarray[tuple[int, ...], np.dtype[np.float64]]  # pragma: no cover
Float64Array1D = np.ndarray[tuple[int], np.dtype[np.float64]]  # pragma: no cover
Float64Array2D = np.ndarray[tuple[int, int], np.dtype[np.float64]]  # pragma: no cover
Int64Array = np.ndarray[tuple[int, ...], np.dtype[np.int64]]  # pragma: no cover
Int64Array1D = np.ndarray[tuple[int], np.dtype[np.int64]]  # pragma: no cover
Int64Array2D = np.ndarray[tuple[int, int], np.dtype[np.int64]]  # pragma: no cover
Int32Array = np.ndarray[tuple[int, ...], np.dtype[np.intc]]  # pragma: no cover
IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int64]]  # pragma: no cover
BoolArray = np.ndarray[tuple[int, ...], np.dtype[np.bool_]]  # pragma: no cover
AnyArray = np.ndarray[tuple[int, ...], Any]  # pragma: no cover
AnyArray1D = np.ndarray[tuple[int], Any]  # pragma: no cover
Uint32Array = np.ndarray[tuple[int, ...], np.dtype[np.uintc]]  # pragma: no cover

BootstrapIndexT = Union[
    Int64Array1D,
    tuple[Int64Array1D, ...],
    tuple[list[Int64Array1D], dict[str, Int64Array1D]],
]
RandomStateState = tuple[str, Uint32Array, int, int, float]

RNGType = Callable[[Union[int, tuple[int, ...]]], Float64Array]
ArrayLike1D = Union[Float64Array1D, Series]
ArrayLike2D = Union[Float64Array2D, DataFrame]
ArrayLike = Union[NDArray, DataFrame, Series]
NDArrayOrFrame = TypeVar("NDArrayOrFrame", Float64Array, DataFrame)
AnyPandas = Union[Series, DataFrame]
DateLike = Union[str, dt.datetime, np.datetime64, Timestamp]
Label = Optional[Hashable]
FloatOrArray = TypeVar("FloatOrArray", float, np.ndarray)
UnitRootTrend = Literal["n", "c", "ct", "ctt"]
ForecastingMethod = Literal["analytic", "simulation", "bootstrap"]
