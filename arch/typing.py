from collections.abc import Hashable
import datetime as dt
from typing import Any, Callable, Literal, Optional, TypeVar, Union

import numpy as np
from pandas import DataFrame, Series, Timestamp

__all__ = [
    "NDArray",
    "ArrayLike",
    "NDArrayOrFrame",
    "AnyPandas",
    "DateLike",
    "ArrayLike1D",
    "ArrayLike2D",
    "Label",
    "FloatOrArray",
    "RNGType",
    "Literal",
    "UnitRootTrend",
    "ForecastingMethod",
    "Float64Array",
    "Int64Array",
    "Int32Array",
    "BoolArray",
    "AnyArray",
    "IntArray",
    "RandomStateState",
    "Uint32Array",
    "BootstrapIndexT",
]

NDArray = Union[np.ndarray]
Float64Array = np.ndarray[Any, np.dtype[np.double]]  # pragma: no cover
Int64Array = np.ndarray[Any, np.dtype[np.longlong]]  # pragma: no cover
Int32Array = np.ndarray[Any, np.dtype[np.intc]]  # pragma: no cover
IntArray = np.ndarray[Any, np.dtype[np.int_]]  # pragma: no cover
BoolArray = np.ndarray[Any, np.dtype[np.bool_]]  # pragma: no cover
AnyArray = np.ndarray[Any, Any]  # pragma: no cover
Uint32Array = np.ndarray[Any, np.dtype[np.uintc]]  # pragma: no cover

BootstrapIndexT = Union[
    Int64Array, tuple[Int64Array, ...], tuple[list[Int64Array], dict[str, Int64Array]]
]
RandomStateState = tuple[str, Uint32Array, int, int, float]

RNGType = Callable[[Union[int, tuple[int, ...]]], Float64Array]
ArrayLike1D = Union[NDArray, Series]
ArrayLike2D = Union[NDArray, DataFrame]
ArrayLike = Union[NDArray, DataFrame, Series]
NDArrayOrFrame = TypeVar("NDArrayOrFrame", Float64Array, DataFrame)
AnyPandas = Union[Series, DataFrame]
DateLike = Union[str, dt.datetime, np.datetime64, Timestamp]
Label = Optional[Hashable]
FloatOrArray = TypeVar("FloatOrArray", float, np.ndarray)
UnitRootTrend = Literal["n", "c", "ct", "ctt"]
ForecastingMethod = Literal["analytic", "simulation", "bootstrap"]
