from __future__ import annotations

import datetime as dt
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Hashable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
from pandas import DataFrame, Series, Timestamp

NP_GTE_121 = np.lib.NumpyVersion(np.__version__) >= np.lib.NumpyVersion("1.21.0")

if sys.version_info >= (3, 8):
    from typing import Literal
elif TYPE_CHECKING:
    from typing_extensions import Literal
else:

    class _Literal:
        def __getitem__(self, item):
            pass

    Literal = _Literal()

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
if NP_GTE_121 and TYPE_CHECKING:
    Float64Array = np.ndarray[Any, np.dtype[np.float64]]
    Int64Array = np.ndarray[Any, np.dtype[np.int64]]
    Int32Array = np.ndarray[Any, np.dtype[np.int32]]
    IntArray = np.ndarray[Any, np.dtype[np.int_]]
    BoolArray = np.ndarray[Any, np.dtype[np.bool_]]
    AnyArray = np.ndarray[Any, Any]
    Uint32Array = np.ndarray[Any, np.dtype[np.uint32]]
else:
    Uint32Array = (
        IntArray
    ) = Float64Array = Int64Array = Int32Array = BoolArray = AnyArray = NDArray

BootstrapIndexT = Union[
    Int64Array, Tuple[Int64Array, ...], Tuple[List[Int64Array], Dict[str, Int64Array]]
]
RandomStateState = Tuple[str, Uint32Array, int, int, float]

RNGType = Callable[[Union[int, Tuple[int, ...]]], Float64Array]
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
