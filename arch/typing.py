import datetime as dt
from typing import Callable, Hashable, Optional, Tuple, TypeVar, Union

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
]
NDArray = Union[np.ndarray]
RNGType = Callable[[Union[int, Tuple[int, ...]]], NDArray]
ArrayLike1D = Union[NDArray, Series]
ArrayLike2D = Union[NDArray, DataFrame]
ArrayLike = Union[NDArray, DataFrame, Series]
NDArrayOrFrame = TypeVar("NDArrayOrFrame", np.ndarray, DataFrame)
AnyPandas = Union[Series, DataFrame]
DateLike = Union[str, dt.datetime, np.datetime64, Timestamp]
Label = Optional[Hashable]
FloatOrArray = TypeVar("FloatOrArray", float, np.ndarray)
