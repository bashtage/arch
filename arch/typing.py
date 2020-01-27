from typing import Union

import numpy as np
from pandas import DataFrame, Series

NDArray = Union[np.ndarray]
ArrayLike = Union[NDArray, DataFrame, Series]
