from distutils.version import LooseVersion
from typing import Any, Union

from numpy import recarray
from pandas import DataFrame
import statsmodels
from statsmodels.tsa import tsatools

from arch.typing import ArrayLike

SM_LT_011 = LooseVersion(statsmodels.__version__) < LooseVersion("0.11")


def dataset_loader(dataset: Any) -> Union[recarray, DataFrame]:
    """Load a dataset using the new syntax is possible"""
    try:
        return dataset.load(as_pandas=True).data
    except TypeError:
        return dataset.load().data


def add_trend(
    x: ArrayLike, trend: str = "c", prepend: bool = False, has_constant: str = "skip"
) -> ArrayLike:
    if trend in ("n", "nc"):
        return x
    return tsatools.add_trend(
        x, trend=trend, prepend=prepend, has_constant=has_constant
    )
