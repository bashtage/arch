from typing import Any, Union

from numpy import recarray
from pandas import DataFrame


def dataset_loader(dataset: Any) -> Union[recarray, DataFrame]:
    """Load a dataset using the new syntax is possible"""
    try:
        return dataset.load(as_pandas=True).data
    except TypeError:
        return dataset.load().data
