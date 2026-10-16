from typing import Any

from numpy import recarray
from pandas import DataFrame


def dataset_loader(dataset: Any) -> recarray | DataFrame:
    """
    Load a statsmodels dataset.

    Parameters
    ----------
    dataset : module
        A dataset module of ``statsmodels.datasets``, e.g.,
        ``statsmodels.datasets.macrodata``.

    Returns
    -------
    recarray or DataFrame
        The data of the dataset. A DataFrame if the dataset supports the
        ``as_pandas`` argument of ``load`` and a recarray otherwise.
    """
    try:
        return dataset.load(as_pandas=True).data
    except TypeError:
        return dataset.load().data
