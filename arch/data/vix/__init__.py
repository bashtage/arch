from pandas import DataFrame

from arch.data.utility import load_file


def load() -> DataFrame:
    """
    Load the VIX Index data used in the examples

    Returns
    -------
    data : DataFrame
        Data set containing historical VIX
    """
    return load_file(__file__, "vix.csv.gz")
