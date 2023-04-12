from pandas import DataFrame

from arch.data.utility import load_file


def load() -> DataFrame:
    """
    Load the West Texas Intermediate crude oil price data used in the examples

    Returns
    -------
    data : DataFrame
        Data set containing the price of WTI

    Notes
    -----
    From the FRED database
    """
    return load_file(__file__, "wti.csv.gz")
