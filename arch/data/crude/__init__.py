from pandas import DataFrame

from arch.data.utility import load_file


def load() -> DataFrame:
    """
    Load the Brent and WTI crude oil price data used in the examples

    Returns
    -------
    data : DataFrame
        Data set containing the daily spot prices of Brent and WTI crude oil

    Notes
    -----
    From the FRED database
    """
    return load_file(__file__, "crude.csv.gz")
