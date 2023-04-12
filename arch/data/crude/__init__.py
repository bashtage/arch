from pandas import DataFrame

from arch.data.utility import load_file


def load() -> DataFrame:
    """
    Load the Core CPI data used in the examples

    Returns
    -------
    data : DataFrame
        Data set containing the CPI less Food and Energy

    Notes
    -----
    From the FRED database
    """
    return load_file(__file__, "crude.csv.gz")
