from pandas import DataFrame

from arch.data.utility import load_file


def load() -> DataFrame:
    """
    Load the AAA and BAA rates used in the examples

    Returns
    -------
    data : DataFrame
        Data set containing the rates on AAA and BAA rated bonds.
    """
    return load_file(__file__, "default.csv.gz")
