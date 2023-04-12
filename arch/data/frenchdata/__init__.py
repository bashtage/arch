from pandas import DataFrame

from arch.data.utility import load_file


def load() -> DataFrame:
    """
    Load the Fama-French factor data used in the examples

    Returns
    -------
    data : DataFrame
        Data set containing excess market, size and value factors and the
        risk-free rate

    Notes
    -----
    Provided by Ken French,
    http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
    """
    return load_file(__file__, "frenchdata.csv.gz")
