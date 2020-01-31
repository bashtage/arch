from pandas import DataFrame

from arch.data.utility import load_file


def load() -> DataFrame:
    """
    Load the graduate school admissions data used in the examples

    Returns
    -------
    data : DataFrame
        Dataset containing GRE, GPA and class rank, and admission decision
    """
    return load_file(__file__, "binary.csv.gz")
