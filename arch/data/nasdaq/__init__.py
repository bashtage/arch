from arch.data.utility import load_file


def load():
    """
    Load the NASDAQ Composite data used in the examples

    Returns
    -------
    data : DataFrame
        Data set containing OHLC, adjusted close and the trading volume.
    """
    return load_file(__file__, 'nasdaq.csv.gz')
