import os

import pandas as pd


def load_file(file_base, filename):
    """
    Parameters
    ----------
    filename : str
        Name of csv.gz to load

    Returns
    -------
    data : DataFrame
        Dataframe containing the loaded data
    """
    curr_dir = os.path.split(os.path.abspath(file_base))[0]
    data = pd.read_csv(os.path.join(curr_dir, filename))
    if 'Date' in data:
        data.Date = pd.to_datetime(data.Date)
        data = data.set_index('Date')
    for col in data:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    return data
