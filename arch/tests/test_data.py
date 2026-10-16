import glob
import os

import pandas as pd
import pytest

from arch.data import frenchdata

CURR_DIR = os.path.split(os.path.abspath(__file__))[0]
FILES = glob.glob(os.path.join(CURR_DIR, "..", "data", "*"))
DATASETS = [os.path.split(f)[-1] for f in FILES if (".py" not in f and "__" not in f)]


@pytest.fixture(params=DATASETS)
def dataset(request):
    return request.param


def test_dataset(dataset):
    mod = __import__(f"arch.data.{dataset}", fromlist=[dataset])
    data = mod.load()
    assert isinstance(data, pd.DataFrame)
    if isinstance(data.index, pd.DatetimeIndex):
        assert data.index.min().year >= 1900
        assert data.index.max().year <= 2100
        assert data.index.is_monotonic_increasing


def test_frenchdata_monthly_dates():
    data = frenchdata.load()
    assert data.index[0] == pd.Timestamp("1926-07-01")
    assert data.index[1] == pd.Timestamp("1926-08-01")
    assert (data.index.day == 1).all()
    months = data.index.year * 12 + data.index.month
    assert (months[1:] - months[:-1] == 1).all()
