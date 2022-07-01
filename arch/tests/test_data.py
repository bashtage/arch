import glob
import os

import pandas as pd
import pytest

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
