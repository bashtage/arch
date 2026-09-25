from arch.univariate.base import format_float_fixed
from arch import arch_model
import pandas as pd
import numpy as np


def test_format_float_fixed():
    out = format_float_fixed(0.0)
    assert out == "0.0000"
    out = format_float_fixed(1.23e-9)
    assert out == "1.2300e-09"
    out = format_float_fixed(123456789.0)
    assert out == "1.2346e+08"


def test_std_resid_numpy_input_returns_series():
    np.random.seed(42)
    data = np.random.randn(500)
    am = arch_model(data, vol="GARCH", p=1, q=1)
    res = am.fit(disp="off")
    sr = res.std_resid
    assert isinstance(sr, pd.Series)
    assert sr.name == "std_resid"
    assert len(sr) == 500


def test_std_resid_pandas_input_still_works():
    np.random.seed(42)
    data = pd.Series(np.random.randn(500), name="returns")
    am = arch_model(data, vol="GARCH", p=1, q=1)
    res = am.fit(disp="off")
    sr = res.std_resid
    assert isinstance(sr, pd.Series)
    assert sr.name == "std_resid"


def test_std_resid_numpy_has_index():
    np.random.seed(42)
    data = np.random.randn(500)
    am = arch_model(data, vol="GARCH", p=1, q=1)
    res = am.fit(disp="off")
    sr = res.std_resid
    assert sr.index is not None
    assert len(sr.index) == 500
