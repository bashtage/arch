import numpy as np
import pytest

from arch.utility.io import pval_format, str_format

CASES = (
    (np.nan, "        "),
    (1, "1.0000"),
    (1.234567890e10, "1.235e+10"),
    (12345678900, "1.235e+10"),
    (123, "123.00"),
    (0.000006789, "6.789e-06"),
)


@pytest.mark.parametrize("case", CASES)
def test_str_format(case):
    assert str_format(case[0]) == case[1]


PVAL_CASES = ((np.nan, "        "), (1e-37, "0.0000"), (0.999999, "1.0000"))


@pytest.mark.parametrize("case", PVAL_CASES)
def test_pval_format(case):
    assert pval_format(case[0]) == case[1]
