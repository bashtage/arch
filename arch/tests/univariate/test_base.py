from arch.univariate.base import format_float_fixed


def test_format_float_fixed():
    out = format_float_fixed(0.0)
    assert out == "0.0000"
    out = format_float_fixed(1.23e-9)
    assert out == "1.2300e-09"
    out = format_float_fixed(123456789.0)
    assert out == "1.2346e+08"
