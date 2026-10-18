import numpy as np

from arch.utility._validation import check_float

__all__ = ["pval_format", "str_format"]


def str_format(v: float) -> str:
    """
    Format a number using the preferred basic formatter.

    Parameters
    ----------
    v : float
        The value to format. NaN is formatted as spaces and inf as ``inf``.

    Returns
    -------
    str
        The formatted value, or 8 spaces if ``v`` is NaN.

    Raises
    ------
    TypeError
        If v is not a real number.
    """
    checked = check_float("v", v, allow_nan=True, allow_inf=True)
    if not isinstance(v, np.floating):
        v = checked
    if np.isnan(v):
        return "        "
    if np.isinf(v):
        return f"{v:8.4g}"
    av = abs(v)
    digits = 0
    if av != 0:
        digits = int(np.ceil(np.log10(av)))
    if digits > 4 or digits <= -4:
        return f"{v:8.4g}"

    if digits > 0:
        d = int(5 - digits)
    else:
        d = 4

    format_str = "{0:" + f"0.{d}f" + "}"
    return format_str.format(v)


def pval_format(v: float) -> str:
    """
    Format a number in [0, 1] using the preferred formatter.

    Parameters
    ----------
    v : float
        The value to format, between 0 and 1. NaN is formatted as spaces.

    Returns
    -------
    str
        The value formatted with 4 decimal places, or 8 spaces if ``v`` is NaN.

    Raises
    ------
    TypeError
        If v is not a real number.
    ValueError
        If v is inf or is not in [0, 1].
    """
    checked = check_float("v", v, lower=0.0, upper=1.0, allow_nan=True)
    if not isinstance(v, np.floating):
        v = checked
    if np.isnan(v):
        return "        "
    return f"{v:4.4f}"
