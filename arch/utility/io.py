import numpy as np

__all__ = ["pval_format", "str_format"]


def str_format(v: float) -> str:
    """
    Format a number using the preferred basic formatter.

    Parameters
    ----------
    v : float
        The value to format.

    Returns
    -------
    str
        The formatted value, or 8 spaces if ``v`` is NaN.
    """
    if np.isnan(v):
        return "        "
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
        The value to format.

    Returns
    -------
    str
        The value formatted with 4 decimal places, or 8 spaces if ``v`` is NaN.
    """
    if np.isnan(v):
        return "        "
    return f"{v:4.4f}"
