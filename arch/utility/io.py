import numpy as np

__all__ = ["str_format", "pval_format"]


def str_format(v: float) -> str:
    """Preferred basic formatter"""
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
        d = int(4)

    format_str = "{0:" + f"0.{d}f" + "}"
    return format_str.format(v)


def pval_format(v: float) -> str:
    """Preferred formatting for x in [0,1]"""
    if np.isnan(v):
        return "        "
    return f"{v:4.4f}"
