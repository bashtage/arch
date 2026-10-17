"""
Validation of options that take one of a fixed set of strings.

Option names are not case sensitive. Every function and class that accepts an
option such as a covariance type, a trend or a kernel compares the lower case
value to a table of lower case names, and raises a ``ValueError`` that names the
option and lists the valid values when the value is not in the table.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

__all__ = ["check_option"]


def _format_choices(choices: Sequence[str]) -> str:
    quoted = [f"'{choice}'" for choice in choices]
    if len(quoted) == 1:
        return quoted[0]
    return ", ".join(quoted[:-1]) + " or " + quoted[-1]


def check_option(
    value: Any,
    name: str,
    valid: Iterable[str] | Mapping[str, str],
    *,
    display: Sequence[str] | None = None,
    normalize: Callable[[str], str] | None = None,
) -> str:
    """
    Compare an option to its valid values without regard to case.

    Parameters
    ----------
    value : object
        The value passed by the user. Anything that is not a string is invalid.
    name : str
        The name of the parameter, which is used in the error message.
    valid : {iterable[str], mapping[str, str]}
        The lower case valid values. A mapping maps each lower case spelling,
        including any alias, to the canonical value that is returned. The
        canonical value may use any case, for example "R" or "Za".
    display : sequence[str], optional
        The values listed in the error message. The default is the lower case
        spellings in ``valid``.
    normalize : callable, optional
        A function applied to the lower case value before it is looked up, for
        example to ignore spaces and hyphens in the names of kernels.

    Returns
    -------
    str
        The canonical value. This is the lower case value when ``valid`` is not
        a mapping.

    Raises
    ------
    ValueError
        If the value is not a string or is not one of the valid values.
    """
    table = dict(valid) if isinstance(valid, Mapping) else {v: v for v in valid}
    if isinstance(value, str):
        key = value.lower()
        if normalize is not None:
            key = normalize(key)
        if key in table:
            return table[key]
    choices = list(table) if display is None else list(display)
    raise ValueError(
        f"{name} must be one of {_format_choices(choices)}. The comparison is "
        f"not case sensitive. Got {value!r}."
    )
