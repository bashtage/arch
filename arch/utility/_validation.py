"""
Validation of the arguments of the public API.

The helpers take the name of the argument first, so that the error messages name
the argument, then the value that was received and then what is accepted.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

__all__ = ["check_option"]

_MAX_REPR = 40


def _short_repr(value: Any) -> str:
    """Return the repr of a value, shortened to keep error messages readable."""
    text = repr(value)
    if len(text) > _MAX_REPR:
        text = text[: _MAX_REPR - 3] + "..."
    return text


def _format_choices(choices: Sequence[str]) -> str:
    quoted = [f"'{choice}'" for choice in choices]
    if len(quoted) == 1:
        return quoted[0]
    return ", ".join(quoted[:-1]) + " or " + quoted[-1]


def check_option(
    name: str,
    value: Any,
    allowed: Iterable[str] | Mapping[str, str],
    *,
    display: Sequence[str] | None = None,
    normalize: Callable[[str], str] | None = None,
) -> str:
    """
    Compare an option to its allowed values without regard to case.

    Parameters
    ----------
    name : str
        The name of the argument, which is used in the error message.
    value : object
        The value passed by the user. It must be a string.
    allowed : iterable of str or mapping of str to str
        The lower case allowed values. A mapping maps each lower case spelling,
        including any alias, to the canonical value that is returned. The
        canonical value may use any case, for example "R" or "Za".
    display : sequence[str], optional
        The values listed in the error message. The default is the lower case
        spellings in ``allowed``.
    normalize : callable, optional
        A function applied to the lower case value before it is looked up, for
        example to ignore spaces and hyphens in the names of kernels.

    Returns
    -------
    str
        The canonical value. This is the lower case value when ``allowed`` is
        not a mapping.

    Raises
    ------
    TypeError
        If the value is not a string.
    ValueError
        If the value is a string that is not one of the allowed values.
    """
    table = dict(allowed) if isinstance(allowed, Mapping) else {a: a for a in allowed}
    if isinstance(value, str):
        key = value.lower()
        if normalize is not None:
            key = normalize(key)
        if key in table:
            return table[key]
    choices = list(table) if display is None else list(display)
    message = (
        f"{name} must be one of {_format_choices(choices)}; got {_short_repr(value)}"
    )
    if not isinstance(value, str):
        raise TypeError(message)
    raise ValueError(message)
