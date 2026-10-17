r"""
Table-driven tests of the input validation of the public API.

Each area adds its own ``test_validation_<area>.py`` and builds rows from this module,
so that no shared test file changes.  A row is a call that must raise (or must not
raise) and the message it must raise with::

    from arch.tests.validation_table import (
        INT_BAD,
        Row,
        bad_value_rows,
        run_row,
        to_params,
    )

    BASE = {"y": Y}
    ROWS = [
        *bad_value_rows("ADF", lambda **kw: ADF(**kw), BASE, "lags", INT_BAD),
        Row("ADF", lambda **kw: ADF(**kw), {**BASE, "trend": "x"}, ValueError,
            r"trend must be one of 'n', 'c', 'ct' or 'ctt'; got 'x' \(str\)"),
    ]


    @pytest.mark.parametrize("row", to_params(ROWS))
    def test_validation(row):
        run_row(row)

The message of every row also has to follow the common form, which `run_row` checks:
``<name> must be <rule>; got <value> (<type>)``.  Use ``form=False`` for a message that
cannot follow it, for example an error raised by a lower level routine.

The ``*_BAD`` lists are ``(value, exception)`` pairs and the ``*_GOOD`` lists are
values that must be accepted, for the policy of decisions 63 and 66-68. A flag accepts
``bool``, ``np.bool_`` and the integers 0 and 1 (decision 66). To change the policy,
change the lists here (and the helpers in ``arch/utility/_validation.py``); the tests of
all areas follow.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import re
from typing import Any

import numpy as np
import pytest

__all__ = [
    "BOOL_BAD",
    "BOOL_GOOD",
    "FLOAT_BAD",
    "FLOAT_GOOD",
    "INT_BAD",
    "INT_GOOD",
    "SEED_BAD",
    "SEED_GOOD",
    "Row",
    "bad_value_rows",
    "good_value_rows",
    "ids_for",
    "message_form",
    "run_row",
    "to_params",
]

# Values that must be rejected: (value, exception). None is a bad value for a
# parameter that does not accept None; remove it from the list for one that does.
BOOL_BAD: list[tuple[Any, type[Exception]]] = [
    (2, ValueError),
    (-1, ValueError),
    (np.int64(2), ValueError),
    (np.uint8(255), ValueError),
    ("no", TypeError),
    ("", TypeError),
    (None, TypeError),
    (np.array([True]), TypeError),
    (np.array(1), TypeError),
    (1.0, TypeError),
    (0.0, TypeError),
    (np.float64(1.0), TypeError),
]
INT_BAD: list[tuple[Any, type[Exception]]] = [
    (2.5, TypeError),
    (5.0, TypeError),
    (np.float64(3.0), TypeError),
    (True, TypeError),
    (np.True_, TypeError),
    ("2", TypeError),
    (None, TypeError),
    (np.nan, TypeError),
    (np.inf, TypeError),
    ([2], TypeError),
]
FLOAT_BAD: list[tuple[Any, type[Exception]]] = [
    ("0.5", TypeError),
    (True, TypeError),
    (np.True_, TypeError),
    (None, TypeError),
    (1j, TypeError),
    ([0.5], TypeError),
    (np.nan, ValueError),
    (np.inf, ValueError),
    (-np.inf, ValueError),
]
SEED_BAD: list[tuple[Any, type[Exception]]] = [
    (-1, ValueError),
    (np.int64(-5), ValueError),
    (2.5, TypeError),
    (3.0, TypeError),
    (True, TypeError),
    ("abc", TypeError),
    ([1, 2], TypeError),
]

# Valid values: Python and NumPy scalars are accepted wherever the Python type is
BOOL_GOOD: list[Any] = [
    True,
    False,
    np.True_,
    np.False_,
    0,
    1,
    np.int8(0),
    np.int64(1),
    np.uint16(1),
]
INT_GOOD: list[Any] = [3, np.int8(3), np.int32(3), np.int64(3), np.uint16(3)]
FLOAT_GOOD: list[Any] = [
    0.5,
    np.float16(0.5),
    np.float32(0.5),
    np.float64(0.5),
    Fraction(1, 2),
]
SEED_GOOD: list[Any] = [
    0,
    12345,
    np.int64(7),
    np.random.default_rng(0),
    np.random.RandomState(0),
]

_FORM = r"^{name} must be .+; got .+ \(\w+\)$"


def message_form(name: str) -> re.Pattern[str]:
    """
    Return the pattern of a message in the common form for an argument.

    Parameters
    ----------
    name : str
        The name of the argument.

    Returns
    -------
    re.Pattern
        A pattern that matches ``<name> must be <rule>; got <value> (<type>)``.
    """
    return re.compile(_FORM.format(name=re.escape(name)), re.DOTALL)


@dataclass(frozen=True)
class Row:
    r"""
    A call and what it must do.

    Parameters
    ----------
    target : str
        A short name of what is called, for the test id.
    func : callable
        Called as ``func(**kwargs)``. Use a lambda to call a method of an object that
        is constructed in the call, or to run a computation such as ``fit``.
    kwargs : mapping
        The keyword arguments. They are copied before the call.
    exc : type of Exception or None
        The exception that must be raised, or None if the call must succeed.
    match : str, optional
        A regular expression searched in the message, for example
        ``r"lags must be an integer >= 0; got 2.5 \(float\)"``.
    id : str, optional
        The id of the test. The default is built from the target and the arguments
        that differ from the first row of the target, which is not always unique, so
        give it for rows that would collide.
    form : bool, optional
        Whether the message must be in the common form for ``name``. The default is
        to check it for all rows that raise and that have a ``name``.
    name : str, optional
        The argument that is invalid. It is the name used for the common form.
    """

    target: str
    func: Callable[..., Any]
    kwargs: Mapping[str, Any]
    exc: type[Exception] | None
    match: str | None = None
    id: str | None = None
    form: bool = True
    name: str | None = None


def _label(value: Any) -> str:
    """A label that is the same in every process, so no memory addresses (xdist)."""
    if isinstance(value, np.generic):
        text = f"np.{type(value).__name__}({value})"
    elif isinstance(value, np.ndarray):
        text = f"array{value.shape}"
    elif value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        text = repr(value)
    elif (
        isinstance(value, (list, tuple))
        and len(value) <= 3
        and all(v is None or isinstance(v, (bool, int, float, str)) for v in value)
    ):
        text = repr(value)
    else:
        text = type(value).__name__
    text = text.replace(" ", "")
    return text if len(text) <= 24 else text[:21] + "..."


def ids_for(values: Iterable[Any]) -> list[str]:
    """
    Build unique test ids for a list of values or of (value, exception) pairs.

    pytest 9 requires unique ids, and ``2``, ``np.int64(2)`` and ``2.0`` would all
    render as ``2``.

    Parameters
    ----------
    values : iterable
        The parameter values, or ``(value, exception)`` pairs like `INT_BAD`.

    Returns
    -------
    list of str
        One id per value, starting with its position.
    """
    ids = []
    for i, item in enumerate(values):
        is_pair = (
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[1], type)
            and issubclass(item[1], Exception)
        )
        ids.append(f"{i}-{_label(item[0] if is_pair else item)}")
    return ids


def bad_value_rows(
    target: str,
    func: Callable[..., Any],
    base: Mapping[str, Any],
    name: str,
    bad: Iterable[tuple[Any, type[Exception]]],
    *,
    rule: str = r".+",
) -> list[Row]:
    """
    Build the rows that set one argument to each bad value.

    Parameters
    ----------
    target : str
        A short name of what is called, for the test id.
    func : callable
        Called as ``func(**kwargs)``.
    base : mapping
        Valid keyword arguments.
    name : str
        The argument to replace.
    bad : iterable of tuple
        The ``(value, exception)`` pairs, for example `INT_BAD`.
    rule : str, optional
        A regular expression for the rule in the message, between ``must be`` and
        ``; got``. The default accepts any rule.

    Returns
    -------
    list of Row
        One row per bad value.
    """
    rows = []
    for i, (value, exc) in enumerate(bad):
        match = rf"^{re.escape(name)} must be {rule}; got "
        rows.append(
            Row(
                target,
                func,
                {**base, name: value},
                exc,
                match,
                id=f"{target}-{name}-{i}-{_label(value)}",
                name=name,
            )
        )
    return rows


def good_value_rows(
    target: str,
    func: Callable[..., Any],
    base: Mapping[str, Any],
    name: str,
    good: Iterable[Any],
) -> list[Row]:
    """
    Build the rows that set one argument to each value that must be accepted.

    Parameters
    ----------
    target : str
        A short name of what is called, for the test id.
    func : callable
        Called as ``func(**kwargs)``.
    base : mapping
        Valid keyword arguments.
    name : str
        The argument to replace.
    good : iterable
        The values, for example `INT_GOOD`.

    Returns
    -------
    list of Row
        One row per value, with no exception.
    """
    return [
        Row(
            target,
            func,
            {**base, name: value},
            None,
            id=f"{target}-{name}-{i}-{_label(value)}-ok",
            name=name,
        )
        for i, value in enumerate(good)
    ]


def to_params(rows: Sequence[Row]) -> list[Any]:
    """
    Convert rows to pytest parameters with unique ids.

    Parameters
    ----------
    rows : sequence of Row
        The rows.

    Returns
    -------
    list
        ``pytest.param`` objects, to use with ``parametrize("row", ...)``.

    Raises
    ------
    ValueError
        If two rows have the same id (pytest 9 requires unique ids).
    """
    params = []
    seen: set[str] = set()
    for i, row in enumerate(rows):
        row_id = row.id or f"{row.target}-{i}"
        if row_id in seen:
            raise ValueError(f"duplicate row id {row_id}")
        seen.add(row_id)
        params.append(pytest.param(row, id=row_id))
    return params


def run_row(row: Row) -> None:
    """
    Run a row.

    Parameters
    ----------
    row : Row
        The call to run.

    Raises
    ------
    AssertionError
        If the call does not do what the row says, or the message is not in the common
        form.
    """
    kwargs = dict(row.kwargs)
    if row.exc is None:
        row.func(**kwargs)
        return
    with pytest.raises(row.exc, match=row.match) as info:
        row.func(**kwargs)
    if row.form and row.name is not None:
        pattern = message_form(row.name)
        assert pattern.match(str(info.value)), str(info.value)
