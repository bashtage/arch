Contributing
============

Docstrings
----------

Public classes, methods, properties and functions have docstrings in the
`numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html>`_ format. These ten
rules apply to all of them. Each rule shows a good and a bad form of a parameter type or
of a line of a docstring.

1. **Optional parameters.** A parameter with a default has ``, optional`` in its type.
   The default value goes in the description only when the signature does not make it
   obvious. A parameter whose default cannot be used, because the value has to be
   provided, is still ``optional``, and its description says that it must be provided.

   - Good: ``reps : int, optional``, and ``adf : ADF, optional`` described as "The ADF
     instance used to perform the test. It must be provided."
   - Bad: ``reps : int, default 1000`` or ``reps : int`` for ``reps=1000``, and
     ``adf : ADF`` for ``adf=None``

2. **Unions.** Write a union of types with ``or``. A union of three or more types is
   written "A, B or C", with no comma before ``or``. Braces are only for a set of
   literal values.

   - Good: ``x : ndarray or Series``, ``seed : int, Generator or RandomState`` and
     ``method : {"aic", "bic"}``
   - Bad: ``x : {ndarray, Series}`` and ``seed : int, Generator, or RandomState``

3. **No subscripted types.** Spell out the contents.

   - Good: ``list of str`` and ``dict of str to DataFrame``
   - Bad: ``list[str]``, ``list (str)`` and ``dict[str, DataFrame]``

4. **Arrays.** ``array_like`` is input that is converted with ``np.asarray``.
   ``ndarray`` is input that must already be an array.

   - Good: ``y : array_like`` when the code starts with ``y = np.asarray(y)``
   - Bad: ``y : ndarray`` for the same code, or ``array-like`` and ``ArrayLike``

5. **Basic types.** Use ``bool``, ``int``, ``float``, ``str`` and ``callable``.
   Document ``*args`` and ``**kwargs`` with their stars, and list the keys that
   ``**kwargs`` accepts.

   - Good: ``**kwargs : dict, optional`` (``*args : tuple, optional``)
   - Bad: ``kwargs : dict``, ``flag : boolean``, ``name : string`` and
     ``func : function``

6. **Sections.** Use the numpydoc section names in the numpydoc order: Parameters,
   Returns (or Yields), Other Parameters, Raises, Warns, See Also, Notes, References,
   Examples. The names are case sensitive.

   - Good: ``See Also`` before ``Notes``, and ``Examples`` last
   - Bad: ``See also``, or ``Examples`` before ``Notes``

7. **Returns.** Give the concrete type of each returned value. A named entry is allowed.

   - Good: ``ndarray`` or ``names : list of str``
   - Bad: ``array`` or ``the fitted values`` as the type

8. **Complete parameters.** Document every parameter in the signature, in signature
   order, and document nothing else. Public properties have a docstring with a summary
   line.

   - Good: ``fit(self, reps, method)`` documents ``reps`` and then ``method``
   - Bad: a missing parameter, one in a different order, ``resids `` with a trailing
     space, or ``method: str`` without a space before the colon

9. **Punctuation.** The summary is one line. The summary of a function or a method
   starts with a verb in the imperative mood, and the summary of a class or a property
   is a noun phrase. The summary, the parameter descriptions and the return descriptions
   start with a capital letter and end with a period.

   - Good: ``Compute the log-likelihood.`` for a method and ``Augmented Dickey-Fuller
     unit root test.`` for a class
   - Bad: ``Computes the log-likelihood``

10. **Bugs.** A bug found while editing a docstring is not fixed in the docstring commit.
    It goes in its own commit with the ``BUG:`` prefix.

Extended summaries, ``See Also`` sections and examples are encouraged but not required.
As an example, this is a compliant docstring.

.. code-block:: python

   def scale(x, factor=1.0, method="mean", **kwargs):
       """
       Scale a series.

       Parameters
       ----------
       x : array_like
           The series to scale.
       factor : float, optional
           Multiplier applied after centering.
       method : {"mean", "median"}, optional
           The statistic used to center the series.
       **kwargs : dict, optional
           Keyword arguments passed to ``numpy.asarray``.

       Returns
       -------
       ndarray
           The scaled series.
       """

Checking docstrings
~~~~~~~~~~~~~~~~~~~

The rules are configured in the ``[tool.numpydoc_validation]`` table of
``pyproject.toml`` and checked in two ways. From the root of the repository, run

.. code-block:: bash

   python -m numpydoc lint $(git ls-files "arch/*.py")
   python -m pytest arch/tests/test_docstrings.py

Both need numpydoc 1.10 or later, which ``requirements-dev.txt`` and the ``dev`` extra
install. The Lint workflow runs the first command, and the second is skipped when
numpydoc is not installed.

``numpydoc lint`` reads the source files. It cannot see a docstring that a class
inherits when it is created or the docstring of a compiled object, so it does not check
for a missing docstring (GL08) or a missing Returns section (RT01).
``arch/tests/test_docstrings.py`` validates the live public objects for these and the
other structural problems. ``KNOWN_FAILURES`` in that file lists the problems that remain.
The test fails for a new problem and for a listed problem that no longer occurs, so
remove an entry when you fix it.

A check that is wrong for one object can be silenced with a comment on the line that
starts its definition, for example ``def method(self):  # numpydoc ignore=PR01``.

Input validation
----------------

Every public function, constructor and method checks its arguments when it is called,
before any computation, so that a bad argument never fails deep inside NumPy, SciPy or
the optimizer. The checks are in ``arch/utility/_validation.py`` and are used in the
same way everywhere.

.. note::

   The rules below are the decisions of the 9.0 release. They are switches in the
   helpers (the private functions ``_is_bool``, ``_is_flag_integer``, ``_is_integer``,
   ``_is_real`` and
   ``_float_in_range`` in ``arch/utility/_validation.py``) and in the lists of values in
   ``arch/tests/validation_table.py``, so that a different answer changes those places
   and not the code that calls them.

The rules
~~~~~~~~~

1. **Flags** accept ``bool``, ``np.bool_`` and the integers ``0`` and ``1`` (Python and
   NumPy integers), and ``check_bool`` returns a Python ``bool``. An integer other than
   ``0`` and ``1`` raises a ``ValueError``. Floats (including ``1.0``), ``"no"`` and
   arrays raise a ``TypeError``. ``None`` is accepted only where it is the documented
   default.
2. **Integers** accept Python and NumPy integers (``operator.index``). A float is
   rejected, including ``5.0``, and so is a ``bool``.
3. **Floats** accept Python and NumPy real numbers, including integers. ``bool`` is
   rejected. NaN and inf are rejected unless the argument is documented to accept them.
4. **NumPy scalars** are accepted wherever the Python type of the same kind is accepted.
5. **Data** is ``array_like`` and is converted to ``float64`` (``as_float64`` in
   ``arch/utility/array.py``). Strings, dates, complex numbers, ``None`` and objects that
   are not numbers cannot be converted. NaN and inf in the data of the tests, the
   estimators and the covariance estimators raise a ``ValueError`` that names the
   argument, except where NaN is handled by design (the data of the bootstraps, the
   diagnostics that drop NaN, outputs padded with NaN), which the docstring says.
6. **String options** are not case sensitive (``check_option``).
7. **Positions and dates** (``first_obs``, ``last_obs``, ``start``) must be in
   ``[-nobs, nobs]`` or, for a data set with a date index, a date within the index. A
   negative position counts from the end with the semantics that Python gives the
   argument: ``first_obs`` and ``last_obs`` are slice bounds, so ``last_obs=-1`` drops
   the last observation, and ``start`` is the index of an observation, so ``start=-1``
   is the last observation. The helper returns the position as a non-negative integer.
   A position outside the range raises a ``ValueError``. Say this in the docstring of
   every argument that takes a position.
8. **When:** everything that can be decided from the arguments (type, range, option,
   shape, finiteness, lags against the number of observations) is checked when the object
   is created or the function is called. Only errors in the numerical work (a singular
   matrix, no convergence) and errors in the order of calls stay lazy.
9. An argument that does not apply to the selected model is ignored, as documented
   (``arch_model`` ignores the arguments of a model that it does not select).
10. Methods of the volatility processes and distributions that are part of the API
    reference (``compute_variance``, ``simulate``, ``forecast``, ``bounds``, ...) check
    the type, ``float64`` dtype, contiguity, dimensions and length of the arrays that they
    pass to the compiled recursions, since a wrong array reads or writes memory that
    does not belong to it. The rules for the values (for example that the variance is
    positive) stay with the caller, because the optimizer tries any value.

The message
~~~~~~~~~~~

Every message has one form, built by the helpers::

    <name> must be <rule>; got <value> (<type>)

``name`` is the name of the argument as the user wrote it, ``rule`` states what is
accepted, and the value received is shortened to about 40 characters. For an array
the value is its shape. Examples:

.. code-block:: text

   lags must be an integer >= 0; got 2.5 (float)
   size must be a finite float in (0, 1); got 1.5 (float)
   method must be one of 'aic', 'bic', 't-stat' or 'maic'; got 'xyz' (str)
   y must be a 1-d array of finite numbers; got 1 NaN (list)
   seed must be None, a non-negative integer, a NumPy Generator or a NumPy RandomState;
   got -1 (int)

The exception
~~~~~~~~~~~~~

* ``TypeError`` for a value of the wrong type, including a non-string for a string option
  and data that cannot be converted to numbers.
* ``ValueError`` for a value of the right type that is not allowed: outside its range,
  not one of the options, the wrong shape or length, NaN or inf where finite values are
  needed.

``InfeasibleTestException`` is both a ``RuntimeError`` and a ``ValueError``.

Using the helpers
~~~~~~~~~~~~~~~~~

Each helper takes the name of the argument first and the value second, checks it, and
returns the value in its canonical type: a Python ``int``, ``float`` or ``bool`` and a
``float64`` ndarray.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Helper
     - Use it for
   * - ``check_bool(name, value, *, allow_none=False)``
     - flags
   * - ``check_int(name, value, *, lower=None, upper=None, allow_none=False)``
     - integers, with inclusive bounds
   * - ``check_float(name, value, *, lower=None, upper=None, inclusive="both",
       allow_nan=False, allow_inf=False, allow_none=False)``
     - real numbers, with bounds and ``inclusive`` in ``"both"``, ``"neither"``,
       ``"left"`` or ``"right"``
   * - ``check_seed(name, value, *, allow_none=True)``
     - ``seed``: None, a non-negative integer, a ``Generator`` or a ``RandomState``
   * - ``check_option(name, value, allowed, *, display=None, normalize=None)``
     - string options, not case sensitive
   * - ``check_str(name, value, *, allow_none=False)``
     - free strings, such as names
   * - ``check_callable(name, value, *, allow_none=False)``
     - functions
   * - ``check_instance(name, value, types, *, allow_none=False)``
     - objects of a class, such as a distribution
   * - ``check_array(name, value, *, ndim=None, shape=None, min_length=None,
       squeeze=True, allow_nan=False, allow_inf=False, allow_none=False)``
     - ``array_like`` data, converted to ``float64``, with its dimensions, length and
       finiteness
   * - ``check_ndarray(name, value, *, ndim, shape=None, writeable=False,
       allow_none=False)``
     - arrays that are passed to compiled code and must not be converted
   * - ``check_index(name, value, nobs, *, index=None, default=None, upper=None,
       allow_none=True)``
     - ``first_obs``, ``last_obs`` and ``start``, with negative positions counted from
       the end

.. code-block:: python

   from arch.utility._validation import check_array, check_float, check_int


   def fit(y, lags=1, size=0.05):
       lags = check_int("lags", lags, lower=0)
       size = check_float("size", size, lower=0, upper=1, inclusive="neither")
       y = check_array("y", y, ndim=1, min_length=lags + 2)
       ...

Call the helpers at the top of the function or in ``__init__``, and use the value that
they return. The helpers are private (``arch.utility._validation`` is not part of the
API), so a library that wants the same checks copies them.

Testing validation
~~~~~~~~~~~~~~~~~~

Add a ``test_validation_<area>.py`` file to the tests of the area. Build a list of rows
that give the callable, its arguments, the exception and a pattern for the message with
``arch/tests/validation_table.py``. The helper checks that every message has the form
above, supplies the usual bad and good values of flags, integers, floats and seeds, and
gives every row a unique id (pytest requires it):

.. code-block:: python

   import pytest

   from arch.tests.validation_table import INT_BAD, INT_GOOD, bad_value_rows
   from arch.tests.validation_table import good_value_rows, run_row, to_params
   from arch.unitroot import ADF

   BASE = {"y": Y}
   ROWS = [
       *bad_value_rows("ADF", lambda **kw: ADF(**kw), BASE, "lags", INT_BAD),
       *good_value_rows("ADF", lambda **kw: ADF(**kw), BASE, "lags", INT_GOOD),
   ]


   @pytest.mark.parametrize("row", to_params(ROWS))
   def test_validation(row):
       run_row(row)

Every fix needs a test that fails before it and passes after it. The pattern of a row
names the argument, which most of the old errors from NumPy and SciPy did not.
