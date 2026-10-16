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
   obvious.

   - Good: ``reps : int, optional``
   - Bad: ``reps : int, default 1000`` or ``reps : int`` for ``reps=1000``

2. **Unions.** Write a union of types with ``or``. Braces are only for a set of literal
   values.

   - Good: ``x : ndarray or Series`` and ``method : {"aic", "bic"}``
   - Bad: ``x : {ndarray, Series}``

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

9. **Punctuation.** The summary is one line that starts with a verb in the imperative
   mood. The summary, the parameter descriptions and the return descriptions start with
   a capital letter and end with a period.

   - Good: ``Compute the log-likelihood.``
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

``numpydoc lint`` reads the source files. It cannot see a docstring that a class
inherits when it is created or the docstring of a compiled object, so it does not check
for a missing docstring (GL08) or a missing Returns section (RT01).
``arch/tests/test_docstrings.py`` validates the live public objects for these and the
other structural problems. ``KNOWN_FAILURES`` in that file lists the problems that remain.
The test fails for a new problem and for a listed problem that no longer occurs, so
remove an entry when you fix it.

A check that is wrong for one object can be silenced with a comment on the line that
starts its definition, for example ``def method(self):  # numpydoc ignore=PR01``.
