Common Type Definitions
=======================

The type aliases below are defined in ``arch._typing`` and are used in the
signatures of the public API. They are documented here so that the names that
appear in type annotations can be looked up. They are not part of the public
API and may change between releases.

.. py:currentmodule:: arch._typing

Array Types
-----------

.. py:data:: NDArray
   :value: numpy.ndarray

   A NumPy array of any shape and data type.

.. py:data:: Float64Array
   :value: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.float64]]

   A NumPy array with any number of dimensions containing 64-bit floating point
   values.

.. py:data:: Float64Array1D
   :value: numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]]

   A 1-dimensional NumPy array containing 64-bit floating point values.

.. py:data:: Float64Array2D
   :value: numpy.ndarray[tuple[int, int], numpy.dtype[numpy.float64]]

   A 2-dimensional NumPy array containing 64-bit floating point values.

.. py:data:: Int64Array
   :value: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.int64]]

   A NumPy array with any number of dimensions containing 64-bit integers.

.. py:data:: Int64Array1D
   :value: numpy.ndarray[tuple[int], numpy.dtype[numpy.int64]]

   A 1-dimensional NumPy array containing 64-bit integers.

.. py:data:: Int64Array2D
   :value: numpy.ndarray[tuple[int, int], numpy.dtype[numpy.int64]]

   A 2-dimensional NumPy array containing 64-bit integers.

.. py:data:: Int32Array
   :value: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.intc]]

   A NumPy array with any number of dimensions containing C ``int`` values,
   which are 32-bit integers on all supported platforms.

.. py:data:: IntArray
   :value: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.int64]]

   A NumPy array with any number of dimensions containing 64-bit integers.

.. py:data:: Uint32Array
   :value: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uintc]]

   A NumPy array with any number of dimensions containing unsigned C ``int``
   values.

.. py:data:: BoolArray
   :value: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]]

   A NumPy array with any number of dimensions containing boolean values.

.. py:data:: AnyArray
   :value: numpy.ndarray[tuple[int, ...], typing.Any]

   A NumPy array with any number of dimensions and any data type.

.. py:data:: AnyArray1D
   :value: numpy.ndarray[tuple[int], typing.Any]

   A 1-dimensional NumPy array with any data type.

Array-like Types
----------------

.. py:data:: ArrayLike1D
   :value: Float64Array1D | pandas.Series

   A 1-dimensional array of floats, either a NumPy array or a pandas ``Series``.

.. py:data:: ArrayLike2D
   :value: Float64Array2D | pandas.DataFrame

   A 2-dimensional array of floats, either a NumPy array or a pandas
   ``DataFrame``.

.. py:data:: ArrayLike
   :value: numpy.ndarray | pandas.DataFrame | pandas.Series

   A NumPy array, a pandas ``DataFrame`` or a pandas ``Series``.

.. py:data:: AnyPandas
   :value: pandas.Series | pandas.DataFrame

   A pandas ``Series`` or ``DataFrame``.

.. py:data:: NDArrayOrFrame
   :type: TypeVar
   :value: TypeVar("NDArrayOrFrame", Float64Array, DataFrame)

   A type variable that is either a NumPy array of floats or a pandas
   ``DataFrame``. It is used when the type of the output matches the type of
   the input.

.. py:data:: FloatOrArray
   :type: TypeVar
   :value: TypeVar("FloatOrArray", float, numpy.ndarray)

   A type variable that is either a Python ``float`` or a NumPy array. It is
   used when the type of the output matches the type of the input.

Other Types
-----------

.. py:data:: RandomStateState
   :value: tuple[str, Uint32Array, int, int, float]

   The state of a legacy :class:`numpy.random.RandomState`, as returned by
   :meth:`numpy.random.RandomState.get_state`.

.. py:data:: RNGType
   :value: Callable[[int | tuple[int, ...]], Float64Array]

   A function that takes a size, either an integer or a tuple of integers, and
   returns an array of random numbers of that size.

.. py:data:: BootstrapIndexT
   :value: Int64Array1D | tuple[Int64Array1D, ...] | tuple[list[Int64Array1D], dict[str, Int64Array1D]]

   The indices used to construct a bootstrap sample. They are either a single
   array of indices, a tuple with an array of indices for each positional
   input, or a tuple containing a list of arrays of indices for the positional
   inputs and a dictionary of arrays of indices for the keyword inputs.

.. py:data:: DateLike
   :value: str | datetime.datetime | numpy.datetime64 | pandas.Timestamp

   A value that can be interpreted as a date.

.. py:data:: Label
   :value: collections.abc.Hashable | None

   A label for an element in a pandas ``Index``, for example a column name.

.. py:data:: UnitRootTrend
   :value: Literal["n", "c", "ct", "ctt"]

   The deterministic trend terms that are available in unit root tests: no
   deterministic terms (``"n"``), a constant (``"c"``), a constant and a linear
   time trend (``"ct"``), and a constant, a linear and a quadratic time trend
   (``"ctt"``).

.. py:data:: ForecastingMethod
   :value: Literal["analytic", "simulation", "bootstrap"]

   The methods that can be used to construct volatility forecasts.
