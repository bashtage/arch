from contextlib import contextmanager
from typing import Any

_BACKEND_ENGINE = "numpy"


def backend():
    return _BACKEND_ENGINE


def set_backend(library_name):
    """
    Set backend engine.

    The function sets the backend engine in global level.

    Parameters
    ----------
    library_name : str
        Library name. Default is `numpy`. Options are `numpy`, `tensorflow`,
        `cupy` and `jax`.
    """
    assert library_name.lower() in ["numpy", "tensorflow", "cupy", "jax"], (
        "Only `numpy`, `tensorflow`, `cupy` and `jax` are supported, but not "
        f"{library_name}"
    )
    global _BACKEND_ENGINE
    _BACKEND_ENGINE = library_name


@contextmanager
def use_backend(library_name="numpy"):
    """
    NumPy engine selection.

    The function is a context manager to enable users to switch to a
    specific library as a replacement of NumPy in CPU.

    Parameters
    ----------
    library_name : str
        Library name. Default is `numpy`. Options are `numpy`, `tensorflow`,
        `cupy` and `jax`.
    """
    assert library_name.lower() in ["numpy", "tensorflow", "cupy", "jax"], (
        "Only `numpy`, `tensorflow`, `cupy` and `jax` are supported, but not "
        f"{library_name}"
    )
    global _BACKEND_ENGINE
    _original = _BACKEND_ENGINE
    try:
        _BACKEND_ENGINE = library_name
        if _BACKEND_ENGINE == "tensorflow":
            import tensorflow.experimental.numpy as np

            np.experimental_enable_numpy_behavior()
        yield
    finally:
        _BACKEND_ENGINE = _original


class NumpyEngine:
    """
    NumPy engine.
    """

    @property
    def name(self):
        """
        Get engine name.
        """
        global _BACKEND_ENGINE
        return _BACKEND_ENGINE

    def __getattribute__(self, __name: str) -> Any:
        global _BACKEND_ENGINE
        try:
            if _BACKEND_ENGINE == "numpy":
                import numpy as anp
            elif _BACKEND_ENGINE == "tensorflow":
                import tensorflow.experimental.numpy as anp
            elif _BACKEND_ENGINE == "cupy":
                import cupy as anp
            elif _BACKEND_ENGINE == "jax":
                import jax.numpy as anp
            else:
                raise ValueError(f"Cannot recognize backend {_BACKEND_ENGINE}")
        except ImportError:
            raise ImportError(
                "Library `numpy` cannot be imported from backend engine "
                f"{_BACKEND_ENGINE}. Please make sure to install the library "
                f"via `pip install {_BACKEND_ENGINE}`."
            )

        try:
            return getattr(anp, __name)
        except AttributeError:
            raise AttributeError(
                "Cannot get attribute / function from numpy library in "
                f"backend engine {_BACKEND_ENGINE}"
            )


class LinAlgEngine:
    """
    Linear algebra engine.
    """

    @property
    def name(self):
        """
        Get engine name.
        """
        global _BACKEND_ENGINE
        return _BACKEND_ENGINE

    def __getattribute__(self, __name: str) -> Any:
        global _BACKEND_ENGINE
        try:
            if _BACKEND_ENGINE == "numpy":
                import numpy.linalg as alinalg
            elif _BACKEND_ENGINE == "tensorflow":
                import tensorflow.linalg as alinalg
            elif _BACKEND_ENGINE == "cupy":
                import cupy.linalg as alinalg
            elif _BACKEND_ENGINE == "jax":
                import jax.numpy.linalg as alinalg
            else:
                raise ValueError(f"Cannot recognize backend {_BACKEND_ENGINE}")
        except ImportError:
            raise ImportError(
                "Library `linalg` cannot be imported from backend engine "
                f"{_BACKEND_ENGINE}. Please make sure to install the library "
                f"via `pip install {_BACKEND_ENGINE}`."
            )

        try:
            return getattr(alinalg, __name)
        except AttributeError:
            raise AttributeError(
                "Cannot get attribute / function from linalg library in "
                f"backend engine {_BACKEND_ENGINE}"
            )


def fori_loop(lower, upper, body_fun, init_val=None):
    global _BACKEND_ENGINE
    if _BACKEND_ENGINE in ["numpy", "cupy"]:
        val = init_val
        for i in range(lower, upper):
            val = body_fun(i, val)
        return val
    elif _BACKEND_ENGINE == "jax":
        import jax.lax

        return jax.lax.fori_loop(lower, upper, body_fun, init_val)
    elif _BACKEND_ENGINE == "tensorflow":
        import tensorflow as tf

        i = tf.constant(lower)
        while_condition = lambda i: tf.less(i, upper)

        def body(i, val):
            return [tf.add(i, 1), body_fun(val)]

        return tf.while_loop(while_condition, body, [i, init_val])

    raise ImportError(f"Cannot recognize backend {_BACKEND_ENGINE}")


numpy = NumpyEngine()
linalg = LinAlgEngine()
