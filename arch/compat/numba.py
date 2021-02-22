import functools
from typing import Any, Callable


class PerformanceWarning(UserWarning):
    """Warning issued if recursions are run in CPython"""


performance_warning: str = """
numba is not available, and this function is being executed without JIT
compilation. Either install numba or reinstalling after installing Cython
is strongly recommended."""

try:
    from numba import jit

    try:

        def f(x: float, y: float) -> float:
            return x + y

        fjit = jit(f, nopython=True, fastmath=True)
        fjit(1.0, 2.0)
        jit = functools.partial(jit, nopython=True, fastmath=True)
    except KeyError:
        jit = functools.partial(jit, nopython=True)
except ImportError:

    def jit(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Callable[..., Any]:
            import warnings

            warnings.warn(performance_warning, PerformanceWarning)
            return func(*args, **kwargs)

        return wrapper


__all__ = ["jit", "PerformanceWarning"]
