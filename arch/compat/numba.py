import functools
import os
from typing import Any, Callable

DISABLE_NUMBA = os.environ.get("ARCH_DISABLE_NUMBA", False) in ("1", "true", "True")


class PerformanceWarning(UserWarning):
    """Warning issued if recursions are run in CPython"""


performance_warning: str = """
numba is not available, and this function is being executed without JIT
compilation. Either install numba or reinstalling after installing Cython
is strongly recommended."""

try:
    if DISABLE_NUMBA:
        raise ImportError

    from numba import jit

    jit = functools.partial(jit, nopython=True)

except ImportError:

    def jit(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Callable[..., Any]:
            import warnings

            warnings.warn(performance_warning, PerformanceWarning)
            return func(*args, **kwargs)

        return wrapper


__all__ = ["jit", "PerformanceWarning"]
