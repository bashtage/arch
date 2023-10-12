import functools
import os
from typing import Any, Callable

from arch.utility.exceptions import PerformanceWarning

DISABLE_NUMBA = os.environ.get("ARCH_DISABLE_NUMBA", False) in ("1", "true", "True")

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

    def jit(
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        def wrap(func):
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Callable[..., Any]:
                import warnings

                warnings.warn(performance_warning, PerformanceWarning)
                return func(*args, **kwargs)

            return wrapper

        return wrap


__all__ = ["jit", "PerformanceWarning"]
