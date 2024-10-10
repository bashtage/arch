import functools
import os
from typing import Any, Callable, Optional

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
        function_or_signature: Optional[Callable[..., Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if function_or_signature is not None and callable(function_or_signature):
            # Used directly, e.g., f_jit = jit(f)
            @functools.wraps(function_or_signature)
            def wrapper(*args: Any, **kwargs: Any) -> Callable[..., Any]:
                import warnings

                warnings.warn(performance_warning, PerformanceWarning)
                return function_or_signature(*args, **kwargs)

            return wrapper

        # Used as a decorator, e.g., @jit
        def wrap(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Callable[..., Any]:
                import warnings

                warnings.warn(performance_warning, PerformanceWarning)
                return func(*args, **kwargs)

            return wrapper

        return wrap


__all__ = ["jit", "PerformanceWarning"]
