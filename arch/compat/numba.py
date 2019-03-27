from __future__ import absolute_import, division

import functools


class PerformanceWarning(UserWarning):
    pass


performance_warning = '''
numba is not available, and this function is being executed without JIT
compilation. Either install numba or reinstalling after installing Cython
is strongly recommended.'''

try:
    from numba import jit

    try:
        def f(x, y):
            return x + y

        fjit = jit(f, nopython=True, fastmath=True)
        fjit(1.0, 2.0)
        jit = functools.partial(jit, nopython=True, fastmath=True)
    except KeyError:
        jit = functools.partial(jit, nopython=True)
except ImportError:
    def jit(func, *args, **kwargs):
        def wrapper(*args, **kwargs):
            import warnings
            warnings.warn(performance_warning, PerformanceWarning)
            return func(*args, **kwargs)

        return wrapper

__all__ = ['jit', 'PerformanceWarning']
