from __future__ import absolute_import, division

try:
    from numba import jit
except ImportError:
    def jit(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

__all__ = ['jit']
