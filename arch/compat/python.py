# Compatibility code from six and pandas
# flake8: noqa
import sys

if sys.version_info[0] == 3:
    def lmap(*args, **kwargs):
        return list(map(*args, **kwargs))
else:
    import __builtin__ as builtins
    lmap = builtins.map

__all__ = ['lmap']
