# Compatibility code from six and pandas
# flake8: noqa
import sys

PY3 = sys.version_info[0] == 3

if PY3:
    from io import StringIO
    range = range
    long = int

    def lmap(*args, **kwargs):
        return list(map(*args, **kwargs))
else:
    from cStringIO import StringIO
    import __builtin__ as builtins
    range = xrange
    long = long
    lmap = builtins.map


def add_metaclass(metaclass):
    """Class decorator for creating a class with a metaclass."""
    def wrapper(cls):
        orig_vars = cls.__dict__.copy()
        slots = orig_vars.get('__slots__')
        if slots is not None:
            if isinstance(slots, str):
                slots = [slots]
            for slots_var in slots:
                orig_vars.pop(slots_var)
        orig_vars.pop('__dict__', None)
        orig_vars.pop('__weakref__', None)
        return metaclass(cls.__name__, cls.__bases__, orig_vars)
    return wrapper


def iteritems(obj, **kwargs):
    """replacement for six's iteritems for Python2/3 compat
       uses 'iteritems' if available and otherwise uses 'items'.

       Passes kwargs to method.
    """
    func = getattr(obj, "iteritems", None)
    if not func:
        func = obj.items
    return func(**kwargs)


def itervalues(obj, **kwargs):
    func = getattr(obj, "itervalues", None)
    if not func:
        func = obj.values
    return func(**kwargs)


__all__ = ['iteritems', 'itervalues', 'add_metaclass', 'lmap', 'long', 'range', 'PY3', 'StringIO']
