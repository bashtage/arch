import numpy.linalg


def lstsq(y, x, rcond=None):
    """
    Wrapper to handle rcond changes in 1.14

    Remove after 1.13 is dropped
    """
    try:
        return numpy.linalg.lstsq(y, x, rcond=rcond)
    except TypeError:
        return numpy.linalg.lstsq(y, x, rcond=-1)
