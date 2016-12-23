from __future__ import absolute_import, division
from ..compat.numba import jit


def stationary_bootstrap_sample_python(indices, u, p):
    """
    Parameters
    -------
    indices: array
        Array containing draws from randint with the same size as the data in
        the range of [0,nobs)
    u : array
        Array of standard uniforms
    p : float
        Probability that a new block is started in the stationary bootstrap.
        The multiplicative reciprocal of the window length

    Returns
    -------
    indices: array
        Indices for an iteration of the stationary bootstrap
    """
    num_items = indices.shape[0]
    for i in range(1, num_items):
        if u[i] > p:
            indices[i] = indices[i - 1] + 1
            if indices[i] == num_items:
                indices[i] = 0

    return indices


stationary_bootstrap_sample = jit(stationary_bootstrap_sample_python)
