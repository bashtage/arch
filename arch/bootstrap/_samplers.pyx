import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def stationary_bootstrap_sample(np.int64_t[:] indices,
                                double[:] u,
                                double p):
    """
    Parameters
    -------
    indices: 1-d array
        Array containing draws from randint with the same size as the data in
        the range of [0,nobs)
    u : 1-d array
        Array of standard uniforms
    p : float
        Probability that a new block is started in the stationary bootstrap.
        The multiplicative reciprocal of the window length

    Returns
    -------
    indices: 1-d array
        Indices for an iteration of the stationary bootstrap
    """
    cdef Py_ssize_t num_items, i
    num_items = indices.shape[0]

    for i in range(1, num_items):
        if u[i] > p:
            indices[i] = indices[i - 1] + 1
            if indices[i] == num_items:
                indices[i] = 0

    return indices