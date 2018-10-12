import numpy as np


def vech(c, loc=False):
    k = c.shape[0]
    sel = np.tril(np.ones((k, k), dtype=np.bool))
    if loc:
        return np.where(sel.flat)[0]
    return c[sel]


def inv_vech(v, symmetric=False):
    n = len(v)
    k = int(np.floor(np.sqrt(2 * n)))
    c = np.zeros((k, k))
    sel = np.tril(np.ones((k, k), dtype=np.bool))
    c[sel] = v
    if symmetric:
        c.T[sel] = v

    return c


def symmetric_matrix_root(m):
    val, vec = np.linalg.eigh(m)
    return np.linalg.multi_dot((vec, np.diag(np.sqrt(val)), vec.T))


def symmetric_matrix_invroot(m):
    val, vec = np.linalg.eigh(m)
    return np.linalg.multi_dot((vec, np.diag(1. / np.sqrt(val)), vec.T))
