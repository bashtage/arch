import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from arch.multivariate.utility import vech, inv_vech, symmetric_matrix_invroot, \
    symmetric_matrix_root

MATRICES = [np.eye(3) + np.diag(np.ones(3)),
            np.eye(2),
            np.eye(4) + np.array([[.3, .4, .5, .6]]).T.dot(np.array([[.3, .4, .5, .6]]))]


@pytest.mark.parametrize('matrix', MATRICES)
def test_roots(matrix):
    m = matrix.shape[0]
    root = symmetric_matrix_root(matrix)
    inv_root = symmetric_matrix_invroot(matrix)
    assert_array_almost_equal(root.dot(inv_root), np.eye(m))
    assert_array_almost_equal(inv_root.dot(root), np.eye(m))
    assert_array_almost_equal(np.linalg.multi_dot((inv_root, matrix, inv_root)), np.eye(m))
    assert_array_almost_equal(np.linalg.multi_dot((root, np.linalg.inv(matrix), root)), np.eye(m))


@pytest.mark.parametrize('matrix', MATRICES)
def test_vech(matrix):
    assert_array_almost_equal(inv_vech(vech(matrix), symmetric=True), matrix)
    assert_array_almost_equal(inv_vech(vech(matrix), symmetric=False), np.tril(matrix))
