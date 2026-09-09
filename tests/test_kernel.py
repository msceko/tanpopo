import numpy as np

from tanpopo.kernel import kernel_matrix_sparse


def test_square_kernel_is_zero_diagonal_and_symmetric():
    coords = np.c_[np.linspace(0, 10, 30), np.zeros(30)]
    K = kernel_matrix_sparse(coords, radius=2.0)
    np.testing.assert_allclose(K.diagonal(), 0.0)
    np.testing.assert_allclose(K.toarray(), K.toarray().T)
    assert K.nnz > 0
