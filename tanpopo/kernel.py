import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


def gaussian_kernel(coords, sigma):
    """
    Kernel for ⟨N(mu_i), N(mu_j)⟩ with isotropic variance sigma^2

    Returns:
        K : (spots, spots)
    """
    d2 = cdist(coords, coords, metric="sqeuclidean")
    return np.exp(-d2 / (4 * sigma**2))


def wendland_c2(r):
    """
    Wendland C^2 (compactly supported, PSD in R^d for d <= 3):
      phi(r) = (1 - r)^4_+ * (4r + 1),  r >= 0
    """
    r = np.asarray(r, dtype=np.float64)
    t = np.maximum(1.0 - r, 0.0)
    return (t**4) * (4.0 * r + 1.0)


def kernel_matrix_sparse(coords, radius, symmetrize=True):
    """
    Build a sparse radius-neighborhood Wendland kernel K (CSR) on spot coordinates.

      K_ij = phi(||xi-xj|| / ell)   for ||xi-xj|| <= radius
      phi = Wendland C^2

    Args:
        coords: (n_spots, d)
        radius: positive length-scale. Support of phi is r<=1, i.e. ||xi-xj|| <= ell.
        symmetrize: (K + K.T)/2

    Returns:
        K: scipy.sparse.csr_matrix, shape (n_spots, n_spots)
    """
    nn = NearestNeighbors(radius=radius, algorithm="ball_tree", metric="euclidean")
    nn.fit(coords)

    K = nn.radius_neighbors_graph(coords, mode="distance")  # CSR
    K.data = wendland_c2(K.data / radius)

    if symmetrize:
        Kt = K.T.tocsr()
        K = K.maximum(Kt)

    K.eliminate_zeros()

    return K
