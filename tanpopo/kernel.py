import warnings

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


def _check_radius(coords, radius, K, threshold=0.5):
    """Check that the chosen radius does not produce points with no neighbours"""
    n_points = coords.shape[0]
    row_counts = np.diff(K.indptr)
    has_self = K.diagonal() == 0
    nonself_counts = row_counts - has_self.astype(row_counts.dtype)
    isolated = nonself_counts == 0
    isolated_fraction = isolated.mean() if n_points else 0.0
    if isolated_fraction > threshold:
        warnings.warn(
            f"radius={radius} produced no non-self neighbors for "
            f"{isolated.sum()} / {n_points} points "
            f"({isolated_fraction:.1%}). The resulting kernel matrix will be "
            "mostly isolated diagonal/self entries. Consider increasing `radius`.",
            UserWarning,
        )


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
    _check_radius(coords, radius, K)
    K.data = wendland_c2(K.data / radius)

    if symmetrize:
        Kt = K.T.tocsr()
        K = K.maximum(Kt)

    K.eliminate_zeros()

    return K
