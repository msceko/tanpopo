import warnings

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree


def _check_radius(D, radius, threshold=0.5):
    """
    Check that radius gives each kernel centre at least one non-self neighbour.
    D is a sparse distance matrix with shape: n_centres × n_spots
    """
    n_centres = D.shape[0]
    if n_centres == 0:
        return

    D = D.tocoo(copy=False)
    nonself_counts = np.bincount(D.row[D.data > 0], minlength=n_centres)
    isolated = nonself_counts == 0
    isolated_fraction = isolated.mean()

    if isolated_fraction > threshold:
        warnings.warn(
            f"radius={radius} produced no non-self neighbours for "
            f"{isolated.sum()} / {n_centres} kernel centers "
            f"({isolated_fraction:.1%}). "
            "The resulting kernel matrix will be mostly isolated "
            "self entries. Consider increasing `radius`.",
            UserWarning,
        )


def wendland_c2(r):
    """
    Wendland C^2 (compactly supported, PSD in R^d for d <= 3):
      phi(r) = (1 - r)^4_+ * (4r + 1),  r >= 0
    """
    r = np.asarray(r)
    out = np.zeros_like(r, dtype=float)

    inside = r < 1
    x = r[inside]

    out[inside] = (1 - x) ** 4 * (4 * x + 1)
    return out


def kernel_matrix_sparse(coords, radius, coords_query=None, dtype=np.float64):
    """Build a sparse Wendland kernel matrix.
    ``coords`` gives the kernel centres. ``coords_query`` gives the spots
    evaluated by each centre. If ``coords_query`` is ``None``, ``coords`` is
    used for both and the result is square.

    Parameters
    ----------
    coords : array-like, shape (n_centres, n_dims)
        Coordinates of kernel centres.
    radius : float
        Kernel support radius.
    coords_query : array-like, shape (n_spots, n_dims), optional
        Coordinates of queried spots.
    dtype : dtype, default=np.float64
        Numeric dtype for coordinates and kernel values.

    Returns
    -------
    scipy.sparse.csr_matrix, shape (n_centres, n_spots)
        Sparse matrix of Wendland kernel weights.
    """
    coords = np.asarray(coords, dtype=dtype)

    if coords_query is None:
        coords_query = coords
    else:
        coords_query = np.asarray(coords_query, dtype=dtype)

    tree_centres = cKDTree(coords)
    tree_query = cKDTree(coords_query)

    D = tree_centres.sparse_distance_matrix(
        tree_query, max_distance=radius, output_type="coo_matrix"
    )
    _check_radius(D, radius)
    weights = wendland_c2(D.data / radius).astype(dtype, copy=False)

    K = sp.csr_matrix(
        (weights, (D.row, D.col)), shape=(coords.shape[0], coords_query.shape[0]), dtype=dtype
    )

    return K
