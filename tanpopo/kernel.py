from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree


def neighbour_spacing(coords):
    coords = np.asarray(coords, dtype=float)
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=2)
    return float(np.mean(distances[:, 1]))


def wendland_c2(r):
    r = np.asarray(r)
    out = np.zeros_like(r, dtype=float)
    inside = r < 1
    x = r[inside]
    out[inside] = (1 - x) ** 4 * (4 * x + 1)
    return out


def _warn_connectivity(K, radius, threshold=0.5):
    if K.shape[0] == 0:
        return
    counts = np.diff(K.indptr)
    fraction = float(np.mean(counts == 0))
    if fraction > threshold:
        warnings.warn(
            f"radius={radius} leaves {fraction:.1%} of kernel centres with no neighbours. "
            "Consider increasing --radius.",
            UserWarning,
        )


def _symmetric_normalize(K, eps=1e-12):
    degree = np.asarray(K.sum(axis=1)).ravel()
    inv = np.divide(1.0, np.sqrt(degree), out=np.zeros_like(degree), where=degree > eps)
    return (sp.diags(inv, format="csr") @ K @ sp.diags(inv, format="csr")).tocsr()


def _bipartite_normalize(K, eps=1e-12):
    row_degree = np.asarray(K.sum(axis=1)).ravel()
    col_degree = np.asarray(K.sum(axis=0)).ravel()
    row_inv = np.divide(
        1.0, np.sqrt(row_degree), out=np.zeros_like(row_degree), where=row_degree > eps
    )
    col_inv = np.divide(
        1.0, np.sqrt(col_degree), out=np.zeros_like(col_degree), where=col_degree > eps
    )
    return (sp.diags(row_inv, format="csr") @ K @ sp.diags(col_inv, format="csr")).tocsr()


def kernel_matrix_sparse(
    coords,
    radius,
    coords_query=None,
    dtype=np.float64,
    normalisation="symmetric",
    zero_diagonal=True,
):
    """Build the spatial adjacency used by Tanpopo.

    The narrowed Tanpopo definition is explicitly cross-cell: square kernels are
    zero-diagonal by default. ``normalisation='symmetric'`` uses D^-1/2 K D^-1/2.
    Rectangular kernels use the analogous bipartite degree normalisation.
    """
    coords = np.asarray(coords, dtype=dtype)
    square = coords_query is None
    query = coords if square else np.asarray(coords_query, dtype=dtype)
    if radius <= 0:
        raise ValueError("radius must be positive")

    tree_centres = cKDTree(coords)
    tree_query = cKDTree(query)
    D = tree_centres.sparse_distance_matrix(
        tree_query, max_distance=radius, output_type="coo_matrix"
    )
    weights = wendland_c2(D.data / radius).astype(dtype, copy=False)
    K = sp.csr_matrix(
        (weights, (D.row, D.col)), shape=(coords.shape[0], query.shape[0]), dtype=dtype
    )
    K.sum_duplicates()

    if square:
        K = K.maximum(K.T).tocsr()
        if zero_diagonal:
            K.setdiag(0)
            K.eliminate_zeros()
        _warn_connectivity(K, radius)
        if normalisation == "symmetric":
            K = _symmetric_normalize(K)
        elif normalisation != "none":
            raise ValueError("normalisation must be 'symmetric' or 'none'")
    else:
        if normalisation == "symmetric":
            K = _bipartite_normalize(K)
        elif normalisation != "none":
            raise ValueError("normalisation must be 'symmetric' or 'none'")

    return K.astype(dtype, copy=False)


def kernel_diagnostics(K):
    K = K.tocsr()
    row_degree = np.asarray(K.sum(axis=1)).ravel()
    return {
        "n_rows": int(K.shape[0]),
        "n_cols": int(K.shape[1]),
        "nnz": int(K.nnz),
        "isolated_fraction": float(np.mean(np.diff(K.indptr) == 0)),
        "median_weighted_degree": float(np.median(row_degree)) if row_degree.size else 0.0,
    }
