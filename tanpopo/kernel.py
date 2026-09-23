from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree


VALID_GEOMETRY_NORMALISATIONS = {"none", "mass", "distance"}
VALID_SPATIAL_STATISTICS = {"mark_correlation", "variogram"}


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
    """Build the base Wendland adjacency used by Tanpopo.

    Square kernels are zero-diagonal by default. ``normalisation='symmetric'``
    uses D^-1/2 K D^-1/2. Rectangular kernels use the analogous bipartite
    degree normalisation.

    Geometry standardisation and conversion to a mark-correlation or variogram
    operator are deliberately separate; see :func:`spatial_statistic_kernels`.
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


def _edge_distances(K, coords):
    """Return COO edge indices and Euclidean distances for nonzero entries of K."""
    C = K.tocoo(copy=False)
    coords = np.asarray(coords, dtype=float)
    delta = coords[C.row] - coords[C.col]
    distances = np.sqrt(np.sum(delta * delta, axis=1))
    return C, distances


def _bin_index(distances, bin_edges):
    idx = np.searchsorted(bin_edges, distances, side="right") - 1
    # Nonzero Wendland edges satisfy distance < radius, but clipping avoids
    # floating-point edge cases exactly at a supplied boundary.
    return np.clip(idx, 0, len(bin_edges) - 2)


def _bin_masses(K, coords, bin_edges):
    C, distances = _edge_distances(K, coords)
    idx = _bin_index(distances, bin_edges)
    masses = np.bincount(idx, weights=C.data, minlength=len(bin_edges) - 1).astype(float)
    return masses, C, idx, distances


def _mass_standardise(K, target_mass, eps=1e-12):
    mass = float(K.sum())
    if mass <= eps:
        raise ValueError("Spatial graph has zero total edge weight")
    return (K * (float(target_mass) / mass)).tocsr()


def _distance_standardise(adjacencies, coords, radius, n_bins, eps=1e-12):
    """Standardise all samples to one common weighted pair-distance measure.

    The reference distance profile is the equal-sample mean of each sample's
    normalised weighted pair-distance profile, restricted to bins represented in
    every sample. Each output adjacency has total mass equal to its number of
    cells, so subsequent ``1 / n_spots`` sample weighting estimates an average
    pair statistic under the same reference measure in every tissue.
    """
    n_bins = int(n_bins)
    if n_bins < 1:
        raise ValueError("distance_bins must be a positive integer")
    bin_edges = np.linspace(0.0, float(radius), n_bins + 1)
    cached = [_bin_masses(K, xy, bin_edges) for K, xy in zip(adjacencies, coords)]
    masses = np.vstack([x[0] for x in cached])
    common = np.all(masses > eps, axis=0)
    if not np.any(common):
        raise ValueError(
            "No distance bin has positive pair weight in every sample; increase --radius, "
            "reduce --distance-bins, or use --geometry-normalisation mass."
        )

    profiles = np.zeros_like(masses, dtype=float)
    for i, row in enumerate(masses):
        total = float(row[common].sum())
        profiles[i, common] = row[common] / total
    reference = profiles.mean(axis=0)
    reference[~common] = 0.0
    reference /= reference.sum()

    outputs = []
    diagnostics = []
    for K, xy, (row_mass, C, idx, distances) in zip(adjacencies, coords, cached):
        keep = common[idx]
        scale = np.zeros(n_bins, dtype=float)
        n = K.shape[0]
        scale[common] = n * reference[common] / row_mass[common]
        data = C.data * scale[idx]
        data[~keep] = 0.0
        out = sp.csr_matrix((data, (C.row, C.col)), shape=K.shape, dtype=K.dtype)
        out.eliminate_zeros()
        outputs.append(out)
        diagnostics.append(
            {
                "raw_pair_mass": float(K.sum()),
                "pair_mass": float(out.sum()),
                "distance_bin_mass_raw": row_mass.copy(),
                "distance_bin_mass": np.bincount(
                    idx[keep], weights=data[keep], minlength=n_bins
                ).astype(float),
                "distance_bin_edges": bin_edges.copy(),
                "distance_reference_weights": reference.copy(),
                "common_distance_bins": common.copy(),
                "weighted_distance_quantiles": _weighted_quantiles(
                    distances[keep], data[keep], [0.1, 0.5, 0.9]
                ),
            }
        )
    return outputs, diagnostics


def _weighted_quantiles(values, weights, quantiles):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    if values.size == 0 or float(weights.sum()) <= 0:
        return np.full(quantiles.shape, np.nan)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    return np.interp(quantiles, cdf, values)


def adjacency_to_variogram(K):
    """Convert a symmetric pair-weight adjacency to its graph Laplacian.

    For residual expression Y,

        Y.T @ (D - K) @ Y

    equals ``1/2 sum_ij K_ij (Y_i-Y_j)(Y_i-Y_j).T``. Thus, after pair-mass
    normalisation, the operator estimates a matrix-valued mark variogram under
    exactly the same pair measure as the mark-correlation statistic.
    """
    K = K.tocsr()
    degree = np.asarray(K.sum(axis=1)).ravel()
    return (sp.diags(degree, format="csr") - K).tocsr()


def spatial_statistic_kernels(
    coords,
    radius,
    *,
    spatial_statistic="mark_correlation",
    geometry_normalisation="distance",
    distance_bins=10,
    graph_normalisation="symmetric",
    dtype=np.float64,
):
    """Build comparable square spatial operators for one or more samples.

    Parameters
    ----------
    coords
        Sequence of ``(n_s, 2)`` coordinate matrices after any cell-type mask.
    spatial_statistic
        ``'mark_correlation'`` returns a zero-diagonal weighted adjacency.
        ``'variogram'`` returns its graph Laplacian.
    geometry_normalisation
        ``'none'`` preserves the historical graph scale; ``'mass'`` fixes each
        adjacency's total pair mass to ``n_s``; ``'distance'`` additionally
        forces every sample to have the same weighted pair-distance profile.
    distance_bins
        Number of equal-width bins on ``[0, radius]`` used for distance
        standardisation.

    Returns
    -------
    operators, diagnostics
        Lists aligned to ``coords``. Diagnostics describe the underlying pair
        adjacency even when ``spatial_statistic='variogram'``.
    """
    if spatial_statistic not in VALID_SPATIAL_STATISTICS:
        raise ValueError(
            f"spatial_statistic must be one of {sorted(VALID_SPATIAL_STATISTICS)}"
        )
    if geometry_normalisation not in VALID_GEOMETRY_NORMALISATIONS:
        raise ValueError(
            "geometry_normalisation must be one of "
            f"{sorted(VALID_GEOMETRY_NORMALISATIONS)}"
        )
    coords = [np.asarray(xy, dtype=dtype) for xy in coords]
    if not coords:
        raise ValueError("At least one coordinate matrix is required")

    adjacency = [
        kernel_matrix_sparse(
            xy,
            radius,
            dtype=dtype,
            normalisation=graph_normalisation,
            zero_diagonal=True,
        )
        for xy in coords
    ]

    if geometry_normalisation == "distance":
        adjacency, diagnostics = _distance_standardise(
            adjacency, coords, radius, distance_bins
        )
    else:
        diagnostics = []
        output = []
        for K, xy in zip(adjacency, coords):
            raw_mass = float(K.sum())
            if geometry_normalisation == "mass":
                K = _mass_standardise(K, K.shape[0])
            C, distances = _edge_distances(K, xy)
            output.append(K)
            diagnostics.append(
                {
                    "raw_pair_mass": raw_mass,
                    "pair_mass": float(K.sum()),
                    "weighted_distance_quantiles": _weighted_quantiles(
                        distances, C.data, [0.1, 0.5, 0.9]
                    ),
                }
            )
        adjacency = output

    for K, diag in zip(adjacency, diagnostics):
        diag.update(kernel_diagnostics(K))
        diag["geometry_normalisation"] = geometry_normalisation
        diag["spatial_statistic"] = spatial_statistic

    if spatial_statistic == "variogram":
        operators = [adjacency_to_variogram(K).astype(dtype, copy=False) for K in adjacency]
    else:
        operators = [K.astype(dtype, copy=False) for K in adjacency]
    return operators, diagnostics


def kernel_diagnostics(K):
    K = K.tocsr()
    row_degree = np.asarray(K.sum(axis=1)).ravel()
    mean_degree = float(np.mean(row_degree)) if row_degree.size else 0.0
    degree_cv = (
        float(np.std(row_degree) / mean_degree) if row_degree.size and mean_degree > 0 else 0.0
    )
    return {
        "n_rows": int(K.shape[0]),
        "n_cols": int(K.shape[1]),
        "nnz": int(K.nnz),
        "isolated_fraction": float(np.mean(np.diff(K.indptr) == 0)),
        "total_edge_weight": float(K.sum()),
        "mean_weighted_degree": mean_degree,
        "median_weighted_degree": float(np.median(row_degree)) if row_degree.size else 0.0,
        "weighted_degree_cv": degree_cv,
    }
