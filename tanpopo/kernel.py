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


def _warn_connectivity(K, radius, threshold=0.5, check_columns=False):
    if K.shape[0] == 0:
        return
    row_counts = np.diff(K.indptr)
    row_fraction = float(np.mean(row_counts == 0))
    if row_fraction > threshold:
        warnings.warn(
            f"radius={radius} leaves {row_fraction:.1%} of kernel rows with no neighbours. "
            "Consider increasing --radius.",
            UserWarning,
        )
    if check_columns:
        col_counts = np.diff(K.tocsc().indptr)
        col_fraction = float(np.mean(col_counts == 0))
        if col_fraction > threshold:
            warnings.warn(
                f"radius={radius} leaves {col_fraction:.1%} of kernel columns with no "
                "neighbours. Consider increasing --radius.",
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


def _remove_identity_pairs(K, row_ids, col_ids):
    """Remove pairs referring to the same original observation.

    This is the rectangular analogue of a zero diagonal. The row and column
    positions need not coincide, so original observation IDs are used rather than
    matrix diagonal indices.
    """
    row_ids = np.asarray(row_ids)
    col_ids = np.asarray(col_ids)
    if row_ids.shape != (K.shape[0],) or col_ids.shape != (K.shape[1],):
        raise ValueError("row_ids and col_ids must align with the pair kernel")
    common, row_pos, col_pos = np.intersect1d(
        row_ids, col_ids, assume_unique=False, return_indices=True
    )
    if common.size == 0:
        return K.tocsr(), 0
    K = K.tolil(copy=True)
    K[row_pos, col_pos] = 0
    K = K.tocsr()
    K.eliminate_zeros()
    return K, int(common.size)


def kernel_matrix_sparse(
    coords,
    radius,
    coords_query=None,
    dtype=np.float64,
    normalisation="symmetric",
    zero_diagonal=True,
    row_ids=None,
    query_ids=None,
    exclude_identity_pairs=False,
):
    """Build a Wendland pair adjacency used by Tanpopo.

    Square kernels are zero-diagonal by default. Rectangular kernels may remove
    biological self-pairs by supplying original ``row_ids``/``query_ids`` and
    ``exclude_identity_pairs=True``. Identity removal occurs before degree
    normalisation, so excluded self-pairs cannot distort the remaining weights.

    ``normalisation='symmetric'`` uses D^-1/2 K D^-1/2 for square graphs and the
    analogous row/column degree normalisation for bipartite graphs.
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
    elif exclude_identity_pairs:
        if row_ids is None or query_ids is None:
            raise ValueError(
                "row_ids and query_ids are required when excluding rectangular identity pairs"
            )
        K, _ = _remove_identity_pairs(K, row_ids, query_ids)

    _warn_connectivity(K, radius, check_columns=not square)
    if normalisation == "symmetric":
        K = _symmetric_normalize(K) if square else _bipartite_normalize(K)
    elif normalisation != "none":
        raise ValueError("normalisation must be 'symmetric' or 'none'")

    return K.astype(dtype, copy=False)


def _edge_distances(K, row_coords, col_coords=None):
    """Return COO edge indices and Euclidean distances for nonzero pair weights."""
    C = K.tocoo(copy=False)
    row_coords = np.asarray(row_coords, dtype=float)
    col_coords = row_coords if col_coords is None else np.asarray(col_coords, dtype=float)
    if row_coords.shape[0] != K.shape[0] or col_coords.shape[0] != K.shape[1]:
        raise ValueError("Coordinate matrices must align with the pair kernel")
    delta = row_coords[C.row] - col_coords[C.col]
    distances = np.sqrt(np.sum(delta * delta, axis=1))
    return C, distances


def _bin_index(distances, bin_edges):
    idx = np.searchsorted(bin_edges, distances, side="right") - 1
    # Nonzero Wendland edges satisfy distance < radius, but clipping avoids
    # floating-point edge cases exactly at a supplied boundary.
    return np.clip(idx, 0, len(bin_edges) - 2)


def _bin_masses(K, row_coords, col_coords, bin_edges):
    C, distances = _edge_distances(K, row_coords, col_coords)
    idx = _bin_index(distances, bin_edges)
    masses = np.bincount(idx, weights=C.data, minlength=len(bin_edges) - 1).astype(float)
    return masses, C, idx, distances


def _mass_standardise(K, target_mass, eps=1e-12):
    mass = float(K.sum())
    if mass <= eps:
        raise ValueError("Spatial graph has zero total pair weight")
    return (K * (float(target_mass) / mass)).tocsr()


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


def _degree_summary(values, prefix):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values)) if values.size else 0.0
    cv = float(np.std(values) / mean) if values.size and mean > 0 else 0.0
    return {
        f"{prefix}_mean_weighted_degree": mean,
        f"{prefix}_median_weighted_degree": float(np.median(values)) if values.size else 0.0,
        f"{prefix}_weighted_degree_cv": cv,
        f"{prefix}_isolated_fraction": float(np.mean(values == 0)) if values.size else 0.0,
    }


def kernel_diagnostics(K):
    """Diagnostics for square or rectangular pair adjacencies."""
    K = K.tocsr()
    row_degree = np.asarray(K.sum(axis=1)).ravel()
    col_degree = np.asarray(K.sum(axis=0)).ravel()
    row = _degree_summary(row_degree, "row")
    col = _degree_summary(col_degree, "column")
    result = {
        "n_rows": int(K.shape[0]),
        "n_cols": int(K.shape[1]),
        "nnz": int(K.nnz),
        "total_edge_weight": float(K.sum()),
        **row,
        **col,
    }
    # Preserve historical square-graph diagnostic names as row aliases.
    result.update(
        {
            "isolated_fraction": row["row_isolated_fraction"],
            "mean_weighted_degree": row["row_mean_weighted_degree"],
            "median_weighted_degree": row["row_median_weighted_degree"],
            "weighted_degree_cv": row["row_weighted_degree_cv"],
        }
    )
    return result


def standardise_pair_measures(
    adjacencies,
    row_coords,
    col_coords,
    target_masses,
    radius,
    *,
    geometry_normalisation="distance",
    distance_bins=10,
    eps=1e-12,
):
    """Standardise square or bipartite pair measures with one implementation.

    ``target_masses`` defines the total pair mass after normalisation. Square
    workflows use ``n_s``; cross workflows use ``sqrt(n_target_s*n_neighbour_s)``.
    Under the existing corresponding sample weights, each biological specimen
    therefore contributes unit total pair mass.
    """
    if geometry_normalisation not in VALID_GEOMETRY_NORMALISATIONS:
        raise ValueError(
            "geometry_normalisation must be one of "
            f"{sorted(VALID_GEOMETRY_NORMALISATIONS)}"
        )
    if not adjacencies:
        raise ValueError("At least one pair adjacency is required")
    if not (
        len(adjacencies)
        == len(row_coords)
        == len(col_coords)
        == len(target_masses)
    ):
        raise ValueError("Pair adjacencies, coordinates and target masses must align")

    row_coords = [np.asarray(xy, dtype=float) for xy in row_coords]
    col_coords = [np.asarray(xy, dtype=float) for xy in col_coords]
    target_masses = np.asarray(target_masses, dtype=float)
    if np.any(~np.isfinite(target_masses)) or np.any(target_masses <= 0):
        raise ValueError("target_masses must be finite and positive")

    diagnostics = []
    if geometry_normalisation == "distance":
        distance_bins = int(distance_bins)
        if distance_bins < 1:
            raise ValueError("distance_bins must be a positive integer")
        bin_edges = np.linspace(0.0, float(radius), distance_bins + 1)
        cached = [
            _bin_masses(K, row_xy, col_xy, bin_edges)
            for K, row_xy, col_xy in zip(adjacencies, row_coords, col_coords)
        ]
        masses = np.vstack([x[0] for x in cached])
        common = np.all(masses > eps, axis=0)
        if not np.any(common):
            raise ValueError(
                "No distance bin has positive pair weight in every sample; increase "
                "--radius, reduce --distance-bins, or use --geometry-normalisation mass."
            )

        profiles = np.zeros_like(masses, dtype=float)
        for i, row in enumerate(masses):
            profiles[i, common] = row[common] / float(row[common].sum())
        reference = profiles.mean(axis=0)
        reference[~common] = 0.0
        reference /= reference.sum()

        outputs = []
        for K, target_mass, (raw_bin_mass, C, idx, distances) in zip(
            adjacencies, target_masses, cached
        ):
            keep = common[idx]
            scale = np.zeros(distance_bins, dtype=float)
            scale[common] = target_mass * reference[common] / raw_bin_mass[common]
            data = C.data * scale[idx]
            data[~keep] = 0.0
            out = sp.csr_matrix((data, (C.row, C.col)), shape=K.shape, dtype=K.dtype)
            out.eliminate_zeros()
            outputs.append(out)
            diagnostics.append(
                {
                    "raw_pair_mass": float(K.sum()),
                    "pair_mass": float(out.sum()),
                    "target_pair_mass": float(target_mass),
                    "distance_bin_mass_raw": raw_bin_mass.copy(),
                    "distance_bin_mass": np.bincount(
                        idx[keep], weights=data[keep], minlength=distance_bins
                    ).astype(float),
                    "distance_bin_edges": bin_edges.copy(),
                    "distance_reference_weights": reference.copy(),
                    "common_distance_bins": common.copy(),
                    "weighted_distance_quantiles": _weighted_quantiles(
                        distances[keep], data[keep], [0.1, 0.5, 0.9]
                    ),
                }
            )
    else:
        outputs = []
        for K, row_xy, col_xy, target_mass in zip(
            adjacencies, row_coords, col_coords, target_masses
        ):
            raw_mass = float(K.sum())
            if geometry_normalisation == "mass":
                K = _mass_standardise(K, target_mass)
            C, distances = _edge_distances(K, row_xy, col_xy)
            outputs.append(K)
            diagnostics.append(
                {
                    "raw_pair_mass": raw_mass,
                    "pair_mass": float(K.sum()),
                    "target_pair_mass": float(target_mass),
                    "weighted_distance_quantiles": _weighted_quantiles(
                        distances, C.data, [0.1, 0.5, 0.9]
                    ),
                }
            )

    for K, diag in zip(outputs, diagnostics):
        diag.update(kernel_diagnostics(K))
        diag["geometry_normalisation"] = geometry_normalisation
    return outputs, diagnostics


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
    """Build comparable square spatial operators for one or more samples."""
    if spatial_statistic not in VALID_SPATIAL_STATISTICS:
        raise ValueError(
            f"spatial_statistic must be one of {sorted(VALID_SPATIAL_STATISTICS)}"
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
    adjacency, diagnostics = standardise_pair_measures(
        adjacency,
        coords,
        coords,
        [len(xy) for xy in coords],
        radius,
        geometry_normalisation=geometry_normalisation,
        distance_bins=distance_bins,
    )
    for diag in diagnostics:
        diag["spatial_statistic"] = spatial_statistic

    if spatial_statistic == "variogram":
        operators = [
            adjacency_to_variogram(K).astype(dtype, copy=False) for K in adjacency
        ]
        return operators, diagnostics
    return [K.astype(dtype, copy=False) for K in adjacency], diagnostics


def cross_mark_correlation_kernels(
    target_coords,
    neighbour_coords,
    radius,
    *,
    target_ids=None,
    neighbour_ids=None,
    geometry_normalisation="distance",
    distance_bins=10,
    graph_normalisation="symmetric",
    dtype=np.float64,
):
    """Build geometry-standardised bipartite target-neighbour pair kernels.

    Biological self-pairs are removed using original observation IDs before
    row/column degree normalisation. The geometry target mass for sample ``s`` is
    ``sqrt(n_target_s * n_neighbour_s)``, matching cross-program sample weighting.
    """
    target_coords = [np.asarray(xy, dtype=dtype) for xy in target_coords]
    neighbour_coords = [np.asarray(xy, dtype=dtype) for xy in neighbour_coords]
    if not target_coords or len(target_coords) != len(neighbour_coords):
        raise ValueError("Target and neighbour coordinate lists must be non-empty and aligned")
    if target_ids is None:
        target_ids = [np.arange(len(xy), dtype=np.int64) for xy in target_coords]
    if neighbour_ids is None:
        # With no shared original ID space, no identities can safely be inferred.
        neighbour_ids = [-(np.arange(len(xy), dtype=np.int64) + 1) for xy in neighbour_coords]
    if len(target_ids) != len(target_coords) or len(neighbour_ids) != len(target_coords):
        raise ValueError("Target/neighbour ID lists must align with coordinate lists")

    adjacency = []
    removed_identity_counts = []
    for target_xy, neighbour_xy, row_ids, col_ids in zip(
        target_coords, neighbour_coords, target_ids, neighbour_ids
    ):
        row_ids = np.asarray(row_ids)
        col_ids = np.asarray(col_ids)
        removed = int(np.intersect1d(row_ids, col_ids).size)
        K = kernel_matrix_sparse(
            target_xy,
            radius,
            coords_query=neighbour_xy,
            dtype=dtype,
            normalisation=graph_normalisation,
            zero_diagonal=False,
            row_ids=row_ids,
            query_ids=col_ids,
            exclude_identity_pairs=True,
        )
        adjacency.append(K)
        removed_identity_counts.append(removed)

    target_masses = [
        np.sqrt(float(len(target_xy) * len(neighbour_xy)))
        for target_xy, neighbour_xy in zip(target_coords, neighbour_coords)
    ]
    adjacency, diagnostics = standardise_pair_measures(
        adjacency,
        target_coords,
        neighbour_coords,
        target_masses,
        radius,
        geometry_normalisation=geometry_normalisation,
        distance_bins=distance_bins,
    )
    for diag, target_xy, neighbour_xy, removed in zip(
        diagnostics, target_coords, neighbour_coords, removed_identity_counts
    ):
        diag.update(
            {
                "spatial_statistic": "mark_correlation",
                "n_target": int(len(target_xy)),
                "n_neighbour": int(len(neighbour_xy)),
                "removed_identity_pairs": int(removed),
                "removed_identity_pair_weight_raw": float(removed),
            }
        )
    return [K.astype(dtype, copy=False) for K in adjacency], diagnostics
