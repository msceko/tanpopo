from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import scipy.sparse as sp

from tanpopo.covariates import compute_covariates
from tanpopo.kernel import cross_mark_correlation_kernels, spatial_statistic_kernels
from tanpopo.utils import as_list, get_counts_matrix, none_to_list, pd_dtype


def subset_by_labels(adata, label_key, labels, mode):
    if label_key not in adata.obs:
        raise ValueError(f"Valid label_key is required when using {mode} labels.")
    labels = np.array(labels, pd_dtype(adata.obs[label_key]))
    subset = adata.obs[label_key].isin(labels)
    if mode == "exclude":
        subset = ~subset
    adata._inplace_subset_obs(subset)


def _gene_filter_mask(X, min_counts=10, min_spot_fraction=None):
    n = X.shape[0]
    keep = np.ones(X.shape[1], dtype=bool)
    if min_counts:
        totals = np.asarray(X.sum(axis=0)).ravel()
        keep &= totals >= min_counts
    if min_spot_fraction:
        if sp.issparse(X):
            detected = np.asarray((X > 0).sum(axis=0)).ravel()
        else:
            detected = np.sum(np.asarray(X) > 0, axis=0)
        keep &= detected >= int(np.ceil(min_spot_fraction * n))
    return keep


def filter_anndata(
    adata,
    min_counts=10,
    min_spot_fraction=None,
    include=None,
    exclude=None,
    label_key=None,
    layer=None,
):
    if include is not None:
        subset_by_labels(adata, label_key, include, mode="include")
    elif exclude is not None:
        subset_by_labels(adata, label_key, exclude, mode="exclude")
    X = get_counts_matrix(adata, sparse=True, layer=layer)
    keep = _gene_filter_mask(X, min_counts, min_spot_fraction)
    adata._inplace_subset_var(keep)


def _normalise_matrix(X, target_sum):
    sparse = sp.issparse(X)
    X = X.tocsr().astype(float, copy=True) if sparse else np.asarray(X, dtype=float).copy()
    totals = np.asarray(X.sum(axis=1)).ravel() if sparse else X.sum(axis=1)
    scale = np.divide(
        target_sum,
        totals,
        out=np.ones_like(totals, dtype=float),
        where=totals > 0,
    )
    if sparse:
        return sp.diags(scale, format="csr") @ X
    return X * scale[:, None]


def _transform_matrix(X, transform):
    sparse = sp.issparse(X)
    if transform is None:
        return X
    if transform == "log1p":
        if sparse:
            X = X.copy()
            X.data = np.log1p(X.data)
            return X
        return np.log1p(X)
    if transform == "sqrt":
        if sparse:
            X = X.copy()
            X.data = np.sqrt(X.data)
            return X
        return np.sqrt(X)
    raise ValueError("transform must be None, 'log1p', or 'sqrt'")


def transform_anndata(adata, target_sum=None, transform=None, layer=None):
    if target_sum in {0, 0.0}:
        target_sum = None
    X = adata.X if layer is None else adata.layers[layer]
    if target_sum is not None:
        X = _normalise_matrix(X, float(target_sum))
    X = _transform_matrix(X, transform)
    if layer is None:
        adata.X = X
    else:
        adata.layers[layer] = X


def preprocess_anndata(
    adata,
    target_sum=None,
    transform=None,
    min_counts=10,
    min_spot_fraction=None,
    covariates=None,
    include=None,
    exclude=None,
    label_key=None,
    layer=None,
):
    filter_anndata(
        adata,
        min_counts,
        min_spot_fraction,
        include,
        exclude,
        label_key,
        layer,
    )
    if covariates:
        compute_covariates(adata, covariates, layer)
    transform_anndata(adata, target_sum, transform, layer)


def preprocess_anndata_shared_genes(adata_list, **kwargs):
    for adata in adata_list:
        filter_anndata(
            adata,
            kwargs.get("min_counts", 10),
            kwargs.get("min_spot_fraction"),
            kwargs.get("include"),
            kwargs.get("exclude"),
            kwargs.get("label_key"),
            kwargs.get("layer"),
        )
    shared = set(adata_list[0].var_names)
    for adata in adata_list[1:]:
        shared &= set(adata.var_names)
    genes = [g for g in adata_list[0].var_names if g in shared]
    for i, adata in enumerate(adata_list):
        adata_list[i] = adata[:, genes].copy()
        if kwargs.get("covariates"):
            compute_covariates(adata_list[i], kwargs["covariates"], kwargs.get("layer"))
        transform_anndata(
            adata_list[i], kwargs.get("target_sum"), kwargs.get("transform"), kwargs.get("layer")
        )


def get_spatial_from_anndata(adata, layer=None, spatial_key="spatial", sparse=True):
    X = get_counts_matrix(adata, sparse=sparse, layer=layer)
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    covariates = adata.obsm.get("tanpopo_covariates", None)
    return X, coords, covariates


@dataclass
class Groups:
    offsets: np.ndarray
    lengths: np.ndarray

    @classmethod
    def single(cls, n):
        return cls(np.array([0], dtype=np.int64), np.array([n], dtype=np.int64))

    @classmethod
    def from_labels(cls, labels):
        labels = np.asarray(labels)
        _, codes = np.unique(labels, return_inverse=True)
        order = np.argsort(codes, kind="stable")
        grouped = codes[order]
        starts = np.flatnonzero(np.r_[True, grouped[1:] != grouped[:-1]])
        stops = np.r_[starts[1:], len(labels)]
        return order, cls(starts.astype(np.int64), (stops - starts).astype(np.int64))

    @property
    def n_groups(self):
        return len(self.offsets)


@dataclass
class SampleData:
    W: sp.csr_matrix
    K: sp.csr_matrix
    inv_order: np.ndarray
    obs_idx: np.ndarray
    labels_groups: Groups
    covariates: np.ndarray | None = None
    geometry_diagnostics: dict | None = None

    @property
    def n_spots(self):
        return self.W.shape[0]

    @property
    def n_genes(self):
        return self.W.shape[1]

    @cached_property
    def Wcsc(self):
        return self.W.tocsc(copy=False)


@dataclass
class CrossSampleData:
    """Prepared target/neighbour data for one biological sample."""

    W_target: sp.csr_matrix
    W_neighbour: sp.csr_matrix
    K: sp.csr_matrix
    target_idx: np.ndarray
    neighbour_idx: np.ndarray
    covariates_target: np.ndarray | None = None
    covariates_neighbour: np.ndarray | None = None
    geometry_diagnostics: dict | None = None

    @property
    def n_target(self):
        return self.W_target.shape[0]

    @property
    def n_neighbour(self):
        return self.W_neighbour.shape[0]

    @property
    def n_genes(self):
        return self.W_target.shape[1]


def _prepare_sample_arrays(
    W,
    coords,
    labels=None,
    covariates=None,
    mask=None,
    dtype=np.float64,
):
    """Apply a hard mask and label ordering without constructing spatial weights."""
    W = sp.csr_matrix(W, dtype=dtype)
    coords = np.asarray(coords, dtype=dtype)
    n = W.shape[0]
    if coords.shape[0] != n:
        raise ValueError("W and coords must contain the same number of spots")
    if mask is None:
        mask = np.ones(n, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (n,) or not np.any(mask):
            raise ValueError("mask must select at least one spot")
    spot_idx = np.flatnonzero(mask)
    if labels is None:
        order = np.arange(spot_idx.size, dtype=np.int64)
        groups = Groups.single(spot_idx.size)
    else:
        order, groups = Groups.from_labels(np.asarray(labels)[spot_idx])
    idx = spot_idx[order]
    inv = np.empty(order.size, dtype=np.int64)
    inv[order] = np.arange(order.size, dtype=np.int64)
    W_sub = sp.csr_matrix(W[idx], dtype=dtype)
    coords_sub = coords[idx]
    cov = None
    if covariates is not None:
        cov = np.asarray(covariates, dtype=dtype)[idx]
        if cov.ndim == 1:
            cov = cov[:, None]
    return W_sub, coords_sub, inv, spot_idx, groups, cov


def prepare_sample(
    W,
    coords,
    radius,
    labels=None,
    covariates=None,
    mask=None,
    dtype=np.float64,
    graph_normalisation="symmetric",
    spatial_statistic="mark_correlation",
    geometry_normalisation="distance",
    distance_bins=10,
):
    """Prepare one hard-masked sample and its geometry-standardised operator."""
    return prepare_samples(
        [W],
        [coords],
        radius,
        labels=None if labels is None else [labels],
        covariates=None if covariates is None else [covariates],
        masks=None if mask is None else [mask],
        dtype=dtype,
        graph_normalisation=graph_normalisation,
        spatial_statistic=spatial_statistic,
        geometry_normalisation=geometry_normalisation,
        distance_bins=distance_bins,
    )[0]


def prepare_samples(
    W,
    coords,
    radius,
    labels=None,
    covariates=None,
    masks=None,
    dtype=np.float64,
    graph_normalisation="symmetric",
    spatial_statistic="mark_correlation",
    geometry_normalisation="distance",
    distance_bins=10,
):
    """Prepare matched samples and build their spatial operators jointly.

    Joint construction is required for ``geometry_normalisation='distance'``:
    the pair-distance target is estimated once from all included biological
    samples and then imposed on every sample before model fitting.
    """
    W = as_list(W)
    coords = as_list(coords)
    if len(coords) != len(W):
        raise ValueError("W and coords must contain the same number of samples")
    labels = none_to_list(labels, len(W))
    covariates = none_to_list(covariates, len(W))
    masks = none_to_list(masks, len(W))

    prepared = [
        _prepare_sample_arrays(w, xy, lab, cov, mask, dtype)
        for w, xy, lab, cov, mask in zip(W, coords, labels, covariates, masks)
    ]
    coords_sub = [item[1] for item in prepared]
    kernels, diagnostics = spatial_statistic_kernels(
        coords_sub,
        radius,
        spatial_statistic=spatial_statistic,
        geometry_normalisation=geometry_normalisation,
        distance_bins=distance_bins,
        graph_normalisation=graph_normalisation,
        dtype=dtype,
    )
    return [
        SampleData(
            W_sub,
            K,
            inv,
            spot_idx,
            groups,
            cov,
            geometry_diagnostics=diag,
        )
        for (W_sub, _, inv, spot_idx, groups, cov), K, diag in zip(
            prepared, kernels, diagnostics
        )
    ]

def _prepare_cross_sample_arrays(
    W,
    coords,
    target_mask,
    neighbour_mask,
    covariates=None,
    dtype=np.float64,
):
    """Subset one sample into target/neighbour matrices without building pair weights."""
    W = sp.csr_matrix(W, dtype=dtype)
    coords = np.asarray(coords, dtype=dtype)
    n = W.shape[0]
    if coords.shape[0] != n:
        raise ValueError("W and coords must contain the same number of spots")
    target_mask = np.asarray(target_mask, dtype=bool)
    neighbour_mask = np.asarray(neighbour_mask, dtype=bool)
    if target_mask.shape != (n,) or neighbour_mask.shape != (n,):
        raise ValueError("target_mask and neighbour_mask must contain one value per spot")
    if not np.any(target_mask) or not np.any(neighbour_mask):
        raise ValueError("target_mask and neighbour_mask must each select at least one spot")

    target_idx = np.flatnonzero(target_mask)
    neighbour_idx = np.flatnonzero(neighbour_mask)
    W_target = sp.csr_matrix(W[target_idx], dtype=dtype)
    W_neighbour = sp.csr_matrix(W[neighbour_idx], dtype=dtype)
    coords_target = coords[target_idx]
    coords_neighbour = coords[neighbour_idx]

    cov_target = cov_neighbour = None
    if covariates is not None:
        covariates = np.asarray(covariates, dtype=dtype)
        if covariates.ndim == 1:
            covariates = covariates[:, None]
        if covariates.shape[0] != n:
            raise ValueError("covariates must contain one row per spot")
        cov_target = covariates[target_idx]
        cov_neighbour = covariates[neighbour_idx]

    return (
        W_target,
        W_neighbour,
        coords_target,
        coords_neighbour,
        target_idx,
        neighbour_idx,
        cov_target,
        cov_neighbour,
    )


def prepare_cross_sample(
    W,
    coords,
    radius,
    target_mask,
    neighbour_mask,
    covariates=None,
    dtype=np.float64,
    graph_normalisation="symmetric",
    geometry_normalisation="distance",
    distance_bins=10,
):
    """Prepare one geometry-standardised bipartite target-neighbour sample."""
    return prepare_cross_samples(
        [W],
        [coords],
        radius,
        [target_mask],
        [neighbour_mask],
        covariates=None if covariates is None else [covariates],
        dtype=dtype,
        graph_normalisation=graph_normalisation,
        geometry_normalisation=geometry_normalisation,
        distance_bins=distance_bins,
    )[0]


def prepare_cross_samples(
    W,
    coords,
    radius,
    target_masks,
    neighbour_masks,
    covariates=None,
    dtype=np.float64,
    graph_normalisation="symmetric",
    geometry_normalisation="distance",
    distance_bins=10,
):
    """Prepare matched target-neighbour data with one shared pair-distance measure."""
    W = as_list(W)
    coords = as_list(coords)
    target_masks = as_list(target_masks)
    neighbour_masks = as_list(neighbour_masks)
    covariates = none_to_list(covariates, len(W))
    lengths = {len(W), len(coords), len(target_masks), len(neighbour_masks), len(covariates)}
    if lengths != {len(W)}:
        raise ValueError("W, coords, masks and covariates must contain the same number of samples")

    prepared = [
        _prepare_cross_sample_arrays(w, xy, target, neighbour, cov, dtype)
        for w, xy, target, neighbour, cov in zip(
            W, coords, target_masks, neighbour_masks, covariates
        )
    ]
    target_coords = [item[2] for item in prepared]
    neighbour_coords = [item[3] for item in prepared]
    target_ids = [item[4] for item in prepared]
    neighbour_ids = [item[5] for item in prepared]
    kernels, diagnostics = cross_mark_correlation_kernels(
        target_coords,
        neighbour_coords,
        radius,
        target_ids=target_ids,
        neighbour_ids=neighbour_ids,
        geometry_normalisation=geometry_normalisation,
        distance_bins=distance_bins,
        graph_normalisation=graph_normalisation,
        dtype=dtype,
    )

    return [
        CrossSampleData(
            W_target=W_target,
            W_neighbour=W_neighbour,
            K=K,
            target_idx=target_idx,
            neighbour_idx=neighbour_idx,
            covariates_target=cov_target,
            covariates_neighbour=cov_neighbour,
            geometry_diagnostics=diag,
        )
        for (
            W_target,
            W_neighbour,
            _,
            _,
            target_idx,
            neighbour_idx,
            cov_target,
            cov_neighbour,
        ), K, diag in zip(prepared, kernels, diagnostics)
    ]

def _stack_cross_covariates(samples, side):
    attr = f"covariates_{side}"
    values = [getattr(sample, attr) for sample in samples]
    if not any(value is not None for value in values):
        return None
    n_cov = next(value.shape[1] for value in values if value is not None)
    rows_attr = "n_target" if side == "target" else "n_neighbour"
    return np.vstack(
        [
            np.zeros((getattr(sample, rows_attr), n_cov)) if value is None else value
            for sample, value in zip(samples, values)
        ]
    )


def concatenate_cross_samples(samples):
    """Concatenate cross-sample matrices while preserving sample-wise centering."""
    if not samples:
        raise ValueError("At least one cross sample is required")
    n_genes = samples[0].n_genes
    if any(sample.n_genes != n_genes for sample in samples):
        raise ValueError("All cross samples must contain the same genes")
    W_target = sp.vstack([sample.W_target for sample in samples], format="csr")
    W_neighbour = sp.vstack([sample.W_neighbour for sample in samples], format="csr")
    target_lengths = np.asarray([sample.n_target for sample in samples], dtype=np.int64)
    neighbour_lengths = np.asarray([sample.n_neighbour for sample in samples], dtype=np.int64)
    target_offsets = np.r_[0, np.cumsum(target_lengths[:-1])].astype(np.int64)
    neighbour_offsets = np.r_[0, np.cumsum(neighbour_lengths[:-1])].astype(np.int64)
    target_groups = Groups(target_offsets, target_lengths)
    neighbour_groups = Groups(neighbour_offsets, neighbour_lengths)

    cov_target = _stack_cross_covariates(samples, "target")
    cov_neighbour = _stack_cross_covariates(samples, "neighbour")
    return (
        W_target,
        W_neighbour,
        target_groups,
        neighbour_groups,
        cov_target,
        cov_neighbour,
    )


def concatenate_samples(samples):
    W = sp.vstack([s.W for s in samples], format="csr")
    K = sp.block_diag([s.K for s in samples], format="csr")
    offsets = np.r_[0, np.cumsum([s.n_spots for s in samples[:-1]])].astype(np.int64)
    sample_groups = Groups(offsets, np.asarray([s.n_spots for s in samples], dtype=np.int64))
    label_offsets, label_lengths = [], []
    covs = []
    has_cov = any(s.covariates is not None for s in samples)
    n_cov = next((s.covariates.shape[1] for s in samples if s.covariates is not None), 0)
    for off, s in zip(offsets, samples):
        label_offsets.extend((off + s.labels_groups.offsets).tolist())
        label_lengths.extend(s.labels_groups.lengths.tolist())
        if has_cov:
            covs.append(
                np.zeros((s.n_spots, n_cov)) if s.covariates is None else s.covariates
            )
    label_groups = Groups(np.asarray(label_offsets), np.asarray(label_lengths))
    covariates = None if not has_cov else np.vstack(covs)
    return W, K, sample_groups, label_groups, covariates
