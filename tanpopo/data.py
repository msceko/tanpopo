from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import scipy.sparse as sp

from tanpopo.covariates import compute_covariates
from tanpopo.kernel import kernel_matrix_sparse
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

    @property
    def n_spots(self):
        return self.W.shape[0]

    @property
    def n_genes(self):
        return self.W.shape[1]

    @cached_property
    def Wcsc(self):
        return self.W.tocsc(copy=False)


def prepare_sample(
    W,
    coords,
    radius,
    labels=None,
    covariates=None,
    mask=None,
    dtype=np.float64,
    graph_normalisation="symmetric",
):
    """Hard-mask a sample, order by labels, and build a zero-diagonal graph."""
    W = sp.csr_matrix(W, dtype=dtype)
    coords = np.asarray(coords, dtype=dtype)
    n = W.shape[0]
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
    K = kernel_matrix_sparse(
        coords_sub,
        radius,
        dtype=dtype,
        normalisation=graph_normalisation,
        zero_diagonal=True,
    )
    return SampleData(W_sub, K, inv, spot_idx, groups, cov)


def prepare_samples(
    W,
    coords,
    radius,
    labels=None,
    covariates=None,
    masks=None,
    dtype=np.float64,
    graph_normalisation="symmetric",
):
    W = as_list(W)
    coords = as_list(coords)
    labels = none_to_list(labels, len(W))
    covariates = none_to_list(covariates, len(W))
    masks = none_to_list(masks, len(W))
    return [
        prepare_sample(
            w,
            xy,
            radius,
            lab,
            cov,
            mask,
            dtype,
            graph_normalisation,
        )
        for w, xy, lab, cov, mask in zip(W, coords, labels, covariates, masks)
    ]


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
