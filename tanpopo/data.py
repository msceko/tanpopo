from dataclasses import dataclass
from functools import cached_property

import numpy as np
import scanpy as sc
import scipy.sparse as sp

from tanpopo.covariates import compute_covariates
from tanpopo.kernel import kernel_matrix_sparse
from tanpopo.utils import as_list, get_counts_matrix


def filter_anndata(adata, min_counts=10, min_spot_fraction=0.01):
    if min_counts:
        sc.pp.filter_genes(adata, min_counts=min_counts)
    if min_spot_fraction:
        min_spots = int(min_spot_fraction * len(adata.obs))
        sc.pp.filter_genes(adata, min_cells=min_spots)


def transform_anndata(adata, target_sum=1e4, transform=None):
    if target_sum:
        sc.pp.normalize_total(adata, target_sum=target_sum)
    if transform == "log1p":
        sc.pp.log1p(adata)
    elif transform == "sqrt":
        sc.pp.sqrt(adata)


def preprocess_anndata(
    adata,
    target_sum=1e4,
    transform=None,
    min_counts=10,
    min_spot_fraction=0.01,
    covariates=None,
    layer=None,
    sparse=True,
):
    """
    Filter genes, compute covariates and transform anndata
    """
    filter_anndata(adata, min_counts, min_spot_fraction)
    if covariates:
        compute_covariates(adata, covariates, layer)
    transform_anndata(adata, target_sum, transform)


def get_spatial_from_anndata(adata, layer=None, spatial_key="spatial", sparse=True):
    X = get_counts_matrix(adata, sparse, layer)
    coords = adata.obsm[spatial_key].astype(np.float64)
    covariates = adata.obsm.get("tanpopo_covariates", None)
    return X, coords, covariates


@dataclass
class Groups:
    """
    Contiguous spot groups.

    offsets and lengths refer to the current row order.
    """

    offsets: np.ndarray
    lengths: np.ndarray

    @classmethod
    def single(cls, n):
        return cls(
            offsets=np.array([0], dtype=np.int64),
            lengths=np.array([n], dtype=np.int64),
        )

    @classmethod
    def from_labels(cls, labels):
        labels = np.asarray(labels)
        _, codes = np.unique(labels, return_inverse=True)
        order = np.argsort(codes, kind="stable")
        grouped = codes[order]

        starts = np.flatnonzero(np.r_[True, grouped[1:] != grouped[:-1]])
        stops = np.r_[starts[1:], len(labels)]
        lengths = stops - starts

        return order, cls(starts.astype(np.int64), lengths.astype(np.int64))

    @property
    def n_groups(self):
        return len(self.offsets)


@dataclass
class SampleData:
    W: sp.csr_matrix
    K: sp.csr_matrix
    inv_order: np.ndarray
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


def prepare_sample(W, coords, radius, labels=None, covariates=None, dtype=np.float64):
    """
    Reorder one sample by labels, build its sparse kernel, and keep inverse order.
    """
    n = W.shape[0]

    if labels is None:
        order = np.arange(n, dtype=np.int64)
        groups = Groups.single(n)
    else:
        order, groups = Groups.from_labels(labels)

    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n, dtype=np.int64)

    W = sp.csr_matrix(W[order], dtype=dtype)
    coords = np.asarray(coords, dtype=dtype)[order]

    if covariates is not None:
        covariates = np.asarray(covariates, dtype=dtype)
        if covariates.ndim == 1:
            covariates = covariates[:, None]
        covariates = covariates[order]

    K = kernel_matrix_sparse(coords, radius).astype(dtype)

    return SampleData(W=W, K=K, inv_order=inv, labels_groups=groups, covariates=covariates)


def prepare_samples(W, coords, radius, labels=None, covariates=None, dtype=np.float64):
    W = as_list(W)
    coords = as_list(coords)
    labels = [None] * len(W) if labels is None else as_list(labels)
    covariates = [None] * len(W) if covariates is None else as_list(covariates)

    return [
        prepare_sample(w, xy, radius, lab, cov, dtype=dtype)
        for w, xy, lab, cov in zip(W, coords, labels, covariates)
    ]


def concatenate_samples(samples):
    """
    Concatenate reordered samples and build block-diagonal K.
    """
    W = sp.vstack([s.W for s in samples], format="csr")
    K = sp.block_diag([s.K for s in samples], format="csr")

    offsets = np.r_[0, np.cumsum([s.n_spots for s in samples[:-1]])].astype(np.int64)

    sample_offsets = []
    sample_lengths = []
    label_offsets = []
    label_lengths = []

    covs = []
    has_cov = any(s.covariates is not None for s in samples)
    n_cov = next((s.covariates.shape[1] for s in samples if s.covariates is not None), 0)

    for off, s in zip(offsets, samples):
        off = int(off)
        n = s.n_spots

        sample_offsets.append(off)
        sample_lengths.append(n)

        label_offsets.extend((off + s.labels_groups.offsets).tolist())
        label_lengths.extend(s.labels_groups.lengths.tolist())

        if has_cov:
            if s.covariates is None:
                covs.append(np.zeros((n, n_cov)))
            else:
                covs.append(s.covariates)

    sample_groups = Groups(np.asarray(sample_offsets), np.asarray(sample_lengths))
    label_groups = Groups(np.asarray(label_offsets), np.asarray(label_lengths))
    covariates = None if not has_cov else np.vstack(covs)

    return W, K, sample_groups, label_groups, covariates
