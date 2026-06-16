from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

from tanpopo.data import Groups
from tanpopo.projection import SpotProjector
from tanpopo.utils import center_rows


class ProjectedKernel:
    """
    Efficient projected spatial kernel: S = P K P
    where P is a SpotProjector (centering + optional covariate removal).
    The diagonal computation is handled separately by gene_spatial_variance.
    """

    def __init__(self, K, projector, dtype=np.float64):
        self.K = K.tocsr().astype(dtype)
        self.projector = projector
        self.dtype = dtype
        self.n = K.shape[0]

    def apply(self, X, residualise=True):
        X = self.projector.apply(X, residualise=residualise)
        X = self.K @ X
        X = self.projector.adjoint(X, residualise=residualise)
        return X

    def gene_spatial_variance(self, W, block_size=None):
        return gene_spatial_variance(W, self.K, self.projector, block_size, dtype=self.dtype)


@dataclass
class SpotOperatorSpec:
    """
    Declarative specification for a projected spot operator.

    centering must be one of 'none', 'sample', or 'label'. Calling build()
    constructs the corresponding SpotProjector and ProjectedKernel, replacing
    the groups_for_operator + ProjectedKernel(...) pattern scattered across
    the original fit methods.
    """

    centering: str

    def build(self, K, sample_groups, label_groups, covariates, tol, dtype):
        groups = groups_for_operator(self.centering, K.shape[0], sample_groups, label_groups)
        projector = SpotProjector(
            K.shape[0],
            groups=groups,
            covariates=covariates,
            tol=tol,
            dtype=dtype,
        )
        return ProjectedKernel(K, projector, dtype=dtype)


class GeneKernel:
    """
    Gene-space linear operator: G = H_g D^{-alpha} W^T S W D^{-alpha} H_g

    Responsible only for matrix-vector products used inside the eigensolver.
    The lift step (mapping eigenvectors to spot/gene space) is handled
    separately by ModeLift.
    """

    def __init__(self, W, S, gene_scale=None, gene_center=True, dtype=np.float64):
        self.W = W.tocsr() if sp.issparse(W) else sp.csr_matrix(W, dtype=dtype)
        self.Wcsc = self.W.tocsc(copy=False)
        self.S = S
        self.gene_scale = gene_scale
        self.gene_center = gene_center
        self.dtype = dtype
        self.shape = (self.W.shape[1], self.W.shape[1])

    def _prepare(self, X):
        X = np.asarray(X, dtype=self.dtype)
        if X.ndim == 1:
            X = X[:, None]

        if self.gene_center:
            X = center_rows(X)

        if self.gene_scale is not None:
            X = self.gene_scale[:, None] * X

        return X

    def _finish(self, Y):
        if self.gene_scale is not None:
            Y = self.gene_scale[:, None] * Y

        if self.gene_center:
            Y = center_rows(Y)

        return Y

    def apply(self, X):
        was_vector = np.asarray(X).ndim == 1

        X = self._prepare(X)
        U = self.W @ X
        U = self.S.apply(U)
        Y = self.Wcsc.T @ U
        Y = self._finish(Y)

        return Y[:, 0] if was_vector else Y


class SumGeneOperator:
    """
    Weighted sum of gene operators: G = sum_i weight_i G_i
    """

    def __init__(self, terms, dtype=np.float64):
        self.terms = [(float(w), op) for w, op in terms if float(w) != 0.0]
        if len(self.terms) == 0:
            raise ValueError("SumGeneOperator received no nonzero terms.")
        self.dtype = dtype
        self.shape = self.terms[0][1].shape

    def apply(self, X):
        was_vector = np.asarray(X).ndim == 1
        X2 = np.asarray(X, dtype=self.dtype)
        if was_vector:
            X2 = X2[:, None]

        Y = np.zeros((self.shape[0], X2.shape[1]), dtype=self.dtype)
        for w, op in self.terms:
            Y += w * op.apply(X2)

        return Y[:, 0] if was_vector else Y

    def as_scipy(self):
        return LinearOperator(
            self.shape,
            matvec=lambda x: self.apply(x),
            matmat=lambda X: self.apply(X),
            dtype=self.dtype,
        )


class SampleOperatorBuilder:
    """
    Builds per-sample (coefficient, GeneKernel) pairs with optional signed
    coefficients and sample weighting.

    Extracted from SpatialGeneSampleCombinedKPCA so the two sample-wise model
    classes can share the logic without an inheritance relationship.
    """

    def __init__(
        self,
        spot_operator,
        sample_weighting,
        normalise_by,
        alpha,
        gene_center,
        eps,
        covariates_tol,
        block_size,
        dtype,
    ):
        self.spot_operator = spot_operator
        self.sample_weighting = sample_weighting
        self.normalise_by = normalise_by
        self.alpha = alpha
        self.gene_center = gene_center
        self.eps = eps
        self.covariates_tol = covariates_tol
        self.block_size = block_size
        self.dtype = dtype

    def build(self, samples, signed_coefficients=None):
        """
        Build per-sample operators and diagnostics.

        Returns (ops, raw_diags, coeff) where ops is a list of
        (coefficient, GeneKernel) pairs ready for SumGeneOperator.
        """
        raw_S_ops = []
        raw_diags = []

        for s in samples:
            spec = SpotOperatorSpec(self.spot_operator)
            S = spec.build(
                s.K,
                sample_groups=Groups.single(s.n_spots),
                label_groups=s.labels_groups,
                covariates=s.covariates,
                tol=self.covariates_tol,
                dtype=self.dtype,
            )
            diag = np.maximum(S.gene_spatial_variance(s.W, self.block_size), self.eps)
            raw_S_ops.append(S)
            raw_diags.append(diag)

        raw_diags = np.vstack(raw_diags)

        weights = self._sample_weights(samples, raw_diags)
        coeff = weights if signed_coefficients is None else weights * signed_coefficients

        if self.normalise_by == "pooled":
            ref = np.sum(np.abs(coeff)[:, None] * raw_diags, axis=0)
            pooled_scale = gene_scale_from_diag(ref, self.alpha, self.eps)
        elif self.normalise_by == "sample":
            pooled_scale = None
        else:
            raise ValueError("normalise_by must be 'sample' or 'pooled'")

        ops = []
        for i, s in enumerate(samples):
            scale = (
                gene_scale_from_diag(raw_diags[i], self.alpha, self.eps)
                if self.normalise_by == "sample"
                else pooled_scale
            )
            Gs = GeneKernel(s.W, raw_S_ops[i], scale, self.gene_center, self.dtype)
            ops.append((coeff[i], Gs))

        return ops, raw_diags, coeff

    def _sample_weights(self, samples, diags):
        if self.sample_weighting == "none":
            return np.ones(len(samples))
        if self.sample_weighting == "n_spots":
            return 1.0 / np.array([s.n_spots for s in samples], dtype=float)
        if self.sample_weighting == "trace":
            return 1.0 / np.maximum(diags.sum(axis=1), self.eps)
        raise ValueError("sample_weighting must be 'none', 'n_spots', or 'trace'")


@dataclass
class GeneCenteringState:
    """Gene-independent state reused by the blocked diagonal calculation."""

    kind: str
    n: int
    K1: object = None
    s11: float = None
    Z: object = None
    A: object = None
    inv_lengths: object = None


def groups_for_operator(operator, n, sample_groups=None, label_groups=None):
    if operator == "none":
        return None
    if operator == "sample":
        return sample_groups if sample_groups is not None else Groups.single(n)
    if operator == "label":
        return label_groups
    raise ValueError("operator must be one of {'none', 'sample', 'label'}.")


def _diag_gene_centered(W, K, projector, dtype=np.float64):
    """
    Compute diag(W^T C K C W), where C is the centering operator in projector.

    W must be sparse (CSC recommended for efficiency).
    Handles no-centering, single-group, and multi-group cases.
    """
    KW = K @ W
    WTKW = np.asarray(W.multiply(KW).sum(axis=0)).ravel().astype(dtype)

    if projector.groups is None:
        return WTKW

    n = K.shape[0]
    offsets = projector.groups.offsets
    lengths = projector.groups.lengths

    if len(offsets) == 1:
        ones = np.ones(n, dtype=dtype)
        K1 = K @ ones
        s11 = float(ones @ K1)

        colsum = np.asarray(W.sum(axis=0)).ravel().astype(dtype)
        mean = colsum / float(n)
        wTK1 = np.asarray(W.T @ K1).ravel().astype(dtype)

        return WTKW - 2.0 * mean * wTK1 + (mean**2) * s11

    # Sparse grouped formula.
    row_to_group = np.empty(n, dtype=np.int64)
    for g, (off, length) in enumerate(zip(offsets, lengths)):
        row_to_group[int(off) : int(off + length)] = g

    rows = np.arange(n, dtype=np.int64)
    data = np.ones(n, dtype=dtype)
    Z = sp.csr_matrix(
        (data, (rows, row_to_group)),
        shape=(n, len(offsets)),
        dtype=dtype,
    )

    A = (Z.T @ K @ Z).tocsr()
    B = (Z.T @ W).tocsr()
    C = (Z.T @ KW).tocsr()

    M = B.multiply((1.0 / lengths.astype(dtype))[:, None])

    MTC = np.asarray(M.multiply(C).sum(axis=0)).ravel().astype(dtype)
    AM = A @ M
    MTAM = np.asarray(M.multiply(AM).sum(axis=0)).ravel().astype(dtype)

    return WTKW - 2.0 * MTC + MTAM


def _prepare_gene_centering_state(K, projector, dtype):
    """Precompute centering quantities that do not depend on the gene slice."""
    groups = projector.groups
    n = K.shape[0]
    if groups is None:
        return GeneCenteringState(kind="none", n=n)

    offsets = groups.offsets
    lengths = groups.lengths
    if len(offsets) == 1:
        ones = np.ones(n, dtype=dtype)
        K1 = K @ ones
        return GeneCenteringState(kind="single", n=n, K1=K1, s11=float(ones @ K1))

    row_to_group = np.empty(n, dtype=np.int64)
    for group, (offset, length) in enumerate(zip(offsets, lengths)):
        row_to_group[int(offset) : int(offset + length)] = group

    rows = np.arange(n, dtype=np.int64)
    Z = sp.csr_matrix(
        (np.ones(n, dtype=dtype), (rows, row_to_group)), shape=(n, len(offsets)), dtype=dtype
    )
    A = (Z.T @ K @ Z).tocsr()
    return GeneCenteringState(
        kind="grouped", n=n, Z=Z, A=A, inv_lengths=1.0 / lengths.astype(dtype)
    )


def _diag_gene_centered_block(W, K, state, dtype):
    """Blocked counterpart of _diag_gene_centered using reusable state."""
    KW = K @ W
    WTKW = np.asarray(W.multiply(KW).sum(axis=0)).ravel().astype(dtype, copy=False)

    if state.kind == "none":
        return WTKW

    if state.kind == "single":
        colsum = np.asarray(W.sum(axis=0)).ravel().astype(dtype, copy=False)
        mean = colsum / float(state.n)
        wTK1 = np.asarray(W.T @ state.K1).ravel().astype(dtype, copy=False)
        return WTKW - 2.0 * mean * wTK1 + (mean**2) * state.s11

    B = (state.Z.T @ W).tocsr()
    C = (state.Z.T @ KW).tocsr()
    M = B.multiply(state.inv_lengths[:, None])
    MTC = np.asarray(M.multiply(C).sum(axis=0)).ravel().astype(dtype, copy=False)
    AM = state.A @ M
    MTAM = np.asarray(M.multiply(AM).sum(axis=0)).ravel().astype(dtype, copy=False)
    return WTKW - 2.0 * MTC + MTAM


def _gene_spatial_variance_full(W, K, projector, dtype=np.float64):
    """Original whole-gene implementation, kept as the no-block fast path."""

    # diag(W^T C K C W), computed by the existing efficient grouped formula.
    diag0 = _diag_gene_centered(W, K, projector, dtype)

    Q = projector.Q
    if Q is None:
        return np.asarray(diag0, dtype=dtype)

    # T = Q^T C W.
    # Since Q is centered, C Q = Q, so Q^T C W = Q^T W.
    T = (W.T @ Q).T  # shape: (r, n_genes)

    # S_cov = Q^T K C W = (C K Q)^T W.
    KQ = K @ Q  # shape: (n_spots, r)
    CKQ = projector._center(KQ)  # C K Q
    S_cov = (W.T @ CKQ).T  # shape: (r, n_genes)

    # M = Q^T K Q.
    M = Q.T @ KQ  # shape: (r, r)

    diag = diag0 - 2.0 * np.sum(T * S_cov, axis=0) + np.sum(T * (M @ T), axis=0)
    return np.asarray(diag, dtype=dtype)


def _gene_spatial_variance_blocked(W, K, projector, block_size, dtype=np.float64):
    """Compute the diagonal in contiguous gene blocks."""
    state = _prepare_gene_centering_state(K, projector, dtype)

    Q = projector.Q
    if Q is None:
        CKQ = None
        M = None
    else:
        KQ = K @ Q
        CKQ = projector._center(KQ)
        M = Q.T @ KQ

    n_genes = W.shape[1]
    diag = np.empty(n_genes, dtype=dtype)
    for start in range(0, n_genes, block_size):
        stop = min(start + block_size, n_genes)
        W_block = W[:, start:stop]
        block_diag = _diag_gene_centered_block(W_block, K, state, dtype)

        if Q is not None:
            T = (W_block.T @ Q).T
            S_cov = (W_block.T @ CKQ).T
            block_diag = block_diag - 2.0 * np.sum(T * S_cov, axis=0) + np.sum(T * (M @ T), axis=0)

        diag[start:stop] = block_diag

    return diag


def gene_spatial_variance(W, K, projector, block_size=None, dtype=np.float64):
    """
    Compute diag(W^T C R K R C W) without densifying W.

    Here:
        C = spot-centering operator
        R = I - Q Q^T, where Q was built from centered covariates
        P = R C
        S = P^T K P = C R K R C

    The result is:
        diag(W^T S W)

    This uses:
        diag(W^T C K C W)
        - 2 diag(T^T S_cov)
        + diag(T^T M T)

    where:
        T     = Q^T W
        S_cov = (C K Q)^T W
        M     = Q^T K Q

    Because C Q = Q, this is equivalent to the dense formula using W_c = C W,
    but avoids materialising W_c.

    block_size controls the maximum number of genes represented by KW and
    the covariate correction temporaries at once. None, or a value greater than
    or equal to the number of genes, uses the original whole-matrix fast path.
    """
    W = W.tocsc() if sp.issparse(W) else sp.csc_matrix(W, dtype=dtype)
    K = K.tocsr().astype(dtype, copy=False)
    if block_size is None:
        return _gene_spatial_variance_full(W, K, projector, dtype)
    if block_size < 0 or not isinstance(block_size, (int, np.integer)):
        raise TypeError("Gene block_size must be a positive integer.")
    return _gene_spatial_variance_blocked(W, K, projector, block_size, dtype)


def gene_scale_from_diag(diag, alpha, eps):
    if alpha <= 0:
        return None
    return np.maximum(diag, eps) ** (-0.5 * alpha)
