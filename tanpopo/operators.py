from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, svds

from tanpopo.data import Groups
from tanpopo.projection import SpotProjector


class ProjectedKernel:
    """Projected zero-diagonal spatial operator S = P^T A P."""

    def __init__(self, K, projector, dtype=np.float64):
        self.K = K.tocsr().astype(dtype, copy=False)
        self.projector = projector
        self.dtype = np.dtype(dtype)
        self.n = K.shape[0]

    def apply(self, X, residualise=True):
        X = self.projector.apply(X, residualise=residualise)
        X = self.K @ X
        return self.projector.adjoint(X, residualise=residualise)


@dataclass
class SpotOperatorSpec:
    centering: str

    def build(self, K, sample_groups, label_groups, covariates, tol, dtype):
        groups = groups_for_operator(
            self.centering, K.shape[0], sample_groups=sample_groups, label_groups=label_groups
        )
        projector = SpotProjector(
            K.shape[0], groups=groups, covariates=covariates, tol=tol, dtype=dtype
        )
        return ProjectedKernel(K, projector, dtype=dtype)


def groups_for_operator(operator, n, sample_groups=None, label_groups=None):
    if operator == "none":
        return None
    if operator == "sample":
        return sample_groups if sample_groups is not None else Groups.single(n)
    if operator == "label":
        if label_groups is None:
            raise ValueError("label centering requires labels")
        return label_groups
    raise ValueError("operator must be one of {'none', 'sample', 'label'}")


def _dense_projected_block(W, projector, start, stop):
    block = W[:, start:stop]
    if sp.issparse(block):
        block = block.toarray()
    return projector.apply(np.asarray(block, dtype=projector.dtype))


def gene_expression_variance(W, projector, block_size=256, eps=0.0):
    """diag(W^T P^T P W), computed in gene blocks.

    This is ordinary residual expression variance, not spatial autocovariance.
    It is therefore a valid positive scaling reference for the
    ``gene_standardized`` objective.
    """
    W = W.tocsr() if sp.issparse(W) else sp.csr_matrix(W)
    if block_size is None:
        block_size = min(256, W.shape[1])
    diag = np.empty(W.shape[1], dtype=float)
    for start in range(0, W.shape[1], block_size):
        stop = min(start + block_size, W.shape[1])
        Y = _dense_projected_block(W, projector, start, stop)
        diag[start:stop] = np.sum(Y * Y, axis=0)
    return np.maximum(diag, eps)


def gene_spatial_covariance_diag(W, S, block_size=256):
    """diag(W^T P^T A P W), without clipping negative values."""
    W = W.tocsr() if sp.issparse(W) else sp.csr_matrix(W)
    if block_size is None:
        block_size = min(256, W.shape[1])
    diag = np.empty(W.shape[1], dtype=float)
    for start in range(0, W.shape[1], block_size):
        stop = min(start + block_size, W.shape[1])
        Y = _dense_projected_block(W, S.projector, start, stop)
        AY = S.K @ Y
        diag[start:stop] = np.sum(Y * AY, axis=0)
    return diag


def expression_scale_from_variance(variance, eps=1e-12):
    variance = np.asarray(variance, dtype=float)
    return np.maximum(variance, eps) ** -0.5


class GeneKernel:
    """Gene-space spatial covariance operator.

    With ``gene_scale=None`` this is G = W^T P^T A P W.
    With a positive expression-derived scale D^-1/2 it is the gene-standardized
    form D^-1/2 G D^-1/2. Spatial autocovariance is never used as a denominator.
    """

    def __init__(self, W, S, gene_scale=None, dtype=np.float64):
        self.W = W.tocsr() if sp.issparse(W) else sp.csr_matrix(W, dtype=dtype)
        self.Wcsc = self.W.tocsc(copy=False)
        self.S = S
        self.gene_scale = None if gene_scale is None else np.asarray(gene_scale, dtype=dtype)
        self.dtype = np.dtype(dtype)
        self.shape = (self.W.shape[1], self.W.shape[1])

    def _prepare(self, X):
        X = np.asarray(X, dtype=self.dtype)
        if X.ndim == 1:
            X = X[:, None]
        if self.gene_scale is not None:
            X = self.gene_scale[:, None] * X
        return X

    def apply(self, X):
        was_vector = np.asarray(X).ndim == 1
        X = self._prepare(X)
        U = self.W @ X
        U = self.S.apply(U)
        Y = self.Wcsc.T @ U
        if self.gene_scale is not None:
            Y = self.gene_scale[:, None] * Y
        return np.asarray(Y[:, 0] if was_vector else Y, dtype=self.dtype)

    def as_scipy(self):
        return LinearOperator(
            self.shape,
            matvec=lambda x: self.apply(x),
            matmat=lambda X: self.apply(X),
            dtype=self.dtype,
        )

    def loadings_from_solver_vectors(self, vectors):
        return self._prepare(vectors)

    def scores_from_loadings(self, loadings):
        return self.S.projector.apply(self.W @ loadings)


class SumGeneOperator:
    def __init__(self, terms, dtype=np.float64):
        self.terms = [(float(w), op) for w, op in terms if float(w) != 0]
        if not self.terms:
            raise ValueError("SumGeneOperator received no nonzero terms")
        self.shape = self.terms[0][1].shape
        self.dtype = np.dtype(dtype)

    def apply(self, X):
        was_vector = np.asarray(X).ndim == 1
        X2 = np.asarray(X, dtype=self.dtype)
        if was_vector:
            X2 = X2[:, None]
        Y = np.zeros((self.shape[0], X2.shape[1]), dtype=self.dtype)
        for weight, op in self.terms:
            Y += weight * op.apply(X2)
        return Y[:, 0] if was_vector else Y

    def as_scipy(self):
        return LinearOperator(
            self.shape,
            matvec=lambda x: self.apply(x),
            matmat=lambda X: self.apply(X),
            dtype=self.dtype,
        )


class ProjectedExpressionOperator(LinearOperator):
    """LinearOperator for Y = P W, optionally row weighted."""

    def __init__(self, W, projector, row_scale=None, dtype=np.float64):
        self.W = W.tocsr() if sp.issparse(W) else sp.csr_matrix(W, dtype=dtype)
        self.projector = projector
        self.row_scale = None if row_scale is None else np.asarray(row_scale, dtype=dtype)
        self.dtype = np.dtype(dtype)
        super().__init__(dtype=self.dtype, shape=self.W.shape)

    def _matvec(self, v):
        y = self.projector.apply(self.W @ v)
        if self.row_scale is not None:
            y = self.row_scale * y
        return np.asarray(y).ravel()

    def _matmat(self, V):
        Y = self.projector.apply(self.W @ V)
        if self.row_scale is not None:
            Y = self.row_scale[:, None] * Y
        return np.asarray(Y)

    def _rmatvec(self, u):
        u = np.asarray(u, dtype=self.dtype)
        if self.row_scale is not None:
            u = self.row_scale * u
        u = self.projector.adjoint(u[:, None])[:, 0]
        return np.asarray(self.W.T @ u).ravel()

    def _rmatmat(self, U):
        U = np.asarray(U, dtype=self.dtype)
        if self.row_scale is not None:
            U = self.row_scale[:, None] * U
        U = self.projector.adjoint(U)
        return np.asarray(self.W.T @ U)


def projected_svd(W, projector, rank, row_scale=None, dtype=np.float64, tol=0):
    """Truncated SVD of P W without materialising the projected expression matrix."""
    n, g = W.shape
    max_rank = min(n, g) - 1
    if max_rank < 1:
        raise ValueError("Projected SVD requires at least two spots and two genes")
    rank = min(int(rank), max_rank)
    if rank < 1:
        raise ValueError("expression_rank must be positive")
    op = ProjectedExpressionOperator(W, projector, row_scale=row_scale, dtype=dtype)
    U, s, Vt = svds(op, k=rank, which="LM", tol=tol, return_singular_vectors=True)
    order = np.argsort(s)[::-1]
    return U[:, order], s[order], Vt[order]


class CrossGeneOperator(LinearOperator):
    """C = Y_target^T A_target,neighbour Y_neighbour."""

    def __init__(
        self,
        W_target,
        W_neighbour,
        K_cross,
        projector_target,
        projector_neighbour,
        scale_target=None,
        scale_neighbour=None,
        dtype=np.float64,
    ):
        self.Wt = W_target.tocsr()
        self.Wu = W_neighbour.tocsr()
        self.K = K_cross.tocsr().astype(dtype, copy=False)
        self.Pt = projector_target
        self.Pu = projector_neighbour
        self.scale_t = None if scale_target is None else np.asarray(scale_target, dtype=dtype)
        self.scale_u = None if scale_neighbour is None else np.asarray(scale_neighbour, dtype=dtype)
        self.dtype = np.dtype(dtype)
        super().__init__(dtype=self.dtype, shape=(self.Wt.shape[1], self.Wu.shape[1]))

    def _prepare_u(self, V):
        V = np.asarray(V, dtype=self.dtype)
        return V if self.scale_u is None else self.scale_u[:, None] * V

    def _prepare_t(self, V):
        V = np.asarray(V, dtype=self.dtype)
        return V if self.scale_t is None else self.scale_t[:, None] * V

    def _matmat(self, V):
        V = self._prepare_u(V)
        y = self.Pu.apply(self.Wu @ V)
        y = self.K @ y
        y = self.Pt.adjoint(y)
        out = np.asarray(self.Wt.T @ y)
        if self.scale_t is not None:
            out = self.scale_t[:, None] * out
        return out

    def _matvec(self, v):
        return self._matmat(np.asarray(v)[:, None])[:, 0]

    def _rmatmat(self, V):
        V = self._prepare_t(V)
        y = self.Pt.apply(self.Wt @ V)
        y = self.K.T @ y
        y = self.Pu.adjoint(y)
        out = np.asarray(self.Wu.T @ y)
        if self.scale_u is not None:
            out = self.scale_u[:, None] * out
        return out

    def _rmatvec(self, v):
        return self._rmatmat(np.asarray(v)[:, None])[:, 0]


class SpotTransform:
    """Abstract spot-space linear transform with explicit adjoint."""

    def apply(self, X):
        raise NotImplementedError

    def adjoint(self, X):
        raise NotImplementedError


class ProjectorTransform(SpotTransform):
    def __init__(self, projector):
        self.projector = projector

    def apply(self, X):
        return self.projector.apply(X)

    def adjoint(self, X):
        return self.projector.adjoint(X)


class WithinLabelTransform(SpotTransform):
    """Within-label component of a common residualized expression matrix."""

    def __init__(self, base_projector, label_groups, dtype=np.float64):
        self.base = base_projector
        self.label_center = SpotProjector(
            base_projector.n, groups=label_groups, covariates=None, dtype=dtype
        )

    def apply(self, X):
        return self.label_center.apply(self.base.apply(X), residualise=False)

    def adjoint(self, X):
        return self.base.adjoint(self.label_center.adjoint(X, residualise=False))


class BetweenLabelTransform(SpotTransform):
    """Between-label means remaining after subtracting within-label residuals."""

    def __init__(self, base_projector, within_transform):
        self.base = base_projector
        self.within = within_transform

    def apply(self, X):
        return self.base.apply(X) - self.within.apply(X)

    def adjoint(self, X):
        return self.base.adjoint(X) - self.within.adjoint(X)


class BilinearGeneOperator:
    """G = W^T T_left^T A T_right W, optionally symmetrized."""

    def __init__(
        self,
        W,
        K,
        left_transform,
        right_transform=None,
        symmetric_pair=False,
        dtype=np.float64,
    ):
        self.W = W.tocsr() if sp.issparse(W) else sp.csr_matrix(W, dtype=dtype)
        self.Wcsc = self.W.tocsc(copy=False)
        self.K = K.tocsr().astype(dtype, copy=False)
        self.left = left_transform
        self.right = left_transform if right_transform is None else right_transform
        self.symmetric_pair = bool(symmetric_pair)
        self.dtype = np.dtype(dtype)
        self.shape = (self.W.shape[1], self.W.shape[1])

    def _one(self, X, left, right, K):
        U = self.W @ X
        U = right.apply(U)
        U = K @ U
        U = left.adjoint(U)
        return np.asarray(self.Wcsc.T @ U)

    def apply(self, X):
        was_vector = np.asarray(X).ndim == 1
        X2 = np.asarray(X, dtype=self.dtype)
        if was_vector:
            X2 = X2[:, None]
        Y = self._one(X2, self.left, self.right, self.K)
        if self.symmetric_pair:
            Y += self._one(X2, self.right, self.left, self.K.T)
        return Y[:, 0] if was_vector else Y

    def as_scipy(self):
        return LinearOperator(
            self.shape,
            matvec=lambda x: self.apply(x),
            matmat=lambda X: self.apply(X),
            dtype=self.dtype,
        )
