import numpy as np
from scipy.linalg import qr


class SpotProjector:
    """
    Unified centering and covariate residualisation.

    Replaces the separate Centering, CovariateResidual, and Projection classes.
    Applies P = R C where C is group centering and R removes centered covariates.

    self.Q holds the orthonormal basis for the covariate column space (or None),
    and is exposed for use by gene_spatial_variance in the low-rank correction.
    """

    def __init__(self, n, groups=None, covariates=None, tol=1e-10, dtype=np.float64):
        self.n = n
        self.groups = groups
        self.dtype = dtype
        self.Q = None
        self._build_residual(covariates, tol)

    def _build_residual(self, covariates, tol):
        if covariates is None:
            return

        X = np.asarray(covariates, dtype=self.dtype)
        if X.ndim == 1:
            X = X[:, None]

        X = self._center(X)
        Q, R, piv = qr(X, mode="economic", pivoting=True)

        if R.size:
            d = np.abs(np.diag(R))
            if d.size == 0 or d.max() == 0:
                return
            keep = d > tol * max(X.shape) * d.max()
            Q = Q[:, keep]
            self.Q = Q if Q.shape[1] else None

    def _center(self, X):
        X = np.asarray(X, dtype=self.dtype)

        if self.groups is None:
            return X

        out = np.array(X, copy=True, dtype=self.dtype)
        for off, length in zip(self.groups.offsets, self.groups.lengths):
            sl = slice(int(off), int(off + length))
            out[sl] -= out[sl].mean(axis=0, keepdims=True)

        return out

    def apply(self, X, residualise=True):
        X = self._center(X)
        if residualise and self.Q is not None:
            X = X - self.Q @ (self.Q.T @ X)
        return X

    def adjoint(self, X, residualise=True):
        if residualise and self.Q is not None:
            X = X - self.Q @ (self.Q.T @ X)
        X = self._center(X)
        return X
