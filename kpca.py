from functools import cached_property

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from kernel import kernel_matrix_sparse
from utils import timed, vector_or_matrix


class SpatialGeneKPCA:
    """
    Spatially informed gene-kernel PCA with optional:
      - spot centering
      - gene centering
      - cosine / partial-whitening normalization

    Fitted gene-space operator:
        G = H_g B^T K_* B H_g

    where
        K_* = K              if spot_center=False
        K_* = H_n K H_n      if spot_center=True

        B   = W              if cosine_normalise=False
        B   = W D^{-alpha}   if cosine_normalise=True

    with
        D_gg = sqrt( diag(W^T K_* W) ).

    If G V = V Lambda and V^T V = I, then the lifted spot modes

        Phi = B H_g V Lambda^{-1/2}

    satisfy approximately

        Phi^T K_* Phi = I

    on the retained positive-eigenvalue subspace.
    """

    def __init__(
        self,
        radius,
        alpha=0.5,
        spot_center=True,
        gene_center=True,
        cosine_normalise=True,
        verbose=False,
        eps=1e-12,
        dtype=np.float64,
    ):
        self.radius = radius
        self.alpha = alpha
        self.spot_center = spot_center
        self.gene_center = gene_center
        self.cosine_normalise = cosine_normalise
        self.verbose = verbose
        self.eps = eps
        self.dtype = dtype

    @cached_property
    def _diagG(self):
        """
        Compute diag(G), where
            G = W^T K_* W
        with K_* = K or H_n K H_n.

        If spot_center=True, we need diag(W^T (H K H) W)
        Use (w - m1)^T K (w - m1) = w^T K w - 2m (w^T K1) + m^2 (1^T K1)
        with m = mean(w). This avoids explicitly constructing Kc.
        """
        ones = np.ones(self.n_spots, dtype=self.dtype)
        K1 = self.K_csr @ ones
        s11 = float(ones @ K1)

        # KW = K @ W  (used for w^T K w)
        KW = self.K_csr @ self.W_csc
        wTKw = np.asarray(self.W_csc.multiply(KW).sum(axis=0)).ravel().astype(self.dtype)

        if self.spot_center:
            colsumW = np.asarray(self.W_csc.sum(axis=0)).ravel().astype(self.dtype)
            meanW = colsumW / float(self.n_spots)
            wTK1 = np.asarray(self.W_csc.T @ K1).ravel().astype(self.dtype)
            diagG = wTKw - 2.0 * meanW * wTK1 + (meanW**2) * s11
        else:
            diagG = wTKw

        return np.maximum(diagG, self.eps)

    @cached_property
    def _norm_factor(self):
        """Columnwise gene normalization factor c_g = diag(G)_g^{-alpha/2}"""
        if not self.cosine_normalise:
            return None
        C = self._diagG ** (-0.5 * self.alpha)
        return C[:, None]

    @cached_property
    def _G(self):
        """Gene-space operator G = H_g B^T K_* B H_g"""
        return LinearOperator((self.n_genes, self.n_genes), matvec=self.matvec, dtype=self.dtype)

    def _clear_cached_properties(self):
        for name in ("_diagG", "_norm_factor", "_G"):
            self.__dict__.pop(name, None)

    @vector_or_matrix
    def _apply_centering(self, U):
        """Apply centering operator H_n = I_n - 1_n columnwise."""
        return U - U.mean(axis=0, keepdims=True)

    @vector_or_matrix
    def _apply_K(self, U):
        """
        Apply K or H_n K H_n.
        """
        if not self.spot_center:
            return self.K_csr @ U
        Uc = self._apply_centering(U)
        return self._apply_centering(self.K_csr @ Uc)

    @vector_or_matrix
    def _apply_B(self, X):
        """
        Apply B H_g to gene-space input X, where:
          B = W              if cosine_normalise=False
          B = W D^{-alpha}   if cosine_normalise=True

        So this map is:
          X -> W [D^{-alpha}] [H_g X]
        """
        if self.gene_center:
            X = self._apply_centering(X)
        if self.cosine_normalise:
            X = self._norm_factor * X
        return self.W_csr @ X

    @vector_or_matrix
    def _apply_BT(self, U):
        """
        Apply H_g B^T to spot-space input U, where:
          B^T = W^T              if cosine_normalise=False
          B^T = D^{-alpha} W^T   if cosine_normalise=True

        So this map is:
          U -> H_g [D^{-alpha}] W^T U
        """
        Y = self.W_csc.T @ U
        if self.cosine_normalise:
            Y = self._norm_factor * Y
        if self.gene_center:
            Y = self._apply_centering(Y)
        return Y

    def _orient_vectors(self, V):
        """Orient vector V according to its sum of squared components."""
        Vpos = sum((V > 0) * (V**2), 1)
        Vneg = sum((V < 0) * (V**2), 1)
        signs = np.sign(Vpos - Vneg)
        return signs * V

    def lift_spot_modes(self, normalise=True):
        """
        Lift gene eigenvectors to spot modes and return the corresponding
        effective loading vectors in the original gene basis.

        If
            A = H_g B^T K_* B H_g
            A V = V Lambda
        then
            Phi = B H_g V Lambda^{-1/2}

        satisfies approximately
            Phi^T K_* Phi = I
        for positive eigenvalues.

        The corresponding loading vectors L are defined so that
            Phi = W L
        exactly.

        if cosine_normalise=False:
            L = H_g V                  (or H_g V Lambda^{-1/2} if normalise=True)
        if cosine_normalise=True:
            L = D^{-alpha} H_g V       (or D^{-alpha} H_g V Lambda^{-1/2} if normalise=True)
        """
        if not hasattr(self, "eigenvectors"):
            raise RuntimeError("Model must be fitted before lifting spot modes.")

        phi = self._apply_B(self.eigenvectors)
        loadings = self.eigenvectors.copy()

        if self.gene_center:
            loadings = self._apply_centering(loadings)
        if self.cosine_normalise:
            loadings = self._norm_factor * loadings
        if normalise:
            sqrt_eigvals = np.sqrt(self.eigenvalues)[None, :]
            phi = phi / sqrt_eigvals
            loadings = loadings / sqrt_eigvals

        return np.asarray(phi, dtype=self.dtype), np.asarray(loadings, dtype=self.dtype)

    def matvec(self, x):
        """G x = H_g B^T K_* B H_g x"""
        return self._apply_BT(self._apply_K(self._apply_B(x)))

    def fit(self, W, coords, n_components, tol=0, maxiter=None):
        """
        Top eigenpairs of symmetric LinearOperator G via ARPACK (eigsh).

        Returns:
        eigvals: (spots,)
        eigvecs: (genes, spots)
        Z:       (genes, spots) KPCA coords = eigvecs * sqrt(eigvals)
        """
        self._clear_cached_properties()

        self.n_spots, self.n_genes = W.shape
        self.W_csc = W.tocsc(copy=False)
        self.W_csr = W.tocsr(copy=False)
        with timed("Constructing kernel matrix", self.verbose):
            self.K_csr = kernel_matrix_sparse(coords, self.radius)

        with timed("Computing eigendecomposition", self.verbose):
            self.eigenvalues, self.eigenvectors = eigsh(
                self._G, k=n_components, which="LA", tol=tol, maxiter=maxiter
            )

        with timed("Processing eigenmodes", self.verbose):
            idx = np.argsort(self.eigenvalues)[::-1]
            self.eigenvalues = np.maximum(self.eigenvalues[idx], 0.0)
            self.eigenvectors = self._orient_vectors(self.eigenvectors[:, idx])
            self.gene_scores = self.eigenvectors * np.sqrt(self.eigenvalues[None, :])

            self.spot_modes, self.gene_loadings = self.lift_spot_modes()

        return self

    def summary(self):
        return {
            "eigenvectors": self.eigenvectors,
            "eigenvalues": self.eigenvalues,
            "gene_scores": self.gene_scores,
            "spot_modes": self.spot_modes,
            "gene_loadings": self.gene_loadings,
            "cfg": {
                "kernel": "wendland_c2",
                "radius": self.radius,
                "alpha": self.alpha,
                "spot_center": self.spot_center,
                "gene_center": self.gene_center,
                "cosine_normalize": self.cosine_normalise,
            },
        }
