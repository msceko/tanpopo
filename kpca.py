from functools import cached_property

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh

from kernel import kernel_matrix_sparse
from utils import timed, vector_or_matrix, all_equal, make_list, order_by_label


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
        global_center=False,
        verbose=False,
        eps=1e-12,
        dtype=np.float64,
    ):
        self.radius = radius
        self.alpha = alpha
        self.spot_center = spot_center
        self.global_center = global_center
        self.gene_center = gene_center
        self.cosine_normalise = cosine_normalise
        self.verbose = verbose
        self.eps = eps
        self.dtype = dtype

    def _validate_inputs(self, W, coords, labels):
        """Put all inputs into sample lists and validate"""
        W, coords = make_list(W), make_list(coords)
        input_lengths = [len(W), len(coords)]

        if labels is None:
            labels = len(W) * [None]
        else:
            labels = make_list(labels)
            input_lengths.append(len(labels))

        if not all_equal(input_lengths):
            raise ValueError("All inputs must have the same number of samples.")
        if not all_equal([w.shape[1] for w in W]):
            raise ValueError("ALl samples must have the same number of genes.")

        return W, coords, labels

    def _setup(self, W, coords, labels):
        """
        Validate inputs, prepare per-sample row ordering, and construct the
        concatenated expression matrix and block-diagonal spatial kernel.

        For each sample:
        - if labels are not provided, the sample is kept in its original order
            and treated as a single centering group
        - if labels are provided, rows are reordered so that equal labels form
            contiguous blocks used for within-sample grouped centering

        This method stores:
        - W_csr, W_csc : concatenated spot-by-gene matrices
        - K_csr        : block-diagonal kernel across samples
        - n_spots      : per-sample spot counts
        - total_spots  : total number of spots
        - group_offsets, group_lengths :
                contiguous centering blocks after reordering
        - inv_row_order:
                inverse permutation used to restore original input order

        If only one group is present after setup, centering is treated as global.
        """
        W, coords, labels = self._validate_inputs(W, coords, labels)

        self.n_samples = len(W)
        self.n_genes = W[0].shape[1]
        self.n_spots = np.array([w.shape[0] for w in W])
        self.total_spots = self.n_spots.sum()

        self.sample_offsets = np.r_[0, np.cumsum(self.n_spots[:-1])]
        self.inv_row_order = np.empty(self.total_spots, dtype=np.int64)
        self.group_offsets = np.empty(0, dtype=np.int64)
        self.group_lengths = np.empty(0, dtype=np.int64)
        K = []

        for i in range(self.n_samples):
            if labels[i] is None:
                idx = np.arange(self.n_spots[i], dtype=np.int64)
                offsets = np.array([0], dtype=np.int64)
                lengths = np.array([self.n_spots[i]])
            else:
                idx, offsets, lengths = order_by_label(labels[i])

            W[i], coords[i] = W[i][idx], coords[i][idx]
            K.append(kernel_matrix_sparse(coords[i], self.radius))

            self.group_offsets = np.append(self.group_offsets, self.sample_offsets[i] + offsets)
            self.group_lengths = np.append(self.group_lengths, lengths)

            sample_inv = self.sample_offsets[i] + np.arange(self.n_spots[i])
            self.inv_row_order[self.sample_offsets[i] + idx] = sample_inv

        self.n_groups = len(self.group_offsets)
        self._global_center = bool(self.global_center or self.n_groups == 1)

        self.W_csr = sp.vstack(W, format="csr").astype(self.dtype, copy=False)
        self.W_csc = self.W_csr.tocsc(copy=False)
        self.K_csr = sp.block_diag(K, format="csr")

    @cached_property
    def _row_to_group(self):
        """
        Row -> group id map for non-global spot centering.

        Groups are contiguous blocks defined by self.group_offsets/self.group_lengths.
        """
        row_to_group = np.empty(self.total_spots, dtype=np.int64)
        for g, (off, length) in enumerate(zip(self.group_offsets, self.group_lengths)):
            off = int(off)
            length = int(length)
            row_to_group[off : off + length] = g
        return row_to_group

    @cached_property
    def _group_membership(self):
        """
        Sparse one-hot membership matrix Z of shape (n_spots, n_groups),
        with Z[i, g] = 1 iff row i belongs to group g.
        """
        rows = np.arange(self.total_spots, dtype=np.int64)
        cols = self._row_to_group
        data = np.ones(self.total_spots, dtype=self.dtype)
        return sp.csr_matrix(
            (data, (rows, cols)), shape=(self.total_spots, self.n_groups), dtype=self.dtype
        )

    @cached_property
    def _group_kernel_mass(self):
        """
        A = Z^T K Z, the group-group kernel mass matrix.
        Shape: (n_groups, n_groups)
        """
        Z = self._group_membership
        return (Z.T @ self.K_csr @ Z).tocsr()

    @cached_property
    def _diagG(self):
        """
        Compute diag(W^T K_* W), where:

        - if spot_center=False:
                K_* = K

        - if spot_center=True and global_center=True:
                K_* = H K H

        - if spot_center=True and global_center=False:
                K_* = R K R
            where R subtracts means within contiguous groups.

        The non-global branch uses the sparse grouped formula:
            diag(W^T R K R W)
        = diag(W^T K W)
            - 2 diag(M^T Z^T K W)
            + diag(M^T (Z^T K Z) M)

        with:
            Z = group membership matrix
            M = D^{-1} Z^T W   (group means of W)
            D = diag(group sizes)
        """
        # Common uncentered term: diag(W^T K W)
        KW = self.K_csr @ self.W_csc
        WTKW = np.asarray(self.W_csc.multiply(KW).sum(axis=0)).ravel().astype(self.dtype)

        if not self.spot_center:
            return np.maximum(WTKW, self.eps)

        if self._global_center:
            # Fast closed form for global centering
            ones = np.ones(self.total_spots, dtype=self.dtype)
            K1 = self.K_csr @ ones
            s11 = float(ones @ K1)

            colsumW = np.asarray(self.W_csc.sum(axis=0)).ravel().astype(self.dtype)
            meanW = colsumW / float(self.total_spots)
            wTK1 = np.asarray(self.W_csc.T @ K1).ravel().astype(self.dtype)

            diagG = WTKW - 2.0 * meanW * wTK1 + (meanW**2) * s11
            return np.maximum(diagG, self.eps)

        # Non-global grouped centering
        Z = self._group_membership
        A = self._group_kernel_mass

        # B = Z^T W      : group sums of W
        # C = Z^T K W    : group sums of K W
        B = (Z.T @ self.W_csc).tocsr()
        C = (Z.T @ KW).tocsr()

        # M = D^{-1} B   : group means of W
        inv_sizes = 1.0 / self.group_lengths.astype(self.dtype)
        M = B.multiply(inv_sizes[:, None])

        # diag(M^T C)
        MTC = np.asarray(M.multiply(C).sum(axis=0)).ravel().astype(self.dtype)
        # diag(M^T A M)
        AM = A @ M
        MTAM = np.asarray(M.multiply(AM).sum(axis=0)).ravel().astype(self.dtype)

        diagG = WTKW - 2.0 * MTC + MTAM
        return np.maximum(diagG, self.eps)

    @cached_property
    def _norm_factor(self):
        """Columnwise gene normalization factor c_g = diag(G)_g^{-alpha/2}"""
        if not self.cosine_normalise:
            return 1
        C = self._diagG ** (-0.5 * self.alpha)
        return C[:, None]

    @cached_property
    def _G(self):
        """Gene-space operator G = H_g B^T K_* B H_g"""
        return LinearOperator((self.n_genes, self.n_genes), matvec=self._matvec, dtype=self.dtype)

    def _clear_cached_properties(self):
        for name in (
            "_row_to_group",
            "_group_membership",
            "_group_kernel_mass",
            "_diagG",
            "_norm_factor",
            "_G",
        ):
            self.__dict__.pop(name, None)

    @vector_or_matrix
    def _apply_centering(self, U):
        """Apply centering operator H_n = I_n - 1_n columnwise."""
        return U - U.mean(axis=0, keepdims=True)

    @vector_or_matrix
    def _apply_centering_per_group(self, U):
        """Apply centering operator to each group separately"""
        out = np.array(U, copy=True, dtype=self.dtype)
        for offset, length in zip(self.group_offsets, self.group_lengths):
            sl = slice(int(offset), int(offset + length))
            out[sl] -= out[sl].mean(axis=0, keepdims=True)
        return out

    @vector_or_matrix
    def _gene_centering(self, U):
        """Apply gene-centering"""
        return self._apply_centering(U)

    @vector_or_matrix
    def _spot_centering(self, U):
        """Apply spot-centering"""
        if self._global_center:
            return self._apply_centering(U)
        return self._apply_centering_per_group(U)

    @vector_or_matrix
    def _apply_K(self, U):
        """
        Apply K or H_n K H_n.
        """
        if not self.spot_center:
            return self.K_csr @ U
        Uc = self._spot_centering(U)
        return self._spot_centering(self.K_csr @ Uc)

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
            X = self._gene_centering(X)
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
            Y = self._gene_centering(Y)
        return Y

    def _matvec(self, x):
        """G x = H_g B^T K_* B H_g x"""
        return self._apply_BT(self._apply_K(self._apply_B(x)))

    def _orient_vectors(self, V):
        """Orient vector V according to its sum of squared components."""
        Vpos = np.sum((V > 0) * (V**2), axis=0)
        Vneg = np.sum((V < 0) * (V**2), axis=0)
        signs = 2 * ((Vpos - Vneg) >= 0) - 1
        return signs * V

    def lift_spot_modes(self, normalise=True, return_input_order=True, split_by_sample=True):
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
            loadings = self._gene_centering(loadings)
        if self.cosine_normalise:
            loadings = self._norm_factor * loadings
        if normalise:
            sqrt_eigvals = np.sqrt(self.eigenvalues)[None, :]
            phi = phi / sqrt_eigvals
            loadings = loadings / sqrt_eigvals

        phi = np.asarray(phi, dtype=self.dtype)
        loadings = np.asarray(loadings, dtype=self.dtype)

        if return_input_order:
            phi = phi[self.inv_row_order]

        if split_by_sample:
            phi_list = []
            for offset, n in zip(self.sample_offsets, self.n_spots):
                phi_list.append(phi[offset : offset + n])
            return phi_list, loadings

        return phi, loadings

    def fit(self, W, coords, n_components, labels=None, tol=0, maxiter=None):
        """
        Top eigenpairs of symmetric LinearOperator G via ARPACK (eigsh).

        Parameters
        ----------
        W : list
            List of spot-by-gene matrices, one per sample.
        coords : list
            List of coordinate arrays, one per sample.
        n_components : int
            Number of leading eigenpairs to compute.
        labels : list or None
            Optional list of per-sample label vectors.
        tol : float
            ARPACK tolerance.
        maxiter : int or None
            ARPACK max iterations.

        Returns:
        ----------
            eigvals: (spots,)
            eigvecs: (genes, spots)
            Z:       (genes, spots) KPCA coords = eigvecs * sqrt(eigvals)
        """
        self._clear_cached_properties()
        with timed("Constructing kernel matrix", self.verbose):
            self._setup(W, coords, labels)

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

    def gene_spatial_scores(self):
        """
        Return gene-wise diagonal spatial scores diag(W^T K_* W).
        """
        return self._diagG.copy()

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
                "global_center": self.global_center,
                "gene_center": self.gene_center,
                "cosine_normalize": self.cosine_normalise,
            },
        }
