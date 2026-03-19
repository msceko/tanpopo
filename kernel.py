import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


def gaussian_kernel(coords, sigma):
    """
    Kernel for ⟨N(mu_i), N(mu_j)⟩ with isotropic variance sigma^2

    Returns:
        K : (spots, spots)
    """
    d2 = cdist(coords, coords, metric="sqeuclidean")
    return np.exp(-d2 / (4 * sigma**2))


def wendland_c2(r):
    """
    Wendland C^2 (compactly supported, PSD in R^d for d <= 3):
      phi(r) = (1 - r)^4_+ * (4r + 1),  r >= 0
    """
    r = np.asarray(r, dtype=np.float64)
    t = np.maximum(1.0 - r, 0.0)
    return (t**4) * (4.0 * r + 1.0)


def kernel_matrix_sparse(coords, radius, symmetrize=True):
    """
    Build a sparse radius-neighborhood Wendland kernel K (CSR) on spot coordinates.

      K_ij = phi(||xi-xj|| / ell)   for ||xi-xj|| <= radius
      phi = Wendland C^2

    Args:
        coords: (n_spots, d)
        radius: positive length-scale. Support of phi is r<=1, i.e. ||xi-xj|| <= ell.
        symmetrize: (K + K.T)/2

    Returns:
        K: scipy.sparse.csr_matrix, shape (n_spots, n_spots)
    """
    nn = NearestNeighbors(radius=radius, algorithm="ball_tree", metric="euclidean")
    nn.fit(coords)

    K = nn.radius_neighbors_graph(coords, mode="distance")  # CSR
    K.data = wendland_c2(K.data / radius)

    if symmetrize:
        Kt = K.T.tocsr()
        K = K.maximum(Kt)

    K.eliminate_zeros()

    return K


def cosine_kernel(W, K):
    """
    ⟨p_i, p_j⟩ = w_i^T K w_j

    Args:
        W : (spots, genes)
        K : (spots, spots)

    Returns:
        S : (genes, genes)
    """
    G = W.T @ (K @ W)  # gene gram metrix
    G = 0.5 * (G + G.T)  # symmetrize for numerical drift
    if sp.issparse(G):
        G = G.toarray()
    norm = np.sqrt(np.outer(np.diag(G), np.diag(G)))
    return G / norm


def cosine_kernel_operator(
    W,
    K,
    spot_center=True,
    gene_center=True,
    cosine_normalise=True,
    eps=1e-12,
    dtype=np.float64,
):
    """
    Build LinearOperator A implementing y = (H_g) * (C) * (H_g) x, where:

      - Optional spot-RKHS centering:
            Kc = H_n K H_n,   H_n = I_n - (1/n) 11^T
        If spot_center=False, Kc = K.

      - Optional cosine normalization:
            G  = W^T Kc W
            C  = D^{-1} G D^{-1},   D_gg = sqrt(G_gg)
        If cosine_normalize=False, C = G.

      - Optional gene centering (applied left and right):
            H_g = I_p - (1/p) 11^T
        If gene_center=False, H_g = I_p.

    The operator is applied without explicitly forming G or C.

    Args:
        W: (n_spots, n_genes) sparse matrix (CSR/CSC ok).
        K: (n_spots, n_spots) sparse matrix (CSR/CSC ok). Symmetric recommended.
        spot_center: apply spot centering in RKHS (H_n K H_n).
        gene_center: apply gene centering (H_g * ... * H_g).
        cosine_normalize: apply cosine normalization in gene-RKHS induced by Kc.
        eps: floor for diagonal in cosine normalization.
        dtype: numerical dtype.

    Returns:
        A: LinearOperator (n_genes, n_genes)
        info: dict with diagnostics (e.g., diagG, scaling c, etc.)
    """
    n_spots, n_genes = W.shape
    W_csc = W.tocsc(copy=False)
    W_csr = W.tocsr(copy=False)
    K_csr = K.tocsr(copy=False)

    def H_spots(u):
        # center over spots
        return u - u.mean()

    def H_genes(x):
        # center over genes
        return x - x.mean()

    def Kc_apply(u):
        # Apply Kc = (H_n K H_n) if spot_center else K
        if not spot_center:
            return K_csr @ u
        u1 = H_spots(u)
        v = K_csr @ u1
        return H_spots(v)

    if cosine_normalise:
        ones = np.ones(n_spots, dtype=dtype)

        # If spot_center=True, we need diag(W^T (H K H) W)
        # Use: (w - m1)^T K (w - m1) = w^T K w - 2m (w^T K1) + m^2 (1^T K1)
        # with m = mean(w). This avoids explicitly constructing Kc.
        K1 = K_csr @ ones
        s11 = float(ones @ K1)

        # KW = K @ W  (used for w^T K w)
        KW = K_csr @ W_csc

        wTKw = np.asarray(W_csc.multiply(KW).sum(axis=0)).ravel().astype(dtype)

        if spot_center:
            colsumW = np.asarray(W_csc.sum(axis=0)).ravel().astype(dtype)
            meanW = colsumW / float(n_spots)
            wTK1 = np.asarray(W_csc.T @ K1).ravel().astype(dtype)
            diagG = wTKw - 2.0 * meanW * wTK1 + (meanW**2) * s11
        else:
            diagG = wTKw

        diagG = np.maximum(diagG, eps)
        c = 1.0 / np.sqrt(diagG)  # elementwise
    else:
        diagG = None
        c = None

    def matvec(x):
        x = np.asarray(x, dtype=dtype).reshape(-1)
        # right gene-centering
        if gene_center:
            x = H_genes(x)
        # right cosine scaling
        if cosine_normalise:
            x = x * c
        # u = W x  (spots)
        u = W_csr @ x
        # v = Kc u  (spots), optionally centered in spot RKHS
        v = Kc_apply(u)
        # y = W^T v (genes)
        y = W_csc.T @ v
        y = np.asarray(y).reshape(-1)
        # left cosine scaling
        if cosine_normalise:
            y = y * c
        # left gene-centering
        if gene_center:
            y = H_genes(y)

        return y

    return LinearOperator((n_genes, n_genes), matvec=matvec, dtype=dtype)


def cs_divergence(S):
    """Compute Cauchy-Schwarz divergence from similarity"""
    return -np.log(S)
