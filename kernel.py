import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

from utils import make_iterable


def gaussian_kernel(coords, sigma):
    """
    Kernel for ⟨N(mu_i), N(mu_j)⟩ with isotropic variance sigma^2

    Returns:
        K : (spots, spots)
    """
    d2 = cdist(coords, coords, metric="sqeuclidean")
    return np.exp(-d2 / (4 * sigma**2))


def gaussian_kernel_sparse(
    coords, sigma, beta=None, radius=None, symmetrize=True, normalise_mass=True
):
    """
    Build a sparse nearest neighbour Gaussian kernel K (CSR) on spot coordinates.

    K_ij = exp(-||xi-xj||^2 / (4*sigma^2)) for j in kNN(i)

    Args:
        coords: (spots, 2) or (spots, d)
        sigma: float
        radius: range of parameter space for cutoff
        symmetrize: make K symmetric via (K + K.T)/2 (recommended for eigensolvers)

    Returns:
        K: scipy.sparse.csr_matrix shape (spots, spots)
    """
    sigma = np.asarray(make_iterable(sigma))
    if beta is None:
        beta = np.ones_like(sigma)
    else:
        beta = np.asarray(make_iterable(beta))
    beta = beta / beta.sum()
    if radius is None:
        radius = 3.0 * max(sigma)

    nn = NearestNeighbors(radius=radius, algorithm="ball_tree", metric="euclidean")
    nn.fit(coords)
    D = nn.radius_neighbors_graph(coords, mode="distance")  # CSR

    K = sp.csr_matrix(D.shape, dtype=np.float64)
    for s, b in zip(sigma, beta):
        Ks = D.copy()
        Ks.data = np.exp(-(D.data**2) / (4 * s**2))
        if normalise_mass:
            Ks = Ks / Ks.sum()
        K = K + Ks.multiply(b)

    if symmetrize:
        K = 0.5 * (K + K.T)
    K.eliminate_zeros()

    return K


def cs_kernel(W, K):
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


def cs_kernel_operator(W, K, eps=1e-12, precompute_KW=True, return_KW=False):
    """
    Build LinearOperator A implementing y = (H C H) x without forming C or G.
    Implicit centered CS-cosine gene kernel operator for eigsh
    C = D^{-1} (W^T K W) D^{-1}, then A = H C H

    Assumes:
      - W is sparse (CSC/CSR), shape (spots, genes)
      - K is sparse CSR/CSC, shape (spots, spots), symmetric recommended

    Returns:
      A: LinearOperator (genes, genes)
      d: (genes,) sqrt(diag(W^T K W)) used for cosine normalization
    """
    n_genes = W.shape[1]

    # W as CSC for fast W.T @ v, and CSR for fast W @ x
    W_csc = W.tocsc(copy=False)
    W_csr = W.tocsr(copy=False)
    K_csr = K.tocsr(copy=False)

    # precompute diag(G) where G = W^T K W:
    # diag(G)_g = w_g^T K w_g
    if precompute_KW:
        KW = K_csr @ W_csc
        # diag is column-wise sum of elementwise product W * (K W)
        diagG = np.asarray(W_csc.multiply(KW).sum(axis=0)).ravel()
    else:
        # slower but lower memory: compute diagG gene-by-gene if needed (not recommended)
        diagG = np.zeros(n_genes, dtype=np.float64)
        for g in range(n_genes):
            wg = W_csc[:, g]
            diagG[g] = float((wg.T @ (K_csr @ wg)).toarray()[0, 0])

    d = np.sqrt(np.maximum(diagG, eps))  # cosine normalization denom per gene

    def H_apply(x):
        # double-centering in feature space corresponds to Hx = x - mean(x)
        return x - np.mean(x)

    def matvec(x):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        # right centering
        x = H_apply(x)

        # right cosine scaling
        x = x / d

        # y = W^T K (W x)
        u = W_csr @ x  # (spots,)
        v = K_csr @ u  # (spots,)
        y = W_csc.T @ v  # (genes,)

        # left cosine scaling
        y = np.asarray(y).reshape(-1) / d

        # left centering
        y = H_apply(y)
        return y

    A = LinearOperator((n_genes, n_genes), matvec=matvec, dtype=np.float64)

    if return_KW:
        return A, KW
    return A


def cs_divergence(S):
    """Compute Cauchy-Schwarz divergence from similarity"""
    return -np.log(S)
