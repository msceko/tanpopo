import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


def gaussian_kernel(coords, sigma):
    """
    Kernel for ⟨N(mu_i), N(mu_j)⟩ with isotropic variance sigma²

    Returns:
        K : (spots, spots)
    """
    d2 = cdist(coords, coords, metric="sqeuclidean")
    if sigma is None:
        sigma = np.median(d2[d2 > 0])
    return np.exp(-d2 / (4 * sigma**2))


def knn_gaussian_kernel(coords, sigma, k=50, symmetrize=True):
    """
    Build a sparse kNN Gaussian kernel K (CSR) on spot coordinates.

    K_ij = exp(-||xi-xj||^2 / (4*sigma^2)) for j in kNN(i)

    Args:
        coords: (spots, 2) or (spots, d)
        sigma: float
        k: number of neighbors (per row)
        include_self: include i->i edge (recommended)
        symmetrize: make K symmetric via (K + K.T)/2 (recommended for eigensolvers)

    Returns:
        K: scipy.sparse.csr_matrix shape (spots, spots)
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]

    nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nn.fit(coords)
    dists, idx = nn.kneighbors(coords)  # (n, n_neighbors)

    rows = np.repeat(np.arange(n), idx.shape[1])
    cols = idx.reshape(-1)

    # weights
    denom = 4.0 * float(sigma) * float(sigma)
    w = np.exp(-(dists.reshape(-1) ** 2) / denom)

    K = sp.csr_matrix((w, (rows, cols)), shape=(n, n))
    if symmetrize:
        K = 0.5 * (K + K.T)
    return K


def cs_kernel(W, K):
    """
    ⟨p_i, p_j⟩ = w_iᵀ K w_j

    Args:
        W : (spots, genes)
        K : (spots, spots)

    Returns:
        S : (genes, genes)
    """
    G = W.T @ (K @ W)  # gene gram metrix
    # G = 0.5 * (G + G.T)  # symmetrize for numerical drift
    if sp.issparse(G):
        G = G.toarray()
    norm = np.sqrt(np.outer(np.diag(G), np.diag(G)))
    return G / norm


def cs_divergence(S):
    """Compute Cauchy-Schwarz divergence from similarity"""
    return -np.log(S)
