import numpy as np
from scipy.spatial.distance import cdist


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
    norm = np.sqrt(np.outer(np.diag(G), np.diag(G)))
    return G / norm


def cs_divergence(S):
    """Compute Cauchy-Schwarz divergence from similarity"""
    return -np.log(S)
