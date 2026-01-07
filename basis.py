import numpy as np
from scipy.linalg import eigh


def project_spatial_basis(X, Z):
    """
    Args:
        X : (spots, genes)
        Z : (genes, components)

    Returns:
        phi : (spots, components)
    """
    return X @ Z


def orthogonalize_spatial_basis(phi, K, eps=1e-10):
    """
    Orthogonalize spatial basis under inner product defined by K.

    Args:
        phi : (spots, components)
        K   : (spots, spots)

    Returns:
        phi_ortho : (spots, components)
    """
    # gram matrix
    G = phi.T @ (K @ phi)
    # eigen-decomposition (G is symmetric)
    evals, evecs = eigh(G)
    # regularize small eigenvalues
    evals = np.maximum(evals, eps)
    # G^{-1/2}
    G_inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    # orthogonalized basis
    phi_ortho = phi @ G_inv_sqrt

    return phi_ortho
