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


def inv_sqrt(G, eps=1e-10):
    """
    Compute the inverse square root of a symmetric positive semi-definite matrix.
    M^{-1/2} = U diag(1 / sqrt(evals)) U^T
    """
    evals, evecs = eigh(G)
    evals = np.maximum(evals, eps)
    return evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T


def orthogonalise_spatial_basis(phi, K, V=None):
    """
    Orthogonalise spatial basis under inner product defined by K.

    Args:
        phi : (spots, components)
        K   : (spots, spots)

    Returns:
        phi_ortho : (spots, components)
    """
    # gram matrix
    G = phi.T @ (K @ phi)
    G_inv_sqrt = inv_sqrt(G)
    # orthogonalized basis
    phi_ortho = phi @ G_inv_sqrt

    if V is None:
        return phi_ortho
    return phi_ortho, V @ G_inv_sqrt


def orient_vectors(V):
    """Orient vector V according to its sum of squared components."""
    Vpos = sum((V > 0) * (V**2), 1)
    Vneg = sum((V < 0) * (V**2), 1)
    signs = np.sign(Vpos - Vneg)
    return signs * V


def fractional_energy(phi, eps=1e-12):
    E = phi**2
    return E / (E.sum(axis=1, keepdims=True) + eps)
