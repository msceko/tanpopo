import numpy as np
from scipy.linalg import eigh

from utils import quad_form


def project_spatial_basis(X, Z):
    """
    Args:
        X : (spots, genes)
        Z : (genes, components)

    Returns:
        phi : (spots, components)
    """
    return X @ Z


def inv_sqrt(G, eps=1e-12):
    """
    Compute the inverse square root of a symmetric positive semi-definite matrix.
    M^{-1/2} = U diag(1 / sqrt(evals)) U^T
    """
    evals, evecs = eigh(G)
    evals = np.maximum(evals, eps)
    return evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T


def whiten_eigenmodes(phi, K, V=None):
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
    """Return fractional energy for each eigenmode"""
    E = phi**2
    return E / (E.sum(axis=1, keepdims=True) + eps)


def gene_loading_mass(eigvecs, eigvals=None):
    """Captured gene loading mass - optionally weighted by eigenvalues"""
    if eigvals is not None:
        return (eigvals * eigvecs**2).sum(1)
    return (eigvecs**2).sum(1)


def split_mode_contributions(S, eigvals, eigvecs, use_overlap_penalty=False, eps=1e-12):
    """
    Rank split halves of KPCA modes.

    S: function (p,r)->(p,r) computing S@X (S = W^T K W or cosine-normalized C)
    eigvals: (m,)
    eigvecs: (p,m)

    Returns list of dicts with per-half contributions.
    """
    out = []
    m = eigvecs.shape[1]
    for k in range(m):
        lam = float(eigvals[k])
        v = eigvecs[:, k]
        vp = np.maximum(v, 0.0)
        vn = np.maximum(-v, 0.0)

        Ep = quad_form(S, vp) if vp.any() else 0.0
        En = quad_form(S, vn) if vn.any() else 0.0

        if use_overlap_penalty and vp.any() and vn.any():
            # overlap = vp^T S vn
            Svn = S(vn.reshape(-1, 1)).ravel()
            overlap = float(vp @ Svn)
            Ep_eff = max(Ep - overlap, 0.0)
            En_eff = max(En - overlap, 0.0)
            denom = max(Ep_eff + En_eff, eps)
            cp = lam * Ep_eff / denom
            cn = lam * En_eff / denom
        else:
            denom = max(Ep + En, eps)
            cp = lam * Ep / denom
            cn = lam * En / denom

        Svn = S(vn.reshape(-1, 1)).ravel()
        overlap = float(vp @ Svn)

        out.append(
            {
                "v": vp,
                "mode": k,
                "half": "+",
                "lambda": lam,
                "E": Ep,
                "contrib": cp,
                "overlap": overlap,
            }
        )
        out.append(
            {
                "v": vn,
                "mode": k,
                "half": "-",
                "lambda": lam,
                "E": En,
                "contrib": cn,
                "overlap": overlap,
            }
        )

    # sort by contribution
    out.sort(key=lambda d: d["contrib"], reverse=True)
    return out
