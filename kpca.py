import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import KernelPCA


def kpca(K, n_components, eigen_solver="auto"):
    """
    Kernel Principal component analysis using sklearn.
    Does not require centering.

    Args:
        K : (spots, spots)
        n_components : int

    Returns:
        gene_scores : (genes, components)
        eigenvalues : (components,)
        eigenvectors : (genes, components)
    """
    kpca = KernelPCA(n_components=n_components, kernel="precomputed", eigen_solver=eigen_solver, n_jobs=-1)
    Z = kpca.fit_transform(K)
    return Z, kpca.eigenvalues_, kpca.eigenvectors_


def center_gram(G):
    """
    Double-centers a Gram matrix in feature space: Gc = H G H.
    """
    n = G.shape[0]
    one = np.ones((n, 1), dtype=G.dtype)
    H = np.eye(n, dtype=G.dtype) - (one @ one.T) / n
    return H @ G @ H


def kernel_pca(G, n_components):
    """
    Eigen-decomposition of gene kernel.
    Requires cetered gram matrix.

    Returns:
        eigenvalues : (components,)
        eigenvectors : (genes, components)
    """
    G = center_gram(G)
    eigvals, eigvecs = eigh(G)
    idx = np.argsort(eigvals)[::-1]

    eigvals = eigvals[idx][:n_components]
    eigvecs = eigvecs[:, idx][:, :n_components]
    Z = eigvecs * np.sqrt(eigvals)

    return Z, eigvals, eigvecs

def kernel_pca_iterative(A, n_components, tol=0, maxiter=None):
    """
    Top eigenpairs of symmetric LinearOperator A via ARPACK (eigsh).

    Returns:
      eigvals: (spots,)
      eigvecs: (genes, spots)
      Z:       (genes, spots) KPCA coords = eigvecs * sqrt(eigvals)
    """
    eigvals, eigvecs = eigsh(A, k=n_components, which="LA", tol=tol, maxiter=maxiter)
    idx = np.argsort(eigvals)[::-1]

    eigvals = np.maximum(eigvals[idx], 0.0)
    eigvecs = eigvecs[:, idx]
    Z = eigvecs * np.sqrt(eigvals[None, :])

    return Z, eigvals, eigvecs
