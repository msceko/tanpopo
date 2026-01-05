import numpy as np
from scipy.linalg import eigh
from sklearn.decomposition import KernelPCA


def gene_kpca(K, n_components, eigen_solver="auto"):
    """
    Kernel Principal component analysis in gene RKHS

    Args:
        K : (spots, spots)
        n_components : int

    Returns:
        gene_scores : (genes, components)
        eigenvalues : (components,)
        eigenvectors : (genes, components)
    """
    kpca = KernelPCA(n_components=n_components, kernel="precomputed", eigen_solver=eigen_solver)
    Z = kpca.fit_transform(K)
    return Z, kpca.eigenvalues_, kpca.eigenvectors_


def center_gram(G):
    """
    Double-centers a Gram matrix in feature space: Gc = H G H
    """
    n = G.shape[0]
    one = np.ones((n, 1), dtype=G.dtype)
    H = np.eye(n, dtype=G.dtype) - (one @ one.T) / n
    return H @ G @ H


def kernel_pca(G, n_components):
    """
    Eigen-decomposition of gene kernel.

    Returns:
        eigenvalues : (components,)
        eigenvectors : (genes, components)
    """
    G = center_gram(G)
    eigvals, eigvecs = eigh(G)
    idx = np.argsort(eigvals)[::-1]

    eigvals = eigvals[idx][:n_components]
    eigvecs = eigvecs[:, idx][:, :n_components]

    return eigvals, eigvecs
