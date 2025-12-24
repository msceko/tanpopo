import argparse
import numpy as np
import scanpy as sc
import squidpy as sq
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from sklearn.decomposition import KernelPCA


def extract_visium_data(
    adata, normalise=True, transform=None, layer=None, min_counts=10, sparse=False
):
    """
    Returns:
        X : (n_spots, n_genes) expression matrix
        coords : (n_spots, 2) spatial coordinates
        gene_names
    """
    if normalise:
        sc.pp.normalize_total(adata, target_sum=1e4)
    if transform == "log1p":
        sc.pp.log1p(adata)
    elif transform == "sqrt":
        sc.pp.sqrt(adata)

    if layer is None:
        X = adata.X
    else:
        X = adata.layers[layer]

    if sparse and not sp.issparse(X):
        X = sp.csr_matrix(X)
    elif not sparse and sp.issparse(X):
        X = X.toarray()

    coords = adata.obsm["spatial"].astype(np.float64)
    gene_names = np.array(adata.var_names)

    # Filter low-count genes
    gene_mask = np.array(X.sum(axis=0)).ravel() >= min_counts
    # gene_mask = (adata.X.toarray() > 0).mean(axis=0) >= 0.05
    X = X[:, gene_mask]
    gene_names = gene_names[gene_mask]

    return X, coords, gene_names


def normalise_gene_weights(X):
    """Normalise columns to sum to 1"""
    return X / X.sum(axis=0, keepdims=True)


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
    eigvals, eigvecs = np.linalg.eigh(G)
    idx = np.argsort(eigvals)[::-1]

    eigvals = eigvals[idx][:n_components]
    eigvecs = eigvecs[:, idx][:, :n_components]

    return eigvals, eigvecs


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
    # Gram matrix
    G = phi.T @ K @ phi
    # Eigen-decomposition (G is symmetric)
    evals, evecs = eigh(G)
    # Regularize small eigenvalues
    evals = np.maximum(evals, eps)
    # G^{-1/2}
    G_inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    # Orthogonalized basis
    phi_ortho = phi @ G_inv_sqrt

    return phi_ortho


def top_genes_per_basis(Z, genes, n_top):
    """Compute top genes for each gene basis"""
    top_genes = []
    for k in range(Z.shape[1]):
        idx = np.argsort(np.abs(Z[:, k]))[::-1][:n_top]
        top_genes.append({genes[i]: Z[i, k] for i in idx})
    return top_genes


def print_top_genes_per_basis(Z, genes, n_top=8):
    """Print top genes for each gene basis"""
    top_genes = top_genes_per_basis(Z, genes, n_top)
    for k in range(Z.shape[1]):
        print(f"\nBasis {k}")
        for g, w in top_genes[k].items():
            print(f"{g:15s} {w:+.3f}")


def plot_spatial_basis(adata, phi, prefix="spatial_basis"):
    """Plot spatial gene basis"""
    keys = []
    for k in range(phi.shape[1]):
        keys.append(f"{prefix}_{k}")
        adata.obs[keys[k]] = phi[:, k]

    sq.pl.spatial_scatter(adata, color=keys)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Spatial RKHS Gene Basis")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Visium data path",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Save .h5ad-formatted hdf5 file",
    )
    parser.add_argument(
        "-s",
        "--sigma",
        type=float,
        help="Standard deviation of Gaussian kernel",
    )
    parser.add_argument(
        "--components",
        type=int,
        default=8,
        help="Number of spatial components",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot spatial gene basis",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    adata = sq.read.visium(args.input)
    adata.var_names_make_unique()
    X, coords, gene_names = extract_visium_data(adata, transform="sqrt")

    W = normalise_gene_weights(X)
    K = gaussian_kernel(coords, args.sigma)
    S = cs_kernel(W, K)
    Z, eigvals, eigvecs = gene_kpca(S, args.components)
    # eigvals, eigvecs = kernel_pca(C, spatial_components)
    # Z = eigvecs * np.sqrt(eigvals)
    phi = project_spatial_basis(X, eigvecs)
    phi = orthogonalize_spatial_basis(phi, K)

    print_top_genes_per_basis(Z, gene_names)

    adata.uns["spatial_basis_genes"] = gene_names
    adata.obsp["spatial_kernel"] = K
    adata.uns["gene_cosine_rkhs"] = S
    adata.uns["spatial_gene_basis"] = eigvecs
    adata.uns["spatial_gene_eigvals"] = eigvals
    adata.uns["spatial_gene_scores"] = Z

    if args.output:
        adata.write(args.output)
    if args.plot:
        plot_spatial_basis(adata, phi)
