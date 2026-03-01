import anndata as ad
import numpy as np
import scanpy as sc
import squidpy as sq
import matplotlib.pyplot as plt

from basis import project_spatial_basis
from utils import cumulative_contribution


def spatial_scatter(adata, keys):
    n_cols = int(np.round(4 / 3 * np.sqrt(len(keys))))
    axs = sc.pl.embedding(
        adata, basis="spatial", color=keys, ncols=n_cols, cmap="PiYG", frameon=False, show=False
    )
    for ax in axs:
        ax.invert_yaxis()


def plot_spatial_basis(adata, phi, prefix="spatial_basis"):
    """Plot spatial gene basis"""
    keys = []
    for k in range(phi.shape[1]):
        keys.append(f"{prefix}_{k}")
        adata.obs[keys[k]] = phi[:, k]

    spatial_scatter(adata, keys)


def plot_spatial_basis_signed(adata, X, eigvecs):
    """Plot positive and negative components of gene basis independently"""
    phi_pos = project_spatial_basis(X, np.maximum(eigvecs, 0))
    plot_spatial_basis(adata, phi_pos, "spatial_basis_positive")
    phi_neg = project_spatial_basis(X, np.maximum(-eigvecs, 0))
    plot_spatial_basis(adata, phi_neg, "spatial_basis_negative")


def plot_gene_clusters(adata, key="leiden"):
    """Plot spatial distributions of gene clusters"""
    keys = []
    for cluster in range(adata.var[key].max() + 1):
        keys.append(f"gene_cluster_{cluster}")
        genes = (adata.var[key] == cluster).tolist()
        adata.obs[keys[cluster]] = np.asarray(adata.X[:, genes].sum(axis=1)).ravel()

    spatial_scatter(adata, keys)


def plot_cumulative_contribution(eigvecs):
    """Plot cumulative distributions for each squared eigenvector"""
    plt.figure()
    plt.plot(cumulative_contribution(eigvecs))
    plt.xlabel("Number of genes")
    plt.ylabel("Cumulative contribution")
    plt.axhline(0.8, linestyle="--", alpha=0.5)
    plt.legend([f"Basis {n}" for n in range(eigvecs.shape[1])])


def plot_umap(X, obs, obs_names, n_neighbors):
    """Plot umap embedding with cluster colours"""
    adata = ad.AnnData(X)
    adata.obs_names = obs_names
    adata.obs["umap"] = obs

    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X")
    sc.tl.umap(adata)
    sc.pl.umap(adata, color="umap", size=20, cmap="tab20")
