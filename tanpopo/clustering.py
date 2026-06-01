import anndata as ad
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from tanpopo.plot import plot_gene_clusters, plot_umap
from tanpopo.utils import timed


def cluster_leiden(X, n_neighbors=15, resolution=1.0, metric="cosine", random_state=0):
    """
    Leiden on kNN graph built in program space.
    Returns integer cluster labels of length n_spots.
    """
    adata = ad.AnnData(X)
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep="X",
        metric=metric,
        random_state=random_state,
    )
    sc.tl.leiden(adata, resolution=resolution, random_state=random_state, key_added="leiden")
    labels = adata.obs["leiden"].astype(int).to_numpy()
    return pd.Categorical(labels, ordered=True)


def cluster_spots(adata, neighbours, resolution, metric, plot, umap, verbose):
    """Cluster spots based on spatial eigenmodes"""

    with timed("Spot clustering", verbose):
        adata.obs["tanpopo_leiden"] = cluster_leiden(
            adata.obsm["tanpopo_spot_modes"], neighbours, resolution, metric
        )

    if plot:
        ax = sc.pl.embedding(
            adata,
            basis="spatial",
            color="tanpopo_leiden",
            palette="tab20",
            frameon=False,
            show=False,
        )
        ax.invert_yaxis()
        ax.axis("equal")
    if umap:
        plot_umap(
            adata.obsm["tanpopo_spot_modes"],
            adata.obs["tanpopo_leiden"],
            adata.obs_names,
            neighbours,
        )
    plt.show()

    return adata


def cluster_genes(adata, neighbours, resolution, metric, plot, umap, verbose):
    """Cluster genes based on eigendecomposition gene scores"""

    with timed("Gene clustering", verbose):
        adata.var["tanpopo_leiden"] = cluster_leiden(
            adata.varm["tanpopo_gene_scores"], neighbours, resolution, metric
        )

    if plot:
        plot_gene_clusters(adata, key="tanpopo_leiden")
    if umap:
        plot_umap(
            adata.varm["tanpopo_gene_scores"],
            adata.var["tanpopo_leiden"],
            adata.var_names,
            neighbours,
        )
    plt.show()

    return adata
