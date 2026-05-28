import anndata as ad
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from tanpopo.plot import plot_gene_clusters, plot_umap
from tanpopo.utils import timed, argtop


def cluster_cmd(args):
    """Cluster spots or genes based on eigenmodes"""
    with timed("Loading data", args.verbose):
        adata = sc.read_h5ad(args.input)

    # filter top n genes
    if args.ngenes:
        idx = argtop((adata.varm["tanpopo_gene_scores"] ** 2).sum(1), args.ngenes, mode="pos")
        adata = adata[:, idx].copy()

    # add dictionary for clustering metadata
    if adata.uns["tanpopo"].get("clustering") is None:
        adata.uns["tanpopo"]["clustering"] = {}

    adata.uns["tanpopo"]["clustering"][args.by] = {
        "n_neighbours": args.neighbours,
        "resolution": args.resolution,
        "metric": args.metric,
    }

    if args.by == "spots":
        cluster_spots(args, adata)
    else:
        cluster_genes(args, adata)


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


def cluster_spots(args, adata):
    """Cluster spots based on spatial eigenmodes"""

    with timed("Spot clustering", args.verbose):
        adata.obs["tanpopo_leiden"] = cluster_leiden(
            adata.obsm["tanpopo_spot_modes"],
            args.neighbours,
            args.resolution,
            args.metric,
        )

    if args.plot:
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
    if args.umap:
        plot_umap(
            adata.obsm["tanpopo_spot_modes"],
            adata.obs["tanpopo_leiden"],
            adata.obs_names,
            args.neighbours,
        )
    plt.show()

    if args.output:
        adata.write(args.output)


def cluster_genes(args, adata):
    """Cluster genes based on eigendecomposition gene scores"""

    with timed("Gene clustering", args.verbose):
        adata.var["tanpopo_leiden"] = cluster_leiden(
            adata.varm["tanpopo_gene_scores"],
            args.neighbours,
            args.resolution,
            args.metric,
        )

    if args.plot:
        plot_gene_clusters(adata, key="tanpopo_leiden")
    if args.umap:
        plot_umap(
            adata.varm["tanpopo_gene_scores"],
            adata.var["tanpopo_leiden"],
            adata.var_names,
            args.neighbours,
        )
    plt.show()

    if args.output:
        adata.write(args.output)
