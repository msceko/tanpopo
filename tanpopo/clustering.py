import argparse

import anndata as ad
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from tanpopo.plot import plot_gene_clusters, plot_umap
from tanpopo.utils import timed, argtop


def parse_args():
    parser = argparse.ArgumentParser(description="Spatial RKHS Gene Basis")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Input .h5ad-formatted hdf5 file with spatial basis information",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Save .h5ad-formatted hdf5 file",
    )
    parser.add_argument(
        "--neighbours",
        type=int,
        default=15,
        help="Number of neighbours",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Leiden resolution",
    )
    parser.add_argument(
        "--ngenes",
        type=int,
        help="Filter top n genes",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        help="Clustering metric",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot spatial gene basis",
    )
    parser.add_argument(
        "--umap",
        action="store_true",
        help="Compute and plot umap",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Time and print each step",
    )
    return parser.parse_args()


def setup(mode):
    args = parse_args()
    with timed("Loading data", args.verbose):
        adata = sc.read_h5ad(args.input)

    if args.ngenes:
        idx = argtop(
            (adata.varm["tanpopo_gene_scores"] ** 2).sum(1), args.ngenes, mode="pos"
        )
        adata = adata[:, idx].copy()

    if adata.uns["tanpopo"].get("clustering") is None:
        adata.uns["tanpopo"]["clustering"] = {}

    adata.uns["tanpopo"]["clustering"][mode] = {
        "n_neighbours": args.neighbours,
        "resolution": args.resolution,
        "metric": args.metric,
    }

    return args, adata


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
    sc.tl.leiden(
        adata, resolution=resolution, random_state=random_state, key_added="leiden"
    )
    labels = adata.obs["leiden"].astype(int).to_numpy()
    return pd.Categorical(labels, ordered=True)


def cluster_spots():
    args, adata = setup("spots")

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


def cluster_genes():
    args, adata = setup("genes")

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
