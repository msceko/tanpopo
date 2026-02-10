import argparse

import anndata as ad
import numpy as np
import scanpy as sc
import squidpy as sq
import matplotlib.pyplot as plt

from plot import plot_gene_clusters, plot_umap
from utils import timed


def choose_k_by_energy(eigvals, energy=0.9):
    """
    Choose smallest k such that cumulative explained 'kernel variance' >= energy.
    Using eigvals of centered Gram.
    """
    if eigvals.size == 0:
        return 0
    cum = np.cumsum(eigvals)
    total = cum[-1]
    if total <= 0:
        return 0
    k = int(np.searchsorted(cum / total, energy) + 1)
    return k


def choose_k_by_elbow(eigvals, k_max=None):
    """
    Lightweight elbow finder on log-eigvals curve:
    pick k where second-difference is most negative (strongest curvature).
    """
    if eigvals.size < 3:
        return int(eigvals.size)
    if k_max is None:
        k_max = eigvals.size
    y = np.log(np.clip(eigvals[:k_max], 1e-30, None))
    # discrete second derivative
    d2 = y[:-2] - 2 * y[1:-1] + y[2:]
    k = int(np.argmin(d2) + 2)  # +2 to map to component index (1-based)
    return max(2, min(k, k_max))


def choose_k(eigvals, method="auto", energy=0.9, k_max=50):
    """
    method:
      - "energy": energy threshold
      - "elbow": curvature elbow
      - "auto": min(elbow, energy-based) with sensible bounds
    """
    eigvals = np.asarray(eigvals, dtype=np.float64)
    eigvals = eigvals[: min(k_max, eigvals.size)]
    if eigvals.size == 0:
        return 0

    k_e = choose_k_by_energy(eigvals, energy=energy)
    k_l = choose_k_by_elbow(eigvals, k_max=eigvals.size)

    if method == "energy":
        return k_e
    if method == "elbow":
        return k_l
    # auto: take the more conservative (smaller) but at least 2
    return int(max(2, min(k_e, k_l, eigvals.size)))


def cluster_leiden(X, n_neighbors=15, resolution=1.0, random_state=0):
    """
    Leiden on kNN graph built in program space.
    Returns integer cluster labels of length n_spots.
    """
    adata = ad.AnnData(X)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X", random_state=random_state)
    sc.tl.leiden(adata, resolution=resolution, random_state=random_state, key_added="leiden")
    labels = adata.obs["leiden"].astype(int).to_numpy()
    return labels


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
        "--ngenes",
        type=int,
        default=15,
        help="Number of gene neighbours",
    )
    parser.add_argument(
        "--nspots",
        type=int,
        default=15,
        help="Number of spot neighbours",
    )
    parser.add_argument(
        "--generes",
        type=float,
        default=1.0,
        help="Leiden resolution for genes",
    )
    parser.add_argument(
        "--spotres",
        type=float,
        default=1.0,
        help="Leiden resolution for spots",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot spatial gene basis",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with timed("Loading data", enabled=True):
        adata = sc.read_h5ad(args.input)

    with timed("Gene clustering", enabled=True):
        adata.uns["gene_features"] = (adata.uns["spatial_eigenmodes"].T @ adata.uns["KW"]).T
        adata.var["leiden"] = cluster_leiden(
            adata.uns["gene_features"], n_neighbors=args.ngenes, resolution=args.generes
        )

    with timed("Spot clustering", enabled=True):
        adata.obs["leiden"] = cluster_leiden(
            adata.uns["spatial_eigenmodes"], n_neighbors=args.nspots, resolution=args.spotres
        )

    if args.output:
        adata.write(args.output)

    if args.plot:
        plot_gene_clusters(adata, key="leiden")
        sq.pl.spatial_scatter(adata, color="leiden", img=None, cmap="tab20")
        plot_umap(adata.uns["gene_features"], adata.var["leiden"], adata.var_names, args.ngenes)
        # plot_umap(adata.uns["spatial_gene_basis"], adata.obs["leiden"], adata.obs_names, args.nspots)
        plt.show()
