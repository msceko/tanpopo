import argparse
import os

import numpy as np
import scanpy as sc
import squidpy as sq
import matplotlib.pyplot as plt

from data import extract_visium_data, load_visium_hd, load_xenium_binned
from kernel import kernel_matrix_sparse, cosine_kernel_operator
from kpca import kernel_pca_iterative
from basis import (
    project_spatial_basis,
    whiten_eigenmodes,
    orient_vectors,
    split_mode_contributions,
    fractional_energy,
)
from plot import plot_spatial_basis, plot_spatial_basis_signed, plot_cumulative_contribution
from utils import timed, str2bool, normalise_gene_weights, print_top_genes_per_basis


def load_data(fname, platform, verbose=False):
    if platform == "visium":
        if os.path.exists(fname):
            adata = sq.read.visium(fname)
        else:
            adata = sq.datasets.visium(fname)
    elif platform == "visiumhd":
        adata = load_visium_hd(fname)
    elif platform == "xenium":
        if args.bin is None:
            args.bin = args.radius / 3
        adata = load_xenium_binned(fname, args.bin)
    else:
        if os.path.exists(fname):
            adata = sc.read_h5ad(fname)
        else:
            adata = getattr(sq.datasets, fname)()

    adata.var_names_make_unique()

    if verbose:
        print(adata)

    return adata


def spatial_rkhs_gene_basis(
    adata,
    output,
    n_components,
    radius,
    transform,
    min_counts,
    min_spot_fraction,
    target_sum,
    spot_center,
    gene_center,
    cosine_normalise,
    whiten,
    split_modes,
    plot,
    verbose=False,
):
    W, coords, gene_names = extract_visium_data(
        adata, target_sum, transform, min_counts, min_spot_fraction
    )

    with timed("Kernel matrix", verbose):
        K = kernel_matrix_sparse(coords, radius)

    with timed("Cosine matrix", verbose):
        S = cosine_kernel_operator(W, K, spot_center, gene_center, cosine_normalise)

    with timed("Kernel PCA", verbose):
        Z, eigvals, eigvecs = kernel_pca_iterative(S, n_components)
        Z, eigvecs = orient_vectors(Z), orient_vectors(eigvecs)
        phi = project_spatial_basis(W, eigvecs)

    if whiten:
        phi, eigvecs = whiten_eigenmodes(phi, K, eigvecs)

    if split_modes:
        split_modes = split_mode_contributions(S, eigvals, eigvecs, use_overlap_penalty=False)
        split_vecs = np.array([mode["v"] for mode in split_modes]).T
        phi_split = project_spatial_basis(W, split_vecs)

    if verbose:
        print_top_genes_per_basis(eigvecs, eigvals, gene_names)

        if split_modes:
            for k, mode in enumerate(split_modes):
                print(
                    f"Non-negative eigenmode {k}: (λ{mode['half']} = {mode['contrib']:.4f}, k = {mode['mode']})"
                )

    adata.uns["srgem"] = {
        "gene_loadings": eigvecs,
        "gene_energy": eigvals,
        "gene_scores": Z,
        "eigenmodes": phi,
        "info": {
            "target_sum": target_sum,
            "transform": transform,
            "min_counts": min_counts,
            "min_spot_fraction": min_spot_fraction,
            "kernel": "wendland_c2",
            "radius": radius,
            "spot_center": spot_center,
            "gene_center": gene_center,
            "cosine_normalize": cosine_normalise,
            "whiten": whiten,
        },
    }

    if output:
        adata.write(output)
    if plot:
        plot_spatial_basis(adata, phi, cmap="PiYG", vcenter=0)
        if split_modes:
            plot_spatial_basis(adata, phi_split, cmap="viridis")
        # plot_spatial_basis_signed(adata, W, eigvecs)
        # plot_spatial_basis(adata, fractional_energy(phi), "fractional_energy")
        # plot_cumulative_contribution(eigvecs)
        plt.show()

    return adata


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
        "--components",
        type=int,
        default=8,
        help="Number of spatial components",
    )
    parser.add_argument(
        "-r",
        "--radius",
        type=float,
        help="Wendland kernel support radius",
    )
    parser.add_argument(
        "--bin",
        type=float,
        help="Bin size (for xenium)",
    )
    parser.add_argument(
        "--transform",
        type=str,
        choices=["sqrt", "log1p"],
        help="Counts transform",
    )
    parser.add_argument(
        "--mincounts",
        type=int,
        default=10,
        help="Minimum number of counts required for a gene to pass filtering",
    )
    parser.add_argument(
        "--minspots",
        type=float,
        help="Minimum fraction of spots required for a gene to pass filtering [0, 1]",
    )
    parser.add_argument(
        "--targetsum",
        type=int,
        default=1e4,
        help="Normalise each spot total to the target sum",
    )
    parser.add_argument(
        "--platform",
        type=str,
        choices=["visium", "visiumhd", "xenium"],
        help="Spatial platform.",
    )
    parser.add_argument(
        "--spotcenter",
        type=str,
        choices=["True", "False"],
        default="True",
        help="Center spatial kernel.",
    )
    parser.add_argument(
        "--genecenter",
        choices=["True", "False"],
        default="True",
        help="Center gene weights.",
    )
    parser.add_argument(
        "--normalise",
        choices=["True", "False"],
        default="True",
        help="Apply cosine normalisation to Gram matrix.",
    )
    parser.add_argument(
        "--whiten",
        action="store_true",
        help="Whiten spatial eigenmodes",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Split modes into positive and negative components",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot spatial gene basis",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Time and print each step",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with timed("Loading data", args.verbose):
        adata = load_data(args.input, args.platform, args.verbose)

    adata = spatial_rkhs_gene_basis(
        adata,
        args.output,
        args.components,
        args.radius,
        args.transform,
        args.mincounts,
        args.minspots,
        args.targetsum,
        str2bool(args.spotcenter),
        str2bool(args.genecenter),
        str2bool(args.normalise),
        args.whiten,
        args.split,
        args.plot,
        args.verbose,
    )
