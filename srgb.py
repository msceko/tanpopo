import argparse
import os
import scanpy as sc
import squidpy as sq
import matplotlib.pyplot as plt

from data import extract_visium_data, load_visium_hd, load_xenium_binned
from kernel import kernel_matrix_sparse, cs_kernel_operator
from kpca import kernel_pca_iterative
from basis import (
    project_spatial_basis,
    whiten_eigenmodes,
    orient_vectors,
    fractional_energy,
)
from plot import plot_spatial_basis, plot_spatial_basis_signed, plot_cumulative_contribution
from utils import timed, normalise_gene_weights, print_top_genes_per_basis


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
            args.bin = args.sigma[0]
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
    sigma,
    n_components,
    radius,
    transform,
    whiten,
    plot,
    verbose=False,
):
    W, coords, gene_names = extract_visium_data(adata, transform=transform)

    with timed("Kernel matrix", verbose):
        K = kernel_matrix_sparse(coords, sigma, radius)

    with timed("Cosine matrix", verbose):
        S, KW = cs_kernel_operator(W, K)

    with timed("Kernel PCA", verbose):
        Z, eigvals, eigvecs = kernel_pca_iterative(S, n_components)
        Z, eigvecs = orient_vectors(Z), orient_vectors(eigvecs)
        phi = project_spatial_basis(W, eigvecs)

    if whiten:
        phi, eigvecs = whiten_eigenmodes(phi, K, eigvecs)

    if verbose:
        print_top_genes_per_basis(eigvecs, eigvals, gene_names)

    adata.uns["spatial_gene_loadings"] = eigvecs
    adata.uns["spatial_gene_energy"] = eigvals
    adata.uns["spatial_gene_scores"] = Z
    adata.uns["spatial_eigenmodes"] = phi
    adata.uns["KW"] = KW

    if output:
        adata.write(output)
    if plot:
        plot_spatial_basis(adata, phi)
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
        "--radius",
        type=float,
        help="Radius of neighbourhood for spatial computation",
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
        "--platform",
        type=str,
        choices=["visium", "visiumhd", "xenium"],
        help="Spatial platform.",
    )
    parser.add_argument(
        "--whiten",
        action="store_true",
        help="Whiten spatial eigenmodes",
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
        args.sigma,
        args.components,
        args.radius,
        args.transform,
        args.whiten,
        args.plot,
        args.verbose,
    )
