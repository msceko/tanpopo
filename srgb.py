import argparse
import squidpy as sq
import matplotlib.pyplot as plt

from data import extract_visium_data, load_visium_hd
from kernel import gaussian_kernel_sparse, cs_kernel_operator
from kpca import kernel_pca_iterative
from basis import project_spatial_basis, orthogonalise_spatial_basis, orient_vectors
from utils import (
    timed,
    normalise_gene_weights,
    print_top_genes_per_basis,
    plot_spatial_basis,
    plot_spatial_basis_signed,
    plot_cumulative_contribution,
)


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
        nargs="+",
        help="Standard deviation of Gaussian kernel",
    )
    parser.add_argument(
        "-b",
        "--beta",
        type=float,
        nargs="+",
        help="Scaling factors for each sigma",
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
        "--transform",
        type=str,
        choices=["sqrt", "log1p"],
        help="Counts transform",
    )
    parser.add_argument(
        "--platform",
        type=str,
        choices=["visium", "visium_hd"],
        default="visium",
        help="Spatial platform.",
    )
    parser.add_argument(
        "--orthogonalise",
        action="store_true",
        help="Orthonalise basis spatially.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot spatial gene basis",
    )
    parser.add_argument(
        "--lowmem",
        action="store_false",
        help="Low memory kernel construction (slower)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Time and print each step.",
    )
    return parser.parse_args()


def spatial_rkhs_gene_basis(
    adata,
    output,
    sigma,
    beta,
    n_components,
    radius,
    transform,
    orthogonalise_basis,
    plot,
    precompute_KW=True,
    verbose=False,
):
    X, coords, gene_names = extract_visium_data(adata, transform=transform)

    with timed("Kernel matrix", verbose):
        W = normalise_gene_weights(X)
        K = gaussian_kernel_sparse(coords, sigma, beta, radius)

    with timed("Cosine matrix", verbose):
        S = cs_kernel_operator(W, K, precompute_KW=precompute_KW)

    with timed("Kernel PCA", verbose):
        Z, eigvals, eigvecs = kernel_pca_iterative(S, n_components)
        Z, eigvecs = orient_vectors(Z), orient_vectors(eigvecs)
        phi = project_spatial_basis(X, eigvecs)

    if orthogonalise_basis:
        phi, eigvecs = orthogonalise_spatial_basis(phi, K, eigvecs)

    if verbose:
        print_top_genes_per_basis(eigvecs, eigvals, gene_names)

    adata.uns["spatial_basis_genes"] = gene_names
    adata.uns["spatial_gene_loadings"] = eigvecs
    adata.uns["spatial_gene_eigvals"] = eigvals
    adata.uns["spatial_gene_scores"] = Z
    adata.uns["spatial_gene_basis"] = phi

    if output:
        adata.write(output)
    if plot:
        plot_spatial_basis(adata, phi)
        plot_spatial_basis_signed(adata, X, eigvecs)
        plot_cumulative_contribution(eigvecs)
        plt.show()


if __name__ == "__main__":
    args = parse_args()

    with timed("Loading data", args.verbose):
        if args.platform == "visium":
            adata = sq.read.visium(args.input)
        elif args.platform == "visium_hd":
            adata = load_visium_hd(args.input)
        adata.var_names_make_unique()

    spatial_rkhs_gene_basis(
        adata,
        args.output,
        args.sigma,
        args.beta,
        args.components,
        args.radius,
        args.transform,
        args.orthogonalise,
        args.plot,
        args.lowmem,
        args.verbose,
    )
