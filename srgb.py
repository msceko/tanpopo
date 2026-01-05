import argparse
import squidpy as sq

from data import extract_visium_data
from kernel import gaussian_kernel, cs_kernel
from kpca import gene_kpca
from basis import project_spatial_basis, orthogonalize_spatial_basis
from utils import normalise_gene_weights, print_top_genes_per_basis, plot_spatial_basis


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
        "--transform",
        type=str,
        choices=["sqrt", "log1p"],
        help="Counts transform",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot spatial gene basis",
    )
    return parser.parse_args()


def spatial_rkhs_gene_basis(fname, output, sigma, n_components, transform, plot):
    adata = sq.read.visium(fname)
    adata.var_names_make_unique()
    X, coords, gene_names = extract_visium_data(adata, transform=transform)

    W = normalise_gene_weights(X)
    K = gaussian_kernel(coords, sigma)
    S = cs_kernel(W, K)
    Z, eigvals, eigvecs = gene_kpca(S, n_components)
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

    if output:
        adata.write(output)
    if plot:
        plot_spatial_basis(adata, phi)


if __name__ == "__main__":
    args = parse_args()
    spatial_rkhs_gene_basis(
        args.input, args.output, args.sigma, args.components, args.transform, args.plot
    )
