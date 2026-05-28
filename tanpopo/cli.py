import argparse

import scanpy as sc
import matplotlib.pyplot as plt

from tanpopo.data import read_anndata
from tanpopo.models import SpatialGeneKPCA
from tanpopo.plot import plot_spatial_basis
from tanpopo.utils import timed, str2bool, print_top_genes_per_basis


def spatial_gene_eigenmodes(
    adata,
    output,
    n_components,
    radius,
    transform,
    min_counts,
    min_spot_fraction,
    target_sum,
    covariates,
    alpha,
    spot_center,
    gene_center,
    plot,
    verbose=False,
):
    W, coords, gene_names, covariates_matrix = read_anndata(
        adata, target_sum, transform, min_counts, min_spot_fraction, covariates
    )

    spot_operator = "sample" if spot_center else "none"
    sgkpca = SpatialGeneKPCA(radius, spot_operator, alpha, gene_center, verbose=verbose)
    sgkpca.fit(W, coords, n_components, covariates=covariates_matrix)

    adata.obsm["tanpopo_spot_modes"] = sgkpca.spot_modes[0]
    adata.varm["tanpopo_eigenvectors"] = sgkpca.eigenvectors
    adata.varm["tanpopo_gene_loadings"] = sgkpca.gene_loadings
    adata.varm["tanpopo_gene_scores"] = sgkpca.gene_scores
    adata.uns["tanpopo"] = {
        "eigenvalues": sgkpca.eigenvalues,
        "preprocessing": {
            "target_sum": target_sum,
            "transform": transform,
            "min_counts": min_counts,
            "min_spot_fraction": min_spot_fraction,
        },
        "cfg": {
            "kernel": "wendland_c2",
            "radius": sgkpca.radius,
            "alpha": sgkpca.alpha,
            "spot_operator": sgkpca.spot_operator,
            "gene_center": sgkpca.gene_center,
            "covariates": covariates,
        },
    }

    if verbose:
        print_top_genes_per_basis(sgkpca.eigenvectors, sgkpca.eigenvalues, gene_names)
    if output:
        adata.write(output)
    if plot:
        for spot_mode in sgkpca.spot_modes:
            plot_spatial_basis(adata, spot_mode, cmap="PiYG", vcenter=0)
        plt.show()

    return adata


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
        "--components",
        type=int,
        default=8,
        help="Number of spatial components",
    )
    parser.add_argument(
        "-r",
        "--radius",
        type=float,
        required=True,
        help="Wendland kernel support radius",
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
        "--covariates",
        type=str,
        choices=[
            "log_total_counts",
            "log_detected_genes",
            "mito_fraction",
            "ribo_fraction",
        ],
        nargs="+",
        help="Include covariates to correct for.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Gene magnitude scaling - uses the operator D^{-alpha} G D^{-alpha}",
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
        default="False",
        help="Center gene weights.",
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


def main():
    args = parse_args()

    with timed("Loading data", args.verbose):
        adata = sc.read_h5ad(args.input)
    if args.verbose:
        print(adata)

    adata = spatial_gene_eigenmodes(
        adata,
        args.output,
        args.components,
        args.radius,
        args.transform,
        args.mincounts,
        args.minspots,
        args.targetsum,
        args.covariates,
        args.alpha,
        str2bool(args.spotcenter),
        str2bool(args.genecenter),
        args.plot,
        args.verbose,
    )


if __name__ == "__main__":
    main()
