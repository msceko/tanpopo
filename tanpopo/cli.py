import argparse

import scanpy as sc
import matplotlib.pyplot as plt

from tanpopo.clustering import cluster_cmd
from tanpopo.data import read_anndata
from tanpopo.models import SpatialGeneKPCA
from tanpopo.plot import plot_spatial_basis
from tanpopo.utils import timed, str2bool, print_top_genes_per_basis


def programs_cmd(args):
    with timed("Loading data", args.verbose):
        adata = sc.read_h5ad(args.input)
    if args.verbose:
        print(adata)
    W, coords, gene_names, covariates_matrix = read_anndata(
        adata,
        args.target_sum,
        args.transform,
        args.min_counts,
        args.min_spot_fraction,
        args.covariates,
    )

    args.spot_center = str2bool(args.spot_center)
    args.gene_center = str2bool(args.gene_center)
    spot_operator = "sample" if args.spot_center else "none"
    sgkpca = SpatialGeneKPCA(
        args.radius, spot_operator, args.alpha, args.gene_center, verbose=args.verbose
    )
    sgkpca.fit(W, coords, args.components, covariates=covariates_matrix)

    adata.obsm["tanpopo_spot_modes"] = sgkpca.spot_modes[0]
    adata.varm["tanpopo_eigenvectors"] = sgkpca.eigenvectors
    adata.varm["tanpopo_gene_loadings"] = sgkpca.gene_loadings
    adata.varm["tanpopo_gene_scores"] = sgkpca.gene_scores
    adata.uns["tanpopo"] = {
        "eigenvalues": sgkpca.eigenvalues,
        "preprocessing": {
            "target_sum": args.target_sum,
            "transform": args.transform,
            "min_counts": args.min_counts,
            "min_spot_fraction": args.min_spot_fraction,
        },
        "cfg": {
            "kernel": "wendland_c2",
            "radius": sgkpca.radius,
            "alpha": sgkpca.alpha,
            "spot_operator": sgkpca.spot_operator,
            "gene_center": sgkpca.gene_center,
            "covariates": args.covariates,
        },
    }

    if args.verbose:
        print_top_genes_per_basis(sgkpca.eigenvectors, sgkpca.eigenvalues, gene_names)
    if args.output:
        adata.write(args.output)
    if args.plot:
        for spot_mode in sgkpca.spot_modes:
            plot_spatial_basis(adata, spot_mode, cmap="PiYG", vcenter=0)
        plt.show()

    return adata


def io_args(parser):
    group = parser.add_argument_group("input/output")
    group.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Input .h5ad-formatted hdf5 file",
    )
    group.add_argument(
        "-o",
        "--output",
        type=str,
        help="Save .h5ad-formatted hdf5 file",
    )
    return group


def io_keys(group):
    group.add_argument(
        "--layer",
        default=None,
        help="AnnData layer to use instead of X",
    )
    group.add_argument(
        "--label-key",
        default=None,
        help="obs column defining labels/cell types",
    )
    return group


def preprocessing_args(parser):
    group = parser.add_argument_group("preprocessing")
    group.add_argument(
        "--transform",
        type=str,
        choices=["sqrt", "log1p"],
        help="Counts transform",
    )
    group.add_argument(
        "--min-counts",
        type=int,
        default=10,
        help="Minimum number of counts required for a gene to pass filtering",
    )
    group.add_argument(
        "--min-spot-fraction",
        type=float,
        help="Minimum fraction of spots required for a gene to pass filtering [0, 1]",
    )
    group.add_argument(
        "--target-sum",
        type=int,
        default=1e4,
        help="Normalise each spot total to the target sum",
    )
    group.add_argument(
        "--covariates",
        type=str,
        choices=[
            "log_total_counts",
            "log_detected_genes",
            "mito_fraction",
            "ribo_fraction",
        ],
        nargs="+",
        help="Include covariates to correct for",
    )
    return group


def model_args(parser):
    group = parser.add_argument_group("model")
    group.add_argument(
        "--components",
        type=int,
        default=8,
        help="Number of spatial components",
    )
    group.add_argument(
        "-r",
        "--radius",
        type=float,
        required=True,
        help="Wendland kernel support radius",
    )
    group.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Gene magnitude scaling exponent",
    )
    group.add_argument(
        "--spot-center",
        type=str,
        choices=["True", "False"],
        default="True",
        help="Center spatial kernel.",
    )
    group.add_argument(
        "--gene-center",
        choices=["True", "False"],
        default="False",
        help="Center gene weights.",
    )
    return group


def clustering_args(parser):
    group = parser.add_argument_group("clustering")
    group.add_argument(
        "--by",
        type=str,
        required=True,
        choices=["spots", "genes"],
        help="Dimension along which to cluster",
    )
    group.add_argument(
        "--neighbours",
        type=int,
        default=15,
        help="Number of neighbours",
    )
    group.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Leiden resolution",
    )
    group.add_argument(
        "--metric",
        type=str,
        default="cosine",
        help="Clustering metric",
    )
    group.add_argument(
        "--ngenes",
        type=int,
        help="Filter top n genes",
    )
    return group


def flag_args(parser):
    group = parser.add_argument_group("flags")
    group.add_argument(
        "--plot",
        action="store_true",
    )
    group.add_argument(
        "--verbose",
        action="store_true",
    )
    return group


def build_parser():
    parser = argparse.ArgumentParser(
        prog="tanpopo",
        description="Spatial gene eigenmode workflows for spatial transcriptomics",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    arg_parser = sub.add_parser("programs", help="Compute spatial gene programs/eigenmodes")
    io_group = io_args(arg_parser)
    io_group = io_keys(io_group)
    preprocessing_group = preprocessing_args(arg_parser)
    model_group = model_args(arg_parser)
    flag_group = flag_args(arg_parser)
    arg_parser.set_defaults(func=programs_cmd)

    arg_parser = sub.add_parser("cluster", help="Cluster spots or genes based on eigenmodes")
    io_group = io_args(arg_parser)
    clustering_group = clustering_args(arg_parser)
    flag_group = flag_args(arg_parser)
    flag_group.add_argument("--umap", action="store_true")
    arg_parser.set_defaults(func=cluster_cmd)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
