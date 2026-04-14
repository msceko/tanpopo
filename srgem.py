import argparse
import os

import scanpy as sc
import squidpy as sq
import matplotlib.pyplot as plt

from data import extract_visium_data, load_visium_hd, load_xenium_binned
from kpca import SpatialGeneKPCA
from plot import plot_spatial_basis
from utils import timed, str2bool, print_top_genes_per_basis


def load_data(fname, platform):
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

    return adata


def spatial_rkhs_gene_eigenmodes(
    adata,
    output,
    n_components,
    radius,
    transform,
    min_counts,
    min_spot_fraction,
    target_sum,
    alpha,
    spot_center,
    gene_center,
    cosine_normalise,
    plot,
    verbose=False,
):
    W, coords, gene_names = extract_visium_data(
        adata, target_sum, transform, min_counts, min_spot_fraction
    )

    sgkpca = SpatialGeneKPCA(
        radius, alpha, spot_center, gene_center, cosine_normalise, verbose=verbose
    )
    sgkpca.fit(W, coords, n_components)

    adata.uns["srgem"] = sgkpca.summary()
    adata.uns["srgem"]["preprocessing"] = {
        "target_sum": target_sum,
        "transform": transform,
        "min_counts": min_counts,
        "min_spot_fraction": min_spot_fraction,
    }

    if verbose:
        print_top_genes_per_basis(sgkpca.eigenvectors, sgkpca.eigenvalues, gene_names)
    if output:
        adata.write(output)
    if plot:
        plot_spatial_basis(adata, sgkpca.spot_modes, cmap="PiYG", vcenter=0)
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
        "--normalise",
        choices=["True", "False"],
        default="True",
        help="Apply cosine normalisation to Gram matrix.",
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
        adata = load_data(args.input, args.platform)
    if args.verbose:
        print(adata)

    adata = spatial_rkhs_gene_eigenmodes(
        adata,
        args.output,
        args.components,
        args.radius,
        args.transform,
        args.mincounts,
        args.minspots,
        args.targetsum,
        args.alpha,
        str2bool(args.spotcenter),
        str2bool(args.genecenter),
        str2bool(args.normalise),
        args.plot,
        args.verbose,
    )
