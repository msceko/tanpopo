import scanpy as sc
import matplotlib.pyplot as plt
import typer


from tanpopo.data import read_anndata
from tanpopo.models import SpatialGeneKPCA
from tanpopo.clustering import cluster_genes, cluster_spots
from tanpopo.plot import plot_spatial_basis
from tanpopo.utils import timed, argtop, print_top_genes_per_basis
from tanpopo.cli import *

app = typer.Typer(
    name="tanpopo",
    help="Spatial gene eigenmode workflows for spatial transcriptomics.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.command()
def fit(
    fname: InputPath,
    radius: Radius,
    output: OutputPath = None,
    n_components: Components = 8,
    transform: Transform = TransformTypes.log1p,
    min_counts: MinCounts = 10,
    min_spot_fraction: MinSpotFraction = None,
    target_sum: TargetSum = 1e4,
    covariates: Covariates = None,
    alpha: Alpha = 1.0,
    spot_operator: SpotOperator = SpotOperatorTypes.sample,
    gene_center: GeneCenter = False,
    plot: Plot = False,
    verbose: Verbose = False,
):
    """Fit spatial eigenmodes"""
    with timed("Loading data", verbose):
        adata = sc.read_h5ad(fname)
    if verbose:
        print(adata)
    W, coords, gene_names, covariates_matrix = read_anndata(
        adata,
        target_sum,
        transform,
        min_counts,
        min_spot_fraction,
        covariates,
    )

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
        print_top_genes_per_basis(
            adata.varm["tanpopo_eigenvectors"], adata.varm["tanpopo_gene_loadings"], gene_names
        )
    if output:
        adata.write(output)
    if plot:
        plot_spatial_basis(adata, adata.obsm["tanpopo_spot_modes"], cmap="PiYG", vcenter=0)
        plt.show()

    return adata


@app.command()
def cluster(
    fname: InputPath,
    by: ClusterBy,
    output: OutputPath = None,
    neighbours: Neighbours = 15,
    resolution: Resolution = 1.0,
    metric: Metric = "cosine",
    ngenes: NGenes = None,
    plot: Plot = False,
    umap: Umap = False,
    verbose: Verbose = False,
):
    """Cluster spots or genes based on eigenmodes"""
    with timed("Loading data", verbose):
        adata = sc.read_h5ad(fname)

    # filter top n genes
    if ngenes:
        idx = argtop((adata.varm["tanpopo_gene_scores"] ** 2).sum(1), ngenes, mode="pos")
        adata = adata[:, idx].copy()

    # add dictionary for clustering metadata
    if adata.uns["tanpopo"].get("clustering") is None:
        adata.uns["tanpopo"]["clustering"] = {}

    adata.uns["tanpopo"]["clustering"][by] = {
        "n_neighbours": neighbours,
        "resolution": resolution,
        "metric": metric,
    }

    if by == "spots":
        cluster_spots(adata, neighbours, resolution, metric, plot, umap, verbose)
    else:
        cluster_genes(adata, neighbours, resolution, metric, plot, umap, verbose)

    if output:
        adata.write(output)


@app.command()
def plot(fname: InputPath, verbose: Verbose = False):
    """Plot spatial eigenmodes"""
    with timed("Loading data", verbose):
        adata = sc.read_h5ad(fname)
    if verbose:
        print_top_genes_per_basis(
            adata.varm["tanpopo_eigenvectors"],
            adata.varm["tanpopo_gene_loadings"],
            adata.var_names,
        )
    plot_spatial_basis(adata, adata.obsm["tanpopo_spot_modes"], cmap="PiYG", vcenter=0)
    plt.show()


def main():
    app()


if __name__ == "__main__":
    main()
