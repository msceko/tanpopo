import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import typer

from tanpopo.data import (
    preprocess_anndata,
    preprocess_anndata_shared_genes,
    get_spatial_from_anndata,
)
from tanpopo.models import SpatialGeneKPCA, SpatialGeneSampleCombinedKPCA
from tanpopo.clustering import cluster_genes, cluster_spots
from tanpopo.plot import plot_spatial_modes
from tanpopo.utils import timed, argtop, print_top_genes_per_basis
from tanpopo.cli import *


def pd_dtype(series: pd.Series):
    if isinstance(series.dtype, pd.CategoricalDtype):
        return series.cat.categories.dtype
    return series.dtype


def _require_obs_key(adata, key, option):
    if key is None:
        raise typer.BadParameter(f"{option} is required for this workflow.")
    if key not in adata.obs:
        raise typer.BadParameter(f"{option}={key} is not present in adata.obs.")
    return key


def _parse_labels(adata, labels, key):
    if labels is None:
        return [""]

    all_labels = list(adata.obs[key].unique())
    if labels == "all":
        return all_labels

    dtype = pd_dtype(adata.obs[key])
    obs_labels = np.array([label.strip() for label in labels.split(",")], dtype)
    not_in_obs = set(obs_labels) - set(all_labels)
    if not_in_obs:
        not_in_obs = ",".join(not_in_obs)
        raise typer.BadParameter(f"--labels={not_in_obs} not in adata.obs['{key}'].")
    return obs_labels


def _name_samples(fnames, sample_names):
    if sample_names is None:
        sample_names = [fname.stem for fname in fnames]
    if len(sample_names) != len(fnames):
        raise typer.BadParameter(f"Supply exactly one --sample-name for each --input.")
    if len(sample_names) != len(set(sample_names)):
        raise typer.BadParameter("Sample names must be unique.")
    return sample_names


def _concat_adata_samples(adatas, sample_names):
    if len(adatas) == 1:
        return adatas[0].copy()
    return sc.concat(
        adatas,
        label="sample",
        keys=[str(x) for x in sample_names],
        index_unique="-",
        join="inner",
        merge="unique",
    )


app = typer.Typer(
    name="tanpopo",
    help="Spatial gene eigenmode workflows for spatial transcriptomics.",
    no_args_is_help=True,
)


@app.command()
def programs(
    fname: InputPath,
    radius: Radius,
    output: OutputPath = None,
    n_components: Components = 8,
    layer: Layer = None,
    label_key: LabelKey = None,
    subset_labels: Labels = None,
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
        preprocess_anndata(
            adata, target_sum, transform, min_counts, min_spot_fraction, covariates, layer
        )
    if verbose:
        print(adata)

    if subset_labels is not None or spot_operator == "label":
        _require_obs_key(adata, label_key, "--label-key")
    obs_labels = _parse_labels(adata, subset_labels, label_key)
    labels = adata.obs[label_key] if spot_operator == "label" and subset_labels is None else None

    model = SpatialGeneKPCA(radius, spot_operator, alpha, gene_center, verbose=verbose)
    adata.uns["tanpopo"] = {
        "preprocessing": {
            "target_sum": target_sum,
            "transform": transform,
            "min_counts": min_counts,
            "min_spot_fraction": min_spot_fraction,
        },
        "cfg": {
            "kernel": "wendland_c2",
            "radius": radius,
            "alpha": alpha,
            "spot_operator": spot_operator,
            "gene_center": gene_center,
            "covariates": covariates,
            "label_key": label_key,
        },
    }
    for label in obs_labels:
        mask = (adata.obs[label_key] == label).to_numpy() if subset_labels else slice(None)
        key = f"_{str(label).replace(' ', '_')}" if subset_labels else ""

        W, coords, covariates_matrix = get_spatial_from_anndata(adata[mask], layer)
        model.fit(W, coords, n_components, labels, covariates_matrix)

        full_mode = np.full((adata.n_obs, model.spot_modes[0].shape[1]), np.nan)
        full_mode[mask] = model.spot_modes[0]
        adata.obsm[f"tanpopo{key}_spot_modes"] = full_mode
        adata.varm[f"tanpopo{key}_eigenvectors"] = model.eigenvectors
        adata.varm[f"tanpopo{key}_gene_loadings"] = model.gene_loadings
        adata.varm[f"tanpopo{key}_gene_scores"] = model.gene_scores
        adata.uns["tanpopo"][f"eigenvalues{key}"] = model.eigenvalues

        if verbose:
            if subset_labels:
                print(f"\n{label_key}: {label}")
            print_top_genes_per_basis(model.eigenvectors, model.eigenvalues, adata.var_names)
        if plot:
            size = 120000 / adata.n_obs
            plot_spatial_modes(adata[mask], full_mode[mask], cmap="coolwarm", vcenter=0, size=size)

    if output:
        adata.write(output)
    if plot:
        plt.show()

    return adata


@app.command()
def shared_programs(
    fnames: InputPaths,
    radius: Radius,
    output: OutputPath = None,
    sample_names: SampleNames = None,
    n_components: Components = 8,
    layer: Layer = None,
    label_key: LabelKey = None,
    transform: Transform = TransformTypes.log1p,
    min_counts: MinCounts = 10,
    min_spot_fraction: MinSpotFraction = None,
    target_sum: TargetSum = 1e4,
    covariates: Covariates = None,
    alpha: Alpha = 1.0,
    spot_operator: SpotOperator = SpotOperatorTypes.sample,
    sample_weighting: SampleWeighting = SampleWeightingTypes.trace,
    normalise_by: SampleNormaliseBy = SampleNormaliseTypes.sample,
    gene_center: GeneCenter = False,
    plot: Plot = False,
    verbose: Verbose = False,
):
    """Fit shared spatial eigenmodes across multiple samples"""
    with timed("Loading data", verbose):
        adata_samples = [sc.read_h5ad(fname) for fname in fnames]
        preprocess_anndata_shared_genes(
            adata_samples, target_sum, transform, min_counts, min_spot_fraction, covariates, layer
        )
        sample_names = _name_samples(fnames, sample_names)
    if verbose:
        for adata, name in zip(adata_samples, sample_names):
            print(name)
            print(adata)

    W, coords, covariates_matrix, labels = [], [], [], [] if spot_operator == "label" else None
    for adata in adata_samples:
        w, pts, cov = get_spatial_from_anndata(adata, layer)
        W.append(w)
        coords.append(pts)
        covariates_matrix.append(cov)

        if spot_operator == "label":
            _require_obs_key(adata, label_key, "--label-key")
            labels.append(adata.obs[label_key])

    model = SpatialGeneSampleCombinedKPCA(
        radius, spot_operator, sample_weighting, normalise_by, alpha, gene_center, verbose=verbose
    ).fit(W, coords, n_components, labels, covariates)

    for adata, name, spot_mode, gene_loadings in zip(
        adata_samples, sample_names, model.spot_modes, model.gene_loadings
    ):
        adata.obsm["tanpopo_spot_modes"] = spot_mode
        adata.varm[f"tanpopo_{name}_gene_loadings"] = gene_loadings
        if plot:
            plot_spatial_modes(adata, spot_mode, cmap="coolwarm", vcenter=0)

    adata_samples = _concat_adata_samples(adata_samples, sample_names)
    adata_samples.varm["tanpopo_eigenvectors"] = model.eigenvectors
    adata_samples.varm["tanpopo_gene_scores"] = model.gene_scores
    adata_samples.uns["tanpopo"] = {
        "eigenvalues": model.eigenvalues,
        "sample_names": sample_names,
        "sample_coefficients": model.sample_coefficients_,
        "preprocessing": {
            "target_sum": target_sum,
            "transform": transform,
            "min_counts": min_counts,
            "min_spot_fraction": min_spot_fraction,
        },
        "cfg": {
            "kernel": "wendland_c2",
            "radius": radius,
            "alpha": alpha,
            "spot_operator": spot_operator,
            "sample_weighting": sample_weighting,
            "normalise_by": normalise_by,
            "gene_center": gene_center,
            "covariates": covariates,
            "label_key": label_key,
        },
    }

    if verbose:
        print_top_genes_per_basis(model.eigenvectors, model.eigenvalues, adata_samples.var_names)
    if output:
        adata_samples.write(output)
    if plot:
        plt.show()


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
    plot_spatial_modes(adata, adata.obsm["tanpopo_spot_modes"], cmap="coolwarm", vcenter=0)
    plt.show()


def main():
    app()


if __name__ == "__main__":
    main()
