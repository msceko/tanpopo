import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import typer
from sklearn.neighbors import NearestNeighbors

from tanpopo.io import (
    load_preprocess_sample,
    load_preprocess_samples,
    preprocess_cfg,
    model_cfg,
    add_metadata,
    store_sample_result,
    store_multi_sample_result,
)
from tanpopo.data import get_spatial_from_anndata
from tanpopo.models import (
    SpatialGeneKPCA,
    SpatialGeneContrastKPCA,
    SpatialGeneSampleContrastKPCA,
    SpatialGeneSampleCombinedKPCA,
)
from tanpopo.clustering import cluster_genes, cluster_spots
from tanpopo.plot import plot_spatial_modes
from tanpopo.utils import argtop, pd_dtype, timed
from tanpopo.analysis import print_top_genes_per_basis
from tanpopo.cli import *


def _require_obs_key(adata, key, option):
    if key is None:
        raise typer.BadParameter(f"{option} is required for this workflow.")
    if key not in adata.obs:
        raise typer.BadParameter(f"{option}={key} is not present in adata.obs.")
    return key


def _require_multi_input(fnames, cmd):
    if len(fnames) < 2:
        raise typer.BadParameter(f"{cmd} requires at least two --input files.")


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


def _get_spatial_inputs_for_samples(adata_samples, layer, spot_operator, label_key=None):
    W, coords, covariates_matrix, labels = [], [], [], [] if spot_operator == "label" else None

    for adata in adata_samples:
        w, pts, cov = get_spatial_from_anndata(adata, layer)
        W.append(w)
        coords.append(pts)
        covariates_matrix.append(cov)

        if spot_operator == "label":
            _require_obs_key(adata, label_key, "--label-key")
            labels.append(adata.obs[label_key])

    return W, coords, covariates_matrix, labels


app = typer.Typer(
    name="tanpopo",
    help="Spatial gene eigenmode workflows for spatial transcriptomics.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.command(no_args_is_help=True)
def spatial_programs(
    fname: InputPath,
    output: OutputPath = None,
    cmd_id: ExperimentId = "spatial",
    radius: Radius = 150,
    n_components: Components = 8,
    layer: Layer = None,
    label_key: LabelKey = None,
    subset_labels: Labels = None,
    exclude: Exclude = None,
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
    """Spatial gene programs in a sample, optionally within labels."""
    exclude_labels = [l.strip() for l in exclude.split(",")] if exclude is not None else None
    pre_args = preprocess_cfg(
        target_sum, transform, min_counts, min_spot_fraction, covariates, label_key, layer
    )
    model_args = model_cfg(radius, alpha, gene_center, spot_operator)

    adata = load_preprocess_sample(fname, exclude=exclude_labels, verbose=verbose, **pre_args)
    if subset_labels is not None or spot_operator == "label":
        _require_obs_key(adata, label_key, "--label-key")
    obs_labels = _parse_labels(adata, subset_labels, label_key)
    labels = adata.obs[label_key] if spot_operator == "label" and subset_labels is None else None

    model = SpatialGeneKPCA(verbose=verbose, **model_args)
    add_metadata(adata, cmd_id, pre_args, model_args)

    for label in obs_labels:
        mask = (adata.obs[label_key] == label).to_numpy() if subset_labels else slice(None)
        key = f"_{str(label).replace(' ', '_')}" if subset_labels else ""

        W, coords, covariates_matrix = get_spatial_from_anndata(adata[mask], layer)
        model.fit(W, coords, n_components, labels, covariates_matrix)
        store_sample_result(adata, model, cmd_id, key, mask)

        if verbose:
            if subset_labels:
                print(f"\n{label_key}: {label}")
            print_top_genes_per_basis(model.eigenvectors, model.eigenvalues, adata.var_names)
        if plot:
            size = 120000 / adata.n_obs
            plot_spatial_modes(adata[mask], model.spot_modes[0], size=size)

    if output:
        adata.write(output)
    if plot:
        plt.show()
    return adata


@app.command(no_args_is_help=True)
def shared_programs(
    fnames: InputPaths,
    output: OutputPath = None,
    cmd_id: ExperimentId = "shared",
    radius: Radius = 150,
    sample_names: SampleNames = None,
    n_components: Components = 8,
    layer: Layer = None,
    label_key: LabelKey = None,
    exclude: Exclude = None,
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
    """Shared spatial gene programs across multiple samples."""
    _require_multi_input(fnames, "shared-programs")
    exclude_labels = [l.strip() for l in exclude.split(",")] if exclude is not None else None
    pre_args = preprocess_cfg(
        target_sum, transform, min_counts, min_spot_fraction, covariates, label_key, layer
    )
    model_args = model_cfg(
        radius, alpha, gene_center, spot_operator, sample_weighting, normalise_by
    )

    adata_samples, sample_names = load_preprocess_samples(
        fnames, sample_names, exclude=exclude_labels, verbose=verbose, **pre_args
    )
    if spot_operator == "label":
        _require_obs_key(adata_samples, label_key, "--label-key")
    W, coords, covariates_matrix, labels = _get_spatial_inputs_for_samples(
        adata_samples, layer, spot_operator, label_key
    )

    model = SpatialGeneSampleCombinedKPCA(verbose=verbose, **model_args).fit(
        W, coords, n_components, labels, covariates_matrix
    )
    adata_samples = store_multi_sample_result(adata_samples, sample_names, model, cmd_id, plot)
    extra = {"sample_names": sample_names, "sample_coefficients": model.sample_coefficients_}
    add_metadata(adata_samples, cmd_id, pre_args, model_args, extra)

    if verbose:
        print_top_genes_per_basis(model.eigenvectors, model.eigenvalues, adata_samples.var_names)
    if output:
        adata_samples.write(output)
    if plot:
        plt.show()
    return adata_samples


@app.command(no_args_is_help=True)
def differential_label_programs(
    fname: InputPath,
    label_key: LabelKey,
    output: OutputPath = None,
    cmd_id: ExperimentId = "differential_label",
    radius: Radius = 150,
    n_components: Components = 8,
    layer: Layer = None,
    exclude: Exclude = None,
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
    """Differential gene programs enriched in one label versus the rest."""
    exclude_labels = [l.strip() for l in exclude.split(",")] if exclude is not None else None
    pre_args = preprocess_cfg(
        target_sum, transform, min_counts, min_spot_fraction, covariates, label_key, layer
    )
    model_args = model_cfg(
        radius, alpha, gene_center, spot_operator, sample_weighting, normalise_by
    )

    adata = load_preprocess_sample(fname, exclude=exclude_labels, verbose=verbose, **pre_args)
    _require_obs_key(adata, label_key, "--label-key")
    obs_labels = adata.obs[label_key].unique()
    W, coords, covariates_matrix = get_spatial_from_anndata(adata, layer)

    model = SpatialGeneSampleContrastKPCA(
        positive_samples=[0], negative_samples=[1], verbose=verbose, **model_args
    )
    add_metadata(adata, cmd_id, pre_args, model_args)

    for label in obs_labels:
        mask = (adata.obs[label_key] == label).to_numpy()
        key = f"_{str(label).replace(' ', '_')}"

        W_list = [W[mask], W[~mask]]
        coords_list = [coords[mask], coords[~mask]]
        if spot_operator == "label":
            label_list = [adata.obs[label_key][mask], adata.obs[label_key][~mask]]
        else:
            label_list = None
        if covariates is not None:
            covariates_list = [covariates_matrix[mask], covariates_matrix[~mask]]
        else:
            covariates_list = None

        model.fit(W_list, coords_list, n_components, label_list, covariates_list)
        store_sample_result(adata, model, cmd_id, key, mask, model.gene_loadings[0])

        if verbose:
            print(f"\n{label_key}: {label}")
            print_top_genes_per_basis(model.eigenvectors, model.eigenvalues, adata.var_names)
        if plot:
            size = 120000 / adata.n_obs
            plot_spatial_modes(adata[mask], model.spot_modes[0], size=size)

    if output:
        adata.write(output)
    if plot:
        plt.show()
    return adata


@app.command(no_args_is_help=True)
def differential_sample_programs(
    fnames_a: InputPathsA,
    fnames_b: InputPathsB,
    output: OutputPath = None,
    cmd_id: ExperimentId = "differential_sample",
    radius: Radius = 150,
    sample_names: SampleNames = None,
    n_components: Components = 8,
    layer: Layer = None,
    label_key: LabelKey = None,
    exclude: Exclude = None,
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
    """Differential gene programs enriched in one sample group or condition versus another."""
    fnames = fnames_a + fnames_b
    n_a, n_b = len(fnames_a), len(fnames_b)
    idx_a, idx_b = range(n_a), range(n_a, n_a + n_b)
    exclude_labels = [l.strip() for l in exclude.split(",")] if exclude is not None else None
    pre_args = preprocess_cfg(
        target_sum, transform, min_counts, min_spot_fraction, covariates, label_key, layer
    )
    model_args = model_cfg(
        radius, alpha, gene_center, spot_operator, sample_weighting, normalise_by
    )

    adata_samples, sample_names = load_preprocess_samples(
        fnames, sample_names, exclude=exclude_labels, verbose=verbose, **pre_args
    )
    W, coords, covariates_matrix, labels = _get_spatial_inputs_for_samples(
        adata_samples, layer, spot_operator, label_key
    )

    model = SpatialGeneSampleContrastKPCA(
        positive_samples=idx_a, negative_samples=idx_b, verbose=verbose, **model_args
    ).fit(W, coords, n_components, labels, covariates_matrix)

    adata_samples = store_multi_sample_result(adata_samples, sample_names, model, cmd_id, plot)
    extra = {
        "eigenvalues": model.eigenvalues,
        "sample_names_group_A": [sample_names[i] for i in idx_a],
        "sample_names_group_B": [sample_names[i] for i in idx_b],
        "sample_coefficients": model.sample_coefficients_,
    }
    add_metadata(adata_samples, cmd_id, pre_args, model_args, extra)

    if verbose:
        print_top_genes_per_basis(model.eigenvectors, model.eigenvalues, adata_samples.var_names)
    if output:
        adata_samples.write(output)
    if plot:
        plt.show()
    return adata_samples


@app.command(no_args_is_help=True)
def marker_programs(
    fname: InputPath,
    label_key: LabelKey,
    output: OutputPath = None,
    cmd_id: ExperimentId = "marker",
    radius: Radius = 150,
    n_components: Components = 8,
    layer: Layer = None,
    exclude: Exclude = None,
    transform: Transform = TransformTypes.log1p,
    min_counts: MinCounts = 10,
    min_spot_fraction: MinSpotFraction = None,
    target_sum: TargetSum = 1e4,
    covariates: Covariates = None,
    alpha: Alpha = 1.0,
    gene_center: GeneCenter = False,
    plot: Plot = False,
    verbose: Verbose = False,
):
    """Marker gene programs that distinguish labelled domains or cell types."""
    exclude_labels = [l.strip() for l in exclude.split(",")] if exclude is not None else None
    pre_args = preprocess_cfg(
        target_sum, transform, min_counts, min_spot_fraction, covariates, label_key, layer
    )
    model_args = model_cfg(radius, alpha, gene_center)

    adata = load_preprocess_sample(fname, exclude=exclude_labels, verbose=verbose, **pre_args)
    _require_obs_key(adata, label_key, "--label-key")
    W, coords, covariates_matrix = get_spatial_from_anndata(adata, layer)
    labels = adata.obs[label_key]

    model = SpatialGeneContrastKPCA.between_labels(verbose=verbose, **model_args).fit(
        W, coords, n_components, labels, covariates_matrix
    )
    add_metadata(adata, cmd_id, pre_args, model_args)
    store_sample_result(adata, model, cmd_id)

    if verbose:
        print_top_genes_per_basis(model.eigenvectors, model.eigenvalues, adata.var_names)
    if output:
        adata.write(output)
    if plot:
        plot_spatial_modes(adata, model.spot_modes[0])
        plt.show()
    return adata


@app.command()
def gene_scores(
    fname: InputPath,
    output: OutputPath = None,
    cmd_id: ExperimentId = "spatial",
    radius: Radius = 150,
    n_components: Components = 8,
    layer: Layer = None,
    label_key: LabelKey = None,
    subset_labels: Labels = None,
    exclude: Exclude = None,
    transform: Transform = TransformTypes.log1p,
    min_counts: MinCounts = 10,
    min_spot_fraction: MinSpotFraction = None,
    target_sum: TargetSum = 1e4,
    covariates: Covariates = None,
    alpha: Alpha = 1.0,
    spot_operator: SpotOperator = SpotOperatorTypes.sample,
    gene_center: GeneCenter = False,
    verbose: Verbose = False,
):
    """Score genes by spatial structure."""
    adata = spatial_programs(
        fname,
        output,
        cmd_id,
        radius,
        n_components,
        layer,
        label_key,
        subset_labels,
        exclude,
        transform,
        min_counts,
        min_spot_fraction,
        target_sum,
        covariates,
        alpha,
        spot_operator,
        gene_center,
    )
    obs_labels = _parse_labels(adata, subset_labels, label_key)
    order = np.argsort(adata.var[f"tanpopo_{cmd_id}_gene_scores"])[::-1]

    if verbose:
        for label in obs_labels:
            if subset_labels:
                print(f"\n{label_key}: {label}")
            top = order[: min(20, len(order))]
            for i in top:
                gene_name = adata.var_names[i]
                gene_score = adata.var[f"tanpopo_{cmd_id}_gene_scores"][i]
                print(f"{gene_name:15s} {gene_score:.2f}")

    return adata


@app.command(no_args_is_help=True)
def estimate_spacing(fname: InputPath):
    """Compute average distance to closest neighbour."""
    with timed("Loading data", enabled=True):
        adata = sc.read_h5ad(fname)
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nn.fit(adata.obsm["spatial"])
    distances, _ = nn.kneighbors(adata.obsm["spatial"])
    avg_dist = np.mean(distances[:, 1])
    print(f"Average neighbour distance = {avg_dist:.2f}")


@app.command(no_args_is_help=True)
def h5ad_summary(fname: InputPath):
    """View all anndata annotations."""
    with timed("Loading data", enabled=True):
        adata = sc.read_h5ad(fname)
        print(adata)
        if "tanpopo" in adata.uns:
            experiment_ids = ", ".join(f"'{key}'" for key in adata.uns["tanpopo"].keys())
            print("    uns['tanpopo']: " + experiment_ids)
            # for key, val in adata.uns["tanpopo"].items():
            #     print(f"    uns['tanpopo']['{key}']: ", val)


@app.command(no_args_is_help=True)
def cluster(
    fname: InputPath,
    by: ClusterBy,
    output: OutputPath = None,
    cmd_id: ExperimentId = "spatial",
    neighbours: Neighbours = 15,
    resolution: Resolution = 1.0,
    metric: Metric = "cosine",
    ngenes: NGenes = None,
    plot: Plot = False,
    umap: Umap = False,
    verbose: Verbose = False,
):
    """Cluster spots or genes based on spatial gene programs."""
    with timed("Loading data", verbose):
        adata = sc.read_h5ad(fname)

    # filter top n genes
    if ngenes:
        idx = argtop(adata.var[f"tanpopo_{cmd_id}_gene_scores"], ngenes, mode="pos")
        adata = adata[:, idx].copy()

    # add dictionary for clustering metadata
    adata.uns["tanpopo"].setdefault("clustering", {})

    adata.uns["tanpopo"]["clustering"][by] = {
        "n_neighbours": neighbours,
        "resolution": resolution,
        "metric": metric,
    }

    key_added = f"tanpopo_{cmd_id}_leiden"
    if by == "spots":
        key = f"tanpopo_{cmd_id}_spot_modes"
        cluster_spots(adata, neighbours, resolution, metric, key, key_added, plot, umap, verbose)
    else:
        key = f"tanpopo_{cmd_id}_gene_scores"
        cluster_genes(adata, neighbours, resolution, metric, key, key_added, plot, umap, verbose)
    if output:
        adata.write(output)


@app.command(no_args_is_help=True)
def plot(fname: InputPath, cmd_id: ExperimentId = "spatial", verbose: Verbose = False):
    """Plot spatial gene programs."""
    with timed("Loading data", verbose):
        adata = sc.read_h5ad(fname)
    if verbose:
        print_top_genes_per_basis(
            adata.varm[f"tanpopo_{cmd_id}_eigenvectors"],
            adata.varm[f"tanpopo_{cmd_id}_gene_loadings"],
            adata.var_names,
        )
    plot_spatial_modes(adata, adata.obsm[f"tanpopo_{cmd_id}_spot_modes"])
    plt.show()


def main():
    app()


if __name__ == "__main__":
    main()
