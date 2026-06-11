import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import typer
from sklearn.neighbors import NearestNeighbors

from tanpopo.data import (
    preprocess_anndata,
    preprocess_anndata_shared_genes,
    get_spatial_from_anndata,
)
from tanpopo.models import (
    SpatialGeneKPCA,
    SpatialGeneContrastKPCA,
    SpatialGeneSampleContrastKPCA,
    SpatialGeneSampleCombinedKPCA,
)
from tanpopo.clustering import cluster_genes, cluster_spots
from tanpopo.plot import plot_spatial_modes
from tanpopo.utils import as_value, timed, argtop, pd_dtype
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


def _full_mode(spot_mode, mask, total_spots):
    full_mode = np.full((total_spots, spot_mode.shape[1]), np.nan)
    full_mode[mask] = spot_mode
    return full_mode


app = typer.Typer(
    name="tanpopo",
    help="Spatial gene eigenmode workflows for spatial transcriptomics.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.command()
def spatial_programs(
    fname: InputPath,
    output: OutputPath = None,
    cmd_id: ExperimentId = "spatial",
    radius: Radius = 150,
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
    """Spatial gene programs in a sample, optionally within labels."""
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
        cmd_id: {
            "preprocessing": {
                "target_sum": target_sum,
                "transform": as_value(transform),
                "min_counts": min_counts,
                "min_spot_fraction": min_spot_fraction,
            },
            "cfg": {
                "kernel": "wendland_c2",
                "radius": radius,
                "alpha": alpha,
                "spot_operator": as_value(spot_operator),
                "gene_center": gene_center,
                "covariates": covariates,
                "label_key": label_key,
            },
        }
    }
    for label in obs_labels:
        mask = (adata.obs[label_key] == label).to_numpy() if subset_labels else slice(None)
        key = f"_{str(label).replace(' ', '_')}" if subset_labels else ""

        W, coords, covariates_matrix = get_spatial_from_anndata(adata[mask], layer)
        model.fit(W, coords, n_components, labels, covariates_matrix)

        adata.obsm[f"tanpopo_{cmd_id}{key}_spot_modes"] = _full_mode(
            model.spot_modes[0], mask, adata.n_obs
        )
        adata.varm[f"tanpopo_{cmd_id}{key}_eigenvectors"] = model.eigenvectors
        adata.varm[f"tanpopo_{cmd_id}{key}_gene_loadings"] = model.gene_loadings
        adata.varm[f"tanpopo_{cmd_id}{key}_gene_scores"] = model.gene_scores
        adata.var[f"tanpopo_{cmd_id}{key}_gene_scores"] = model.gene_spatial_scores()
        adata.uns["tanpopo"][cmd_id][f"eigenvalues{key}"] = model.eigenvalues

        if verbose:
            if subset_labels:
                print(f"\n{label_key}: {label}")
            print_top_genes_per_basis(model.eigenvectors, model.eigenvalues, adata.var_names)
        if plot:
            size = 120000 / adata.n_obs
            plot_spatial_modes(
                adata[mask], model.spot_modes[0], cmap="coolwarm", vcenter=0, size=size
            )

    if output:
        adata.write(output)
    if plot:
        plt.show()
    return adata


@app.command()
def shared_programs(
    fnames: InputPaths,
    output: OutputPath = None,
    cmd_id: ExperimentId = "shared",
    radius: Radius = 150,
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
    """Shared spatial gene programs across multiple samples."""
    _require_multi_input(fnames, "shared-programs")
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
    ).fit(W, coords, n_components, labels, covariates_matrix)

    for adata, name, spot_mode, gene_loadings in zip(
        adata_samples, sample_names, model.spot_modes, model.gene_loadings
    ):
        adata.obsm[f"tanpopo_{cmd_id}_spot_modes"] = spot_mode
        adata.varm[f"tanpopo_{cmd_id}_{name}_gene_loadings"] = gene_loadings
        if plot:
            plot_spatial_modes(adata, spot_mode, cmap="coolwarm", vcenter=0)

    adata_samples = _concat_adata_samples(adata_samples, sample_names)
    adata_samples.varm[f"tanpopo_{cmd_id}_eigenvectors"] = model.eigenvectors
    adata_samples.varm[f"tanpopo_{cmd_id}_gene_scores"] = model.gene_scores
    adata_samples.var[f"tanpopo_{cmd_id}_gene_scores"] = model.gene_spatial_scores()
    adata_samples.uns["tanpopo"] = {
        cmd_id: {
            "eigenvalues": model.eigenvalues,
            "sample_names": sample_names,
            "sample_coefficients": model.sample_coefficients_,
            "preprocessing": {
                "target_sum": target_sum,
                "transform": as_value(transform),
                "min_counts": min_counts,
                "min_spot_fraction": min_spot_fraction,
            },
            "cfg": {
                "kernel": "wendland_c2",
                "radius": radius,
                "alpha": alpha,
                "spot_operator": as_value(spot_operator),
                "sample_weighting": as_value(sample_weighting),
                "normalise_by": as_value(normalise_by),
                "gene_center": gene_center,
                "covariates": covariates,
                "label_key": label_key,
            },
        }
    }

    if verbose:
        print_top_genes_per_basis(model.eigenvectors, model.eigenvalues, adata_samples.var_names)
    if output:
        adata_samples.write(output)
    if plot:
        plt.show()
    return adata_samples


@app.command()
def differential_label_programs(
    fname: InputPath,
    label_key: LabelKey,
    output: OutputPath = None,
    cmd_id: ExperimentId = "differential_label",
    radius: Radius = 150,
    n_components: Components = 8,
    layer: Layer = None,
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
    with timed("Loading data", verbose):
        adata = sc.read_h5ad(fname)
        preprocess_anndata(
            adata, target_sum, transform, min_counts, min_spot_fraction, covariates, layer
        )
    if verbose:
        print(adata)

    _require_obs_key(adata, label_key, "--label-key")
    obs_labels = adata.obs[label_key].unique()
    W, coords, covariates_matrix = get_spatial_from_anndata(adata, layer)

    model = SpatialGeneSampleContrastKPCA(
        radius=radius,
        positive_samples=[0],
        negative_samples=[1],
        spot_operator=spot_operator,
        sample_weighting=sample_weighting,
        normalise_by=normalise_by,
        alpha=alpha,
        gene_center=gene_center,
        verbose=verbose,
    )
    adata.uns["tanpopo"] = {
        cmd_id: {
            "preprocessing": {
                "target_sum": target_sum,
                "transform": as_value(transform),
                "min_counts": min_counts,
                "min_spot_fraction": min_spot_fraction,
            },
            "cfg": {
                "kernel": "wendland_c2",
                "radius": radius,
                "alpha": alpha,
                "spot_operator": as_value(spot_operator),
                "sample_weighting": as_value(sample_weighting),
                "normalise_by": as_value(normalise_by),
                "gene_center": gene_center,
                "covariates": covariates,
                "label_key": label_key,
            },
        }
    }

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

        adata.obsm[f"tanpopo_{cmd_id}{key}_spot_modes"] = _full_mode(
            model.spot_modes[0], mask, adata.n_obs
        )
        adata.varm[f"tanpopo_{cmd_id}{key}_eigenvectors"] = model.eigenvectors
        adata.varm[f"tanpopo_{cmd_id}{key}_gene_loadings"] = model.gene_loadings[0]
        adata.varm[f"tanpopo_{cmd_id}{key}_gene_scores"] = model.gene_scores
        adata.var[f"tanpopo_{cmd_id}{key}_gene_scores"] = model.gene_spatial_scores()
        adata.uns["tanpopo"][cmd_id][f"eigenvalues{key}"] = model.eigenvalues

        if verbose:
            print(f"\n{label_key}: {label}")
            print_top_genes_per_basis(model.eigenvectors, model.eigenvalues, adata.var_names)
        if plot:
            size = 120000 / adata.n_obs
            plot_spatial_modes(
                adata[mask], model.spot_modes[0], cmap="coolwarm", vcenter=0, size=size
            )

    if output:
        adata.write(output)
    if plot:
        plt.show()
    return adata


@app.command()
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

    model = SpatialGeneSampleContrastKPCA(
        radius=radius,
        positive_samples=idx_a,
        negative_samples=idx_b,
        spot_operator=spot_operator,
        sample_weighting=sample_weighting,
        normalise_by=normalise_by,
        alpha=alpha,
        gene_center=gene_center,
        verbose=verbose,
    ).fit(W, coords, n_components, labels, covariates_matrix)

    for adata, name, spot_mode, gene_loadings in zip(
        adata_samples, sample_names, model.spot_modes, model.gene_loadings
    ):
        adata.obsm[f"tanpopo_{cmd_id}_spot_modes"] = spot_mode
        adata.varm[f"tanpopo_{cmd_id}_{name}_gene_loadings"] = gene_loadings
        if plot:
            plot_spatial_modes(adata, spot_mode, cmap="coolwarm", vcenter=0)

    adata_samples = _concat_adata_samples(adata_samples, sample_names)
    adata_samples.varm[f"tanpopo_{cmd_id}_eigenvectors"] = model.eigenvectors
    adata_samples.varm[f"tanpopo_{cmd_id}_gene_scores"] = model.gene_scores
    adata_samples.var[f"tanpopo_{cmd_id}_gene_scores"] = model.gene_spatial_scores()
    adata_samples.uns[f"tanpopo"] = {
        cmd_id: {
            "eigenvalues": model.eigenvalues,
            "sample_names_group_A": [sample_names[i] for i in idx_a],
            "sample_names_group_B": [sample_names[i] for i in idx_b],
            "sample_coefficients": model.sample_coefficients_,
            "preprocessing": {
                "target_sum": target_sum,
                "transform": as_value(transform),
                "min_counts": min_counts,
                "min_spot_fraction": min_spot_fraction,
            },
            "cfg": {
                "kernel": "wendland_c2",
                "radius": radius,
                "alpha": alpha,
                "spot_operator": as_value(spot_operator),
                "sample_weighting": as_value(sample_weighting),
                "normalise_by": as_value(normalise_by),
                "gene_center": gene_center,
                "covariates": covariates,
                "label_key": label_key,
            },
        }
    }

    if verbose:
        print_top_genes_per_basis(model.eigenvectors, model.eigenvalues, adata_samples.var_names)
    if output:
        adata_samples.write(output)
    if plot:
        plt.show()
    return adata_samples


@app.command()
def marker_programs(
    fname: InputPath,
    label_key: LabelKey,
    output: OutputPath = None,
    cmd_id: ExperimentId = "marker",
    radius: Radius = 150,
    n_components: Components = 8,
    layer: Layer = None,
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
    with timed("Loading data", verbose):
        adata = sc.read_h5ad(fname)
        preprocess_anndata(
            adata, target_sum, transform, min_counts, min_spot_fraction, covariates, layer
        )
    if verbose:
        print(adata)

    _require_obs_key(adata, label_key, "--label-key")
    W, coords, covariates_matrix = get_spatial_from_anndata(adata, layer)
    labels = adata.obs[label_key]

    model = SpatialGeneContrastKPCA.between_labels(
        radius=radius,
        alpha=alpha,
        gene_center=gene_center,
        verbose=verbose,
    ).fit(W, coords, n_components, labels, covariates_matrix)

    adata.obsm[f"tanpopo_{cmd_id}_spot_modes"] = model.spot_modes[0]
    adata.varm[f"tanpopo_{cmd_id}_gene_loadings"] = model.gene_loadings
    adata.varm[f"tanpopo_{cmd_id}_eigenvectors"] = model.eigenvectors
    adata.varm[f"tanpopo_{cmd_id}_gene_scores"] = model.gene_scores
    adata.var[f"tanpopo_{cmd_id}_gene_scores"] = model.gene_spatial_scores()
    adata.uns["tanpopo"] = {
        cmd_id: {
            "eigenvalues": model.eigenvalues,
            "preprocessing": {
                "target_sum": target_sum,
                "transform": as_value(transform),
                "min_counts": min_counts,
                "min_spot_fraction": min_spot_fraction,
            },
            "cfg": {
                "kernel": "wendland_c2",
                "radius": radius,
                "alpha": alpha,
                "gene_center": gene_center,
                "covariates": covariates,
                "label_key": label_key,
            },
        }
    }

    if verbose:
        print_top_genes_per_basis(model.eigenvectors, model.eigenvalues, adata.var_names)
    if output:
        adata.write(output)
    if plot:
        plot_spatial_modes(adata, model.spot_modes[0], cmap="coolwarm", vcenter=0)
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


@app.command()
def estimate_spacing(fname: InputPath):
    """Compute average distance to closest neighbour."""
    with timed("Loading data", enabled=True):
        adata = sc.read_h5ad(fname)
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nn.fit(adata.obsm["spatial"])
    distances, _ = nn.kneighbors(adata.obsm["spatial"])
    avg_dist = np.mean(distances[:, 1])
    print(f"Average neighbour distance = {avg_dist:.2f}")


@app.command()
def h5ad_summary(fname: InputPath):
    """View all anndata annotations."""
    with timed("Loading data", enabled=True):
        adata = sc.read_h5ad(fname)
        print(adata)
        if "tanpopo" in adata.uns:
            experiment_ids = ", ".join(f"'{key}'" for key in adata.uns["tanpopo"].keys())
            print("    uns['tanpopo']: " + experiment_ids)


@app.command()
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
    if adata.uns["tanpopo"].get("clustering") is None:
        adata.uns["tanpopo"]["clustering"] = {}

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


@app.command()
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
    plot_spatial_modes(
        adata, adata.obsm[f"tanpopo_{cmd_id}_spot_modes"], cmap="coolwarm", vcenter=0
    )
    plt.show()


def main():
    app()


if __name__ == "__main__":
    main()
