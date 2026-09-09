from __future__ import annotations

from pathlib import Path

import numpy as np
import typer

from tanpopo.cli import *
from tanpopo.data import get_spatial_from_anndata
from tanpopo.io import (
    add_metadata,
    concat_adata_samples,
    load_preprocess_sample,
    load_preprocess_samples,
    model_cfg,
    preprocess_cfg,
    store_cross_result,
    store_sample_result,
)
from tanpopo.kernel import neighbour_spacing
from tanpopo.models import (
    CrossSpatialProgramModel,
    DifferentialSpatialProgramModel,
    LabelDecompositionModel,
    SharedSpatialProgramModel,
    SpatialProgramModel,
)
from tanpopo.utils import as_value, pd_dtype


app = typer.Typer(
    name="tanpopo",
    help=(
        "Conditional spatial covariance programs for spatial transcriptomics. "
        "Spatial graphs are zero-diagonal by default."
    ),
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _require_obs_key(adata, key, option="--label-key"):
    if key is None or key not in adata.obs:
        raise typer.BadParameter(f"{option} must name a column in adata.obs")
    return key


def _parse_csv_labels(adata, text, label_key, allow_all=True):
    if text is None:
        return None
    _require_obs_key(adata, label_key)
    values = adata.obs[label_key]
    unique = [x for x in values.dropna().unique()]
    if allow_all and text.strip().lower() == "all":
        return unique
    requested = [x.strip() for x in text.split(",") if x.strip()]
    lookup = {}
    for value in unique:
        key = str(value)
        if key in lookup and lookup[key] != value:
            raise ValueError(
                f"Ambiguous label representation {key!r} in " f"adata.obs[{label_key!r}]"
            )
        lookup[key] = value
    missing = [label for label in requested if label not in lookup]
    if missing:
        available = [str(x) for x in unique]
        raise typer.BadParameter(
            f"Label(s) not present in adata.obs[{label_key!r}]: {missing}. "
            f"Available labels include: {available[:20]}"
        )
    return [lookup[label] for label in requested]


def _single_multisample_label(adatas, text, label_key):
    if text is None:
        return None
    labels = _parse_csv_labels(adatas[0], text, label_key, allow_all=False)
    if len(labels) != 1:
        raise typer.BadParameter(
            "shared-programs and differential-sample-programs currently accept one --labels value"
        )
    label = labels[0]
    for adata in adatas[1:]:
        _require_obs_key(adata, label_key)
        if label not in set(adata.obs[label_key].unique()):
            raise typer.BadParameter(f"Label {label!r} is not present in every sample")
    return label


def _resolve_objective(objective, alpha):
    objective = as_value(objective)
    if alpha is None:
        return objective
    if float(alpha) == 0:
        typer.echo(
            "Warning: --alpha is deprecated; alpha=0 maps to --objective covariance.", err=True
        )
        return "covariance"
    raise typer.BadParameter(
        "The historical nonzero --alpha normalization has been removed. "
        "Use --objective gene_standardized or --objective gain."
    )


def _pre_args(
    target_sum,
    transform,
    min_counts,
    min_spot_fraction,
    covariates,
    label_key,
    layer,
    include,
    exclude,
):
    return preprocess_cfg(
        target_sum,
        transform,
        min_counts,
        min_spot_fraction,
        covariates,
        label_key,
        layer,
        include,
        exclude,
    )


def _radius_from_mask(adata, mask=None):
    coords = np.asarray(adata.obsm["spatial"])
    if mask is not None:
        coords = coords[np.asarray(mask, dtype=bool)]
    if len(coords) < 2:
        raise typer.BadParameter("At least two selected cells are required to estimate radius")
    spacing = neighbour_spacing(coords)
    radius = 3.5 * spacing
    typer.echo(
        f"No --radius specified; using {radius:.3g} (3.5 x mean nearest-neighbour distance)"
    )
    return radius


def _model_kwargs(
    objective, spot_operator, expression_rank, gain_ridge, graph_normalisation, dtype, verbose
):
    return {
        "objective": _resolve_objective(objective, None),
        "spot_operator": as_value(spot_operator),
        "expression_rank": expression_rank,
        "gain_ridge": gain_ridge,
        "graph_normalisation": as_value(graph_normalisation),
        "dtype": as_value(dtype),
        "verbose": verbose,
    }


@app.command("spatial-programs", no_args_is_help=True)
def spatial_programs(
    fname: InputPath,
    output: OutputPath = None,
    cmd_id: ExperimentId = "spatial",
    radius: Radius = None,
    n_components: Components = 8,
    layer: Layer = None,
    label_key: LabelKey = None,
    subset_labels: Labels = None,
    objective: Objective = ObjectiveTypes.covariance,
    expression_rank: ExpressionRank = 50,
    gain_ridge: GainRidge = 1e-8,
    graph_normalisation: GraphNormalisation = GraphNormalisationTypes.symmetric,
    spot_operator: SpotOperator = SpotOperatorTypes.sample,
    alpha: Alpha = None,
    include: Include = None,
    exclude: Exclude = None,
    transform: Transform = None,
    min_counts: MinCounts = 10,
    min_spot_fraction: MinSpotFraction = None,
    target_sum: TargetSum = None,
    covariates: Covariates = None,
    dtype: Dtype = Dtypes.float64,
    verbose: Verbose = False,
):
    """Spatial programs globally or within selected cell types."""
    objective_value = _resolve_objective(objective, alpha)
    pre = _pre_args(
        target_sum,
        transform,
        min_counts,
        min_spot_fraction,
        covariates,
        label_key,
        layer,
        include,
        exclude,
    )
    adata = load_preprocess_sample(fname, verbose=verbose, **pre)
    targets = _parse_csv_labels(adata, subset_labels, label_key) if subset_labels else [None]
    W, coords, cov = get_spatial_from_anndata(adata, layer)

    for target in targets:
        mask = None if target is None else (adata.obs[label_key] == target).to_numpy()
        local_radius = _radius_from_mask(adata, mask) if radius is None else radius
        labels = None
        if as_value(spot_operator) == "label":
            _require_obs_key(adata, label_key)
            labels = np.asarray(adata.obs[label_key])
        model = SpatialProgramModel(
            local_radius,
            objective=objective_value,
            spot_operator=as_value(spot_operator),
            expression_rank=expression_rank,
            gain_ridge=gain_ridge,
            graph_normalisation=as_value(graph_normalisation),
            dtype=as_value(dtype),
            verbose=verbose,
        ).fit(W, coords, n_components, labels=labels, covariates=cov, masks=mask)
        key = "" if target is None else f"_{str(target).replace(' ', '_')}"
        store_sample_result(adata, model, cmd_id, key)
        add_metadata(
            adata,
            cmd_id,
            pre,
            model_cfg(model),
            {"conditioned_label": None if target is None else str(target)},
        )
    if output is not None:
        adata.write_h5ad(output)
    return adata


@app.command("shared-programs", no_args_is_help=True)
def shared_programs(
    fnames: InputPaths,
    output: OutputPath = None,
    cmd_id: ExperimentId = "shared",
    radius: Radius = None,
    sample_names: SampleNames = None,
    n_components: Components = 8,
    layer: Layer = None,
    label_key: LabelKey = None,
    subset_labels: Labels = None,
    objective: Objective = ObjectiveTypes.covariance,
    expression_rank: ExpressionRank = 50,
    gain_ridge: GainRidge = 1e-8,
    graph_normalisation: GraphNormalisation = GraphNormalisationTypes.symmetric,
    spot_operator: SpotOperator = SpotOperatorTypes.sample,
    sample_weighting: SampleWeighting = SampleWeightingTypes.n_spots,
    alpha: Alpha = None,
    include: Include = None,
    exclude: Exclude = None,
    transform: Transform = None,
    min_counts: MinCounts = 10,
    min_spot_fraction: MinSpotFraction = None,
    target_sum: TargetSum = None,
    covariates: Covariates = None,
    dtype: Dtype = Dtypes.float64,
    verbose: Verbose = False,
):
    """Shared spatial programs across biological samples, optionally within one cell type."""
    if len(fnames) < 2:
        raise typer.BadParameter("shared-programs requires at least two inputs")
    objective_value = _resolve_objective(objective, alpha)
    pre = _pre_args(
        target_sum,
        transform,
        min_counts,
        min_spot_fraction,
        covariates,
        label_key,
        layer,
        include,
        exclude,
    )
    adatas, sample_names = load_preprocess_samples(fnames, sample_names, verbose=verbose, **pre)
    target = _single_multisample_label(adatas, subset_labels, label_key)
    W, coords, covs, labels, masks = [], [], [], [], []
    for adata in adatas:
        w, xy, cov = get_spatial_from_anndata(adata, layer)
        W.append(w)
        coords.append(xy)
        covs.append(cov)
        masks.append(None if target is None else (adata.obs[label_key] == target).to_numpy())
        if as_value(spot_operator) == "label":
            _require_obs_key(adata, label_key)
            labels.append(np.asarray(adata.obs[label_key]))
    if as_value(spot_operator) != "label":
        labels = None
    if radius is None:
        radius = _radius_from_mask(adatas[0], masks[0])
    model = SharedSpatialProgramModel(
        radius,
        objective=objective_value,
        spot_operator=as_value(spot_operator),
        sample_weighting=as_value(sample_weighting),
        expression_rank=expression_rank,
        gain_ridge=gain_ridge,
        graph_normalisation=as_value(graph_normalisation),
        dtype=as_value(dtype),
        verbose=verbose,
    ).fit(W, coords, n_components, labels=labels, covariates=covs, masks=masks)

    key = "" if target is None else f"_{str(target).replace(' ', '_')}"
    prefix = f"tanpopo_{cmd_id}{key}"
    for i, adata in enumerate(adatas):
        full = np.full((adata.n_obs, model.spot_modes[i].shape[1]), np.nan)
        full[model.samples[i].obs_idx] = model.spot_modes[i]
        adata.obsm[f"{prefix}_spot_modes"] = full
    combined = concat_adata_samples(adatas, sample_names)
    combined.varm[f"{prefix}_gene_loadings"] = model.gene_loadings
    combined.varm[f"{prefix}_gene_scores"] = model.gene_scores
    combined.var[f"{prefix}_gene_spatial_covariance"] = model.gene_spatial_scores()
    add_metadata(
        combined,
        cmd_id,
        pre,
        model_cfg(model),
        {
            "sample_names": sample_names,
            "sample_coefficients": model.sample_coefficients_,
            "conditioned_label": None if target is None else str(target),
        },
    )
    combined.uns["tanpopo"][cmd_id][f"eigenvalues{key}"] = model.eigenvalues
    if output is not None:
        combined.write_h5ad(output)
    return combined


@app.command("differential-sample-programs", no_args_is_help=True)
def differential_sample_programs(
    fnames_a: InputPathsA,
    fnames_b: InputPathsB,
    output: OutputPath = None,
    cmd_id: ExperimentId = "differential_sample",
    radius: Radius = None,
    sample_names: SampleNames = None,
    n_components: Components = 8,
    layer: Layer = None,
    label_key: LabelKey = None,
    subset_labels: Labels = None,
    objective: Objective = ObjectiveTypes.covariance,
    expression_rank: ExpressionRank = 50,
    gain_ridge: GainRidge = 1e-8,
    graph_normalisation: GraphNormalisation = GraphNormalisationTypes.symmetric,
    spot_operator: SpotOperator = SpotOperatorTypes.sample,
    sample_weighting: SampleWeighting = SampleWeightingTypes.n_spots,
    alpha: Alpha = None,
    include: Include = None,
    exclude: Exclude = None,
    transform: Transform = None,
    min_counts: MinCounts = 10,
    min_spot_fraction: MinSpotFraction = None,
    target_sum: TargetSum = None,
    covariates: Covariates = None,
    permutations: Permutations = 0,
    seed: RandomSeed = 0,
    dtype: Dtype = Dtypes.float64,
    verbose: Verbose = False,
):
    """Spatial programs whose covariance differs between biological sample groups."""
    if not fnames_a or not fnames_b:
        raise typer.BadParameter("At least one sample is required in each group")
    fnames = list(fnames_a) + list(fnames_b)
    objective_value = _resolve_objective(objective, alpha)
    pre = _pre_args(
        target_sum,
        transform,
        min_counts,
        min_spot_fraction,
        covariates,
        label_key,
        layer,
        include,
        exclude,
    )
    adatas, sample_names = load_preprocess_samples(fnames, sample_names, verbose=verbose, **pre)
    target = _single_multisample_label(adatas, subset_labels, label_key)
    W, coords, covs, labels, masks = [], [], [], [], []
    for adata in adatas:
        w, xy, cov = get_spatial_from_anndata(adata, layer)
        W.append(w)
        coords.append(xy)
        covs.append(cov)
        masks.append(None if target is None else (adata.obs[label_key] == target).to_numpy())
        if as_value(spot_operator) == "label":
            labels.append(np.asarray(adata.obs[label_key]))
    if as_value(spot_operator) != "label":
        labels = None
    if radius is None:
        radius = _radius_from_mask(adatas[0], masks[0])
    n_a = len(fnames_a)
    model = DifferentialSpatialProgramModel(
        radius,
        positive_samples=range(n_a),
        negative_samples=range(n_a, len(fnames)),
        objective=objective_value,
        spot_operator=as_value(spot_operator),
        sample_weighting=as_value(sample_weighting),
        expression_rank=expression_rank,
        gain_ridge=gain_ridge,
        graph_normalisation=as_value(graph_normalisation),
        dtype=as_value(dtype),
        verbose=verbose,
    ).fit(W, coords, n_components, labels=labels, covariates=covs, masks=masks)
    if permutations:
        model.permutation_test(permutations, seed=seed)
    key = "" if target is None else f"_{str(target).replace(' ', '_')}"
    prefix = f"tanpopo_{cmd_id}{key}"
    for i, adata in enumerate(adatas):
        full = np.full((adata.n_obs, model.spot_modes[i].shape[1]), np.nan)
        full[model.samples[i].obs_idx] = model.spot_modes[i]
        adata.obsm[f"{prefix}_spot_modes"] = full
    combined = concat_adata_samples(adatas, sample_names)
    combined.varm[f"{prefix}_gene_loadings"] = model.gene_loadings
    combined.varm[f"{prefix}_gene_scores"] = model.gene_scores
    combined.var[f"{prefix}_gene_spatial_covariance"] = model.gene_spatial_scores()
    add_metadata(
        combined,
        cmd_id,
        pre,
        model_cfg(model),
        {
            "sample_names": sample_names,
            "group_a": sample_names[:n_a],
            "group_b": sample_names[n_a:],
            "sample_coefficients": model.sample_coefficients_,
            "conditioned_label": None if target is None else str(target),
            "permutations": int(permutations),
        },
    )
    combined.uns["tanpopo"][cmd_id][f"eigenvalues{key}"] = model.eigenvalues
    if permutations:
        combined.uns["tanpopo"][cmd_id][f"permutation_pvalues{key}"] = model.permutation_pvalues_
        combined.uns["tanpopo"][cmd_id][
            f"permutation_max_abs_eigenvalues{key}"
        ] = model.permutation_max_abs_eigenvalues_
    if output is not None:
        combined.write_h5ad(output)
    return combined


@app.command("label-decomposition", no_args_is_help=True)
def label_decomposition(
    fname: InputPath,
    label_key: LabelKey,
    output: OutputPath = None,
    cmd_id: ExperimentId = "label_decomposition",
    radius: Radius = None,
    n_components: Components = 8,
    layer: Layer = None,
    graph_normalisation: GraphNormalisation = GraphNormalisationTypes.symmetric,
    include: Include = None,
    exclude: Exclude = None,
    transform: Transform = None,
    min_counts: MinCounts = 10,
    min_spot_fraction: MinSpotFraction = None,
    target_sum: TargetSum = None,
    covariates: Covariates = None,
    dtype: Dtype = Dtypes.float64,
    verbose: Verbose = False,
):
    """Decompose total spatial covariance into between-label, within-label and coupling terms."""
    pre = _pre_args(
        target_sum,
        transform,
        min_counts,
        min_spot_fraction,
        covariates,
        label_key,
        layer,
        include,
        exclude,
    )
    adata = load_preprocess_sample(fname, verbose=verbose, **pre)
    _require_obs_key(adata, label_key)
    W, coords, cov = get_spatial_from_anndata(adata, layer)
    if radius is None:
        radius = _radius_from_mask(adata)
    model = LabelDecompositionModel(
        radius,
        graph_normalisation=as_value(graph_normalisation),
        dtype=as_value(dtype),
        verbose=verbose,
    ).fit(W, coords, np.asarray(adata.obs[label_key]), n_components, covariates=cov)
    for name, result in model.results_.items():
        prefix = f"tanpopo_{cmd_id}_{name}"
        adata.varm[f"{prefix}_gene_loadings"] = result["gene_loadings"]
        full = np.full((adata.n_obs, result["spot_modes"].shape[1]), np.nan)
        full[model.samples[0].obs_idx] = result["spot_modes"]
        adata.obsm[f"{prefix}_spot_modes"] = full
        adata.uns.setdefault("tanpopo", {}).setdefault(cmd_id, {})
        adata.uns["tanpopo"][cmd_id][f"eigenvalues_{name}"] = result["eigenvalues"]
    add_metadata(
        adata,
        cmd_id,
        pre,
        {
            "radius": radius,
            "graph_normalisation": as_value(graph_normalisation),
            "decomposition": "total = between + within + coupling",
        },
        {"label_key": label_key, "operator_additivity_error": model.operator_additivity_error()},
    )
    if output is not None:
        adata.write_h5ad(output)
    return adata


@app.command("cross-programs", no_args_is_help=True)
def cross_programs(
    fname: InputPath,
    label_key: LabelKey,
    target_labels: TargetLabels,
    neighbour_labels: NeighbourLabels,
    output: OutputPath = None,
    cmd_id: ExperimentId = "cross",
    radius: Radius = None,
    n_components: Components = 8,
    layer: Layer = None,
    objective: Objective = ObjectiveTypes.covariance,
    expression_rank: ExpressionRank = 50,
    gain_ridge: GainRidge = 1e-8,
    graph_normalisation: GraphNormalisation = GraphNormalisationTypes.symmetric,
    include: Include = None,
    exclude: Exclude = None,
    transform: Transform = None,
    min_counts: MinCounts = 10,
    min_spot_fraction: MinSpotFraction = None,
    target_sum: TargetSum = None,
    covariates: Covariates = None,
    dtype: Dtype = Dtypes.float64,
    verbose: Verbose = False,
):
    """Paired neighbour-context and target-response gene programs."""
    pre = _pre_args(
        target_sum,
        transform,
        min_counts,
        min_spot_fraction,
        covariates,
        label_key,
        layer,
        include,
        exclude,
    )
    adata = load_preprocess_sample(fname, verbose=verbose, **pre)
    _require_obs_key(adata, label_key)
    targets = _parse_csv_labels(adata, target_labels, label_key, allow_all=False)
    neighbours = _parse_csv_labels(adata, neighbour_labels, label_key, allow_all=False)
    target_mask = adata.obs[label_key].isin(targets).to_numpy()
    neighbour_mask = adata.obs[label_key].isin(neighbours).to_numpy()
    W, coords, cov = get_spatial_from_anndata(adata, layer)
    if radius is None:
        radius = _radius_from_mask(adata, target_mask | neighbour_mask)
    model = CrossSpatialProgramModel(
        radius,
        objective=as_value(objective),
        expression_rank=expression_rank,
        gain_ridge=gain_ridge,
        graph_normalisation=as_value(graph_normalisation),
        dtype=as_value(dtype),
        verbose=verbose,
    ).fit(
        W,
        coords,
        n_components,
        target_mask=target_mask,
        neighbour_mask=neighbour_mask,
        covariates=cov,
    )
    store_cross_result(adata, model, cmd_id)
    add_metadata(
        adata,
        cmd_id,
        pre,
        {
            "radius": radius,
            "objective": as_value(objective),
            "expression_rank": expression_rank,
            "gain_ridge": gain_ridge,
            "graph_normalisation": as_value(graph_normalisation),
        },
        {
            "target_labels": [str(x) for x in targets],
            "neighbour_labels": [str(x) for x in neighbours],
        },
    )
    if output is not None:
        adata.write_h5ad(output)
    return adata


@app.command("pca-programs", no_args_is_help=True)
def pca_programs(
    fname: InputPath,
    output: OutputPath = None,
    cmd_id: ExperimentId = "pca",
    n_components: Components = 20,
    layer: Layer = None,
    label_key: LabelKey = None,
    subset_labels: Labels = None,
    include: Include = None,
    exclude: Exclude = None,
    transform: Transform = None,
    min_counts: MinCounts = 10,
    min_spot_fraction: MinSpotFraction = None,
    target_sum: TargetSum = None,
    dtype: Dtype = Dtypes.float64,
    verbose: Verbose = False,
):
    """Ordinary PCA globally or within selected cell types.

    This is the non-spatial baseline for spatial-programs. PCA is performed
    on exactly the selected expression layer and, when --labels is supplied,
    only on cells belonging to that label.
    """
    import scanpy as sc

    pre = _pre_args(
        target_sum,
        transform,
        min_counts,
        min_spot_fraction,
        None,  # no covariate residualisation for ordinary PCA
        label_key,
        layer,
        include,
        exclude,
    )

    adata = load_preprocess_sample(fname, verbose=verbose, **pre)

    targets = _parse_csv_labels(adata, subset_labels, label_key) if subset_labels else [None]

    for target in targets:
        if target is None:
            mask = np.ones(adata.n_obs, dtype=bool)
            key = ""
        else:
            mask = (adata.obs[label_key] == target).to_numpy()
            key = f"_{str(target).replace(' ', '_')}"

        if mask.sum() < 2:
            raise typer.BadParameter(f"Not enough cells selected for PCA: {target!r}")

        # Work on a copy so Scanpy's standard PCA keys do not overwrite
        # anything else in the output AnnData.
        work = adata[mask].copy()

        pca_key = "_tanpopo_pca"

        sc.pp.pca(
            work,
            n_comps=n_components,
            layer=layer,
            zero_center=True,
            svd_solver="arpack",
            mask_var=None,
            dtype=as_value(dtype),
            key_added=pca_key,
        )

        prefix = f"tanpopo_{cmd_id}{key}"

        # Same shape/naming convention as Tanpopo gene loadings, so
        # tanpopo-sim evaluate-subspace can be used unchanged.
        adata.varm[f"{prefix}_gene_loadings"] = np.asarray(work.varm[pca_key])

        # Store PCA cell scores too. Cells outside a restricted analysis are NaN.
        modes = np.full(
            (adata.n_obs, n_components),
            np.nan,
            dtype=float,
        )
        modes[mask] = np.asarray(work.obsm[pca_key])
        adata.obsm[f"{prefix}_spot_modes"] = modes

        # Keep ordinary PCA variance diagnostics.
        adata.uns.setdefault("tanpopo", {}).setdefault(cmd_id, {})
        adata.uns["tanpopo"][cmd_id][f"variance{key}"] = np.asarray(work.uns[pca_key]["variance"])
        adata.uns["tanpopo"][cmd_id][f"variance_ratio{key}"] = np.asarray(
            work.uns[pca_key]["variance_ratio"]
        )
        adata.uns["tanpopo"][cmd_id][f"conditioned_label{key}"] = (
            None if target is None else str(target)
        )

    if output is not None:
        adata.write_h5ad(output)

    return adata


@app.command("estimate-spacing")
def estimate_spacing(fname: InputPath):
    """Mean nearest-neighbour distance for a spatial sample."""
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError("estimate-spacing requires scanpy") from exc
    adata = sc.read_h5ad(fname)
    value = neighbour_spacing(adata.obsm["spatial"])
    typer.echo(f"{value:.8g}")


def main():
    app()


if __name__ == "__main__":
    main()
