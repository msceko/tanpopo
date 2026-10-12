from __future__ import annotations

from pathlib import Path

import numpy as np
import typer

from tanpopo.cli import (
    Components,
    Covariates,
    DistanceBins,
    Dtype,
    Dtypes,
    Exclude,
    ExperimentId,
    ExpressionRank,
    FixedComponents,
    GainRidge,
    GeometryNormalisation,
    GeometryNormalisationTypes,
    GraphNormalisation,
    GraphNormalisationTypes,
    Include,
    InputPath,
    InputPaths,
    InputPathsA,
    InputPathsB,
    LabelKey,
    Labels,
    Layer,
    MaxComponents,
    MinCounts,
    MinSpotFraction,
    ModeAlpha,
    NeighbourLabels,
    Objective,
    ObjectiveTypes,
    OutputPath,
    Permutations,
    Radius,
    RandomSeed,
    SampleNames,
    SampleWeighting,
    SampleWeightingTypes,
    SpotOperator,
    SpotOperatorTypes,
    TargetLabels,
    TargetSum,
    Transform,
    Verbose,
)
from tanpopo.data import Groups, get_spatial_from_anndata
from tanpopo.io import (
    add_metadata,
    load_preprocess_sample,
    load_preprocess_samples,
    geometry_diagnostics_cfg,
    model_cfg,
    preprocess_cfg,
    store_cross_result,
    store_multi_sample_result,
    store_sample_result,
    store_multi_sample_cross_result,
)
from tanpopo.geometry import neighbour_spacing
from tanpopo.analysis import (
    compare_cross_results,
    compare_program_results,
    compare_spot_modes as compare_spot_mode_results,
)
from tanpopo.diagnostics import (
    spatial_information_curve_from_arrays,
    spatial_information_from_arrays,
)
from tanpopo.results import (
    CrossProgramResult,
    ProgramResult,
    list_results,
    load_result,
    register_program_result,
)
from tanpopo.operators import gene_expression_variance, projected_svd
from tanpopo.projection import SpotProjector
from tanpopo.models import (
    CrossSpatialProgramModel,
    DifferentialCrossSpatialProgramModel,
    DifferentialSpatialProgramModel,
    SharedCrossSpatialProgramModel,
    LabelDecompositionModel,
    SharedSpatialProgramModel,
    SpatialProgramModel,
)
from tanpopo.utils import as_value, pd_dtype


app = typer.Typer(
    name="tanpopo",
    help=("Geometry-standardised spatial covariance programs for spatial transcriptomics."),
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
        raise typer.BadParameter("shared and differential currently accept one --labels value")
    label = labels[0]
    for adata in adatas[1:]:
        _require_obs_key(adata, label_key)
        if label not in set(adata.obs[label_key].unique()):
            raise typer.BadParameter(f"Label {label!r} is not present in every sample")
    return label


def _multisample_labels(adatas, text, label_key, option_name):
    if text is None:
        raise typer.BadParameter(f"{option_name} is required")
    labels = _parse_csv_labels(adatas[0], text, label_key, allow_all=False)
    for adata in adatas[1:]:
        _require_obs_key(adata, label_key)
        available = set(adata.obs[label_key].unique())
        missing = [label for label in labels if label not in available]
        if missing:
            raise typer.BadParameter(
                f"{option_name} label(s) {missing} are not present in every sample"
            )
    return labels


def _component_plan(components, max_components):
    text = str(components).strip().lower()
    if text == "auto":
        if int(max_components) < 1:
            raise typer.BadParameter("--max-components must be positive")
        return int(max_components), True
    try:
        value = int(text)
    except ValueError as exc:
        raise typer.BadParameter("--components must be a positive integer or 'auto'") from exc
    if value < 1:
        raise typer.BadParameter("--components must be positive")
    return value, False


def _apply_mode_inference(model, automatic, permutations, alpha, seed):
    permutations = int(permutations)
    if automatic and not 0 < float(alpha) < 1:
        raise typer.BadParameter("--mode-alpha must lie strictly between 0 and 1")
    if automatic and permutations <= 0:
        raise typer.BadParameter("--components auto requires --permutations > 0")
    if automatic and 1.0 / (permutations + 1.0) > float(alpha):
        minimum = int(np.ceil(1.0 / float(alpha) - 1.0))
        raise typer.BadParameter(
            f"--components auto at --mode-alpha {alpha:g} requires at least "
            f"{minimum} permutations for sufficient p-value resolution"
        )
    if permutations > 0:
        model.permutation_test(permutations, seed=seed)
    metadata = {"automatic": bool(automatic), "permutations": permutations}
    if automatic:
        selection = model.select_modes(alpha=alpha)
        metadata.update(selection.summary())
        metadata["pvalues"] = selection.pvalues
        metadata["null_maxima"] = selection.null_maxima
        model.truncate_modes(selection.selected_modes)
    elif permutations > 0:
        metadata["pvalues"] = np.asarray(model.permutation_pvalues_)
        if hasattr(model, "permutation_max_eigenvalues_"):
            metadata["null_maxima"] = np.asarray(model.permutation_max_eigenvalues_)
        elif hasattr(model, "permutation_max_singular_values_"):
            metadata["null_maxima"] = np.asarray(model.permutation_max_singular_values_)
    metadata["reported_modes"] = int(
        len(model.singular_values) if hasattr(model, "singular_values") else len(model.eigenvalues)
    )
    return metadata


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


def _radius_from_masks(adatas, masks):
    """Choose one fallback radius without privileging the first biological sample."""
    spacings = []
    for adata, mask in zip(adatas, masks):
        coords = np.asarray(adata.obsm["spatial"])
        if mask is not None:
            coords = coords[np.asarray(mask, dtype=bool)]
        if len(coords) < 2:
            raise typer.BadParameter(
                "At least two selected cells are required in every sample to estimate radius"
            )
        spacings.append(neighbour_spacing(coords))
    radius = 3.5 * float(np.median(spacings))
    typer.echo(
        "Warning: no --radius specified for a multi-sample analysis; using "
        f"{radius:.3g} (3.5 x median sample mean nearest-neighbour distance). "
        "For interpretable cross-tissue comparisons, specify a physical radius explicitly.",
        err=True,
    )
    return radius


@app.command("programs", no_args_is_help=True)
def programs(
    fname: InputPath,
    output: OutputPath = None,
    analysis_id: ExperimentId = "spatial",
    radius: Radius = None,
    n_components: Components = "8",
    max_components: MaxComponents = 30,
    mode_alpha: ModeAlpha = 0.05,
    layer: Layer = None,
    label_key: LabelKey = None,
    subset_labels: Labels = None,
    objective: Objective = ObjectiveTypes.covariance,
    expression_rank: ExpressionRank = 50,
    gain_ridge: GainRidge = 1e-8,
    graph_normalisation: GraphNormalisation = GraphNormalisationTypes.symmetric,
    geometry_normalisation: GeometryNormalisation = GeometryNormalisationTypes.distance,
    distance_bins: DistanceBins = 10,
    spot_operator: SpotOperator = SpotOperatorTypes.sample,
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
    """Spatial programs globally or within selected cell types."""
    objective_value = as_value(objective)
    fit_components, automatic = _component_plan(n_components, max_components)
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
            geometry_normalisation=as_value(geometry_normalisation),
            distance_bins=distance_bins,
            dtype=as_value(dtype),
            verbose=verbose,
        ).fit(W, coords, fit_components, labels=labels, covariates=cov, masks=mask)
        inference = _apply_mode_inference(model, automatic, permutations, mode_alpha, seed)
        key = "" if target is None else f"_{str(target).replace(' ', '_')}"
        store_sample_result(adata, model, analysis_id, key)
        add_metadata(
            adata,
            analysis_id,
            pre,
            model_cfg(model),
            {
                "conditioned_label": None if target is None else str(target),
                "geometry_diagnostics": geometry_diagnostics_cfg(model),
                "mode_inference": inference,
            },
            variant="default" if target is None else key.lstrip("_"),
            analysis_type="program",
        )
    if output is not None:
        adata.write_h5ad(output)
    return adata


@app.command("shared", no_args_is_help=True)
def shared(
    fnames: InputPaths,
    output: OutputPath = None,
    analysis_id: ExperimentId = "shared",
    radius: Radius = None,
    sample_names: SampleNames = None,
    n_components: Components = "8",
    max_components: MaxComponents = 30,
    mode_alpha: ModeAlpha = 0.05,
    layer: Layer = None,
    label_key: LabelKey = None,
    subset_labels: Labels = None,
    objective: Objective = ObjectiveTypes.covariance,
    expression_rank: ExpressionRank = 50,
    gain_ridge: GainRidge = 1e-8,
    graph_normalisation: GraphNormalisation = GraphNormalisationTypes.symmetric,
    geometry_normalisation: GeometryNormalisation = GeometryNormalisationTypes.distance,
    distance_bins: DistanceBins = 10,
    spot_operator: SpotOperator = SpotOperatorTypes.sample,
    sample_weighting: SampleWeighting = SampleWeightingTypes.n_spots,
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
    """Shared spatial programs across biological samples, optionally within one cell type."""
    if len(fnames) < 2:
        raise typer.BadParameter("shared requires at least two inputs")
    objective_value = as_value(objective)
    fit_components, automatic = _component_plan(n_components, max_components)
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
        radius = _radius_from_masks(adatas, masks)
    model = SharedSpatialProgramModel(
        radius,
        objective=objective_value,
        spot_operator=as_value(spot_operator),
        sample_weighting=as_value(sample_weighting),
        expression_rank=expression_rank,
        gain_ridge=gain_ridge,
        graph_normalisation=as_value(graph_normalisation),
        geometry_normalisation=as_value(geometry_normalisation),
        distance_bins=distance_bins,
        dtype=as_value(dtype),
        verbose=verbose,
    ).fit(W, coords, fit_components, labels=labels, covariates=covs, masks=masks)
    inference = _apply_mode_inference(model, automatic, permutations, mode_alpha, seed)

    key = "" if target is None else f"_{str(target).replace(' ', '_')}"
    combined = store_multi_sample_result(adatas, sample_names, model, analysis_id, key)
    add_metadata(
        combined,
        analysis_id,
        pre,
        model_cfg(model),
        {
            "sample_names": sample_names,
            "sample_coefficients": model.sample_coefficients_,
            "conditioned_label": None if target is None else str(target),
            "geometry_diagnostics": geometry_diagnostics_cfg(model, sample_names),
            "mode_inference": inference,
        },
        variant="default" if target is None else key.lstrip("_"),
        analysis_type="shared_program",
    )
    if output is not None:
        combined.write_h5ad(output)
    return combined


@app.command("differential", no_args_is_help=True)
def differential(
    fnames_a: InputPathsA,
    fnames_b: InputPathsB,
    output: OutputPath = None,
    analysis_id: ExperimentId = "differential",
    radius: Radius = None,
    sample_names: SampleNames = None,
    n_components: Components = "8",
    max_components: MaxComponents = 30,
    mode_alpha: ModeAlpha = 0.05,
    layer: Layer = None,
    label_key: LabelKey = None,
    subset_labels: Labels = None,
    objective: Objective = ObjectiveTypes.covariance,
    expression_rank: ExpressionRank = 50,
    gain_ridge: GainRidge = 1e-8,
    graph_normalisation: GraphNormalisation = GraphNormalisationTypes.symmetric,
    geometry_normalisation: GeometryNormalisation = GeometryNormalisationTypes.distance,
    distance_bins: DistanceBins = 10,
    spot_operator: SpotOperator = SpotOperatorTypes.sample,
    sample_weighting: SampleWeighting = SampleWeightingTypes.n_spots,
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
    """Spatial covariance programs differing between biological sample groups."""
    if not fnames_a or not fnames_b:
        raise typer.BadParameter("At least one sample is required in each group")
    fnames = list(fnames_a) + list(fnames_b)
    objective_value = as_value(objective)
    fit_components, automatic = _component_plan(n_components, max_components)
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
        radius = _radius_from_masks(adatas, masks)
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
        geometry_normalisation=as_value(geometry_normalisation),
        distance_bins=distance_bins,
        dtype=as_value(dtype),
        verbose=verbose,
    ).fit(W, coords, fit_components, labels=labels, covariates=covs, masks=masks)
    inference = _apply_mode_inference(model, automatic, permutations, mode_alpha, seed)
    key = "" if target is None else f"_{str(target).replace(' ', '_')}"
    combined = store_multi_sample_result(adatas, sample_names, model, analysis_id, key)
    add_metadata(
        combined,
        analysis_id,
        pre,
        model_cfg(model),
        {
            "sample_names": sample_names,
            "group_a": sample_names[:n_a],
            "group_b": sample_names[n_a:],
            "sample_coefficients": model.sample_coefficients_,
            "conditioned_label": None if target is None else str(target),
            "permutations": int(permutations),
            "geometry_diagnostics": geometry_diagnostics_cfg(model, sample_names),
            "mode_inference": inference,
        },
        variant="default" if target is None else key.lstrip("_"),
        analysis_type="differential_program",
    )
    if output is not None:
        combined.write_h5ad(output)
    return combined


@app.command("decompose-labels", no_args_is_help=True)
def decompose_labels(
    fname: InputPath,
    label_key: LabelKey,
    output: OutputPath = None,
    analysis_id: ExperimentId = "decomposition",
    radius: Radius = None,
    n_components: FixedComponents = 8,
    layer: Layer = None,
    graph_normalisation: GraphNormalisation = GraphNormalisationTypes.symmetric,
    geometry_normalisation: GeometryNormalisation = GeometryNormalisationTypes.distance,
    distance_bins: DistanceBins = 10,
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
    """Decompose spatial covariance into between-label, within-label and coupling terms."""
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
        geometry_normalisation=as_value(geometry_normalisation),
        distance_bins=distance_bins,
        dtype=as_value(dtype),
        verbose=verbose,
    ).fit(W, coords, np.asarray(adata.obs[label_key]), n_components, covariates=cov)
    for name, result in model.results_.items():
        prefix = f"tanpopo_{analysis_id}_{name}"
        loadings_key = f"{prefix}_gene_loadings"
        modes_key = f"{prefix}_spot_modes"
        adata.varm[loadings_key] = result["gene_loadings"]
        full = np.full((adata.n_obs, result["spot_modes"].shape[1]), np.nan)
        full[model.samples[0].obs_idx] = result["spot_modes"]
        adata.obsm[modes_key] = full
        register_program_result(
            adata,
            analysis_id,
            loadings_key=loadings_key,
            modes_key=modes_key,
            values=result["eigenvalues"],
            variant=name,
            analysis_type="label_decomposition",
        )
        add_metadata(
            adata,
            analysis_id,
            pre,
            {
                "radius": radius,
                "graph_normalisation": as_value(graph_normalisation),
                "geometry_normalisation": as_value(geometry_normalisation),
                "distance_bins": distance_bins,
                "decomposition": "total = between + within + coupling",
            },
            {
                "label_key": label_key,
                "operator_additivity_error": model.operator_additivity_error(),
                "geometry_diagnostics": geometry_diagnostics_cfg(model),
                "component": name,
            },
            variant=name,
            analysis_type="label_decomposition",
        )
    if output is not None:
        adata.write_h5ad(output)
    return adata


def _multisample_cross_inputs(adatas, label_key, target_labels, neighbour_labels, layer):
    """Resolve labels and assemble target/neighbour masks for multi-sample workflows."""
    _require_obs_key(adatas[0], label_key)
    targets = _multisample_labels(adatas, target_labels, label_key, "--target-labels")
    neighbours = _multisample_labels(adatas, neighbour_labels, label_key, "--neighbour-labels")
    W, coords, covs, target_masks, neighbour_masks = [], [], [], [], []
    for adata in adatas:
        w, xy, cov = get_spatial_from_anndata(adata, layer)
        W.append(w)
        coords.append(xy)
        covs.append(cov)
        target_masks.append(adata.obs[label_key].isin(targets).to_numpy())
        neighbour_masks.append(adata.obs[label_key].isin(neighbours).to_numpy())
    return targets, neighbours, W, coords, covs, target_masks, neighbour_masks


def _multisample_cross_radius(adatas, target_masks, neighbour_masks, radius):
    if radius is not None:
        return radius
    union_masks = [target | neighbour for target, neighbour in zip(target_masks, neighbour_masks)]
    return _radius_from_masks(adatas, union_masks)


@app.command("cross-shared", no_args_is_help=True)
def cross_shared(
    fnames: InputPaths,
    label_key: LabelKey,
    target_labels: TargetLabels,
    neighbour_labels: NeighbourLabels,
    output: OutputPath = None,
    analysis_id: ExperimentId = "cross_shared",
    radius: Radius = None,
    sample_names: SampleNames = None,
    n_components: Components = "8",
    max_components: MaxComponents = 30,
    mode_alpha: ModeAlpha = 0.05,
    layer: Layer = None,
    objective: Objective = ObjectiveTypes.covariance,
    expression_rank: ExpressionRank = 50,
    gain_ridge: GainRidge = 1e-8,
    graph_normalisation: GraphNormalisation = GraphNormalisationTypes.symmetric,
    geometry_normalisation: GeometryNormalisation = GeometryNormalisationTypes.distance,
    distance_bins: DistanceBins = 10,
    sample_weighting: SampleWeighting = SampleWeightingTypes.n_spots,
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
    """Paired target-neighbour gene programs shared across biological samples."""
    if len(fnames) < 2:
        raise typer.BadParameter("cross-shared requires at least two inputs")
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
    (
        targets,
        neighbours,
        W,
        coords,
        covs,
        target_masks,
        neighbour_masks,
    ) = _multisample_cross_inputs(adatas, label_key, target_labels, neighbour_labels, layer)
    radius = _multisample_cross_radius(adatas, target_masks, neighbour_masks, radius)
    fit_components, automatic = _component_plan(n_components, max_components)
    model = SharedCrossSpatialProgramModel(
        radius,
        objective=as_value(objective),
        sample_weighting=as_value(sample_weighting),
        expression_rank=expression_rank,
        gain_ridge=gain_ridge,
        graph_normalisation=as_value(graph_normalisation),
        geometry_normalisation=as_value(geometry_normalisation),
        distance_bins=distance_bins,
        dtype=as_value(dtype),
        verbose=verbose,
    ).fit(
        W,
        coords,
        fit_components,
        target_masks=target_masks,
        neighbour_masks=neighbour_masks,
        covariates=covs,
    )
    inference = _apply_mode_inference(model, automatic, permutations, mode_alpha, seed)
    combined = store_multi_sample_cross_result(adatas, sample_names, model, analysis_id)
    add_metadata(
        combined,
        analysis_id,
        pre,
        model_cfg(model),
        {
            "sample_names": sample_names,
            "target_labels": [str(x) for x in targets],
            "neighbour_labels": [str(x) for x in neighbours],
            "geometry_diagnostics": geometry_diagnostics_cfg(model, sample_names),
            "mode_inference": inference,
        },
        analysis_type="shared_cross",
    )
    if output is not None:
        combined.write_h5ad(output)
    return combined


@app.command("cross-differential", no_args_is_help=True)
def cross_differential(
    fnames_a: InputPathsA,
    fnames_b: InputPathsB,
    label_key: LabelKey,
    target_labels: TargetLabels,
    neighbour_labels: NeighbourLabels,
    output: OutputPath = None,
    analysis_id: ExperimentId = "cross_differential",
    radius: Radius = None,
    sample_names: SampleNames = None,
    n_components: Components = "8",
    max_components: MaxComponents = 30,
    mode_alpha: ModeAlpha = 0.05,
    layer: Layer = None,
    objective: Objective = ObjectiveTypes.covariance,
    expression_rank: ExpressionRank = 50,
    gain_ridge: GainRidge = 1e-8,
    graph_normalisation: GraphNormalisation = GraphNormalisationTypes.symmetric,
    geometry_normalisation: GeometryNormalisation = GeometryNormalisationTypes.distance,
    distance_bins: DistanceBins = 10,
    sample_weighting: SampleWeighting = SampleWeightingTypes.n_spots,
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
    """Target-neighbour cross-covariance programs differing between sample groups."""
    if not fnames_a or not fnames_b:
        raise typer.BadParameter("At least one sample is required in each group")
    fnames = list(fnames_a) + list(fnames_b)
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
    (
        targets,
        neighbours,
        W,
        coords,
        covs,
        target_masks,
        neighbour_masks,
    ) = _multisample_cross_inputs(adatas, label_key, target_labels, neighbour_labels, layer)
    radius = _multisample_cross_radius(adatas, target_masks, neighbour_masks, radius)
    n_a = len(fnames_a)
    fit_components, automatic = _component_plan(n_components, max_components)
    model = DifferentialCrossSpatialProgramModel(
        radius,
        positive_samples=range(n_a),
        negative_samples=range(n_a, len(fnames)),
        objective=as_value(objective),
        sample_weighting=as_value(sample_weighting),
        expression_rank=expression_rank,
        gain_ridge=gain_ridge,
        graph_normalisation=as_value(graph_normalisation),
        geometry_normalisation=as_value(geometry_normalisation),
        distance_bins=distance_bins,
        dtype=as_value(dtype),
        verbose=verbose,
    ).fit(
        W,
        coords,
        fit_components,
        target_masks=target_masks,
        neighbour_masks=neighbour_masks,
        covariates=covs,
    )
    inference = _apply_mode_inference(model, automatic, permutations, mode_alpha, seed)
    combined = store_multi_sample_cross_result(adatas, sample_names, model, analysis_id)
    add_metadata(
        combined,
        analysis_id,
        pre,
        model_cfg(model),
        {
            "sample_names": sample_names,
            "group_a": sample_names[:n_a],
            "group_b": sample_names[n_a:],
            "target_labels": [str(x) for x in targets],
            "neighbour_labels": [str(x) for x in neighbours],
            "permutations": int(permutations),
            "geometry_diagnostics": geometry_diagnostics_cfg(model, sample_names),
            "mode_inference": inference,
        },
        analysis_type="differential_cross",
    )
    if output is not None:
        combined.write_h5ad(output)
    return combined


@app.command("cross", no_args_is_help=True)
def cross(
    fname: InputPath,
    label_key: LabelKey,
    target_labels: TargetLabels,
    neighbour_labels: NeighbourLabels,
    output: OutputPath = None,
    analysis_id: ExperimentId = "cross",
    radius: Radius = None,
    n_components: Components = "8",
    max_components: MaxComponents = 30,
    mode_alpha: ModeAlpha = 0.05,
    layer: Layer = None,
    objective: Objective = ObjectiveTypes.covariance,
    expression_rank: ExpressionRank = 50,
    gain_ridge: GainRidge = 1e-8,
    graph_normalisation: GraphNormalisation = GraphNormalisationTypes.symmetric,
    geometry_normalisation: GeometryNormalisation = GeometryNormalisationTypes.distance,
    distance_bins: DistanceBins = 10,
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
    fit_components, automatic = _component_plan(n_components, max_components)
    model = CrossSpatialProgramModel(
        radius,
        objective=as_value(objective),
        expression_rank=expression_rank,
        gain_ridge=gain_ridge,
        graph_normalisation=as_value(graph_normalisation),
        geometry_normalisation=as_value(geometry_normalisation),
        distance_bins=distance_bins,
        dtype=as_value(dtype),
        verbose=verbose,
    ).fit(
        W,
        coords,
        fit_components,
        target_mask=target_mask,
        neighbour_mask=neighbour_mask,
        covariates=cov,
    )
    inference = _apply_mode_inference(model, automatic, permutations, mode_alpha, seed)
    store_cross_result(adata, model, analysis_id)
    add_metadata(
        adata,
        analysis_id,
        pre,
        model_cfg(model),
        {
            "target_labels": [str(x) for x in targets],
            "neighbour_labels": [str(x) for x in neighbours],
            "geometry_diagnostics": geometry_diagnostics_cfg(model),
            "mode_inference": inference,
        },
    )
    if output is not None:
        adata.write_h5ad(output)
    return adata


@app.command("pca", no_args_is_help=True)
def pca(
    fname: InputPath,
    output: OutputPath = None,
    analysis_id: ExperimentId = "pca",
    n_components: FixedComponents = 20,
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

    This is the non-spatial baseline for programs. PCA is performed
    on exactly the selected expression layer and, when --labels is supplied,
    only on cells belonging to that label.
    """
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
    W, _, _ = get_spatial_from_anndata(adata, layer)

    for target in targets:
        if target is None:
            mask = np.ones(adata.n_obs, dtype=bool)
            key = ""
        else:
            mask = (adata.obs[label_key] == target).to_numpy()
            key = f"_{str(target).replace(' ', '_')}"
        n_selected = int(mask.sum())
        if n_selected < 2:
            raise typer.BadParameter(f"Not enough cells selected for PCA: {target!r}")

        W_sub = W[mask]
        projector = SpotProjector(n_selected, Groups.single(n_selected), dtype=as_value(dtype))
        U, singular, Vt = projected_svd(
            W_sub,
            projector,
            n_components,
            dtype=np.dtype(as_value(dtype)),
        )
        k = len(singular)
        loadings = np.asarray(Vt.T)
        scores = np.asarray(U * singular[None, :])
        variance = np.square(singular) / max(n_selected - 1, 1)
        total_ss = float(gene_expression_variance(W_sub, projector, block_size=256).sum())
        variance_ratio = np.divide(
            np.square(singular),
            total_ss,
            out=np.zeros_like(singular),
            where=total_ss > 0,
        )

        prefix = f"tanpopo_{analysis_id}{key}"
        adata.varm[f"{prefix}_gene_loadings"] = loadings
        modes = np.full((adata.n_obs, k), np.nan, dtype=float)
        modes[mask] = scores
        adata.obsm[f"{prefix}_spot_modes"] = modes

        variant = "default" if target is None else key.lstrip("_")
        register_program_result(
            adata,
            analysis_id,
            loadings_key=f"{prefix}_gene_loadings",
            modes_key=f"{prefix}_spot_modes",
            values=variance,
            value_name="variance",
            variant=variant,
            analysis_type="pca",
        )
        add_metadata(
            adata,
            analysis_id,
            pre,
            {"components": int(k), "baseline": "pca"},
            {
                "variance_ratio": variance_ratio,
                "conditioned_label": None if target is None else str(target),
            },
            variant=variant,
            analysis_type="pca",
        )

    if output is not None:
        adata.write_h5ad(output)

    return adata


def _read_h5ad(path):
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("AnnData utilities require anndata") from exc
    return ad.read_h5ad(path)


def _parse_mode_indices(text, n_modes):
    if text is None:
        return list(range(n_modes))
    values = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 0 or value >= n_modes:
            raise typer.BadParameter(f"Mode {value} is outside 0..{n_modes - 1}")
        values.append(value)
    return values


@app.command("inspect")
def inspect_sample(
    fname: InputPath,
    spacing: bool = typer.Option(True, "--spacing/--no-spacing"),
):
    """Inspect AnnData annotations, Tanpopo analyses and spatial spacing."""
    adata = _read_h5ad(fname)
    typer.echo(str(adata))
    results = list_results(adata)
    if results:
        typer.echo("\nTanpopo analyses:")
        for result in results:
            typer.echo(
                "  {analysis_id} [{analysis_type}/{variant}] {n_modes} modes".format(**result)
            )
    else:
        typer.echo("\nNo Tanpopo analyses found.")
    if spacing and "spatial" in adata.obsm and adata.n_obs > 1:
        spacing_value = neighbour_spacing(adata.obsm["spatial"])
        typer.echo(f"Mean nearest-neighbour spacing: {spacing_value:.6g}")


@app.command("preprocess", no_args_is_help=True)
def preprocess_sample(
    fname: InputPath,
    output: OutputPath,
    layer: Layer = None,
    label_key: LabelKey = None,
    include: Include = None,
    exclude: Exclude = None,
    transform: Transform = None,
    min_counts: MinCounts = 10,
    min_spot_fraction: MinSpotFraction = None,
    target_sum: TargetSum = None,
    covariates: Covariates = None,
    verbose: Verbose = False,
):
    """Preprocess a sample without fitting spatial programs."""
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
    if output is None:
        raise typer.BadParameter("--output is required")
    adata.write_h5ad(output)
    return adata


@app.command("export", no_args_is_help=True)
def export_result(
    fname: InputPath,
    output: OutputPath,
    analysis_id: str = typer.Option("spatial", "--analysis-id", "-id"),
    variant: str | None = typer.Option(None, "--variant"),
    side: str | None = typer.Option(None, "--side", help="target or neighbour for cross results"),
):
    """Export gene loadings from any tanpopo result to CSV."""
    adata = _read_h5ad(fname)
    result = load_result(adata, analysis_id, variant=variant)
    if isinstance(result, ProgramResult):
        frame = result.loadings_frame()
    else:
        if side not in {"target", "neighbour"}:
            raise typer.BadParameter("Cross results require --side target or --side neighbour")
        frame = result.target_frame() if side == "target" else result.neighbour_frame()
    frame.to_csv(output, index=True, index_label="gene")


@app.command("compare", no_args_is_help=True)
def compare_results(
    input_a: Path = typer.Option(..., "--input-a", exists=True, dir_okay=False),
    input_b: Path = typer.Option(..., "--input-b", exists=True, dir_okay=False),
    analysis_a: str = typer.Option(..., "--analysis-a"),
    analysis_b: str = typer.Option(..., "--analysis-b"),
    variant_a: str | None = typer.Option(None, "--variant-a"),
    variant_b: str | None = typer.Option(None, "--variant-b"),
    output: Path | None = typer.Option(None, "--output", "-o", dir_okay=False),
    top_genes: int = typer.Option(20, "--top-genes", min=1),
):
    """Compare two Tanpopo gene-program spaces with one-to-one mode matching."""
    a = load_result(_read_h5ad(input_a), analysis_a, variant=variant_a)
    b = load_result(_read_h5ad(input_b), analysis_b, variant=variant_b)
    if isinstance(a, ProgramResult) and isinstance(b, ProgramResult):
        frame = compare_program_results(a, b, top_genes=top_genes)
        for key, value in frame.attrs.items():
            typer.echo(f"{key}: {value}")
    elif isinstance(a, CrossProgramResult) and isinstance(b, CrossProgramResult):
        frame = compare_cross_results(a, b, top_genes=top_genes)
    else:
        raise typer.BadParameter("Both analyses must be the same result kind")
    typer.echo(frame.to_string(index=False))
    if output is not None:
        frame.to_csv(output, index=False)


@app.command("compare-spots", no_args_is_help=True)
def compare_spots(
    input_a: Path = typer.Option(..., "--input-a", exists=True, dir_okay=False),
    input_b: Path = typer.Option(..., "--input-b", exists=True, dir_okay=False),
    analysis_a: str = typer.Option(..., "--analysis-a"),
    analysis_b: str = typer.Option(..., "--analysis-b"),
    variant_a: str | None = typer.Option(None, "--variant-a"),
    variant_b: str | None = typer.Option(None, "--variant-b"),
    output: Path | None = typer.Option(None, "--output", "-o", dir_okay=False),
):
    """Compare spot-mode spaces from analyses of the same observations."""
    a = load_result(_read_h5ad(input_a), analysis_a, variant=variant_a)
    b = load_result(_read_h5ad(input_b), analysis_b, variant=variant_b)
    if not isinstance(a, ProgramResult) or not isinstance(b, ProgramResult):
        raise typer.BadParameter("compare-spots requires two ordinary program results")
    frame = compare_spot_mode_results(a, b)
    typer.echo(frame.to_string(index=False))
    if output is not None:
        frame.to_csv(output, index=False)


@app.command("plot", no_args_is_help=True)
def plot_result(
    fname: InputPath,
    analysis_id: str | None = typer.Option(None, "--analysis-id", "-id"),
    variant: str | None = typer.Option(None, "--variant"),
    label_key: str | None = typer.Option(None, "--label-key"),
    modes: str | None = typer.Option(None, "--modes", help="Comma-separated zero-based modes"),
    side: str = typer.Option("target", "--side", help="target or neighbour for cross results"),
    output: Path | None = typer.Option(None, "--output", "-o", dir_okay=False),
    show: bool = typer.Option(True, "--show/--no-show"),
):
    """Plot stored spatial modes or categorical labels."""
    from tanpopo.plot import plot_labels, plot_spatial_modes

    adata = _read_h5ad(fname)
    if label_key is not None:
        fig = plot_labels(adata, label_key)
    elif analysis_id is not None:
        result = load_result(adata, analysis_id, variant=variant)
        indices = _parse_mode_indices(modes, result.n_modes)
        if isinstance(result, ProgramResult):
            if result.spot_modes is None:
                raise typer.BadParameter("Stored result does not contain spot modes")
            fig = plot_spatial_modes(adata, result.spot_modes, indices=indices)
        else:
            if side not in {"target", "neighbour"}:
                raise typer.BadParameter("--side must be target or neighbour")
            values = result.target_modes if side == "target" else result.neighbour_modes
            if values is None:
                raise typer.BadParameter("Stored cross result does not contain spot modes")
            fig = plot_spatial_modes(adata, values, indices=indices)
    else:
        raise typer.BadParameter("Supply --analysis-id or --label-key")
    if output is not None:
        fig.savefig(output, bbox_inches="tight", dpi=200)
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    return fig


@app.command("cluster", no_args_is_help=True)
def cluster_result(
    fname: InputPath,
    analysis_id: str = typer.Option("spatial", "--analysis-id", "-id"),
    variant: str | None = typer.Option(None, "--variant"),
    by: str = typer.Option("spots", "--by", help="spots or genes"),
    output: OutputPath = None,
    key_added: str | None = typer.Option(None, "--key-added"),
    neighbours: int = typer.Option(15, "--neighbours", min=2),
    resolution: float = typer.Option(1.0, "--resolution", min=0.0),
    seed: RandomSeed = 0,
):
    """Leiden-cluster spots or genes in a stored Tanpopo program space."""
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError("Clustering requires scanpy plus the 'cluster' extra") from exc
    adata = _read_h5ad(fname)
    result = load_result(adata, analysis_id, variant=variant)
    if not isinstance(result, ProgramResult):
        raise typer.BadParameter("cluster currently supports ordinary program results")
    if by == "spots":
        values = result.spot_modes
        if values is None:
            raise typer.BadParameter("Stored result does not contain spot modes")
        valid = np.all(np.isfinite(values), axis=1)
        work = sc.AnnData(np.asarray(values[valid]))
        if work.n_obs < 3:
            raise typer.BadParameter("At least three valid observations are required to cluster")
        sc.pp.neighbors(work, n_neighbors=min(neighbours, work.n_obs - 1), use_rep="X")
        sc.tl.leiden(
            work,
            resolution=resolution,
            random_state=seed,
            key_added="cluster",
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )
        key_added = key_added or f"tanpopo_{analysis_id}_cluster"
        labels = np.full(adata.n_obs, None, dtype=object)
        labels[valid] = work.obs["cluster"].astype(str).to_numpy()
        adata.obs[key_added] = labels
    elif by == "genes":
        work = sc.AnnData(np.asarray(result.gene_loadings))
        if work.n_obs < 3:
            raise typer.BadParameter("At least three genes are required to cluster")
        sc.pp.neighbors(work, n_neighbors=min(neighbours, work.n_obs - 1), use_rep="X")
        sc.tl.leiden(
            work,
            resolution=resolution,
            random_state=seed,
            key_added="cluster",
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )
        key_added = key_added or f"tanpopo_{analysis_id}_gene_cluster"
        adata.var[key_added] = work.obs["cluster"].astype(str).to_numpy()
    else:
        raise typer.BadParameter("--by must be spots or genes")
    if output is not None:
        adata.write_h5ad(output)
    return adata


@app.command("gsea", no_args_is_help=True)
def gsea_result(
    fname: InputPath,
    analysis_id: str = typer.Option("spatial", "--analysis-id", "-id"),
    variant: str | None = typer.Option(None, "--variant"),
    side: str | None = typer.Option(None, "--side"),
    modes: str | None = typer.Option(None, "--modes"),
    gene_sets: str = typer.Option("MSigDB_Hallmark_2020", "--gene-sets"),
    permutations: int = typer.Option(1000, "--permutations", min=1),
    output: Path | None = typer.Option(None, "--output", "-o", dir_okay=False),
    seed: RandomSeed = 0,
):
    """Preranked gene-set enrichment using stored program loadings."""
    try:
        import gseapy as gp
    except ImportError as exc:
        raise ImportError("GSEA requires the optional 'gsea' dependencies") from exc
    adata = _read_h5ad(fname)
    result = load_result(adata, analysis_id, variant=variant)
    if isinstance(result, ProgramResult):
        loadings = result.gene_loadings
    else:
        if side not in {"target", "neighbour"}:
            raise typer.BadParameter("Cross results require --side target or neighbour")
        loadings = result.target_loadings if side == "target" else result.neighbour_loadings
    indices = _parse_mode_indices(modes, loadings.shape[1])
    frames = []
    for mode in indices:
        import pandas as pd

        rank = pd.Series(loadings[:, mode], index=result.gene_names).sort_values(ascending=False)
        pre = gp.prerank(
            rnk=rank,
            gene_sets=gene_sets,
            permutation_num=permutations,
            outdir=None,
            seed=seed,
            verbose=False,
        )
        frame = pre.res2d.copy()
        frame.insert(0, "mode", mode)
        frames.append(frame)
    import pandas as pd

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    typer.echo(combined.to_string(index=False))
    if output is not None:
        combined.to_csv(output, index=False)
    return combined


@app.command("spatial-info", no_args_is_help=True)
def spatial_info(
    fname: InputPath,
    radius: Radius = None,
    radii: str | None = typer.Option(None, "--radii", help="Comma-separated radius curve"),
    layer: Layer = None,
    label_key: LabelKey = None,
    spot_operator: SpotOperator = SpotOperatorTypes.sample,
    expression_rank: ExpressionRank = 100,
    max_components: MaxComponents = 30,
    mode_alpha: ModeAlpha = 0.05,
    graph_normalisation: GraphNormalisation = GraphNormalisationTypes.symmetric,
    geometry_normalisation: GeometryNormalisation = GeometryNormalisationTypes.distance,
    distance_bins: DistanceBins = 10,
    include: Include = None,
    exclude: Exclude = None,
    transform: Transform = None,
    min_counts: MinCounts = 10,
    min_spot_fraction: MinSpotFraction = None,
    target_sum: TargetSum = None,
    covariates: Covariates = None,
    permutations: Permutations = 200,
    seed: RandomSeed = 0,
    dtype: Dtype = Dtypes.float64,
    output: Path | None = typer.Option(None, "--output", "-o", dir_okay=False),
):
    """Quantify gain, effective dimension and supported spatial mode count."""
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
    adata = load_preprocess_sample(fname, **pre)
    W, coords, cov = get_spatial_from_anndata(adata, layer)
    labels = None
    if as_value(spot_operator) == "label":
        _require_obs_key(adata, label_key)
        labels = np.asarray(adata.obs[label_key])
    kwargs = dict(
        labels=labels,
        covariates=cov,
        max_components=max_components,
        expression_rank=expression_rank,
        graph_normalisation=as_value(graph_normalisation),
        geometry_normalisation=as_value(geometry_normalisation),
        distance_bins=distance_bins,
        spot_operator=as_value(spot_operator),
        permutations=permutations,
        alpha=mode_alpha,
        seed=seed,
        dtype=np.dtype(as_value(dtype)),
    )
    if radii:
        values = [float(value.strip()) for value in radii.split(",") if value.strip()]
        curve = spatial_information_curve_from_arrays(W, coords, values, **kwargs)
        frame = curve.summary_frame()
        typer.echo(frame.to_string(index=False))
        if output is not None:
            frame.to_csv(output, index=False)
        return curve
    result = spatial_information_from_arrays(W, coords, radius=radius, **kwargs)
    for key, value in result.summary().items():
        typer.echo(f"{key}: {value}")
    if output is not None:
        import pandas as pd

        pd.DataFrame([result.summary()]).to_csv(output, index=False)
    return result


def main():
    app()


if __name__ == "__main__":
    main()
