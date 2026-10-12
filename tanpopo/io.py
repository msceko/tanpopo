from __future__ import annotations

import numpy as np

from tanpopo.data import preprocess_anndata, preprocess_anndata_shared_genes
from tanpopo.results import register_cross_result, register_program_result
from tanpopo.utils import as_value, timed


def _anndata():
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("AnnData workflows require anndata") from exc
    return ad


def name_samples(fnames, sample_names):
    if sample_names is None:
        sample_names = [fname.stem for fname in fnames]
    if len(sample_names) != len(fnames):
        raise ValueError("Supply exactly one --name for each input")
    if len(set(sample_names)) != len(sample_names):
        raise ValueError("Sample names must be unique")
    return sample_names


def load_preprocess_sample(fname, verbose=False, **kwargs):
    ad = _anndata()
    with timed("Loading data", verbose):
        adata = ad.read_h5ad(fname)
        preprocess_anndata(adata, **kwargs)
    return adata


def load_preprocess_samples(fnames, sample_names=None, verbose=False, **kwargs):
    ad = _anndata()
    with timed("Loading data", verbose):
        adatas = [ad.read_h5ad(fname) for fname in fnames]
        preprocess_anndata_shared_genes(adatas, **kwargs)
        sample_names = name_samples(fnames, sample_names)
    return adatas, sample_names


def preprocess_cfg(
    target_sum,
    transform,
    min_counts,
    min_spot_fraction,
    covariates,
    label_key,
    layer,
    include=None,
    exclude=None,
):
    cov = None if covariates is None else [x.strip() for x in covariates.split(",") if x.strip()]
    return {
        "target_sum": target_sum,
        "transform": as_value(transform),
        "min_counts": min_counts,
        "min_spot_fraction": min_spot_fraction,
        "covariates": cov,
        "include": include,
        "exclude": exclude,
        "label_key": label_key,
        "layer": layer,
    }


def model_cfg(model):
    keys = (
        "radius",
        "objective",
        "spot_operator",
        "sample_weighting",
        "expression_rank",
        "gain_ridge",
        "graph_normalisation",
        "geometry_normalisation",
        "distance_bins",
    )
    return {key: getattr(model, key) for key in keys if hasattr(model, key)}


def add_metadata(
    adata,
    analysis_id,
    preprocessing,
    model,
    extra=None,
    *,
    variant="default",
    analysis_type=None,
):
    """Attach stable metadata to a stored analysis result."""
    record = adata.uns.setdefault("tanpopo", {}).setdefault(str(analysis_id), {})
    if analysis_type is not None:
        record["analysis_type"] = str(analysis_type)
    record["preprocessing"] = preprocessing
    record["scope"] = "geometry_standardised_spatial_covariance"
    variant_metadata = record.setdefault("result_metadata", {}).setdefault(str(variant), {})
    variant_metadata["model"] = model
    if extra:
        variant_metadata.update(extra)
    return record


def full_mode(mode, obs_idx, n_obs):
    out = np.full((n_obs, mode.shape[1]), np.nan, dtype=float)
    out[np.asarray(obs_idx, dtype=int)] = mode
    return out


def _store_gene_spatial_covariance(adata, prefix, model):
    adata.var[f"{prefix}_gene_spatial_covariance"] = model.gene_spatial_scores()


def geometry_diagnostics_cfg(model, sample_names=None):
    """Return H5AD-friendly per-sample geometry diagnostics."""
    diagnostics = getattr(model, "geometry_diagnostics_", None)
    if diagnostics is None:
        return None
    if sample_names is None:
        sample_names = [f"sample_{i}" for i in range(len(diagnostics))]
    return {str(name): diag for name, diag in zip(sample_names, diagnostics)}


def store_sample_result(adata, model, analysis_id, key="", sample_index=0):
    prefix = f"tanpopo_{analysis_id}{key}"
    variant = "default" if not key else key.lstrip("_")
    sample = model.samples[sample_index]
    modes_key = f"{prefix}_spot_modes"
    loadings_key = f"{prefix}_gene_loadings"
    scores_key = f"{prefix}_gene_scores"
    covariance_key = f"{prefix}_gene_spatial_covariance"
    adata.obsm[modes_key] = full_mode(model.spot_modes[sample_index], sample.obs_idx, adata.n_obs)
    adata.varm[loadings_key] = model.gene_loadings
    adata.varm[scores_key] = model.gene_scores
    _store_gene_spatial_covariance(adata, prefix, model)
    register_program_result(
        adata,
        analysis_id,
        loadings_key=loadings_key,
        modes_key=modes_key,
        gene_scores_key=scores_key,
        gene_covariance_key=covariance_key,
        values=model.eigenvalues,
        variant=variant,
        analysis_type="program",
    )


def concat_adata_samples(adatas, sample_names):
    ad = _anndata()
    if len(adatas) == 1:
        return adatas[0].copy()
    return ad.concat(
        adatas,
        label="sample",
        keys=[str(x) for x in sample_names],
        index_unique="-",
        join="inner",
        merge="unique",
    )


def store_multi_sample_result(adatas, sample_names, model, analysis_id, key=""):
    prefix = f"tanpopo_{analysis_id}{key}"
    variant = "default" if not key else key.lstrip("_")
    modes_key = f"{prefix}_spot_modes"
    loadings_key = f"{prefix}_gene_loadings"
    scores_key = f"{prefix}_gene_scores"
    covariance_key = f"{prefix}_gene_spatial_covariance"
    for i, adata in enumerate(adatas):
        adata.obsm[modes_key] = full_mode(
            model.spot_modes[i], model.samples[i].obs_idx, adata.n_obs
        )
    combined = concat_adata_samples(adatas, sample_names)
    combined.varm[loadings_key] = model.gene_loadings
    combined.varm[scores_key] = model.gene_scores
    _store_gene_spatial_covariance(combined, prefix, model)
    record = register_program_result(
        combined,
        analysis_id,
        loadings_key=loadings_key,
        modes_key=modes_key,
        gene_scores_key=scores_key,
        gene_covariance_key=covariance_key,
        values=model.eigenvalues,
        variant=variant,
        analysis_type="program",
    )
    record.setdefault("result_metadata", {}).setdefault(variant, {})["sample_coefficients"] = (
        np.asarray(model.sample_coefficients_)
    )
    return combined


def store_multi_sample_cross_result(adatas, sample_names, model, analysis_id):
    """Store shared or differential cross-program results across samples."""
    prefix = f"tanpopo_{analysis_id}"
    for adata, sample, target_modes, neighbour_modes in zip(
        adatas, model.samples, model.target_modes, model.neighbour_modes
    ):
        target = np.full((adata.n_obs, target_modes.shape[1]), np.nan)
        neighbour = np.full((adata.n_obs, neighbour_modes.shape[1]), np.nan)
        target[sample.target_idx] = target_modes
        neighbour[sample.neighbour_idx] = neighbour_modes
        adata.obsm[f"{prefix}_target_modes"] = target
        adata.obsm[f"{prefix}_neighbour_modes"] = neighbour

    combined = concat_adata_samples(adatas, sample_names)
    combined.varm[f"{prefix}_target_loadings"] = model.target_loadings
    combined.varm[f"{prefix}_neighbour_loadings"] = model.neighbour_loadings
    record = register_cross_result(
        combined,
        analysis_id=analysis_id,
        target_loadings_key=f"{prefix}_target_loadings",
        neighbour_loadings_key=f"{prefix}_neighbour_loadings",
        target_modes_key=f"{prefix}_target_modes",
        neighbour_modes_key=f"{prefix}_neighbour_modes",
        singular_values=model.singular_values,
        analysis_type="cross",
    )
    metadata = record.setdefault("result_metadata", {}).setdefault("default", {})
    metadata["sample_coefficients"] = np.asarray(model.sample_coefficients_)
    if hasattr(model, "base_sample_weights_"):
        metadata["base_sample_weights"] = np.asarray(model.base_sample_weights_)
    if hasattr(model, "contrast_coefficients_"):
        metadata["contrast_coefficients"] = np.asarray(model.contrast_coefficients_)
    metadata["sample_mode_covariance"] = np.asarray(model.sample_mode_covariance_)
    for attr, key in (
        ("sample_mode_statistic_", "sample_mode_statistic"),
        ("aggregate_mode_statistic_", "aggregate_mode_statistic"),
        ("contrast_mode_statistic_", "contrast_mode_statistic"),
        ("group_a_mode_statistic_", "group_a_mode_statistic"),
        ("group_b_mode_statistic_", "group_b_mode_statistic"),
    ):
        if hasattr(model, attr):
            metadata[key] = np.asarray(getattr(model, attr))
    return combined


def store_cross_result(adata, model, analysis_id):
    prefix = f"tanpopo_{analysis_id}"
    target = np.full((adata.n_obs, model.target_modes.shape[1]), np.nan)
    neighbour = np.full((adata.n_obs, model.neighbour_modes.shape[1]), np.nan)
    target[model.target_idx] = model.target_modes
    neighbour[model.neighbour_idx] = model.neighbour_modes
    adata.obsm[f"{prefix}_target_modes"] = target
    adata.obsm[f"{prefix}_neighbour_modes"] = neighbour
    adata.varm[f"{prefix}_target_loadings"] = model.target_loadings
    adata.varm[f"{prefix}_neighbour_loadings"] = model.neighbour_loadings
    register_cross_result(
        adata,
        analysis_id=analysis_id,
        target_loadings_key=f"{prefix}_target_loadings",
        neighbour_loadings_key=f"{prefix}_neighbour_loadings",
        target_modes_key=f"{prefix}_target_modes",
        neighbour_modes_key=f"{prefix}_neighbour_modes",
        singular_values=model.singular_values,
        analysis_type="cross",
    )
