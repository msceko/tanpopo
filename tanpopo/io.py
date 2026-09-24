from __future__ import annotations

import json

import numpy as np

from tanpopo.data import preprocess_anndata, preprocess_anndata_shared_genes
from tanpopo.utils import as_value, timed


def _scanpy():
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError("AnnData workflows require scanpy and anndata") from exc
    return sc


def name_samples(fnames, sample_names):
    if sample_names is None:
        sample_names = [fname.stem for fname in fnames]
    if len(sample_names) != len(fnames):
        raise ValueError("Supply exactly one --name for each input")
    if len(set(sample_names)) != len(sample_names):
        raise ValueError("Sample names must be unique")
    return sample_names


def load_preprocess_sample(fname, verbose=False, **kwargs):
    sc = _scanpy()
    with timed("Loading data", verbose):
        adata = sc.read_h5ad(fname)
        preprocess_anndata(adata, **kwargs)
    return adata


def load_preprocess_samples(fnames, sample_names=None, verbose=False, **kwargs):
    sc = _scanpy()
    with timed("Loading data", verbose):
        adatas = [sc.read_h5ad(fname) for fname in fnames]
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
        "spatial_statistic",
        "geometry_normalisation",
        "distance_bins",
        "null_center",
    )
    return {key: getattr(model, key) for key in keys if hasattr(model, key)}


def add_metadata(adata, cmd_id, preprocessing, model, extra=None):
    adata.uns.setdefault("tanpopo", {}).setdefault(cmd_id, {})
    metadata = {
        "preprocessing": preprocessing,
        "model": model,
        "tanpopo_scope": "conditional_spatial_pair_statistics",
    }
    if extra:
        metadata.update(extra)
    adata.uns["tanpopo"][cmd_id].update(metadata)


def full_mode(mode, obs_idx, n_obs):
    out = np.full((n_obs, mode.shape[1]), np.nan, dtype=float)
    out[np.asarray(obs_idx, dtype=int)] = mode
    return out


def _store_gene_spatial_statistic(adata, prefix, model):
    values = model.gene_spatial_scores()
    adata.var[f"{prefix}_gene_spatial_statistic"] = values
    # Retain the historical covariance key for mark correlation and provide an
    # explicit variogram key for the alternative pair statistic.
    if getattr(model, "spatial_statistic", "mark_correlation") == "mark_correlation":
        adata.var[f"{prefix}_gene_spatial_covariance"] = values
    else:
        adata.var[f"{prefix}_gene_mark_variogram"] = values


def geometry_diagnostics_cfg(model, sample_names=None):
    """Return H5AD-friendly per-sample geometry diagnostics."""
    diagnostics = getattr(model, "geometry_diagnostics_", None)
    if diagnostics is None:
        return None
    if sample_names is None:
        sample_names = [f"sample_{i}" for i in range(len(diagnostics))]
    return {str(name): diag for name, diag in zip(sample_names, diagnostics)}


def store_sample_result(adata, model, cmd_id, key="", sample_index=0):
    prefix = f"tanpopo_{cmd_id}{key}"
    sample = model.samples[sample_index]
    adata.obsm[f"{prefix}_spot_modes"] = full_mode(
        model.spot_modes[sample_index], sample.obs_idx, adata.n_obs
    )
    adata.varm[f"{prefix}_gene_loadings"] = model.gene_loadings
    adata.varm[f"{prefix}_gene_scores"] = model.gene_scores
    # Keep historical eigenvector key where dimensions permit; gene_loadings is the
    # scientifically interpretable key for all objectives.
    if model.eigenvectors.shape[0] == adata.n_vars:
        adata.varm[f"{prefix}_eigenvectors"] = model.eigenvectors
    _store_gene_spatial_statistic(adata, prefix, model)
    adata.uns.setdefault("tanpopo", {}).setdefault(cmd_id, {})
    adata.uns["tanpopo"][cmd_id][f"eigenvalues{key}"] = np.asarray(model.eigenvalues)


def concat_adata_samples(adatas, sample_names):
    sc = _scanpy()
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


def store_multi_sample_result(adatas, sample_names, model, cmd_id, key=""):
    prefix = f"tanpopo_{cmd_id}{key}"
    for i, adata in enumerate(adatas):
        adata.obsm[f"{prefix}_spot_modes"] = full_mode(
            model.spot_modes[i], model.samples[i].obs_idx, adata.n_obs
        )
    combined = concat_adata_samples(adatas, sample_names)
    combined.varm[f"{prefix}_gene_loadings"] = model.gene_loadings
    combined.varm[f"{prefix}_gene_scores"] = model.gene_scores
    _store_gene_spatial_statistic(combined, prefix, model)
    combined.uns.setdefault("tanpopo", {}).setdefault(cmd_id, {})
    combined.uns["tanpopo"][cmd_id][f"eigenvalues{key}"] = np.asarray(model.eigenvalues)
    combined.uns["tanpopo"][cmd_id][f"sample_coefficients{key}"] = np.asarray(
        model.sample_coefficients_
    )
    return combined


def store_multi_sample_cross_result(adatas, sample_names, model, cmd_id):
    """Store shared or differential cross-program results across samples."""
    prefix = f"tanpopo_{cmd_id}"
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
    result = combined.uns.setdefault("tanpopo", {}).setdefault(cmd_id, {})
    result["singular_values"] = np.asarray(model.singular_values)
    result["sample_coefficients"] = np.asarray(model.sample_coefficients_)
    if hasattr(model, "base_sample_weights_"):
        result["base_sample_weights"] = np.asarray(model.base_sample_weights_)
    if hasattr(model, "contrast_coefficients_"):
        result["contrast_coefficients"] = np.asarray(model.contrast_coefficients_)
    result["sample_mode_covariance"] = np.asarray(model.sample_mode_covariance_)
    if hasattr(model, "sample_mode_statistic_"):
        result["sample_mode_statistic"] = np.asarray(model.sample_mode_statistic_)
    if hasattr(model, "aggregate_mode_statistic_"):
        result["aggregate_mode_statistic"] = np.asarray(model.aggregate_mode_statistic_)
    if hasattr(model, "contrast_mode_statistic_"):
        result["contrast_mode_statistic"] = np.asarray(model.contrast_mode_statistic_)
    if hasattr(model, "group_a_mode_statistic_"):
        result["group_a_mode_statistic"] = np.asarray(model.group_a_mode_statistic_)
    if hasattr(model, "group_b_mode_statistic_"):
        result["group_b_mode_statistic"] = np.asarray(model.group_b_mode_statistic_)
    return combined


# Backward-compatible internal alias.
store_shared_cross_result = store_multi_sample_cross_result


def store_cross_result(adata, model, cmd_id):
    prefix = f"tanpopo_{cmd_id}"
    target = np.full((adata.n_obs, model.target_modes.shape[1]), np.nan)
    neighbour = np.full((adata.n_obs, model.neighbour_modes.shape[1]), np.nan)
    target[model.target_idx] = model.target_modes
    neighbour[model.neighbour_idx] = model.neighbour_modes
    adata.obsm[f"{prefix}_target_modes"] = target
    adata.obsm[f"{prefix}_neighbour_modes"] = neighbour
    adata.varm[f"{prefix}_target_loadings"] = model.target_loadings
    adata.varm[f"{prefix}_neighbour_loadings"] = model.neighbour_loadings
    adata.uns.setdefault("tanpopo", {}).setdefault(cmd_id, {})
    adata.uns["tanpopo"][cmd_id]["singular_values"] = np.asarray(model.singular_values)
