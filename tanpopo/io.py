import typer
import numpy as np
import scanpy as sc

from tanpopo.data import preprocess_anndata, preprocess_anndata_shared_genes
from tanpopo.plot import plot_spatial_modes
from tanpopo.utils import timed, as_value


def name_samples(fnames, sample_names):
    if sample_names is None:
        sample_names = [fname.stem for fname in fnames]
    if len(sample_names) != len(fnames):
        raise typer.BadParameter(f"Supply exactly one --sample-name for each --input.")
    if len(sample_names) != len(set(sample_names)):
        raise typer.BadParameter("Sample names must be unique.")
    return sample_names


def load_preprocess_sample(
    fname,
    target_sum,
    transform,
    min_counts,
    min_spot_fraction,
    covariates,
    exclude=None,
    label_key=None,
    layer=None,
    verbose=False,
):
    with timed("Loading data", verbose):
        adata = sc.read_h5ad(fname)
        preprocess_anndata(
            adata,
            target_sum,
            transform,
            min_counts,
            min_spot_fraction,
            covariates,
            exclude,
            label_key,
            layer,
        )
    if verbose:
        print(adata)
    return adata


def load_preprocess_samples(
    fnames,
    sample_names,
    target_sum,
    transform,
    min_counts,
    min_spot_fraction,
    covariates,
    exclude=None,
    label_key=None,
    layer=None,
    verbose=False,
):
    with timed("Loading data", verbose):
        adata_samples = [sc.read_h5ad(fname) for fname in fnames]
        preprocess_anndata_shared_genes(
            adata_samples,
            target_sum,
            transform,
            min_counts,
            min_spot_fraction,
            covariates,
            exclude,
            label_key,
            layer,
        )
        sample_names = name_samples(fnames, sample_names)
    if verbose:
        for adata, name in zip(adata_samples, sample_names):
            print(name)
            print(adata)
    return adata_samples, sample_names


def preprocess_cfg(
    target_sum, transform, min_counts, min_spot_fraction, covariates, label_key, layer
):
    return {
        "target_sum": target_sum,
        "transform": as_value(transform),
        "min_counts": min_counts,
        "min_spot_fraction": min_spot_fraction,
        "covariates": covariates,
        "label_key": label_key,
        "layer": layer,
    }


def model_cfg(
    radius,
    alpha,
    gene_center,
    spot_operator=None,
    sample_weighting=None,
    normalise_by=None,
):
    cfg = {
        "kernel": "wendland_c2",
        "radius": radius,
        "alpha": alpha,
        "gene_center": gene_center,
    }
    if spot_operator is not None:
        cfg["spot_operator"] = as_value(spot_operator)
    if sample_weighting is not None:
        cfg["sample_weighting"] = as_value(sample_weighting)
    if normalise_by is not None:
        cfg["normalise_by"] = as_value(normalise_by)

    return cfg


def add_metadata(adata, cmd_id, preprocessing, model, extra=None):
    adata.uns.setdefault("tanpopo", {}).setdefault(cmd_id, {})
    adata.uns["tanpopo"][cmd_id].update({"preprocessing": preprocessing, "model": model})
    if extra:
        adata.uns["tanpopo"][cmd_id].update(extra)


def full_mode(spot_mode, mask, total_spots):
    if isinstance(mask, np.ndarray):
        full_mode = np.full((total_spots, spot_mode.shape[1]), np.nan)
        full_mode[mask] = spot_mode
        return full_mode
    return spot_mode


def concat_adata_samples(adatas, sample_names):
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


def store_sample_result(adata, model, cmd_id, key="", mask=None, gene_loadings=None):
    prefix = f"tanpopo_{cmd_id}{key}"
    adata.obsm[f"{prefix}_spot_modes"] = full_mode(model.spot_modes[0], mask, adata.n_obs)
    adata.varm[f"{prefix}_eigenvectors"] = model.eigenvectors
    adata.varm[f"{prefix}_gene_loadings"] = (
        model.gene_loadings if gene_loadings is None else gene_loadings
    )
    adata.varm[f"{prefix}_gene_scores"] = model.gene_scores
    adata.var[f"{prefix}_gene_scores"] = model.gene_spatial_scores()
    adata.uns["tanpopo"][cmd_id][f"eigenvalues{key}"] = model.eigenvalues


def store_multi_sample_result(adata_samples, sample_names, model, cmd_id, plot=False):
    for adata, name, spot_mode, gene_loadings in zip(
        adata_samples, sample_names, model.spot_modes, model.gene_loadings
    ):
        adata.obsm[f"tanpopo_{cmd_id}_spot_modes"] = spot_mode
        adata.varm[f"tanpopo_{cmd_id}_{name}_gene_loadings"] = gene_loadings

        if plot:
            plot_spatial_modes(adata, spot_mode, cmap="coolwarm", vcenter=0)

    combined = concat_adata_samples(adata_samples, sample_names)
    combined.uns.setdefault("tanpopo", {}).setdefault(cmd_id, {})
    combined.varm[f"tanpopo_{cmd_id}_eigenvectors"] = model.eigenvectors
    combined.varm[f"tanpopo_{cmd_id}_gene_scores"] = model.gene_scores
    combined.var[f"tanpopo_{cmd_id}_gene_scores"] = model.gene_spatial_scores()
    combined.uns["tanpopo"][cmd_id]["eigenvalues"] = model.eigenvalues

    return combined
