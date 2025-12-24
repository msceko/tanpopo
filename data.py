import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from matplotlib.image import imread


def extract_visium_data(
    adata, normalise=True, transform=None, layer=None, min_counts=10, sparse=False
):
    """
    Returns:
        X : (n_spots, n_genes) expression matrix
        coords : (n_spots, 2) spatial coordinates
        gene_names
    """
    if normalise:
        sc.pp.normalize_total(adata, target_sum=1e4)
    if transform == "log1p":
        sc.pp.log1p(adata)
    elif transform == "sqrt":
        sc.pp.sqrt(adata)

    if layer is None:
        X = adata.X
    else:
        X = adata.layers[layer]

    if sparse and not sp.issparse(X):
        X = sp.csr_matrix(X)
    elif not sparse and sp.issparse(X):
        X = X.toarray()

    coords = adata.obsm["spatial"].astype(np.float64)
    gene_names = np.array(adata.var_names)

    # Filter low-count genes
    gene_mask = np.array(X.sum(axis=0)).ravel() >= min_counts
    # gene_mask = (adata.X.toarray() > 0).mean(axis=0) >= 0.05
    X = X[:, gene_mask]
    gene_names = gene_names[gene_mask]

    return X, coords, gene_names


def load_visium_hd(
    bin_path,
    sample_id=None,
    load_images=True,
):
    """
    Load 10x Visium HD binned output into an AnnData object.

    Parameters
    ----------
    bin_path : str or Path
        Path to a Visium HD bin directory (e.g. square_016um).
    sample_id : str, optional
        Sample name to store under adata.uns['spatial'].
        Defaults to bin directory name.
    load_images : bool
        Whether to load tissue images and scalefactors.

    Returns
    -------
    adata : AnnData
        AnnData object with spatial coordinates attached.
    """

    bin_path = Path(bin_path)
    spatial_path = bin_path / "spatial"

    if sample_id is None:
        sample_id = bin_path.name

    # -----------------------------
    # 1. Load expression matrix
    # -----------------------------
    h5_path = bin_path / "filtered_feature_bc_matrix.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing {h5_path}")

    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()

    # -----------------------------
    # 2. Load spatial coordinates
    # -----------------------------
    pos_path = spatial_path / "tissue_positions.parquet"
    if not pos_path.exists():
        raise FileNotFoundError(f"Missing {pos_path}")

    pos = pd.read_parquet(pos_path)

    pos = pos.set_index("barcode").loc[adata.obs_names]

    adata.obsm["spatial"] = pos[["pxl_row_in_fullres", "pxl_col_in_fullres"]].to_numpy()

    adata.obs["in_tissue"] = pos["in_tissue"].values
    adata.obs["array_row"] = pos["array_row"].values
    adata.obs["array_col"] = pos["array_col"].values

    # -----------------------------
    # 3. Load images + scalefactors
    # -----------------------------
    if load_images:
        sf_path = spatial_path / "scalefactors_json.json"
        if sf_path.exists():
            with open(sf_path) as f:
                scalefactors = json.load(f)
        else:
            scalefactors = {}

        images = {}
        hires = spatial_path / "tissue_hires_image.png"
        lowres = spatial_path / "tissue_lowres_image.png"

        if hires.exists():
            images["hires"] = imread(hires)
        if lowres.exists():
            images["lowres"] = imread(lowres)

        adata.uns["spatial"] = {
            sample_id: {
                "scalefactors": scalefactors,
                "images": images,
            }
        }

    return adata
