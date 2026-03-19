import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from matplotlib.image import imread


def extract_visium_data(
    adata,
    target_sum=1e4,
    transform=None,
    min_counts=10,
    min_spot_fraction=0.01,
    layer=None,
    sparse=True,
):
    """
    Returns:
        X : (n_spots, n_genes) expression matrix
        coords : (n_spots, 2) spatial coordinates
        gene_names
    """
    if min_counts:
        sc.pp.filter_genes(adata, min_counts=min_counts)
    if min_spot_fraction:
        min_spots = int(min_spot_fraction * len(adata.obs))
        sc.pp.filter_genes(adata, min_cells=min_spots)

    if target_sum:
        sc.pp.normalize_total(adata, target_sum=target_sum)
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


def xenium_adata(data, meta, genes, x, y, library_id: str = "library_id"):
    """Construct visium-like anndata with xenium data"""
    # Spot/bin IDs (Visium uses barcodes; here we create stable names)
    obs_names = pd.Index(
        [f"bin_y{yb}_x{xb}" for yb, xb in zip(meta["y_bin"], meta["x_bin"])], name="spot_id"
    )
    var_names = pd.Index(genes.astype(str), name="gene_ids")

    obs = pd.DataFrame(meta, index=obs_names)
    var = pd.DataFrame(index=var_names)
    adata = ad.AnnData(X=data, obs=obs, var=var)

    # Visium-like spatial coordinates: (x, y)
    adata.obsm["spatial"] = np.column_stack([x, y]).astype(np.float32)

    # Minimal Visium-ish uns['spatial'] stub (real Visium includes images/scalefactors)
    adata.uns["spatial"] = {
        library_id: {
            "images": {},  # add "hires"/"lowres" arrays if you have them
            "scalefactors": {
                "tissue_hires_scalef": 1.0,
                "tissue_lowres_scalef": 1.0,
                "spot_diameter_fullres": meta["bin_size"],
            },  # add scalefactors if you want Visium tooling compatibility
            "metadata": {
                "source": "binned_transcripts",
                "bin_size": meta["bin_size"],
            },
        }
    }

    return adata


def load_xenium_binned(
    fname: str,
    bin_size: float,
    x_col: str = "x_location",
    y_col: str = "y_location",
    gene_col: str = "feature_name",
    qv_col: str = "qv",
    library_id: str = "library_id",
    remove=["Control", "Codeword", "BLANK"],
    dtype=np.float32,
):
    """
    Sparse gene-by-grid matrix using from Xenium transcripts.parquet.

    Returns:
      X: scipy.sparse.csr_matrix, shape (M_occ, N_genes)
      meta: dict with mappings to interpret rows/cols
    """
    df = pd.read_parquet(fname, columns=[gene_col, x_col, y_col, qv_col])
    df[gene_col] = df[gene_col].astype("str").astype("category")
    df = df[~df[gene_col].str.contains("|".join(remove))]
    df[gene_col] = df[gene_col].cat.remove_unused_categories()

    x, y = df[x_col].to_numpy(), df[y_col].to_numpy()
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    n_bins_x = int(np.ceil((x_max - x_min) / bin_size))
    n_bins_y = int(np.ceil((y_max - y_min) / bin_size))
    x_edges = x_min + bin_size * np.arange(n_bins_x + 1)
    y_edges = y_min + bin_size * np.arange(n_bins_y + 1)

    # Bin indices
    xb = np.searchsorted(x_edges, x, side="right") - 1
    yb = np.searchsorted(y_edges, y, side="right") - 1
    xb = np.clip(xb, 0, n_bins_x - 1)
    yb = np.clip(yb, 0, n_bins_y - 1)

    # Flatten full-grid row id
    full_row_id = yb * n_bins_x + xb

    # Reindex to occupied-only row ids: 0..M_occ-1
    occ_row_codes, occ_row_uniques = pd.factorize(full_row_id, sort=True)

    # Encode genes to columns
    gene_codes, genes = pd.factorize(df[gene_col], sort=True)

    qv_vals = df[qv_col].to_numpy()
    weights = 1 - 10 ** (-qv_vals / 10)

    # Aggregate duplicates (occupied_row, gene)
    agg = (
        pd.DataFrame({"row": occ_row_codes, "col": gene_codes, "val": weights})
        .groupby(["row", "col"], sort=False, as_index=False)["val"]
        .sum()
    )

    X = sp.coo_matrix(
        (agg["val"].to_numpy(), (agg["row"].to_numpy(), agg["col"].to_numpy())),
        shape=(len(occ_row_uniques), len(genes)),
        dtype=dtype,
    ).tocsr()

    y_bin = (occ_row_uniques // n_bins_x).astype(np.int32)
    x_bin = (occ_row_uniques % n_bins_x).astype(np.int32)
    x_center = (x_edges[x_bin] + x_edges[x_bin + 1]) / 2.0
    y_center = (y_edges[y_bin] + y_edges[y_bin + 1]) / 2.0

    meta = {
        "bin_size": bin_size,
        "x_bin": x_bin,
        "y_bin": y_bin,
        "x_center": x_center,
        "y_center": y_center,
        "full_row_id": occ_row_uniques,
    }

    return xenium_adata(X, meta, genes, x_center, y_center, library_id)
