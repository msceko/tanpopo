import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from matplotlib.image import imread

from kernel import kernel_matrix_sparse
from utils import as_list


def get_counts_matrix(adata, sparse=True, layer=None):
    """
    Return the counts matrix from adata or adata.layers[layer].
    Always returned as CSR sparse matrix.
    """
    X = adata.X if layer is None else adata.layers[layer]
    if sparse and not sp.issparse(X):
        X = sp.csr_matrix(X)
    elif not sparse and sp.issparse(X):
        X = X.toarray()
    return X


def compute_log_total_counts(X_counts):
    """
    Spot-level log1p total counts.
    """
    total_counts = np.asarray(X_counts.sum(axis=1)).ravel().astype(np.float64)
    return np.log1p(total_counts)


def compute_log_detected_genes(X_counts):
    """
    Spot-level log1p number of detected genes.
    """
    detected_genes = np.asarray((X_counts > 0).sum(axis=1)).ravel().astype(np.float64)
    return np.log1p(detected_genes)


def compute_mito_fraction(X_counts, gene_names, eps=1e-12):
    """
    Spot-level mitochondrial fraction using genes starting with 'MT-'.
    """
    gene_names_upper = np.char.upper(np.asarray(gene_names).astype(str))
    mito_mask = np.char.startswith(gene_names_upper, "MT-")
    if mito_mask.sum() == 0:
        raise ValueError("No mitochondrial genes found for 'mito_fraction'.")

    total_counts = np.asarray(X_counts.sum(axis=1)).ravel().astype(np.float64)
    mito_counts = np.asarray(X_counts[:, mito_mask].sum(axis=1)).ravel().astype(np.float64)
    return mito_counts / (total_counts + eps)


def compute_ribo_fraction(X_counts, gene_names, eps=1e-12):
    """
    Spot-level ribosomal fraction using genes starting with 'RPS' or 'RPL'.
    """
    gene_names_upper = np.char.upper(np.asarray(gene_names).astype(str))
    ribo_mask = np.char.startswith(gene_names_upper, "RPS") | np.char.startswith(
        gene_names_upper, "RPL"
    )
    if ribo_mask.sum() == 0:
        raise ValueError("No ribosomal genes found for 'ribo_fraction'.")

    total_counts = np.asarray(X_counts.sum(axis=1)).ravel().astype(np.float64)
    ribo_counts = np.asarray(X_counts[:, ribo_mask].sum(axis=1)).ravel().astype(np.float64)
    return ribo_counts / (total_counts + eps)


def extract_covariates(adata, covariates, layer=None):
    """
    Extract a spot-by-covariate matrix from pre-normalized counts in adata.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object. Should already have any desired gene filtering
        applied so the returned covariates align with the returned expression.
    covariates : list[str]
        Predefined covariates to compute. Supported values:
            - "log_total_counts"
            - "log_detected_genes"
            - "mito_fraction"
            - "ribo_fraction"
    layer : str or None
        Layer to use as the counts source. If None, use adata.X.

    Returns
    -------
    covariate_matrix : (n_spots, n_covariates) ndarray
    """
    covariates = list(covariates)
    X_counts = get_counts_matrix(adata, layer=layer)
    gene_names = np.array(adata.var_names)

    covariate_cols = []
    for cov in covariates:
        if cov == "log_total_counts":
            covariate_cols.append(compute_log_total_counts(X_counts))
        elif cov == "log_detected_genes":
            covariate_cols.append(compute_log_detected_genes(X_counts))
        elif cov == "mito_fraction":
            covariate_cols.append(compute_mito_fraction(X_counts, gene_names))
        elif cov == "ribo_fraction":
            covariate_cols.append(compute_ribo_fraction(X_counts, gene_names))
        else:
            raise ValueError(
                f"Unknown covariate '{cov}'. Supported values are: "
                "'log_total_counts', 'log_detected_genes', "
                "'mito_fraction', 'ribo_fraction'."
            )

    return np.column_stack(covariate_cols).astype(np.float64, copy=False)


def extract_visium_data(
    adata,
    target_sum=1e4,
    transform=None,
    min_counts=10,
    min_spot_fraction=0.01,
    covariates=None,
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

    covariate_matrix = extract_covariates(adata, covariates, layer) if covariates else None

    if target_sum:
        sc.pp.normalize_total(adata, target_sum=target_sum)
    if transform == "log1p":
        sc.pp.log1p(adata)
    elif transform == "sqrt":
        sc.pp.sqrt(adata)

    X = get_counts_matrix(adata, sparse, layer)

    coords = adata.obsm["spatial"].astype(np.float64)
    gene_names = np.array(adata.var_names)

    return X, coords, gene_names, covariate_matrix


@dataclass
class Groups:
    """
    Contiguous spot groups.

    offsets and lengths refer to the current row order.
    """

    offsets: np.ndarray
    lengths: np.ndarray

    @classmethod
    def single(cls, n):
        return cls(
            offsets=np.array([0], dtype=np.int64),
            lengths=np.array([n], dtype=np.int64),
        )

    @classmethod
    def from_labels(cls, labels):
        labels = np.asarray(labels)
        _, codes = np.unique(labels, return_inverse=True)
        order = np.argsort(codes, kind="stable")
        grouped = codes[order]

        starts = np.flatnonzero(np.r_[True, grouped[1:] != grouped[:-1]])
        stops = np.r_[starts[1:], len(labels)]
        lengths = stops - starts

        return order, cls(starts.astype(np.int64), lengths.astype(np.int64))

    @property
    def n_groups(self):
        return len(self.offsets)


@dataclass
class SampleData:
    W: sp.csr_matrix
    K: sp.csr_matrix
    inv_order: np.ndarray
    labels_groups: Groups
    covariates: np.ndarray | None = None

    @property
    def n_spots(self):
        return self.W.shape[0]

    @property
    def n_genes(self):
        return self.W.shape[1]

    @cached_property
    def Wcsc(self):
        return self.W.tocsc(copy=False)


def prepare_sample(W, coords, radius, labels=None, covariates=None, dtype=np.float64):
    """
    Reorder one sample by labels, build its sparse kernel, and keep inverse order.
    """
    n = W.shape[0]

    if labels is None:
        order = np.arange(n, dtype=np.int64)
        groups = Groups.single(n)
    else:
        order, groups = Groups.from_labels(labels)

    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n, dtype=np.int64)

    W = sp.csr_matrix(W[order], dtype=dtype)
    coords = np.asarray(coords, dtype=dtype)[order]

    if covariates is not None:
        covariates = np.asarray(covariates, dtype=dtype)
        if covariates.ndim == 1:
            covariates = covariates[:, None]
        covariates = covariates[order]

    K = kernel_matrix_sparse(coords, radius).astype(dtype)

    return SampleData(W=W, K=K, inv_order=inv, labels_groups=groups, covariates=covariates)


def prepare_samples(W, coords, radius, labels=None, covariates=None, dtype=np.float64):
    W = as_list(W)
    coords = as_list(coords)
    labels = [None] * len(W) if labels is None else as_list(labels)
    covariates = [None] * len(W) if covariates is None else as_list(covariates)

    return [
        prepare_sample(w, xy, radius, lab, cov, dtype=dtype)
        for w, xy, lab, cov in zip(W, coords, labels, covariates)
    ]


def concatenate_samples(samples):
    """
    Concatenate reordered samples and build block-diagonal K.
    """
    W = sp.vstack([s.W for s in samples], format="csr")
    K = sp.block_diag([s.K for s in samples], format="csr")

    offsets = np.r_[0, np.cumsum([s.n_spots for s in samples[:-1]])].astype(np.int64)

    sample_offsets = []
    sample_lengths = []
    label_offsets = []
    label_lengths = []

    covs = []
    has_cov = any(s.covariates is not None for s in samples)
    n_cov = next((s.covariates.shape[1] for s in samples if s.covariates is not None), 0)

    for off, s in zip(offsets, samples):
        off = int(off)
        n = s.n_spots

        sample_offsets.append(off)
        sample_lengths.append(n)

        label_offsets.extend((off + s.labels_groups.offsets).tolist())
        label_lengths.extend(s.labels_groups.lengths.tolist())

        if has_cov:
            if s.covariates is None:
                covs.append(np.zeros((n, n_cov)))
            else:
                covs.append(s.covariates)

    sample_groups = Groups(np.asarray(sample_offsets), np.asarray(sample_lengths))
    label_groups = Groups(np.asarray(label_offsets), np.asarray(label_lengths))
    covariates = None if not has_cov else np.vstack(covs)

    return W, K, sample_groups, label_groups, covariates


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
