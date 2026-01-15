from collections.abc import Iterable
from contextlib import contextmanager
from time import perf_counter

import anndata as ad
import numpy as np
import pandas as pd
import squidpy as sq
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

from basis import project_spatial_basis


@contextmanager
def timed(label: str, enabled: bool, sink=print, unit="s"):
    if not enabled:
        yield
        return
    scale = 1000.0 if unit == "ms" else 1.0
    sink(f"{label}", end="")
    t0 = perf_counter()
    try:
        yield
    finally:
        dt = (perf_counter() - t0) * scale
        sink(f": {dt:.2f} {unit}")


def make_iterable(obj):
    """
    Ensure obj is iterable.
    """
    if isinstance(obj, (str, bytes)):
        return [obj]
    if isinstance(obj, Iterable):
        return obj
    return [obj]


def normalise_gene_weights(X):
    """Normalise columns to sum to 1"""
    return normalize(X, norm="l1", axis=0)


def bin_spatial_basis(
    adata,
    prefix="spatial_basis_",
    bin_size=16,
    reducer="mean",  # "mean" or "median"
):
    """Bin spatial basis vectors for plotting"""
    # find all matching obs columns
    basis_cols = [c for c in adata.obs.columns if c.startswith(prefix)]
    if not basis_cols:
        raise KeyError(f"No adata.obs columns start with {prefix!r}")

    xy = np.asarray(adata.obsm["spatial"])
    bx = np.floor_divide(xy[:, 0], bin_size).astype(np.int32)
    by = np.floor_divide(xy[:, 1], bin_size).astype(np.int32)
    bin_id = bx.astype(str) + "_" + by.astype(str)

    # build a compact dataframe: coords + only the columns you need
    df = pd.DataFrame(
        {
            "bin": bin_id,
            "x": xy[:, 0],
            "y": xy[:, 1],
        },
        index=adata.obs_names,
    )
    df[basis_cols] = adata.obs[basis_cols].to_numpy()  # avoids pandas alignment surprises

    g = df.groupby("bin", sort=False)

    # bin centroids
    coords = g[["x", "y"]].mean()

    # reduce each basis column per bin
    if reducer == "mean":
        vals = g[basis_cols].mean()
    elif reducer == "median":
        vals = g[basis_cols].median()
    else:
        raise ValueError("reducer must be 'mean' or 'median'")

    # minimal AnnData for plotting
    out = ad.AnnData(X=np.zeros((coords.shape[0], 0), dtype=np.float32))
    out.obsm["spatial"] = coords.to_numpy()
    out.obs = vals  # includes all spatial_basis_* columns
    out.obs_names = coords.index.astype(str)
    out.uns["spatial"] = adata.uns.get("spatial", {})

    return out, basis_cols


def argtop(v, n_top, mode):
    """Return top n components of vector v"""
    if mode == "abs":
        idx = np.argsort(np.abs(v))[::-1]
    elif mode == "pos":
        idx = np.argsort(v)[::-1]
    elif mode == "neg":
        idx = np.argsort(v)
    else:
        raise ValueError("mode must be 'abs', 'pos', or 'neg'")

    return idx[:n_top]


def top_genes_per_basis(eigvecs, genes, n_top, mode="abs"):
    """Compute top genes for each gene basis"""
    top_genes = []
    for k in range(eigvecs.shape[1]):
        idx = argtop(eigvecs[:, k], n_top, mode)
        top_genes.append({genes[i]: eigvecs[i, k] for i in idx})
    return top_genes


def print_top_genes_per_basis(eigvecs, eigvals, genes, n_top=8):
    """Print top genes for each gene basis"""
    # top_genes = top_genes_per_basis(eigvecs, genes, n_top)
    # for k in range(eigvecs.shape[1]):
    #     print(f"\nBasis {k} (λ = {eigvals[k]:.4f})")
    #     for g, w in top_genes[k].items():
    #         print(f"{g:15s} {w:+.3f}")

    top_genes_abs = top_genes_per_basis(eigvecs, genes, n_top, "abs")
    top_genes_pos = top_genes_per_basis(eigvecs, genes, n_top, "pos")
    top_genes_neg = top_genes_per_basis(eigvecs, genes, n_top, "neg")
    for k in range(eigvecs.shape[1]):
        print(f"\nBasis {k} (λ = {eigvals[k]:.4f})")
        for (g_abs, w_abs), (g_pos, w_pos), (g_neg, w_neg) in zip(
            top_genes_abs[k].items(), top_genes_pos[k].items(), top_genes_neg[k].items()
        ):
            print(f"{g_abs:15s} {w_abs:+.3f}", end="  |  ")
            print(f"{g_pos:15s} {w_pos:+.3f}", end="  |  ")
            print(f"{g_neg:15s} {w_neg:+.3f}")


def plot_spatial_basis(adata, phi, prefix="spatial_basis"):
    """Plot spatial gene basis"""
    keys = []
    for k in range(phi.shape[1]):
        keys.append(f"{prefix}_{k}")
        adata.obs[keys[k]] = phi[:, k]

    sq.pl.spatial_scatter(adata, color=keys, img=None)


def plot_spatial_basis_signed(adata, X, eigvecs):
    """Plot positive and negative components of gene basis independently"""
    phi_pos = project_spatial_basis(X, np.maximum(eigvecs, 0))
    plot_spatial_basis(adata, phi_pos, "spatial_basis_positive")
    phi_neg = project_spatial_basis(X, np.maximum(-eigvecs, 0))
    plot_spatial_basis(adata, phi_neg, "spatial_basis_negative")


def cumulative_contribution(eigvecs):
    """Cumulative squared-loading contribution curve for one component"""
    sq_sorted = np.sort(eigvecs**2, axis=0)[::-1]
    return np.cumsum(sq_sorted, axis=0) / sq_sorted.sum(0)


def plot_cumulative_contribution(eigvecs):
    """Plot cumulative distributions for each squared eigenvector"""
    plt.figure()
    plt.plot(cumulative_contribution(eigvecs))
    plt.xlabel("Number of genes")
    plt.ylabel("Cumulative contribution")
    plt.axhline(0.8, linestyle="--", alpha=0.5)
    plt.legend([f"Basis {n}" for n in range(eigvecs.shape[1])])
