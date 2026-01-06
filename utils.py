from contextlib import contextmanager
from time import perf_counter

import numpy as np
import squidpy as sq
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


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


def normalise_gene_weights(X):
    """Normalise columns to sum to 1"""
    return normalize(X, norm="l1", axis=0)


def top_genes_per_basis(Z, genes, n_top):
    """Compute top genes for each gene basis"""
    top_genes = []
    for k in range(Z.shape[1]):
        idx = np.argsort(np.abs(Z[:, k]))[::-1][:n_top]
        top_genes.append({genes[i]: Z[i, k] for i in idx})
    return top_genes


def print_top_genes_per_basis(Z, genes, n_top=8):
    """Print top genes for each gene basis"""
    top_genes = top_genes_per_basis(Z, genes, n_top)
    for k in range(Z.shape[1]):
        print(f"\nBasis {k}")
        for g, w in top_genes[k].items():
            print(f"{g:15s} {w:+.3f}")


def plot_spatial_basis(adata, phi, prefix="spatial_basis"):
    """Plot spatial gene basis"""
    keys = []
    for k in range(phi.shape[1]):
        keys.append(f"{prefix}_{k}")
        adata.obs[keys[k]] = phi[:, k]

    sq.pl.spatial_scatter(adata, color=keys)
    plt.show()
