from collections.abc import Iterable
from contextlib import contextmanager
from functools import wraps
from time import perf_counter

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize


def vector_or_matrix(method):
    """
    Decorator for methods that should accept either:
      - a vector of shape (n,)
      - a matrix of shape (n, m)

    The wrapped method always receives a 2D ndarray of shape (n, m),
    and the output is squeezed back to 1D if the input was 1D.
    """

    @wraps(method)
    def wrapper(self, X, *args, **kwargs):
        X = np.asarray(X, dtype=self.dtype)
        was_vector = X.ndim == 1

        if was_vector:
            X = X[:, None]
        elif X.ndim != 2:
            raise ValueError(f"Expected 1D or 2D input, got shape {X.shape}.")

        out = method(self, X, *args, **kwargs)
        out = np.asarray(out, dtype=self.dtype)

        if was_vector:
            return out[:, 0]
        return out

    return wrapper


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


def as_list(x):
    """Return x as [x] if not a list"""
    return list(x) if isinstance(x, (list, tuple)) else [x]


def all_equal(X):
    """Check if all elements in list X are equal"""
    return all(x == X[0] for x in X)


def str2bool(arg):
    ua = str(arg).upper()
    if "TRUE".startswith(ua):
        return True
    elif "FALSE".startswith(ua):
        return False
    else:
        raise ValueError("Argument must be 'True' or 'False'")


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


def order_by_label(labels):
    """
    Reorder by label so each label block is contiguous.

    Returns
    -------
    order : (n_rows,) int ndarray
        Permutation applied within the sample.
    offsets : (n_blocks,) int ndarray
        Block start offsets after reordering.
    lengths : (n_blocks,) int ndarray
        Block lengths after reordering.
    """
    n_rows = len(labels)
    order = np.arange(n_rows, dtype=np.int64)

    _, codes = np.unique(labels, return_inverse=True)
    order = np.argsort(codes, kind="stable")
    grouped_codes = codes[order]

    starts = np.flatnonzero(np.r_[True, grouped_codes[1:] != grouped_codes[:-1]])
    stops = np.r_[starts[1:], n_rows]
    lengths = stops - starts

    return order, starts.astype(np.int64), lengths.astype(np.int64)


def normalise_gene_weights(X):
    """Normalise columns to sum to 1"""
    return normalize(X, norm="l1", axis=0)


def quad_form(applyS, a):
    """Compute a^T S a using only S@a."""
    Sa = applyS(a.reshape(-1, 1)).ravel()
    return float(a @ Sa)


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


def top_scored_genes(scores, genes, n_top, mode="pos"):
    idx = argtop(scores, n_top, mode)
    return list(genes[idx]), list(scores[idx])


def top_genes_per_basis(eigvecs, genes, n_top, mode="abs"):
    """Compute top genes for each gene basis"""
    top_genes = []
    for k in range(eigvecs.shape[1]):
        idx = argtop(eigvecs[:, k], n_top, mode)
        top_genes.append({genes[i]: eigvecs[i, k] for i in idx})
    return top_genes


def print_top_genes(scores, genes, n_top):
    top_genes, top_scores = top_scored_genes(scores, genes, n_top)
    for gene, score in zip(top_genes, top_scores):
        print(f"{gene:15s} {score:+.3f}")


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
        print(f"\nEigenmode {k} (λ = {eigvals[k]:.4f})")
        for (g_abs, w_abs), (g_pos, w_pos), (g_neg, w_neg) in zip(
            top_genes_abs[k].items(), top_genes_pos[k].items(), top_genes_neg[k].items()
        ):
            print(f"{g_abs:15s} {w_abs:+.3f}", end="  |  ")
            print(f"{g_pos:15s} {w_pos:+.3f}", end="  |  ")
            print(f"{g_neg:15s} {w_neg:+.3f}")


def cumulative_contribution(eigvecs):
    """Cumulative squared-loading contribution curve for one component"""
    sq_sorted = np.sort(eigvecs**2, axis=0)[::-1]
    return np.cumsum(sq_sorted, axis=0) / sq_sorted.sum(0)


def choose_k_by_energy(eigvals, energy=0.9):
    """
    Choose smallest k such that cumulative explained 'kernel variance' >= energy.
    Using eigvals of centered Gram.
    """
    if eigvals.size == 0:
        return 0
    cum = np.cumsum(eigvals)
    total = cum[-1]
    if total <= 0:
        return 0
    k = int(np.searchsorted(cum / total, energy) + 1)
    return k


def choose_k_by_elbow(eigvals, k_max=None):
    """
    Lightweight elbow finder on log-eigvals curve:
    pick k where second-difference is most negative (strongest curvature).
    """
    if eigvals.size < 3:
        return int(eigvals.size)
    if k_max is None:
        k_max = eigvals.size
    y = np.log(np.clip(eigvals[:k_max], 1e-30, None))
    # discrete second derivative
    d2 = y[:-2] - 2 * y[1:-1] + y[2:]
    k = int(np.argmin(d2) + 2)  # +2 to map to component index (1-based)
    return max(2, min(k, k_max))


def choose_k(eigvals, method="auto", energy=0.9, k_max=50):
    """
    method:
      - "energy": energy threshold
      - "elbow": curvature elbow
      - "auto": min(elbow, energy-based) with sensible bounds
    """
    eigvals = np.asarray(eigvals, dtype=np.float64)
    eigvals = eigvals[: min(k_max, eigvals.size)]
    if eigvals.size == 0:
        return 0

    k_e = choose_k_by_energy(eigvals, energy=energy)
    k_l = choose_k_by_elbow(eigvals, k_max=eigvals.size)

    if method == "energy":
        return k_e
    if method == "elbow":
        return k_l
    # auto: take the more conservative (smaller) but at least 2
    return int(max(2, min(k_e, k_l, eigvals.size)))
