from collections.abc import Iterable
from contextlib import contextmanager
from functools import wraps
from time import perf_counter

import numpy as np
import pandas as pd
import scipy.sparse as sp


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


def pd_dtype(series: pd.Series):
    if isinstance(series.dtype, pd.CategoricalDtype):
        return series.cat.categories.dtype
    return series.dtype


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
