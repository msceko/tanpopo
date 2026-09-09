from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
from time import perf_counter

import numpy as np
import pandas as pd
import scipy.sparse as sp


def as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def none_to_list(value, n):
    return [None] * n if value is None else as_list(value)


def as_value(x):
    return x.value if hasattr(x, "value") else x


def pd_dtype(series: pd.Series):
    if isinstance(series.dtype, pd.CategoricalDtype):
        return series.cat.categories.dtype
    return series.dtype


def get_counts_matrix(adata, sparse=True, layer=None):
    X = adata.X if layer is None else adata.layers[layer]
    if sparse and not sp.issparse(X):
        X = sp.csr_matrix(X)
    elif not sparse and sp.issparse(X):
        X = X.toarray()
    return X


def center_columns(X):
    X = np.asarray(X)
    return X - X.mean(axis=0, keepdims=True)


def column_normalize(X, eps=1e-12):
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=0, keepdims=True)
    return X / np.maximum(norms, eps)


@contextmanager
def timed(label: str, enabled: bool, sink=print):
    if not enabled:
        yield
        return
    sink(label, end="")
    t0 = perf_counter()
    try:
        yield
    finally:
        sink(f": {perf_counter() - t0:.2f} s")
