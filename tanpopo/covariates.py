from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from tanpopo.utils import get_counts_matrix


def _row_sum(X):
    return np.asarray(X.sum(axis=1)).ravel() if sp.issparse(X) else np.asarray(X).sum(axis=1)


def _detected(X):
    if sp.issparse(X):
        return np.asarray((X > 0).sum(axis=1)).ravel()
    return np.sum(np.asarray(X) > 0, axis=1)


def _fraction(X, mask):
    total = _row_sum(X)
    sub = _row_sum(X[:, mask]) if np.any(mask) else np.zeros_like(total, dtype=float)
    return np.divide(sub, total, out=np.zeros_like(sub, dtype=float), where=total > 0)


def compute_covariates(adata, covariates, layer=None):
    X = get_counts_matrix(adata, sparse=True, layer=layer)
    names = np.asarray(adata.var_names).astype(str)
    columns = []
    for covariate in covariates:
        value = getattr(covariate, "value", covariate)
        if value == "log_total_counts":
            columns.append(np.log1p(_row_sum(X)))
        elif value == "log_detected_genes":
            columns.append(np.log1p(_detected(X)))
        elif value == "mito_fraction":
            mask = np.char.startswith(np.char.upper(names), "MT-")
            columns.append(_fraction(X, mask))
        elif value == "ribo_fraction":
            upper = np.char.upper(names)
            mask = np.char.startswith(upper, "RPL") | np.char.startswith(upper, "RPS")
            columns.append(_fraction(X, mask))
        else:
            raise ValueError(f"Unknown covariate: {value}")
    adata.obsm["tanpopo_covariates"] = np.column_stack(columns) if columns else None
    return adata.obsm.get("tanpopo_covariates")
