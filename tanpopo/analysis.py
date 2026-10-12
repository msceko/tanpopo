from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from tanpopo.results import CrossProgramResult, ProgramResult


def _unit_columns(values):
    values = np.asarray(values, dtype=float)
    norms = np.linalg.norm(values, axis=0)
    if np.any(norms == 0):
        raise ValueError("Cannot compare a zero-norm mode")
    return values / norms


def _top_gene_jaccard(a, b, n=20):
    n = min(int(n), len(a), len(b))
    if n <= 0:
        return float("nan")
    ia = set(np.argsort(np.abs(a))[-n:])
    ib = set(np.argsort(np.abs(b))[-n:])
    return len(ia & ib) / len(ia | ib)


def _subspace_overlap(a, b, k):
    k = min(int(k), a.shape[1], b.shape[1])
    if k <= 0:
        return float("nan")
    qa = np.linalg.qr(a[:, :k])[0]
    qb = np.linalg.qr(b[:, :k])[0]
    singular = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return float(np.mean(np.square(singular)))


def compare_program_results(a: ProgramResult, b: ProgramResult, top_genes=20):
    """One-to-one mode matching with direction, gene and spectrum diagnostics."""
    common, ia, ib = np.intersect1d(a.gene_names, b.gene_names, return_indices=True)
    if common.size == 0:
        raise ValueError("The analyses do not share any genes")
    A = _unit_columns(a.gene_loadings[ia])
    B = _unit_columns(b.gene_loadings[ib])
    signed = A.T @ B
    rows, cols = linear_sum_assignment(-np.abs(signed))
    records = []
    for i, j in zip(rows, cols):
        records.append(
            {
                "mode_a": int(i),
                "mode_b": int(j),
                "signed_cosine": float(signed[i, j]),
                "absolute_cosine": float(abs(signed[i, j])),
                "top_gene_jaccard": _top_gene_jaccard(A[:, i], B[:, j], top_genes),
                "value_a": float(a.values[i]),
                "value_b": float(b.values[j]),
            }
        )
    frame = pd.DataFrame(records).sort_values("mode_a").reset_index(drop=True)
    frame.attrs["common_genes"] = int(common.size)
    frame.attrs["top3_subspace_overlap"] = _subspace_overlap(A, B, 3)
    frame.attrs["top5_subspace_overlap"] = _subspace_overlap(A, B, 5)
    frame.attrs["top10_subspace_overlap"] = _subspace_overlap(A, B, 10)
    return frame


def compare_cross_results(a: CrossProgramResult, b: CrossProgramResult, top_genes=20):
    """Compare paired target-neighbour rank-one cross-program components."""
    common, ia, ib = np.intersect1d(a.gene_names, b.gene_names, return_indices=True)
    if common.size == 0:
        raise ValueError("The analyses do not share any genes")
    At = _unit_columns(a.target_loadings[ia])
    An = _unit_columns(a.neighbour_loadings[ia])
    Bt = _unit_columns(b.target_loadings[ib])
    Bn = _unit_columns(b.neighbour_loadings[ib])
    target_cosine = At.T @ Bt
    neighbour_cosine = An.T @ Bn
    rank_one_similarity = target_cosine * neighbour_cosine
    rows, cols = linear_sum_assignment(-np.abs(rank_one_similarity))
    records = []
    for i, j in zip(rows, cols):
        records.append(
            {
                "mode_a": int(i),
                "mode_b": int(j),
                "rank_one_similarity": float(rank_one_similarity[i, j]),
                "absolute_rank_one_similarity": float(abs(rank_one_similarity[i, j])),
                "target_cosine": float(target_cosine[i, j]),
                "neighbour_cosine": float(neighbour_cosine[i, j]),
                "target_top_gene_jaccard": _top_gene_jaccard(At[:, i], Bt[:, j], top_genes),
                "neighbour_top_gene_jaccard": _top_gene_jaccard(An[:, i], Bn[:, j], top_genes),
                "value_a": float(a.values[i]),
                "value_b": float(b.values[j]),
            }
        )
    return pd.DataFrame(records).sort_values("mode_a").reset_index(drop=True)


def compare_spot_modes(a: ProgramResult, b: ProgramResult):
    if a.spot_modes is None or b.spot_modes is None:
        raise ValueError("Both analyses must contain spot modes")
    if a.spot_modes.shape[0] != b.spot_modes.shape[0]:
        raise ValueError("Spot-mode comparison requires analyses of the same observations")
    valid = np.all(np.isfinite(a.spot_modes), axis=1) & np.all(np.isfinite(b.spot_modes), axis=1)
    A = a.spot_modes[valid]
    B = b.spot_modes[valid]
    A = _unit_columns(A - A.mean(axis=0, keepdims=True))
    B = _unit_columns(B - B.mean(axis=0, keepdims=True))
    cosine = A.T @ B
    rows, cols = linear_sum_assignment(-np.abs(cosine))
    return (
        pd.DataFrame(
            {
                "mode_a": rows.astype(int),
                "mode_b": cols.astype(int),
                "signed_cosine": cosine[rows, cols],
                "absolute_cosine": np.abs(cosine[rows, cols]),
            }
        )
        .sort_values("mode_a")
        .reset_index(drop=True)
    )
