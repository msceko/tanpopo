import numpy as np

from tanpopo.utils import argtop


def fractional_energy(phi, eps=1e-12):
    """Return fractional energy for each eigenmode"""
    E = phi**2
    return E / (E.sum(axis=1, keepdims=True) + eps)


def gene_loading_mass(eigvecs, eigvals=None):
    """Captured gene loading mass - optionally weighted by eigenvalues"""
    if eigvals is not None:
        return (eigvals * eigvecs**2).sum(1)
    return (eigvecs**2).sum(1)


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


def top_scored_genes(scores, genes, n_top, mode="pos"):
    """Return n top genes from their scores"""
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
    top_genes_abs = top_genes_per_basis(eigvecs, genes, n_top, "abs")
    top_genes_pos = top_genes_per_basis(eigvecs, genes, n_top, "pos")
    top_genes_neg = top_genes_per_basis(eigvecs, genes, n_top, "neg")
    for k in range(eigvecs.shape[1]):
        print(f"\nEigenmode {k} (λ = {eigvals[k]:.6e})")
        for (g_abs, w_abs), (g_pos, w_pos), (g_neg, w_neg) in zip(
            top_genes_abs[k].items(), top_genes_pos[k].items(), top_genes_neg[k].items()
        ):
            print(f"{g_abs:15s} {w_abs:+.3f}", end="  |  ")
            print(f"{g_pos:15s} {w_pos:+.3f}", end="  |  ")
            print(f"{g_neg:15s} {w_neg:+.3f}")
