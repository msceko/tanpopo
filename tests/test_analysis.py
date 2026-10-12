import numpy as np

from tanpopo.analysis import compare_cross_results, compare_program_results
from tanpopo.results import CrossProgramResult, ProgramResult


def _program(loadings, values=None):
    loadings = np.asarray(loadings, dtype=float)
    if values is None:
        values = np.arange(loadings.shape[1], 0, -1, dtype=float)
    return ProgramResult(
        analysis_id="x",
        variant="default",
        values=np.asarray(values, dtype=float),
        value_name="eigenvalue",
        gene_loadings=loadings,
        spot_modes=None,
        gene_names=np.asarray([f"g{i}" for i in range(loadings.shape[0])]),
    )


def _cross(target, neighbour, values=None):
    target = np.asarray(target, dtype=float)
    neighbour = np.asarray(neighbour, dtype=float)
    if values is None:
        values = np.arange(target.shape[1], 0, -1, dtype=float)
    return CrossProgramResult(
        analysis_id="x",
        values=np.asarray(values, dtype=float),
        target_loadings=target,
        neighbour_loadings=neighbour,
        target_modes=None,
        neighbour_modes=None,
        gene_names=np.asarray([f"g{i}" for i in range(target.shape[0])]),
    )


def test_program_comparison_recovers_permutation_and_sign_flip():
    rng = np.random.default_rng(10)
    q, _ = np.linalg.qr(rng.normal(size=(30, 4)))
    a = _program(q, [4, 3, 2, 1])
    permutation = [2, 0, 1, 3]
    signs = np.array([-1, 1, -1, 1])
    b = _program(q[:, permutation] * signs, np.array([2, 4, 3, 1]))
    frame = compare_program_results(a, b, top_genes=10)
    np.testing.assert_allclose(frame["absolute_cosine"], 1.0, atol=1e-12)
    np.testing.assert_allclose(frame.attrs["top3_subspace_overlap"], 1.0, atol=1e-12)
    np.testing.assert_allclose(frame.attrs["top5_subspace_overlap"], 1.0, atol=1e-12)


def test_cross_comparison_is_invariant_to_paired_sign_flips():
    rng = np.random.default_rng(11)
    target, _ = np.linalg.qr(rng.normal(size=(24, 3)))
    neighbour, _ = np.linalg.qr(rng.normal(size=(24, 3)))
    a = _cross(target, neighbour, [3, 2, 1])
    permutation = [1, 2, 0]
    signs = np.array([-1, 1, -1])
    b = _cross(
        target[:, permutation] * signs,
        neighbour[:, permutation] * signs,
        [2, 1, 3],
    )
    frame = compare_cross_results(a, b, top_genes=8)
    np.testing.assert_allclose(frame["rank_one_similarity"], 1.0, atol=1e-12)
    np.testing.assert_allclose(frame["absolute_rank_one_similarity"], 1.0, atol=1e-12)
