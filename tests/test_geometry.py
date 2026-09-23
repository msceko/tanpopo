import itertools

import numpy as np

from tanpopo.data import prepare_sample
from tanpopo.kernel import spatial_statistic_kernels
from tanpopo.models import (
    DifferentialSpatialProgramModel,
    SpatialProgramModel,
    _sample_statistic_kernel,
)
from tanpopo.projection import SpotProjector


def cosine(a, b):
    return abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_mass_normalisation_fixes_pair_mass_to_n_spots():
    rng = np.random.default_rng(100)
    coords = rng.uniform(size=(80, 2))
    kernels, diagnostics = spatial_statistic_kernels(
        [coords],
        0.35,
        geometry_normalisation="mass",
        spatial_statistic="mark_correlation",
    )
    np.testing.assert_allclose(kernels[0].sum(), len(coords), rtol=1e-12)
    np.testing.assert_allclose(diagnostics[0]["pair_mass"], len(coords), rtol=1e-12)


def test_distance_normalisation_matches_reference_profile_across_geometries():
    rng = np.random.default_rng(101)
    square = rng.uniform(size=(180, 2))
    rectangle = np.c_[rng.uniform(0, 2.0, 240), rng.uniform(0, 0.5, 240)]
    _, diagnostics = spatial_statistic_kernels(
        [square, rectangle],
        0.45,
        geometry_normalisation="distance",
        spatial_statistic="mark_correlation",
        distance_bins=6,
    )
    reference = diagnostics[0]["distance_reference_weights"]
    common = diagnostics[0]["common_distance_bins"]
    assert np.any(common)
    for n, diag in zip([len(square), len(rectangle)], diagnostics):
        np.testing.assert_allclose(diag["pair_mass"], n, rtol=1e-12, atol=1e-12)
        observed = diag["distance_bin_mass"] / diag["pair_mass"]
        np.testing.assert_allclose(observed[common], reference[common], rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(observed[~common], 0.0, atol=1e-12)


def test_single_sample_distance_normalisation_equals_mass_normalisation():
    rng = np.random.default_rng(102)
    coords = rng.uniform(size=(100, 2))
    mass, _ = spatial_statistic_kernels(
        [coords], 0.3, geometry_normalisation="mass", distance_bins=7
    )
    distance, _ = spatial_statistic_kernels(
        [coords], 0.3, geometry_normalisation="distance", distance_bins=7
    )
    np.testing.assert_allclose(distance[0].toarray(), mass[0].toarray(), rtol=1e-12, atol=1e-12)


def test_variogram_operator_matches_weighted_pairwise_difference_identity():
    rng = np.random.default_rng(103)
    coords = rng.uniform(size=(60, 2))
    adjacency, _ = spatial_statistic_kernels(
        [coords], 0.4, spatial_statistic="mark_correlation", geometry_normalisation="mass"
    )
    laplacian, _ = spatial_statistic_kernels(
        [coords], 0.4, spatial_statistic="variogram", geometry_normalisation="mass"
    )
    A = adjacency[0].tocoo()
    L = laplacian[0]
    y = rng.normal(size=len(coords))
    lhs = float(y @ (L @ y))
    rhs = 0.5 * np.sum(A.data * (y[A.row] - y[A.col]) ** 2)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(L.sum(axis=1)).ravel(), 0.0, atol=1e-12)


def test_null_center_has_zero_random_labelling_expectation():
    rng = np.random.default_rng(104)
    n = 6
    coords = np.c_[np.arange(n, dtype=float), np.zeros(n)]
    sample = prepare_sample(
        rng.normal(size=(n, 2)),
        coords,
        radius=2.5,
        geometry_normalisation="mass",
        spatial_statistic="mark_correlation",
    )
    K = _sample_statistic_kernel(sample, null_center=True)
    projector = SpotProjector(n, groups=sample.labels_groups)
    y = projector.apply(rng.normal(size=(n, 1)))[:, 0]
    values = []
    for perm in itertools.permutations(range(n)):
        yp = y[np.asarray(perm)]
        values.append(float(yp @ (K @ yp)))
    np.testing.assert_allclose(np.mean(values), 0.0, atol=1e-12)


def test_variogram_differential_recovers_local_roughness_program():
    rng = np.random.default_rng(105)
    g = 24
    v = np.zeros(g)
    v[:5] = rng.normal(size=5)
    v /= np.linalg.norm(v)
    Ws, coords = [], []
    for rough in [True, True, False, False]:
        n = 180
        x = np.linspace(0, 18, n)
        xy = np.c_[x, np.zeros(n)]
        z = (-1.0) ** np.arange(n) if rough else np.sin(x / 4.0)
        X = rng.normal(scale=0.25, size=(n, g)) + 1.8 * z[:, None] * v
        Ws.append(X)
        coords.append(xy)
    model = DifferentialSpatialProgramModel(
        0.5,
        positive_samples=[0, 1],
        negative_samples=[2, 3],
        spatial_statistic="variogram",
        geometry_normalisation="distance",
        distance_bins=4,
    ).fit(Ws, coords, 3)
    best = max(cosine(v, model.gene_loadings[:, j]) for j in range(3))
    assert best > 0.8
    assert np.any(model.eigenvalues > 0)


def test_null_center_rejects_variogram():
    try:
        SpatialProgramModel(1.0, spatial_statistic="variogram", null_center=True)
    except ValueError as exc:
        assert "mark_correlation" in str(exc)
    else:
        raise AssertionError("variogram null centering should be rejected")


def test_geometry_none_preserves_historical_square_kernel():
    from tanpopo.kernel import kernel_matrix_sparse

    rng = np.random.default_rng(106)
    coords = rng.uniform(size=(70, 2))
    old = kernel_matrix_sparse(coords, 0.35, normalisation="symmetric", zero_diagonal=True)
    new, _ = spatial_statistic_kernels(
        [coords],
        0.35,
        spatial_statistic="mark_correlation",
        geometry_normalisation="none",
        graph_normalisation="symmetric",
    )
    np.testing.assert_allclose(new[0].toarray(), old.toarray(), rtol=0, atol=0)
