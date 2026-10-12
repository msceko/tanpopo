import numpy as np

from tanpopo.geometry import kernel_matrix_sparse, spatial_covariance_kernels


def test_mass_normalisation_fixes_pair_mass_to_n_spots():
    rng = np.random.default_rng(100)
    coords = rng.uniform(size=(80, 2))
    kernels, diagnostics = spatial_covariance_kernels(
        [coords], 0.35, geometry_normalisation="mass"
    )
    np.testing.assert_allclose(kernels[0].sum(), len(coords), rtol=1e-12)
    np.testing.assert_allclose(diagnostics[0]["pair_mass"], len(coords), rtol=1e-12)
    assert np.allclose(kernels[0].diagonal(), 0.0)


def test_distance_normalisation_matches_reference_profile_across_geometries():
    rng = np.random.default_rng(101)
    square = rng.uniform(size=(180, 2))
    rectangle = np.c_[rng.uniform(0, 2.0, 240), rng.uniform(0, 0.5, 240)]
    _, diagnostics = spatial_covariance_kernels(
        [square, rectangle],
        0.45,
        geometry_normalisation="distance",
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
    mass, _ = spatial_covariance_kernels(
        [coords], 0.3, geometry_normalisation="mass", distance_bins=7
    )
    distance, _ = spatial_covariance_kernels(
        [coords], 0.3, geometry_normalisation="distance", distance_bins=7
    )
    np.testing.assert_allclose(distance[0].toarray(), mass[0].toarray(), rtol=1e-12, atol=1e-12)


def test_geometry_none_preserves_base_square_kernel():
    rng = np.random.default_rng(106)
    coords = rng.uniform(size=(70, 2))
    base = kernel_matrix_sparse(coords, 0.35, normalisation="symmetric", zero_diagonal=True)
    new, _ = spatial_covariance_kernels(
        [coords],
        0.35,
        geometry_normalisation="none",
        graph_normalisation="symmetric",
    )
    np.testing.assert_allclose(new[0].toarray(), base.toarray(), rtol=0, atol=0)
