import numpy as np

from tanpopo.data import prepare_cross_sample, prepare_cross_samples, prepare_sample
from tanpopo.kernel import kernel_matrix_sparse
from tanpopo.models import (
    CrossSpatialProgramModel,
    DifferentialCrossSpatialProgramModel,
    SharedCrossSpatialProgramModel,
)


def cosine(a, b):
    return abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_cross_programs_recover_paired_gene_directions():
    rng = np.random.default_rng(20)
    n, g = 500, 50
    coords = np.c_[np.linspace(0, 25, n), np.zeros(n)]
    target = np.arange(n) % 2 == 0
    neighbour = ~target
    vt = np.zeros(g)
    vu = np.zeros(g)
    vt[:7] = rng.normal(size=7)
    vu[15:22] = rng.normal(size=7)
    vt /= np.linalg.norm(vt)
    vu /= np.linalg.norm(vu)
    field = np.sin(coords[:, 0] / 2.5)
    X = rng.normal(scale=0.3, size=(n, g))
    X[target] += 2.5 * field[target, None] * vt
    X[neighbour] += 2.5 * field[neighbour, None] * vu

    model = CrossSpatialProgramModel(1.5, objective="covariance").fit(
        X, coords, 2, target, neighbour
    )
    assert cosine(vt, model.target_loadings[:, 0]) > 0.8
    assert cosine(vu, model.neighbour_loadings[:, 0]) > 0.8

def make_shared_cross_samples(seed=100, sizes=(320, 420, 520), n_genes=60):
    rng = np.random.default_rng(seed)
    vt = np.zeros(n_genes)
    vu = np.zeros(n_genes)
    vt[:8] = rng.normal(size=8)
    vu[15:24] = rng.normal(size=9)
    vt /= np.linalg.norm(vt)
    vu /= np.linalg.norm(vu)

    Xs, coords, target_masks, neighbour_masks = [], [], [], []
    for sample_index, n in enumerate(sizes):
        x = np.linspace(0, 24, n)
        xy = np.c_[x, np.zeros(n)]
        target = np.arange(n) % 2 == 0
        neighbour = ~target
        field = np.sin(x / 2.8 + 0.35 * sample_index)
        X = rng.normal(scale=0.35, size=(n, n_genes))
        X[target] += 2.2 * field[target, None] * vt
        X[neighbour] += 2.4 * field[neighbour, None] * vu
        Xs.append(X)
        coords.append(xy)
        target_masks.append(target)
        neighbour_masks.append(neighbour)
    return Xs, coords, target_masks, neighbour_masks, vt, vu


def test_shared_cross_programs_recover_recurrent_paired_directions():
    Xs, coords, targets, neighbours, vt, vu = make_shared_cross_samples()
    model = SharedCrossSpatialProgramModel(1.5, objective="covariance").fit(
        Xs,
        coords,
        3,
        target_masks=targets,
        neighbour_masks=neighbours,
    )

    assert cosine(vt, model.target_loadings[:, 0]) > 0.9
    assert cosine(vu, model.neighbour_loadings[:, 0]) > 0.9
    assert len(model.target_modes) == len(Xs)
    assert len(model.neighbour_modes) == len(Xs)
    assert model.sample_mode_covariance_.shape == (len(Xs), 3)
    assert np.all(model.sample_mode_covariance_[:, 0] > 0)

    effective_sizes = [
        target.sum() * neighbour.sum()
        for target, neighbour in zip(targets, neighbours)
    ]
    expected_weights = 1.0 / np.sqrt(np.asarray(effective_sizes))
    np.testing.assert_allclose(model.sample_coefficients_, expected_weights)


def test_shared_cross_single_sample_matches_cross_model():
    Xs, coords, targets, neighbours, _, _ = make_shared_cross_samples(sizes=(360,))
    single = CrossSpatialProgramModel(1.5, objective="covariance").fit(
        Xs[0], coords[0], 3, targets[0], neighbours[0]
    )
    shared = SharedCrossSpatialProgramModel(
        1.5, objective="covariance", sample_weighting="none"
    ).fit(
        Xs,
        coords,
        3,
        target_masks=targets,
        neighbour_masks=neighbours,
    )

    np.testing.assert_allclose(single.singular_values, shared.singular_values, rtol=1e-6)
    for mode in range(3):
        assert cosine(single.target_loadings[:, mode], shared.target_loadings[:, mode]) > 0.999
        assert (
            cosine(single.neighbour_loadings[:, mode], shared.neighbour_loadings[:, mode])
            > 0.999
        )


def test_shared_cross_supports_all_objectives():
    Xs, coords, targets, neighbours, vt, vu = make_shared_cross_samples(
        seed=120, sizes=(300, 360, 420), n_genes=50
    )
    for objective in ("covariance", "gene_standardized", "gain"):
        model = SharedCrossSpatialProgramModel(
            1.5,
            objective=objective,
            expression_rank=20,
        ).fit(
            Xs,
            coords,
            3,
            target_masks=targets,
            neighbour_masks=neighbours,
        )
        best_target = max(cosine(vt, model.target_loadings[:, j]) for j in range(3))
        best_neighbour = max(cosine(vu, model.neighbour_loadings[:, j]) for j in range(3))
        assert best_target > 0.65
        assert best_neighbour > 0.65


def test_shared_cross_covariance_matches_explicit_weighted_sum():
    Xs, coords, targets, neighbours, _, _ = make_shared_cross_samples(
        seed=140, sizes=(180, 260), n_genes=24
    )
    model = SharedCrossSpatialProgramModel(1.8, objective="covariance").fit(
        Xs,
        coords,
        4,
        target_masks=targets,
        neighbour_masks=neighbours,
    )

    explicit = np.zeros((24, 24))
    for weight, sample in zip(model.sample_coefficients_, model.samples):
        target = sample.W_target.toarray()
        neighbour = sample.W_neighbour.toarray()
        target -= target.mean(axis=0, keepdims=True)
        neighbour -= neighbour.mean(axis=0, keepdims=True)
        explicit += weight * target.T @ (sample.K @ neighbour)

    expected = np.linalg.svd(explicit, compute_uv=False)[:4]
    np.testing.assert_allclose(model.singular_values, expected, rtol=1e-6, atol=1e-8)


def test_cross_mass_normalisation_uses_geometric_mean_pair_mass():
    rng = np.random.default_rng(150)
    n, g = 120, 8
    X = rng.normal(size=(n, g))
    coords = rng.uniform(size=(n, 2))
    target = np.zeros(n, dtype=bool)
    neighbour = np.zeros(n, dtype=bool)
    target[:45] = True
    neighbour[45:] = True
    sample = prepare_cross_sample(
        X,
        coords,
        0.5,
        target,
        neighbour,
        geometry_normalisation="mass",
    )
    expected = np.sqrt(target.sum() * neighbour.sum())
    np.testing.assert_allclose(sample.K.sum(), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        sample.geometry_diagnostics["target_pair_mass"], expected, rtol=1e-12
    )


def test_shared_cross_distance_normalisation_matches_pair_distance_profile():
    rng = np.random.default_rng(151)
    Xs, coords, targets, neighbours = [], [], [], []
    geometries = [
        rng.uniform(size=(180, 2)),
        np.c_[rng.uniform(0, 2.0, 240), rng.uniform(0, 0.5, 240)],
    ]
    for xy in geometries:
        n = len(xy)
        target = np.arange(n) % 3 != 0
        neighbour = np.arange(n) % 3 == 0
        Xs.append(rng.normal(size=(n, 6)))
        coords.append(xy)
        targets.append(target)
        neighbours.append(neighbour)

    samples = prepare_cross_samples(
        Xs,
        coords,
        0.55,
        targets,
        neighbours,
        geometry_normalisation="distance",
        distance_bins=6,
    )
    reference = samples[0].geometry_diagnostics["distance_reference_weights"]
    common = samples[0].geometry_diagnostics["common_distance_bins"]
    assert np.any(common)
    for sample in samples:
        diag = sample.geometry_diagnostics
        target_mass = np.sqrt(sample.n_target * sample.n_neighbour)
        np.testing.assert_allclose(diag["pair_mass"], target_mass, rtol=1e-12)
        observed = diag["distance_bin_mass"] / diag["pair_mass"]
        np.testing.assert_allclose(observed[common], reference[common], rtol=1e-11)
        np.testing.assert_allclose(observed[~common], 0.0, atol=1e-12)


def test_cross_overlapping_masks_remove_biological_self_pairs_before_normalisation():
    n = 6
    X = np.arange(n * 3, dtype=float).reshape(n, 3)
    coords = np.c_[np.arange(n, dtype=float), np.zeros(n)]
    target = np.array([True, True, True, True, False, False])
    neighbour = np.array([False, True, True, False, True, True])
    sample = prepare_cross_sample(
        X,
        coords,
        radius=3.0,
        target_mask=target,
        neighbour_mask=neighbour,
        graph_normalisation="none",
        geometry_normalisation="none",
    )
    target_lookup = {obs: i for i, obs in enumerate(sample.target_idx)}
    neighbour_lookup = {obs: j for j, obs in enumerate(sample.neighbour_idx)}
    overlap = set(sample.target_idx) & set(sample.neighbour_idx)
    assert overlap == {1, 2}
    for obs in overlap:
        assert sample.K[target_lookup[obs], neighbour_lookup[obs]] == 0
    assert sample.geometry_diagnostics["removed_identity_pairs"] == len(overlap)


def test_identical_cross_masks_match_square_pair_adjacency():
    rng = np.random.default_rng(152)
    n = 70
    X = rng.normal(size=(n, 5))
    coords = rng.uniform(size=(n, 2))
    mask = np.ones(n, dtype=bool)
    cross = prepare_cross_sample(
        X,
        coords,
        0.4,
        mask,
        mask,
        geometry_normalisation="none",
    )
    square = prepare_sample(
        X,
        coords,
        0.4,
        geometry_normalisation="none",
        spatial_statistic="mark_correlation",
    )
    np.testing.assert_allclose(cross.K.toarray(), square.K.toarray(), rtol=1e-14, atol=1e-15)
    assert cross.geometry_diagnostics["removed_identity_pairs"] == n


def test_cross_geometry_none_preserves_historical_disjoint_kernel():
    rng = np.random.default_rng(153)
    n = 90
    X = rng.normal(size=(n, 4))
    coords = rng.uniform(size=(n, 2))
    target = np.arange(n) % 2 == 0
    neighbour = ~target
    sample = prepare_cross_sample(
        X,
        coords,
        0.45,
        target,
        neighbour,
        geometry_normalisation="none",
        graph_normalisation="symmetric",
    )
    historical = kernel_matrix_sparse(
        coords[target],
        0.45,
        coords_query=coords[neighbour],
        normalisation="symmetric",
        zero_diagonal=False,
    )
    np.testing.assert_allclose(sample.K.toarray(), historical.toarray(), rtol=0, atol=0)


def test_shared_cross_default_weight_cancels_geometry_target_mass():
    Xs, coords, targets, neighbours, _, _ = make_shared_cross_samples(
        seed=154, sizes=(180, 260, 340), n_genes=24
    )
    model = SharedCrossSpatialProgramModel(
        1.8,
        objective="covariance",
        geometry_normalisation="distance",
        distance_bins=5,
    ).fit(
        Xs,
        coords,
        2,
        target_masks=targets,
        neighbour_masks=neighbours,
    )
    for weight, sample in zip(model.sample_coefficients_, model.samples):
        np.testing.assert_allclose(weight * sample.K.sum(), 1.0, rtol=1e-12, atol=1e-12)


def test_single_cross_distance_normalisation_equals_mass_normalisation():
    rng = np.random.default_rng(155)
    n = 100
    X = rng.normal(size=(n, 6))
    coords = rng.uniform(size=(n, 2))
    target = np.arange(n) % 2 == 0
    neighbour = ~target
    mass = prepare_cross_sample(
        X,
        coords,
        0.5,
        target,
        neighbour,
        geometry_normalisation="mass",
        distance_bins=7,
    )
    distance = prepare_cross_sample(
        X,
        coords,
        0.5,
        target,
        neighbour,
        geometry_normalisation="distance",
        distance_bins=7,
    )
    np.testing.assert_allclose(
        distance.K.toarray(), mass.K.toarray(), rtol=1e-12, atol=1e-12
    )


def make_differential_cross_samples(seed=170, group_a=3, group_b=2, n_genes=36):
    rng = np.random.default_rng(seed)
    vt = np.zeros(n_genes)
    vu = np.zeros(n_genes)
    vt[:6] = rng.normal(size=6)
    neighbour_start = min(12, n_genes - 6)
    neighbour_stop = min(neighbour_start + 6, n_genes)
    vu[neighbour_start:neighbour_stop] = rng.normal(
        size=neighbour_stop - neighbour_start
    )
    vt /= np.linalg.norm(vt)
    vu /= np.linalg.norm(vu)
    Xs, coords, targets, neighbours = [], [], [], []
    for sample_index in range(group_a + group_b):
        n = 180 + 20 * sample_index
        x = np.linspace(0, 20, n)
        xy = np.c_[x, np.zeros(n)]
        target = np.arange(n) % 2 == 0
        neighbour = ~target
        X = rng.normal(scale=0.35, size=(n, n_genes))
        if sample_index < group_a:
            field = np.sin(x / 2.4 + 0.2 * sample_index)
            X[target] += 2.3 * field[target, None] * vt
            X[neighbour] += 2.5 * field[neighbour, None] * vu
        else:
            target_field = rng.normal(size=n)
            neighbour_field = rng.normal(size=n)
            X[target] += 1.6 * target_field[target, None] * vt
            X[neighbour] += 1.6 * neighbour_field[neighbour, None] * vu
        Xs.append(X)
        coords.append(xy)
        targets.append(target)
        neighbours.append(neighbour)
    return Xs, coords, targets, neighbours, vt, vu


def test_differential_cross_recovers_group_a_specific_pair():
    Xs, coords, targets, neighbours, vt, vu = make_differential_cross_samples()
    model = DifferentialCrossSpatialProgramModel(
        1.5,
        positive_samples=[0, 1, 2],
        negative_samples=[3, 4],
        objective="covariance",
    ).fit(
        Xs,
        coords,
        3,
        target_masks=targets,
        neighbour_masks=neighbours,
    )
    assert max(cosine(vt, model.target_loadings[:, j]) for j in range(3)) > 0.85
    assert max(cosine(vu, model.neighbour_loadings[:, j]) for j in range(3)) > 0.85
    np.testing.assert_allclose(model.contrast_coefficients_[:3], 1.0 / 3.0)
    np.testing.assert_allclose(model.contrast_coefficients_[3:], -1.0 / 2.0)
    assert model.group_a_mode_statistic_.shape == (3,)
    assert model.group_b_mode_statistic_.shape == (3,)
    assert model.contrast_mode_statistic_[0] > 0


def test_differential_cross_covariance_matches_explicit_group_mean_contrast():
    Xs, coords, targets, neighbours, _, _ = make_differential_cross_samples(
        seed=171, group_a=3, group_b=2, n_genes=20
    )
    model = DifferentialCrossSpatialProgramModel(
        1.8,
        positive_samples=[0, 1, 2],
        negative_samples=[3, 4],
        objective="covariance",
    ).fit(
        Xs,
        coords,
        4,
        target_masks=targets,
        neighbour_masks=neighbours,
    )

    explicit = np.zeros((20, 20))
    for coefficient, sample in zip(model.sample_coefficients_, model.samples):
        target = sample.W_target.toarray()
        neighbour = sample.W_neighbour.toarray()
        target -= target.mean(axis=0, keepdims=True)
        neighbour -= neighbour.mean(axis=0, keepdims=True)
        explicit += coefficient * target.T @ (sample.K @ neighbour)
    expected = np.linalg.svd(explicit, compute_uv=False)[:4]
    np.testing.assert_allclose(model.singular_values, expected, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(
        model.contrast_mode_statistic_, model.singular_values, rtol=1e-6, atol=1e-8
    )


def test_differential_cross_supports_all_objectives_and_permutations():
    Xs, coords, targets, neighbours, _, _ = make_differential_cross_samples(
        seed=172, group_a=2, group_b=2, n_genes=24
    )
    for objective in ("covariance", "gene_standardized", "gain"):
        model = DifferentialCrossSpatialProgramModel(
            1.5,
            positive_samples=[0, 1],
            negative_samples=[2, 3],
            objective=objective,
            expression_rank=16,
        ).fit(
            Xs,
            coords,
            2,
            target_masks=targets,
            neighbour_masks=neighbours,
        )
        assert model.singular_values.shape == (2,)
        assert model.sample_mode_statistic_.shape == (4, 2)
        pvalues = model.permutation_test(4, seed=3)
        assert pvalues.shape == (2,)
        assert np.all((pvalues > 0) & (pvalues <= 1))
        assert model.permutation_max_singular_values_.shape == (4,)


def test_differential_cross_preserves_common_geometry_reference():
    Xs, coords, targets, neighbours, _, _ = make_differential_cross_samples(
        seed=173, group_a=2, group_b=2, n_genes=18
    )
    model = DifferentialCrossSpatialProgramModel(
        1.7,
        positive_samples=[0, 1],
        negative_samples=[2, 3],
        geometry_normalisation="distance",
        distance_bins=5,
    ).fit(
        Xs,
        coords,
        2,
        target_masks=targets,
        neighbour_masks=neighbours,
    )
    reference = model.geometry_diagnostics_[0]["distance_reference_weights"]
    for diagnostic in model.geometry_diagnostics_[1:]:
        np.testing.assert_allclose(diagnostic["distance_reference_weights"], reference)
