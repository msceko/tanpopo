import numpy as np

from tanpopo.models import CrossSpatialProgramModel, SharedCrossSpatialProgramModel


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
