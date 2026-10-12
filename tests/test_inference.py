import numpy as np

from tanpopo.diagnostics import spatial_information_from_arrays
from tanpopo.inference import effective_dimension, select_modes
from tanpopo.models import CrossSpatialProgramModel, SpatialProgramModel


def test_max_statistic_selection_and_effective_dimension():
    observed = np.array([5.0, 3.0, 0.5])
    null = np.array([1.0, 1.5, 2.0, 2.5, 1.2, 1.8, 2.1, 1.7, 2.2, 1.9])
    selection = select_modes(observed, null, alpha=0.2)
    assert selection.selected_modes == 2
    assert selection.pvalues[0] <= selection.pvalues[1] < selection.pvalues[2]
    assert selection.effective_dimension > 1.0
    assert effective_dimension([1, 1]) == 2.0


def make_spatial(seed=1, n=140, g=18):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 14, n)
    coords = np.c_[x, np.zeros(n)]
    v = np.zeros(g)
    v[:4] = 1.0
    v /= np.linalg.norm(v)
    X = rng.normal(scale=0.2, size=(n, g)) + 3.0 * np.sin(x / 2)[:, None] * v
    return X, coords


def test_spatial_model_permutation_selection_recovers_planted_mode():
    X, coords = make_spatial()
    model = SpatialProgramModel(1.5).fit(X, coords, 5)
    pvalues = model.permutation_test(39, seed=1)
    selection = model.select_modes(alpha=0.05)
    assert pvalues[0] <= 0.05
    assert selection.selected_modes >= 1
    model.truncate_modes(selection.selected_modes)
    assert model.gene_loadings.shape[1] == selection.selected_modes
    assert model.spot_modes[0].shape[1] == selection.selected_modes


def test_cross_permutation_selection_recovers_planted_pair():
    rng = np.random.default_rng(2)
    n, g = 180, 20
    x = np.linspace(0, 18, n)
    coords = np.c_[x, np.zeros(n)]
    target = np.arange(n) % 2 == 0
    neighbour = ~target
    vt = np.zeros(g)
    vu = np.zeros(g)
    vt[:4] = 1
    vu[8:12] = 1
    vt /= np.linalg.norm(vt)
    vu /= np.linalg.norm(vu)
    z = np.sin(x / 2)
    X = rng.normal(scale=0.2, size=(n, g))
    X[target] += 3 * z[target, None] * vt
    X[neighbour] += 3 * z[neighbour, None] * vu
    model = CrossSpatialProgramModel(1.5).fit(X, coords, 4, target, neighbour)
    model.permutation_test(39, seed=1)
    selection = model.select_modes(alpha=0.05)
    assert selection.selected_modes >= 1


def test_spatial_information_reuses_native_gain_and_inference():
    X, coords = make_spatial(seed=3, n=120, g=16)
    result = spatial_information_from_arrays(
        X,
        coords,
        radius=1.5,
        max_components=5,
        expression_rank=10,
        permutations=19,
        alpha=0.1,
        seed=0,
    )
    summary = result.summary()
    assert summary["selected_modes"] >= 1
    assert summary["effective_spatial_dimension"] >= 1.0
    assert np.isfinite(summary["spectral_gain"])
    assert summary["n_permutations"] == 19


def test_gain_permutations_reuse_square_expression_basis(monkeypatch):
    X, coords = make_spatial(seed=4, n=90, g=14)
    model = SpatialProgramModel(1.5, objective="gain", expression_rank=8).fit(
        X, coords, 4
    )

    def fail(*args, **kwargs):
        raise AssertionError("projected_svd should not be recomputed during permutations")

    monkeypatch.setattr("tanpopo.models.projected_svd", fail)
    model.permutation_test(3, seed=0)
    assert model.permutation_max_eigenvalues_.shape == (3,)


def test_gain_permutations_reuse_cross_expression_basis(monkeypatch):
    rng = np.random.default_rng(5)
    n, g = 120, 16
    x = np.linspace(0, 12, n)
    coords = np.c_[x, np.zeros(n)]
    target = np.arange(n) % 2 == 0
    neighbour = ~target
    X = rng.normal(scale=0.3, size=(n, g))
    X[target, :3] += np.sin(x[target, None])
    X[neighbour, 6:9] += np.sin(x[neighbour, None])
    model = CrossSpatialProgramModel(1.5, objective="gain", expression_rank=8).fit(
        X, coords, 3, target, neighbour
    )

    def fail(*args, **kwargs):
        raise AssertionError("projected_svd should not be recomputed during permutations")

    monkeypatch.setattr("tanpopo.models.projected_svd", fail)
    model.permutation_test(3, seed=0)
    assert model.permutation_max_singular_values_.shape == (3,)
