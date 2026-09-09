import numpy as np

from tanpopo.models import DifferentialSpatialProgramModel


def cosine(a, b):
    return abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_differential_program_detects_spatial_reorganisation():
    rng = np.random.default_rng(30)
    g = 35
    v = np.zeros(g)
    v[:6] = rng.normal(size=6)
    v /= np.linalg.norm(v)
    Ws, coords = [], []
    for spatial in [True, True, False, False]:
        n = 260
        x = np.linspace(0, 20, n)
        xy = np.c_[x, np.zeros(n)]
        z = np.sin(x / 2) if spatial else rng.normal(size=n)
        X = rng.normal(scale=0.35, size=(n, g)) + 2.5 * z[:, None] * v
        Ws.append(X)
        coords.append(xy)
    model = DifferentialSpatialProgramModel(
        1.5,
        positive_samples=[0, 1],
        negative_samples=[2, 3],
        objective="covariance",
    ).fit(Ws, coords, 3)
    best = max(cosine(v, model.gene_loadings[:, j]) for j in range(3))
    assert best > 0.75
    assert np.any(model.eigenvalues > 0)


def test_differential_permutation_test_runs_at_sample_level():
    rng = np.random.default_rng(31)
    g = 20
    Ws, coords = [], []
    for i in range(4):
        n = 100
        x = np.linspace(0, 10, n)
        coords.append(np.c_[x, np.zeros(n)])
        Ws.append(rng.normal(size=(n, g)))
    model = DifferentialSpatialProgramModel(
        1.0,
        positive_samples=[0, 1],
        negative_samples=[2, 3],
        objective="covariance",
    ).fit(Ws, coords, 2)
    p = model.permutation_test(4, seed=1)
    assert p.shape == (2,)
    assert np.all((p > 0) & (p <= 1))
    assert model.permutation_max_abs_eigenvalues_.shape == (4,)
