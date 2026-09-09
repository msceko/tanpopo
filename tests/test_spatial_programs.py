import numpy as np

from tanpopo.models import SpatialProgramModel


def cosine(a, b):
    return abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def make_spatial(seed=1):
    rng = np.random.default_rng(seed)
    n, g = 400, 40
    x = np.linspace(0, 20, n)
    coords = np.c_[x, np.zeros(n)]
    v = np.zeros(g)
    v[:8] = rng.normal(size=8)
    v /= np.linalg.norm(v)
    z = np.sin(x / 2.0)
    X = rng.normal(scale=0.35, size=(n, g)) + 2.5 * z[:, None] * v[None, :]
    return X, coords, v


def test_covariance_recovers_spatial_program():
    X, coords, v = make_spatial()
    model = SpatialProgramModel(
        1.5, objective="covariance", graph_normalisation="symmetric"
    ).fit(X, coords, 3)
    best = max(cosine(v, model.gene_loadings[:, j]) for j in range(3))
    assert best > 0.85


def test_gain_recovers_spatial_program_despite_large_nonspatial_variance():
    rng = np.random.default_rng(3)
    n, g = 450, 50
    x = np.linspace(0, 20, n)
    coords = np.c_[x, np.zeros(n)]
    v_sp = np.zeros(g)
    v_sp[:7] = rng.normal(size=7)
    v_sp /= np.linalg.norm(v_sp)
    v_ns = np.zeros(g)
    v_ns[10:18] = rng.normal(size=8)
    v_ns /= np.linalg.norm(v_ns)
    z_sp = np.sin(x / 2)
    z_ns = rng.normal(size=n)
    X = (
        rng.normal(scale=0.25, size=(n, g))
        + 1.2 * z_sp[:, None] * v_sp
        + 5.0 * z_ns[:, None] * v_ns
    )
    model = SpatialProgramModel(
        1.5,
        objective="gain",
        expression_rank=20,
        graph_normalisation="symmetric",
    ).fit(X, coords, 3)
    best = max(cosine(v_sp, model.gene_loadings[:, j]) for j in range(3))
    assert best > 0.6
