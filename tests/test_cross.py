import numpy as np

from tanpopo.models import CrossSpatialProgramModel


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
