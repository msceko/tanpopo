import numpy as np

from tanpopo.models import SpatialProgramModel


def cosine(a, b):
    return abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_hard_mask_exposes_within_type_spatial_program():
    rng = np.random.default_rng(10)
    n, g = 500, 45
    coords = np.c_[np.linspace(0, 25, n), np.zeros(n)]
    labels = np.where(np.arange(n) % 2 == 0, "target", "other")
    target = labels == "target"
    v = np.zeros(g)
    v[:6] = rng.normal(size=6)
    v /= np.linalg.norm(v)
    nuisance = np.zeros(g)
    nuisance[15:25] = rng.normal(size=10)
    nuisance /= np.linalg.norm(nuisance)
    X = rng.normal(scale=0.35, size=(n, g))
    X += 4 * np.sin(coords[:, 0] / 1.5)[:, None] * nuisance
    X[target] += 2.5 * np.sin(coords[target, 0] / 3)[:, None] * v

    model = SpatialProgramModel(1.8, objective="covariance").fit(
        X, coords, 3, masks=target
    )
    best = max(cosine(v, model.gene_loadings[:, j]) for j in range(3))
    assert best > 0.8
    assert model.samples[0].n_spots == target.sum()
