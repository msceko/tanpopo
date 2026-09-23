import numpy as np

from tanpopo.models import LabelDecompositionModel


def test_label_decomposition_is_operator_additive():
    rng = np.random.default_rng(55)
    n, g = 160, 25
    coords = np.c_[np.linspace(0, 12, n), np.zeros(n)]
    labels = np.where(np.arange(n) < n // 2, "A", "B")
    X = rng.normal(size=(n, g))
    X[labels == "B", :4] += 1.5
    model = LabelDecompositionModel(1.2).fit(X, coords, labels, 3)
    assert model.operator_additivity_error() < 1e-10
    assert set(model.results_) == {"total", "between", "within", "coupling"}


def test_label_decomposition_is_additive_for_variogram():
    rng = np.random.default_rng(56)
    n, g = 140, 20
    coords = np.c_[np.linspace(0, 10, n), np.zeros(n)]
    labels = np.where(np.arange(n) < n // 2, "A", "B")
    X = rng.normal(size=(n, g))
    model = LabelDecompositionModel(
        1.0,
        spatial_statistic="variogram",
        geometry_normalisation="distance",
        distance_bins=4,
    ).fit(X, coords, labels, 3)
    assert model.operator_additivity_error() < 1e-10
