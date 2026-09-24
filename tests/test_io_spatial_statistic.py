import numpy as np
import pandas as pd

from tanpopo.io import (
    geometry_diagnostics_cfg,
    model_cfg,
    store_multi_sample_cross_result,
    store_sample_result,
)
from tanpopo.models import DifferentialCrossSpatialProgramModel, SpatialProgramModel


class DummyAnnData:
    def __init__(self, n_obs, n_vars):
        self.n_obs = n_obs
        self.n_vars = n_vars
        self.var = pd.DataFrame(index=np.arange(n_vars))
        self.varm = {}
        self.obsm = {}
        self.uns = {}


def _fit(statistic):
    rng = np.random.default_rng(200)
    n, g = 50, 8
    coords = np.c_[np.linspace(0, 8, n), np.zeros(n)]
    X = rng.normal(size=(n, g))
    return SpatialProgramModel(
        1.0,
        spatial_statistic=statistic,
        geometry_normalisation="distance",
        distance_bins=4,
    ).fit(X, coords, 2)


def test_store_variogram_uses_generic_and_specific_statistic_keys():
    model = _fit("variogram")
    adata = DummyAnnData(model.samples[0].n_spots, model.samples[0].n_genes)
    store_sample_result(adata, model, "test")
    assert "tanpopo_test_gene_spatial_statistic" in adata.var
    assert "tanpopo_test_gene_mark_variogram" in adata.var
    assert "tanpopo_test_gene_spatial_covariance" not in adata.var


def test_store_mark_correlation_retains_compatibility_key_and_metadata():
    model = _fit("mark_correlation")
    adata = DummyAnnData(model.samples[0].n_spots, model.samples[0].n_genes)
    store_sample_result(adata, model, "test")
    assert "tanpopo_test_gene_spatial_statistic" in adata.var
    assert "tanpopo_test_gene_spatial_covariance" in adata.var
    cfg = model_cfg(model)
    assert cfg["spatial_statistic"] == "mark_correlation"
    assert cfg["geometry_normalisation"] == "distance"
    diagnostics = geometry_diagnostics_cfg(model)
    assert np.isclose(diagnostics["sample_0"]["pair_mass"], model.samples[0].n_spots)


def test_store_differential_cross_result_includes_group_mode_statistics(monkeypatch):
    rng = np.random.default_rng(201)
    Xs, coords, targets, neighbours = [], [], [], []
    for i in range(4):
        n = 60 + 10 * i
        x = np.linspace(0, 8, n)
        target = np.arange(n) % 2 == 0
        neighbour = ~target
        X = rng.normal(size=(n, 7))
        if i < 2:
            field = np.sin(x)
            X[target, 0] += field[target]
            X[neighbour, 1] += field[neighbour]
        Xs.append(X)
        coords.append(np.c_[x, np.zeros(n)])
        targets.append(target)
        neighbours.append(neighbour)
    model = DifferentialCrossSpatialProgramModel(
        1.2,
        positive_samples=[0, 1],
        negative_samples=[2, 3],
        geometry_normalisation="mass",
    ).fit(
        Xs,
        coords,
        2,
        target_masks=targets,
        neighbour_masks=neighbours,
    )
    adatas = [DummyAnnData(len(X), X.shape[1]) for X in Xs]
    combined = DummyAnnData(sum(len(X) for X in Xs), Xs[0].shape[1])
    monkeypatch.setattr(
        "tanpopo.io.concat_adata_samples", lambda adatas, sample_names: combined
    )
    result = store_multi_sample_cross_result(
        adatas, ["a1", "a2", "b1", "b2"], model, "diff_cross"
    )
    stored = result.uns["tanpopo"]["diff_cross"]
    for key in (
        "singular_values",
        "sample_coefficients",
        "base_sample_weights",
        "contrast_coefficients",
        "sample_mode_covariance",
        "sample_mode_statistic",
        "aggregate_mode_statistic",
        "contrast_mode_statistic",
        "group_a_mode_statistic",
        "group_b_mode_statistic",
    ):
        assert key in stored
    assert "tanpopo_diff_cross_target_loadings" in result.varm
    assert "tanpopo_diff_cross_neighbour_loadings" in result.varm
