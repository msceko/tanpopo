import numpy as np
import pandas as pd

from tanpopo.io import geometry_diagnostics_cfg, model_cfg, store_sample_result
from tanpopo.models import SpatialProgramModel


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
