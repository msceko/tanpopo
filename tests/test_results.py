import numpy as np
import pandas as pd

from tanpopo.io import (
    add_metadata,
    geometry_diagnostics_cfg,
    model_cfg,
    store_multi_sample_cross_result,
    store_sample_result,
)
from tanpopo.models import DifferentialCrossSpatialProgramModel, SpatialProgramModel
from tanpopo.results import CrossProgramResult, ProgramResult, load_result


class DummyAnnData:
    def __init__(self, n_obs, n_vars):
        self.n_obs = n_obs
        self.n_vars = n_vars
        self.var = pd.DataFrame(index=pd.Index([f"g{i}" for i in range(n_vars)]))
        self.varm = {}
        self.obsm = {}
        self.uns = {}

    @property
    def var_names(self):
        return self.var.index


def test_program_result_schema_round_trip_in_memory():
    rng = np.random.default_rng(200)
    n, g = 50, 8
    coords = np.c_[np.linspace(0, 8, n), np.zeros(n)]
    X = rng.normal(size=(n, g))
    model = SpatialProgramModel(1.0, geometry_normalisation="distance", distance_bins=4).fit(
        X, coords, 2
    )
    adata = DummyAnnData(n, g)
    store_sample_result(adata, model, "test")

    record = adata.uns["tanpopo"]["test"]
    assert record["tanpopo_version"] == "0.3.0"
    assert record["analysis_type"] == "program"
    assert "tanpopo_test_gene_spatial_covariance" in adata.var
    assert "tanpopo_test_gene_spatial_statistic" not in adata.var

    result = load_result(adata, "test")
    assert isinstance(result, ProgramResult)
    np.testing.assert_allclose(result.values, model.eigenvalues)
    np.testing.assert_allclose(result.gene_loadings, model.gene_loadings)
    assert result.gene_names.tolist() == list(adata.var_names)

    cfg = model_cfg(model)
    assert "spatial_statistic" not in cfg
    assert "null_center" not in cfg
    assert cfg["geometry_normalisation"] == "distance"
    diagnostics = geometry_diagnostics_cfg(model)
    assert np.isclose(diagnostics["sample_0"]["pair_mass"], n)


def test_store_differential_cross_registers_cross_schema(monkeypatch):
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
    ).fit(Xs, coords, 2, target_masks=targets, neighbour_masks=neighbours)
    adatas = [DummyAnnData(len(X), X.shape[1]) for X in Xs]
    combined = DummyAnnData(sum(len(X) for X in Xs), Xs[0].shape[1])
    monkeypatch.setattr("tanpopo.io.concat_adata_samples", lambda adatas, sample_names: combined)
    stored = store_multi_sample_cross_result(adatas, ["a1", "a2", "b1", "b2"], model, "diff_cross")
    result = load_result(stored, "diff_cross")
    assert isinstance(result, CrossProgramResult)
    np.testing.assert_allclose(result.values, model.singular_values)
    np.testing.assert_allclose(result.target_loadings, model.target_loadings)
    np.testing.assert_allclose(result.neighbour_loadings, model.neighbour_loadings)


def test_metadata_is_scoped_to_each_result_variant():
    adata = DummyAnnData(5, 3)
    add_metadata(
        adata,
        "multi",
        {"layer": "logcounts"},
        {"radius": 10.0},
        {"conditioned_label": "A"},
        variant="A",
        analysis_type="program",
    )
    add_metadata(
        adata,
        "multi",
        {"layer": "logcounts"},
        {"radius": 20.0},
        {"conditioned_label": "B"},
        variant="B",
        analysis_type="program",
    )
    record = adata.uns["tanpopo"]["multi"]
    assert "model" not in record
    assert record["result_metadata"]["A"]["model"]["radius"] == 10.0
    assert record["result_metadata"]["B"]["model"]["radius"] == 20.0
    assert record["result_metadata"]["A"]["conditioned_label"] == "A"
    assert record["result_metadata"]["B"]["conditioned_label"] == "B"
