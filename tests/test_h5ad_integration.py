from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

ad = pytest.importorskip("anndata")
from tanpopo.results import ProgramResult, load_result
from tanpopo.workflows import app


runner = CliRunner()


def _example_adata(seed=500):
    rng = np.random.default_rng(seed)
    n, g = 48, 12
    x = np.linspace(0, 8, n)
    X = rng.poisson(2.0, size=(n, g)).astype(float)
    X[:, :3] += 2.0 * (1 + np.sin(x))[:, None]
    adata = ad.AnnData(X=X)
    adata.var_names = [f"g{i}" for i in range(g)]
    adata.obs_names = [f"c{i}" for i in range(n)]
    adata.obsm["spatial"] = np.c_[x, np.zeros(n)]
    return adata


def test_program_cli_h5ad_round_trip_and_export(tmp_path: Path):
    input_path = tmp_path / "input.h5ad"
    output_path = tmp_path / "output.h5ad"
    csv_path = tmp_path / "loadings.csv"
    _example_adata().write_h5ad(input_path)

    fitted = runner.invoke(
        app,
        [
            "programs",
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "--analysis-id",
            "test",
            "--radius",
            "1.5",
            "--components",
            "3",
            "--geometry-normalisation",
            "mass",
            "--min-counts",
            "0",
        ],
    )
    assert fitted.exit_code == 0, fitted.output

    stored = ad.read_h5ad(output_path)
    result = load_result(stored, "test")
    assert isinstance(result, ProgramResult)
    assert result.gene_loadings.shape == (stored.n_vars, 3)
    assert result.spot_modes.shape == (stored.n_obs, 3)

    exported = runner.invoke(
        app,
        [
            "export",
            "-i",
            str(output_path),
            "-o",
            str(csv_path),
            "--analysis-id",
            "test",
        ],
    )
    assert exported.exit_code == 0, exported.output
    assert csv_path.exists()
    assert csv_path.read_text().startswith("gene,Mode 0")


def test_baseline_pca_uses_schema_without_scanpy(tmp_path: Path):
    input_path = tmp_path / "input_pca.h5ad"
    output_path = tmp_path / "pca.h5ad"
    _example_adata(seed=501).write_h5ad(input_path)
    fitted = runner.invoke(
        app,
        [
            "baseline-pca",
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "--analysis-id",
            "pca",
            "--components",
            "4",
            "--min-counts",
            "0",
        ],
    )
    assert fitted.exit_code == 0, fitted.output
    stored = ad.read_h5ad(output_path)
    result = load_result(stored, "pca")
    assert isinstance(result, ProgramResult)
    assert result.value_name == "variance"
    assert result.gene_loadings.shape == (stored.n_vars, 4)
    assert result.spot_modes.shape == (stored.n_obs, 4)
    variance_ratio = stored.uns["tanpopo"]["pca"]["result_metadata"]["default"]["variance_ratio"]
    assert np.all(np.asarray(variance_ratio) >= 0)
    assert np.asarray(variance_ratio).sum() <= 1.0 + 1e-8
