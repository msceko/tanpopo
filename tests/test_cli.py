import pytest
import typer
from typer.testing import CliRunner

from tanpopo.workflows import _apply_mode_inference, app


runner = CliRunner()


def test_release_cli_lists_core_and_utility_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in (
        "programs",
        "shared",
        "differential",
        "cross",
        "cross-shared",
        "cross-differential",
        "decompose-labels",
        "baseline-pca",
        "inspect",
        "preprocess",
        "export",
        "compare",
        "compare-spots",
        "plot",
        "cluster",
        "gsea",
        "spatial-info",
    ):
        assert command in result.stdout


def test_program_help_exposes_auto_selection_without_removed_experiments():
    result = runner.invoke(app, ["programs", "--help"])
    assert result.exit_code == 0
    assert "--components" in result.stdout
    assert "--max-components" in result.stdout
    assert "--mode-alpha" in result.stdout
    assert "--permutations" in result.stdout
    assert "Geometry balancing" in result.stdout
    for removed in ("--spatial-statistic", "--null-center", "--alpha", "variogram"):
        assert removed not in result.stdout


def test_cross_help_uses_covariance_geometry_surface_only():
    result = runner.invoke(app, ["cross", "--help"])
    assert result.exit_code == 0
    assert "--target-labels" in result.stdout
    assert "--neighbour-labels" in result.stdout
    assert "Geometry balancing" in result.stdout
    assert "--components" in result.stdout
    assert "--spatial-statistic" not in result.stdout


def test_auto_mode_selection_rejects_insufficient_permutation_resolution():
    with pytest.raises(typer.BadParameter, match="at least 19 permutations"):
        _apply_mode_inference(object(), True, 10, 0.05, 0)
