from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from tanpopo.version import __version__


def _root(adata):
    return adata.uns.setdefault("tanpopo", {})


def _analysis(adata, analysis_id):
    root = _root(adata)
    record = root.setdefault(str(analysis_id), {})
    record["tanpopo_version"] = __version__
    return record


def register_program_result(
    adata,
    analysis_id,
    *,
    loadings_key,
    modes_key,
    values,
    value_name="eigenvalue",
    gene_scores_key=None,
    gene_covariance_key=None,
    variant="default",
    analysis_type="program",
):
    record = _analysis(adata, analysis_id)
    record.setdefault("analysis_type", analysis_type)
    variants = record.setdefault("results", {})
    variants[str(variant)] = {
        "kind": "program",
        "loadings_key": str(loadings_key),
        "modes_key": str(modes_key),
        "gene_scores_key": None if gene_scores_key is None else str(gene_scores_key),
        "gene_covariance_key": (None if gene_covariance_key is None else str(gene_covariance_key)),
        "value_name": str(value_name),
        "values": np.asarray(values, dtype=float),
    }
    return record


def register_cross_result(
    adata,
    analysis_id,
    *,
    target_loadings_key,
    neighbour_loadings_key,
    target_modes_key,
    neighbour_modes_key,
    singular_values,
    analysis_type="cross",
):
    record = _analysis(adata, analysis_id)
    record["analysis_type"] = analysis_type
    record["results"] = {
        "default": {
            "kind": "cross",
            "target_loadings_key": str(target_loadings_key),
            "neighbour_loadings_key": str(neighbour_loadings_key),
            "target_modes_key": str(target_modes_key),
            "neighbour_modes_key": str(neighbour_modes_key),
            "value_name": "singular_value",
            "values": np.asarray(singular_values, dtype=float),
        }
    }
    return record


@dataclass(frozen=True)
class ProgramResult:
    analysis_id: str
    variant: str
    values: np.ndarray
    value_name: str
    gene_loadings: np.ndarray
    spot_modes: np.ndarray | None
    gene_names: np.ndarray
    gene_scores: np.ndarray | None = None
    gene_spatial_covariance: np.ndarray | None = None
    metadata: dict[str, Any] | None = None

    @property
    def n_modes(self):
        return int(self.gene_loadings.shape[1])

    def loadings_frame(self):
        columns = [
            f"Mode {i} ({self.value_name} = {value:.6e})" for i, value in enumerate(self.values)
        ]
        return pd.DataFrame(self.gene_loadings, index=self.gene_names, columns=columns)


@dataclass(frozen=True)
class CrossProgramResult:
    analysis_id: str
    values: np.ndarray
    target_loadings: np.ndarray
    neighbour_loadings: np.ndarray
    target_modes: np.ndarray | None
    neighbour_modes: np.ndarray | None
    gene_names: np.ndarray
    metadata: dict[str, Any] | None = None

    @property
    def n_modes(self):
        return int(self.target_loadings.shape[1])

    def target_frame(self):
        columns = [
            f"Mode {i} (singular_value = {value:.6e})" for i, value in enumerate(self.values)
        ]
        return pd.DataFrame(self.target_loadings, index=self.gene_names, columns=columns)

    def neighbour_frame(self):
        columns = [
            f"Mode {i} (singular_value = {value:.6e})" for i, value in enumerate(self.values)
        ]
        return pd.DataFrame(self.neighbour_loadings, index=self.gene_names, columns=columns)


def list_results(adata):
    root = adata.uns.get("tanpopo", {})
    output = []
    for analysis_id, record in root.items():
        if not isinstance(record, dict) or "results" not in record:
            continue
        for variant, spec in record["results"].items():
            output.append(
                {
                    "analysis_id": str(analysis_id),
                    "analysis_type": record.get("analysis_type", "unknown"),
                    "variant": str(variant),
                    "kind": spec.get("kind", "unknown"),
                    "n_modes": int(len(spec.get("values", []))),
                }
            )
    return output


def _get_variant(record, variant):
    results = record.get("results", {})
    if not results:
        raise KeyError("Tanpopo analysis does not contain a result record")
    if variant is None:
        if len(results) != 1:
            choices = ", ".join(results)
            raise ValueError(f"Analysis has multiple variants; choose one of: {choices}")
        variant = next(iter(results))
    if variant not in results:
        raise KeyError(f"Unknown result variant {variant!r}")
    return str(variant), results[variant]


def load_result(adata, analysis_id, variant=None):
    root = adata.uns.get("tanpopo", {})
    if analysis_id not in root:
        raise KeyError(f"Tanpopo analysis {analysis_id!r} was not found")
    record = root[analysis_id]
    variant, spec = _get_variant(record, variant)
    values = np.asarray(spec["values"], dtype=float)
    gene_names = np.asarray(adata.var_names).astype(str)
    if spec["kind"] == "program":
        gene_scores_key = spec.get("gene_scores_key")
        gene_covariance_key = spec.get("gene_covariance_key")
        modes_key = spec.get("modes_key")
        return ProgramResult(
            analysis_id=str(analysis_id),
            variant=variant,
            values=values,
            value_name=str(spec.get("value_name", "eigenvalue")),
            gene_loadings=np.asarray(adata.varm[spec["loadings_key"]]),
            spot_modes=(
                None
                if modes_key is None or modes_key not in adata.obsm
                else np.asarray(adata.obsm[modes_key])
            ),
            gene_names=gene_names,
            gene_scores=(
                None if gene_scores_key is None else np.asarray(adata.varm[gene_scores_key])
            ),
            gene_spatial_covariance=(
                None if gene_covariance_key is None else np.asarray(adata.var[gene_covariance_key])
            ),
            metadata=record,
        )
    if spec["kind"] == "cross":
        target_modes_key = spec.get("target_modes_key")
        neighbour_modes_key = spec.get("neighbour_modes_key")
        return CrossProgramResult(
            analysis_id=str(analysis_id),
            values=values,
            target_loadings=np.asarray(adata.varm[spec["target_loadings_key"]]),
            neighbour_loadings=np.asarray(adata.varm[spec["neighbour_loadings_key"]]),
            target_modes=(
                None
                if target_modes_key is None or target_modes_key not in adata.obsm
                else np.asarray(adata.obsm[target_modes_key])
            ),
            neighbour_modes=(
                None
                if neighbour_modes_key is None or neighbour_modes_key not in adata.obsm
                else np.asarray(adata.obsm[neighbour_modes_key])
            ),
            gene_names=gene_names,
            metadata=record,
        )
    raise ValueError(f"Unknown Tanpopo result kind {spec['kind']!r}")
