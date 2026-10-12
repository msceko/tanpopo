from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tanpopo.inference import effective_dimension, select_modes
from tanpopo.geometry import neighbour_spacing
from tanpopo.models import SpatialProgramModel


@dataclass(frozen=True)
class SpatialInformationResult:
    """Spatial-gain spectrum and permutation-derived information summaries."""

    radius: float
    eigenvalues: np.ndarray
    pvalues: np.ndarray
    null_maxima: np.ndarray
    alpha: float
    selected_modes: int
    effective_dimension: float
    spectral_gain: float
    maximum_gain: float
    threshold: float
    graph: dict

    def summary(self):
        scalar_graph = {
            key: value
            for key, value in self.graph.items()
            if np.isscalar(value) and not isinstance(value, (str, bytes))
        }
        return {
            "radius": float(self.radius),
            "maximum_gain": float(self.maximum_gain),
            "spectral_gain": float(self.spectral_gain),
            "effective_spatial_dimension": float(self.effective_dimension),
            "selected_modes": int(self.selected_modes),
            "mode_alpha": float(self.alpha),
            "null_threshold": float(self.threshold),
            "n_permutations": int(self.null_maxima.size),
            **scalar_graph,
        }


@dataclass(frozen=True)
class SpatialInformationCurveResult:
    results: tuple[SpatialInformationResult, ...]

    def best(self, metric="spectral_gain"):
        if not self.results:
            raise ValueError("No spatial-information results are available")
        values = np.asarray([getattr(result, metric) for result in self.results], dtype=float)
        if not np.any(np.isfinite(values)):
            raise ValueError(f"Metric {metric!r} is non-finite for every radius")
        return self.results[int(np.nanargmax(values))]

    def summary_frame(self):
        import pandas as pd

        return pd.DataFrame([result.summary() for result in self.results])


def spatial_information_from_arrays(
    X,
    coords,
    radius=None,
    *,
    labels=None,
    covariates=None,
    mask=None,
    max_components=30,
    expression_rank=100,
    gain_ridge=1e-6,
    graph_normalisation="symmetric",
    geometry_normalisation="distance",
    distance_bins=10,
    spot_operator="sample",
    permutations=200,
    alpha=0.05,
    seed=0,
    dtype=np.float64,
):
    """Quantify spatial information with Tanpopo's native gain objective.

    This intentionally reuses the same geometry, residualisation, gain operator and
    max-statistic permutation machinery as program fitting. The returned effective
    dimension is the participation ratio of gain above the FWER null threshold.
    """
    coords = np.asarray(coords, dtype=float)
    if not 0 < float(alpha) < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if radius is None:
        radius = 3.5 * neighbour_spacing(coords if mask is None else coords[np.asarray(mask)])
    model = SpatialProgramModel(
        radius,
        objective="gain",
        spot_operator=spot_operator,
        expression_rank=expression_rank,
        gain_ridge=gain_ridge,
        graph_normalisation=graph_normalisation,
        geometry_normalisation=geometry_normalisation,
        distance_bins=distance_bins,
        dtype=dtype,
    ).fit(
        X,
        coords,
        max_components,
        labels=labels,
        covariates=covariates,
        masks=mask,
    )
    if int(permutations) <= 0:
        raise ValueError("spatial information requires at least one permutation")
    if 1.0 / (int(permutations) + 1.0) > float(alpha):
        minimum = int(np.ceil(1.0 / float(alpha) - 1.0))
        raise ValueError(
            f"alpha={alpha:g} requires at least {minimum} permutations for "
            "sufficient p-value resolution"
        )
    model.permutation_test(permutations, seed=seed)
    selection = select_modes(model.eigenvalues, model.permutation_max_eigenvalues_, alpha=alpha)
    corrected = np.maximum(model.eigenvalues - selection.threshold, 0.0)
    return SpatialInformationResult(
        radius=float(radius),
        eigenvalues=np.asarray(model.eigenvalues),
        pvalues=np.asarray(selection.pvalues),
        null_maxima=np.asarray(selection.null_maxima),
        alpha=float(alpha),
        selected_modes=int(selection.selected_modes),
        effective_dimension=float(effective_dimension(corrected)),
        spectral_gain=float(corrected.sum()),
        maximum_gain=float(model.eigenvalues[0]) if len(model.eigenvalues) else float("nan"),
        threshold=float(selection.threshold),
        graph=dict(model.geometry_diagnostics_[0]),
    )


def spatial_information_curve_from_arrays(X, coords, radii, **kwargs):
    """Evaluate matched spatial-information summaries over candidate radii."""
    radii = tuple(float(radius) for radius in radii)
    if not radii or any(radius <= 0 for radius in radii):
        raise ValueError("radii must contain positive values")
    results = tuple(
        spatial_information_from_arrays(X, coords, radius=radius, **kwargs) for radius in radii
    )
    return SpatialInformationCurveResult(results)
