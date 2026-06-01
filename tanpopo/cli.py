from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer


class TransformTypes(str, Enum):
    sqrt = "sqrt"
    log1p = "log1p"


class SpotOperatorTypes(str, Enum):
    none = "none"
    sample = "sample"
    label = "label"


class CovariateTypes(str, Enum):
    log_total_counts = "log_total_counts"
    log_detected_genes = "log_detected_genes"
    mito_fraction = "mito_fraction"
    ribo_fraction = "ribo_fraction"


class ClusterTypes(str, Enum):
    spots = "spots"
    genes = "genes"


InputPath = Annotated[
    Path,
    typer.Option(
        "--input",
        "-i",
        help="Input .h5ad file.",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
]
OutputPath = Annotated[
    Path | None,
    typer.Option(
        "--output",
        "-o",
        help="Optional output .h5ad file.",
        dir_okay=False,
        writable=True,
    ),
]
Layer = Annotated[
    str | None,
    typer.Option("--layer", help="AnnData layer to use instead of X."),
]
LabelKey = Annotated[
    str | None,
    typer.Option("--label-key", help="obs column defining labels/cell types."),
]
Components = Annotated[
    int,
    typer.Option("--components", "-k", help="Number of spatial components."),
]
Radius = Annotated[
    float,
    typer.Option("--radius", "-r", help="Wendland kernel support radius."),
]
Alpha = Annotated[
    float,
    typer.Option("--alpha", help="Gene magnitude scaling exponent."),
]
Transform = Annotated[
    TransformTypes | None,
    typer.Option("--transform", help="Counts transform after normalisation."),
]
MinCounts = Annotated[
    int | None,
    typer.Option("--min-counts", help="Minimum counts per gene."),
]
MinSpotFraction = Annotated[
    float | None,
    typer.Option("--min-spot-fraction", help="Minimum fraction of spots per gene."),
]
TargetSum = Annotated[
    float | None,
    typer.Option("--target-sum", help="Per-spot normalisation target. Use 0 to disable."),
]
Covariates = Annotated[
    list[CovariateTypes] | None,
    typer.Option(
        "--covariate",
        "--covariates",
        help="Covariate to correct for. Can be supplied multiple times.",
    ),
]
SpotOperator = Annotated[
    SpotOperatorTypes,
    typer.Option("--operator", help="Spot operator."),
]
GeneCenter = Annotated[
    bool,
    typer.Option("--gene-center/--no-gene-center", help="Center gene weights."),
]
Plot = Annotated[
    bool,
    typer.Option("--plot", help="Plot results."),
]
Verbose = Annotated[
    bool,
    typer.Option("--verbose", help="Print timing information."),
]
ClusterBy = Annotated[
    ClusterTypes,
    typer.Option("--by", help="Dimension to cluster."),
]
Neighbours = Annotated[
    int,
    typer.Option("--neighbours", help="Number of neighbours."),
]
Resolution = Annotated[
    float,
    typer.Option("--resolution", help="Leiden resolution."),
]
Metric = Annotated[
    str,
    typer.Option("--metric", help="Neighbour graph metric."),
]
NGenes = Annotated[
    int | None,
    typer.Option("--ngenes", help="Restrict to top n genes for clustering."),
]
Umap = Annotated[
    bool,
    typer.Option("--umap", help="Compute and plot UMAP."),
]
