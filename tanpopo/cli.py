from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Callable, TypeVar

import typer

E = TypeVar("E", bound=Enum)


def comma_separated_enum(enum_cls: type[E]) -> Callable[[str | None], list[E] | None]:
    valid = {item.value: item for item in enum_cls}
    choices = ", ".join(str(item.value) for item in enum_cls)

    def parser(value: str | None) -> list[E] | None:
        if value is None:
            return None

        raw_values = [item.strip() for item in value.split(",")]

        if any(item == "" for item in raw_values):
            raise typer.BadParameter(f"Empty values are not allowed. Choices: {choices}")

        invalid = [item for item in raw_values if item not in valid]
        if invalid:
            raise typer.BadParameter(
                f"Invalid value(s): {', '.join(invalid)}. " f"Choose from: {choices}"
            )

        return [valid[item].value for item in raw_values]

    return parser


# ------------------------------------------------------------------------------
# IO
# ------------------------------------------------------------------------------
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
InputPaths = Annotated[
    list[Path],
    typer.Option(
        "--input",
        "-i",
        help="Input .h5ad file. Supply once per sample for multi-sample workflows.",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
]
InputPathsA = Annotated[
    list[Path],
    typer.Option(
        "--input-a",
        "-ia",
        help="Input .h5ad file for group A. Can be specified multiple times.",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
]
InputPathsB = Annotated[
    list[Path],
    typer.Option(
        "--input-b",
        "-ib",
        help="Input .h5ad file for group B. Can be specified multiple times.",
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
Labels = Annotated[
    str | None,
    typer.Option(
        "--labels",
        help="Subset spatial programs to label(s) (comma separated). "
        "Omit for whole-sample analysis, use 'all' for every label in --label-key.",
    ),
]
SampleNames = Annotated[
    list[str] | None,
    typer.Option(
        "--name",
        help="Name for each --input, in the same order. Defaults to input file stems.",
    ),
]


# ------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------
class SpotOperatorTypes(str, Enum):
    none = "none"
    sample = "sample"
    label = "label"


class SampleWeightingTypes(str, Enum):
    none = "none"
    n_spots = "n_spots"
    trace = "trace"


class SampleNormaliseTypes(str, Enum):
    sample = "sample"
    pooled = "pooled"


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
SpotOperator = Annotated[
    SpotOperatorTypes,
    typer.Option("--operator", help="Spot operator."),
]
GeneCenter = Annotated[
    bool,
    typer.Option("--gene-center/--no-gene-center", help="Center gene weights."),
]
SampleWeighting = Annotated[
    SampleWeightingTypes,
    typer.Option("--sample-weighting", help="How to balance samples."),
]
SampleNormaliseBy = Annotated[
    SampleNormaliseTypes,
    typer.Option(
        "--normalise-by",
        help="Gene scaling reference for sample-wise models.",
    ),
]


# ------------------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------------------
class TransformTypes(str, Enum):
    sqrt = "sqrt"
    log1p = "log1p"


class CovariateTypes(str, Enum):
    log_total_counts = "log_total_counts"
    log_detected_genes = "log_detected_genes"
    mito_fraction = "mito_fraction"
    ribo_fraction = "ribo_fraction"


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
    str | None,
    typer.Option(
        "--covariates",
        callback=comma_separated_enum(CovariateTypes),
        help="Covariate(s) to correct for (comma separated).",
    ),
]


# ------------------------------------------------------------------------------
# Clustering
# ------------------------------------------------------------------------------
class ClusterTypes(str, Enum):
    spots = "spots"
    genes = "genes"


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

# ------------------------------------------------------------------------------
# Misc
# ------------------------------------------------------------------------------
ExperimentId = Annotated[
    str,
    typer.Option(
        "--experiment-id",
        "-id",
        help=("Experiment ID used to namespace Tanpopo outputs. Defaults to the workflow name."),
    ),
]
Plot = Annotated[
    bool,
    typer.Option("--plot", help="Plot results."),
]
Verbose = Annotated[
    bool,
    typer.Option("--verbose", help="Print timing information."),
]
