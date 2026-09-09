from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer


class ObjectiveTypes(str, Enum):
    covariance = "covariance"
    gene_standardized = "gene_standardized"
    gain = "gain"


class SpotOperatorTypes(str, Enum):
    none = "none"
    sample = "sample"
    label = "label"


class SampleWeightingTypes(str, Enum):
    none = "none"
    n_spots = "n_spots"


class GraphNormalisationTypes(str, Enum):
    symmetric = "symmetric"
    none = "none"


class TransformTypes(str, Enum):
    sqrt = "sqrt"
    log1p = "log1p"


class Dtypes(str, Enum):
    float32 = "float32"
    float64 = "float64"


InputPath = Annotated[
    Path,
    typer.Option("--input", "-i", exists=True, dir_okay=False, readable=True),
]
InputPaths = Annotated[
    list[Path],
    typer.Option("--input", "-i", exists=True, dir_okay=False, readable=True),
]
InputPathsA = Annotated[
    list[Path],
    typer.Option("--input-a", "-ia", exists=True, dir_okay=False, readable=True),
]
InputPathsB = Annotated[
    list[Path],
    typer.Option("--input-b", "-ib", exists=True, dir_okay=False, readable=True),
]
OutputPath = Annotated[Path | None, typer.Option("--output", "-o", dir_okay=False)]
Layer = Annotated[str | None, typer.Option("--layer", help="AnnData layer to analyse instead of X.")]
LabelKey = Annotated[str | None, typer.Option("--label-key", help="obs column defining cell types.")]
Labels = Annotated[
    str | None,
    typer.Option(
        "--labels",
        help="Cell type(s) to analyse, comma separated. 'all' analyses each label separately.",
    ),
]
TargetLabels = Annotated[str, typer.Option("--target-labels", help="Target cell label(s), comma separated.")]
NeighbourLabels = Annotated[
    str, typer.Option("--neighbour-labels", help="Neighbour cell label(s), comma separated.")
]
SampleNames = Annotated[list[str] | None, typer.Option("--name", help="Name for each input sample.")]
ExperimentId = Annotated[
    str,
    typer.Option("--experiment-id", "--cmd-id", "-id", help="Namespace for Tanpopo outputs."),
]
Radius = Annotated[float | None, typer.Option("--radius", "-r", help="Wendland support radius.")]
Components = Annotated[int, typer.Option("--components", "-k", help="Number of programs.")]
Objective = Annotated[
    ObjectiveTypes,
    typer.Option(
        "--objective",
        help="Spatial objective: covariance, gene_standardized, or gain.",
    ),
]
ExpressionRank = Annotated[
    int,
    typer.Option(
        "--expression-rank",
        help="Expression subspace rank used by objective=gain.",
    ),
]
GainRidge = Annotated[
    float,
    typer.Option("--gain-ridge", help="Ridge added to the expression metric for gain."),
]
GraphNormalisation = Annotated[
    GraphNormalisationTypes,
    typer.Option("--graph-normalisation", help="Spatial graph degree normalisation."),
]
SpotOperator = Annotated[
    SpotOperatorTypes,
    typer.Option("--operator", help="Spot-space centering: sample, label, or none."),
]
SampleWeighting = Annotated[
    SampleWeightingTypes,
    typer.Option("--sample-weighting", help="How biological samples are balanced."),
]
Alpha = Annotated[
    float | None,
    typer.Option(
        "--alpha",
        help="Deprecated. alpha=0 maps to objective=covariance; nonzero alpha is unsupported.",
    ),
]
Transform = Annotated[TransformTypes | None, typer.Option("--transform")]
MinCounts = Annotated[int | None, typer.Option("--min-counts")]
MinSpotFraction = Annotated[float | None, typer.Option("--min-spot-fraction")]
TargetSum = Annotated[
    float | None,
    typer.Option("--target-sum", help="Per-cell normalisation target; omitted by default."),
]
Covariates = Annotated[
    str | None,
    typer.Option(
        "--covariates",
        help="Comma-separated: log_total_counts,log_detected_genes,mito_fraction,ribo_fraction.",
    ),
]
Include = Annotated[str | None, typer.Option("--include")]
Exclude = Annotated[str | None, typer.Option("--exclude")]
Dtype = Annotated[Dtypes, typer.Option("--dtype")]
Verbose = Annotated[bool, typer.Option("--verbose")]

Permutations = Annotated[int, typer.Option("--permutations", help="Sample-label permutations for differential FWER p-values.")]
RandomSeed = Annotated[int, typer.Option("--seed", help="Random seed.")]
