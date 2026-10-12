# Tanpopo 0.3

Tanpopo discovers multigene spatial covariance programs from spatial transcriptomics while
standardising the cell-pair geometry used to measure them. The same framework supports a
single tissue, replicated tissues, condition contrasts, and paired target-neighbour
(cell-cell context) programs.

The release API is intentionally narrow: one spatial statistic, a small set of related
models, explicit geometry normalisation, permutation-based mode inference, and utilities
for inspecting and interpreting stored results.

## Installation

```bash
pip install .
```

Optional utilities have small extras:

```bash
pip install '.[plot]'
pip install '.[cluster]'
pip install '.[gsea]'
pip install '.[dev]'
```

## Scientific model

For sample `s`, let `Y_s` be centred/residualised spot-by-gene expression and `A_s` a
zero-diagonal spatial pair-weight matrix. Tanpopo's primary gene-space operator is

\[
G_s = Y_s^T A_s Y_s.
\]

An eigenvector of `G_s` is a multigene direction whose scores covary across nearby,
distinct cells. Self-edges are always removed before graph normalisation.

For target-neighbour analyses, target expression `Y_t` and neighbour expression `Y_u`
are connected by a rectangular pair matrix `A_tu`:

\[
C_s = Y_{t,s}^T A_{tu,s}Y_{u,s}.
\]

Its SVD produces paired target and neighbour gene programs.

Tanpopo does **not** expose a mark-variogram objective in this release. Variograms answer a
different biological question and require different interpretation, especially for
bipartite target-neighbour analyses.

## Geometry normalisation

Spatial covariance can change because biology changes or because different tissues offer
different numbers and distances of cell pairs. Tanpopo therefore separates construction
of the pair measure from expression analysis.

`--geometry-normalisation` has three values:

- `distance` (default): all biological samples use the same weighted pair-distance
  distribution over distance bins that have support in every sample;
- `mass`: only total pair mass is standardised;
- `none`: retain the degree-normalised pair graph without cross-sample pair-measure
  standardisation. This is useful mainly for ablations.

For a square sample the normalised pair mass is `n_s`. For target-neighbour samples it is
`sqrt(n_target * n_neighbour)`. With the default multi-sample `n_spots` weighting this
makes every biological sample contribute one unit of total pair measure before a group
contrast is applied.

Distance normalisation changes pair weights, not expression values. It therefore avoids
regressing genuine boundary-, compartment-, or gradient-associated expression out of the
features themselves.

## Objectives

All program models expose three gene-space objectives.

### `covariance` (default)

Directly eigendecomposes the geometry-standardised spatial covariance operator. This is
the primary Tanpopo estimand and the recommended default.

### `gene_standardized`

Scales genes by residual expression variance before solving the covariance problem. This
can be useful when high-variance genes otherwise dominate the spectrum.

### `gain`

Solves the generalized spatial-gain problem

\[
Y^TAYv = \lambda (Y^TY + \tau I)v
\]

in a truncated expression SVD. This asks for spatial covariance relative to available
non-spatial expression variance and is also used by the `spatial-info` diagnostic.

## Core workflows

Run `tanpopo --help` for the complete command list.

### Single-sample programs

```bash
tanpopo programs \
  -i sample.h5ad \
  -o sample_programs.h5ad \
  --layer logcounts \
  --radius 20 \
  --components 10 \
  --analysis-id spatial
```

Restrict to a cell type with `--label-key` and `--labels`.

### Shared programs across biological samples

```bash
tanpopo shared \
  -i donor1.h5ad -i donor2.h5ad -i donor3.h5ad \
  -o shared.h5ad \
  --layer logcounts \
  --radius 20 \
  --components 10
```

Each biological sample contributes separately; cells are not concatenated into one graph.

### Differential programs

```bash
tanpopo differential \
  -ia conditionA_1.h5ad -ia conditionA_2.h5ad \
  -ib conditionB_1.h5ad -ib conditionB_2.h5ad \
  -o differential.h5ad \
  --layer logcounts \
  --radius 20 \
  --components 10 \
  --permutations 1000
```

The signed operator is the weighted group-A minus group-B spatial covariance. Positive
eigenvalues correspond to stronger covariance in group A and negative eigenvalues to
group B. Sample-label permutations use the maximum absolute eigenvalue for family-wise
mode inference.

### Cross programs

```bash
tanpopo cross \
  -i sample.h5ad \
  -o cross.h5ad \
  --layer logcounts \
  --label-key CellAnnotation.Level0 \
  --target-labels Fibroblasts \
  --neighbour-labels Macrophages \
  --radius 20 \
  --components 10
```

Overlapping target and neighbour masks are allowed, but a biological cell is never paired
with itself. Identity pairs are removed by original observation ID before bipartite degree
normalisation.

### Shared and differential cross programs

```bash
tanpopo cross-shared \
  -i donor1.h5ad -i donor2.h5ad \
  -o shared_cross.h5ad \
  --label-key CellAnnotation.Level0 \
  --target-labels Fibroblasts \
  --neighbour-labels Macrophages \
  --radius 20
```

```bash
tanpopo cross-differential \
  -ia conditionA_1.h5ad -ia conditionA_2.h5ad \
  -ib conditionB_1.h5ad -ib conditionB_2.h5ad \
  -o differential_cross.h5ad \
  --label-key CellAnnotation.Level0 \
  --target-labels Fibroblasts \
  --neighbour-labels Macrophages \
  --radius 20 \
  --permutations 1000
```

A differential cross operator is rectangular. Its singular values are non-negative, so
they do not carry a group-direction sign. The signed biological object is each rank-one
component of the group-A minus group-B cross-covariance matrix. Per-sample mode
covariances are stored to aid interpretation.

## Automatic spatial mode selection

A fixed mode count is still accepted:

```bash
--components 10
```

For inferential selection use:

```bash
--components auto --max-components 30 --permutations 500 --mode-alpha 0.05
```

Tanpopo fits up to `--max-components`, constructs a permutation null for the maximum
spatial statistic, and retains modes with family-wise p-values at or below
`--mode-alpha`.

The null statistic follows the fitted model:

| workflow | permutation | maximum null statistic |
| --- | --- | --- |
| `programs`, `shared` | random labelling within each biological sample | largest eigenvalue |
| `differential` | biological-sample group labels | largest absolute eigenvalue |
| `cross`, `cross-shared` | neighbour identities within each sample | largest singular value |
| `cross-differential` | biological-sample group labels | largest singular value |

Automatic selection can return zero modes. This is intentional: a dataset with no
supported spatial structure should not be forced to return a program. For the `gain`
objective, the projected expression basis is cached and reused across permutations; only
the low-dimensional spatial numerator is recomputed.

## Spatial-information diagnostics

`spatial-info` uses the same projector, geometry normalisation, gain objective and
max-statistic inference as the program models; it is not a parallel implementation.

```bash
tanpopo spatial-info \
  -i sample.h5ad \
  --layer logcounts \
  --radius 20 \
  --max-components 30 \
  --permutations 500
```

It reports:

- maximum spatial gain;
- the max-statistic null threshold;
- null-threshold-corrected spectral gain;
- selected mode count;
- effective spatial dimension (participation ratio of positive gain above the null);
- graph/pair-measure diagnostics.

A radius curve can reuse the same diagnostic definition:

```bash
tanpopo spatial-info -i sample.h5ad --radii 10,20,40,80 --permutations 200
```

The radius with the largest in-sample gain should not automatically be treated as an
unbiased scale estimate. The curve is a diagnostic; radius choice should also consider
physical interpretation and robustness.

## Diagnostics and baselines

### Label decomposition

`decompose-labels` exactly decomposes sample-centred covariance into between-label,
within-label and coupling operators. It is intended as a diagnostic for asking whether a
spatial signal is explained mostly by cell-type organization or by within-cell-type
expression structure.

```bash
tanpopo decompose-labels \
  -i sample.h5ad \
  --label-key CellAnnotation.Level0 \
  --radius 20
```

### PCA baseline

`baseline-pca` stores an ordinary PCA result using the same schema as Tanpopo program
results, so downstream comparison utilities can use it directly.

## Result schema

Tanpopo 0.3 writes a versioned result record under:

```text
adata.uns['tanpopo'][analysis_id]
```

with:

```text
tanpopo_version
analysis_type
preprocessing
scope
results[variant]
result_metadata[variant]
```

The `results` entry identifies the corresponding `varm`, `obsm` and gene-level fields.
Use `tanpopo.results.load_result()` rather than hard-coding storage keys.

```python
from tanpopo.results import load_result

result = load_result(adata, "spatial")
print(result.values)
print(result.gene_loadings)
```

Analyses with multiple outputs, such as per-label programs or label decomposition, store
named `variant`s. Model settings are recorded per variant so analyses whose masks imply
different automatic radii remain unambiguous.

## Utilities

### Inspect

```bash
tanpopo inspect -i sample.h5ad
```

Shows AnnData metadata, Tanpopo results and mean nearest-neighbour spacing.

### Preprocess

```bash
tanpopo preprocess -i raw.h5ad -o processed.h5ad --transform log1p --target-sum 10000
```

This exposes Tanpopo's normalisation/filtering path without fitting a model.

### Export

```bash
tanpopo export -i result.h5ad -o loadings.csv --analysis-id spatial
```

For cross results add `--side target` or `--side neighbour`.

### Compare programs

```bash
tanpopo compare \
  --input-a result_a.h5ad --analysis-a spatial \
  --input-b result_b.h5ad --analysis-b spatial
```

Ordinary programs are matched one-to-one with the Hungarian algorithm and compared by
signed/absolute cosine similarity, top-gene overlap and top-k subspace overlap. Cross
programs compare the paired rank-one target-neighbour component, which is invariant to a
simultaneous sign flip of the two loading vectors.

`compare-spots` performs analogous one-to-one comparison of stored spot-mode spaces.

### Plot, cluster and GSEA

`plot`, `cluster` and `gsea` consume the versioned result schema. Plotting uses only
Matplotlib. Scanpy/Leiden clustering and GSEA are optional extras so they do not expand
the numerical core dependency surface.

## Python API

```python
from tanpopo import SpatialProgramModel

model = SpatialProgramModel(
    radius=20,
    objective="covariance",
    geometry_normalisation="distance",
    distance_bins=10,
).fit(X, coords, 20)

model.permutation_test(500, seed=1)
selection = model.select_modes(alpha=0.05)
model.truncate_modes(selection.selected_modes)
```

Related public models are:

- `SharedSpatialProgramModel`
- `DifferentialSpatialProgramModel`
- `CrossSpatialProgramModel`
- `SharedCrossSpatialProgramModel`
- `DifferentialCrossSpatialProgramModel`
- `LabelDecompositionModel`

## Geometry diagnostics

Stored diagnostics include pair mass, weighted distance quantiles, degree variation and
isolated fractions. Distance-normalised models additionally store raw/standardised bin
masses, common distance support and reference weights. Bipartite analyses report row and
column connectivity separately and the number of removed identity pairs.

Tanpopo currently receives coordinates rather than a reliable tissue observation window,
so classical observation-window edge corrections are not applied.

## Development

```bash
pytest -q
```

The tests cover square and bipartite geometry normalisation, differential and shared
program recovery, cross-program identity handling, exact label decomposition, result
schema loading, permutation mode selection and spatial-information diagnostics.

Before publishing a result, prefer an explicit physical radius, sufficient biological
replicates, and enough permutations for the p-value resolution required by the analysis.
