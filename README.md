![logo](tanpopo.webp)

tanpopo is a python package for discovering spatially structured gene programs in spatial transcriptomics data. It computes spatial gene eigenmodes from an expression matrix and spot coordinates, using sparse spatial kernels to identify genes and programs whose expression varies coherently across tissue space.

## Installation

```
pip install git+https://github.com/msceko/tanpopo.git
```

Check if it has installed correctly by running `tanpopo` to get a list of available tools:

```
╭─ Commands ────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ spatial-programs              Spatial programs globally or within selected cell types.                                │
│ shared-programs               Shared spatial programs across biological samples, optionally within one cell type.     │
│ differential-sample-programs  Spatial programs whose covariance differs between biological sample groups.             │
│ label-decomposition           Decompose total spatial covariance into between-label, within-label and coupling terms. │
│ cross-programs                Paired neighbour-context and target-response gene programs.                             │
│ pca-programs                  Ordinary PCA globally or within selected cell types.                                    │
│ estimate-spacing              Mean nearest-neighbour distance for a spatial sample.                                   │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

> [!NOTE]
> You may need to deactivate and source your environment for `tanpopo` to appear.

# tanpopo 0.2 — conditional spatial covariance programs

This branch narrows Tanpopo to the part of the project supported by the current
simulation results: **conditional and comparative multigene spatial covariance**.

The package keeps the original Tanpopo architecture where possible:

```text
AnnData / NumPy input
      ↓
prepare_samples
      ↓
zero-diagonal spatial graph
      ↓
SpotProjector (centering + covariates)
      ↓
spatial gene operator
      ↓
eigensolver / SVD
      ↓
gene loadings + cell modes
      ↓
AnnData output
```

The old diagonal-containing kernel and spatial-diagonal `alpha` normalisation are no
longer part of the default method.

## Why this rewrite

With the historical kernel, `alpha=0` used

\[
X^T K X = X^T X + X^T A X
\]

because the Wendland self weight is one. Strong non-spatial expression programs were
therefore recovered almost as well as genuinely spatial programs. Tanpopo 0.2 defines
spatial programs using relationships **between different cells**:

\[
G_{sp}=Y^T A Y, \qquad A_{ii}=0.
\]

Square spatial graphs are zero-diagonal and symmetrically degree-normalised by
default:

\[
A=D^{-1/2}A_0D^{-1/2}.
\]

This separates ordinary molecular variance from spatial covariance and reduces
sensitivity to local cell density.

## Three explicit objectives

### `covariance` — default

\[
\max_v v^T Y^T A Y v.
\]

Finds molecular programs accounting for the largest positive cross-cell spatial
covariance. This is the principal Tanpopo objective and the one most directly supported
by the current within-cell-type simulations.

### `gene_standardized`

Each gene is standardised using **ordinary residual expression variance** before
spatial covariance is computed:

\[
D_0 = \operatorname{diag}(Y^T Y),
\qquad
D_0^{-1/2}Y^TAYD_0^{-1/2}.
\]

This replaces the old `alpha=1` path. The denominator is positive ordinary expression
variance; spatial autocovariance is never treated as a variance.

### `gain`

Tanpopo solves the generalized spatial-gain problem on a truncated expression subspace:

\[
Y^TAYv = \lambda(Y^TY + \tau I)v.
\]

It is implemented through a truncated SVD of the projected expression matrix rather
than inversion of a large gene covariance matrix. This objective asks which molecular
directions are disproportionately spatial relative to their expression variance.

The global `gain` objective overlaps conceptually with methods such as SPACO; its value
inside Tanpopo is primarily in conditioning and comparison rather than as a standalone
novel spatial PCA.

## Core workflows

### 1. Spatial programs within a cell type

```bash
tanpopo spatial-programs \
  -i sample.h5ad \
  -o result.h5ad \
  --layer log1p \
  --label-key cell_type \
  --labels Fibroblasts \
  --radius 20 \
  --components 10 \
  --objective covariance
```

Hard masking is the only masking used by `spatial-programs`. This directly asks:

> Which multigene states are spatially coordinated within this cell population?

`--labels all` runs the analysis separately for each label.

The historical `--soft-mask` workflow is intentionally removed; target-neighbour
questions are handled explicitly by `cross-programs`.

### 2. Shared conditional programs across samples

```bash
tanpopo shared-programs \
  -i patient1.h5ad \
  -i patient2.h5ad \
  -i patient3.h5ad \
  -o shared.h5ad \
  --label-key cell_type \
  --labels Fibroblasts \
  --radius 20 \
  --objective covariance
```

Biological samples are balanced by `1 / n_spots` by default so one large tissue does
not automatically dominate the shared operator.

### 3. Differential spatial organization across sample groups

```bash
tanpopo differential-sample-programs \
  -ia responder1.h5ad \
  -ia responder2.h5ad \
  -ib nonresponder1.h5ad \
  -ib nonresponder2.h5ad \
  -o differential.h5ad \
  --label-key cell_type \
  --labels Fibroblasts \
  --radius 20 \
  --components 8 \
  --permutations 1000
```

The contrast is a difference of biological-sample spatial operators. Positive and
negative eigenvalues identify structure enriched on opposite sides of the contrast.
Optional sample-label permutations use the maximum absolute eigenvalue to provide
two-sided family-wise corrected mode p-values.

### 4. Paired target-neighbour programs

```bash
tanpopo cross-programs \
  -i sample.h5ad \
  -o cross.h5ad \
  --label-key cell_type \
  --target-labels Fibroblasts \
  --neighbour-labels Macrophages \
  --radius 30 \
  --components 5 \
  --objective covariance
```

This estimates the bipartite cross-covariance

\[
C_{t\leftarrow u}=Y_t^T A_{tu}Y_u
\]

and returns paired target and neighbour gene programs through an SVD.

Outputs:

- `varm['tanpopo_<id>_target_loadings']`
- `varm['tanpopo_<id>_neighbour_loadings']`
- `obsm['tanpopo_<id>_target_modes']`
- `obsm['tanpopo_<id>_neighbour_modes']`
- `uns['tanpopo'][<id>]['singular_values']`

The bipartite graph uses the same Wendland kernel with row/column degree
normalisation.

### 5. Exact label-source decomposition

```bash
tanpopo label-decomposition \
  -i sample.h5ad \
  -o decomposition.h5ad \
  --label-key cell_type \
  --radius 20
```

After a common sample-level projection, expression is decomposed as

\[
Y=B+W
\]

where `B` is the between-label mean component and `W` is the within-label residual.
The spatial covariance then decomposes exactly:

\[
Y^TAY
=
B^TAB
+
W^TAW
+
B^TAW + W^TAB.
\]

Tanpopo stores programs for `total`, `between`, `within`, and `coupling`, together
with a numerical operator-additivity error in `uns`.

## AnnData outputs

The existing Tanpopo naming pattern is retained where possible. For
`--experiment-id spatial`:

```text
varm['tanpopo_spatial_gene_loadings']
varm['tanpopo_spatial_gene_scores']
var ['tanpopo_spatial_gene_spatial_covariance']
obsm['tanpopo_spatial_spot_modes']
uns ['tanpopo']['spatial']['eigenvalues']
```

For cell-type-restricted analyses, the label is appended to the key, matching the old
workflow style.

`gene_loadings` is the key to use for comparisons across all objectives. For
`objective=gain`, solver eigenvectors live in the truncated expression basis and are
therefore not stored as a gene-space `eigenvectors` matrix.

## Preprocessing change

When `--layer` is supplied, filtering, normalization and transformation now operate on
that layer. The old behavior could transform `adata.X` while the model subsequently
analysed an unchanged layer.

Preprocessing is conservative by default:

- `--target-sum` defaults to disabled;
- `--transform` defaults to disabled;
- `--min-counts 10` remains the default gene filter.

For pre-normalized/log-transformed simulation layers, simply provide the layer and do
not request another transform.

## Migration from Tanpopo 0.1

### Removed as primary concepts

- self-edges in spatial kernels;
- spatial-diagonal `alpha` normalization;
- generic marker-program workflow;
- differential-label-versus-rest workflow;
- soft masking through `A.T @ A`;
- clustering/plotting as core CLI responsibilities.

These are removed or demoted because they are not central to the narrowed scientific
claim.

### `--alpha`

`--alpha 0` is accepted temporarily and maps to `--objective covariance`.
Non-zero alpha values raise an error. Use:

```text
--objective gene_standardized
```

or

```text
--objective gain
```

for explicit normalized estimands.

## Array-level API

All model classes work directly on NumPy/SciPy matrices without AnnData:

```python
from tanpopo import SpatialProgramModel

model = SpatialProgramModel(
    radius=20,
    objective="covariance",
).fit(
    X,
    coords,
    n_components=10,
    masks=cell_type == "Fibroblasts",
)

loadings = model.gene_loadings
scores = model.spot_modes[0]
```

Cross-population programs:

```python
from tanpopo import CrossSpatialProgramModel

model = CrossSpatialProgramModel(30, objective="gain").fit(
    X,
    coords,
    n_components=5,
    target_mask=cell_type == "Fibroblasts",
    neighbour_mask=cell_type == "Macrophages",
)
```

## Validation priorities

This rewrite is intentionally narrow. Before adding more functionality, the following
benchmarks should pass:

1. spatial versus matched non-spatial injections across many seeds;
2. within-cell-type recovery against fibroblast-only PCA and SPACO;
3. expression-strength × spatial-strength sweeps;
4. target-neighbour recovery against NiCo/MISTy/Niche-DE where outputs are comparable;
5. differential spatial covariance with no differential mean expression;
6. held-out biological-sample replication.

The package should not grow back into a general spatial-transcriptomics toolbox until
those tests establish that these conditional operators add information beyond existing
specialist methods.
