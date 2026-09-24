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
│ differential-sample-programs  Spatial programs whose pair statistic differs between biological sample groups.        │
│ label-decomposition           Decompose a spatial pair statistic into between-label, within-label and coupling terms. │
│ cross-programs                Paired neighbour-context and target-response gene programs.                             │
│ shared-cross-programs         Paired target-neighbour programs shared across biological samples.                     │
│ differential-cross-programs   Target-neighbour cross-covariance programs differing between sample groups.            │
│ pca-programs                  Ordinary PCA globally or within selected cell types.                                    │
│ estimate-spacing              Mean nearest-neighbour distance for a spatial sample.                                   │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

> [!NOTE]
> You may need to deactivate and source your environment for `tanpopo` to appear.

# tanpopo 0.2 — conditional spatial pair-statistic programs

This branch narrows Tanpopo to the part of the project supported by the current
simulation results: **conditional and comparative multigene spatial pair statistics**.

The package keeps the original Tanpopo architecture where possible:

```text
AnnData / NumPy input
      ↓
prepare_samples
      ↓
zero-diagonal pair adjacency
      ↓
geometry standardisation
      ↓
mark-correlation adjacency OR variogram Laplacian
      ↓
SpotProjector (centering + covariates)
      ↓
gene-space spatial-statistic operator
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

## Geometry-standardised pair statistics

Tanpopo now separates **what is measured between cells** from **how tissue geometry
determines which cell pairs are available**. The base Wendland adjacency remains
zero-diagonal. Before expression enters the model, the adjacency can be standardised
across biological samples and then interpreted in one of two ways.

### `--spatial-statistic mark_correlation` — default

For residual expression `Y` and a pair-weight adjacency `A`, Tanpopo uses

\[
G_s = Y_s^T A_s Y_s.
\]

Despite the CLI name, this is a matrix-valued, pair-conditioned **mark covariance**
operator rather than a Pearson correlation matrix. With geometry standardisation and
`--sample-weighting n_spots`, each biological sample contributes an average weighted
cross-cell product under a common pair-distance measure.

### `--spatial-statistic variogram`

The same pair weights are converted to the graph Laplacian

\[
L_s=D_s-A_s.
\]

Then

\[
Y_s^T L_s Y_s
=
\frac12\sum_{i,j} A_{s,ij}
(Y_{s,i}-Y_{s,j})(Y_{s,i}-Y_{s,j})^T,
\]

which is a matrix-valued mark variogram. For a single/shared analysis the largest
eigenvalues identify gene directions with the greatest local pairwise variation. For a
differential analysis, the sign of an eigenvalue indicates which sample group has the
larger variogram along that gene program.

### `--geometry-normalisation`

Three geometry treatments are available for both square and bipartite spatial
workflows:

- `none`: preserve the historical graph scale after degree normalisation;
- `mass`: fix the total pair mass to a workflow-specific reference;
- `distance` (default): additionally force every biological sample to have the same
  weighted pair-distance profile.

For square programs the target pair mass is `n_s`. For target-neighbour cross programs
it is `sqrt(n_target_s * n_neighbour_s)`. These choices are deliberate: the default
multi-sample weights `1/n_s` and `1/sqrt(n_target_s*n_neighbour_s)` respectively cancel
the normalised pair mass, so each tissue contributes one unit of total pair measure.

For `distance`, `[0, radius]` is divided into `--distance-bins` equal-width shells.
Let `m_sb` be sample `s`'s total pair weight in shell `b`. Bins not represented in
every sample are excluded. The common reference `q_b` is the equal-sample mean of
each sample's normalised shell profile on the common support. Each shell is then
rescaled so

\[
\sum_{ij:\ d_{ij}\in B_b} A^*_{s,ij}=M_s q_b,
\]

where `M_s=n_s` for square programs and
`M_s=sqrt(n_target_s*n_neighbour_s)` for cross programs. Consequently

\[
\sum_{ij} A^*_{s,ij}=M_s.
\]

With only one sample, `distance` reduces exactly to total-mass normalisation.

This standardisation changes the **measurement distribution over cell pairs**, not the
expression values themselves. It therefore does not regress away genuine expression
gradients associated with boundaries, compartments or other tissue coordinates.

### Optional random-labelling null centering

For mark correlation only, `--null-center` adds the exact finite-sample correction for
permuting sample-centred residual marks over the fixed point pattern. If `S_s=sum(A_s)`
and `n_s` cells are present,

\[
E_\pi[Y_\pi^T A_sY_\pi]
=-\frac{S_s}{n_s(n_s-1)}Y^TY.
\]

Tanpopo therefore adds `S_s / (n_s(n_s-1))` times the identity to the spatial
operator. This diagonal is an analytical null correction, not a biological self-edge.
`--null-center` requires sample- or label-centred expression and is not defined for
the variogram statistic.

Geometry diagnostics (pair mass, weighted-degree CV, weighted distance quantiles, and
when applicable the shell masses/reference weights) are stored under the analysis
metadata in `uns['tanpopo']`. Bipartite diagnostics report target/row and
neighbour/column connectivity separately and record removed identity pairs when the
target and neighbour masks overlap. Classical observation-window edge corrections are
not implemented because Tanpopo currently receives cell coordinates but not a reliable
tissue observation window/mask.

## Three explicit gene-space objectives

### `covariance` — default

This historical objective name now means the **unstandardised gene-space pair
statistic**. For mark correlation,

\[
\max_v v^T Y^T A Y v,
\]

so it finds directions with large positive cross-cell covariance. For variogram, the
same objective maximises

\[
\max_v v^T Y^T L Y v,
\]

so it finds directions with large local pairwise variation. The CLI name is retained
for backwards compatibility with the existing objective API.

### `gene_standardized`

Each gene is standardised using **ordinary residual expression variance** before
spatial covariance is computed:

\[
D_0 = \operatorname{diag}(Y^T Y),
\qquad
D_0^{-1/2}Y^T K_{stat}YD_0^{-1/2},
\]

This replaces the old `alpha=1` path. The denominator is positive ordinary expression
variance; spatial autocovariance is never treated as a variance.

### `gain`

Tanpopo solves the generalized spatial-gain problem on a truncated expression subspace:

\[
Y^T K_{stat}Yv = \lambda(Y^TY + \tau I)v,
\]

where `K_stat` is the mark-correlation adjacency or variogram Laplacian.

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
  --spatial-statistic mark_correlation \
  --geometry-normalisation distance \
  --distance-bins 10 \
  --permutations 1000
```

The contrast is a **difference of biological-sample group means**. Each sample is
first balanced according to `--sample-weighting`; group-A coefficients then sum to
`+1` and group-B coefficients to `-1`. Positive and negative eigenvalues identify
structure enriched on opposite sides of the contrast. Optional sample-label
permutations preserve group sizes and use the maximum absolute eigenvalue to provide
two-sided family-wise corrected mode p-values.

To compare local expression roughness rather than cross-cell covariance, use for
example:

```bash
tanpopo differential-sample-programs \
  -ia groupA_1.h5ad -ia groupA_2.h5ad \
  -ib groupB_1.h5ad -ib groupB_2.h5ad \
  --radius 20 \
  --spatial-statistic variogram \
  --geometry-normalisation distance \
  --distance-bins 10
```

If a common physical `--radius` is omitted in a multi-sample analysis, Tanpopo now
uses `3.5 x` the **median** of the sample-wise mean nearest-neighbour spacings rather
than deriving the radius from the first sample. An explicit physical radius is still
strongly preferred for cross-tissue interpretation.

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
  --objective covariance \
  --geometry-normalisation distance \
  --distance-bins 10
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
normalisation. Cross programs remain **mark-covariance** methods rather than
variograms, but their pair geometry is standardised by the same pair-measure machinery
as square programs. For sample `s`, distance-bin masses are reweighted to the common
reference `q_b` with total mass

\[
M_s=\sqrt{n_{t,s}n_{u,s}}.
\]

Thus `--geometry-normalisation distance` compares target-neighbour expression
relationships under the same distribution of physical separations across tissues.
`--geometry-normalisation mass` fixes only total pair mass, while `none` recovers the
historical rectangular-kernel scale.

If target and neighbour masks overlap, a cell is never paired with itself: identity
pairs are detected by the original observation index and removed **before** row/column
degree normalisation. This is the bipartite analogue of the zero diagonal in square
programs.

### 5. Shared target-neighbour programs across samples

For a recurrent niche interaction, estimate one pair of gene programs jointly across
biological samples rather than independently rotating each sample's SVD:

```bash
tanpopo shared-cross-programs \
  -i patient1.h5ad \
  -i patient2.h5ad \
  -i patient3.h5ad \
  -o shared_cross.h5ad \
  --label-key cell_type \
  --target-labels Fibroblasts \
  --neighbour-labels Macrophages \
  --radius 30 \
  --components 5 \
  --objective covariance \
  --geometry-normalisation distance \
  --distance-bins 10
```

The pooled operator is

\[
C_{\mathrm{shared}} = \sum_s w_s\,Y_{t,s}^T A_{tu,s}Y_{u,s}.
\]

With the default `--sample-weighting n_spots`,

\[
w_s = \frac{1}{\sqrt{n_{t,s}n_{u,s}}}.
\]

Under `mass` or `distance` geometry normalisation, the standardised cross kernel has
`sum(A_tu,s)=sqrt(n_t,s*n_u,s)`, so `w_s sum(A_tu,s)=1` for every biological sample.
With `distance`, the samples additionally share the same target-neighbour distance
profile. `--sample-weighting none` is still available when this balancing is not
desired. The same `covariance`, `gene_standardized`, and `gain` gene-space objectives
are available as for single-sample cross programs.

In addition to the shared target/neighbour loadings, Tanpopo stores each shared mode's
raw cross-spatial covariance in every biological sample. This is useful for checking
that a pooled program is recurrent rather than driven by one patient.

Outputs include:

```text
varm['tanpopo_<id>_target_loadings']
varm['tanpopo_<id>_neighbour_loadings']
obsm['tanpopo_<id>_target_modes']
obsm['tanpopo_<id>_neighbour_modes']
uns ['tanpopo'][<id>]['singular_values']
uns ['tanpopo'][<id>]['sample_coefficients']
uns ['tanpopo'][<id>]['sample_mode_covariance']
uns ['tanpopo'][<id>]['sample_mode_statistic']
uns ['tanpopo'][<id>]['aggregate_mode_statistic']
```

### 6. Differential target-neighbour programs across sample groups

```bash
tanpopo differential-cross-programs \
  -ia responder1.h5ad -ia responder2.h5ad \
  -ib nonresponder1.h5ad -ib nonresponder2.h5ad \
  -o differential_cross.h5ad \
  --label-key cell_type \
  --target-labels Fibroblasts \
  --neighbour-labels Macrophages \
  --radius 30 \
  --components 5 \
  --objective covariance \
  --geometry-normalisation distance \
  --distance-bins 10 \
  --permutations 1000
```

For each biological sample, let

\[
C_s = w_s\,Y_{t,s}^T A_{tu,s}Y_{u,s},
\]

where the default `w_s=1/sqrt(n_target*n_neighbour)` cancels the standardised
bipartite pair mass. Differential cross programs use the group-mean contrast

\[
\Delta C
=
\frac{1}{n_A}\sum_{s\in A} C_s
-
\frac{1}{n_B}\sum_{s\in B} C_s,
\]

and return the leading paired programs from

\[
\Delta C = U\Sigma V^T.
\]

Unlike the symmetric differential operator, `Sigma` contains non-negative singular
values: there is no meaningful positive/negative singular-value sign. Tanpopo therefore
stores the geometry/sample-normalised statistic for every biological sample and the
corresponding group-A mean, group-B mean, and contrast for each fitted mode. For the
`covariance` objective the stored `contrast_mode_statistic` equals the fitted singular
value up to numerical precision. The group-specific values should be used to determine
which side of the comparison carries the paired target-neighbour relationship.

Sample-label permutations retain the observed group sizes and use the maximum singular
value in each permutation to provide family-wise corrected mode p-values. Geometry is
prepared once across all samples and remains fixed while group labels are permuted.

Additional outputs are:

```text
uns ['tanpopo'][<id>]['group_a_mode_statistic']
uns ['tanpopo'][<id>]['group_b_mode_statistic']
uns ['tanpopo'][<id>]['contrast_mode_statistic']
uns ['tanpopo'][<id>]['permutation_pvalues']
uns ['tanpopo'][<id>]['permutation_max_singular_values']
```

### 7. Exact label-source decomposition

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
For either the mark-correlation adjacency or variogram Laplacian, write the chosen
spatial operator as `K_stat`. The statistic decomposes exactly:

\[
Y^T K_{stat}Y
=
B^T K_{stat}B
+
W^T K_{stat}W
+
B^T K_{stat}W + W^T K_{stat}B.
\]

`label-decomposition` accepts the same `--spatial-statistic`,
`--geometry-normalisation`, `--distance-bins`, and `--null-center` options as the
other square-graph workflows.

Tanpopo stores programs for `total`, `between`, `within`, and `coupling`, together
with a numerical operator-additivity error in `uns`.

## AnnData outputs

The existing Tanpopo naming pattern is retained where possible. For
`--experiment-id spatial`:

```text
varm['tanpopo_spatial_gene_loadings']
varm['tanpopo_spatial_gene_scores']
var ['tanpopo_spatial_gene_spatial_statistic']
var ['tanpopo_spatial_gene_spatial_covariance']  # mark_correlation compatibility key
var ['tanpopo_spatial_gene_mark_variogram']       # variogram-specific key
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
    spatial_statistic="mark_correlation",  # or "variogram"
    geometry_normalisation="distance",
    distance_bins=10,
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

model = CrossSpatialProgramModel(
    30,
    objective="gain",
    geometry_normalisation="distance",
    distance_bins=10,
).fit(
    X,
    coords,
    n_components=5,
    target_mask=cell_type == "Fibroblasts",
    neighbour_mask=cell_type == "Macrophages",
)
```

Shared cross-population programs:

```python
from tanpopo import SharedCrossSpatialProgramModel

model = SharedCrossSpatialProgramModel(
    30,
    objective="covariance",
    geometry_normalisation="distance",
    distance_bins=10,
).fit(
    [X1, X2, X3],
    [coords1, coords2, coords3],
    n_components=5,
    target_masks=[ct1 == "Fibroblasts", ct2 == "Fibroblasts", ct3 == "Fibroblasts"],
    neighbour_masks=[ct1 == "Macrophages", ct2 == "Macrophages", ct3 == "Macrophages"],
)

# rows are patients, columns are shared paired modes
patient_covariance = model.sample_mode_covariance_
```

Differential cross-population programs:

```python
from tanpopo import DifferentialCrossSpatialProgramModel

model = DifferentialCrossSpatialProgramModel(
    30,
    positive_samples=[0, 1, 2],
    negative_samples=[3, 4],
    objective="covariance",
    geometry_normalisation="distance",
).fit(
    [X1, X2, X3, X4, X5],
    [coords1, coords2, coords3, coords4, coords5],
    n_components=5,
    target_masks=target_masks,
    neighbour_masks=neighbour_masks,
)

# group-normalised target-neighbour statistic along each fitted mode
group_a = model.group_a_mode_statistic_
group_b = model.group_b_mode_statistic_
contrast = model.contrast_mode_statistic_
```

## Validation priorities

This rewrite is intentionally narrow. Before adding more functionality, the following
benchmarks should pass:

1. spatial versus matched non-spatial injections across many seeds;
2. within-cell-type recovery against fibroblast-only PCA and SPACO;
3. expression-strength × spatial-strength sweeps;
4. target-neighbour recovery against NiCo/MISTy/Niche-DE where outputs are comparable;
5. differential spatial covariance with no differential mean expression;
6. geometry-invariance simulations with matched conditional mark covariance but
   different tissue shapes/densities;
7. covariance-versus-variogram consistency and scale-specific simulations;
8. held-out biological-sample replication.

The package should not grow back into a general spatial-transcriptomics toolbox until
those tests establish that these conditional operators add information beyond existing
specialist methods.
