# Migration notes: Tanpopo 0.1 -> narrowed 0.2

## Geometry-standardised spatial statistics

Square spatial workflows expose two independent choices:

- `--spatial-statistic mark_correlation|variogram`;
- `--geometry-normalisation none|mass|distance`.

`cross-programs` and `shared-cross-programs` now expose the same geometry-normalisation
choice, but remain mark-covariance/SVD methods rather than variograms.

`distance` is the new default. It reweights distance shells to a shared reference
profile across biological samples and fixes each sample's total pair mass to a
workflow-specific reference (`n_s` for square graphs and
`sqrt(n_target*n_neighbour)` for cross graphs). This changes eigenvalue/statistic scales
relative to the historical raw graph, although single-sample mark-correlation loadings are unchanged
up to numerical eigensolver variation because one-sample distance normalisation is a
global graph rescaling. Use `--geometry-normalisation none` to recover the previous
graph scaling.

The base mark-correlation adjacency remains zero-diagonal. `variogram` converts the
same pair weights to `L = D - A`; its diagonal is therefore part of the variogram
operator rather than a restored self-edge.

For cross programs, the normalised bipartite pair mass is
`sqrt(n_target*n_neighbour)`, matching the reciprocal default sample weight. Distance
normalisation learns one common target-neighbour distance profile across samples. If
target and neighbour masks overlap, same-observation pairs are now removed before
bipartite degree normalisation; use `--geometry-normalisation none` to retain historical
pair scaling, but biological self-pairs remain excluded under the zero-self-edge
semantics.

Multi-sample workflows no longer estimate an omitted radius from the first sample
only. The fallback is based on the median sample nearest-neighbour spacing; an explicit
physical radius is recommended for comparative analyses.

## Preserved structure

- `tanpopo.kernel`: Wendland kernel and neighbour spacing.
- `tanpopo.projection`: group centering and covariate residualisation.
- `tanpopo.data`: sample preparation, label ordering, shared-gene preprocessing.
- `tanpopo.operators`: matrix-free gene/spot operators.
- `tanpopo.models`: array-level model classes.
- `tanpopo.io`: AnnData loading/storage and Tanpopo key names.
- `tanpopo.cli`: Typer annotations and enums.
- `tanpopo.workflows`: command-line workflows.

The central `prepare_samples -> operator -> solver -> modes/loadings` design is retained.

## Changed semantics

### Spatial graph

Old: Wendland self edges were retained.

New: square graphs are zero-diagonal by default and symmetrically degree-normalised.

### Gene normalization

Old: `alpha` scaled by the diagonal of the spatial gene operator.

New:

- `covariance`: no gene magnitude normalization;
- `gene_standardized`: ordinary residual expression variance;
- `gain`: generalized spatial gain in a truncated expression basis.

`--alpha 0` is temporarily accepted as an alias for `--objective covariance`.

### Cell-type workflows

Old: hard and soft masks were both available.

New: `spatial-programs --labels` is hard-mask only. Niche/context analysis moves to
`cross-programs`, which returns a paired target and neighbour program.

### Multi-sample workflows

Shared and differential workflows now support a selected cell type directly. Differential
sample contrasts can use sample-label permutations; cells are not treated as independent
replicates.

## Removed from the core CLI

- marker programs;
- differential label-versus-rest programs;
- clustering;
- plotting;
- soft masking.

These can be restored later as auxiliary tools if they remain scientifically useful, but
they are deliberately excluded from the narrowed method surface.

## Output compatibility

For `spatial-programs --cmd-id spatial`, the primary loading key remains:

`adata.varm['tanpopo_spatial_gene_loadings']`

and cell-type suffixing remains compatible with the existing simulation evaluator, e.g.:

`adata.varm['tanpopo_spatial_Fibroblasts_gene_loadings']`.

The old per-gene `*_gene_scores` scalar column is renamed to
`*_gene_spatial_covariance` because a zero-diagonal spatial autocovariance may be negative.
