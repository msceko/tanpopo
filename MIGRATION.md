# Migration notes: Tanpopo 0.1 -> narrowed 0.2

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
