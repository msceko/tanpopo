# Changelog

## 0.2.0

- Redefined the core spatial graph as zero-diagonal.
- Added symmetric degree normalisation for square graphs and bipartite normalisation
  for cross-population graphs.
- Replaced `alpha` with explicit `covariance`, `gene_standardized`, and `gain`
  objectives.
- `gene_standardized` now scales by ordinary residual expression variance.
- Added truncated-SVD generalized spatial-gain solver.
- Hard masking is the only cell-type restriction in `spatial-programs`.
- Added paired target-neighbour `cross-programs`.
- Added exact total/between/within/coupling label decomposition.
- Shared and differential workflows can operate within a selected cell type.
- Differential sample contrasts use biological-sample weights and optional sample-label
  permutation tests.
- Preprocessing now acts on the selected AnnData layer.
- Removed clustering, plotting, marker programs, and ambiguous soft-mask workflows from
  the core CLI.
