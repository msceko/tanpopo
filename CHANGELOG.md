# Unreleased

- Added geometry-standardised square spatial operators with `none`, `mass`, and
  distance-profile normalisation.
- Added matrix-valued `mark_correlation` and graph-Laplacian `variogram` statistics.
- Added optional exact random-labelling null centering for centred mark correlation.
- Multi-sample fallback radius now uses the median sample nearest-neighbour spacing
  rather than the first sample only.
- Added pair-geometry diagnostics and generic spatial-statistic output keys.
- Added numerical tests for geometry balancing, variogram identities, null centering,
  differential variogram recovery, and variogram label decomposition.
- Generalised pair-measure geometry standardisation to rectangular target-neighbour
  kernels; cross programs now support `none`, `mass`, and shared distance-profile
  normalisation with target mass `sqrt(n_target*n_neighbour)`.
- Cross workflows remove overlapping target/neighbour self-pairs by original
  observation ID before bipartite degree normalisation and report separate row/column
  geometry diagnostics.
- Added `differential-cross-programs` and `DifferentialCrossSpatialProgramModel` for
  biological-sample group contrasts of target-neighbour spatial cross-covariance, with
  geometry balancing, all cross objectives, per-sample/group mode statistics, and
  max-singular-value permutation inference.
- Differential square and cross workflows now use differences of group means
  (`+1/n_A`, `-1/n_B`) rather than differences of sample sums, removing replicate-count
  bias for unequal groups.
- Refactored shared/differential cross result storage and multi-sample cross input
  preparation to avoid duplicated workflow logic.

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
