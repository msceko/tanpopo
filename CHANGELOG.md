# Changelog

## 0.3.0

Release-oriented geometry branch cleanup.

### Scientific core

- Made geometry-standardised spatial covariance the single primary statistic.
- Kept `covariance`, `gene_standardized` and `gain` gene-space objectives.
- Retained square, shared, differential, cross, shared-cross and differential-cross
  models.
- Removed mark-variogram support and the analytical `null_center` experimental path.
- Retained exact label decomposition and PCA as diagnostics/baselines rather than primary
  program-discovery methods.
- Standardised square and bipartite pair measures through common `none`, `mass` and
  `distance` geometry-normalisation modes.

### Inference and diagnostics

- Added maximum-statistic permutation inference and automatic mode selection.
- Added `--components auto`, `--max-components` and `--mode-alpha` to core program
  workflows.
- Added effective spatial dimension as a participation-ratio diagnostic.
- Cached projected expression bases for `gain` permutation tests so automatic inference
  recomputes only the low-dimensional spatial numerator.
- Integrated spatial-information analysis on top of the native gain model, projector,
  geometry and inference machinery instead of maintaining a second implementation.

### Result and utility layer

- Added versioned result schema `schema_version = 1` and typed `ProgramResult` /
  `CrossProgramResult` loaders.
- Added schema-driven `inspect`, `export`, `compare`, `compare-spots`, `plot`, `cluster`
  and `gsea` utilities.
- Restored a standalone preprocessing workflow.
- Added optional dependency groups for plotting, Leiden clustering and GSEA.
- Renamed the public CLI around a small related command family: `programs`, `shared`,
  `differential`, `cross`, `cross-shared`, `cross-differential`, `decompose-labels` and
  `baseline-pca`.

### Cleanup

- Removed deprecated `alpha` compatibility handling, legacy class aliases and historical
  result-key aliases.
- Removed the old migration document; this release intentionally establishes a new
  schema/API because no external compatibility contract exists yet.
- Bumped the package version to `0.3.0`.
