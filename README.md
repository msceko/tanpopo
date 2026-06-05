![logo](tanpopo.webp)

tanpopo is a python package for discovering spatially structured gene programs in spatial transcriptomics data. It computes spatial gene eigenmodes from an expression matrix and spot coordinates, using sparse spatial kernels to identify genes and programs whose expression varies coherently across tissue space.

## Installation

```
pip install git+https://github.com/msceko/tanpopo.git
```

Check if it has installed correctly by running `tanpopo` to get a list of available tools:

```
╭─ Commands ─────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ spatial-programs              Spatial gene programs in a sample, optionally within labels.                         │
│ shared-programs               Shared spatial gene programs across multiple samples.                                │
│ differential-label-programs   Differential gene programs enriched in one label versus the rest.                    │
│ differential-sample-programs  Differential gene programs enriched in one sample group or condition versus another. │
│ marker-programs               Marker gene programs that distinguish labelled domains or cell types.                │
│ estimate-spacing              Compute average distance to closest neighbour.                                       │
│ cluster                       Cluster spots or genes based on spatial gene programs.                               │
│ plot                          Plot spatial gene programs.                                                          │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

> [!NOTE]
> You may need to deactivate and source your environment for `tanpopo` to appear.
