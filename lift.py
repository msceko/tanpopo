import numpy as np


class ModeLift:
    """
    Maps gene-space eigenvectors to spot embeddings and gene loadings.

    Separated from GeneKernel so that class is solely responsible for the
    matrix-vector products used inside eigsh.
    """

    def __init__(self, gene_kernel, dtype=np.float64):
        self.gene_kernel = gene_kernel
        self.dtype = dtype

    def lift(self, V, eigenvalues=None, normalise=True):
        X = self.gene_kernel._prepare(V)
        phi = self.gene_kernel.W @ X
        loadings = X.copy()

        if normalise and eigenvalues is not None:
            scale = np.sqrt(np.maximum(np.abs(eigenvalues), 1e-12))[None, :]
            phi = phi / scale
            loadings = loadings / scale

        return np.asarray(phi, dtype=self.dtype), np.asarray(loadings, dtype=self.dtype)


class ConcatenatedModeLifter:
    """
    Lift modes from one concatenated GeneKernel and return per-sample spot modes
    in original input order.
    """

    def __init__(self, gene_kernel, samples, dtype=np.float64):
        self.gene_kernel = gene_kernel
        self.samples = samples
        self.dtype = dtype
        self.mode_lift = ModeLift(gene_kernel, dtype=dtype)

    def lift(self, eigenvectors, eigenvalues, normalise=True):
        phi, loadings = self.mode_lift.lift(
            eigenvectors,
            eigenvalues,
            normalise=normalise,
        )

        spot_modes = []
        start = 0
        for s in self.samples:
            stop = start + s.n_spots
            spot_modes.append(phi[start:stop][s.inv_order])
            start = stop

        return spot_modes, loadings


class SamplewiseModeLifter:
    """
    Lift modes separately through one GeneKernel per sample.

    If normalise_by='pooled', loadings are common and a single matrix is returned.
    If normalise_by='sample', loadings are sample-specific and a list is returned.
    """

    def __init__(self, ops, samples, normalise_by="sample", dtype=np.float64):
        self.ops = ops
        self.samples = samples
        self.normalise_by = normalise_by
        self.dtype = dtype

    def lift(self, eigenvectors, eigenvalues, normalise=True):
        spot_modes = []
        gene_loadings = []

        for s, (_, Gs) in zip(self.samples, self.ops):
            lifter = ModeLift(Gs, dtype=self.dtype)
            phi, load = lifter.lift(
                eigenvectors,
                eigenvalues,
                normalise=normalise,
            )
            spot_modes.append(phi[s.inv_order])
            gene_loadings.append(load)

        if self.normalise_by == "pooled":
            gene_loadings = gene_loadings[0]

        return spot_modes, gene_loadings
