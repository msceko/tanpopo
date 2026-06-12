import numpy as np
from scipy.sparse.linalg import eigsh

from tanpopo.data import prepare_samples, concatenate_samples
from tanpopo.operators import (
    SpotOperatorSpec,
    GeneKernel,
    SampleOperatorBuilder,
    SumGeneOperator,
    gene_scale_from_diag,
)
from tanpopo.lift import ConcatenatedModeLifter, SamplewiseModeLifter
from tanpopo.utils import timed


class KPCAModel:
    """
    Base class for all spatial gene KPCA models.

    Implements the shared fit pipeline:
        prepare_samples -> _build_operator -> eigensolver -> lift

    Subclasses implement _build_operator(samples), which must:
      - Construct and return a (SumGeneOperator, lift_fn) tuple
      - Store any score attributes (e.g. gene_spatial_scores_) on self as a
        side effect, since scores depend on intermediate values only available
        during operator construction

    lift_fn(eigenvectors, eigenvalues) -> (spot_modes, gene_loadings)
    where spot_modes is a list of per-sample arrays.

    The class attribute _signed controls whether the eigensolver targets the
    largest-magnitude (indefinite/contrast) or largest-positive (PSD) eigenvalues.

    Subclasses implement _build_operator(samples), which must return:

        G, lifter

    where:
        G      : SumGeneOperator or compatible object with as_scipy()
        lifter : object with lift(eigenvectors, eigenvalues, normalise=True)

    The lifter must return:
        spot_modes, gene_loadings
    """

    _signed = False

    def _orient_vectors(self, V):
        pos = np.sum((V > 0) * V**2, axis=0)
        neg = np.sum((V < 0) * V**2, axis=0)
        signs = np.where(pos >= neg, 1.0, -1.0)
        return V * signs[None, :]

    def _fit_operator(self, G, n_components, signed=False, tol=0, maxiter=None, which=None):
        which = ("LM" if signed else "LA") if which is None else which

        vals, vecs = eigsh(
            G.as_scipy(),
            k=n_components,
            which=which,
            tol=tol,
            maxiter=maxiter,
        )

        if signed:
            if which == "LM":
                idx = np.argsort(np.abs(vals))[::-1]
            elif which == "LA":
                idx = np.argsort(vals)[::-1]
            elif which == "SA":
                idx = np.argsort(vals)
            else:
                raise ValueError("which must be one of {'LM', 'LA', 'SA'}")
            vals = vals[idx]
        else:
            idx = np.argsort(vals)[::-1]
            vals = np.maximum(vals[idx], 0.0)

        vecs = self._orient_vectors(vecs[:, idx])

        self.eigenvalues = vals
        self.eigenvectors = vecs
        self.gene_scores = vecs * np.sqrt(np.maximum(np.abs(vals), 1e-12))[None, :]

        if signed:
            self.mode_sign = np.sign(vals)
            self.positive_mode_index = np.flatnonzero(vals > 0)
            self.negative_mode_index = np.flatnonzero(vals < 0)

        return self

    def fit(self, W, coords, n_components, labels=None, covariates=None, tol=0, maxiter=None):
        with timed("Preparing samples", self.verbose):
            self.samples = prepare_samples(W, coords, self.radius, labels, covariates, self.dtype)

        with timed("Building operator", self.verbose):
            G, lifter = self._build_operator(self.samples)

        with timed("Solving eigenproblem", self.verbose):
            self._fit_operator(G, n_components, signed=self._signed, tol=tol, maxiter=maxiter)

        self.spot_modes, self.gene_loadings = lifter.lift(
            self.eigenvectors, self.eigenvalues, normalise=True
        )
        return self

    def _build_operator(self, samples):
        raise NotImplementedError


class SpatialGeneKPCA(KPCAModel):
    """
    Concatenated/block-diagonal spatial gene KPCA.

    Uses one projected spot operator: G = B^T S B
    """

    def __init__(
        self,
        radius,
        spot_operator="sample",
        alpha=1.0,
        gene_center=True,
        kernel="wendland_c2",
        eps=1e-12,
        covariates_tol=1e-10,
        dtype=np.float64,
        verbose=False,
    ):
        self.radius = radius
        self.spot_operator = spot_operator
        self.alpha = alpha
        self.gene_center = gene_center
        self.kernel = kernel
        self.eps = eps
        self.covariates_tol = covariates_tol
        self.dtype = dtype
        self.verbose = verbose

    def _build_operator(self, samples):
        Wc, Kc, sample_groups, label_groups, covc = concatenate_samples(samples)

        spec = SpotOperatorSpec(self.spot_operator)
        S = spec.build(Kc, sample_groups, label_groups, covc, self.covariates_tol, self.dtype)
        diag = np.maximum(S.gene_spatial_variance(Wc), self.eps)
        scale = gene_scale_from_diag(diag, self.alpha, self.eps)

        op = GeneKernel(Wc, S, scale, self.gene_center, self.dtype)
        G = SumGeneOperator([(1.0, op)], dtype=self.dtype)

        self.gene_spatial_scores_ = diag

        lifter = ConcatenatedModeLifter(
            gene_kernel=op,
            samples=samples,
            dtype=self.dtype,
        )

        return G, lifter

    def gene_spatial_scores(self):
        return self.gene_spatial_scores_.copy()


class SpatialGeneContrastKPCA(KPCAModel):
    """
    Contrast two spot operators on the same concatenated expression matrix:

        G = B^T S_left B - B^T S_right B

    Example:
        between labels: S_sample - S_label
    """

    _signed = True

    def __init__(
        self,
        radius,
        left_operator,
        right_operator,
        alpha=1.0,
        gene_center=True,
        normalise_by="left",
        kernel="wendland_c2",
        eps=1e-12,
        covariates_tol=1e-10,
        dtype=np.float64,
        verbose=False,
    ):
        self.radius = radius
        self.left_operator = left_operator
        self.right_operator = right_operator
        self.alpha = alpha
        self.gene_center = gene_center
        self.normalise_by = normalise_by
        self.kernel = kernel
        self.eps = eps
        self.covariates_tol = covariates_tol
        self.dtype = dtype
        self.verbose = verbose

    @classmethod
    def between_labels(cls, radius, **kwargs):
        return cls(radius, left_operator="sample", right_operator="label", **kwargs)

    @classmethod
    def between_samples(cls, radius, **kwargs):
        return cls(radius, left_operator="none", right_operator="sample", **kwargs)

    def _build_operator(self, samples):
        Wc, Kc, sample_groups, label_groups, covc = concatenate_samples(samples)

        def make_S(centering):
            spec = SpotOperatorSpec(centering)
            return spec.build(
                Kc, sample_groups, label_groups, covc, self.covariates_tol, self.dtype
            )

        S_left = make_S(self.left_operator)
        S_right = make_S(self.right_operator)

        diag_left = np.maximum(S_left.gene_spatial_variance(Wc), self.eps)
        diag_right = np.maximum(S_right.gene_spatial_variance(Wc), self.eps)

        ref = diag_left if self.normalise_by == "left" else diag_right
        scale = gene_scale_from_diag(ref, self.alpha, self.eps)

        G_left = GeneKernel(Wc, S_left, scale, self.gene_center, self.dtype)
        G_right = GeneKernel(Wc, S_right, scale, self.gene_center, self.dtype)

        G = SumGeneOperator([(1.0, G_left), (-1.0, G_right)], dtype=self.dtype)

        self.gene_spatial_scores_left_ = diag_left
        self.gene_spatial_scores_right_ = diag_right
        self.gene_spatial_scores_contrast_ = diag_left - diag_right

        # The spot operator is irrelevant for lifting; ModeLift only uses W,
        # gene_scale, and gene_center. G_left and G_right share these.
        lifter = ConcatenatedModeLifter(gene_kernel=G_left, samples=samples, dtype=self.dtype)

        return G, lifter

    def gene_spatial_scores(self, kind="contrast"):
        if kind == "left":
            return self.gene_spatial_scores_left_.copy()
        if kind == "right":
            return self.gene_spatial_scores_right_.copy()
        if kind == "contrast":
            return self.gene_spatial_scores_contrast_.copy()
        raise ValueError("kind must be 'left', 'right', or 'contrast'")


class SpatialGeneSampleCombinedKPCA(KPCAModel):
    """
    Sample-wise combined modes:

        G = sum_s w_s B_s^T S_s B_s

    This avoids block-diagonal construction and gives explicit sample-balanced
    combined modes.
    """

    def __init__(
        self,
        radius,
        spot_operator="sample",
        sample_weighting="trace",
        normalise_by="sample",
        alpha=1.0,
        gene_center=True,
        kernel="wendland_c2",
        eps=1e-12,
        covariates_tol=1e-10,
        dtype=np.float64,
        verbose=False,
    ):
        self.radius = radius
        self.spot_operator = spot_operator
        self.sample_weighting = sample_weighting
        self.normalise_by = normalise_by
        self.alpha = alpha
        self.gene_center = gene_center
        self.kernel = kernel
        self.eps = eps
        self.covariates_tol = covariates_tol
        self.dtype = dtype
        self.verbose = verbose

    def _build_operator(self, samples):
        builder = SampleOperatorBuilder(
            self.spot_operator,
            self.sample_weighting,
            self.normalise_by,
            self.alpha,
            self.gene_center,
            self.eps,
            self.covariates_tol,
            self.dtype,
        )
        ops, diags, coeff = builder.build(samples)
        G = SumGeneOperator(ops, dtype=self.dtype)

        self.sample_coefficients_ = coeff
        self.gene_spatial_scores_samples_ = diags
        self.gene_spatial_scores_ = sum(w * d for (w, _), d in zip(ops, diags))

        lifter = SamplewiseModeLifter(
            ops=ops,
            samples=samples,
            normalise_by=self.normalise_by,
            dtype=self.dtype,
        )

        return G, lifter

    def gene_spatial_scores(self, kind="combined"):
        if kind in {"combined", "model"}:
            return self.gene_spatial_scores_.copy()
        if kind == "samples":
            return self.gene_spatial_scores_samples_.copy()
        raise ValueError("kind must be 'combined' or 'samples'")


class SpatialGeneSampleContrastKPCA(KPCAModel):
    """
    Sample-wise contrast modes:

        G = sum_{s in positive} w_s G_s - sum_{s in negative} w_s G_s

    Positive eigenvalues are enriched on the positive side.
    Negative eigenvalues are enriched on the negative side.

    Sits at the same level as SpatialGeneSampleCombinedKPCA rather than
    inheriting from it; shared logic lives in SampleOperatorBuilder.
    """

    _signed = True

    def __init__(
        self,
        radius,
        positive_samples,
        negative_samples,
        spot_operator="sample",
        sample_weighting="trace",
        normalise_by="sample",
        alpha=1.0,
        gene_center=True,
        kernel="wendland_c2",
        eps=1e-12,
        covariates_tol=1e-10,
        dtype=np.float64,
        verbose=False,
    ):
        self.radius = radius
        self.positive_samples = list(positive_samples)
        self.negative_samples = list(negative_samples)
        self.spot_operator = spot_operator
        self.sample_weighting = sample_weighting
        self.normalise_by = normalise_by
        self.alpha = alpha
        self.gene_center = gene_center
        self.kernel = kernel
        self.eps = eps
        self.covariates_tol = covariates_tol
        self.dtype = dtype
        self.verbose = verbose

    @classmethod
    def paired(cls, radius, positive_sample, negative_sample, **kwargs):
        return cls(radius, [positive_sample], [negative_sample], **kwargs)

    @classmethod
    def from_condition_labels(
        cls, radius, condition_labels, positive_label, negative_label, **kwargs
    ):
        labels = np.asarray(condition_labels)
        pos = np.flatnonzero(labels == positive_label).tolist()
        neg = np.flatnonzero(labels == negative_label).tolist()
        return cls(radius, pos, neg, **kwargs)

    def _build_operator(self, samples):
        signed = np.zeros(len(samples), dtype=float)
        signed[self.positive_samples] = 1.0 / len(self.positive_samples)
        signed[self.negative_samples] = -1.0 / len(self.negative_samples)

        builder = SampleOperatorBuilder(
            self.spot_operator,
            self.sample_weighting,
            self.normalise_by,
            self.alpha,
            self.gene_center,
            self.eps,
            self.covariates_tol,
            self.dtype,
        )
        ops, diags, coeff = builder.build(samples, signed_coefficients=signed)
        G = SumGeneOperator(ops, dtype=self.dtype)

        self.sample_coefficients_ = coeff
        self.gene_spatial_scores_samples_ = diags
        self.gene_spatial_scores_weighted_ = np.vstack([w * d for (w, _), d in zip(ops, diags)])
        self.gene_spatial_scores_contrast_ = self.gene_spatial_scores_weighted_.sum(axis=0)

        lifter = SamplewiseModeLifter(
            ops=ops, samples=samples, normalise_by=self.normalise_by, dtype=self.dtype
        )

        return G, lifter

    def gene_spatial_scores(self, kind="contrast"):
        if kind in {"contrast", "model"}:
            return self.gene_spatial_scores_contrast_.copy()
        if kind == "samples":
            return self.gene_spatial_scores_samples_.copy()
        if kind == "weighted":
            return self.gene_spatial_scores_weighted_.copy()
        raise ValueError("kind must be 'contrast', 'samples', or 'weighted'")
