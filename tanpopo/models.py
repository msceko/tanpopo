from __future__ import annotations

from dataclasses import replace

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh, svd
from scipy.sparse.linalg import eigsh, svds

from tanpopo.data import (
    Groups,
    concatenate_cross_samples,
    concatenate_samples,
    prepare_cross_samples,
    prepare_samples,
)
from tanpopo.geometry import VALID_GEOMETRY_NORMALISATIONS
from tanpopo.inference import select_modes
from tanpopo.operators import (
    BetweenLabelTransform,
    BilinearGeneOperator,
    CrossGeneOperator,
    GeneKernel,
    ProjectedKernel,
    ProjectorTransform,
    SpotOperatorSpec,
    WithinLabelTransform,
    expression_scale_from_variance,
    gene_expression_variance,
    gene_spatial_covariance_diag,
    projected_svd,
)
from tanpopo.projection import SpotProjector
from tanpopo.utils import column_normalize, timed


VALID_OBJECTIVES = {"covariance", "gene_standardized", "gain"}


def _group_contrast_coefficients(n_samples, positive_samples, negative_samples):
    """Return a difference-of-group-means contrast over biological samples.

    Positive/group-A coefficients sum to +1 and negative/group-B coefficients
    sum to -1. This keeps the estimand invariant to unequal replicate counts.
    """
    n_samples = int(n_samples)
    positive = np.asarray(list(positive_samples), dtype=int)
    negative = np.asarray(list(negative_samples), dtype=int)
    if positive.size == 0 or negative.size == 0:
        raise ValueError("Both differential groups must contain at least one sample")
    if np.any(positive < 0) or np.any(positive >= n_samples):
        raise ValueError("positive_samples contains an out-of-range sample index")
    if np.any(negative < 0) or np.any(negative >= n_samples):
        raise ValueError("negative_samples contains an out-of-range sample index")
    repeated = len(np.unique(positive)) != len(positive) or len(np.unique(negative)) != len(
        negative
    )
    if repeated:
        raise ValueError("Sample indices must not be repeated within a differential group")
    if np.intersect1d(positive, negative).size:
        raise ValueError("Differential groups must be disjoint")
    assigned = np.zeros(n_samples, dtype=bool)
    assigned[positive] = True
    assigned[negative] = True
    if not np.all(assigned):
        raise ValueError("Every sample must be assigned to exactly one differential group")
    contrast = np.zeros(n_samples, dtype=float)
    contrast[positive] = 1.0 / positive.size
    contrast[negative] = -1.0 / negative.size
    return contrast


def _permuted_group_contrast(rng, n_samples, n_positive):
    positive = rng.choice(n_samples, size=n_positive, replace=False)
    negative = np.setdiff1d(np.arange(n_samples), positive, assume_unique=True)
    return _group_contrast_coefficients(n_samples, positive, negative)


def _sample_weights(samples, mode):
    if mode == "none":
        return np.ones(len(samples), dtype=float)
    if mode == "n_spots":
        return 1.0 / np.asarray([s.n_spots for s in samples], dtype=float)
    raise ValueError("sample_weighting must be 'none' or 'n_spots'")


def _weighted_block_kernel(samples, weights, signs=None):
    signs = np.ones(len(samples)) if signs is None else np.asarray(signs, dtype=float)
    return sp.block_diag(
        [float(w * sign) * sample.K for sample, w, sign in zip(samples, weights, signs)],
        format="csr",
    )


def _row_scale(samples, weights):
    return np.concatenate(
        [np.full(s.n_spots, np.sqrt(float(w)), dtype=float) for s, w in zip(samples, weights)]
    )


def _cross_sample_weights(samples, mode):
    if mode == "none":
        return np.ones(len(samples), dtype=float)
    if mode == "n_spots":
        effective_n = np.sqrt(
            np.asarray([sample.n_target * sample.n_neighbour for sample in samples], dtype=float)
        )
        return 1.0 / effective_n
    raise ValueError("sample_weighting must be 'none' or 'n_spots'")


def _cross_row_scale(samples, weights, side):
    attr = "n_target" if side == "target" else "n_neighbour"
    return np.concatenate(
        [
            np.full(getattr(sample, attr), np.sqrt(float(weight)), dtype=float)
            for sample, weight in zip(samples, weights)
        ]
    )


def _split_cross_modes(phi, samples, side):
    attr = "n_target" if side == "target" else "n_neighbour"
    output = []
    start = 0
    for sample in samples:
        stop = start + getattr(sample, attr)
        output.append(np.asarray(phi[start:stop]))
        start = stop
    return output


def _orient_cross_pairs(target_loadings, neighbour_loadings):
    target_loadings = column_normalize(target_loadings)
    neighbour_loadings = column_normalize(neighbour_loadings)
    orient_idx = np.argmax(np.abs(target_loadings), axis=0)
    pair_sign = np.sign(target_loadings[orient_idx, np.arange(target_loadings.shape[1])])
    pair_sign[pair_sign == 0] = 1
    return (
        target_loadings * pair_sign[None, :],
        neighbour_loadings * pair_sign[None, :],
    )


def _orient_columns(V):
    V = np.asarray(V, dtype=float)
    if V.ndim == 1:
        V = V[:, None]
    idx = np.argmax(np.abs(V), axis=0)
    signs = np.sign(V[idx, np.arange(V.shape[1])])
    signs[signs == 0] = 1
    return V * signs[None, :]


def _split_spot_modes(phi, samples):
    output = []
    start = 0
    for sample in samples:
        stop = start + sample.n_spots
        output.append(np.asarray(phi[start:stop])[sample.inv_order])
        start = stop
    return output


def _random_label_permutation(sample, rng, within_labels=False):
    """Permutation of spatial identities for a prepared sample."""
    if not within_labels:
        return rng.permutation(sample.n_spots)
    permutation = np.arange(sample.n_spots)
    for offset, length in zip(sample.labels_groups.offsets, sample.labels_groups.lengths):
        indices = np.arange(int(offset), int(offset + length))
        permutation[indices] = rng.permutation(indices)
    return permutation


def _permuted_square_samples(samples, rng, within_labels=False):
    output = []
    for sample in samples:
        permutation = _random_label_permutation(sample, rng, within_labels=within_labels)
        K = sample.K[permutation][:, permutation].tocsr()
        output.append(replace(sample, K=K))
    return output


def _permuted_cross_samples(samples, rng):
    output = []
    for sample in samples:
        permutation = rng.permutation(sample.n_neighbour)
        K = sample.K[:, permutation].tocsr()
        output.append(replace(sample, K=K))
    return output


def _solve_gene_operator(op, n_components, signed=False, tol=0, maxiter=None):
    g = op.shape[0]
    k = min(int(n_components), g - 1)
    if k < 1:
        raise ValueError("At least two genes are required")
    which = "LM" if signed else "LA"
    vals, vecs = eigsh(op.as_scipy(), k=k, which=which, tol=tol, maxiter=maxiter)
    order = np.argsort(np.abs(vals))[::-1] if signed else np.argsort(vals)[::-1]
    return vals[order], _orient_columns(vecs[:, order])


def _solve_dense(H, n_components, signed=False):
    vals, vecs = eigh(np.asarray(H, dtype=float))
    order = np.argsort(np.abs(vals))[::-1] if signed else np.argsort(vals)[::-1]
    order = order[: min(int(n_components), len(order))]
    return vals[order], _orient_columns(vecs[:, order])


class SpatialProgramModel:
    """Geometry-standardised multigene spatial covariance programs.

    The core statistic is a zero-self-edge pair covariance ``Y.T @ A @ Y``.
    Geometry normalisation is applied to ``A`` before expression enters the
    gene-space eigensolver.

    Objectives
    ----------
    covariance
        Maximise the geometry-standardised cross-cell spatial covariance
        ``v.T @ Y.T @ A @ Y @ v``.
    gene_standardized
        The same numerator after scaling each gene by ordinary residual expression
        variance. The denominator is positive expression variance rather than spatial
        autocovariance.
    gain
        Solve the spatial-gain generalized problem on a truncated expression
        subspace: Y^T A Y v = lambda (Y^T Y + ridge I) v.
    """

    def __init__(
        self,
        radius,
        objective="covariance",
        spot_operator="sample",
        sample_weighting="none",
        expression_rank=50,
        gain_ridge=1e-8,
        graph_normalisation="symmetric",
        geometry_normalisation="distance",
        distance_bins=10,
        covariates_tol=1e-10,
        block_size=256,
        dtype=np.float64,
        verbose=False,
    ):
        if objective not in VALID_OBJECTIVES:
            raise ValueError(f"objective must be one of {sorted(VALID_OBJECTIVES)}")
        if geometry_normalisation not in VALID_GEOMETRY_NORMALISATIONS:
            raise ValueError(
                "geometry_normalisation must be one of " f"{sorted(VALID_GEOMETRY_NORMALISATIONS)}"
            )
        if int(distance_bins) < 1:
            raise ValueError("distance_bins must be positive")
        self.radius = float(radius)
        self.objective = objective
        self.spot_operator = spot_operator
        self.sample_weighting = sample_weighting
        self.expression_rank = int(expression_rank)
        self.gain_ridge = float(gain_ridge)
        self.graph_normalisation = graph_normalisation
        self.geometry_normalisation = geometry_normalisation
        self.distance_bins = int(distance_bins)
        self.covariates_tol = covariates_tol
        self.block_size = block_size
        self.dtype = np.dtype(dtype)
        self.verbose = verbose

    def fit(
        self,
        W,
        coords,
        n_components,
        labels=None,
        covariates=None,
        masks=None,
        tol=0,
        maxiter=None,
        signs=None,
    ):
        with timed("Preparing samples", self.verbose):
            self.samples = prepare_samples(
                W,
                coords,
                self.radius,
                labels=labels,
                covariates=covariates,
                masks=masks,
                dtype=self.dtype,
                graph_normalisation=self.graph_normalisation,
                geometry_normalisation=self.geometry_normalisation,
                distance_bins=self.distance_bins,
            )
        self.geometry_diagnostics_ = [sample.geometry_diagnostics for sample in self.samples]
        if signs is not None and len(signs) != len(self.samples):
            raise ValueError("signs must contain one value per sample")
        signed = signs is not None and np.any(np.asarray(signs) < 0)
        with timed("Building and solving spatial objective", self.verbose):
            if self.objective == "gain":
                self._fit_gain(n_components, signs=signs, signed=signed, tol=tol)
            else:
                self._fit_covariance(
                    n_components,
                    signs=signs,
                    signed=signed,
                    tol=tol,
                    maxiter=maxiter,
                )
        return self

    def _concatenated_state(self, signs=None):
        Wc, _, sample_groups, label_groups, covc = concatenate_samples(self.samples)
        weights = _sample_weights(self.samples, self.sample_weighting)
        K = _weighted_block_kernel(self.samples, weights, signs=signs)
        spec = SpotOperatorSpec(self.spot_operator)
        S = spec.build(
            K,
            sample_groups=sample_groups,
            label_groups=label_groups,
            covariates=covc,
            tol=self.covariates_tol,
            dtype=self.dtype,
        )
        return Wc, S, sample_groups, label_groups, covc, weights

    def _fit_covariance(self, n_components, signs, signed, tol, maxiter):
        Wc, S, _, _, _, weights = self._concatenated_state(signs=signs)
        scale = None
        if self.objective == "gene_standardized":
            # Expression standardisation uses positive ordinary residual variance.
            W_for_scale = Wc.copy().tocsr()
            rowscale = _row_scale(self.samples, weights)
            W_for_scale = sp.diags(rowscale, format="csr") @ W_for_scale
            # Build the same centering/covariate projector but no spatial kernel is needed.
            variance = gene_expression_variance(
                W_for_scale, S.projector, block_size=self.block_size, eps=1e-12
            )
            scale = expression_scale_from_variance(variance)
            self.gene_expression_variance_ = variance

        op = GeneKernel(Wc, S, gene_scale=scale, dtype=self.dtype)
        vals, solver_vectors = _solve_gene_operator(
            op, n_components, signed=signed, tol=tol, maxiter=maxiter
        )
        loadings = column_normalize(op.loadings_from_solver_vectors(solver_vectors))
        phi = S.projector.apply(Wc @ loadings)

        self.eigenvalues = vals
        self.eigenvectors = solver_vectors
        self.gene_loadings = loadings
        self.spot_modes = _split_spot_modes(phi, self.samples)
        self.gene_scores = loadings * np.sqrt(np.maximum(np.abs(vals), 1e-12))[None, :]
        self.gene_spatial_scores_ = gene_spatial_covariance_diag(Wc, S, block_size=self.block_size)
        self.sample_coefficients_ = weights * (
            np.ones(len(weights)) if signs is None else np.asarray(signs, dtype=float)
        )

    def _prepare_gain_basis(self, n_components, tol):
        """Cache the expression-side basis shared by all spatial permutations."""
        Wc, _, sample_groups, label_groups, covc = concatenate_samples(self.samples)
        weights = _sample_weights(self.samples, self.sample_weighting)
        base_K = sp.block_diag([sample.K for sample in self.samples], format="csr")
        spec = SpotOperatorSpec(self.spot_operator)
        S_base = spec.build(
            base_K,
            sample_groups=sample_groups,
            label_groups=label_groups,
            covariates=covc,
            tol=self.covariates_tol,
            dtype=self.dtype,
        )
        row_scale = _row_scale(self.samples, weights)
        rank = max(self.expression_rank, int(n_components) + 2)
        U, singular, Vt = projected_svd(
            Wc,
            S_base.projector,
            rank=rank,
            row_scale=row_scale,
            dtype=self.dtype,
            tol=tol,
        )
        denom = np.sqrt(np.square(singular) + self.gain_ridge)
        gain_scale = np.divide(singular, denom, out=np.zeros_like(singular), where=denom > 0)
        cache = {
            "Wc": Wc,
            "sample_groups": sample_groups,
            "label_groups": label_groups,
            "covariates": covc,
            "weights": weights,
            "spec": spec,
            "S_base": S_base,
            "U": U,
            "singular": singular,
            "Vt": Vt,
            "denom": denom,
            "gain_scale": gain_scale,
        }
        self._gain_basis_ = cache
        return cache

    def _gain_solve(self, cache, kernels, coefficients, n_components, signed):
        K_signed = sp.block_diag(
            [float(coefficient) * K for K, coefficient in zip(kernels, coefficients)],
            format="csr",
        )
        S_signed = cache["spec"].build(
            K_signed,
            sample_groups=cache["sample_groups"],
            label_groups=cache["label_groups"],
            covariates=cache["covariates"],
            tol=self.covariates_tol,
            dtype=self.dtype,
        )
        U = cache["U"]
        gain_scale = cache["gain_scale"]
        H0 = U.T @ S_signed.apply(U)
        H = gain_scale[:, None] * H0 * gain_scale[None, :]
        return _solve_dense(H, n_components, signed=signed)

    def _fit_gain(self, n_components, signs, signed, tol):
        cache = self._prepare_gain_basis(n_components, tol)
        coefficients = (
            np.ones(len(self.samples), dtype=float)
            if signs is None
            else np.asarray(signs, dtype=float)
        )
        vals, a = self._gain_solve(
            cache,
            [sample.K for sample in self.samples],
            coefficients,
            n_components,
            signed,
        )
        denom = cache["denom"]
        loadings = cache["Vt"].T @ ((1.0 / np.maximum(denom, 1e-12))[:, None] * a)
        loadings = column_normalize(loadings)
        phi = cache["S_base"].projector.apply(cache["Wc"] @ loadings)

        self.eigenvalues = vals
        self.eigenvectors = a
        self.gene_loadings = loadings
        self.spot_modes = _split_spot_modes(phi, self.samples)
        self.gene_scores = loadings * np.sqrt(np.maximum(np.abs(vals), 1e-12))[None, :]
        self.gene_spatial_scores_ = gene_spatial_covariance_diag(
            cache["Wc"], cache["S_base"], block_size=self.block_size
        )
        self.expression_singular_values_ = cache["singular"]
        self.sample_coefficients_ = cache["weights"] * coefficients

    def gene_spatial_scores(self):
        return self.gene_spatial_scores_.copy()

    def truncate_modes(self, n_modes):
        """Keep the first ``n_modes`` fitted programs, allowing zero."""
        n_modes = int(n_modes)
        if n_modes < 0 or n_modes > len(self.eigenvalues):
            raise ValueError("n_modes is outside the fitted mode range")
        self.eigenvalues = self.eigenvalues[:n_modes]
        self.gene_loadings = self.gene_loadings[:, :n_modes]
        self.gene_scores = self.gene_scores[:, :n_modes]
        self.eigenvectors = self.eigenvectors[:, :n_modes]
        self.spot_modes = [mode[:, :n_modes] for mode in self.spot_modes]
        if hasattr(self, "permutation_pvalues_"):
            self.permutation_pvalues_ = self.permutation_pvalues_[:n_modes]
        self.selected_modes_ = n_modes
        return self

    def permutation_test(self, n_permutations=100, seed=0, tol=0, maxiter=None):
        """Random-label max-eigenvalue test for fitted spatial programs.

        Spatial identities are permuted independently within biological samples.
        For label-centred analyses the permutation is restricted within labels,
        testing residual spatial information beyond the label structure.
        """
        n_permutations = int(n_permutations)
        if n_permutations <= 0:
            self.permutation_pvalues_ = np.full(len(self.eigenvalues), np.nan)
            self.permutation_max_eigenvalues_ = np.empty(0)
            return self.permutation_pvalues_
        rng = np.random.default_rng(seed)
        k = len(self.eigenvalues)
        null = np.empty(n_permutations, dtype=float)
        within_labels = self.spot_operator == "label"
        for b in range(n_permutations):
            samples = _permuted_square_samples(self.samples, rng, within_labels=within_labels)
            if self.objective == "gain":
                values, _ = self._gain_solve(
                    self._gain_basis_,
                    [sample.K for sample in samples],
                    np.ones(len(samples), dtype=float),
                    k,
                    signed=False,
                )
                null[b] = float(values[0])
                continue
            tmp = SpatialProgramModel(
                self.radius,
                objective=self.objective,
                spot_operator=self.spot_operator,
                sample_weighting=self.sample_weighting,
                expression_rank=self.expression_rank,
                gain_ridge=self.gain_ridge,
                graph_normalisation=self.graph_normalisation,
                geometry_normalisation=self.geometry_normalisation,
                distance_bins=self.distance_bins,
                covariates_tol=self.covariates_tol,
                block_size=self.block_size,
                dtype=self.dtype,
                verbose=False,
            )
            tmp.samples = samples
            tmp._fit_covariance(k, signs=None, signed=False, tol=tol, maxiter=maxiter)
            null[b] = float(tmp.eigenvalues[0])
        observed = np.asarray(self.eigenvalues, dtype=float)
        selection = select_modes(observed, null, alpha=0.05)
        self.permutation_max_eigenvalues_ = null
        self.permutation_pvalues_ = selection.pvalues
        return self.permutation_pvalues_

    def select_modes(self, alpha=0.05):
        if not hasattr(self, "permutation_max_eigenvalues_"):
            raise RuntimeError("Run permutation_test before select_modes")
        observed = getattr(self, "_selection_observed_", self.eigenvalues)
        selection = select_modes(observed, self.permutation_max_eigenvalues_, alpha=alpha)
        self.mode_selection_ = selection
        return selection


class SharedSpatialProgramModel(SpatialProgramModel):
    def __init__(self, *args, sample_weighting="n_spots", **kwargs):
        super().__init__(*args, sample_weighting=sample_weighting, **kwargs)


class DifferentialSpatialProgramModel(SpatialProgramModel):
    def __init__(
        self,
        radius,
        positive_samples,
        negative_samples,
        *args,
        sample_weighting="n_spots",
        **kwargs,
    ):
        super().__init__(radius, *args, sample_weighting=sample_weighting, **kwargs)
        self.positive_samples = list(positive_samples)
        self.negative_samples = list(negative_samples)

    def fit(self, W, coords, n_components, labels=None, covariates=None, masks=None, **kwargs):
        n = len(W) if isinstance(W, (list, tuple)) else 1
        contrast = _group_contrast_coefficients(n, self.positive_samples, self.negative_samples)
        result = super().fit(
            W,
            coords,
            n_components,
            labels=labels,
            covariates=covariates,
            masks=masks,
            signs=contrast,
            **kwargs,
        )
        self.contrast_coefficients_ = contrast
        self._observed_signs = np.sign(contrast)
        return result

    def permutation_test(self, n_permutations=100, seed=0, tol=0, maxiter=None):
        """Two-sided sample-label permutation test using max-|eigenvalue| FWER control."""
        n_permutations = int(n_permutations)
        if n_permutations <= 0:
            self.permutation_pvalues_ = np.full(len(self.eigenvalues), np.nan)
            self.permutation_max_abs_eigenvalues_ = np.empty(0)
            return self.permutation_pvalues_
        rng = np.random.default_rng(seed)
        n = len(self.samples)
        n_positive = len(self.positive_samples)
        k = len(self.eigenvalues)
        null = np.empty(n_permutations, dtype=float)
        for b in range(n_permutations):
            contrast = _permuted_group_contrast(rng, n, n_positive)
            if self.objective == "gain":
                values, _ = self._gain_solve(
                    self._gain_basis_,
                    [sample.K for sample in self.samples],
                    contrast,
                    k,
                    signed=True,
                )
                null[b] = np.max(np.abs(values))
                continue
            tmp = SpatialProgramModel(
                self.radius,
                objective=self.objective,
                spot_operator=self.spot_operator,
                sample_weighting=self.sample_weighting,
                expression_rank=self.expression_rank,
                gain_ridge=self.gain_ridge,
                graph_normalisation=self.graph_normalisation,
                geometry_normalisation=self.geometry_normalisation,
                distance_bins=self.distance_bins,
                covariates_tol=self.covariates_tol,
                block_size=self.block_size,
                dtype=self.dtype,
                verbose=False,
            )
            tmp.samples = self.samples
            tmp._fit_covariance(k, signs=contrast, signed=True, tol=tol, maxiter=maxiter)
            null[b] = np.max(np.abs(tmp.eigenvalues))
        observed = np.abs(self.eigenvalues)
        pvalues = (1.0 + np.sum(null[:, None] >= observed[None, :], axis=0)) / (
            n_permutations + 1.0
        )
        self.permutation_max_abs_eigenvalues_ = null
        self.permutation_max_eigenvalues_ = null
        self._selection_observed_ = observed
        self.permutation_pvalues_ = pvalues
        return pvalues


class _CrossSpatialProgramBase:
    """Shared implementation for single- and multi-sample cross programs."""

    def __init__(
        self,
        radius,
        objective="covariance",
        sample_weighting="none",
        expression_rank=50,
        gain_ridge=1e-8,
        graph_normalisation="symmetric",
        geometry_normalisation="distance",
        distance_bins=10,
        covariates_tol=1e-10,
        block_size=256,
        dtype=np.float64,
        verbose=False,
    ):
        if objective not in VALID_OBJECTIVES:
            raise ValueError(f"objective must be one of {sorted(VALID_OBJECTIVES)}")
        if geometry_normalisation not in VALID_GEOMETRY_NORMALISATIONS:
            raise ValueError(
                "geometry_normalisation must be one of " f"{sorted(VALID_GEOMETRY_NORMALISATIONS)}"
            )
        if int(distance_bins) < 1:
            raise ValueError("distance_bins must be positive")
        self.radius = float(radius)
        self.objective = objective
        self.sample_weighting = sample_weighting
        self.expression_rank = int(expression_rank)
        self.gain_ridge = float(gain_ridge)
        self.graph_normalisation = graph_normalisation
        self.geometry_normalisation = geometry_normalisation
        self.distance_bins = int(distance_bins)
        self.covariates_tol = covariates_tol
        self.block_size = block_size
        self.dtype = np.dtype(dtype)
        self.verbose = verbose

    def _fit_prepared(self, samples, n_components, tol=0, contrast=None):
        self.samples = samples
        self.geometry_diagnostics_ = [sample.geometry_diagnostics for sample in samples]
        (
            W_target,
            W_neighbour,
            target_groups,
            neighbour_groups,
            cov_target,
            cov_neighbour,
        ) = concatenate_cross_samples(samples)
        weights = _cross_sample_weights(samples, self.sample_weighting)
        if contrast is None:
            contrast = np.ones(len(samples), dtype=float)
        else:
            contrast = np.asarray(contrast, dtype=float)
            if contrast.shape != (len(samples),):
                raise ValueError("contrast must contain one coefficient per sample")
        Pt = SpotProjector(
            W_target.shape[0],
            target_groups,
            cov_target,
            self.covariates_tol,
            dtype=self.dtype,
        )
        Pu = SpotProjector(
            W_neighbour.shape[0],
            neighbour_groups,
            cov_neighbour,
            self.covariates_tol,
            dtype=self.dtype,
        )

        if self.objective == "gain":
            self._fit_gain(
                W_target,
                W_neighbour,
                Pt,
                Pu,
                samples,
                weights,
                contrast,
                n_components,
                tol,
            )
        else:
            self._fit_covariance(
                W_target,
                W_neighbour,
                Pt,
                Pu,
                samples,
                weights,
                contrast,
                n_components,
                tol,
            )

        target_phi = Pt.apply(W_target @ self.target_loadings)
        neighbour_phi = Pu.apply(W_neighbour @ self.neighbour_loadings)
        self.target_modes = _split_cross_modes(target_phi, samples, "target")
        self.neighbour_modes = _split_cross_modes(neighbour_phi, samples, "neighbour")
        self.base_sample_weights_ = weights
        self.contrast_coefficients_ = contrast
        self.sample_coefficients_ = weights * contrast
        self.sample_mode_covariance_ = self._sample_mode_covariance()
        self.sample_mode_statistic_ = weights[:, None] * self.sample_mode_covariance_
        self.aggregate_mode_statistic_ = contrast @ self.sample_mode_statistic_
        return self

    def _fit_covariance(
        self,
        W_target,
        W_neighbour,
        Pt,
        Pu,
        samples,
        weights,
        contrast,
        n_components,
        tol,
    ):
        scale_target = scale_neighbour = None
        if self.objective == "gene_standardized":
            target_scale = _cross_row_scale(samples, weights, "target")
            neighbour_scale = _cross_row_scale(samples, weights, "neighbour")
            W_target_scaled = sp.diags(target_scale, format="csr") @ W_target
            W_neighbour_scaled = sp.diags(neighbour_scale, format="csr") @ W_neighbour
            var_target = gene_expression_variance(W_target_scaled, Pt, self.block_size, eps=1e-12)
            var_neighbour = gene_expression_variance(
                W_neighbour_scaled, Pu, self.block_size, eps=1e-12
            )
            scale_target = expression_scale_from_variance(var_target)
            scale_neighbour = expression_scale_from_variance(var_neighbour)
            self.target_gene_expression_variance_ = var_target
            self.neighbour_gene_expression_variance_ = var_neighbour

        K_weighted = sp.block_diag(
            [
                float(weight * coefficient) * sample.K
                for sample, weight, coefficient in zip(samples, weights, contrast)
            ],
            format="csr",
        )
        operator = CrossGeneOperator(
            W_target,
            W_neighbour,
            K_weighted,
            Pt,
            Pu,
            scale_target,
            scale_neighbour,
            dtype=self.dtype,
        )
        k = min(int(n_components), min(operator.shape) - 1)
        if k < 1:
            raise ValueError("At least two genes are required for cross-program analysis")
        U, singular, Vt = svds(operator, k=k, which="LM", tol=tol)
        order = np.argsort(singular)[::-1]
        singular, U, Vt = singular[order], U[:, order], Vt[order]
        target_loadings = U if scale_target is None else scale_target[:, None] * U
        neighbour_loadings = Vt.T if scale_neighbour is None else scale_neighbour[:, None] * Vt.T
        self.singular_values = singular
        self.target_loadings, self.neighbour_loadings = _orient_cross_pairs(
            target_loadings, neighbour_loadings
        )

    def _prepare_cross_gain_basis(
        self,
        W_target,
        W_neighbour,
        Pt,
        Pu,
        samples,
        weights,
        n_components,
        tol,
    ):
        """Cache expression bases shared by cross-program spatial permutations."""
        target_row_scale = _cross_row_scale(samples, weights, "target")
        neighbour_row_scale = _cross_row_scale(samples, weights, "neighbour")
        rank_target = max(self.expression_rank, int(n_components) + 2)
        rank_neighbour = max(self.expression_rank, int(n_components) + 2)
        Ut, st, Vtt = projected_svd(
            W_target,
            Pt,
            rank_target,
            row_scale=target_row_scale,
            dtype=self.dtype,
            tol=tol,
        )
        Uu, su, Vtu = projected_svd(
            W_neighbour,
            Pu,
            rank_neighbour,
            row_scale=neighbour_row_scale,
            dtype=self.dtype,
            tol=tol,
        )
        denom_target = np.sqrt(np.square(st) + self.gain_ridge)
        denom_neighbour = np.sqrt(np.square(su) + self.gain_ridge)
        cache = {
            "Ut": Ut,
            "st": st,
            "Vtt": Vtt,
            "Uu": Uu,
            "su": su,
            "Vtu": Vtu,
            "dt": np.divide(st, denom_target, out=np.zeros_like(st), where=denom_target > 0),
            "du": np.divide(su, denom_neighbour, out=np.zeros_like(su), where=denom_neighbour > 0),
            "denom_target": denom_target,
            "denom_neighbour": denom_neighbour,
        }
        self._cross_gain_basis_ = cache
        return cache

    @staticmethod
    def _cross_gain_solve(cache, kernels, contrast, n_components):
        K_contrast = sp.block_diag(
            [float(coefficient) * K for K, coefficient in zip(kernels, contrast)],
            format="csr",
        )
        H = (
            cache["dt"][:, None]
            * (cache["Ut"].T @ (K_contrast @ cache["Uu"]))
            * cache["du"][None, :]
        )
        left, singular, right_t = svd(H, full_matrices=False)
        k = min(int(n_components), len(singular))
        return left[:, :k], singular[:k], right_t[:k]

    def _fit_gain(
        self,
        W_target,
        W_neighbour,
        Pt,
        Pu,
        samples,
        weights,
        contrast,
        n_components,
        tol,
    ):
        cache = self._prepare_cross_gain_basis(
            W_target,
            W_neighbour,
            Pt,
            Pu,
            samples,
            weights,
            n_components,
            tol,
        )
        left, singular, right_t = self._cross_gain_solve(
            cache,
            [sample.K for sample in samples],
            contrast,
            n_components,
        )
        target_loadings = cache["Vtt"].T @ (
            (1.0 / np.maximum(cache["denom_target"], 1e-12))[:, None] * left
        )
        neighbour_loadings = cache["Vtu"].T @ (
            (1.0 / np.maximum(cache["denom_neighbour"], 1e-12))[:, None] * right_t.T
        )
        self.singular_values = singular
        self.target_loadings, self.neighbour_loadings = _orient_cross_pairs(
            target_loadings, neighbour_loadings
        )
        self.target_expression_singular_values_ = cache["st"]
        self.neighbour_expression_singular_values_ = cache["su"]

    def _sample_mode_covariance(self):
        values = []
        for sample, target_modes, neighbour_modes in zip(
            self.samples, self.target_modes, self.neighbour_modes
        ):
            spatial_neighbour = sample.K @ neighbour_modes
            values.append(np.sum(target_modes * spatial_neighbour, axis=0))
        return np.vstack(values)

    def truncate_modes(self, n_modes):
        n_modes = int(n_modes)
        if n_modes < 0 or n_modes > len(self.singular_values):
            raise ValueError("n_modes is outside the fitted mode range")
        self.singular_values = self.singular_values[:n_modes]
        self.target_loadings = self.target_loadings[:, :n_modes]
        self.neighbour_loadings = self.neighbour_loadings[:, :n_modes]
        if isinstance(self.target_modes, list):
            self.target_modes = [mode[:, :n_modes] for mode in self.target_modes]
            self.neighbour_modes = [mode[:, :n_modes] for mode in self.neighbour_modes]
        else:
            self.target_modes = self.target_modes[:, :n_modes]
            self.neighbour_modes = self.neighbour_modes[:, :n_modes]
        for name in (
            "sample_mode_covariance_",
            "sample_mode_statistic_",
        ):
            if hasattr(self, name):
                setattr(self, name, getattr(self, name)[:, :n_modes])
        for name in (
            "aggregate_mode_statistic_",
            "contrast_mode_statistic_",
            "group_a_mode_statistic_",
            "group_b_mode_statistic_",
            "permutation_pvalues_",
        ):
            if hasattr(self, name):
                setattr(self, name, getattr(self, name)[:n_modes])
        self.selected_modes_ = n_modes
        return self

    def permutation_test(self, n_permutations=100, seed=0, tol=0):
        """Random-label max-singular-value test for non-differential cross programs."""
        n_permutations = int(n_permutations)
        if n_permutations <= 0:
            self.permutation_pvalues_ = np.full(len(self.singular_values), np.nan)
            self.permutation_max_singular_values_ = np.empty(0)
            return self.permutation_pvalues_
        rng = np.random.default_rng(seed)
        k = len(self.singular_values)
        null = np.empty(n_permutations, dtype=float)
        for b in range(n_permutations):
            samples = _permuted_cross_samples(self.samples, rng)
            if self.objective == "gain":
                _, singular, _ = self._cross_gain_solve(
                    self._cross_gain_basis_,
                    [sample.K for sample in samples],
                    np.ones(len(samples), dtype=float),
                    k,
                )
                null[b] = float(singular[0])
                continue
            tmp = _CrossSpatialProgramBase(
                self.radius,
                objective=self.objective,
                sample_weighting=self.sample_weighting,
                expression_rank=self.expression_rank,
                gain_ridge=self.gain_ridge,
                graph_normalisation=self.graph_normalisation,
                geometry_normalisation=self.geometry_normalisation,
                distance_bins=self.distance_bins,
                covariates_tol=self.covariates_tol,
                block_size=self.block_size,
                dtype=self.dtype,
                verbose=False,
            )
            tmp._fit_prepared(samples, k, tol=tol)
            null[b] = float(tmp.singular_values[0])
        selection = select_modes(self.singular_values, null, alpha=0.05)
        self.permutation_max_singular_values_ = null
        self.permutation_pvalues_ = selection.pvalues
        return self.permutation_pvalues_

    def select_modes(self, alpha=0.05):
        if not hasattr(self, "permutation_max_singular_values_"):
            raise RuntimeError("Run permutation_test before select_modes")
        selection = select_modes(
            self.singular_values, self.permutation_max_singular_values_, alpha=alpha
        )
        self.mode_selection_ = selection
        return selection


class SharedCrossSpatialProgramModel(_CrossSpatialProgramBase):
    """Paired target-neighbour programs shared across biological samples.

    Each sample contributes its bipartite cross-covariance independently. Pair
    geometry can be standardised before expression enters the model. With the
    default ``geometry_normalisation='distance'``, all samples use one common
    target-neighbour distance profile and total pair mass
    ``sqrt(n_target_s * n_neighbour_s)``. The default ``sample_weighting='n_spots'``
    is its reciprocal, so every biological sample contributes unit total pair mass.
    """

    def __init__(self, *args, sample_weighting="n_spots", **kwargs):
        super().__init__(*args, sample_weighting=sample_weighting, **kwargs)

    def fit(
        self,
        W,
        coords,
        n_components,
        target_masks,
        neighbour_masks,
        covariates=None,
        tol=0,
    ):
        with timed("Preparing cross samples", self.verbose):
            samples = prepare_cross_samples(
                W,
                coords,
                self.radius,
                target_masks,
                neighbour_masks,
                covariates=covariates,
                dtype=self.dtype,
                graph_normalisation=self.graph_normalisation,
                geometry_normalisation=self.geometry_normalisation,
                distance_bins=self.distance_bins,
            )
        with timed("Solving shared cross-program objective", self.verbose):
            return self._fit_prepared(samples, n_components, tol=tol)


class DifferentialCrossSpatialProgramModel(_CrossSpatialProgramBase):
    """Target-neighbour programs whose cross-covariance differs between sample groups.

    The contrast is a difference of group means over geometry-normalised biological
    sample statistics. Singular values are non-negative; the paired target/neighbour
    loading orientation is chosen so the fitted group-A minus group-B contrast is
    positive along each returned pair. Group-specific mode statistics are retained
    explicitly for interpretation.
    """

    def __init__(
        self,
        radius,
        positive_samples,
        negative_samples,
        *args,
        sample_weighting="n_spots",
        **kwargs,
    ):
        super().__init__(radius, *args, sample_weighting=sample_weighting, **kwargs)
        self.positive_samples = list(positive_samples)
        self.negative_samples = list(negative_samples)

    def fit(
        self,
        W,
        coords,
        n_components,
        target_masks,
        neighbour_masks,
        covariates=None,
        tol=0,
    ):
        with timed("Preparing differential cross samples", self.verbose):
            samples = prepare_cross_samples(
                W,
                coords,
                self.radius,
                target_masks,
                neighbour_masks,
                covariates=covariates,
                dtype=self.dtype,
                graph_normalisation=self.graph_normalisation,
                geometry_normalisation=self.geometry_normalisation,
                distance_bins=self.distance_bins,
            )
        contrast = _group_contrast_coefficients(
            len(samples), self.positive_samples, self.negative_samples
        )
        with timed("Solving differential cross-program objective", self.verbose):
            self._fit_prepared(samples, n_components, tol=tol, contrast=contrast)
        self.contrast_mode_statistic_ = self.aggregate_mode_statistic_.copy()
        self.group_a_mode_statistic_ = np.mean(
            self.sample_mode_statistic_[self.positive_samples], axis=0
        )
        self.group_b_mode_statistic_ = np.mean(
            self.sample_mode_statistic_[self.negative_samples], axis=0
        )
        return self

    def permutation_test(self, n_permutations=100, seed=0, tol=0):
        """Sample-label permutation test with max-singular-value FWER control."""
        n_permutations = int(n_permutations)
        if n_permutations <= 0:
            self.permutation_pvalues_ = np.full(len(self.singular_values), np.nan)
            self.permutation_max_singular_values_ = np.empty(0)
            return self.permutation_pvalues_
        rng = np.random.default_rng(seed)
        n = len(self.samples)
        n_positive = len(self.positive_samples)
        k = len(self.singular_values)
        null = np.empty(n_permutations, dtype=float)
        for b in range(n_permutations):
            contrast = _permuted_group_contrast(rng, n, n_positive)
            if self.objective == "gain":
                _, singular, _ = self._cross_gain_solve(
                    self._cross_gain_basis_,
                    [sample.K for sample in self.samples],
                    contrast,
                    k,
                )
                null[b] = float(singular[0])
                continue
            tmp = _CrossSpatialProgramBase(
                self.radius,
                objective=self.objective,
                sample_weighting=self.sample_weighting,
                expression_rank=self.expression_rank,
                gain_ridge=self.gain_ridge,
                graph_normalisation=self.graph_normalisation,
                geometry_normalisation=self.geometry_normalisation,
                distance_bins=self.distance_bins,
                covariates_tol=self.covariates_tol,
                block_size=self.block_size,
                dtype=self.dtype,
                verbose=False,
            )
            tmp._fit_prepared(self.samples, k, tol=tol, contrast=contrast)
            null[b] = np.max(tmp.singular_values)
        observed = np.asarray(self.singular_values)
        pvalues = (1.0 + np.sum(null[:, None] >= observed[None, :], axis=0)) / (
            n_permutations + 1.0
        )
        self.permutation_max_singular_values_ = null
        self.permutation_pvalues_ = pvalues
        return pvalues


class CrossSpatialProgramModel(_CrossSpatialProgramBase):
    """Paired target-neighbour gene programs from one biological sample."""

    def __init__(self, *args, **kwargs):
        kwargs.pop("sample_weighting", None)
        super().__init__(*args, sample_weighting="none", **kwargs)

    def fit(
        self,
        W,
        coords,
        n_components,
        target_mask,
        neighbour_mask,
        covariates=None,
        tol=0,
    ):
        with timed("Preparing cross sample", self.verbose):
            samples = prepare_cross_samples(
                [W],
                [coords],
                self.radius,
                [target_mask],
                [neighbour_mask],
                covariates=None if covariates is None else [covariates],
                dtype=self.dtype,
                graph_normalisation=self.graph_normalisation,
                geometry_normalisation=self.geometry_normalisation,
                distance_bins=self.distance_bins,
            )
        with timed("Solving cross-program objective", self.verbose):
            self._fit_prepared(samples, n_components, tol=tol)
        sample = self.samples[0]
        self.K_cross_ = sample.K
        self.target_idx = sample.target_idx
        self.neighbour_idx = sample.neighbour_idx
        self.target_modes = self.target_modes[0]
        self.neighbour_modes = self.neighbour_modes[0]
        return self


class LabelDecompositionModel:
    """Exact decomposition of sample-centred spatial covariance by label structure.

    Let Y be the commonly sample-centered/covariate-residualized expression,
    W the within-label residual, and B = Y - W the between-label mean component.
    Then

        Y^T A Y = B^T A B + W^T A W + B^T A W + W^T A B.

    The four returned operators are therefore exactly additive in the common
    projected expression space.
    """

    components = ("total", "between", "within", "coupling")

    def __init__(
        self,
        radius,
        graph_normalisation="symmetric",
        geometry_normalisation="distance",
        distance_bins=10,
        covariates_tol=1e-10,
        dtype=np.float64,
        verbose=False,
    ):
        if geometry_normalisation not in VALID_GEOMETRY_NORMALISATIONS:
            raise ValueError(
                "geometry_normalisation must be one of " f"{sorted(VALID_GEOMETRY_NORMALISATIONS)}"
            )
        if int(distance_bins) < 1:
            raise ValueError("distance_bins must be positive")
        self.radius = float(radius)
        self.graph_normalisation = graph_normalisation
        self.geometry_normalisation = geometry_normalisation
        self.distance_bins = int(distance_bins)
        self.covariates_tol = covariates_tol
        self.dtype = np.dtype(dtype)
        self.verbose = verbose

    def fit(self, W, coords, labels, n_components, covariates=None, tol=0, maxiter=None):
        samples = prepare_samples(
            W,
            coords,
            self.radius,
            labels=labels,
            covariates=covariates,
            dtype=self.dtype,
            graph_normalisation=self.graph_normalisation,
            geometry_normalisation=self.geometry_normalisation,
            distance_bins=self.distance_bins,
        )
        if len(samples) != 1:
            raise ValueError("LabelDecompositionModel currently operates on one sample")
        self.samples = samples
        sample = samples[0]
        self.geometry_diagnostics_ = [sample.geometry_diagnostics]
        K = sample.K
        base_projector = SpotProjector(
            sample.n_spots,
            groups=Groups.single(sample.n_spots),
            covariates=sample.covariates,
            tol=self.covariates_tol,
            dtype=self.dtype,
        )
        total_t = ProjectorTransform(base_projector)
        within_t = WithinLabelTransform(base_projector, sample.labels_groups, self.dtype)
        between_t = BetweenLabelTransform(base_projector, within_t)

        ops = {
            "total": BilinearGeneOperator(sample.W, K, total_t, dtype=self.dtype),
            "between": BilinearGeneOperator(sample.W, K, between_t, dtype=self.dtype),
            "within": BilinearGeneOperator(sample.W, K, within_t, dtype=self.dtype),
            "coupling": BilinearGeneOperator(
                sample.W,
                K,
                between_t,
                right_transform=within_t,
                symmetric_pair=True,
                dtype=self.dtype,
            ),
        }
        self.results_ = {}
        for name, op in ops.items():
            signed = name == "coupling"
            vals, vecs = _solve_gene_operator(
                op, n_components, signed=signed, tol=tol, maxiter=maxiter
            )
            loadings = column_normalize(vecs)
            transform = {
                "total": total_t,
                "between": between_t,
                "within": within_t,
                "coupling": total_t,
            }[name]
            scores = transform.apply(sample.W @ loadings)
            self.results_[name] = {
                "eigenvalues": vals,
                "gene_loadings": loadings,
                "spot_modes": np.asarray(scores)[sample.inv_order],
            }
        return self

    def operator_additivity_error(self, vectors=None):
        """Numerically verify total = between + within + coupling."""
        sample = self.samples[0]
        K = sample.K
        base = SpotProjector(
            sample.n_spots,
            groups=Groups.single(sample.n_spots),
            covariates=sample.covariates,
            tol=self.covariates_tol,
            dtype=self.dtype,
        )
        within = WithinLabelTransform(base, sample.labels_groups, self.dtype)
        between = BetweenLabelTransform(base, within)
        total_op = BilinearGeneOperator(sample.W, K, ProjectorTransform(base), dtype=self.dtype)
        between_op = BilinearGeneOperator(sample.W, K, between, dtype=self.dtype)
        within_op = BilinearGeneOperator(sample.W, K, within, dtype=self.dtype)
        coupling_op = BilinearGeneOperator(
            sample.W, K, between, within, symmetric_pair=True, dtype=self.dtype
        )
        if vectors is None:
            rng = np.random.default_rng(0)
            vectors = rng.normal(size=(sample.n_genes, min(3, sample.n_genes)))
        residual = total_op.apply(vectors) - (
            between_op.apply(vectors) + within_op.apply(vectors) + coupling_op.apply(vectors)
        )
        denom = max(np.linalg.norm(total_op.apply(vectors)), 1e-12)
        return float(np.linalg.norm(residual) / denom)
