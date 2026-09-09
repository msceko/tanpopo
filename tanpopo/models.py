from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh, svd
from scipy.sparse.linalg import eigsh, svds

from tanpopo.data import Groups, concatenate_samples, prepare_samples
from tanpopo.kernel import kernel_matrix_sparse
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


def _sample_weights(samples, mode):
    if mode == "none":
        return np.ones(len(samples), dtype=float)
    if mode == "n_spots":
        return 1.0 / np.asarray([s.n_spots for s in samples], dtype=float)
    raise ValueError("sample_weighting must be 'none' or 'n_spots'")


def _weighted_block_kernel(samples, weights, signs=None):
    signs = np.ones(len(samples)) if signs is None else np.asarray(signs, dtype=float)
    return sp.block_diag(
        [float(w * sign) * s.K for s, w, sign in zip(samples, weights, signs)],
        format="csr",
    )


def _row_scale(samples, weights):
    return np.concatenate(
        [np.full(s.n_spots, np.sqrt(float(w)), dtype=float) for s, w in zip(samples, weights)]
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
    """Conditional spatial gene programs.

    Objectives
    ----------
    covariance
        Maximise cross-cell spatial covariance, v^T Y^T A Y v.
    gene_standardized
        The same numerator after scaling each gene by ordinary residual expression
        variance. Unlike the historical alpha=1 path, the denominator is positive
        expression variance rather than spatial autocovariance.
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
        covariates_tol=1e-10,
        block_size=256,
        dtype=np.float64,
        verbose=False,
    ):
        if objective not in VALID_OBJECTIVES:
            raise ValueError(f"objective must be one of {sorted(VALID_OBJECTIVES)}")
        self.radius = float(radius)
        self.objective = objective
        self.spot_operator = spot_operator
        self.sample_weighting = sample_weighting
        self.expression_rank = int(expression_rank)
        self.gain_ridge = float(gain_ridge)
        self.graph_normalisation = graph_normalisation
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
            )
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
        self.gene_spatial_scores_ = gene_spatial_covariance_diag(
            Wc, S, block_size=self.block_size
        )
        self.sample_coefficients_ = weights * (
            np.ones(len(weights)) if signs is None else np.asarray(signs, dtype=float)
        )

    def _fit_gain(self, n_components, signs, signed, tol):
        Wc, _, sample_groups, label_groups, covc = concatenate_samples(self.samples)
        weights = _sample_weights(self.samples, self.sample_weighting)
        base_K = sp.block_diag([s.K for s in self.samples], format="csr")
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
        ridge = self.gain_ridge
        denom = np.sqrt(np.square(singular) + ridge)
        gain_scale = np.divide(singular, denom, out=np.zeros_like(singular), where=denom > 0)

        signs_array = np.ones(len(self.samples)) if signs is None else np.asarray(signs, dtype=float)
        # In weighted expression coordinates the base sample weights cancel, leaving
        # only the sign/contrast coefficient in the compressed spatial numerator.
        K_signed = sp.block_diag(
            [float(sign) * s.K for s, sign in zip(self.samples, signs_array)], format="csr"
        )
        S_signed = spec.build(
            K_signed,
            sample_groups=sample_groups,
            label_groups=label_groups,
            covariates=covc,
            tol=self.covariates_tol,
            dtype=self.dtype,
        )
        H0 = U.T @ S_signed.apply(U)
        H = gain_scale[:, None] * H0 * gain_scale[None, :]
        vals, a = _solve_dense(H, n_components, signed=signed)

        loadings = Vt.T @ ((1.0 / np.maximum(denom, 1e-12))[:, None] * a)
        loadings = column_normalize(loadings)
        phi = S_base.projector.apply(Wc @ loadings)

        self.eigenvalues = vals
        self.eigenvectors = a
        self.gene_loadings = loadings
        self.spot_modes = _split_spot_modes(phi, self.samples)
        self.gene_scores = loadings * np.sqrt(np.maximum(np.abs(vals), 1e-12))[None, :]
        # Report unweighted raw spatial autocovariance per gene for diagnostics.
        self.gene_spatial_scores_ = gene_spatial_covariance_diag(
            Wc, S_base, block_size=self.block_size
        )
        self.expression_singular_values_ = singular
        self.sample_coefficients_ = weights * signs_array

    def gene_spatial_scores(self):
        return self.gene_spatial_scores_.copy()


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
        signs = np.zeros(n, dtype=float)
        signs[self.positive_samples] = 1.0
        signs[self.negative_samples] = -1.0
        if np.any(signs == 0):
            raise ValueError("Every sample must be assigned to positive or negative group")
        result = super().fit(
            W,
            coords,
            n_components,
            labels=labels,
            covariates=covariates,
            masks=masks,
            signs=signs,
            **kwargs,
        )
        self._observed_signs = signs
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
            positive = rng.choice(n, size=n_positive, replace=False)
            signs = -np.ones(n, dtype=float)
            signs[positive] = 1.0
            tmp = SpatialProgramModel(
                self.radius,
                objective=self.objective,
                spot_operator=self.spot_operator,
                sample_weighting=self.sample_weighting,
                expression_rank=self.expression_rank,
                gain_ridge=self.gain_ridge,
                graph_normalisation=self.graph_normalisation,
                covariates_tol=self.covariates_tol,
                block_size=self.block_size,
                dtype=self.dtype,
                verbose=False,
            )
            tmp.samples = self.samples
            if self.objective == "gain":
                tmp._fit_gain(k, signs=signs, signed=True, tol=tol)
            else:
                tmp._fit_covariance(
                    k, signs=signs, signed=True, tol=tol, maxiter=maxiter
                )
            null[b] = np.max(np.abs(tmp.eigenvalues))
        observed = np.abs(self.eigenvalues)
        pvalues = (1.0 + np.sum(null[:, None] >= observed[None, :], axis=0)) / (
            n_permutations + 1.0
        )
        self.permutation_max_abs_eigenvalues_ = null
        self.permutation_pvalues_ = pvalues
        return pvalues


class CrossSpatialProgramModel:
    """Paired target-neighbour gene programs from a bipartite spatial graph."""

    def __init__(
        self,
        radius,
        objective="covariance",
        expression_rank=50,
        gain_ridge=1e-8,
        graph_normalisation="symmetric",
        covariates_tol=1e-10,
        block_size=256,
        dtype=np.float64,
        verbose=False,
    ):
        if objective not in VALID_OBJECTIVES:
            raise ValueError(f"objective must be one of {sorted(VALID_OBJECTIVES)}")
        self.radius = float(radius)
        self.objective = objective
        self.expression_rank = int(expression_rank)
        self.gain_ridge = float(gain_ridge)
        self.graph_normalisation = graph_normalisation
        self.covariates_tol = covariates_tol
        self.block_size = block_size
        self.dtype = np.dtype(dtype)
        self.verbose = verbose

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
        W = sp.csr_matrix(W, dtype=self.dtype)
        coords = np.asarray(coords, dtype=self.dtype)
        target_mask = np.asarray(target_mask, dtype=bool)
        neighbour_mask = np.asarray(neighbour_mask, dtype=bool)
        if not np.any(target_mask) or not np.any(neighbour_mask):
            raise ValueError("target_mask and neighbour_mask must each select cells")
        self.target_idx = np.flatnonzero(target_mask)
        self.neighbour_idx = np.flatnonzero(neighbour_mask)
        Wt, Wu = W[target_mask], W[neighbour_mask]
        Ct, Cu = coords[target_mask], coords[neighbour_mask]
        cov_t = None if covariates is None else np.asarray(covariates)[target_mask]
        cov_u = None if covariates is None else np.asarray(covariates)[neighbour_mask]
        Pt = SpotProjector(Wt.shape[0], Groups.single(Wt.shape[0]), cov_t, self.covariates_tol)
        Pu = SpotProjector(Wu.shape[0], Groups.single(Wu.shape[0]), cov_u, self.covariates_tol)

        if np.array_equal(self.target_idx, self.neighbour_idx):
            K = kernel_matrix_sparse(
                Ct,
                self.radius,
                dtype=self.dtype,
                normalisation=self.graph_normalisation,
                zero_diagonal=True,
            )
        else:
            K = kernel_matrix_sparse(
                Ct,
                self.radius,
                coords_query=Cu,
                dtype=self.dtype,
                normalisation=self.graph_normalisation,
                zero_diagonal=False,
            )
        self.K_cross_ = K

        if self.objective == "gain":
            self._fit_gain(Wt, Wu, Pt, Pu, K, n_components, tol)
        else:
            self._fit_covariance(Wt, Wu, Pt, Pu, K, n_components, tol)
        return self

    def _fit_covariance(self, Wt, Wu, Pt, Pu, K, n_components, tol):
        scale_t = scale_u = None
        if self.objective == "gene_standardized":
            var_t = gene_expression_variance(Wt, Pt, self.block_size, eps=1e-12)
            var_u = gene_expression_variance(Wu, Pu, self.block_size, eps=1e-12)
            scale_t = expression_scale_from_variance(var_t)
            scale_u = expression_scale_from_variance(var_u)
        C = CrossGeneOperator(Wt, Wu, K, Pt, Pu, scale_t, scale_u, dtype=self.dtype)
        k = min(int(n_components), min(C.shape) - 1)
        U, singular, Vt = svds(C, k=k, which="LM", tol=tol)
        order = np.argsort(singular)[::-1]
        singular, U, Vt = singular[order], U[:, order], Vt[order]
        target_loadings = U if scale_t is None else scale_t[:, None] * U
        neighbour_loadings = Vt.T if scale_u is None else scale_u[:, None] * Vt.T
        self.singular_values = singular
        target_loadings = column_normalize(target_loadings)
        neighbour_loadings = column_normalize(neighbour_loadings)
        orient_idx = np.argmax(np.abs(target_loadings), axis=0)
        pair_sign = np.sign(target_loadings[orient_idx, np.arange(target_loadings.shape[1])])
        pair_sign[pair_sign == 0] = 1
        self.target_loadings = target_loadings * pair_sign[None, :]
        self.neighbour_loadings = neighbour_loadings * pair_sign[None, :]
        self.target_modes = Pt.apply(Wt @ self.target_loadings)
        self.neighbour_modes = Pu.apply(Wu @ self.neighbour_loadings)

    def _fit_gain(self, Wt, Wu, Pt, Pu, K, n_components, tol):
        rank_t = max(self.expression_rank, int(n_components) + 2)
        rank_u = max(self.expression_rank, int(n_components) + 2)
        Ut, st, Vtt = projected_svd(Wt, Pt, rank_t, dtype=self.dtype, tol=tol)
        Uu, su, Vtu = projected_svd(Wu, Pu, rank_u, dtype=self.dtype, tol=tol)
        dt = st / np.sqrt(np.square(st) + self.gain_ridge)
        du = su / np.sqrt(np.square(su) + self.gain_ridge)
        H = dt[:, None] * (Ut.T @ (K @ Uu)) * du[None, :]
        left, singular, right_t = svd(H, full_matrices=False)
        k = min(int(n_components), len(singular))
        left, singular, right_t = left[:, :k], singular[:k], right_t[:k]
        denom_t = np.sqrt(np.square(st) + self.gain_ridge)
        denom_u = np.sqrt(np.square(su) + self.gain_ridge)
        target_loadings = Vtt.T @ ((1.0 / np.maximum(denom_t, 1e-12))[:, None] * left)
        neighbour_loadings = Vtu.T @ (
            (1.0 / np.maximum(denom_u, 1e-12))[:, None] * right_t.T
        )
        self.singular_values = singular
        target_loadings = column_normalize(target_loadings)
        neighbour_loadings = column_normalize(neighbour_loadings)
        orient_idx = np.argmax(np.abs(target_loadings), axis=0)
        pair_sign = np.sign(target_loadings[orient_idx, np.arange(target_loadings.shape[1])])
        pair_sign[pair_sign == 0] = 1
        self.target_loadings = target_loadings * pair_sign[None, :]
        self.neighbour_loadings = neighbour_loadings * pair_sign[None, :]
        self.target_modes = Pt.apply(Wt @ self.target_loadings)
        self.neighbour_modes = Pu.apply(Wu @ self.neighbour_loadings)
        self.target_expression_singular_values_ = st
        self.neighbour_expression_singular_values_ = su


class LabelDecompositionModel:
    """Exact decomposition of sample-centered spatial covariance by label structure.

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
        covariates_tol=1e-10,
        dtype=np.float64,
        verbose=False,
    ):
        self.radius = float(radius)
        self.graph_normalisation = graph_normalisation
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
        )
        if len(samples) != 1:
            raise ValueError("LabelDecompositionModel currently operates on one sample")
        self.samples = samples
        sample = samples[0]
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
            "total": BilinearGeneOperator(sample.W, sample.K, total_t, dtype=self.dtype),
            "between": BilinearGeneOperator(sample.W, sample.K, between_t, dtype=self.dtype),
            "within": BilinearGeneOperator(sample.W, sample.K, within_t, dtype=self.dtype),
            "coupling": BilinearGeneOperator(
                sample.W,
                sample.K,
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
        base = SpotProjector(
            sample.n_spots,
            groups=Groups.single(sample.n_spots),
            covariates=sample.covariates,
            tol=self.covariates_tol,
            dtype=self.dtype,
        )
        within = WithinLabelTransform(base, sample.labels_groups, self.dtype)
        between = BetweenLabelTransform(base, within)
        total_op = BilinearGeneOperator(sample.W, sample.K, ProjectorTransform(base), dtype=self.dtype)
        between_op = BilinearGeneOperator(sample.W, sample.K, between, dtype=self.dtype)
        within_op = BilinearGeneOperator(sample.W, sample.K, within, dtype=self.dtype)
        coupling_op = BilinearGeneOperator(
            sample.W, sample.K, between, within, symmetric_pair=True, dtype=self.dtype
        )
        if vectors is None:
            rng = np.random.default_rng(0)
            vectors = rng.normal(size=(sample.n_genes, min(3, sample.n_genes)))
        residual = total_op.apply(vectors) - (
            between_op.apply(vectors) + within_op.apply(vectors) + coupling_op.apply(vectors)
        )
        denom = max(np.linalg.norm(total_op.apply(vectors)), 1e-12)
        return float(np.linalg.norm(residual) / denom)


# Backwards-compatible class names for downstream imports. Their semantics now use
# zero-diagonal spatial covariance and explicit objectives rather than alpha scaling.
SpatialGeneKPCA = SpatialProgramModel
SpatialGeneSampleCombinedKPCA = SharedSpatialProgramModel
SpatialGeneSampleContrastKPCA = DifferentialSpatialProgramModel
