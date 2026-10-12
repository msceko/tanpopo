from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModeSelectionResult:
    """Permutation-based spatial mode selection summary."""

    observed: np.ndarray
    null_maxima: np.ndarray
    pvalues: np.ndarray
    alpha: float
    selected_modes: int

    @property
    def threshold(self):
        if self.null_maxima.size == 0:
            return float("nan")
        return float(np.quantile(self.null_maxima, 1.0 - self.alpha, method="higher"))

    @property
    def effective_dimension(self):
        return effective_dimension(np.maximum(self.observed - self.threshold, 0.0))

    def summary(self):
        return {
            "alpha": float(self.alpha),
            "selected_modes": int(self.selected_modes),
            "threshold": self.threshold,
            "effective_dimension": self.effective_dimension,
            "n_permutations": int(self.null_maxima.size),
        }


def max_statistic_pvalues(observed, null_maxima):
    """Family-wise p-values from a maximum-statistic permutation null."""
    observed = np.asarray(observed, dtype=float)
    null_maxima = np.asarray(null_maxima, dtype=float)
    if null_maxima.ndim != 1:
        raise ValueError("null_maxima must be one-dimensional")
    if null_maxima.size == 0:
        return np.full(observed.shape, np.nan, dtype=float)
    return (1.0 + np.sum(null_maxima[:, None] >= observed[None, :], axis=0)) / (
        null_maxima.size + 1.0
    )


def select_modes(observed, null_maxima, alpha=0.05):
    """Select all modes exceeding a max-statistic FWER threshold."""
    observed = np.asarray(observed, dtype=float)
    null_maxima = np.asarray(null_maxima, dtype=float)
    alpha = float(alpha)
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    pvalues = max_statistic_pvalues(observed, null_maxima)
    selected = int(np.sum(np.isfinite(pvalues) & (pvalues <= alpha)))
    return ModeSelectionResult(observed, null_maxima, pvalues, alpha, selected)


def effective_dimension(values):
    """Participation-ratio effective dimension of non-negative spatial signal."""
    values = np.asarray(values, dtype=float)
    values = np.maximum(values[np.isfinite(values)], 0.0)
    total = float(values.sum())
    squared = float(np.square(values).sum())
    if total <= 0 or squared <= 0:
        return 0.0
    return total * total / squared
