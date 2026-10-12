from __future__ import annotations

import math

import numpy as np


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Plotting requires the optional 'plot' dependencies") from exc
    return plt


def plot_spatial_modes(adata, modes, indices=None, clip=(1, 99), cmap="coolwarm"):
    """Plot one or more spatial program scores using only matplotlib."""
    plt = _plt()
    coords = np.asarray(adata.obsm["spatial"])
    modes = np.asarray(modes)
    if indices is None:
        indices = list(range(modes.shape[1]))
    indices = list(indices)
    n = len(indices)
    if n == 0:
        raise ValueError("No modes selected for plotting")
    ncols = min(4, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=(4 * ncols, 4 * nrows))
    for ax, mode in zip(axes.ravel(), indices):
        values = modes[:, mode]
        finite = np.isfinite(values)
        if finite.any():
            lo, hi = np.nanpercentile(values, clip)
            bound = max(abs(float(lo)), abs(float(hi)), 1e-12)
            scatter = ax.scatter(
                coords[finite, 0],
                coords[finite, 1],
                c=values[finite],
                s=4,
                cmap=cmap,
                vmin=-bound,
                vmax=bound,
                linewidths=0,
            )
            fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"Mode {mode}")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_labels(adata, key):
    """Plot categorical labels in tissue coordinates."""
    plt = _plt()
    from matplotlib.lines import Line2D

    if key not in adata.obs:
        raise KeyError(f"obs column {key!r} was not found")
    coords = np.asarray(adata.obsm["spatial"])
    labels = adata.obs[key].astype("category")
    codes = labels.cat.codes.to_numpy()
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=codes, s=4, cmap="tab20", linewidths=0)
    handles = []
    for code, category in enumerate(labels.cat.categories):
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                label=str(category),
                color=scatter.cmap(scatter.norm(code)),
            )
        )
    if handles:
        ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    fig.tight_layout()
    return fig
