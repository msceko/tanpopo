import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

from tanpopo.analysis import cumulative_contribution
from tanpopo.utils import make_iterable


def _match_labels_to_reference(adata, key, reference_key):
    """Match each label to the reference group with greatest overlap."""
    overlap = pd.crosstab(adata.obs[key], adata.obs[reference_key])
    return overlap.idxmax(axis=1)


def _make_matched_labels(labels, matched_reference):
    """Rename labels according to their matched reference group."""
    matched_names = {}

    for reference_label, group in matched_reference.groupby(matched_reference, sort=False):
        group_labels = list(group.index)

        for number, label in enumerate(group_labels, start=1):
            matched_names[label] = (
                str(reference_label) if len(group_labels) == 1 else f"{reference_label} {number}"
            )

    categories = [matched_names[label] for label in labels.cat.categories]

    matched_labels = pd.Categorical(labels.map(matched_names), categories=categories)

    return matched_labels, matched_names


def _make_matched_palette(adata, reference_key, matched_names, matched_reference):
    """Give each subtype the colour of its matched reference group."""
    reference_categories = list(adata.obs[reference_key].cat.categories)

    cmap = plt.get_cmap("tab20")
    reference_colours = {
        reference: to_hex(cmap(i % cmap.N)) for i, reference in enumerate(reference_categories)
    }

    palette = dict(reference_colours)

    for label, matched_name in matched_names.items():
        reference = matched_reference[label]
        palette[matched_name] = reference_colours[reference]

    return palette


def spatial_scatter(adata, keys, **kwargs):
    n_cols = int(np.round(4 / 3 * np.sqrt(len(keys))))
    axs = sc.pl.embedding(
        adata,
        basis="spatial",
        color=keys,
        ncols=n_cols,
        wspace=0,
        hspace=0,
        colorbar_loc=None,
        title=len(keys) * [""],
        frameon=False,
        show=False,
        **kwargs,
    )
    for i, ax in enumerate(make_iterable(axs)):
        ax.invert_yaxis()
        ax.axis("equal")
        ax.text(
            0.05,
            0.95,
            f"{i}.",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
        )
        ax.axis("off")


def plot_spatial_modes(adata, phi, clip=(1, 99), cmap="coolwarm", vcenter=0, **kwargs):
    """Plot spatial gene eigenmodes"""
    prefix = "spatial_mode"
    n_modes = phi.shape[1]
    empty_adata = ad.AnnData(X=np.empty(adata.shape), obs=adata.obs.copy(), var=adata.var.copy())
    empty_adata.obsm["spatial"] = adata.obsm["spatial"]

    keys, vmin, vmax = [], [], []
    for k in range(n_modes):
        keys.append(f"{prefix}_{k}")
        empty_adata.obs[keys[k]] = phi[:, k]
        a, b = np.nanpercentile(phi[:, k], clip)
        vmin.append(min(a, -1e-12))
        vmax.append(max(b, 1e-12))

    spatial_scatter(empty_adata, keys, vmin=vmin, vmax=vmax, cmap=cmap, vcenter=vcenter, **kwargs)


def plot_gene_clusters(adata, key="leiden", **kwargs):
    """Plot spatial distributions of gene clusters"""
    keys = []
    for cluster in range(adata.var[key].max() + 1):
        keys.append(f"gene_cluster_{cluster}")
        genes = (adata.var[key] == cluster).tolist()
        adata.obs[keys[cluster]] = np.asarray(adata.X[:, genes].sum(axis=1)).ravel()

    spatial_scatter(adata, keys, **kwargs)


def plot_cumulative_contribution(eigvecs):
    """Plot cumulative distributions for each squared eigenvector"""
    plt.figure()
    plt.plot(cumulative_contribution(eigvecs))
    plt.xlabel("Number of genes")
    plt.ylabel("Cumulative contribution")
    plt.axhline(0.8, linestyle="--", alpha=0.5)
    plt.legend([f"Basis {n}" for n in range(eigvecs.shape[1])])


def plot_umap(X, obs, obs_names, n_neighbors, min_dist=0.1, spread=1.0, **kwargs):
    """Plot umap embedding with cluster colours"""
    adata = ad.AnnData(X)
    adata.obs_names = obs_names
    adata.obs["umap"] = obs

    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X")
    sc.tl.umap(adata, min_dist=min_dist, spread=spread)
    sc.pl.umap(adata, color="umap", size=20, palette="tab20", **kwargs)


def plot_labels(adata, key, **kwargs):
    """Plot spot labels"""
    embedding_args = {
        "basis": "spatial",
        "color": key,
        "palette": "tab20",
        "frameon": False,
        "show": False,
    }
    for key, value in kwargs.items():
        embedding_args[key] = value
    axs = sc.pl.embedding(adata, **embedding_args)
    axs = make_iterable(axs)
    for ax in axs:
        ax.invert_yaxis()
        ax.axis("equal")


def plot_labels_to_reference(adata, key, reference, **kwargs):
    adata.obs[reference] = adata.obs[reference].astype("category")

    matched_reference = _match_labels_to_reference(adata, key, reference)
    matched_key = f"{key}_matched"
    matched_labels, matched_names = _make_matched_labels(adata.obs[key], matched_reference)
    adata.obs[matched_key] = matched_labels
    palette = _make_matched_palette(adata, reference, matched_names, matched_reference)

    plot_labels(adata, [reference, matched_key], palette=palette)
