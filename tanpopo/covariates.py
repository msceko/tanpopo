import numpy as np

from tanpopo.utils import get_counts_matrix


def compute_log_total_counts(X_counts):
    """
    Spot-level log1p total counts.
    """
    total_counts = np.asarray(X_counts.sum(axis=1)).ravel().astype(np.float64)
    return np.log1p(total_counts)


def compute_log_detected_genes(X_counts):
    """
    Spot-level log1p number of detected genes.
    """
    detected_genes = np.asarray((X_counts > 0).sum(axis=1)).ravel().astype(np.float64)
    return np.log1p(detected_genes)


def compute_mito_fraction(X_counts, gene_names, eps=1e-12):
    """
    Spot-level mitochondrial fraction using genes starting with 'MT-'.
    """
    gene_names_upper = np.char.upper(np.asarray(gene_names).astype(str))
    mito_mask = np.char.startswith(gene_names_upper, "MT-")
    if mito_mask.sum() == 0:
        raise ValueError("No mitochondrial genes found for 'mito_fraction'.")

    total_counts = np.asarray(X_counts.sum(axis=1)).ravel().astype(np.float64)
    mito_counts = np.asarray(X_counts[:, mito_mask].sum(axis=1)).ravel().astype(np.float64)
    return mito_counts / (total_counts + eps)


def compute_ribo_fraction(X_counts, gene_names, eps=1e-12):
    """
    Spot-level ribosomal fraction using genes starting with 'RPS' or 'RPL'.
    """
    gene_names_upper = np.char.upper(np.asarray(gene_names).astype(str))
    ribo_mask = np.char.startswith(gene_names_upper, "RPS") | np.char.startswith(
        gene_names_upper, "RPL"
    )
    if ribo_mask.sum() == 0:
        raise ValueError("No ribosomal genes found for 'ribo_fraction'.")

    total_counts = np.asarray(X_counts.sum(axis=1)).ravel().astype(np.float64)
    ribo_counts = np.asarray(X_counts[:, ribo_mask].sum(axis=1)).ravel().astype(np.float64)
    return ribo_counts / (total_counts + eps)


def compute_covariates(adata, covariates, layer=None, key="tanpopo_covariates"):
    """
    Extract a spot-by-covariate matrix from pre-normalized counts in adata.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object. Should already have any desired gene filtering
        applied so the returned covariates align with the returned expression.
    covariates : list[str]
        Predefined covariates to compute. Supported values:
            - "log_total_counts"
            - "log_detected_genes"
            - "mito_fraction"
            - "ribo_fraction"
    layer : str or None
        Layer to use as the counts source. If None, use adata.X.

    Returns
    -------
    covariate_matrix : (n_spots, n_covariates) ndarray
    """
    covariates = list(covariates)
    X_counts = get_counts_matrix(adata, layer=layer)
    gene_names = np.array(adata.var_names)

    covariate_cols = []
    for cov in covariates:
        if cov == "log_total_counts":
            covariate_cols.append(compute_log_total_counts(X_counts))
        elif cov == "log_detected_genes":
            covariate_cols.append(compute_log_detected_genes(X_counts))
        elif cov == "mito_fraction":
            covariate_cols.append(compute_mito_fraction(X_counts, gene_names))
        elif cov == "ribo_fraction":
            covariate_cols.append(compute_ribo_fraction(X_counts, gene_names))
        else:
            raise ValueError(
                f"Unknown covariate '{cov}'. Supported values are: "
                "'log_total_counts', 'log_detected_genes', "
                "'mito_fraction', 'ribo_fraction'."
            )

    adata.obsm[key] = np.column_stack(covariate_cols).astype(np.float64, copy=False)
