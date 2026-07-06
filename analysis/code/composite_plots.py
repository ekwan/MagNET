"""Correlation-heatmap panels for the composite-model ablation SI figure (Figure S15).

Two seaborn heatmaps behind Figure S15's feature-correlation panels: a per-nucleus lower-triangle
correlation matrix among the five composite-model features (stationary shielding, PCM, Desmond, its
vibrational analogue, QCD), and a PCM-vs-Desmond Pearson r table by nucleus and solvent. The notebook
supplies the correlation table(s) computed by delta22.py and calls these functions directly.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_feature_correlation_heatmap(corr_matrix, vmin, vmax, cmap, title="", figsize=(6, 5), save_path=None):
    """Lower-triangle seaborn heatmap of a correlation matrix (Pearson r)."""
    # In a strict lower triangle the first row and last column hold no cells, so drop them to avoid
    # a blank leading/trailing strip; the diagonal of the trimmed matrix is a real below-diagonal value.
    trimmed = corr_matrix.iloc[1:, :-1]
    mask = np.triu(np.ones(trimmed.shape, dtype=bool), k=1)   # keep the diagonal of the trimmed matrix
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(trimmed, mask=mask, annot=True, fmt=".3f", cmap=cmap, vmin=vmin, vmax=vmax,
                square=True, cbar=False, ax=ax)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_pcm_desmond_correlation_table(corr_table, figsize=(13, 4.5), save_path=None):
    """Annotated seaborn heatmap of PCM-vs-Desmond Pearson r: rows = nucleus, columns = solvent."""
    # tall figsize keeps the two rows from being squashed by the rotated solvent labels
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_table, annot=True, fmt=".3f", cmap="RdBu", vmin=-1, vmax=1, square=True,
                linewidths=0.5, cbar=False, ax=ax)
    ax.set_title("PCM vs. Desmond Pearson $r$ by Solvent")
    _sup = {"H": "$^{1}$H", "C": "$^{13}$C"}
    ax.set_yticklabels([_sup.get(t.get_text(), t.get_text()) for t in ax.get_yticklabels()], rotation=0)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
