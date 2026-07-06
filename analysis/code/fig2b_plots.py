"""Correlation-matrix plotting engine for Figure 2B.

The notebook computes the pairwise Pearson correlation table across methods (delta-22 1H stationary
shieldings, pcSseg-2 basis) and calls `plot_correlation_matrix(...)` to draw the lower-triangle
heatmap.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_correlation_matrix(corr_matrix, save_png, figsize=(6, 6)):
    """Draw and save the Figure 2B lower-triangle heatmap of pairwise Pearson correlations."""
    arr = corr_matrix.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(arr, np.nan)                    # ignore the self-correlation diagonal
    transformed = pd.DataFrame(-np.log10(1 - arr),   # spread values near 1 so differences are visible
                               index=corr_matrix.index, columns=corr_matrix.columns)
    # Tight lower triangle: the first row and last method have no pairs below the diagonal, so drop the
    # first row and last column, then hide the strict upper triangle. Every drawn cell is then a real
    # method pair, with no empty self-correlation squares on the axes.
    transformed = transformed.iloc[1:, :-1]
    mask = np.triu(np.ones(transformed.shape, dtype=bool), k=1)

    sns.set_theme(style="white", context="paper")
    fig, ax = plt.subplots(figsize=figsize)

    r_ticks = [0.99, 0.999, 0.9999]
    cb_values = [-np.log10(1 - r) for r in r_ticks]
    hm = sns.heatmap(transformed, mask=mask, cmap="Reds", vmin=1, vmax=4, square=True,
                     linewidths=1, cbar_kws={"shrink": 0.65, "pad": -0.05, "ticks": cb_values}, ax=ax)

    cbar = hm.collections[0].colorbar
    cbar.set_ticks(cb_values)
    cbar.set_ticklabels([f"{r:g}" for r in r_ticks])  # ticks are Pearson r values
    cax = cbar.ax
    pos = cax.get_position()
    cax.set_position([pos.x0, pos.y0 + 0.04, pos.width, pos.height])
    cax.text(0.5, 1.08, r"Pearson $\boldsymbol{r}$", transform=cax.transAxes, ha="center", va="bottom",
             fontsize=11, bbox=dict(boxstyle="round,pad=0.25", facecolor="#EBEBEB", edgecolor="none", alpha=0.3))

    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(axis="x", length=1.5, width=0.6, labelsize=10)
    ax.tick_params(axis="y", length=1.5, width=0.6, labelsize=10)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor", va="top")
    plt.setp(ax.get_yticklabels(), rotation=0, va="center")
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)

    fig.savefig(save_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.show()
