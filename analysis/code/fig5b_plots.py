"""Plotting engines for Figure 5B (MagNET-Zero vs DFT on the DFT8K benchmark).

The notebook computes residuals / functional-group error tables via `dft8k_residuals` and calls the
drawing functions here; `show_dft8k_molecule` renders the inline-only RDKit structure callouts (not
saved to disk).
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_dft8k_residual_histogram(errors, save_path, color="#5D737E", bin_width=0.0025, xlim=0.23,
                                  grey_box=0.1, x_tick=0.05, figsize=(8, 3)):
    """Draw the DFT8K residual histogram for one nucleus and save it to save_path (or skip saving
    if save_path is None)."""
    # The published panel is slate (#5D737E) with a shaded +/-0.1 ppm window; the "MagNET-Zero Closely
    # Reproduces DFT" title and "X% of Compounds" bracket are added in the figure compositing step.
    sns.set_theme(context="paper", style="white")
    fig, ax = plt.subplots(figsize=figsize)
    edges = np.arange(-xlim, xlim + bin_width, bin_width)
    sns.histplot(x=errors, bins=edges, stat="count", element="bars", ax=ax,
                 color=color, edgecolor=None, alpha=0.9, linewidth=0)
    ax.set_xlim(-xlim, xlim)
    ax.set_xlabel("Residual vs. DFT8K (ppm)", fontsize=13)
    ax.set_ylabel("Counts", fontsize=13)
    if grey_box is not None:                      # shade the +/- grey_box ppm window
        ax.axvspan(-grey_box, grey_box, alpha=0.05, facecolor="k", edgecolor="none", zorder=0)
        for xv in (-grey_box, grey_box):
            ax.axvline(xv, color="k", ls="--", lw=1.0, alpha=0.35, zorder=1)
    if x_tick is not None:
        ax.set_xticks(np.arange(-np.floor(xlim / x_tick) * x_tick, xlim + x_tick / 2, x_tick))
    sns.despine(ax=ax, top=True, right=True)
    ax.spines["left"].set_linewidth(0.6); ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="both", which="major", direction="out", length=1.5, width=0.6, labelsize=10)
    ax.minorticks_off()
    fig.tight_layout()
    if save_path:                                 # 13C is an inline-only companion, not saved
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_dft8k_functional_group_errors(group_errors, nucleus, save_path, figsize=(7, 3.5)):
    """Bar chart of mean |residual| by functional group for one nucleus, saved to save_path."""
    # The panel composites these mean |residual| values as molecule-icon insets; this bar chart is
    # the plain reproduction of the numbers.
    names = list(group_errors.keys())
    values = [group_errors[n]["mean_abs_error"] for n in names]
    counts = [group_errors[n]["n_molecules"] for n in names]
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(names, values, color="#4C9B8F")
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"n={n:,}",
                ha="center", va="bottom", fontsize=8)
    lbl = "¹H" if nucleus == "H" else "¹³C"
    ax.set_ylabel(f"mean |{lbl} residual| (ppm)")
    ax.set_title(f"{lbl} error by functional group (whole DFT8K set)")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    if save_path:                                 # only the 1H panel is a saved deliverable
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def show_dft8k_molecule(smiles, size=(350, 300)):
    """Display an RDKit structure drawing for the given SMILES inline; nothing is saved to disk."""
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    from IPython.display import Image, display
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    drawer.drawOptions().addStereoAnnotation = False
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    display(Image(drawer.GetDrawingText()))       # inline only; whole-molecule, no atom highlighting
