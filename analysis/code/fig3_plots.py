"""Plotting engines for Figure 3 panels 3A, 3B, and 3D (delta-22 solvent-correction figures).

Each panel's notebook (fig3a_pcm_benefit.ipynb, fig3b_shifts.ipynb, fig3d_solvent_corrections.ipynb)
supplies the data table and styling dicts and calls the corresponding function here.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.ticker import MaxNLocator


def plot_pcm_benefit_with_arrows(benefit_df, combos, colors, save_path,
                                  xlimits=(-20, 35), figsize=(6, 6)):
    """Figure 3A: implicit-solvent (PCM) benefit by method, chloroform vs benzene."""
    solvents = ["chloroform", "benzene"]
    labels = [m for (m, _b, _g) in combos]
    arrow_label = "__arrow__"
    ordered_labels = [arrow_label] + labels        # blank spacer row at the top holds the arrows

    rows = []
    for method, basis, geom in combos:
        for solvent in solvents:
            sub = benefit_df[(benefit_df["sap_nmr_method"] == method) &
                             (benefit_df["sap_basis"] == basis) &
                             (benefit_df["sap_geometry_type"] == geom) &
                             (benefit_df["solvent"] == solvent)]
            for v in sub["percent_benefit"]:
                rows.append({"Label": method, "Solvent": solvent, "PCM_Benefit": float(v)})
    long_df = pd.DataFrame(rows)
    long_df = pd.concat([long_df, pd.DataFrame([{"Label": arrow_label, "Solvent": solvents[0],
                                                 "PCM_Benefit": float(np.mean(xlimits))}])],
                        ignore_index=True)
    long_df["Label"] = pd.Categorical(long_df["Label"], categories=ordered_labels, ordered=True)

    sns.set_theme(context="paper", style="white")
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=long_df, x="PCM_Benefit", y="Label", order=ordered_labels,
                hue="Solvent", hue_order=solvents, palette=colors, dodge=False, width=0.3,
                fliersize=0, linewidth=0.5, whiskerprops=dict(alpha=0.5, linewidth=0.8),
                capprops=dict(alpha=0.5, linewidth=1), medianprops=dict(alpha=0.9, linewidth=0.4),
                boxprops=dict(alpha=0.9), ax=ax, orient="h")

    # blank the spacer row's tick label
    ticks = ax.get_yticks()
    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(mticker.FixedFormatter(
        ["" if t.get_text() == arrow_label else t.get_text() for t in ax.get_yticklabels()]))

    leg = ax.get_legend()
    if leg is not None:
        leg.set_title("")
        for txt in leg.get_texts():
            txt.set_text(txt.get_text().capitalize())
        leg.set_frame_on(True); leg.get_frame().set_alpha(0.95)
        for t in leg.get_texts():
            t.set_fontsize(13)
        leg.set_bbox_to_anchor((0.8, 1.02)); leg._loc = 9

    # arrows on the blank spacer row
    y_spacer = ordered_labels.index(arrow_label)
    span = xlimits[1] - xlimits[0]
    offset = 0.02 * span

    def box_arrow(text, x, ha, facecolor):
        ax.text(x, y_spacer, text, ha=ha, va="center", fontsize=11, fontweight="bold",
                color="white", clip_on=False, zorder=10,
                bbox=dict(boxstyle=f"{'larrow' if ha=='right' else 'rarrow'},pad=0.35",
                          fc=facecolor, ec="none", alpha=0.85))
    box_arrow("Improves Acc.", +offset, "left", "#708B75")
    box_arrow("Decreases Acc.", -offset, "right", "#C3A29E")

    ax.axvline(0, color="black", lw=0.2, alpha=0.4, zorder=0)
    ax.set_xlim(*xlimits)
    ax.margins(x=0.01)
    ax.set_xlabel("Benefit of PCM (%)", fontsize=13)
    ax.set_ylabel("")
    sns.despine(ax=ax, left=False, bottom=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.tick_params(axis="both", which="major", length=1.5, width=0.6, labelsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_shift_vs_pcm(shift_df, save_path, xlimits=(-0.2, 0.2), ylimits=(-1.6, 0.3), figsize=(8, 4)):
    """Figure 3B: measured vs PCM-predicted solvent-induced shift differences."""
    sns.set_theme(style="white", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    # left: experimental solvent-induced shifts (teal) -- large and uncorrelated
    sns.scatterplot(data=shift_df, x="exp_diff_x", y="exp_diff_y", ax=axes[0],
                    color="#44AA99", s=15, alpha=1.0, edgecolor=None)
    # right: PCM solvent corrections (dark red) -- small and linearly correlated. Same axis limits
    # as the left panel, so the tiny PCM predictions are visible against the real shifts.
    pcm = shift_df.dropna(subset=["pcm_diff_x", "pcm_diff_y"])
    sns.scatterplot(data=pcm, x="pcm_diff_x", y="pcm_diff_y", ax=axes[1],
                    color="#A72608", s=15, alpha=0.5, edgecolor=None)
    for ax in axes:
        ax.set_xlim(*xlimits); ax.set_ylim(*ylimits)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
        ax.tick_params(axis="both", which="major", direction="out", length=1.5, width=0.6, labelsize=10)
    axes[0].set_xlabel(r"$\delta_{MeOD} - \delta_{CDCl_3}$ ($^{1}\mathrm{H}$ ppm)", fontsize=13)
    axes[0].set_ylabel(r"$\delta_{benzene} - \delta_{CDCl_3}$ ($^{1}\mathrm{H}$ ppm)", fontsize=13)
    axes[1].set_xlabel(r"$\Delta\sigma_{MeOH} - \Delta\sigma_{CHCl_3}$ ($^{1}\mathrm{H}$ ppm)", fontsize=13)
    axes[1].set_ylabel(r"$\Delta\sigma_{benzene} - \Delta\sigma_{CHCl_3}$ ($^{1}\mathrm{H}$ ppm)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    sns.despine()
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_solvent_correction_boxplot(results_df, formula_labels, solvent_order, colors,
                                    solvent_labels, save_path, box_widths=0.70, figsize=(8, 6)):
    """Figure 3D: implicit vs explicit solvent corrections, one box per solvent."""
    labels = list(formula_labels.values())          # implicit, implicit+vib, explicit, explicit+vib
    palette = dict(zip(labels, colors))
    df = results_df[results_df["formula"].isin(formula_labels)].copy()
    df["Model"] = df["formula"].map(formula_labels)
    df = df[df["solvent"].isin(solvent_order)].copy()
    df["solvent"] = pd.Categorical(df["solvent"], categories=solvent_order, ordered=True)
    xpos = {s: float(i) for i, s in enumerate(solvent_order)}
    df["xpos"] = df["solvent"].map(xpos).astype(float)

    sns.set_theme(context="paper", style="white")
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=df, x="xpos", y="test_RMSE", hue="Model", hue_order=labels, palette=palette,
                width=box_widths, showfliers=False, linewidth=0.5,
                whiskerprops=dict(alpha=0.5, linewidth=0.8), capprops=dict(alpha=0.5, linewidth=1),
                medianprops=dict(alpha=0.9, linewidth=0.4), boxprops=dict(alpha=0.9), ax=ax)

    # center each solvent's tick under its "Explicit Solvent" box
    per_box = box_widths / len(labels)
    offset = (labels.index("Explicit Solvent") - (len(labels) - 1) / 2) * per_box
    ax.set_xticks([xpos[s] + offset for s in solvent_order])
    ax.set_xticklabels([solvent_labels.get(s, s) for s in solvent_order], rotation=45, fontsize=8, ha="right")

    ax.set_ylabel(r"Accuracy vs. Benchmark (RMSE, $^{1}\mathrm{H}$ ppm)", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylim(bottom=0)
    ax.set_title("Accuracy Improves Broadly with Explicit Solvation", fontsize=14, fontweight="bold")
    leg = ax.get_legend()
    if leg is not None:
        leg.set_title("")
        for t in leg.get_texts():
            t.set_fontsize(13)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.tick_params(axis="both", which="major", direction="out", length=1.5, width=0.6, labelsize=10)
    ax.grid(False)
    sns.despine(ax=ax)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
