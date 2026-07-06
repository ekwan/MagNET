"""Figure 4 panel-drawing engines.

4A/4B share one scatter engine (predicted vs measured solvent-induced shift, implicit vs explicit
prediction column); 4C is a dumbbell plot of per-solvent fit RMSE vs experimental shift range. The
notebook supplies the data table and the styling dicts (highlight map, order, palette, markers /
colors) as arguments and calls `save_panel(...)` to draw and write the PNG.
"""
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text


def draw_shift_scatter(ax, shift_df, pred_col, highlight_map, order, palette, markers, limits=(-0.25, 2.0)):
    """Figure 4A/4B: scatter predicted vs experimental solvent-induced shift, colored by solvent."""
    df = shift_df[shift_df["solvent"].isin(highlight_map)].copy()
    df["Solvent"] = df["solvent"].map(highlight_map)
    sns.scatterplot(data=df, x="exp_diff", y=pred_col, hue="Solvent", style="Solvent",
                    hue_order=order, style_order=order, palette=palette,
                    markers=markers, s=60, alpha=0.6, edgecolor=None, ax=ax)
    lo, hi = limits
    ax.plot([lo, hi], [lo, hi], linestyle="--", alpha=0.35, linewidth=0.6, color="k")  # ideal y = x
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_box_aspect(1)
    ax.set_xlabel(r"Experimental Solvent-Induced Shift (Δδ vs. CDCl3, $^{1}\mathrm{H}$ ppm)",
                  fontsize=13, labelpad=10)
    ax.set_ylabel(r"Predicted Solvent-Induced Shift (Δδ vs. CDCl3, $^{1}\mathrm{H}$ ppm)",
                  fontsize=13, labelpad=10)
    leg = ax.get_legend()
    if leg is not None:
        leg.set_title(None)
    for side in ax.spines:
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(labelsize=10, length=1.5, width=0.6)


def save_panel(draw_fn, save_path, figsize=(6, 6)):
    """Create a figure/axes, call `draw_fn(ax)`, then save and show the panel as a PNG."""
    sns.set_theme(style="ticks", context="paper")
    fig, ax = plt.subplots(figsize=figsize)
    draw_fn(ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def draw_rmse_dumbbell(ax, rows, labels, colors):
    """Figure 4C: dumbbell plot of implicit vs explicit solvent-correction RMSE per solvent."""
    rows_sorted = sorted(rows, key=lambda r: r["range"])
    xs = [r["range"] for r in rows_sorted]
    for r in rows_sorted:                                   # gray dumbbell connectors
        ax.plot([r["range"], r["range"]], [r["implicit"], r["explicit"]],
                color="#cccccc", lw=1.5, zorder=1)
    ax.scatter(xs, [r["implicit"] for r in rows_sorted], s=60, marker="D",
               color=colors["Implicit (PCM)"], alpha=0.85, edgecolors="none",
               label="Implicit (PCM)", zorder=2)
    ax.scatter(xs, [r["explicit"] for r in rows_sorted], s=60, marker="o",
               color=colors["Explicit (Desmond)"], alpha=0.85, edgecolors="none",
               label="Explicit (Desmond)", zorder=2)
    # frame the data, then label each dumbbell at its midpoint (adjustText avoids overlaps)
    xr = max(xs) - min(xs); yv = [r[k] for r in rows_sorted for k in ("implicit", "explicit")]
    ax.set_xlim(min(xs) - 0.06 * xr, max(xs) + 0.10 * xr)
    ax.set_ylim(min(yv) - 0.06 * (max(yv) - min(yv)), max(yv) + 0.10 * (max(yv) - min(yv)))
    texts = [ax.text(r["range"], 0.5 * (r["implicit"] + r["explicit"]), labels[r["solvent"]],
                     fontsize=10, color="black", va="center", ha="left") for r in rows_sorted]
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.8))
    ax.legend(loc="upper left", framealpha=0.9, fontsize=11)
    ax.set_box_aspect(1)
    ax.set_xlabel(r"Range of Solvent-Induced Shifts ($^{1}\mathrm{H}$ ppm)", fontsize=13, labelpad=10)
    ax.set_ylabel(r"Predicted Solvent-Induced Shift RMSE ($^{1}\mathrm{H}$ ppm)", fontsize=13, labelpad=10)
    for side in ax.spines:
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(labelsize=10, length=1.5, width=0.6)
