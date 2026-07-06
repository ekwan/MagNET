"""Broken-axis accuracy-vs-cost Pareto panel used by Figure 2A and SI Figure S1.

Both figures draw the same panel (fitting RMSE vs total compute time, colored by method type, shaped by
geometry source, sized by basis) and differ only in their config dicts and which panels they show, so
that shared drawing engine lives here instead of being duplicated in each notebook. Call
`plot_pareto_panel(...)`; the notebook supplies the points table and the styling dicts.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

# methods colored as double hybrids; dlpno_mp2 is classified ab initio instead (checked first)
DOUBLE_HYBRID_METHODS = ["dsd_pbep86", "B2GP_PLYP", "B2PLYP", "mPW2PLYP", "revdsd_pbep86", "dlpno_mp2"]


def _format_seconds(seconds):
    if not np.isfinite(seconds) or seconds <= 0:
        return ""
    nice = lambda v: str(int(round(v)))
    if seconds < 60:
        return f"{nice(seconds)} s"
    if seconds < 3600:
        return f"{nice(seconds / 60)} min"
    if seconds < 86400:
        return f"{nice(seconds / 3600)} h"
    return f"{nice(seconds / 86400)} day"


def _apply_time_xticklabels(ax, xticks):
    ax.set_xticks(xticks)
    ax.set_xticklabels([_format_seconds(10 ** float(t)) for t in xticks], fontsize=10)


def _classify_method(method):
    # returns (label_text, category); category drives color, label_text is the annotation
    if method in ["hf", "mp2", "dlpno_mp2"]:
        return method, "ab initio"
    if method in DOUBLE_HYBRID_METHODS:
        return method.lower(), "double hybrids"
    if method.startswith("MagNET"):
        return "MagNET-Zero", "MagNET-Zero"
    if method.startswith(("wp", "wc")):
        return method, "NMR-specific"
    return method.split("_")[0], "DFT"


def _classify_and_plot_points(df, x_range1, ax1, ax2, color_map, geometry_marker_map,
                              marker_alpha, manual_label_positions, labels_per_category=6):
    size_dict = {"pcSseg1": 20, "pcSseg2": 50, "pcSseg3": 100, "N/A": 110}
    color_counts = {k: 0 for k in ["ab initio", "double hybrids", "MagNET-Zero", "NMR-specific", "DFT"]}
    basis_counts = {k: 0 for k in ["pcSseg1", "pcSseg2", "pcSseg3"]}
    geometry_counts = {}

    x_all = np.log10(df["total_time"].astype(float).values)
    x_leftmost = np.nanmin(x_all) if len(x_all) else np.nan

    # label selection: the min-RMSE point per category plus a few spread across time
    tmp = df.copy()
    tmp["_log10_time"] = np.log10(tmp["total_time"].astype(float))
    tmp["_rmse"] = tmp["fitting_RMSE"].astype(float)
    tmp["_category"] = tmp["nmr_method"].astype(str).apply(lambda m: _classify_method(m)[1])
    label_indices = set()
    for cat in ["MagNET-Zero", "ab initio", "double hybrids", "NMR-specific", "DFT"]:
        sub = tmp[tmp["_category"] == cat]
        if sub.empty:
            continue
        picks = [int(sub["_rmse"].idxmin())]
        n_extra = max(0, labels_per_category - 1)
        if n_extra > 0 and len(sub) > 1:
            tvals = sub["_log10_time"].to_numpy()
            for q in np.linspace(0.05, 0.95, num=n_extra):
                picks.append(int((sub["_log10_time"] - np.quantile(tvals, q)).abs().idxmin()))
        seen, unique = set(), []
        for p in picks:
            if p not in seen:
                unique.append(p); seen.add(p)
        label_indices.update(unique[:labels_per_category])

    for idx, row in df.iterrows():
        x_val = float(np.log10(float(row.total_time)))
        y_val = float(row["fitting_RMSE"])
        geometry_type = str(row["geometry_type"]); method_raw = str(row["nmr_method"]); basis = str(row["basis"])
        label_text, category = _classify_method(method_raw)
        color_counts[category] += 1
        point_color = color_map[category]
        marker = geometry_marker_map.get(geometry_type, "o")
        geometry_counts[geometry_type] = geometry_counts.get(geometry_type, 0) + 1
        if basis in basis_counts:
            basis_counts[basis] += 1
        base_size = size_dict.get(basis, size_dict.get("N/A", 30))
        size = base_size * (2.8 if abs(x_val - x_leftmost) <= 1e-12 else 1.0)
        ax = ax1 if (x_range1[0] <= x_val <= x_range1[1]) else ax2
        unfilled = marker in ["x", "+", "1", "2", "3", "4", "|", "_"]
        kws = dict(x=[x_val], y=[y_val], ax=ax, color=point_color, s=size, marker=marker,
                   legend=False, alpha=marker_alpha, zorder=3)
        kws.update(dict(linewidth=1.2) if unfilled else dict(edgecolor="white", linewidth=0.6))
        sns.scatterplot(**kws)

        override = None
        if manual_label_positions:
            for key in [label_text, method_raw, (method_raw, basis),
                        (method_raw, basis, geometry_type), (label_text, basis, geometry_type)]:
                if key in manual_label_positions:
                    override = manual_label_positions[key]; break
        if (idx in label_indices) or (override is not None):
            xytext = tuple(override["xytext"]) if isinstance(override, dict) else (4, 3)
            ax.annotate(label_text, xy=(x_val, y_val), xytext=xytext, textcoords="offset points",
                        ha="left", va="bottom", fontsize=8, color=point_color, zorder=4,
                        annotation_clip=False)
    return color_counts, basis_counts, geometry_counts


def _add_legends(ax1, ax2, color_counts, basis_counts, geometry_counts, color_map,
                 geometry_marker_map, method_legend_label_map, geometry_legend_label_map):
    method_order = ["MagNET-Zero", "ab initio", "double hybrids", "NMR-specific", "DFT"]
    method_elems = [Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[c],
                           markeredgecolor="white", markeredgewidth=0.6, markersize=8, linestyle="None",
                           label=method_legend_label_map.get(c, c))
                    for c in method_order if color_counts.get(c, 0) > 0]
    basis_elems = [Line2D([0], [0], marker="o", color="none", markerfacecolor="k", markersize=ms, label=b)
                   for b, ms in [("pcSseg1", 3), ("pcSseg2", 6), ("pcSseg3", 8)] if basis_counts.get(b, 0) > 0]
    geometry_elems = []
    for g in ["aimnet2", "pbe0_tz"]:
        if geometry_counts.get(g, 0) > 0:
            gm = geometry_marker_map.get(g, "o")
            geometry_elems.append(Line2D([0], [0], marker=gm, color="k",
                                         markerfacecolor="none" if gm == "x" else "k",
                                         markeredgecolor="k", markersize=7, linestyle="None",
                                         label=geometry_legend_label_map.get(g, g)))
    ax1.add_artist(ax1.legend(handles=method_elems, loc="upper left", bbox_to_anchor=(0.005, 0.995),
                              frameon=True, fontsize=9, title="Method Type", title_fontsize=9,
                              borderaxespad=0.0, handletextpad=0.4, labelspacing=0.3))
    ax2.add_artist(ax2.legend(handles=basis_elems, loc="upper right", bbox_to_anchor=(0.995, 0.995),
                              frameon=True, fontsize=9, title="Basis Set", title_fontsize=9,
                              borderaxespad=0.0, handletextpad=0.4, labelspacing=0.3))
    if geometry_elems:
        ax2.add_artist(ax2.legend(handles=geometry_elems, loc="upper right", bbox_to_anchor=(0.80, 0.995),
                                  frameon=True, fontsize=9, title="Geometries", title_fontsize=9,
                                  borderaxespad=0.0, handletextpad=0.4, labelspacing=0.3))


def plot_pareto_panel(pareto_df, query_str, nucleus, method_color_map, method_legend_label_map,
                      geometry_legend_label_map, geometry_marker_map, xlim_left, xlim_right, ylim,
                      figsize, marker_alpha, manual_label_positions, save_png=None,
                      suptitle="MagNET-Zero Extends a Flat Pareto Frontier",
                      supxlabel="Total delta-22 calculation time"):
    """Draw the broken-axis accuracy-vs-cost Pareto panel and optionally save it as a PNG."""
    sns.set_theme(style="ticks", context="paper", rc={"axes.grid": False})
    query_df = pareto_df.query(query_str.format(nucleus=nucleus), engine="python").reset_index(drop=True)
    default_color_map = {"ab initio": "black", "double hybrids": "#ede6ba", "MagNET-Zero": "#61a89a",
                         "NMR-specific": "#a72608", "DFT": "black"}
    color_map = {**default_color_map, **method_color_map}

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 2, width_ratios=[1, 5], wspace=0.05)
    ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1], sharey=ax1)

    cc, bc, gc = _classify_and_plot_points(query_df, xlim_left, ax1, ax2, color_map,
                                           geometry_marker_map, marker_alpha, manual_label_positions)

    ax1.set_xlim(xlim_left); ax2.set_xlim(xlim_right)
    ax1.set_ylim(ylim); ax2.set_ylim(ylim)
    y_ticks = np.linspace(ylim[0], ylim[1], num=6)
    ax1.set_yticks(y_ticks); ax1.set_yticklabels([f"{t:.2f}" for t in y_ticks], fontsize=10)

    _apply_time_xticklabels(ax1, np.array([xlim_left[0], (xlim_left[0] + xlim_left[1]) / 2, xlim_left[1]]))
    _apply_time_xticklabels(ax2, np.arange(xlim_right[0] + 0.25, xlim_right[1] + 0.25, 0.25))
    ax1.tick_params(axis="both", which="both", length=1.5, width=0.6, labelsize=10)
    ax2.tick_params(axis="x", which="both", length=1.5, width=0.6, labelsize=10)
    ax2.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
    ax1.grid(False); ax2.grid(False)

    for ax in (ax1, ax2):
        for side in ("left", "right", "top", "bottom"):
            ax.spines[side].set_visible(True); ax.spines[side].set_linewidth(0.6)
    ax1.spines["right"].set_visible(False); ax2.spines["left"].set_visible(False)

    # broken-axis break marks
    width_ratio = ax2.get_position().width / ax1.get_position().width
    d1 = 0.01; d2 = d1 / width_ratio; slope2 = width_ratio
    kw = dict(color="k", clip_on=False, lw=0.6, transform=ax1.transAxes)
    ax1.plot((1 - d1, 1 + d1), (-d1, d1), **kw); ax1.plot((1 - d1, 1 + d1), (1 - d1, 1 + d1), **kw)
    kw.update(transform=ax2.transAxes)
    ax2.plot((-d2, d2), (-d2 * slope2, d2 * slope2), **kw); ax2.plot((-d2, d2), (1 - d2 * slope2, 1 + d2 * slope2), **kw)

    fig.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.12)
    fig.suptitle(suptitle, fontsize=14)
    fig.supxlabel(supxlabel, fontsize=13)
    ax1.set_ylabel(r"RMSE ($^{1}$H ppm, CDCl$_3$)" if nucleus == "H" else r"RMSE ($^{13}$C ppm, CDCl$_3$)",
                   fontsize=13)

    _add_legends(ax1, ax2, cc, bc, gc, color_map, geometry_marker_map,
                 method_legend_label_map, geometry_legend_label_map)
    if save_png:
        fig.savefig(save_png, dpi=300, bbox_inches="tight")
    plt.show()
