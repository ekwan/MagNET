"""Plotting engines for the delta-22 SI figure notebooks (analysis/si_figures/*.ipynb).

Each function is the drawing engine behind one delta-22 SI figure panel. Notebook globals an engine
needs (color maps, axis-label lookups, N_SPLITS, solvent lists, etc.) are explicit function
parameters, passed in by the notebook at the call site. The numbers come from delta22.py; this
module only draws, except `ss_fits`, a small data-prep helper used only by the si_figure_s14
notebook.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from scipy.stats import pearsonr, linregress
from adjustText import adjust_text

import delta22


# ----------------------------------------------------------------------------
# color helpers shared by the engine/source panels (si_figure_s08, si_figure_s13)
# ----------------------------------------------------------------------------
def darken_color(hex_color, factor=0.7):
    """Scale a "#rrggbb" color's channels by factor (< 1 darkens)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(c * factor) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def lighten_color(hex_color, amount=0.45):
    """Blend a "#rrggbb" color toward white by amount (0-1)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(c + (255 - c) * amount) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def display_solvent_name(name):
    """Human-readable solvent label ("Water (TIP4P)" for TIP4P, else capitalized)."""
    return "Water (TIP4P)" if name == "TIP4P" else name.capitalize()


# ----------------------------------------------------------------------------
# SI Figure S4: correlation of implicit (PCM) corrections across solvents and methods
# ----------------------------------------------------------------------------
def plot_correlation_matrix(corr_matrix, title, caption, colormap="Reds", show_values=True, save_path=None):
    """Lower-triangle heatmap of -log10(1 - r) with a Pearson-R colorbar; a value of 3 means r=0.999."""
    arr = corr_matrix.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(arr, np.nan)
    transformed = pd.DataFrame(-np.log10(1 - arr), index=corr_matrix.index, columns=corr_matrix.columns)
    mask = np.tril(np.ones_like(transformed, dtype=bool), k=0).T
    vmin, vmax = -np.log10(1 - 0.9), -np.log10(1 - 0.9999)
    fig = plt.figure(figsize=(8, 8))
    ax = sns.heatmap(transformed, mask=mask, annot=transformed if show_values else False, fmt=".1f",
                     cmap=colormap, cbar_kws={"shrink": 0.5, "pad": -0.13},
                     vmin=vmin, vmax=vmax, square=True)
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([-np.log10(1 - r) for r in (0.99, 0.999, 0.9999)])
    cbar.set_ticklabels(["0.99", "0.999", "0.9999"])
    cbar.ax.set_title("Pearson R", fontweight="bold", pad=25)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold")
    ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold")
    ax.set_yticks(ax.get_yticks()[1:]); ax.set_xticks(ax.get_xticks()[:-1])
    plt.title(title, fontweight="bold")
    plt.figtext(0.59, 0.895, caption, wrap=True, horizontalalignment="center", fontsize=10)
    for side in ("top", "right", "bottom", "left"):
        plt.gca().spines[side].set_visible(True)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_pcm_scatter(x_vals, y_vals, solvent_x, solvent_y, nucleus, save_path=None):
    """Scatter of PCM corrections in one solvent vs another, with a best-fit line, slope, and Pearson R."""
    pearson_r = x_vals.corr(y_vals)
    slope, _ = np.polyfit(x_vals, y_vals, 1)
    cap = lambda s: s[0].upper() + s[1:]
    fig = plt.figure(figsize=(6.5, 6.5))
    sns.scatterplot(x=x_vals, y=y_vals, s=60, color="black", edgecolor="none")
    sns.regplot(x=x_vals, y=y_vals, scatter=False, color="black", ci=None,
                line_kws={"linewidth": 1, "linestyle": "--"})
    plt.xlabel(f"{cap(solvent_x)} PCM correction")
    plt.ylabel(f"{cap(solvent_y)} PCM correction")
    plt.title(f"PCM corrections: {cap(solvent_x)} vs {cap(solvent_y)} ({nucleus})", fontweight="bold")
    plt.text(0.05, 0.96, f"Slope = {slope:.3f}", transform=plt.gca().transAxes, va="top")
    plt.text(0.05, 0.92, f"Pearson R = {pearson_r:.3f}", transform=plt.gca().transAxes, va="top")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ----------------------------------------------------------------------------
# SI Figure S5: per-solvent PCM benefit vs bulk dielectric constant and polarizability
# ----------------------------------------------------------------------------
def plot_pcm_benefit_vs_properties(benefit, dielectric, polarizability, nucleus_label,
                                   exclude=(), title_extra="", figsize=(14, 6), save_path=None):
    """Per-solvent PCM benefit (percent test-RMSE reduction) vs dielectric constant and
    polarizability, one subplot each, with a best-fit line and Pearson R. exclude drops solvents."""
    solvents = [s for s in benefit.index if s not in set(exclude) and s in dielectric]
    y = np.array([benefit[s] for s in solvents], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, prop, xlabel in [(axes[0], dielectric, "Dielectric Constant"),
                             (axes[1], polarizability, "Polarizability")]:
        x = np.array([prop[s] for s in solvents], dtype=float)
        r, _ = pearsonr(x, y)
        slope, intercept, _, _, _ = linregress(x, y)
        ax.scatter(x, y, s=100, color="black", alpha=1, edgecolors="black", linewidth=1, zorder=3)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.4, linewidth=1, zorder=1)
        xr = np.linspace(x.min(), x.max(), 100)
        ax.plot(xr, slope * xr + intercept, color="gray", linestyle="--", linewidth=2,
                label=f"R = {r:.3f}", zorder=2)
        ax.set_xlabel(xlabel, fontsize=13, fontweight="bold")
        ax.set_ylabel("Percent Reduction in Test RMSE (%)", fontsize=13, fontweight="bold")
        title = f"{nucleus_label} Nucleus: PCM Benefit vs {xlabel}"
        if title_extra:
            title += f"\n({title_extra})"
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        texts = [ax.text(sx, sy, name, fontsize=9, fontweight="bold")
                 for sx, sy, name in zip(x, y, solvents)]
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig


# ----------------------------------------------------------------------------
# SI Figure S6: per-solvent test RMSE of six solvent-correction models
# ----------------------------------------------------------------------------
def plot_formula_ladder_boxplot(results_df, formula_labels, solvent_groups, nucleus="H",
                                colors=None, figsize=(14, 7), title="", save_path=None):
    """Per-solvent box-and-whisker of the test RMSE for a solvent-correction formula ladder, with
    solvents grouped by class (a dashed line separates classes)."""
    formulas = list(formula_labels)
    n = len(formulas)
    if colors is None:
        colors = sns.color_palette("colorblind", n)
    # lay the solvents out class by class (only those present), tracking where classes end
    flat, boundaries = [], []
    for group in solvent_groups:
        present = [s for s in solvent_groups[group] if s in set(results_df["solvent"])]
        flat.extend(present)
        boundaries.append(len(flat))
    fig, ax = plt.subplots(figsize=figsize)
    box_width = 0.8 / n
    gap = 0.6
    centers, separators = [], []
    pos = 0.0
    for si, solvent in enumerate(flat):
        centers.append(pos)
        for fi, formula in enumerate(formulas):
            vals = results_df[(results_df["formula"] == formula)
                              & (results_df["solvent"] == solvent)]["test_RMSE"].dropna().to_numpy()
            bp = ax.boxplot([vals], positions=[pos + (fi - (n - 1) / 2) * box_width],
                            widths=box_width * 0.9, showfliers=False, patch_artist=True,
                            manage_ticks=False)
            bp["boxes"][0].set(facecolor=colors[fi], alpha=0.9)
            bp["medians"][0].set(color="black")
            if si == 0:
                bp["boxes"][0].set_label(formula_labels[formula])
        pos += 1.0
        if (si + 1) in boundaries[:-1]:   # a class just ended (not the last)
            separators.append(pos - 0.5 + gap / 2)
            pos += gap
    for x in separators:
        ax.axvline(x, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(centers)
    ax.set_xticklabels(flat, rotation=60, ha="right", fontweight="bold")
    ax.set_ylabel("Test RMSE (ppm)", fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------------
# SI Figure S7: DFT 13C explicit-solvent correction, Desmond vs OpenMM
# ----------------------------------------------------------------------------
def plot_desmond_vs_openmm_grid(pairs_by_solvent, figsize=(10, 10), save_path=None):
    """Per OpenMM solvent, a scatter of the Desmond vs OpenMM explicit correction at each site.
    Tight clustering on the diagonal shows the correction is independent of the MD engine."""
    solvents = list(pairs_by_solvent)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()
    for ax, solvent in zip(axes, solvents):
        p = pairs_by_solvent[solvent]
        ax.scatter(p["openMM"], p["desmond"], c="k", s=5)
        ax.set_xlabel("OpenMM solvent correction (ppm)", fontweight="bold")
        ax.set_ylabel("Desmond solvent correction (ppm)", fontweight="bold")
        title = "Water (TIP4P)" if solvent == "TIP4P" else solvent.capitalize()
        ax.set_title(title, fontweight="bold")
    for ax in axes[len(solvents):]:
        ax.set_visible(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------------
# SI Figure S8 (panels B-D): explicit-solvent correction vs MD-frame count
# ----------------------------------------------------------------------------
def plot_frame_convergence(running_by_label, finals_by_label, label_colors, label_names,
                           title="", xlabel="Number of Frames", figsize=(6, 5), save_path=None):
    """Running-average correction vs number of frames, one line per label, with each label's
    converged value drawn as a dashed horizontal line."""
    fig, ax = plt.subplots(figsize=figsize)
    for label, running in running_by_label.items():
        color = label_colors.get(label)
        valid = running[~np.isnan(running)]
        display = label_names.get(label, label)
        ax.plot(np.arange(1, len(valid) + 1), valid, color=color, lw=1.1, label=display)
        final = finals_by_label[label]
        ax.axhline(final, color=color, ls="--", lw=1, label=f"{display} final: {final:.3f} ppm")
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel("Running Average Correction (ppm)", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_frame_correction_histogram(values_by_label, total_frames_by_label, label_colors, label_names,
                                    bins=30, title="", figsize=(6, 4.5), save_path=None):
    """Distribution of valid per-frame corrections, one overlaid histogram per label; each bin's
    height is a fraction of that label's total frame count, with mean/std in the legend."""
    fig, ax = plt.subplots(figsize=figsize)
    for label, values in values_by_label.items():
        valid = values[~np.isnan(values)]
        weights = np.full(len(valid), 1.0 / total_frames_by_label[label])
        display = label_names.get(label, label)
        ax.hist(valid, bins=bins, weights=weights, color=label_colors.get(label), alpha=0.55,
                edgecolor="black", linewidth=0.3,
                label=f"{display} ($\\mu$={valid.mean():.3f}, $\\sigma$={valid.std():.3f})")
    ax.set_xlabel("Correction per Frame (ppm)", fontweight="bold")
    ax.set_ylabel("Frequency / Frame Count", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_frame_autocorrelation(autocorr_by_label, label_colors, label_names, title="",
                               figsize=(8, 4.5), save_path=None):
    """Autocorrelation of the per-frame correction vs lag (frames), one line per label."""
    fig, ax = plt.subplots(figsize=figsize)
    for label, autocorr in autocorr_by_label.items():
        ax.plot(np.arange(len(autocorr)), autocorr, color=label_colors.get(label), lw=1,
                label=label_names.get(label, label))
    ax.axhline(0, color="0.8", lw=0.8)
    ax.set_xlabel("Lag (frames)", fontweight="bold")
    ax.set_ylabel("Autocorrelation", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_frame_validity_heatmaps(grids_by_engine, solvents_by_engine, engine_labels, solute="AcOH",
                                 figsize=(10, 8), save_path=None):
    """One heatmap per MD engine: which trajectory frames have computed DFT shielding data (light
    blue) vs not (dark gray), one row per solvent and one column per frame index."""
    engines = list(grids_by_engine)
    fig, axes = plt.subplots(len(engines), 1, figsize=figsize, squeeze=False)
    cmap = ListedColormap(["#1a1a1a", "#a8dadc"])   # False = dark gray, True = light blue
    for ax, engine in zip(axes[:, 0], engines):
        ax.imshow(grids_by_engine[engine], aspect="auto", cmap=cmap, vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_yticks(range(len(solvents_by_engine[engine])))
        ax.set_yticklabels(solvents_by_engine[engine], fontsize=8)
        ax.set_title(f"{solute} Frame Validities ({engine_labels.get(engine, engine)})",
                     fontweight="bold", fontsize=10)
    axes[-1, 0].set_xlabel("Frame index")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------------
# SI Figure S11: MagNET vs DFT rovibrational (QCD) corrections
# ----------------------------------------------------------------------------
def plot_qcd_scatter(qcd, save_path=None):
    """NN vs DFT QCD correction, one subplot per nucleus, with a y=x guide and an R^2/RMSE/MAE
    stats box measured against that line."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, nucleus in zip(axes, ["H", "C"]):
        sub = qcd[qcd["nucleus"] == nucleus]
        x, y = sub["qcd_dft"].to_numpy(), sub["qcd_nn"].to_numpy()
        ss_res, ss_tot = np.sum((y - x) ** 2), np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
        rmse, mae = np.sqrt(np.mean((y - x) ** 2)), np.mean(np.abs(y - x))
        print(f"{nucleus}: R^2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
        ax.scatter(x, y, color="black")
        raw = [min(x.min(), y.min()), max(x.max(), y.max())]
        buf = 0.05 * (raw[1] - raw[0])
        lims = [raw[0] - buf, raw[1] + buf]
        ax.plot(lims, lims, linestyle="--", color="gray", label="NN matches DFT", zorder=-1)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("DFT QCD"); ax.set_ylabel("NN QCD")
        ax.set_title(f"DFT vs NN QCD ({nucleus})", fontweight="bold")
        ax.legend()
        ax.text(0.02, 0.98, f"R^2 = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_qcd_error_histogram(qcd, save_path=None):
    """Distribution of the NN-minus-DFT QCD error, one subplot per nucleus."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, nucleus in zip(axes, ["H", "C"]):
        data = qcd[qcd["nucleus"] == nucleus]["error"]
        ax.hist(data, bins=30, color="gray", alpha=0.8, edgecolor="black")
        ax.set_title(f"QCD Error Distribution (NN - DFT) for {nucleus}", fontweight="bold")
        ax.set_xlabel("Error"); ax.set_ylabel("Count")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_qcd_correction_by_site(by_site, nucleus, legend_loc, dft_color="#A0A0A0", nn_color="#707070",
                                save_path=None):
    """The DFT and NN QCD correction for every site, one column of points per solute; sites within
    a solute are jittered horizontally and joined by a faint vertical line."""
    fig, ax = plt.subplots(figsize=(12, 6))
    solute_order = sorted(by_site["solute"].unique())
    x_pos = {s: i for i, s in enumerate(solute_order)}
    jitter = 0.02
    site_offset = {}
    for solute, group in by_site.groupby("solute"):
        sites = sorted(group["site"].unique())
        if len(sites) == 1:
            site_offset[(solute, sites[0])] = 0.0
        else:
            for site, off in zip(sites, np.linspace(-jitter, jitter, len(sites))):
                site_offset[(solute, site)] = off
    for _, row in by_site.iterrows():
        x = x_pos[row["solute"]] + site_offset[(row["solute"], row["site"])]
        ax.vlines(x, row["qcd_dft"], row["qcd_nn"], color=dft_color, alpha=0.2, linewidth=1)
        ax.scatter(x, row["qcd_dft"], color=dft_color, s=32, edgecolor="black", linewidth=0.3, alpha=0.85)
        ax.scatter(x, row["qcd_nn"], color=nn_color, s=32, edgecolor="black", linewidth=0.3, alpha=0.85)
    ax.set_xticks(range(len(solute_order)))
    ax.set_xticklabels(solute_order, rotation=65, ha="right", fontweight="bold")
    ax.set_ylabel("QCD correction (ppm)")
    ax.set_title(f"QCD: DFT vs NN ({nucleus})", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    handles = [Line2D([0], [0], color=dft_color, marker="o", linestyle="-", label="DFT QCD"),
               Line2D([0], [0], color=nn_color, marker="o", linestyle="-", label="NN QCD")]
    ax.legend(handles=handles, ncol=1, fontsize=9, frameon=True, loc=legend_loc)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------------
# SI Figure S13: MagNET-x vs DFT explicit-solvent corrections
# ----------------------------------------------------------------------------
def plot_dft_nn_scatter_by_engine(compare_df, engine_colors, engine_labels, save_path=None):
    """NN vs DFT explicit correction at each site, both nuclei, Desmond and OpenMM overlaid, with
    a y=x guide."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, nucleus in zip(axes, ["H", "C"]):
        sub = compare_df[compare_df["nucleus"] == nucleus]
        allv = []
        for engine in ["desmond", "openMM"]:
            d = sub[sub["engine"] == engine]
            ax.scatter(d["dft_value"], d["nn_value"], alpha=0.9, color=engine_colors[engine],
                       label=engine_labels[engine])
            allv += [d["dft_value"].to_numpy(), d["nn_value"].to_numpy()]
        allv = np.concatenate(allv)
        raw = [allv.min(), allv.max()]
        buf = 0.05 * (raw[1] - raw[0])
        lims = [raw[0] - buf, raw[1] + buf]
        ax.plot(lims, lims, linestyle="--", color="gray", label="NN matches DFT", zorder=-1)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("DFT Explicit Correction"); ax.set_ylabel("NN Explicit Correction")
        ax.set_title(f"DFT vs NN Explicit Corrections ({nucleus})", fontweight="bold")
        ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_dft_nn_error_histogram_by_engine(compare_df, engine_colors, engine_labels, save_path=None):
    """Distribution of the NN-minus-DFT explicit-correction error, both nuclei, Desmond and OpenMM
    overlaid."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, nucleus in zip(axes, ["H", "C"]):
        sub = compare_df[compare_df["nucleus"] == nucleus]
        for engine in ["desmond", "openMM"]:
            data = sub[sub["engine"] == engine]["error"].to_numpy()
            if len(data) == 0:
                continue
            ax.hist(data, bins=30, color=engine_colors[engine], alpha=0.6, edgecolor="black",
                    label=engine_labels[engine])
        ax.set_title(f"Explicit Correction Error (NN - DFT) for {nucleus}", fontweight="bold")
        ax.set_xlabel("Error"); ax.set_ylabel("Count")
        ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_explicit_correction_by_site(pairs_df, title, engine_colors, save_path=None):
    """The Desmond and OpenMM explicit correction from both DFT (base hue) and NN (darkened), one
    column per solute; the two engines are offset, sites are jittered, and each DFT/NN pair is
    joined by a faint vertical line."""
    fig, ax = plt.subplots(figsize=(12, 6))
    solute_order = sorted(pairs_df["solute"].unique())
    x_pos = {s: i for i, s in enumerate(solute_order)}
    engine_offset = {"desmond": -0.12, "openMM": 0.12}
    jitter = 0.02
    site_offset = {}
    for solute, group in pairs_df.groupby("solute"):
        sites = sorted(group["site"].unique())
        if len(sites) == 1:
            site_offset[(solute, sites[0])] = 0.0
        else:
            for site, off in zip(sites, np.linspace(-jitter, jitter, len(sites))):
                site_offset[(solute, site)] = off
    dft_color = {e: engine_colors[e] for e in ("desmond", "openMM")}
    nn_color = {e: darken_color(engine_colors[e]) for e in ("desmond", "openMM")}
    for engine in ["desmond", "openMM"]:
        for _, row in pairs_df.iterrows():
            x = x_pos[row["solute"]] + engine_offset[engine] + site_offset[(row["solute"], row["site"])]
            dft_val, nn_val = row[f"{engine}_dft"], row[f"{engine}_nn"]
            ax.vlines(x, dft_val, nn_val, color=dft_color[engine], alpha=0.18, linewidth=1)
            ax.scatter(x, dft_val, color=dft_color[engine], s=32, edgecolor="black", linewidth=0.3, alpha=0.85)
            ax.scatter(x, nn_val, color=nn_color[engine], s=32, edgecolor="black", linewidth=0.3, alpha=0.85)
    ax.set_xticks(range(len(solute_order)))
    ax.set_xticklabels(solute_order, rotation=65, ha="right", fontweight="bold")
    ax.set_ylabel("Explicit Solvent Correction (ppm)")
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    handles = [Line2D([0], [0], color=dft_color["desmond"], marker="o", linestyle="-", label="DFT Desmond"),
               Line2D([0], [0], color=nn_color["desmond"], marker="o", linestyle="-", label="NN Desmond"),
               Line2D([0], [0], color=dft_color["openMM"], marker="o", linestyle="-", label="DFT OpenMM"),
               Line2D([0], [0], color=nn_color["openMM"], marker="o", linestyle="-", label="NN OpenMM")]
    ax.legend(handles=handles, ncol=2, fontsize=9, frameon=True, loc="upper left")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_fitting_accuracy_boxplot(results_df, solvents, nuc_label, engine_colors, save_path=None):
    """The semi-parsimonious composite model's test RMSE vs experiment, four boxes per solvent
    (Desmond DFT/NN, OpenMM DFT/NN) so DFT-based and MagNET-x-based explicit terms sit side by side."""
    groups = [("desmond", "DFT", "Desmond (DFT)"), ("desmond", "NN", "Desmond (NN)"),
              ("openMM", "DFT", "OpenMM (DFT)"), ("openMM", "NN", "OpenMM (NN)")]
    colors = [lighten_color(engine_colors["desmond"], 0.15), lighten_color(engine_colors["desmond"], 0.55),
              lighten_color(engine_colors["openMM"], 0.15), lighten_color(engine_colors["openMM"], 0.55)]
    n = len(groups)
    width = 0.8 / n
    x_base = np.arange(len(solvents))
    fig, ax = plt.subplots(figsize=(11, 5))
    for gi, (engine, source, label) in enumerate(groups):
        data = [results_df[(results_df["solvent"] == sv) & (results_df["engine"] == engine)
                           & (results_df["source"] == source)]["test_RMSE"].dropna().to_numpy()
                for sv in solvents]
        positions = x_base + (gi - (n - 1) / 2) * width
        bp = ax.boxplot(data, positions=positions, widths=width * 0.9, patch_artist=True,
                        showfliers=False, manage_ticks=False)
        for box in bp["boxes"]:
            box.set(facecolor=colors[gi], alpha=0.9)
        for median in bp["medians"]:
            median.set(color="black")
        bp["boxes"][0].set_label(label)
    ax.set_xticks(x_base)
    ax.set_xticklabels(solvents)
    ax.set_ylabel(f"Accuracy vs. Experiment (RMSE, {nuc_label} ppm)")
    ax.set_title(f"DFT vs NN Explicit Corrections, Fitting Accuracy ({nuc_label})")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------------
# SI Figure S14: test RMSE with DFT vs MagNET features, by solvent
# ----------------------------------------------------------------------------
def ss_fits(query, nucleus, formulas, solvents, n_splits, dft=False):
    """Solvent-specific test RMSEs over the explicit solvents (SI Figure S14), nitromethane dropped."""
    q = query[(query["nucleus"] == nucleus) & (query["solute"] != "nitromethane")]
    if dft:
        # DFT features need one method per nucleus (the MagNET-Zero training reference: WP04/pcSseg2
        # for 1H, wB97X-D/pcSseg2 for 13C, AIMNet2 geometries) or the fit pools every method together.
        # The NN table has only MagNET, so it needs no such filter.
        method = delta22.MAGNET_PCM_OUTPUT_METHODS[nucleus]
        q = q[(q["sap_nmr_method"] == method) & (q["sap_basis"] == "pcSseg2")
              & (q["sap_geometry_type"] == "aimnet2")]
    solutes = sorted(q["solute"].unique())
    return delta22.run_fits(q, solvents, formulas, n_splits=n_splits, solutes=solutes)


def plot_ss_boxplot_dft_vs_nn(dft_df, nn_df, solvents, formulas, labels, nucleus,
                              solvent_labels=None, colors=("#A72608", "#5D737E", "#D9FFF5"),
                              figsize=(14, 8), box_span=0.7, group_gap=0.35, save_path=None):
    """Per solvent, box plots of the solvent-specific test-RMSE distributions for each composite
    formula, computed with DFT features (solid) and MagNET/NN features (lightened), side by side."""
    def series(df, formula, solvent):
        return df[(df["formula"] == formula) & (df["solvent"] == solvent)]["test_RMSE"].dropna().values

    labels_x = [(solvent_labels or {}).get(s, s) for s in solvents]
    n_formulas = len(formulas)
    n_boxes = 2 * n_formulas                          # all DFT boxes, then all NN boxes, per solvent
    box_w = box_span / n_boxes
    centers = [i * (box_span + group_gap) for i in range(len(solvents))]
    fig, ax = plt.subplots(figsize=figsize)
    handles = []
    # draw the DFT boxes (base colors) first, then the NN boxes (lightened), matching the SI legend
    for source_index, (df, tag, lighten) in enumerate([(dft_df, "DFT", False), (nn_df, "NN", True)]):
        for fi, (formula, label, color) in enumerate(zip(formulas, labels, colors)):
            rgb = plt.cm.colors.to_rgb(color)
            face = tuple(min(1.0, c + 0.3) for c in rgb) if lighten else color
            slot = source_index * n_formulas + fi
            offset = (slot - (n_boxes - 1) / 2) * box_w
            data = [series(df, formula, s) for s in solvents]
            ax.boxplot(data, positions=[c + offset for c in centers], widths=box_w * 0.9,
                       patch_artist=True,
                       boxprops=dict(color="gray", facecolor=face, alpha=0.9),
                       medianprops=dict(color="black", linewidth=0.5),
                       whiskerprops=dict(linewidth=0.4), capprops=dict(linewidth=0.4),
                       flierprops=dict(marker="o", markersize=0))
            handles.append(plt.Rectangle((0, 0), 1, 1, fc=face, ec="gray", alpha=0.9,
                                         label=f"{label} ({tag})"))
    ax.set_xticks(centers)
    ax.set_xticklabels(labels_x, rotation=45, ha="right", fontweight="bold")
    ax.set_ylabel("Test RMSE (ppm)", fontweight="bold")
    ax.set_title(f"DFT vs NN Solvent-Specific Test RMSEs\nModel Comparison ({nucleus} nucleus)",
                 fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.legend(handles=handles, loc="upper left", fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------------
# SI Figures S16-S18: predicted vs experimental solvent-induced shifts by reference solvent
# ----------------------------------------------------------------------------
def plot_shift_prediction_scatter_grid(diff_df, solvents, reference_label, axis_label, title_label,
                                       n_cols=4, save_path=None):
    """Per solvent, the implicit (PCM) and explicit (Desmond) predicted solvent-induced shift
    differences (y) against the measured ones (x), with a y=x guide."""
    n = len(solvents)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    series = [("implicit_diff", "#FF6B6B", "D"), ("explicit_diff", "#4ECDC4", "o")]
    for ax, solvent in zip(axes, solvents):
        d = diff_df[diff_df["solvent"] == solvent]
        vals = np.concatenate([d["exp_diff"].to_numpy()] + [d[c].to_numpy() for c, *_ in series])
        lim = float(np.nanmax(np.abs(vals))) * 1.05 if len(vals) else 1.0
        ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3, lw=1.5, zorder=0)
        for col, color, marker in series:
            ax.scatter(d["exp_diff"], d[col], s=40, color=color, marker=marker, zorder=2)
        tok = axis_label[solvent]
        ax.set_xlabel(f"Experimental delta ({tok} - {reference_label}) [ppm]")
        ax.set_ylabel(f"Correction delta ({reference_label} - {tok}) [ppm]")
        ax.set_title(title_label[solvent], fontweight="bold")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    legend_handles = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#FF6B6B", markersize=9, label="Implicit"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4ECDC4", markersize=9, label="Explicit"),
        Line2D([0], [0], color="gray", linestyle="--", label="Ideal (y = x)")]
    empty = list(axes[n:])
    for ax in empty:
        ax.set_visible(False)
    if empty:                      # put the legend in the first empty grid slot (S16/S17)
        empty[0].set_visible(True); empty[0].axis("off")
        empty[0].legend(handles=legend_handles, loc="center", frameon=False, fontsize=12)
        fig.tight_layout()
    else:                          # full 12-panel grid (S18): legend centred below the grid
        fig.tight_layout(rect=[0, 0.045, 1, 1])
        fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, fontsize=12)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
