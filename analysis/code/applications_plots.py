"""Plotting for the natural-products figures (main-text Figures 5C/5D, SI Figure S15).

Holds every plotting engine for the applications (natural-products) notebooks: the delta-22-vs-test-set
benefit bar chart (Figure 5C), the shared per-solute box-plot engine (Figure 5D and SI Figure S15's
bootstrap-RMSE panels), and SI Figure S15's fitting-RMSE comparison bars, feature-space/residual scatter
grids, and distribution-shift bars. Engines are carried over from the source notebooks with notebook
globals turned into explicit arguments. The numbers come from applications.py; this module is kept
separate so applications.py stays plotting-free.
"""
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_nps_on_boxplot_delta22_simplified(
    regression_ss_df,
    nps_rmse_df,
    best_possible_df,  # expects a 'rmse' column
    nucleus,
    formulas,
    colors,
    site_counts,
    solvents_filter=None,
    box_width=0.20,
    box_gap=0.05,
    formula_gap=0.15,
    figsize=(14, 8),
    formula_remap=None,
    solute_remap=None,
    solute_order=None,
    title=None,
    save_path=None,
    solute_color_remap=None,
    max_bar_height=0.05,

    # --- SELF-SCALING BASELINE CONTROLS ---
    show_baseline=True,
    baseline_annotation_text='Self-scaling baseline (dotted)',
    baseline_annotation_x=0.985,
    baseline_annotation_y=0.93,
    baseline_annotation_fontsize=9,
    
    # --- FULL FIT LINE CONTROLS ---
    show_full_fit_line=False,
    all_solute_fitting_results=None,  # {nucleus: per-solvent fit df}; required when show_full_fit_line
    full_fit_line_label='Full Fit RMSE',
    full_fit_line_label_x=0.99,

    # --- INSET AXIS CONTROLS ---
    site_count_axis_mode='attached',
    site_count_inset_axis_offset=0.0,
    site_count_inset_area_frac=0.1,
    site_count_inset_axis_label_pad=0.04,
    site_count_alpha=0.25,
    site_count_label='Site count',
    site_count_preserve_visual_height=True,

    # ---- font sizes ----
    xtick_fontsize=10,
    delta22_fontsize=10,
    ylabel_fontsize=13,
    title_fontsize=16,

    # ---- SAVE SVG ----
    save_svg_path=None,
    default_svg_name="Figure_5C_extras.svg",
    svg_transparent=True,
):
    """
    Simplified box and whisker plot:
    - Plots all solutes separately (no grouping)
    - Averages RMSE across solvents for each bootstrap seed
    - Shows "best possible" RMSE as dashed baseline for each solute

    Parameters:
    - regression_ss_df: Delta-22 regression samples (Bootstrap_RMSE)
    - nps_rmse_df: Should have columns: solvent, nucleus, formula, seed, solute, Bootstrap_RMSE
    - best_possible_df: DataFrame with best possible RMSEs per solute/solvent/formula
                       Should have columns: formula, solvent, solute, rmse, params
    - nucleus: Which nucleus to filter for
    - formulas: List of formulas to compare
    - colors: Colors for each formula
    - site_counts: DataFrame indexed by solute with nucleus count columns (e.g., 'H', 'C')
    - solvents_filter: Optional list of solvents to include in averaging
    - show_baseline: Whether to show the 'best possible df' baseline
    - show_full_fit_line: Whether to draw a line representing the full-dataset fit RMSE across all solutes
    - full_fit_line_label: Label for the full fit line
    - full_fit_line_label_x: X-coordinate for the full fit line label
    - box_width: Width of each box
    - box_gap: Gap between boxes
    - formula_gap: Gap between formulas
    - figsize: Figure size
    - formula_remap: Optional dict to remap formula names
    - max_bar_height: Maximum visual height target for site-count bars relative to RMSE axis
    - site_count_axis_mode: One of {'attached', 'inset', 'off'}
    - site_count_inset_axis_offset: Horizontal inset-axis adjustment in axes coordinates (clipped to stay inside in inset mode)
    - site_count_inset_area_frac: Fraction of total figure area to allocate to inset axis (inset mode only)
    - site_count_alpha: Alpha for site-count bars
    - site_count_label: Label for site-count axis
    - site_count_preserve_visual_height: Keep the old visual bar footprint when using a right axis
    """
    if formula_remap is None:
        formula_remap = {}

    if solute_remap is None:
        solute_remap = {}

    if solute_color_remap is None:
        solute_color_remap = {}

    valid_axis_modes = {'attached', 'inset', 'off'}
    if site_count_axis_mode not in valid_axis_modes:
        raise ValueError(f"site_count_axis_mode must be one of {sorted(valid_axis_modes)}")

    if site_counts is not None and not isinstance(site_counts, pd.DataFrame):
        raise TypeError("site_counts must be a pandas DataFrame or None")

    if site_count_inset_area_frac < 0 or site_count_inset_area_frac > 1:
        raise ValueError("site_count_inset_area_frac must be between 0 and 1")

    def get_rounded_count_ticks(max_count, target_ticks=7):
        if max_count <= 0:
            return np.array([0.0, 1.0]), 1.0

        rough_step = max_count / max(target_ticks - 1, 1)
        magnitude = 10 ** np.floor(np.log10(rough_step)) if rough_step > 0 else 1.0
        normalized = rough_step / magnitude
        if normalized <= 1:
            nice_base = 1
        elif normalized <= 2:
            nice_base = 2
        elif normalized <= 5:
            nice_base = 5
        else:
            nice_base = 10

        step = nice_base * magnitude
        rounded_top = np.ceil(max_count / step) * step
        ticks = np.arange(0, rounded_top + 0.5 * step, step)
        return ticks, rounded_top

    n_formulas = len(formulas)

    if solvents_filter is not None:
        regression_ss_df = regression_ss_df[regression_ss_df['solvent'].isin(solvents_filter)].copy()
        nps_rmse_df = nps_rmse_df[nps_rmse_df['solvent'].isin(solvents_filter)].copy()
        best_possible_df = best_possible_df[best_possible_df['solvent'].isin(solvents_filter)].copy()

        if regression_ss_df.empty:
            raise ValueError(f"No regression data found for solvents: {solvents_filter}")
        if nps_rmse_df.empty:
            raise ValueError(f"No NPS RMSE data found for solvents: {solvents_filter}")

    def get_delta22_averaged_rmse(regression_df, formula):
        df = regression_df[regression_df['formula'] == formula]
        return df.groupby('seed')['Bootstrap_RMSE'].mean().values

    def get_solute_bootstrap_averaged(nps_df, formula, nucleus):
        np_formula = formula_remap.get(formula, formula)
        df = nps_df[
            (nps_df['formula'] == np_formula) &
            (nps_df['nucleus'] == nucleus)
        ]
        if df.empty:
            return {}

        # average over solvent per solute/seed, then collect each solute's per-seed array
        averaged_df = df.groupby(['solute', 'seed'])['Bootstrap_RMSE'].mean().reset_index()
        solute_dict = {}
        for solute in averaged_df['solute'].unique():
            arr = averaged_df[averaged_df['solute'] == solute]['Bootstrap_RMSE'].values
            if arr.size > 0:
                solute_dict[solute] = arr
        return solute_dict

    def get_best_possible_averaged(best_df, solute, formula):
        np_formula = formula_remap.get(formula, formula)
        df = best_df[
            (best_df['formula'] == np_formula) &
            (best_df['solute'] == solute)
        ]

        if df.empty:
            # fall back to the raw formula name in case best_possible_df wasn't remapped
            df = best_df[
                (best_df['formula'] == formula) &
                (best_df['solute'] == solute)
            ]

        if df.empty:
            return None
        return df['rmse'].mean()

    all_solutes = set()
    for formula in formulas:
        np_formula = formula_remap.get(formula, formula)
        df_filtered = nps_rmse_df[
            (nps_rmse_df['formula'] == np_formula) &
            (nps_rmse_df['nucleus'] == nucleus)
        ]
        all_solutes.update(df_filtered['solute'].unique())
    all_solutes = sorted(list(all_solutes))
    if solute_order is not None:                 # explicit ordering (e.g. main-text Figure 5D)
        ordered = [s for s in solute_order if s in all_solutes]
        all_solutes = ordered + [s for s in all_solutes if s not in ordered]

    delta22_by_formula = [get_delta22_averaged_rmse(regression_ss_df, f) for f in formulas]

    # each formula block is 1 delta-22 box plus one box per solute
    total_boxes_per_formula = 1 + len(all_solutes)

    all_data = []
    all_positions = []
    all_colors = []
    all_alphas = []
    tick_positions = []
    tick_labels = []
    baseline_segments = []  # Store (x_start, x_end, y_value, solute_label) for baseline segments

    count_bar_positions = []
    count_bar_values = []
    delta22_count_by_nucleus = {'H': 43.0, 'C': 50.0}

    current_position = 0.0

    for j, formula in enumerate(formulas):

        # 1) Delta-22 box (solvent-averaged)
        box_pos = current_position + box_width / 2
        all_data.append(delta22_by_formula[j])
        all_positions.append(box_pos)
        all_colors.append(colors[j])
        all_alphas.append(1.0)
        tick_positions.append(box_pos)
        tick_labels.append('delta-22')

        delta22_count = delta22_count_by_nucleus.get(nucleus)
        if delta22_count is not None:
            count_bar_positions.append(box_pos)
            count_bar_values.append(float(delta22_count))

        current_position += box_width + box_gap

        # 2) Individual solute boxes (solvent-averaged, lighter alpha)
        solute_data = get_solute_bootstrap_averaged(nps_rmse_df, formula, nucleus)

        count_col = None
        if site_counts is not None:
            if nucleus in site_counts.columns:
                count_col = nucleus
            elif 'H' in site_counts.columns:
                count_col = 'H'

        for solute in all_solutes:
            box_pos = current_position + box_width / 2
            if solute in solute_data and len(solute_data[solute]) > 0:
                all_data.append(solute_data[solute])
                all_positions.append(box_pos)
                all_colors.append(solute_color_remap.get(solute, colors[j]))
                all_alphas.append(0.5)
                tick_positions.append(box_pos)
                tick_labels.append(solute_remap.get(solute, solute))

                if (
                    count_col is not None
                    and solute in site_counts.index
                    and pd.notna(site_counts.loc[solute, count_col])
                ):
                    count_bar_positions.append(box_pos)
                    count_bar_values.append(float(site_counts.loc[solute, count_col]))
                else:
                    count_bar_positions.append(box_pos)
                    count_bar_values.append(0.0)

                # best_possible_df has no nucleus column, so this doesn't filter by nucleus
                if show_baseline:
                    best_rmse = get_best_possible_averaged(best_possible_df, solute, formula)
                    if best_rmse is not None:
                        # baseline segment spans the box's full width including its gap
                        x_start = current_position - box_gap / 2
                        x_end = current_position + box_width + box_gap / 2
                        baseline_segments.append((x_start, x_end, best_rmse, solute))

            current_position += box_width + box_gap

        # Add gap after formula (except last)
        if j < n_formulas - 1:
            current_position += formula_gap

    plot_figsize = figsize
    fig, ax = plt.subplots(figsize=plot_figsize)

    # skip boxes with no data
    valid_indices = [i for i, data in enumerate(all_data) if len(data) > 0]
    valid_data = [all_data[i] for i in valid_indices]
    valid_positions = [all_positions[i] for i in valid_indices]
    valid_colors = [all_colors[i] for i in valid_indices]
    valid_alphas = [all_alphas[i] for i in valid_indices]

    if valid_data:
        box = ax.boxplot(
            valid_data,
            positions=valid_positions,
            widths=box_width,
            showfliers=False,
            patch_artist=True,
            whiskerprops=dict(color='#404040'),
            capprops=dict(color='#404040')
        )

        for patch, color, alpha in zip(box['boxes'], valid_colors, valid_alphas):
            patch.set_facecolor(color)
            patch.set_alpha(alpha)
        for median in box['medians']:
            median.set_color('#404040')

    if show_baseline:
        for i, (x_start, x_end, y_value, _) in enumerate(baseline_segments):
            ax.plot([x_start, x_end], [y_value, y_value],
                    color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

            # connect to the previous segment unless this is the first solute of a formula block
            # (connects across formula gaps too, not just within one)
            if len(formulas) > 0 and len(baseline_segments) > 0 and i % (len(baseline_segments) / len(formulas)) != 0:
                _, prev_x_end, prev_y, _ = baseline_segments[i - 1]
                ax.plot([prev_x_end, x_start], [prev_y, y_value],
                        color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    if show_baseline and baseline_segments and baseline_annotation_text:
        ax.text(
            baseline_annotation_x,
            baseline_annotation_y,
            baseline_annotation_text,
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=baseline_annotation_fontsize,
            color='gray',
            style='italic',
            alpha=0.85,
        )

    # the full-fit-line RMSE uses the first formula's delta-22 fit as its reference
    if show_full_fit_line:
        formula_remapped = formula_remap.get(formulas[0], formulas[0])
        query = f'formula == "{formula_remapped}"' + (f' and solvent in {solvents_filter}' if solvents_filter is not None else '')
        full_fit_rmse = all_solute_fitting_results[nucleus].query(query)['rmse'].mean()
        ax.axhline(y=full_fit_rmse, color='red', linestyle='-.', linewidth=1.5, alpha=0.7)
        ax.text(full_fit_line_label_x,
                full_fit_rmse, f'{full_fit_line_label}: {full_fit_rmse:.3f} ppm',
                transform=ax.get_yaxis_transform(),
                ha='right', va='bottom', fontsize=9, color='red', style='italic', alpha=0.7)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=75, fontsize=xtick_fontsize, ha='center')
    for i, label_obj in enumerate(ax.get_xticklabels()):
        if tick_labels[i] == 'delta-22':
            label_obj.set_weight('bold')
            label_obj.set_fontsize(delta22_fontsize)
            label_obj.set_style('italic')

    # vertical separators between formula blocks
    for j in range(1, n_formulas):
        x_pos = j * (total_boxes_per_formula * box_width + formula_gap) - formula_gap / 2
        ax.axvline(x=x_pos, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    if solvents_filter is not None:
        solvents_str = ', '.join(solvents_filter)
        ylabel = f"1H RMSE (ppm)\n(Averaged across: {solvents_str})"
    else:
        ylabel = "1H RMSE (ppm)"
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize, fontweight='bold')
    ax.set_ylim(bottom=0)

    # caller may override via title=; the default text reflects the solvent filter
    if title is None:
        if solvents_filter is not None:
            title = f"Delta22 vs Individual Solute Bootstrap Sample RMSEs ({nucleus} nucleus; {solvents_filter})"
        else:
            title = f"Delta22 vs Individual Solute Bootstrap Sample RMSEs ({nucleus} nucleus, solvent-averaged)"
    ax.set_title(title, fontsize=title_fontsize, fontweight='bold')

    ax.set_xlim(-0.2, current_position + 0.2)

    # site-count bars use a dedicated right (twin) axis
    added_site_count_bars = False
    site_count_ax = None
    max_count = max(count_bar_values) if count_bar_values else 0.0
    if site_count_axis_mode != 'off' and max_bar_height > 0 and len(count_bar_positions) > 0 and max_count > 0:
        site_count_ax = ax.twinx()

        tick_counts, rounded_count_top = get_rounded_count_ticks(max_count)
        if site_count_preserve_visual_height:
            rmse_top = ax.get_ylim()[1]
            rmse_top = max(rmse_top, 1e-6)
            bar_target = max(max_bar_height, 1e-6)
            # Keep bars in the lower band while right-axis tick labels remain true counts.
            axis_top = max(rounded_count_top * (rmse_top / bar_target), rounded_count_top * 1.05)
        else:
            axis_top = max(rounded_count_top, 1.0)

        site_count_ax.bar(
            count_bar_positions,
            count_bar_values,
            width=box_width + box_gap,
            bottom=0,
            color='#404040',
            alpha=site_count_alpha,
            edgecolor='none',
            zorder=0,
        )

        site_count_ax.set_ylim(0, axis_top)

        if site_count_axis_mode == 'inset':
            # Hide inset twin-axis renderer and draw a short attached axis manually.
            for spine_name in ('left', 'right', 'top', 'bottom'):
                site_count_ax.spines[spine_name].set_visible(False)
            site_count_ax.patch.set_visible(False)
            site_count_ax.yaxis.set_visible(False)

            inset_axis_top_frac = min(rounded_count_top / axis_top, 1.0)
            # clip to keep the inset axis inside the plotting area near the right edge
            inset_x = float(np.clip(1.0 + site_count_inset_axis_offset, 0.0, 1.0))

            ax.plot(
                [inset_x, inset_x],
                [0.0, inset_axis_top_frac],
                transform=ax.transAxes,
                color='#404040',
                linewidth=1.0,
                clip_on=False,
                zorder=3,
            )

            # Manual ticks and tick labels, confined to the short segment.
            tick_len = 0.008
            tick_label_pad = 0.012
            for t in tick_counts:
                if t == 0:
                    continue
                y_frac = t / axis_top
                if y_frac <= inset_axis_top_frac + 1e-9:
                    ax.plot(
                        [inset_x, inset_x + tick_len],
                        [y_frac, y_frac],
                        transform=ax.transAxes,
                        color='#404040',
                        linewidth=0.9,
                        clip_on=False,
                        zorder=3,
                    )
                    ax.text(
                        inset_x + tick_len + tick_label_pad,
                        y_frac,
                        f"{int(t)}",
                        transform=ax.transAxes,
                        va='center',
                        ha='left',
                        fontsize=10,
                        color='#404040',
                        clip_on=False,
                    )

            label_x = inset_x + site_count_inset_axis_label_pad
            ax.text(
                label_x,
                inset_axis_top_frac / 2,
                site_count_label,
                transform=ax.transAxes,
                rotation=90,
                va='center',
                ha='left',
                fontsize=11,
                fontweight='bold',
                color='#404040',
                clip_on=False,
            )
        else:
            site_count_ax.set_yticks(tick_counts)
            site_count_ax.set_yticklabels([f"{int(t)}" for t in tick_counts])
            site_count_ax.set_ylabel(site_count_label, fontsize=13, fontweight='bold', color='#404040')
            site_count_ax.tick_params(axis='y', labelcolor='#404040')
            site_count_ax.spines['right'].set_color('#404040')

        site_count_ax.grid(False)

        # Keep RMSE artists visually above site-count bars.
        ax.set_zorder(2)
        ax.patch.set_alpha(0)
        site_count_ax.set_zorder(1)
        added_site_count_bars = True


    if site_count_axis_mode == 'inset' and added_site_count_bars:
        ax.set_xlim(left=-box_width/2-box_gap/2, right=4.55)
    plt.tight_layout()
    # PNG is the release format; SVG is also supported for standalone vector export
    if save_path is not None:
        outdir = os.path.dirname(save_path)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if save_svg_path is not None:
        # If user passed a directory, save into it with default name
        if os.path.isdir(save_svg_path) or save_svg_path.endswith(os.sep):
            save_svg_path = os.path.join(save_svg_path, default_svg_name)

        if not save_svg_path.lower().endswith(".svg"):
            save_svg_path = save_svg_path + ".svg"

        outdir = os.path.dirname(save_svg_path)
        if outdir:
            os.makedirs(outdir, exist_ok=True)

        fig.savefig(
            save_svg_path,
            format="svg",
            bbox_inches="tight",
            pad_inches=0.02,
            transparent=svg_transparent,
        )
        print(f"Saved SVG to: {os.path.abspath(save_svg_path)}")

    plt.show()


def plot_nps_benefit_barplot(regression_ss_df, nps_rmse_df, nucleus, solvents, np_solute_groups,
                             formulas, labels, colors, formula_remap=None, figsize=(6, 5),
                             box_width=0.18, formula_gap=0.04, solvent_gap=0.35, spacer_width=0.0,
                             y_min=0.0, y_max=None, save_path=None):
    """Grouped bar chart of mean bootstrap RMSE per (solvent, dataset, formula), with 2.5-97.5
    percentile bootstrap error bars. Delta-22 is drawn first within each solvent, then the
    NP/complex bin. Used by main-text Figure 5C (implicit vs explicit correction, delta-22 vs the
    test set)."""
    sns.set_theme(style="ticks", context="paper")
    formula_remap = formula_remap or {}
    n_formulas = len(formulas)
    label_remap = {labels[0]: "Implicit", labels[1]: "Explicit"}
    label_to_color = {label_remap.get(labels[i], labels[i]): colors[i] for i in range(len(labels))}
    flat_solvents = [s for g in solvents for s in g] if isinstance(solvents[0], list) else list(solvents)
    solvent_groups_iter = solvents if isinstance(solvents[0], list) else [solvents]

    def d22_by_solvent(formula):
        df = regression_ss_df[regression_ss_df["formula"] == formula]
        return {s: df[df["solvent"] == s]["Bootstrap_RMSE"].dropna().values for s in flat_solvents}

    def np_grouped(formula, solvent):
        df = nps_rmse_df[(nps_rmse_df["formula"] == formula) & (nps_rmse_df["solvent"] == solvent)
                         & (nps_rmse_df["nucleus"] == nucleus)]
        if df.empty:
            return {}
        out = {}
        for g in np_solute_groups.keys():
            arr = df[df["solute_group"] == g]["Bootstrap_RMSE"].dropna().values
            if arr.size:
                out[g] = arr
        return out

    def summ(vals):
        vals = np.asarray(vals, float); vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            return None
        m = np.mean(vals); lo, hi = np.percentile(vals, [2.5, 97.5])
        return m, m - lo, hi - m

    d22 = [d22_by_solvent(f) for f in formulas]
    fig, ax = plt.subplots(figsize=figsize)
    category_gap = 0.55
    x = 0.0
    ticks, ticklabels, solvent_centers, seen = [], [], {}, {}

    def draw_pair(center, getter):
        drew = False
        for j, formula in enumerate(formulas):
            stats = getter(j, formula)
            if stats is None:
                continue
            mean, elo, ehi = stats
            disp = label_remap.get(labels[j], labels[j])
            offset = (j - (n_formulas - 1) / 2) * (box_width + formula_gap)
            bar = ax.bar(center + offset, mean, width=box_width,
                         yerr=np.array([[elo], [ehi]]), capsize=3, color=label_to_color[disp],
                         edgecolor="black", linewidth=0.9, zorder=3,
                         label=disp if disp not in seen else None)
            seen.setdefault(disp, bar)
            drew = True
        return drew

    for solvent_group in solvent_groups_iter:
        for solvent in solvent_group:
            centers = [x]
            draw_pair(x, lambda j, f: summ(d22[j].get(solvent, np.array([]))))
            ticks.append(x); ticklabels.append("delta-22"); x += category_gap
            for bin_label in np_solute_groups.keys():
                if draw_pair(x, lambda j, f, s=solvent, b=bin_label:
                             summ(np_grouped(formula_remap.get(f, f), s).get(b, [])) if
                             np_grouped(formula_remap.get(f, f), s).get(b) is not None else None):
                    centers.append(x); ticks.append(x)
                    ticklabels.append("complex" if bin_label == "Test Set" else bin_label)
                    x += category_gap
            solvent_centers[solvent] = np.mean(centers)
            x += solvent_gap
        x += spacer_width

    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels, rotation=45, fontsize=9, style="italic", ha="center")
    for lab in ax.get_xticklabels():
        if lab.get_text() == "delta-22":
            lab.set_fontweight("bold"); lab.set_fontsize(10); lab.set_style("normal")
    for solvent, center in solvent_centers.items():
        ax.text(center, -0.17, solvent, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=12, fontweight="bold")
    nuc = "^{1}\\mathrm{H}" if nucleus == "H" else "^{13}\\mathrm{C}"
    ax.set_ylabel(r"RMSE $(" + nuc + r"$ ppm)", fontsize=13)
    ax.set_ylim(bottom=y_min, top=y_max)
    disp_nuc = "1H" if nucleus == "H" else "13C"
    ax.set_title(f"Delta-22 vs NP/Isomer Bootstrap RMSE by Solvent ({disp_nuc} nucleus)",
                 fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=10, length=4, width=0.8)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.4, alpha=0.35, zorder=0)
    ax.xaxis.grid(False)
    sns.despine(ax=ax, top=True, right=True)
    for side in ["left", "bottom", "right", "top"]:
        ax.spines[side].set_linewidth(0.9)
    leg = ax.legend(frameon=False, fontsize=10)
    if leg is not None:
        leg.set_title("")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_fitting_rmse_comparison_bars(table, nucleus, solvent, figsize=(11, 4), save_path=None):
    """One group of three bars per test-set solute: fit on that solute alone ("Scaled to Solute"),
    fit once across the whole test set ("Scaled to Test Set"), and delta-22's bootstrap
    coefficients applied cold ("Extrapolated from delta22"). Used by SI Figure S15's "Fitting RMSE
    Comparisons" panel."""
    colors = {"Scaled to Solute": "#4C72B0", "Scaled to Test Set": "#DD8452",
              "Extrapolated from delta22": "#55A868"}
    columns = list(colors.keys())
    x = np.arange(len(table))
    width = 0.27
    fig, ax = plt.subplots(figsize=figsize)
    for i, col in enumerate(columns):
        ax.bar(x + (i - 1) * width, table[col].to_numpy(), width,
               label=col.replace("delta22", "Δ22"), color=colors[col])
    ax.set_xticks(x)
    ax.set_xticklabels([str(s).replace("\n", " ") for s in table.index], rotation=30, ha="right")
    ax.set_ylabel(f"{'¹H' if nucleus == 'H' else '¹³C'} RMSE (ppm)")
    ax.set_title(f"Fitting RMSE Comparisons ({solvent}, {nucleus})")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, ax


# --- SI Figure S15 feature-space / residual 2x2 grids + distribution-shift bars ---
# style constants private to the two _dataset_scatter_grid callers below.
_DATASET_COLORS = {"Test Set": "#7e57c2", "Delta22": "#2ca02c"}   # purple circles / green squares
_DATASET_MARKERS = {"Test Set": "o", "Delta22": "s"}
_SOLVENT_DISPLAY = {"chloroform": "Chloroform", "benzene": "Benzene",
                    "methanol": "Methanol", "TIP4P": "Water (TIP4P)"}


def _dataset_scatter_grid(table, x_col, y_col, solvents, title, subtitle, x_label, y_label,
                          figsize, zero_lines):
    """2x2 solvent grid overlaying the Test Set and delta-22 point clouds, shared x/y across panels."""
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, solvent in zip(axes, solvents):
        sdf = table[table["solvent"] == solvent]
        for dataset in ["Test Set", "Delta22"]:
            sub = sdf[sdf["dataset"] == dataset]
            if not sub.empty:
                ax.scatter(sub[x_col], sub[y_col], s=32, alpha=0.75, marker=_DATASET_MARKERS[dataset],
                           c=_DATASET_COLORS[dataset], edgecolors="white", linewidths=0.5, label=dataset)
        if zero_lines:
            ax.axhline(0, color="0.35", linestyle="--", linewidth=1.0, zorder=0)
            ax.axvline(0, color="0.35", linestyle="--", linewidth=1.0, zorder=0)
        else:
            ax.axhline(0, color="0.5", linestyle="--", linewidth=1.0, zorder=0)
        ax.set_title(_SOLVENT_DISPLAY.get(solvent, solvent), fontsize=11, fontweight="semibold")
        ax.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.88, 0.5), frameon=False, title="Dataset")
    fig.supxlabel(x_label, fontsize=11)
    fig.supylabel(y_label, fontsize=11)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.text(0.5, 0.94, subtitle, ha="center", fontsize=10, color="0.35")
    fig.tight_layout(rect=[0.02, 0.02, 0.86, 0.92])
    return fig, axes


def plot_feature_space_coverage_grid(table, nucleus, x_label,
                                     solvents=("chloroform", "benzene", "methanol", "TIP4P"),
                                     figsize=(11, 9), save_path=None):
    """SI Figure S15's "Feature Space Coverage by Solvent" panel."""
    subtitle = f"{'Hydrogen' if nucleus == 'H' else 'Carbon'} sites: Test Set vs Delta22"
    fig, axes = _dataset_scatter_grid(table, "x", "y", solvents, "Feature Space Coverage by Solvent",
                                      subtitle, x_label, "OpenMM Correction (centered, ppm)",
                                      figsize, zero_lines=True)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, axes


def plot_delta22_plane_residuals_grid(table, nucleus,
                                      solvents=("chloroform", "benzene", "methanol", "TIP4P"),
                                      figsize=(11, 9), save_path=None):
    """SI Figure S15's "Residuals for Delta22 Fitting Coefficients" panel."""
    subtitle = f"{'Hydrogen' if nucleus == 'H' else 'Carbon'} sites: Test Set vs Delta22"
    fig, axes = _dataset_scatter_grid(table, "experimental", "residual", solvents,
                                      "Residuals for Delta22 Fitting Coefficients", subtitle,
                                      "Experimental Shielding (ppm)",
                                      "Residual vs Delta22 Plane (experimental - predicted)",
                                      figsize, zero_lines=False)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, axes


def plot_distribution_shift_by_solvent_bars(table, nucleus, figsize=(7, 5), save_path=None):
    """One group of three bars per solvent: extrapolated-from-delta22 / scaled-to-test-set /
    scaled-to-solute RMSE. The three sitting close together (except water) is the transferability
    story. Used by SI Figure S15's "Distribution Shift by Solvent" panel."""
    colors = {"Extrapolated from delta22": "#A72608", "Scaled to Test Set": "#C93240",
              "Scaled to Solute": "#61a89a"}
    columns = list(colors.keys())
    x = np.arange(len(table))
    width = 0.26
    fig, ax = plt.subplots(figsize=figsize)
    for i, col in enumerate(columns):
        ax.bar(x + (i - 1) * width, table[col].to_numpy(), width,
               label=col.replace("delta22", "Δ22"), color=colors[col])
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in table.index])
    ax.set_ylabel(f"{'¹H' if nucleus == 'H' else '¹³C'} RMSE (ppm)")
    ax.set_title(f"Distribution Shift by Solvent ({nucleus})")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax

