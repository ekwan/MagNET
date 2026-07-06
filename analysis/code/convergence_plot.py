"""3D accuracy-vs-slope-vs-year convergence panel for Figure 2D.

The notebook supplies the per-method statistics table, the method->rung map, and the rung->color map as
arguments (`results_df`, `methods_categorization`, `method_color_map`) and calls
`plot_statistics_3d_emphasize_rungs(...)`.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)


def plot_statistics_3d_emphasize_rungs(
    x_param="Slope",
    y_param="RMSE (scaled)",
    z_param=None,
    results_df=None,
    methods_categorization=None,
    method_color_map=None,
    figsize=(9, 7),
    elev=12,
    azim=20,

    # =========================
    # SAVING
    # =========================
    save_path=None,
    save_formats=("png",),
    dpi=300,
    svg_transparent=True,
    png_transparent=False,

    # =========================
    # DROPLINES (ALL POINTS)
    # =========================
    droplines_for_all=True,
    dropline_alpha=0.1,
    dropline_lw=0.8,
    dropline_color="0.4",
    dropline_shadow_alpha=0.25,
    dropline_shadow_size=18,

    xlim=None,
    ylim=None,
    zlim=None,

    # =========================
    # YEAR CONTROL
    # =========================
    year_map=None,
    year_default=np.nan,
    year_tick_step=2,
    drop_missing_years=False,

    # =========================
    # Z LANES
    # =========================
    z_lane_by_category=True,
    z_lane_width=0.35,
    z_jitter_by_category=0.0,

    # =========================
    # XY LANES
    # =========================
    xy_lane_by_category=True,
    xy_lane_strength=0.25,
    xy_lane_style="circle",
    lane_exclude_categories=("ab initio",),

    auto_expand_xy_limits=True,
    xy_limit_pad_frac=0.06,

    # =========================
    # BROKEN Z
    # =========================
    z_break=None,
    z_break_gap=2.0,
    drop_years_in_break=False,

    # =========================
    # DECADE GROUPING
    # =========================
    group_by_decade=True,
    decade_bin=10,
    decade_alpha=0.85,
    decade_marker_size=80,
    decade_sort_ascending=True,

    # =========================
    # XY GRID PLANE
    # =========================
    show_xy_grid_plane=True,
    xy_grid_z=0.0,
    grid_alpha=0.15,
    grid_lw=0.4,
    show_plot=True,
    auto_xy_grid_z_if_years=True,

    # =========================
    # BASELINE SHADOW COLOR
    # =========================
    baseline_tint_strength=0.35,
    baseline_gray="#7A7A7A",

    # =========================
    # MAJOR TICKS (CUSTOMIZABLE)
    # =========================
    x_major_nbins=5,
    y_major_nbins=5,
    major_tick_length=0.1,
    major_tick_width=0.1,
    major_tick_labelsize=10,
    major_tick_pad=2,
    major_tick_color="0.2",
    major_tick_labelcolor="0.1",
    major_tick_rotation_x=0,
    major_tick_rotation_y=0,
    major_tick_rotation_z=0,

    # =========================
    # "SPINES"/BORDERS FOR 3D AXES (CUSTOMIZABLE)
    # =========================
    axisline_lw=0.3,
    axisline_color="0.05",

    pane_fill=False,
    pane_alpha=0.0,
    pane_color="white",
    pane_edgecolor="0.15",
    pane_edge_lw=0.3,

    # alias (so passing axis_linewidth won't crash)
    axis_linewidth=None,

    # =========================
    # HIGHLIGHT + LABEL METHODS
    # =========================
    highlight_methods=None,          # list/tuple of method names to highlight
    highlight_marker_size=140,       # size for highlighted markers
    highlight_edgecolor="k",         # outline color
    highlight_edgewidth=1.2,         # outline thickness
    highlight_alpha=1.0,             # alpha for highlighted markers
    highlight_text=True,             # whether to draw labels
    highlight_text_size=8,           # label fontsize
    highlight_text_color="k",        # label color
    highlight_text_dx_frac=0.008,    # x-offset for text (fraction of x-range)
    highlight_text_dy_frac=0.008,    # y-offset for text (fraction of y-range)
    highlight_text_dz=0.0,           # z-offset in z-units (mapped units)
    highlight_text_bbox=True,        # draw a small white box behind text
    highlight_text_bbox_alpha=0.65,  # bbox alpha

    # legend
    show_legend=False,
    legend_title="Rung Type",
    legend_fontsize=8,
    legend_title_fontsize=9,
    legend_loc="center left",
    legend_bbox_to_anchor=(1.02, 0.5),
):
    """Draw the Figure 2D 3D scatter of RMSE vs. slope vs. year, grouped and colored by rung."""
    sns.set_theme(context="paper", style="ticks", font_scale=1.0)

    def _coerce_pair(val, name, allow_none=True):
        if val is None:
            if allow_none:
                return None
            raise ValueError(f"{name} cannot be None")
        if np.isscalar(val):
            raise ValueError(
                f"{name} must be a 2-tuple like ({name}_min, {name}_max), not a scalar: {val}"
            )
        if len(val) != 2:
            raise ValueError(f"{name} must be length-2 like (min, max). Got: {val}")
        return (float(val[0]), float(val[1]))

    # apply alias if user passed axis_linewidth
    if axis_linewidth is not None:
        axisline_lw = float(axis_linewidth)

    # coerce user limits early
    xlim2 = _coerce_pair(xlim, "xlim", allow_none=True)
    ylim2 = _coerce_pair(ylim, "ylim", allow_none=True)
    zlim2 = _coerce_pair(zlim, "zlim", allow_none=True)

    def _draw_xy_grid_plane(ax, xlim, ylim, z=0.0, alpha=0.25, lw=0.8):
        x_ticks = ax.get_xticks()
        y_ticks = ax.get_yticks()
        for x in x_ticks:
            if xlim[0] <= x <= xlim[1]:
                ax.plot([x, x], [ylim[0], ylim[1]], [z, z],
                        color="0.0", alpha=alpha, linewidth=lw, zorder=1)
        for y in y_ticks:
            if ylim[0] <= y <= ylim[1]:
                ax.plot([xlim[0], xlim[1]], [y, y], [z, z],
                        color="0.0", alpha=alpha, linewidth=lw, zorder=1)

    def _blend_hex(c1, c2, t):
        c1 = np.array(mcolors.to_rgb(c1))
        c2 = np.array(mcolors.to_rgb(c2))
        return (1 - t) * c1 + t * c2

    def _make_z_break_mapper(z_break, gap, drop_in_break):
        z_break2 = _coerce_pair(z_break, "z_break", allow_none=True)
        if z_break2 is None:
            return (lambda y: float(y)), (lambda z: float(z))

        lo, hi = z_break2
        if not (hi > lo):
            raise ValueError(f"z_break must be (lo, hi) with hi>lo; got {z_break2}")
        span = hi - lo
        gap = float(gap)

        def fwd(y):
            y = float(y)
            if lo < y < hi:
                if drop_in_break:
                    return None
                y = lo if (y - lo) <= (hi - y) else hi
            if y <= lo:
                return y
            return y - span + gap

        def inv(z):
            z = float(z)
            if z <= lo:
                return z
            return z + span - gap

        return fwd, inv

    def _norm_cat(s: str) -> str:
        s = str(s).lower()
        return "".join(ch for ch in s if ch.isalnum())

    exclude_norm = {_norm_cat(c) for c in (lane_exclude_categories or ())}

    z_map, _ = _make_z_break_mapper(z_break, z_break_gap, drop_years_in_break)

    # --- Prepare dataframe ---
    df = results_df.copy()
    df["category"] = df["Method"].map(methods_categorization).fillna("Unknown")
    df["_cat_norm"] = df["category"].apply(_norm_cat)

    rung_order = [c for c in method_color_map.keys() if c in df["category"].unique()]
    rung_order += [c for c in df["category"].unique() if c not in rung_order]

    # --- Z assignment ---
    if z_param is not None:
        df["_z_raw"] = df[z_param].astype(float)
        df["_year"] = df["_z_raw"].copy()
    else:
        if year_map is None:
            raise ValueError("z_param is None, so you must provide year_map={Method: year}.")
        df["_z_raw"] = df["Method"].map(year_map).fillna(year_default)

        missing_methods = df[df["_z_raw"].isna()]["Method"].unique()
        if len(missing_methods) > 0:
            if drop_missing_years:
                df = df.dropna(subset=["_z_raw"]).copy()
            else:
                examples = missing_methods[:12].tolist()
                raise ValueError(
                    f"Missing years for {len(missing_methods)} methods. "
                    f"Examples: {examples}. "
                    f"Either add them to year_map or set drop_missing_years=True."
                )

        df["_z_raw"] = df["_z_raw"].astype(float)
        df["_year"] = df["_z_raw"].copy()

        if z_lane_by_category:
            cat_index = {cat: i for i, cat in enumerate(rung_order)}
            offsets = df["category"].map(lambda c: cat_index.get(c, 0)).astype(float)
            offsets = offsets - offsets.mean()
            is_excluded = df["_cat_norm"].isin(exclude_norm)
            offsets = offsets.where(~is_excluded, 0.0)
            df["_z_raw"] = df["_z_raw"] + offsets * float(z_lane_width)

        if z_jitter_by_category and z_jitter_by_category != 0.0:
            rng = np.random.default_rng(0)
            df["_z_raw"] = df["_z_raw"] + rng.uniform(-1, 1, size=len(df)) * float(z_jitter_by_category)

    # --- apply z break mapping ---
    if z_break is not None:
        z_mapped = df["_z_raw"].apply(z_map)
        if drop_years_in_break:
            df = df.loc[~z_mapped.isna()].copy()
            z_mapped = df["_z_raw"].apply(z_map)
        df["_z"] = z_mapped.astype(float)
    else:
        df["_z"] = df["_z_raw"].astype(float)

    # --- Base limits from ORIGINAL (unshifted) data ---
    x_min, x_max = df[x_param].min(), df[x_param].max()
    y_min, y_max = df[y_param].min(), df[y_param].max()
    x_pad = 0.15 * (x_max - x_min) if x_max > x_min else 0.5
    y_pad = 0.15 * (y_max - y_min) if y_max > y_min else 0.5
    xm_default = (x_min - x_pad, x_max + x_pad)
    ym_default = (y_min - y_pad, y_max + y_pad)

    # Use provided limits (or defaults) to define lane sizes
    x0l, x1l = (xm_default if xlim2 is None else xlim2)
    y0l, y1l = (ym_default if ylim2 is None else ylim2)
    xr = (x1l - x0l) if (x1l - x0l) != 0 else 1.0
    yr = (y1l - y0l) if (y1l - y0l) != 0 else 1.0

    # --- XY category lanes ---
    if xy_lane_by_category:
        n = max(1, len(rung_order))
        cat_index = {cat: i for i, cat in enumerate(rung_order)}

        if xy_lane_style.lower() == "grid":
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))

            def grid_xy(i):
                r = i // cols
                c = i % cols
                cx = c - (cols - 1) / 2
                cy = r - (rows - 1) / 2
                return cx, cy

            coords = {cat: grid_xy(cat_index[cat]) for cat in rung_order}
        else:
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            coords = {cat: (np.cos(angles[cat_index[cat]]), np.sin(angles[cat_index[cat]])) for cat in rung_order}

        dx = float(xy_lane_strength) * xr
        dy = float(xy_lane_strength) * yr
        is_excluded = df["_cat_norm"].isin(exclude_norm)

        def _dx(c): return coords.get(c, (0.0, 0.0))[0] * dx
        def _dy(c): return coords.get(c, (0.0, 0.0))[1] * dy

        df["_x_plot"] = df[x_param].astype(float) + np.where(is_excluded, 0.0, df["category"].map(_dx).astype(float))
        df["_y_plot"] = df[y_param].astype(float) + np.where(is_excluded, 0.0, df["category"].map(_dy).astype(float))
    else:
        df["_x_plot"] = df[x_param].astype(float)
        df["_y_plot"] = df[y_param].astype(float)

    # Expand axis limits to include shifted points
    if auto_expand_xy_limits:
        xmn, xmx = float(np.nanmin(df["_x_plot"])), float(np.nanmax(df["_x_plot"]))
        ymn, ymx = float(np.nanmin(df["_y_plot"])), float(np.nanmax(df["_y_plot"]))
        xr2 = (xmx - xmn) if (xmx - xmn) != 0 else 1.0
        yr2 = (ymx - ymn) if (ymx - ymn) != 0 else 1.0
        xpad2 = xy_limit_pad_frac * xr2
        ypad2 = xy_limit_pad_frac * yr2
        xm_shifted = (xmn - xpad2, xmx + xpad2)
        ym_shifted = (ymn - ypad2, ymx + ypad2)
    else:
        xm_shifted = xm_default
        ym_shifted = ym_default

    # baseline plane fix
    if zlim2 is not None:
        z_base = z_map(zlim2[0]) if (z_break is not None) else float(zlim2[0])
    else:
        z_base = float(np.nanmin(df["_z"]))
    Z_BASELINE = z_base
    if auto_xy_grid_z_if_years and (z_param is None):
        xy_grid_z = Z_BASELINE

    # decade grouping
    if group_by_decade and (z_param is None):
        df["_decade"] = (np.floor(df["_year"] / decade_bin) * decade_bin).astype(int)
        decade_order = sorted(df["_decade"].unique(), reverse=not decade_sort_ascending)
    else:
        df["_decade"] = np.nan
        decade_order = []

    # --- Figure ---
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1.5, 2, 1.5))

    # axis lines
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.line.set_linewidth(axisline_lw)
            axis.line.set_color(axisline_color)
        except Exception:
            pass

    # panes
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.set_edgecolor(pane_edgecolor)
            axis.pane.set_linewidth(pane_edge_lw)
            if pane_fill:
                axis.pane.set_facecolor(mcolors.to_rgba(pane_color, pane_alpha))
            else:
                axis.pane.set_facecolor((1, 1, 1, 0))
        except Exception:
            pass

    # --- SCATTER ---
    if group_by_decade and (z_param is None):
        for d in decade_order:
            sub_d = df[df["_decade"] == d]
            if sub_d.empty:
                continue
            for cat in rung_order:
                sub = sub_d[sub_d["category"] == cat]
                if sub.empty:
                    continue
                ax.scatter(
                    sub["_x_plot"], sub["_y_plot"], sub["_z"],
                    s=decade_marker_size,
                    depthshade=False,
                    color=method_color_map.get(cat, "#777777"),
                    edgecolor="none",
                    alpha=decade_alpha,
                    zorder=5,
                )
    else:
        for cat in rung_order:
            sub = df[df["category"] == cat]
            if sub.empty:
                continue
            ax.scatter(
                sub["_x_plot"], sub["_y_plot"], sub["_z"],
                s=80,
                depthshade=False,
                color=method_color_map.get(cat, "#777777"),
                edgecolor="none",
                alpha=0.85,
                zorder=5,
            )

    # --- DROPLINES FOR ALL POINTS ---
    if droplines_for_all:
        cat_cols = df["category"].map(lambda c: method_color_map.get(c, "#777777"))
        shadow_cols = [_blend_hex(baseline_gray, c, baseline_tint_strength) for c in cat_cols]
        for (x0, y0, z0, shcol) in zip(df["_x_plot"].to_numpy(float),
                                      df["_y_plot"].to_numpy(float),
                                      df["_z"].to_numpy(float),
                                      shadow_cols):
            ax.plot([x0, x0], [y0, y0], [z0, Z_BASELINE],
                    color=dropline_color, alpha=dropline_alpha,
                    linewidth=dropline_lw, zorder=2)
            ax.scatter([x0], [y0], [Z_BASELINE],
                       s=dropline_shadow_size,
                       color=shcol,
                       alpha=dropline_shadow_alpha,
                       depthshade=False,
                       zorder=2)

    # --- limits ---
    ax.set_xlim(xm_shifted if xlim2 is None else xlim2)
    ax.set_ylim(ym_shifted if ylim2 is None else ylim2)
    if zlim2 is not None:
        ax.set_zlim(z_map(zlim2[0]), z_map(zlim2[1])) if (z_break is not None) else ax.set_zlim(zlim2)
    else:
        zmin = float(np.nanmin(df["_z"]))
        zmax = float(np.nanmax(df["_z"]))
        ax.set_zlim(zmin - 0.5, zmax + 0.5)

    ax.set_ylabel("Scaled RMSE (" + r"$^{1}\mathrm{H}$" + " ppm)", fontsize=13, labelpad=10)
    ax.set_xlabel("Slope vs. CCSD(T)", fontsize=13, labelpad=10)
    ax.set_zlabel("Year", fontsize=13, labelpad=8)

    # --- major tick locations ---
    ax.xaxis.set_major_locator(MaxNLocator(nbins=x_major_nbins))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=y_major_nbins))

    # --- major tick appearance ---
    for axis_name in ("x", "y", "z"):
        ax.tick_params(
            axis=axis_name,
            which="major",
            length=major_tick_length,
            width=major_tick_width,
            pad=major_tick_pad,
            colors=major_tick_color,
            labelsize=major_tick_labelsize,
            labelcolor=major_tick_labelcolor,
        )

    for lbl in ax.get_xticklabels():
        lbl.set_rotation(major_tick_rotation_x)
    for lbl in ax.get_yticklabels():
        lbl.set_rotation(major_tick_rotation_y)
    for lbl in ax.get_zticklabels():
        lbl.set_rotation(major_tick_rotation_z)

    # --- Z ticks as years ---
    if (z_param is None) or (z_break is not None):
        if zlim2 is not None:
            real_min, real_max = float(zlim2[0]), float(zlim2[1])
        else:
            real_min, real_max = float(np.nanmin(df["_year"])), float(np.nanmax(df["_year"]))

        if z_break is not None:
            lo, hi = map(float, _coerce_pair(z_break, "z_break", allow_none=False))
            step = int(year_tick_step)
            ticks_real = list(np.arange(np.floor(real_min), np.floor(min(real_max, lo)) + 1, step))
            ticks_real += list(np.arange(np.ceil(max(real_min, hi)), np.ceil(real_max) + 1, step))
            ticks_real = [int(t) for t in ticks_real]
            ticks_pos = [z_map(t) for t in ticks_real]
            ax.set_zticks(ticks_pos)
            ax.set_zticklabels([str(t) for t in ticks_real], fontsize=major_tick_labelsize, color=major_tick_labelcolor)
        else:
            ticks = np.arange(int(np.floor(real_min)), int(np.ceil(real_max)) + 1, int(year_tick_step))
            ax.set_zticks(ticks)
            ax.set_zticklabels([str(t) for t in ticks], fontsize=major_tick_labelsize, color=major_tick_labelcolor)

    ax.grid(False)

    if show_xy_grid_plane:
        _draw_xy_grid_plane(ax, xlim=ax.get_xlim(), ylim=ax.get_ylim(), z=xy_grid_z,
                            alpha=grid_alpha, lw=grid_lw)

    # =========================
    # HIGHLIGHT + LABEL
    # =========================
    if highlight_methods:
        hm = set(highlight_methods)
        sub_h = df[df["Method"].isin(hm)].copy()
        if not sub_h.empty:
            ax.scatter(
                sub_h["_x_plot"], sub_h["_y_plot"], sub_h["_z"],
                s=highlight_marker_size,
                depthshade=False,
                color=sub_h["category"].map(lambda c: method_color_map.get(c, "#777777")),
                edgecolor=highlight_edgecolor,
                linewidths=highlight_edgewidth,
                alpha=highlight_alpha,
                zorder=20,
            )

            if highlight_text:
                dx_text = float(highlight_text_dx_frac) * xr
                dy_text = float(highlight_text_dy_frac) * yr
                for _, r in sub_h.iterrows():
                    px, py, pz = float(r["_x_plot"]), float(r["_y_plot"]), float(r["_z"])
                    tx, ty, tz = px + dx_text, py + dy_text, pz + float(highlight_text_dz)
                    # thin leader line so the label reads in clear space, not under the marker
                    ax.plot([px, tx], [py, ty], [pz, tz], color="0.45", linewidth=0.6, zorder=25)
                    txt_kwargs = dict(
                        fontsize=highlight_text_size,
                        color=highlight_text_color,
                        zorder=30, ha="left", va="center",
                    )
                    if highlight_text_bbox:
                        txt_kwargs["bbox"] = dict(
                            boxstyle="round,pad=0.15",
                            facecolor=(1, 1, 1, highlight_text_bbox_alpha),
                            edgecolor="none",
                        )
                    ax.text(tx, ty, tz, str(r["Method"]), **txt_kwargs)

    if show_legend:
        import matplotlib.patches as mpatches
        patches = [mpatches.Patch(color=method_color_map.get(cat, "#777777"), label=cat) for cat in rung_order]
        ax.legend(handles=patches,
                  title=legend_title,
                  fontsize=legend_fontsize,
                  title_fontsize=legend_title_fontsize,
                  loc=legend_loc,
                  bbox_to_anchor=legend_bbox_to_anchor,
                  frameon=True)

    # =========================
    # SAVE
    # =========================
    def _save_figure(fig, save_path, formats=("svg", "png"), dpi=300,
                     svg_transparent=True, png_transparent=False):
        if not save_path:
            return []
        root, ext = os.path.splitext(save_path)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        saved = []

        def _save_one(outpath, fmt):
            fmt = fmt.lower().lstrip(".")
            if fmt == "svg":
                fig.savefig(outpath, format="svg", bbox_inches="tight", pad_inches=0.12,
                            dpi=dpi, transparent=bool(svg_transparent))
            elif fmt in {"png", "jpg", "jpeg", "tif", "tiff"}:
                fig.savefig(outpath, format=fmt, bbox_inches="tight", pad_inches=0.12,
                            dpi=dpi, transparent=bool(png_transparent))
            else:
                fig.savefig(outpath, format=fmt, bbox_inches="tight", pad_inches=0.12,
                            dpi=dpi, transparent=False)

        if ext.lower() in {".png", ".svg", ".pdf", ".jpg", ".jpeg", ".tif", ".tiff"}:
            fmt = ext.lower().lstrip(".")
            _save_one(save_path, fmt)
            saved.append(save_path)
            return saved

        for f in formats:
            f = f.lower().lstrip(".")
            out = f"{save_path}.{f}"
            _save_one(out, f)
            saved.append(out)
        return saved

    saved_files = _save_figure(
        fig,
        save_path=save_path,
        formats=save_formats,
        dpi=dpi,
        svg_transparent=svg_transparent,
        png_transparent=png_transparent,
    )
    if saved_files:
        print("Saved:", *saved_files, sep="\n  - ")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax, saved_files
