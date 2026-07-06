"""Correlation-matrix and PCA-loadings plotting engine for the leveling-effect study (SI Figure S3).

Draws the two figure types built by build_nb_leveling.py: per-nucleus inter-method Pearson
correlation matrices (log-scaled, with a scaled-RMSE side bar and a colour-scale histogram inset)
and PC1/PC2 method-loading panels (with a fitted parabola and adjustText-repelled labels). The
notebook supplies the family-colour map, nucleus display labels, and adjustText tuning as arguments.
"""
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text

from leveling import log_r, parabola_curve


def _adjust_pca_labels(texts, ax, nuc, pca_adjust, pca_adjust_default):
    params = pca_adjust.get(nuc, pca_adjust_default)
    adjust_text(texts, ax=ax, max_move=None, min_arrow_len=0,
                ensure_inside_axes=True,
                arrowprops=dict(arrowstyle='-', color='0.55', lw=0.35),
                **params)


def _family_legend(ax, families, family_colors, include_ref=True):
    present = []
    for f in families:
        if f == 'ref' and not include_ref:
            continue
        if f not in present:
            present.append(f)
    handles = [plt.Line2D([], [], marker='*' if f=='ref' else 'o', ls='',
                          mfc=family_colors.get(f,'#333333'), mec='k',
                          ms=12 if f=='ref' else 6.5,
                          label='CCSD(T)' if f=='ref' else f)
               for f in present]
    ax.legend(handles=handles, loc='best', fontsize=6.5, ncol=2,
              frameon=True, framealpha=0.9)


def _pca_panel(ax, name, nuc, r, families, family_colors, nuc_display, pca_adjust, pca_adjust_default):
    pc1, pc2 = r['pc1'], r['pc2']
    methods = r['methods']
    cx, cy = parabola_curve(r['par'])
    # small padding: the dense central cluster fills more of the plot in inches,
    # giving adjustText more room per label
    xpad = 0.08*(pc1.max()-pc1.min()+1e-9)
    ypad = 0.08*(pc2.max()-pc2.min()+1e-9)
    xlim = (pc1.min()-xpad, pc1.max()+xpad)
    ylim = (pc2.min()-ypad, pc2.max()+ypad)
    m = (cx>=xlim[0])&(cx<=xlim[1])&(cy>=ylim[0])&(cy<=ylim[1])
    ax.plot(cx[m], cy[m], color='0.75', ls='--', lw=1.2, alpha=0.7, zorder=1)
    texts = []
    for i, lab in enumerate(methods):
        fam = families[i]
        if fam == 'ref':
            ax.scatter(pc1[i], pc2[i], s=75, marker='*',
                       c=family_colors['ref'], edgecolors='white',
                       linewidths=0.8, zorder=5)
        else:
            ax.scatter(pc1[i], pc2[i], s=11,
                       c=family_colors.get(fam, '#333333'),
                       edgecolors='black', linewidths=0.3, zorder=4)
        texts.append(ax.text(pc1[i], pc2[i], lab, fontsize=3.8, zorder=6))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    _adjust_pca_labels(texts, ax, nuc, pca_adjust, pca_adjust_default)
    _family_legend(ax, families, family_colors, include_ref=True)
    ev = r['ev']
    ax.set_title(f'{name}  {nuc_display.get(nuc, nuc)} - '
                 f'method loadings in PC1/PC2 space\n'
                 f'PC1 = {ev[0]*100:.2f}%,   PC2 = {ev[1]*100:.2f}%,   '
                 f'parabola R² = {r["par"]["r2"]:.3f}', fontsize=9)
    ax.set_xlabel('PC1 loading', fontsize=8)
    ax.set_ylabel('PC2 loading', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.axhline(0, color='gray', lw=0.4); ax.axvline(0, color='gray', lw=0.4)
    ax.grid(True, ls=':', alpha=0.3)


def plot_pca_pair(name, nucs, res, fam_lookup, family_colors, nuc_display, pca_adjust, pca_adjust_default):
    """PC1/PC2 method-loading panels, one nucleus per row, stacked in a single figure."""
    n = len(nucs)
    fig, axes = plt.subplots(n, 1, figsize=(7.0, 5.2*n), squeeze=False)
    for k, nuc in enumerate(nucs):
        _pca_panel(axes[k][0], name, nuc, res[nuc], fam_lookup[nuc],
                  family_colors, nuc_display, pca_adjust, pca_adjust_default)
    fig.tight_layout(h_pad=3.6)
    return fig


def plot_corr_matrix(name, nucs, res, fam_lookup, vmax, ref_name, nuc_display):
    """Per-nucleus inter-method correlation heatmap with a scaled-RMSE side bar.

    Layout: [ RMSE bar (viridis, bars grow leftward) | heatmap (inferno) ]. Methods are reordered so
    the reference is first and the rest are sorted by ascending scaled RMSE vs the reference. The
    heatmap is the lower triangle with the diagonal removed; its upper-right corner carries an inset
    histogram of all pairwise log-correlations with the colour bar below it, showing two tick sets
    (bottom = -log10(1-|r|) integers, top = the equivalent raw r values). All matrices share `vmax`.
    """
    nm = len(res[nucs[0]]['methods'])
    m  = nm - 1
    hm_h, hm_w, bar_w = 0.135*nm + 1.9, 6.0, 1.5
    fig_w = bar_w + hm_w
    fig_h = hm_h * len(nucs) + 0.4
    fig, axes = plt.subplots(len(nucs), 2, sharey='row',
                             gridspec_kw={'width_ratios': [bar_w, hm_w]},
                             figsize=(fig_w, fig_h),
                             layout='constrained', squeeze=False)
    fig.get_layout_engine().set(wspace=0.01)        # tighten gap to y-labels
    cmap_corr = plt.cm.inferno.copy(); cmap_corr.set_bad('white')
    cmap_rmse = plt.cm.viridis
    fs = 4.3 if m > 24 else 6.0
    ticks_log = [t for t in [1, 2, 3, 4, 5] if t <= vmax + 0.01]
    for k, nuc in enumerate(nucs):
        axR = axes[k, 0]; ax = axes[k, 1]
        r = res[nuc]; R = r['R']; methods = r['methods']
        rmse_dict = r['scaled_rmse']
        # ----- reorder: reference first, others by ascending RMSE vs ref -----
        ref_idx = methods.index(ref_name)
        others  = [i for i in range(nm) if i != ref_idx]
        others_sorted = sorted(others, key=lambda i: rmse_dict[methods[i]])
        new_order = [ref_idx] + others_sorted
        methods_s = [methods[i] for i in new_order]
        R_s = R[np.ix_(new_order, new_order)]
        rmse_s = np.array([rmse_dict[mth] for mth in methods_s])
        # ----- heatmap (drop row 0 = ref, col -1 = worst-RMSE method) -----
        sub  = R_s[1:, :-1]
        hide = np.triu(np.ones((m, m), dtype=bool), k=1)
        lr   = log_r(sub)
        im = ax.imshow(np.ma.masked_array(lr, hide), cmap=cmap_corr,
                       vmin=0, vmax=vmax, aspect='auto')
        ax.set_xticks(range(m))
        ax.set_xticklabels(methods_s[:-1], rotation=90, fontsize=fs)
        ax.tick_params(left=False, labelleft=False)
        if m <= 20:
            for i in range(m):
                for j in range(i+1):
                    ax.text(j, i, f'{sub[i,j]:.5f}', ha='center', va='center',
                            fontsize=3.4,
                            color='white' if lr[i,j] < vmax*0.5 else 'black')
        ax.set_title(f'{name}  {nuc_display.get(nuc, nuc)} - '
                     f'inter-method Pearson correlation', fontsize=9)
        # ----- histogram + colour bar inset, jammed into the upper-right -----
        # NS372 (kaupp): scale the inset up 20% so it fills more of the corner.
        # A small visible gap is left between hist bottom and cb top so the
        # histogram's tick marks are visible; the dual stacked tick labels
        # under the cb (top line = -log10(1-|r|), bottom line = Pearson r)
        # serve both axes since hist and cb share x-extent.
        all_lr = lr[~hide]
        counts, edges = np.histogram(all_lr, bins=28, range=(0.0, vmax))
        centers = 0.5*(edges[:-1] + edges[1:])
        scale  = 1.2 if name == 'NS372' else 1.0
        hist_w = 0.50 * scale
        hist_h = 0.18 * scale
        cb_h   = 0.014                  # thinner colour bar
        gap    = 0.012                  # visible gap for hist tick marks
        hist_top = 0.97
        x0 = 0.98 - hist_w
        hist_y0 = hist_top - hist_h
        cb_y0   = hist_y0 - gap - cb_h
        hax = ax.inset_axes([x0, hist_y0, hist_w, hist_h])
        hax.set_facecolor('white')
        hax.bar(centers, counts, width=(edges[1]-edges[0])*0.92,
                color=cmap_corr(centers / vmax), edgecolor='black', linewidth=0.3)
        hax.axvline(float(np.median(all_lr)), color='black', lw=1.6, ls='--')
        hax.set_xlim(0, vmax)
        hax.set_xticks(ticks_log)
        hax.tick_params(axis='x', labelbottom=False, bottom=True, length=3)
        hax.set_ylabel('count', fontsize=6, labelpad=1)
        hax.tick_params(axis='y', labelsize=5)
        hax.grid(axis='y', ls=':', alpha=0.4)
        # colour bar, slightly below the histogram with a small visible gap
        cbax = ax.inset_axes([x0, cb_y0, hist_w, cb_h])
        cb = fig.colorbar(im, cax=cbax, orientation='horizontal')
        cb.set_ticks(ticks_log)
        cb.ax.set_xticklabels([f'{t}\n0.{"9"*t}' for t in ticks_log])
        cb.ax.tick_params(axis='x', labelsize=5.5, pad=1,
                          top=False, labeltop=False)
        cb.set_label('-log10(1-|r|)  /  Pearson r', fontsize=6, labelpad=2)
        # ----- RMSE bar on the LEFT (viridis, bars grow leftward) -----
        row_methods = methods_s[1:]
        rmse_row = rmse_s[1:]
        rmse_max = max(rmse_row.max(), 1e-6)
        bar_colors = cmap_rmse(rmse_row / rmse_max)
        axR.barh(range(m), rmse_row, color=bar_colors,
                 edgecolor='black', linewidth=0.3)
        axR.yaxis.set_ticks_position('right')
        axR.yaxis.set_label_position('right')
        axR.set_yticks(range(m))
        axR.set_yticklabels(row_methods, fontsize=fs)
        axR.set_xlim(rmse_max * 1.06, 0)                # reversed: 0 on the right
        axR.set_xlabel(f'scaled RMSE (ppm)\nvs {ref_name}', fontsize=6.5)
        axR.tick_params(axis='x', labelsize=5.5)
        axR.tick_params(axis='y', length=0, pad=2)
        axR.grid(axis='x', ls=':', alpha=0.4)
        axR.set_ylim(m - 0.5, -0.5)                     # imshow has inverted y
    return fig
