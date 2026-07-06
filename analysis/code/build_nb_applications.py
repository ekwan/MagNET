# build_nb_applications.py -- generator for the natural-products figure notebooks.
#
# Source of truth for the notebooks: edit the cell sources below, then regenerate the notebook(s) you
# touched (regenerating overwrites the .ipynb, so edit this script, not the notebook):
#
#     python3 build_nb_applications.py si_figure_s08_vomicine
#
# Run with no arguments to list the notebook names; the NOTEBOOKS dict maps each to its destination.
# The numbers come from applications.py; the notebooks arrange and plot them.
from nb_build import md, code, save_notebook as _save


# ----------------------------------------------------------------------------
# SI Figure S8, panel A: explicit-solvent shell-size convergence for vomicine
# ----------------------------------------------------------------------------
si_figure_s08 = [
    md(r"""
# SI Figure S8 (panel A): explicit-solvent correction vs solvent-shell size (vomicine)

The MagNET-x explicit-solvent correction (solvated minus isolated, averaged over a proton site's
atoms and MD frames) vs solvent-shell size (heavy-atom count of the solute-plus-solvent cluster),
for vomicine, one panel per solvent.
"""),
    code(r"""
import os, sys

# make the in-repo modules importable (not pip-installed)
REPO = os.path.abspath("../..")
for _p in ("data/applications", "analysis/code", "analysis/code/shared"):
    sys.path.insert(0, os.path.join(REPO, _p))
"""),
    code(r"""
import matplotlib.pyplot as plt

from applications_reader import Applications, SHELL_SIZES
import applications
import paths
"""),
    code(r"""
DATA = os.path.join(REPO, "data", "applications")

def figure_path(name):
    os.makedirs("figures", exist_ok=True)
    return os.path.join("figures", name)
"""),
    code(r"""
loader = Applications(paths.dataset_file("applications", root=REPO),
                      os.path.join(DATA, "applications_experimental.xlsx"))

# compute the per-site, per-solvent shell-convergence corrections for vomicine (proton sites)
SOLUTE, NUCLEUS = "vomicine", "H"
corrections = applications.shell_convergence_corrections(loader, SOLUTE, NUCLEUS)
corrections.head()
"""),
    code(r"""
# panel layout and font styling for the 2x2 grid (one panel per solvent)
solvent_order = ["chloroform", "methanol", "TIP4P", "benzene"]
shell_cols = [f"shell_{s}" for s in SHELL_SIZES]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Arial", "DejaVu Serif"],
                     "axes.labelsize": 12, "axes.titlesize": 14})
"""),
    code(r"""
# 2x2 panel, one solvent each; one line per proton site
fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
for ax, solvent in zip(axes.ravel(), solvent_order):
    sub = corrections.xs(solvent, level="solvent")
    sub = sub.dropna(how="all")
    ax.set_title("TIP4P" if solvent == "TIP4P" else solvent.capitalize(), fontweight="bold")
    for site, row in sub.iterrows():
        ax.plot(SHELL_SIZES, row[shell_cols].values.astype(float),
                marker="o", linewidth=1.5, alpha=0.75, label=site)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Shell Size (Heavy Atom Count)", fontweight="bold")
    ax.set_ylabel("OpenMM correction (ppm)", fontweight="bold")
fig.suptitle("Shell-size Convergence for Vomicine", fontweight="bold", fontsize=18)
fig.tight_layout()
fig.savefig(figure_path("si_figure_s08_vomicine.png"), dpi=300, bbox_inches="tight")
plt.show()
"""),
]

# ----------------------------------------------------------------------------
# SI Figure S15: transferability of composite-model coefficients to the test set (1H + 13C)
# ----------------------------------------------------------------------------
si_figure_s15 = [
    md(r"""
# SI Figure S15: per-solute RMSE with delta-22 composite coefficients applied

Per-solute ¹H and ¹³C RMSE for the natural-products / olefin-isomer test set with delta-22
coefficients applied (reproduces Figure 5C in ¹H, plus the ¹³C analogue), plus fitting-RMSE
comparisons across coefficient choices, feature-space coverage by solvent, delta-22-plane residuals,
and the RMSE distribution shift by solvent.
"""),
    code(r"""
import os, sys

# make the in-repo modules importable (not pip-installed)
REPO = os.path.abspath("../..")
for _p in ("data/applications", "data/delta22", "analysis/code", "analysis/code/shared"):
    sys.path.insert(0, os.path.join(REPO, _p))
"""),
    code(r"""
import matplotlib.pyplot as plt

from applications_reader import Applications
import applications
import applications_plots
import delta22
import paths
"""),
    code(r"""
DATA = os.path.join(REPO, "data", "applications")
DELTA22_HDF5 = paths.dataset_file("delta22", root=REPO)
DELTA22_XLSX = os.path.join(REPO, "data", "delta22", "delta22_experimental.xlsx")

def figure_path(name):
    os.makedirs("figures", exist_ok=True)
    return os.path.join("figures", name)
"""),
    code(r"""
loader = Applications(paths.dataset_file("applications", root=REPO),
                      os.path.join(DATA, "applications_experimental.xlsx"))

# assemble the feature table and run the composite-model fits + bootstrap
query_df_nn = applications.build_query_df_nn(loader)
site_counts = (query_df_nn.drop_duplicates(subset=["solute", "nucleus", "site"])
               .groupby(["solute", "nucleus"]).size().unstack(fill_value=0))

per_solute = applications.per_solute_fits(query_df_nn)      # "scaled to solute" baseline
all_solute = applications.per_solvent_fits(query_df_nn)     # "scaled to test set" full fit

seed = applications.build_bootstrap_seed_coeffs(loader)
bootstrap_rmses = {}
for nuc in ["H", "C"]:
    preds = applications.apply_bootstrap_params_to_full_dataset(query_df_nn, seed[nuc], nucleus=nuc)
    bootstrap_rmses[nuc] = applications.compute_solute_rmses(preds)
"""),
    code(r"""
# published solute order (isomers, then the two pyridines, then the natural products); matches
# main-text Figure 5D. delta-22 is drawn first by the box-plot engine, so it is not listed here.
S15_SOLUTE_ORDER = ["isomer_1E", "isomer_1Z", "isomer_2E", "isomer_2Z", "isomer_3E", "isomer_3Z",
                    "isomer_4N", "isomer_4O", "vomicine", "prednisone", "peptide", "flavone",
                    "dihydrotanshinone_I"]
"""),
    code(r"""
# 1H panel (reproduces Figure 5C)
applications_plots.plot_nps_on_boxplot_delta22_simplified(
    loader.rmse_distribution("H"), bootstrap_rmses["H"], per_solute["H"],
    nucleus="H", formulas=["stationary_plus_qcd + openMM"], colors=["#61a89a"],
    site_counts=site_counts, formula_remap=applications.FORMULA_REMAP,
    solute_remap=applications.SOLUTE_DISPLAY, solute_order=S15_SOLUTE_ORDER,
    solute_color_remap=applications.PEPTIDE_HIGHLIGHT_H,
    figsize=(14, 8), box_width=0.20, box_gap=0.1,
    show_baseline=True, baseline_annotation_text="Scaled to solute",
    baseline_annotation_x=0.215, baseline_annotation_y=0.39,
    show_full_fit_line=True, full_fit_line_label="Scaled to Test Set", full_fit_line_label_x=0.835,
    all_solute_fitting_results=all_solute,
    max_bar_height=0.06, site_count_axis_mode="inset", site_count_inset_area_frac=0.1,
    site_count_inset_axis_offset=-0.06, site_count_inset_axis_label_pad=0.045,
    save_path=figure_path("si_figure_s15_1H.png"),
)
"""),
    code(r"""
# 13C panel
applications_plots.plot_nps_on_boxplot_delta22_simplified(
    loader.rmse_distribution("C"), bootstrap_rmses["C"], per_solute["C"],
    nucleus="C", formulas=["stationary_plus_op_vib + openMM"], colors=["#61a89a"],
    site_counts=site_counts, formula_remap=applications.FORMULA_REMAP,
    solute_remap=applications.SOLUTE_DISPLAY, solute_order=S15_SOLUTE_ORDER,
    solute_color_remap=applications.PEPTIDE_HIGHLIGHT_C,
    figsize=(14, 8), box_width=0.20, box_gap=0.1,
    show_baseline=True, baseline_annotation_text="Scaled to solute",
    baseline_annotation_x=0.28, baseline_annotation_y=0.355,
    show_full_fit_line=True, full_fit_line_label="Scaled to Test Set", full_fit_line_label_x=0.98,
    all_solute_fitting_results=all_solute,
    max_bar_height=0.8, site_count_axis_mode="inset", site_count_inset_area_frac=0.1,
    site_count_inset_axis_offset=-0.06, site_count_inset_axis_label_pad=0.045,
    save_path=figure_path("si_figure_s15_13C.png"),
)
"""),
    md(r"""
## Fitting RMSE Comparisons (chloroform)

Per-solute RMSE under three coefficient choices (Scaled to Solute / Scaled to Test Set / Extrapolated
from delta22).
"""),
    code(r"""
for nucleus, formula in [("H", "stationary_plus_qcd + openMM"), ("C", "stationary_plus_op_vib + openMM")]:
    table = applications.fitting_rmse_comparison_table(query_df_nn, per_solute, all_solute, bootstrap_rmses[nucleus],
                                            nucleus, "chloroform", formula)
    table = table.rename(index=applications.SOLUTE_DISPLAY)
    order = [applications.SOLUTE_DISPLAY.get(k, k) for k in S15_SOLUTE_ORDER]   # published order; peptide has no chloroform data
    table = table.reindex([n for n in order if n in table.index])
    print(f"--- {nucleus} ---")
    display(table.round(3))
    applications_plots.plot_fitting_rmse_comparison_bars(
        table, nucleus, "chloroform",
        save_path=figure_path(f"si_figure_s15_fitting_rmse_comparisons_{'1H' if nucleus == 'H' else '13C'}.png"))
plt.show()
"""),
    md(r"""
## Feature Space Coverage by Solvent

Test-set vs delta-22 mean-centered feature values across the four explicit-solvent solvents.
"""),
    code(r"""
delta22_query_df_nn = delta22.add_composite_columns(delta22.load_query_df_nn(
    DELTA22_HDF5, DELTA22_XLSX, verbose=False))
"""),
    code(r"""
_FEATURE_X_LABELS = {
    "H": "Gas-Phase Shielding + QCD Correction (centered, ppm)",
    "C": "Gas-Phase Shielding + OpenMM Vibrational Correction (centered, ppm)",
}
for nucleus in ["H", "C"]:
    coverage = applications.feature_space_coverage_table(query_df_nn, delta22_query_df_nn, nucleus)
    applications_plots.plot_feature_space_coverage_grid(
        coverage, nucleus, _FEATURE_X_LABELS[nucleus],
        save_path=figure_path(f"si_figure_s15_feature_space_coverage_{'1H' if nucleus == 'H' else '13C'}.png"))
plt.show()
"""),
    md(r"""
## Residuals for Delta22 Fitting Coefficients

Residuals of a delta-22-only 2-feature plane applied to both delta-22 and the test set, vs
experimental shielding.
"""),
    code(r"""
for nucleus in ["H", "C"]:
    residuals = applications.delta22_plane_residuals_table(query_df_nn, delta22_query_df_nn, nucleus)
    applications_plots.plot_delta22_plane_residuals_grid(
        residuals, nucleus,
        save_path=figure_path(f"si_figure_s15_delta22_plane_residuals_{'1H' if nucleus == 'H' else '13C'}.png"))
plt.show()
"""),
    md(r"""
## Distribution Shift by Solvent

Per-solvent mean test-set RMSE under the three coefficient choices.
"""),
    code(r"""
for nucleus, formula in [("H", "stationary_plus_qcd + openMM"), ("C", "stationary_plus_op_vib + openMM")]:
    shift = applications.distribution_shift_by_solvent_table(per_solute, all_solute, bootstrap_rmses[nucleus],
                                                  nucleus, formula)
    print(f"--- {nucleus} ---")
    display(shift.round(3))
    applications_plots.plot_distribution_shift_by_solvent_bars(
        shift, nucleus,
        save_path=figure_path(f"si_figure_s15_distribution_shift_by_solvent_{'1H' if nucleus == 'H' else '13C'}.png"))
plt.show()
"""),
]

# NOTE: main-text Figure 5 panels C and D (analysis/manuscript_figures/fig5c.ipynb, fig5d.ipynb) are
# HAND-AUTHORED, not generated here. Their plotting engines (plot_nps_benefit_barplot for 5C,
# plot_nps_on_boxplot_delta22_simplified for 5D, also used by SI S15) live in applications_plots.py;
# the notebooks only load data and call them. The SI notebooks above are still generated here.


NOTEBOOKS = {
    "si_figure_s08_vomicine": (si_figure_s08, "analysis/si_figures/si_figure_s08_panelA_vomicine.ipynb"),
    "si_figure_s15": (si_figure_s15, "analysis/si_figures/si_figure_s15.ipynb"),
}

if __name__ == "__main__":
    import os
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    names = sys.argv[1:]
    if not names:
        print("usage: python3 build_nb_applications.py <name> [<name> ...]")
        print("regenerates ONLY the named notebook(s) -- pick just the one(s) you edited,")
        print("since regenerating clears a notebook's execution outputs.")
        print("available names:", ", ".join(NOTEBOOKS))
        raise SystemExit(1)
    unknown = [n for n in names if n not in NOTEBOOKS]
    if unknown:
        raise SystemExit(f"unknown notebook name(s): {unknown}; available: {', '.join(NOTEBOOKS)}")
    for name in names:
        cells, relpath = NOTEBOOKS[name]
        _save(cells, os.path.join(repo, relpath))
