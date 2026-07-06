# build_nb_delta22.py -- generator for the delta-22 figure notebooks.
#
# This script is the source of truth for the notebooks. Edit the cell sources here, then
# regenerate and re-execute ONLY the notebook(s) you touched, e.g.:
#
#     python3 build_nb_delta22.py si_figure_s04
#     jupyter nbconvert --to notebook --execute --inplace ../si_figures/si_figure_s04.ipynb
#
# Regenerating overwrites the .ipynb; do not hand-edit it. All real code lives in delta22.py
# (numbers); plotting is inline in each notebook.
#
# Run with no arguments to list the available notebook names. Regenerating a notebook clears its
# execution outputs, so there is no "build everything" shortcut -- pick just the notebook(s) touched.
from nb_build import md, code, save_notebook as _save


# under nbconvert the CWD is the notebook's own directory, so sys.path is set relative to it
_BOOTSTRAP = r"""
import os, sys

# make the in-repo modules importable (not pip-installed)
REPO = os.path.abspath("../..")
for _p in ("data/delta22", "analysis/code", "analysis/code/shared"):
    sys.path.insert(0, os.path.join(REPO, _p))
"""

# pure imports for the figure notebooks
_IMPORTS = r"""
import matplotlib.pyplot as plt

import delta22
import delta22_plots
import paths
"""

# data-file path constants + the figure_path helper, for the figure notebooks
_SETUP = r"""
DELTA22_HDF5 = paths.dataset_file("delta22", root=REPO)
XLSX = os.path.join(REPO, "data", "delta22", "delta22_experimental.xlsx")

def figure_path(name):
    os.makedirs("figures", exist_ok=True)
    return os.path.join("figures", name)
"""

# pure imports for the table notebooks (no matplotlib -- they only print and write spreadsheets)
_IMPORTS_TABLE = r"""
import delta22
import paths
"""

# figure imports plus numpy, for the few notebooks that use numpy directly in a notebook cell
_IMPORTS_NP = _IMPORTS.replace("import matplotlib.pyplot as plt",
                               "import numpy as np\nimport matplotlib.pyplot as plt", 1)
# table imports plus numpy/pandas (or just pandas), used directly when building the table dataframes
_IMPORTS_TABLE_NPPD = _IMPORTS_TABLE.replace("import delta22",
                                             "import numpy as np\nimport pandas as pd\n\nimport delta22", 1)
_IMPORTS_TABLE_PD = _IMPORTS_TABLE.replace("import delta22",
                                           "import pandas as pd\n\nimport delta22", 1)

# data-file path constants + the document_path helper, for the table notebooks
_SETUP_TABLE = r"""
DELTA22_HDF5 = paths.dataset_file("delta22", root=REPO)
XLSX = os.path.join(REPO, "data", "delta22", "delta22_experimental.xlsx")

def document_path(name):
    os.makedirs("documents", exist_ok=True)
    return os.path.join("documents", name)
"""

# only the notebooks that use N_SPLITS include this cell (fig2a_pareto uses its own, smaller
# PARETO_N_SPLITS instead).
_N_SPLITS_CELL = r"""
# the published panels use 250 seeded train/test splits
N_SPLITS = 250
"""


# Figure 3 (analysis/manuscript_figures/fig3.ipynb) is hand-authored with its plotting inline; its
# data comes from delta22.py (fig3a_pcm_benefit_by_solvent, fig3b_shift_differences,
# fig3d_formula_regressions).


# ----------------------------------------------------------------------------
# Figure 4 + SI S16/S17/S18: Explicit Solvent Corrections are Physically Meaningful
# ----------------------------------------------------------------------------
_FIG4_LOAD = r"""
# Figure 4 uses one DFT method for the proton sites; differences are taken against a reference
# solvent, and the solvent_mean pseudo-solvent is added for the solvent-averaged reference.
METHOD, BASIS, GEOM = "b3lyp_d3bj", "pcSseg2", "aimnet2"
q = delta22.load_query_df_dft(DELTA22_HDF5, XLSX, verbose=False)
one = q[(q["sap_nmr_method"] == METHOD) & (q["sap_basis"] == BASIS) & (q["sap_geometry_type"] == GEOM)]
one = delta22.add_solvent_mean(one)
print(len(one), "rows for", METHOD, BASIS, GEOM)
"""

# shared display labels for the 12 delta-22 solvents, used by Figure 4C and SI Figures S16-S18
_SOLVENT_LABELS = r"""
FIG4C_LABELS = {
    "chloroform": "CDCl3", "dichloromethane": "DCM", "tetrahydrofuran": "THF",
    "acetonitrile": "MeCN", "dimethylsulfoxide": "DMSO", "acetone": "acetone",
    "methanol": "MeOH", "TIP4P": "TIP4P", "trifluoroethanol": "TFE",
    "benzene": "benzene", "toluene": "toluene", "chlorobenzene": "chlorobenzene",
}
"""

# Figure 4 (analysis/manuscript_figures/fig4.ipynb) is hand-authored with its plotting inline; its
# data comes from delta22.py (solvent_induced_shifts, solvent_pair_differences,
# fit_differences_to_experimental). The _FIG4_LOAD / _SOLVENT_LABELS cells above are shared with the
# SI S16-S18 notebook below.


si_figure_s16_s18 = [
    md(r"""
# SI Figures S16-S18: predicted vs experimental solvent-induced shifts by reference solvent

Per solvent, implicit (PCM) and explicit (Desmond) predicted solvent-induced shift differences vs
measured (y=x ideal), for three reference solvents: **S16** chloroform, **S17** benzene, **S18**
solvent-averaged. One panel per solvent, each point a proton site.
"""),
    code(_BOOTSTRAP),
    code(_IMPORTS),
    code(_SETUP),
    code(_FIG4_LOAD),
    code(_SOLVENT_LABELS),
    code(r"""
# panel order matches the canonical figures; a different order from delta22.DESMOND_SOLVENTS
SI_S16_18_SOLVENT_ORDER = [
    "chloroform", "tetrahydrofuran", "dichloromethane", "acetone", "acetonitrile",
    "dimethylsulfoxide", "trifluoroethanol", "methanol", "TIP4P", "benzene", "toluene", "chlorobenzene",
]

# panel titles use the full solvent name; axis labels use the compact NMR abbreviations (THF, DCM,
# DMSO, TFE, CDCl3); the reference is CDCl3 / Benzene / "Avg Corr." for S16 / S17 / S18.
TITLE_LABEL = {"chloroform": "Chloroform", "tetrahydrofuran": "Tetrahydrofuran",
               "dichloromethane": "Dichloromethane", "acetone": "Acetone", "acetonitrile": "Acetonitrile",
               "dimethylsulfoxide": "Dimethylsulfoxide", "trifluoroethanol": "Trifluoroethanol",
               "methanol": "Methanol", "TIP4P": "TIP4P", "benzene": "Benzene", "toluene": "Toluene",
               "chlorobenzene": "Chlorobenzene"}
AXIS_LABEL = {**TITLE_LABEL, "chloroform": "CDCl3", "tetrahydrofuran": "THF",
              "dichloromethane": "DCM", "dimethylsulfoxide": "DMSO", "trifluoroethanol": "TFE"}
REF_LABEL = {"chloroform": "CDCl3", "benzene": "Benzene", "solvent_mean": "Avg Corr."}
"""),
    code(r"""
for figure, reference in [("S16", "chloroform"), ("S17", "benzene"), ("S18", "solvent_mean")]:
    sh = delta22.solvent_induced_shifts(one, reference, SI_S16_18_SOLVENT_ORDER, nucleus="H", explicit="desmond")
    implicit_fit = delta22.fit_differences_to_experimental(sh, "implicit_diff")
    explicit_fit = delta22.fit_differences_to_experimental(sh, "explicit_diff")
    print(f"{figure}: reference={reference:12s} n={explicit_fit['n']:4d}  "
          f"implicit fit RMSE={implicit_fit['rmse']:.3f}  explicit fit RMSE={explicit_fit['rmse']:.3f}")
    panel_solvents = [s for s in SI_S16_18_SOLVENT_ORDER if s != reference]
    delta22_plots.plot_shift_prediction_scatter_grid(
        sh, panel_solvents, REF_LABEL[reference], AXIS_LABEL, TITLE_LABEL,
        save_path=figure_path(f"si_figure_{figure.lower()}.png"))
    plt.show()
"""),
]


# ----------------------------------------------------------------------------
# SI Figure S7: Explicit Solvent Corrections are Method-Independent
# ----------------------------------------------------------------------------
si_figure_s07 = [
    md(r"""
# SI Figure S7: DFT ¹³C explicit-solvent correction, Desmond vs OpenMM

For the four OpenMM solvents, the DFT ¹³C explicit-solvent correction from Desmond MD vs from OpenMM
(y=x reference line).
"""),
    code(_BOOTSTRAP),
    code(_IMPORTS_NP),
    code(_SETUP),
    code(r"""
dft = delta22.load_query_df_dft(DELTA22_HDF5, XLSX, verbose=False)
"""),
    md("## DFT explicit corrections: Desmond vs OpenMM"),
    code(r"""
SOLVENT_ORDER = ["chloroform", "methanol", "TIP4P", "benzene"]
dft_pairs = {sv: delta22.explicit_correction_pairs(dft, sv, "C") for sv in SOLVENT_ORDER}
for sv, p in dft_pairs.items():
    print(f"DFT {sv:12s} n={len(p):3d}  r={np.corrcoef(p['desmond'], p['openMM'])[0,1]:.3f}")
delta22_plots.plot_desmond_vs_openmm_grid(dft_pairs, save_path=figure_path("si_figure_s07.png"))
plt.show()
"""),
]


# ----------------------------------------------------------------------------
# SI Figure S4: Implicit Solvent Corrections are Highly Correlated
# ----------------------------------------------------------------------------
si_figure_s04 = [
    md(r"""
# SI Figure S4: correlation of implicit (PCM) corrections across solvents and methods

**S4A** correlation across solvents (both nuclei), **S4B** one solvent vs another (¹H), **S4C**
correlation across methods (both nuclei). Cells show -log10(1-r), so 3 means r=0.999.
"""),
    code(_BOOTSTRAP),
    code(_IMPORTS_NP),
    code(_SETUP),
    code(r"""
METHOD, BASIS, GEOM = "b3lyp_d3bj", "pcSseg2", "pbe0_tz"
dft = delta22.load_query_df_dft(DELTA22_HDF5, XLSX, verbose=False)
base = dft[(dft["sap_nmr_method"] == METHOD) & (dft["sap_basis"] == BASIS)
           & (dft["sap_geometry_type"] == GEOM)]
one = base[base["nucleus"] == "H"]

# solvent order that groups polar-aprotic -> polar-protic -> aromatic (matches the published panel)
ORDERED_SOLVENTS = ["chloroform", "tetrahydrofuran", "dichloromethane", "acetone", "acetonitrile",
                    "dimethylsulfoxide", "trifluoroethanol", "methanol", "TIP4P",
                    "benzene", "toluene", "chlorobenzene"]
"""),
    md("## S4A: solvent-vs-solvent PCM correlation (¹H and ¹³C)"),
    code(r"""
for nucleus, label in [("H", "1H"), ("C", "13C")]:
    solvent_corr = delta22.correlation_matrix(base[base["nucleus"] == nucleus], "pcm", ["solute", "site"], "solvent")
    solvent_corr = solvent_corr.reindex(index=ORDERED_SOLVENTS, columns=ORDERED_SOLVENTS)
    vals = solvent_corr.values[np.triu_indices_from(solvent_corr.values, k=1)]
    print(f"solvent PCM correlation ({label}): mean r={np.nanmean(vals):.4f}  min r={np.nanmin(vals):.4f}")
    caption = ("Correlation coefficients between solvents across all 22 solutes.\n"
               f"PCM corrections computed with {METHOD} with the {BASIS} basis.\n"
               "A value of 3 means the coefficient is 0.999.")
    delta22_plots.plot_correlation_matrix(solvent_corr, f"Solvents are Highly Correlated ({nucleus})", caption,
                                          colormap="Reds", show_values=True,
                                          save_path=figure_path(f"si_figure_s04a_{label}.png"))
"""),
    md("## S4B: PCM corrections, chloroform vs acetonitrile (1H) -- one square of S4A"),
    code(r"""
pair = one.pivot_table(index=["solute", "site"], columns="solvent", values="pcm")[
    ["chloroform", "acetonitrile"]].dropna()
delta22_plots.plot_pcm_scatter(pair["chloroform"], pair["acetonitrile"], "chloroform", "acetonitrile", "H",
                               save_path=figure_path("si_figure_s04b_1H.png"))
"""),
    md("## S4C: method-vs-method correlation (chloroform, double hybrids excluded, ¹H and ¹³C)"),
    code(r"""
# one solvent, exclude the double hybrids whose PCM is the substituted reference value
for nucleus, label in [("H", "1H"), ("C", "13C")]:
    chcl3 = dft[(dft["sap_basis"] == BASIS) & (dft["sap_geometry_type"] == GEOM)
                & (dft["nucleus"] == nucleus) & (dft["solvent"] == "chloroform")
                & (~dft["sap_nmr_method"].isin(delta22.DOUBLE_HYBRID_METHODS))]
    method_corr = delta22.correlation_matrix(chcl3, "pcm", ["solute", "site"], "sap_nmr_method")
    method_order = sorted(method_corr.index)          # alphabetical, matches the published panel
    method_corr = method_corr.reindex(index=method_order, columns=method_order)
    mvals = method_corr.values[np.triu_indices_from(method_corr.values, k=1)]
    print(f"method PCM correlation ({label}): mean r={np.nanmean(mvals):.4f}  min r={np.nanmin(mvals):.4f}")
    caption = ("Correlation coefficients between NMR methods across all 22 solutes.\n"
               f"PCM corrections for chloroform with the {BASIS} basis.")
    delta22_plots.plot_correlation_matrix(method_corr, f"NMR Methods are Highly Correlated ({nucleus})", caption,
                                          colormap="Reds", show_values=True,
                                          save_path=figure_path(f"si_figure_s04c_{label}.png"))
"""),
]


# ----------------------------------------------------------------------------
# SI Figures S11 and S13: MagNET reproduces the DFT corrections
# ----------------------------------------------------------------------------
_DFT_NN_LOAD = r"""
dft = delta22.load_query_df_dft(DELTA22_HDF5, XLSX, verbose=False)
nn = delta22.load_query_df_nn(DELTA22_HDF5, XLSX, verbose=False)
"""

si_figure_s11 = [
    md(r"""
# SI Figure S11: MagNET vs DFT rovibrational (QCD) corrections

Four panels: **A** MagNET (NN) vs B3LYP/cc-pVDZ rovibrational (QCD) correction per site, both nuclei;
**B** the NN-minus-DFT error distribution; **C/D** every proton/carbon site's DFT and NN QCD
correction, one column per solute. (QCD is gas-phase, so no solvent axis.)
"""),
    code(_BOOTSTRAP),
    code(_IMPORTS),
    code(_SETUP),
    code(_DFT_NN_LOAD),
    md("## Panels A and B: MagNET vs DFT QCD correction, both nuclei"),
    code(r"""
qcd = delta22.compare_dft_nn(dft, nn, "qcd", keys=("solute", "site", "nucleus"))
delta22_plots.plot_qcd_scatter(qcd, save_path=figure_path("si_figure_s11a_scatter.png"))
delta22_plots.plot_qcd_error_histogram(qcd, save_path=figure_path("si_figure_s11b_hist.png"))
plt.show()
"""),
    md("## Panel C: every proton site's QCD correction, one column per solute"),
    code(r"""
delta22_plots.plot_qcd_correction_by_site(delta22.qcd_correction_by_site(dft, nn, nucleus="H"), "H", "upper left",
                                          save_path=figure_path("si_figure_s11c_sites_1H.png"))
plt.show()
"""),
    md("## Panel D: every carbon site's QCD correction, one column per solute"),
    code(r"""
delta22_plots.plot_qcd_correction_by_site(delta22.qcd_correction_by_site(dft, nn, nucleus="C"), "C", "upper right",
                                          save_path=figure_path("si_figure_s11d_sites_13C.png"))
plt.show()
"""),
]

si_figure_s13 = [
    md(r"""
# SI Figure S13: MagNET-x vs DFT explicit-solvent corrections

Four panels, both nuclei: **A** MagNET-x (NN) vs DFT explicit-solvent correction per site/solvent,
Desmond and OpenMM overlaid; **B** the NN-minus-DFT error distribution; **C** every site's DFT and NN
correction in chloroform, one column per solute; **D** the semi-parsimonious composite model's fit accuracy vs experiment across the four OpenMM solvents, split by engine and
DFT-vs-NN source.
"""),
    code(_BOOTSTRAP),
    code(_IMPORTS),
    code(_SETUP),
    code(_DFT_NN_LOAD),
    code(r"""
# openMM/Desmond engine hues (used by the scatter/histogram/site/fitting-accuracy panels below)
ENGINE_COLORS = {"openMM": "#2E86AB", "desmond": "#A23B72"}
ENGINE_LABELS = {"desmond": "Desmond", "openMM": "OpenMM"}
"""),
    md("## Panels A and B: MagNET-x vs DFT explicit corrections, Desmond and OpenMM overlaid"),
    code(r"""
explicit_by_engine = delta22.compare_dft_nn_by_engine(dft, nn, keys=("solute", "site", "nucleus", "solvent"))
delta22_plots.plot_dft_nn_scatter_by_engine(explicit_by_engine, ENGINE_COLORS, ENGINE_LABELS,
                                            save_path=figure_path("si_figure_s13a_scatter.png"))
delta22_plots.plot_dft_nn_error_histogram_by_engine(explicit_by_engine, ENGINE_COLORS, ENGINE_LABELS,
                                                    save_path=figure_path("si_figure_s13b_hist.png"))
plt.show()
"""),
    md("## Panel C: every proton/carbon site's explicit correction in chloroform"),
    code(r"""
for nucleus, label in [("H", "Proton"), ("C", "Carbon")]:
    pairs = delta22.explicit_correction_dft_nn_pairs(dft, nn, "chloroform", nucleus)
    delta22_plots.plot_explicit_correction_by_site(
        pairs, f"All {label} Sites: DFT vs NN Explicit Corrections (chloroform)", ENGINE_COLORS,
        save_path=figure_path(f"si_figure_s13c_sites_{'1H' if nucleus == 'H' else '13C'}.png"))
plt.show()
"""),
    md("## Panel D: fitting accuracy of the semi-parsimonious composite model"),
    code(_N_SPLITS_CELL),
    code(r"""
SOLVENT_ORDER = ["chloroform", "methanol", "TIP4P", "benzene"]
solutes = sorted(dft["solute"].unique())
for nucleus, label in [("H", "1H"), ("C", "13C")]:
    fitting = delta22.si_s13d_fitting_accuracy(dft, nn, SOLVENT_ORDER, N_SPLITS, solutes, nucleus=nucleus)
    delta22_plots.plot_fitting_accuracy_boxplot(
        fitting, SOLVENT_ORDER, label, ENGINE_COLORS,
        save_path=figure_path(f"si_figure_s13d_fitting_{'1H' if nucleus == 'H' else '13C'}.png"))
plt.show()
"""),
]


# ----------------------------------------------------------------------------
# SI Figure S5: PCM Captures Bulk Solvent Effects
# ----------------------------------------------------------------------------
si_figure_s05 = [
    md(r"""
# SI Figure S5: per-solvent PCM benefit vs bulk dielectric constant and polarizability

Per-solvent PCM benefit (% test-RMSE reduction) vs bulk dielectric constant and polarizability, for
the MagNET-Zero reference method per nucleus (WP04 ¹H, wB97X-D ¹³C). Panels A (¹H) and B (¹³C)
show all solvents; C repeats ¹H excluding aromatics and trifluoroethanol.
"""),
    code(_BOOTSTRAP),
    code(_IMPORTS),
    code(_SETUP),
    code(_N_SPLITS_CELL),
    code(r"""
dft = delta22.load_query_df_dft(DELTA22_HDF5, XLSX, verbose=False)
solutes = sorted(dft["solute"].unique())
benefit = {}
for nucleus, label in [("H", "1H"), ("C", "13C")]:
    method = delta22.MAGNET_PCM_OUTPUT_METHODS[nucleus]
    benefit[nucleus] = delta22.pcm_benefit_per_solvent(dft, method, "pcSseg2", "aimnet2",
                                                       delta22.DESMOND_SOLVENTS, n_splits=N_SPLITS,
                                                       solutes=solutes, nucleus=nucleus)
    print(f"{label} ({method}):"); print(benefit[nucleus].round(1).to_string())
"""),
    md("## S5A (1H, all solvents), S5B (13C, all solvents), and S5C (1H, excluding aromatics and trifluoroethanol)"),
    code(r"""
# panels A (1H) and B (13C): all solvents
for nucleus, letter, nuc_label in [("H", "A", "H"), ("C", "B", "C")]:
    delta22_plots.plot_pcm_benefit_vs_properties(
        benefit[nucleus], delta22.SOLVENT_DIELECTRIC, delta22.SOLVENT_POLARIZABILITY, nuc_label,
        save_path=figure_path(f"si_figure_s05{letter}_all.png"))
# panel C is 1H only, excluding the aromatic solvents and trifluoroethanol (matches the published SI)
delta22_plots.plot_pcm_benefit_vs_properties(
    benefit["H"], delta22.SOLVENT_DIELECTRIC, delta22.SOLVENT_POLARIZABILITY, "H",
    exclude=["benzene", "toluene", "chlorobenzene", "trifluoroethanol"],
    title_extra="Excluding Aromatic Solvents and Trifluoroethanol",
    save_path=figure_path("si_figure_s05A_no_aromatics_tfe.png"))
"""),
]


# ----------------------------------------------------------------------------
# SI Figure S8 (panels B-D): Convergence of Explicit Solvent Corrections
# ----------------------------------------------------------------------------
si_figure_s08 = [
    md(r"""
# SI Figure S8 (panels B-D): explicit-solvent correction vs MD-frame count

For an example site (AcOH proton in chloroform), the explicit-solvent correction vs number of MD frames included (running average): OpenMM vs Desmond (B1) and DFT vs MagNET-x/"NN" at fixed OpenMM (B3), with per-frame distributions (B2/B4) and autocorrelation (C). Panel D shows which frames have computed DFT data per solvent (also the source of B's histogram normalization).
"""),
    code(_BOOTSTRAP),
    code(r"""
import numpy as np
import matplotlib.pyplot as plt

import delta22
import delta22_reader
import delta22_plots
import paths
"""),
    code(_SETUP),
    code(r"""
site_atoms = delta22_reader.load_site_atom_data(XLSX, verbose=False)
idx = delta22.site_atom_indices(site_atoms.loc[("AcOH", "H", "H"), "atom_numbers"])
SOLUTE, SOLVENT = "AcOH", "chloroform"
"""),
    code(r"""
# OpenMM/Desmond engine hues; DFT vs NN reuse the OpenMM hue (DFT lightened, NN full strength)
ENGINE_COLORS = {"openMM": "#2E86AB", "desmond": "#A23B72"}
ENGINE_LABELS = {"openMM": "OpenMM", "desmond": "Desmond"}

SOURCE_COLORS = {"dft": delta22_plots.lighten_color(ENGINE_COLORS["openMM"]), "nn": ENGINE_COLORS["openMM"]}
SOURCE_LABELS = {"dft": "DFT", "nn": "NN"}
LABEL_COLORS = {**ENGINE_COLORS, **SOURCE_COLORS}
LABEL_NAMES = {**ENGINE_LABELS, **SOURCE_LABELS}
"""),
    md(r"""
## Panel B1/B2: OpenMM vs Desmond

Running-average convergence (B1) and per-frame distribution (B2), comparing the two MD engines.
"""),
    code(r"""
running_by_engine, finals_by_engine, per_frame_by_engine, n_frames_by_engine = {}, {}, {}, {}
for engine in ["openMM", "desmond"]:
    perturbed = delta22_reader.load_perturbed_shieldings(DELTA22_HDF5, SOLUTE, SOLVENT, engine, "dft")
    per_frame = delta22.frame_corrections(perturbed, idx)
    running = delta22.running_average(per_frame)
    running_by_engine[engine] = running
    finals_by_engine[engine] = running[~np.isnan(running)][-1]
    per_frame_by_engine[engine] = per_frame
    n_frames_by_engine[engine] = len(per_frame)          # the total trajectory length, valid or not
    print(f"{engine}: {int(np.sum(~np.isnan(per_frame)))} / {len(per_frame)} valid frames, "
          f"converged correction {finals_by_engine[engine]:.4f} ppm")

delta22_plots.plot_frame_convergence(
    running_by_engine, finals_by_engine, LABEL_COLORS, LABEL_NAMES,
    title=f"Convergence of Explicit Solvent Corrections\n({SOLUTE} in {SOLVENT})",
    save_path=figure_path("si_figure_s08_b1_convergence_engine.png"))
plt.show()

delta22_plots.plot_frame_correction_histogram(
    per_frame_by_engine, n_frames_by_engine, LABEL_COLORS, LABEL_NAMES,
    title="Distribution of Frame-wise Corrections",
    save_path=figure_path("si_figure_s08_b2_histogram_engine.png"))
plt.show()
"""),
    md(r"""
## Panel B3/B4: DFT vs NN

Same running average (B3) and per-frame distribution (B4), at fixed OpenMM engine, comparing DFT vs
MagNET-x ("NN") shieldings.
"""),
    code(r"""
running_by_source, finals_by_source, per_frame_by_source, n_frames_by_source = {}, {}, {}, {}
for source in ["dft", "nn"]:
    perturbed = delta22_reader.load_perturbed_shieldings(DELTA22_HDF5, SOLUTE, SOLVENT, "openMM", source)
    per_frame = delta22.frame_corrections(perturbed, idx)
    running = delta22.running_average(per_frame)
    running_by_source[source] = running
    finals_by_source[source] = running[~np.isnan(running)][-1]
    per_frame_by_source[source] = per_frame
    n_frames_by_source[source] = len(per_frame)
    print(f"{source}: {int(np.sum(~np.isnan(per_frame)))} / {len(per_frame)} valid frames, "
          f"converged correction {finals_by_source[source]:.4f} ppm")

delta22_plots.plot_frame_convergence(
    running_by_source, finals_by_source, LABEL_COLORS, LABEL_NAMES, xlabel="Number of MD Frames",
    title=f"OpenMM DFT vs. NN Convergence Comparison\n({SOLUTE} in {SOLVENT})",
    save_path=figure_path("si_figure_s08_b3_convergence_source.png"))
plt.show()

delta22_plots.plot_frame_correction_histogram(
    per_frame_by_source, n_frames_by_source, LABEL_COLORS, LABEL_NAMES,
    title="Distribution of Frame-wise Corrections",
    save_path=figure_path("si_figure_s08_b4_histogram_source.png"))
plt.show()
"""),
    md(r"""
## Panel C: autocorrelation, DFT vs NN

Autocorrelation of the per-frame correction to lag 200 (~100 frames per trajectory repeat), DFT vs NN,
same OpenMM trajectory as B3/B4.
"""),
    code(r"""
autocorr_by_source = {source: delta22.autocorrelation(per_frame_by_source[source], max_lag=200)
                      for source in ["dft", "nn"]}
for source, autocorr in autocorr_by_source.items():
    print(f"{source}: lag-1 autocorrelation {autocorr[1]:.3f}")

delta22_plots.plot_frame_autocorrelation(
    autocorr_by_source, LABEL_COLORS, LABEL_NAMES,
    title=f"Autocorrelation of Frame-wise Corrections\n({SOLUTE} in {SOLVENT})",
    save_path=figure_path("si_figure_s08_c_autocorrelation.png"))
plt.show()
"""),
    md(r"""
## Panel D: frame data availability

Which trajectory frames have computed DFT shielding for AcOH, across all 12 Desmond and 4 OpenMM
solvents. DFT was computed for a non-contiguous subset (compute-budget limits; jobs queued randomly).
"""),
    code(r"""
# row order matching the published panel: chloroform first, then by solvent class (aprotic ->
# protic -> aromatic); OpenMM only has the four explicit-solvent solvents
DESMOND_ORDER = ["chloroform", "tetrahydrofuran", "dichloromethane", "acetone", "acetonitrile",
                 "dimethylsulfoxide", "trifluoroethanol", "methanol", "TIP4P",
                 "benzene", "toluene", "chlorobenzene"]
OPENMM_ORDER = ["chloroform", "methanol", "TIP4P", "benzene"]

grids_by_engine, solvents_by_engine = {}, {}
for engine, solvents in [("desmond", DESMOND_ORDER), ("openMM", OPENMM_ORDER)]:
    grid, frame_counts = delta22.frame_validity_grid(DELTA22_HDF5, SOLUTE, solvents, engine, shield_type="dft")
    grids_by_engine[engine] = grid
    solvents_by_engine[engine] = [delta22_plots.display_solvent_name(s) for s in solvents]
    print(f"{engine}: {len(solvents)} solvents, frame counts {dict(zip(solvents, frame_counts))}")

delta22_plots.plot_frame_validity_heatmaps(grids_by_engine, solvents_by_engine, ENGINE_LABELS, solute=SOLUTE,
                                           save_path=figure_path("si_figure_s08_d_frame_validity.png"))
plt.show()
"""),
]


# ----------------------------------------------------------------------------
# Figure 2A + SI Figure S1: the accuracy-vs-cost Pareto frontier
# ----------------------------------------------------------------------------
_PARETO_LOAD = r"""
# Load the flat DFT and MagNET tables and the timing tables.
dft = delta22.load_query_df_dft(DELTA22_HDF5, XLSX, verbose=False)
nn = delta22.load_query_df_nn(DELTA22_HDF5, XLSX, verbose=False)
dft_gas_timings, nn_timings = delta22.load_pareto_timings(DELTA22_HDF5)

# One point per method/basis/geometry/nucleus/solvent, plus the solvent-averaged rows, each with its
# mean test RMSE and total compute time. MagNET appears as a single method (aimnet2, basis "N/A").
# Figure 2A uses 100 seeded splits, not the 250 the other delta-22 panels use, and trains on the
# first 10 shuffled solutes / tests on the rest; fig2a_pareto_points handles that split convention.
PARETO_N_SPLITS = 100
points = delta22.fig2a_pareto_points(dft, nn, dft_gas_timings, nn_timings, n_splits=PARETO_N_SPLITS)
print(points["nmr_method"].nunique(), "methods;",
      "MagNET total time", float(points.query("nmr_method=='MagNET'")["total_time"].iloc[0]), "s")
"""

# Figure 2A (analysis/manuscript_figures/fig2a_pareto.ipynb) is hand-authored with its plotting
# inline; its data comes from delta22.py.

# SI Figure S1 (analysis/si_figures/si_figure_s01.ipynb) is hand-authored too, with the same Pareto
# panel engine as Figure 2A run for both nuclei; its data comes from delta22.py.

# ----------------------------------------------------------------------------
# SI Tables S1/S2: the Pareto plot data (same chloroform rows as SI Figure S1)
# ----------------------------------------------------------------------------
si_table_s01_s02_pareto = [
    md(r"""
# SI Tables S1 and S2: the Pareto plot data

Each method's total compute time and CDCl3 test RMSE, for proton (S1) and carbon (S2): a curated
55-row subset (MagNET, one AIMNet2-geometry row, and the DFT grid on PBE0/cc-pVTZ geometries).
"""),
    code(_BOOTSTRAP),
    code(_IMPORTS_TABLE_NPPD),
    code(_SETUP_TABLE),
    code(_PARETO_LOAD),
    md("## Tables S1 and S2"),
    code(r"""
# full column set matching the SI tables: identity + fitting RMSE + the three time components and
# their log10 (all solvents here are chloroform). geometry_time and nmr_time sum to total_time.
COLUMNS = ["geometry_type", "nmr_method", "basis", "solvent", "fitting_RMSE",
           "geometry_time", "nmr_time", "total_time", "log10_total_time"]

curated = {}
for nucleus, label in [("H", "S1"), ("C", "S2")]:
    sub = delta22.pareto_table_curated(points, nucleus).copy()
    sub["log10_total_time"] = np.log10(sub["total_time"])
    sub = sub[COLUMNS]
    curated[label] = sub
    print(f"Table {label} ({nucleus}): {len(sub)} rows x {sub.shape[1]} columns")
    print(sub.to_string(index=False), "\n")

# write the two tables to this notebook's documents/ folder, one sheet per SI table
out = document_path("si_table_s01_s02_pareto.xlsx")
with pd.ExcelWriter(out) as writer:
    curated["S1"].to_excel(writer, sheet_name="Table S1 (1H)", index=False)
    curated["S2"].to_excel(writer, sheet_name="Table S2 (13C)", index=False)
print("wrote", os.path.relpath(out, REPO))
"""),
]


# ----------------------------------------------------------------------------
# SI Figure S6: Explicit Solvation is Broadly Beneficial
# ----------------------------------------------------------------------------
si_figure_s06 = [
    md(r"""
# SI Figure S6: per-solvent test RMSE of six solvent-correction models

Per nucleus, the per-solvent test RMSE of six solvent-correction models: SotA implicit (single-slope
PCM), the paper's implicit fit (stationary + PCM as separate terms), and explicit (Desmond), each
with and without a vibrational correction (QCD for ¹H, Desmond vibration for ¹³C). Level of theory:
dsd_pbep86/pcSseg3 on pbe0_tz geometries, PCM substituted from b3lyp_d3bj/pcSseg3.
"""),
    code(_BOOTSTRAP),
    code(_IMPORTS),
    code(_SETUP),
    code(r"""
query = delta22.add_composite_columns(delta22.load_query_df_dft(
    DELTA22_HDF5, XLSX, pcm_reference_method="b3lyp_d3bj", pcm_reference_basis="pcSseg3", verbose=False))
solutes = sorted(query["solute"].unique())

METHOD, BASIS, GEOM = "dsd_pbep86", "pcSseg3", "pbe0_tz"

# same six categories for both nuclei; the "+ Vibrations" term differs (qcd for H, desmond_vib for C)
LADDER = {
    "H": {
        "stationary_plus_pcm": "SotA Implicit Solvent",
        "stationary_plus_pcm_plus_qcd": "SotA Implicit Solvent + Vibrations",
        "stationary + pcm": "Implicit Solvent",
        "stationary_plus_qcd + pcm": "Implicit Solvent + Vibrations",
        "stationary + desmond": "Explicit Solvent",
        "stationary_plus_qcd + desmond": "Explicit Solvent + Vibrations"},
    "C": {
        "stationary_plus_pcm": "SotA Implicit Solvent",
        "stationary_plus_pcm_plus_des_vib": "SotA Implicit Solvent + Vibrations",
        "stationary + pcm": "Implicit Solvent",
        "stationary_plus_des_vib + pcm": "Implicit Solvent + Vibrations",
        "stationary + desmond": "Explicit Solvent",
        "stationary_plus_des_vib + desmond": "Explicit Solvent + Vibrations"},
}
"""),
    code(r"""
# one dark/light pair per solvent-treatment family (SotA implicit / implicit / explicit), light = +Vibrations
S6_COLORS = ["#C4B037", "#F5EDA0", "#A72608", "#E4A0A0", "#5F7C8A", "#B9E7DF"]
"""),
    md("## Formula ladder per nucleus"),
    code(_N_SPLITS_CELL),
    code(r"""
for nucleus, labels in LADDER.items():
    results = delta22.fig3d_formula_regressions(query, METHOD, BASIS, GEOM, list(labels),
                                                delta22.DESMOND_SOLVENTS, n_splits=N_SPLITS,
                                                solutes=solutes, nucleus=nucleus)
    med = results.groupby("formula")["test_RMSE"].median()
    print(f"--- {nucleus} ({METHOD}) median test RMSE ---")
    for formula in labels:
        print(f"  {labels[formula]:38s} {med[formula]:.4f}")
    delta22_plots.plot_formula_ladder_boxplot(
        results, labels, delta22.SOLVENT_GROUPS, nucleus=nucleus, colors=S6_COLORS,
        title=f"Explicit Solvation + Vibrational Effects Improve Predictions Across Solvents ({nucleus} Nucleus)",
        save_path=figure_path(f"si_figure_s06_ladder_{'1H' if nucleus == 'H' else '13C'}.png"))
plt.show()
"""),
]


# ----------------------------------------------------------------------------
# SI Table S6: Explicit solvent corrections are superior for predicting solvent-induced shifts
# ----------------------------------------------------------------------------
# SI table row order (chloroform is the reference and so is excluded)
_S6_SOLVENT_ORDER = [
    "tetrahydrofuran", "dichloromethane", "acetone", "acetonitrile", "dimethylsulfoxide",
    "trifluoroethanol", "methanol", "TIP4P", "benzene", "toluene", "chlorobenzene",
]
si_table_s06 = [
    md(r"""
# SI Table S6: per-solvent implicit vs explicit fit RMSE (chloroform reference, ¹H)

Per solvent (chloroform reference), fit RMSE of experimental solvent-induced ¹H shifts using implicit
(PCM) vs explicit (Desmond), and the percent improvement.
"""),
    code(_BOOTSTRAP),
    code(_IMPORTS_TABLE_PD),
    code(_SETUP_TABLE),
    code(_FIG4_LOAD),
    md("## Table S6: per-solvent implicit vs explicit fit RMSE (chloroform reference, 1H)"),
    code(r"""
rows = []
for solvent in _S6_SOLVENT_ORDER:
    sp = delta22.solvent_pair_differences(one, solvent, "chloroform", nucleus="H", explicit="desmond")
    implicit = delta22.fit_differences_to_experimental(sp, "implicit_diff")["rmse"]
    explicit = delta22.fit_differences_to_experimental(sp, "explicit_diff")["rmse"]
    rows.append({"solvent": solvent,
                 "implicit_fitting_rmse": round(implicit, 4),
                 "explicit_fitting_rmse": round(explicit, 4),
                 "explicit_improvement_pct": round(100.0 * (implicit - explicit) / implicit, 2)})
table_s6 = pd.DataFrame(rows)
print(table_s6.to_string(index=False))

# write the table to this notebook's documents/ folder
out = document_path("si_table_s06_solvent_corrections.xlsx")
with pd.ExcelWriter(out) as writer:
    table_s6.to_excel(writer, sheet_name="Table S6", index=False)
print("wrote", os.path.relpath(out, REPO))
""".replace("_S6_SOLVENT_ORDER", repr(_S6_SOLVENT_ORDER))),
]


# ----------------------------------------------------------------------------
# SI Figure S14: End-to-End Neural Network Acceleration Delivers Equivalent Performance
# ----------------------------------------------------------------------------
si_figure_s14 = [
    md(r"""
# SI Figure S14: test RMSE with DFT vs MagNET features, by solvent

For the four explicit-solvent solvents (chloroform, methanol, TIP4P water, benzene), per-solvent test
RMSE of predicting experimental ¹H/¹³C shifts with **DFT** features (solid) vs end-to-end **MagNET/NN**
features (lightened), each under three conditions (implicit SOTA, explicit, explicit + vibrations).
Two pages, one per MD engine (Desmond, OpenMM); nitromethane dropped.
"""),
    code(_BOOTSTRAP),
    code(_IMPORTS),
    code(_SETUP),
    code(_N_SPLITS_CELL),
    code(r"""
dft = delta22.add_composite_columns(delta22.load_query_df_dft(DELTA22_HDF5, XLSX, verbose=False))
nn = delta22.add_composite_columns(delta22.load_query_df_nn(DELTA22_HDF5, XLSX, verbose=False))
print(len(dft), "DFT rows;", len(nn), "NN rows")

LABELS = ["Implicit Solvent (SOTA)", "Explicit Solvent", "Explicit Solvent + Vibrations"]
SS_LABELS = {"TIP4P": "Water (TIP4P)"}
S14_SOLVENTS = ["chloroform", "methanol", "TIP4P", "benzene"]   # the 4 explicit solvents, SI order
"""),
    md("## Desmond page (DFT vs NN)"),
    code(r"""
# implicit = 2-term (stationary + pcm); explicit = stationary + desmond;
# explicit + vibrations = stationary_plus_qcd + desmond (1H) / stationary_plus_des_vib + desmond (13C)
DESMOND_FORMULAS = {
    "H":  ["stationary + pcm", "stationary + desmond", "stationary_plus_qcd + desmond"],
    "C":  ["stationary + pcm", "stationary + desmond", "stationary_plus_des_vib + desmond"],
}
for nucleus, label in [("H", "1H"), ("C", "13C")]:
    formulas = DESMOND_FORMULAS[nucleus]
    delta22_plots.plot_ss_boxplot_dft_vs_nn(
        delta22_plots.ss_fits(dft, nucleus, formulas, S14_SOLVENTS, N_SPLITS, dft=True),
        delta22_plots.ss_fits(nn, nucleus, formulas, S14_SOLVENTS, N_SPLITS),
        S14_SOLVENTS, formulas, LABELS, label, solvent_labels=SS_LABELS,
        save_path=figure_path(f"si_figure_s14_desmond_{label}.png"))
plt.show()
"""),
    md("## OpenMM page (DFT vs NN)"),
    code(r"""
# implicit = 1-term composite (stationary_plus_pcm); explicit = stationary + openMM;
# explicit + vibrations = stationary_plus_qcd + openMM (1H) / stationary_plus_op_vib + openMM (13C)
OPENMM_FORMULAS = {
    "H":  ["stationary_plus_pcm", "stationary + openMM", "stationary_plus_qcd + openMM"],
    "C":  ["stationary_plus_pcm", "stationary + openMM", "stationary_plus_op_vib + openMM"],
}
for nucleus, label in [("H", "1H"), ("C", "13C")]:
    formulas = OPENMM_FORMULAS[nucleus]
    delta22_plots.plot_ss_boxplot_dft_vs_nn(
        delta22_plots.ss_fits(dft, nucleus, formulas, S14_SOLVENTS, N_SPLITS, dft=True),
        delta22_plots.ss_fits(nn, nucleus, formulas, S14_SOLVENTS, N_SPLITS),
        S14_SOLVENTS, formulas, LABELS, label, solvent_labels=SS_LABELS,
        save_path=figure_path(f"si_figure_s14_openmm_{label}.png"))
plt.show()
"""),
]


# name -> (cells, path relative to repo root). Notebooks are grouped by role (manuscript/si_figures/
# si_tables), not by dataset.
NOTEBOOKS = {
    "si_figure_s06": (si_figure_s06, "analysis/si_figures/si_figure_s06.ipynb"),
    "si_figure_s14": (si_figure_s14, "analysis/si_figures/si_figure_s14.ipynb"),
    "si_table_s06": (si_table_s06, "analysis/si_tables/si_table_s06_solvent_corrections.ipynb"),
    "si_table_s01_s02_pareto": (si_table_s01_s02_pareto, "analysis/si_tables/si_table_s01_s02_pareto.ipynb"),
    "si_figure_s16_s18": (si_figure_s16_s18, "analysis/si_figures/si_figure_s16_s18.ipynb"),
    "si_figure_s07": (si_figure_s07, "analysis/si_figures/si_figure_s07.ipynb"),
    "si_figure_s04": (si_figure_s04, "analysis/si_figures/si_figure_s04.ipynb"),
    "si_figure_s11": (si_figure_s11, "analysis/si_figures/si_figure_s11.ipynb"),
    "si_figure_s13": (si_figure_s13, "analysis/si_figures/si_figure_s13.ipynb"),
    "si_figure_s05": (si_figure_s05, "analysis/si_figures/si_figure_s05.ipynb"),
    "si_figure_s08": (si_figure_s08, "analysis/si_figures/si_figure_s08_panelsBCD.ipynb"),
}

if __name__ == "__main__":
    import os
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    names = sys.argv[1:]
    if not names:
        print("usage: python3 build_nb_delta22.py <name> [<name> ...]")
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
