# build_nb_leveling.py -- generator for the SI Figure S3 leveling notebook.
#
# Source of truth for the notebook. Edit the cell sources here, then regenerate and re-execute:
#
#     python3 build_nb_leveling.py si_figure_s03_leveling
#     jupyter nbconvert --to notebook --execute --inplace ../si_figures/si_figure_s03_leveling.ipynb
#
# Regenerating overwrites the .ipynb (clearing its execution outputs), so do not hand-edit it.
# All real code lives in leveling.py; the notebook only runs the protocol and shows results.
#
# Covers NS372 across all 8 nuclei, plus delta22 (the same leveling.py analysis run on both
# datasets). Main-text Figure 2B (the clean delta22 correlation matrix) is built separately in
# analysis/manuscript_figures/fig2b_correlation.ipynb; Figure 2C is a hand-drawn schematic, not
# reproduced in code. This notebook computes both res_ns372 and res_delta22 (via the shared
# setup/load cells below) so its combined cross-dataset summary table is self-contained.
from nb_build import md, code, save_notebook as _save


_TITLE_S3 = r"""
# SI Figure S3: inter-method correlation and PCA of NS372 and delta-22 shieldings

Inter-method correlation matrices and PC1/PC2 loadings for NS372 (44 functionals x 8 nuclei) and
delta22 (18 functionals, ¹H/¹³C), plus a combined summary table.
"""

_METHOD_INTRO = r"""
Two independent NMR shielding datasets, analyzed separately (never pooled):

| Dataset | Source | Methods | Nuclei | Reference |
|---|---|---|---|---|
| **NS372** | Schattenberg & Kaupp, *JCTC* **17**, 7602 (2021) | 44 DFT/WFT functionals | ¹H ¹¹B ¹³C ¹⁵N ¹⁷O ¹⁹F ³¹P ³³S | CCSD(T)/pcSseg-3 |
| **delta22** | in-house `delta22.hdf5` | 18 gas-phase functionals, largest basis (pcSseg-3; mp2 → pcSseg-2), PBE0/cc-pVTZ geometry | ¹H ¹³C | DSD-PBEP86 (highest-rung in-set method, stands in for CCSD(T)) |
"""

_CONFIG_MD = "## 1. Configuration"

_BOOTSTRAP = r"""
import os, sys

REPO = os.path.abspath("../..")
for _p in ("analysis/code", "analysis/code/shared"):
    sys.path.insert(0, os.path.join(REPO, _p))
"""

_IMPORTS = r"""
import glob
import pandas as pd
import matplotlib.pyplot as plt

import paths
import leveling
import leveling_plots
"""

_PATH_SETUP = r"""
# inputs:
#   - the Kaupp NS372 spreadsheet is small and ships in the repo's data/ns372/ folder
#   - delta22.hdf5 is large: it resolves from the repo's data/delta22/ folder, carried by the
#     Hugging Face checkout via Git LFS (see analysis/code/paths.py)
# the delta22 file is encoded as whole numbers (real value x 10,000); the loader
# in leveling.py decodes it on read (a plain-decimal copy also works, since the
# decode step is a no-op on floating-point data).
KAUPP_XLSX = os.path.join(REPO, "data", "ns372", "ct1c00919_si_002.xlsx")
DELTA22_H5 = paths.dataset_file("delta22", root=REPO)
SAVE_DPI   = 200        # SI-quality raster output

def figure_path(name):
    os.makedirs("figures", exist_ok=True)
    return os.path.join("figures", name)

# self-clean: this notebook builds PNG names dynamically (one per nucleus), so drop any
# previously written figures before regenerating (glob on a missing folder returns [])
for _stale in glob.glob(os.path.join("figures", "*.png")):
    os.remove(_stale)

pd.set_option('display.width', 150)
pd.set_option('display.max_columns', 25)
"""

_STYLE_CONFIG = r"""
# functional-family colours used for every figure (family ordering lives in leveling.py)
FAMILY_COLORS = {'LDA':'#777777','GGA':'#1f77b4','mGGA':'#17becf','GH':'#2ca02c',
                 'RSH':'#9467bd','LH':'#8c564b','DH':'#ff7f0e','WFT':'#d62728',
                 'ref':'#FF1493'}

# proper Unicode-superscript labels for nuclei in figure titles
NUC_DISPLAY = {'1H':'¹H','11B':'¹¹B','13C':'¹³C',
               '15N':'¹⁵N','17O':'¹⁷O','19F':'¹⁹F',
               '31P':'³¹P','33S':'³³S'}

# adjustText parameters tuned per nucleus -- 1H and 13C have very dense central
# clusters and need stronger expansion; the paramagnetic-shielding nuclei
# (15N/17O/19F) already spread methods along the parabola and need a gentler
# pass to avoid over-flinging labels.
PCA_ADJUST_DEFAULT = dict(
    force_text=(0.35, 0.55), force_explode=(0.25, 0.40),
    force_static=(0.10, 0.15), force_pull=(0.02, 0.02),
    expand=(1.25, 1.35), time_lim=4,
)
PCA_ADJUST = {
    '1H':  {**PCA_ADJUST_DEFAULT, 'force_text':(0.75, 1.05),
            'force_explode':(0.65, 0.90), 'expand':(1.7, 1.9), 'time_lim':7},
    '11B': {**PCA_ADJUST_DEFAULT, 'force_text':(0.50, 0.75),
            'force_explode':(0.40, 0.60), 'expand':(1.4, 1.55), 'time_lim':5},
    '13C': {**PCA_ADJUST_DEFAULT, 'force_text':(0.70, 1.00),
            'force_explode':(0.60, 0.85), 'expand':(1.6, 1.8), 'time_lim':6},
    '15N': {**PCA_ADJUST_DEFAULT, 'force_text':(0.60, 0.85),
            'force_explode':(0.50, 0.70), 'expand':(1.5, 1.65), 'time_lim':6},
    '17O': {**PCA_ADJUST_DEFAULT, 'force_text':(0.60, 0.85),
            'force_explode':(0.50, 0.70), 'expand':(1.5, 1.65), 'time_lim':6},
    '31P': {**PCA_ADJUST_DEFAULT, 'force_text':(0.65, 0.90),
            'force_explode':(0.55, 0.75), 'expand':(1.55, 1.7), 'time_lim':6},
    '33S': {**PCA_ADJUST_DEFAULT, 'force_text':(0.65, 0.90),
            'force_explode':(0.55, 0.75), 'expand':(1.55, 1.7), 'time_lim':6},
}
"""

_NS372_DEF_MD = r"""
## 2. NS372 (Kaupp) - definitions

Conventional GIAO shieldings for 44 functionals across 8 main-group nuclei, with a
CCSD(T)/pcSseg-3 reference. Kaupp Reduced-Set exclusions (F₃⁻, O₃, BH - multireference
outliers) are applied. Input: `ct1c00919_si_002.xlsx`.
"""

_DELTA22_DEF_MD = r"""
## 3. delta22 - definitions

Gas-phase conventional GIAO shieldings from `delta22.hdf5`: 18 functionals at their
largest available basis (pcSseg-3; plain `mp2` only to pcSseg-2), at the PBE0/cc-pVTZ
geometry. Observations are pooled ¹H / ¹³C atom sites across all 22 solutes. There is no
CCSD(T) reference in the file; **DSD-PBEP86 is used as the reference** for delta22 - it
is the highest-rung double-hybrid available in this method set and stands in for CCSD(T)
in the same role. The stored whole-number shieldings are decoded on read.
"""

_LOAD_MD = r"""
## 4. Load datasets + global colour scale

Run the loaders, analyse every nucleus, and compute the figure-wide
`-log10(1-|r|)` maximum (`GLOBAL_VMAX`) used as the colour scale on every correlation matrix
below, for NS372 and delta22 alike.
"""

_LOAD = r"""
ns372   = leveling.load_ns372(KAUPP_XLSX)
delta22 = leveling.load_delta22(DELTA22_H5)

res_ns372   = {nuc: leveling.analyze_nucleus(d['M'], d['methods'], d['ref'])
               for nuc, d in ns372.items()}
res_delta22 = {nuc: leveling.analyze_nucleus(d['M'], d['methods'], d['ref'])
               for nuc, d in delta22.items()}

GLOBAL_VMAX = max(leveling.dataset_logr_max(res_ns372), leveling.dataset_logr_max(res_delta22))
print(f'NS372  : {len(ns372)} nuclei')
print(f'delta22: {len(delta22)} nuclei  (reference = {leveling.DELTA22_REF})')
print(f'global colour scale: 0 -> {GLOBAL_VMAX} on the -log10(1-|r|) axis')
for nuc, d in ns372.items():
    print(f'  NS372 {nuc:4s}: {d["M"].shape[0]:4d} mols  x  {d["M"].shape[1]-1} methods + CCSD(T)')
for nuc, d in delta22.items():
    print(f'  delta22 {nuc:4s}: {d["M"].shape[0]:4d} sites x  {d["M"].shape[1]} methods')
"""

SHARED = [
    md(_CONFIG_MD),
    code(_BOOTSTRAP),
    code(_IMPORTS),
    code(_PATH_SETUP),
    code(_STYLE_CONFIG),
    md(_NS372_DEF_MD),
    md(_DELTA22_DEF_MD),
    md(_LOAD_MD),
    code(_LOAD),
]

# ----------------------------------------------------------------------------
# SI Figure S3: NS372 results across all 8 nuclei + the combined cross-dataset summary
# ----------------------------------------------------------------------------
si_figure_s03_leveling = [
    md(_TITLE_S3),
    md(_METHOD_INTRO),
    *SHARED,
    md('## 5. NS372 results'),
    md('### 5.1  Leveling diagnostics - NS372'),
    code(r"""
sum_ns372 = leveling.summarize(ns372, res_ns372)
print('NS372 - leveling diagnostics (CCSD(T) included as a method column):')
sum_ns372.round(5)
"""),
    md(r"""
### 5.2  Per-method scaled RMSE vs CCSD(T)

`scaled_rmse` = RMSE of residuals after a per-method linear fit
`sigma_method ~ a*sigma_CCSD(T) + b` - the error that survives empirical linear scaling.
"""),
    code(r"""
rmse_ns372 = pd.DataFrame({nuc: res_ns372[nuc]['scaled_rmse'] for nuc in res_ns372})
rmse_ns372 = rmse_ns372.drop(index='CCSD(T)', errors='ignore')
rmse_ns372 = rmse_ns372.reindex(ns372['1H']['methods'][:-1])    # family order
print('NS372 - per-method scaled RMSE vs CCSD(T) (ppm):')
rmse_ns372.round(3)
"""),
    md('### 5.3  Inter-method correlation matrices - NS372 (one nucleus per file)'),
    code(r"""
NS372_NUCS = ['1H', '11B', '13C', '15N', '17O', '19F', '31P', '33S']
for nuc in NS372_NUCS:
    fams = {nuc: ns372[nuc]['families']}
    fig = leveling_plots.plot_corr_matrix('NS372', [nuc], res_ns372, fams,
                                          GLOBAL_VMAX, ref_name='CCSD(T)',
                                          nuc_display=NUC_DISPLAY)
    fig.savefig(figure_path(f'si_figure_s03_corr_{nuc}.png'),
                dpi=SAVE_DPI, bbox_inches='tight')
    plt.show()
"""),
    md('### 5.4  PC1/PC2 structure - NS372 (one nucleus per file)'),
    code(r"""
for nuc in NS372_NUCS:
    fams = {nuc: ns372[nuc]['families']}
    fig = leveling_plots.plot_pca_pair('NS372', [nuc], res_ns372, fams,
                                       family_colors=FAMILY_COLORS, nuc_display=NUC_DISPLAY,
                                       pca_adjust=PCA_ADJUST, pca_adjust_default=PCA_ADJUST_DEFAULT)
    fig.savefig(figure_path(f'si_figure_s03_pca_{nuc}.png'),
                dpi=SAVE_DPI, bbox_inches='tight')
    plt.show()
"""),
    md(r"""
### 5.5  delta22 panels (lead the published figure: corr ¹H/¹³C + PCA)

The same correlation + PCA layout on the delta22 gas-phase set. DSD-PBEP86 is the reference (delta22
has no CCSD(T)); it appears as an ordinary double-hybrid point in the PCA, with no CCSD(T) star.
"""),
    code(r"""
# delta22 correlation matrices (canonical S3 A = 1H, B = 13C)
for nuc in ['1H', '13C']:
    fams = {nuc: delta22[nuc]['families']}
    fig = leveling_plots.plot_corr_matrix('delta22', [nuc], res_delta22, fams,
                                          GLOBAL_VMAX, ref_name=leveling.DELTA22_REF,
                                          nuc_display=NUC_DISPLAY)
    fig.savefig(figure_path(f'si_figure_s03_delta22_corr_{nuc}.png'),
                dpi=SAVE_DPI, bbox_inches='tight')
    plt.show()

# delta22 PCA projection: 1H and 13C stacked in one figure (canonical S3 C)
fams = {nuc: delta22[nuc]['families'] for nuc in ['1H', '13C']}
fig = leveling_plots.plot_pca_pair('delta22', ['1H', '13C'], res_delta22, fams,
                                   family_colors=FAMILY_COLORS, nuc_display=NUC_DISPLAY,
                                   pca_adjust=PCA_ADJUST, pca_adjust_default=PCA_ADJUST_DEFAULT)
fig.savefig(figure_path('si_figure_s03_delta22_pca.png'),
            dpi=SAVE_DPI, bbox_inches='tight')
plt.show()
"""),
    md('## 6. Combined summary (both datasets)'),
    code(r"""
# delta22's per-nucleus diagnostics (same summarize() used for NS372 above), computed here
# too so this combined table is self-contained -- it reuses res_delta22 from the shared load
# cell above, so this is cheap (no new correlation/PCA computation).
sum_delta22 = leveling.summarize(delta22, res_delta22)

combined = pd.concat([sum_ns372.assign(dataset='NS372'),
                      sum_delta22.assign(dataset='delta22')], ignore_index=True)
combined = combined[['dataset','nucleus','n_obs','n_methods','r_min','r_median',
                     'PC1_pct','PC2_pct','PC3plus_pct','parabola_R2']]
print('Leveling effect - both datasets (analyzed separately):')
print(f'  PC1 range : {combined.PC1_pct.min():.2f}% - {combined.PC1_pct.max():.2f}%')
print(f'  min pairwise r : {combined.r_min.min():.5f}  (worst case, all nuclei)')
print(f'  parabola R2 range : {combined.parabola_R2.min():.3f} - {combined.parabola_R2.max():.3f}')
print(f'  global colour scale : 0 -> {GLOBAL_VMAX} on -log10(1-|r|)')
combined.round(5)
"""),
]

# name -> (cells, path relative to repo root). Notebooks are grouped by role (manuscript/si_figures/
# si_tables), not by dataset.
NOTEBOOKS = {
    "si_figure_s03_leveling": (si_figure_s03_leveling, "analysis/si_figures/si_figure_s03_leveling.ipynb"),
}

if __name__ == "__main__":
    import os
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    names = sys.argv[1:]
    if not names:
        print("usage: python3 build_nb_leveling.py <name> [<name> ...]")
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
