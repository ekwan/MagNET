"""Source of truth for the composite-model ablation-study notebook. Edit the cell sources here,
then regenerate:

    python3 build_nb_composite_models.py --write
    jupyter nbconvert --to notebook --execute --inplace ../si_figures/si_composite_models_ablations.ipynb

Regenerating overwrites the .ipynb, so do not hand-edit the notebook. All real code lives in
composite_models.py (numbers and the openpyxl table-writing layer, which itself calls delta22.py's
existing fitting harness); the notebook only runs it. The notebook itself lives in
analysis/si_figures/ (grouped by role), not here.

Run with no arguments to see this usage message; pass --write to regenerate. This generator produces
a single notebook and takes no name argument, so a bare run must not silently overwrite it.
"""
import os
import sys

from nb_build import md, code, save_notebook as _save


_BOOTSTRAP = r"""
import os, sys

# make the in-repo modules importable (not pip-installed)
REPO = os.path.abspath("../..")
for _p in ("data/delta22", "analysis/code", "analysis/code/shared"):
    sys.path.insert(0, os.path.join(REPO, _p))
"""

_IMPORTS = r"""
import matplotlib.pyplot as plt

import delta22
import composite_models
import composite_plots
import paths
"""

_SETUP = r"""
DELTA22_HDF5 = paths.dataset_file("delta22", root=REPO)
XLSX = os.path.join(REPO, "data", "delta22", "delta22_experimental.xlsx")

def figure_path(name):
    os.makedirs("figures", exist_ok=True)
    return os.path.join("figures", name)

def document_path(name):
    os.makedirs("documents", exist_ok=True)
    return os.path.join("documents", name)

# the shipped ablations.xlsx used 250 seeded train/test splits per (formula, solvent)
N_SPLITS = 250
"""

si_composite_models_ablations = [
    md(r"""
# Composite-formula ablation workbook

Rebuilds the composite-formula ablation grid: ~25 composite-formula variants (stationary geometry
plus some combination of implicit PCM, explicit Desmond, and rovibrational QCD corrections) across
all 12 delta-22 solvents, at two reference levels: DSD-PBEP86/pcSseg-3 (geometry PBE0/tz) and the
MagNET-Zero training reference (WP04/pcSseg-2 for ¹H, ωB97X-D/pcSseg-2 for ¹³C). Also reports the
solvent-averaged correlations between the composite-model features.
"""),
    code(_BOOTSTRAP),
    code(_IMPORTS),
    code(_SETUP),
    code(r"""
query_df_dft = delta22.add_composite_columns(delta22.load_query_df_dft(DELTA22_HDF5, XLSX, verbose=False))
solutes = delta22.delta22_solutes(DELTA22_HDF5)
print(len(query_df_dft), "rows;", len(solutes), "solutes")
"""),
    md("## Preview: one formula's mean test RMSE at the DSD-PBEP86 reference level"),
    code(r"""
# preview before the full build (all ~25 formulas x 12 solvents x 250 splits x 2 nuclei x 2
# reference levels)
level = composite_models.REFERENCE_LEVELS["dsd"]
preview = composite_models.ablation_rmse_table(query_df_dft, "H", level["method_h"], level["basis_h"],
                                 level["geometry_h"], solutes, n_splits=20)
preview[["chloroform", "benzene", "Mean Test RMSE"]].round(3)
"""),
    md("## Build the full ablations workbook (both reference levels, both nuclei)"),
    code(r"""
output_path = document_path("ablations.xlsx")
composite_models.build_ablations_workbook(query_df_dft, solutes, output_path, n_splits=N_SPLITS)
"""),
    md(r"""
## Correlations Between Features

Solvent-averaged Pearson r correlation matrix between the five composite-model features (stationary
shielding, PCM, Desmond, its vibrational analogue, QCD), both nuclei, plus a per-solvent
PCM-vs-Desmond table.
"""),
    code(r"""
for nucleus, label in [("H", "Proton"), ("C", "Carbon")]:
    corr = delta22.ablations_feature_correlations(query_df_dft, nucleus)
    nuc_label = "1H" if nucleus == "H" else "13C"
    nuc_title = "$^{1}$H" if nucleus == "H" else "$^{13}$C"
    composite_plots.plot_feature_correlation_heatmap(
        corr["r"], vmin=-1, vmax=1, cmap="RdBu",
        title=f"Solvent-Averaged Pearson $r$ Correlation Matrix for {nuc_title}",
        save_path=figure_path(f"ablations_feature_corr_r_{nuc_label}.png"))
plt.show()
"""),
    code(r"""
pcm_desmond_corr = delta22.pcm_desmond_correlation_by_solvent(query_df_dft)
print("PCM vs. Desmond Pearson R by nucleus and solvent:")
display(pcm_desmond_corr.round(3))
composite_plots.plot_pcm_desmond_correlation_table(pcm_desmond_corr,
                                   save_path=figure_path("ablations_pcm_desmond_corr_table.png"))
plt.show()
"""),
]


if __name__ == "__main__":
    if "--write" not in sys.argv:
        print("usage: python3 build_nb_composite_models.py --write")
        print("regenerates analysis/si_figures/si_composite_models_ablations.ipynb -- pass --write to actually do it,")
        print("since regenerating clears the notebook's execution outputs.")
        sys.exit(0)
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    _save(si_composite_models_ablations, os.path.join(repo, "analysis", "si_figures", "si_composite_models_ablations.ipynb"))
