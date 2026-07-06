"""Source of truth for the Tables S10/S11 notebook (MagNET-Zero/MagNET-PCM scaling factors). Edit
the cell sources here, then regenerate:

    python3 build_nb_scaling.py si_table_s10_s11_scaling
    jupyter nbconvert --to notebook --execute --inplace ../si_tables/si_table_s10_s11_scaling.ipynb

Regenerating overwrites the .ipynb (clearing its execution outputs). All real code lives in
scaling_factors.py (the published tables and the re-derivation from delta-22); the notebook only
runs it. The notebook lives in analysis/si_tables/ (grouped by role), not here.

Run with no arguments to list the available notebook names.
"""
import os
import sys

from nb_build import md, code, save_notebook as _save


_BOOTSTRAP = r"""
import os, sys

# make the in-repo modules importable (not pip-installed)
REPO = os.path.abspath("../..")
for _p in ("data/delta22", "data/scaling_factors", "data/applications",
           "analysis/code", "analysis/code/shared"):
    sys.path.insert(0, os.path.join(REPO, _p))
"""

_IMPORTS = r"""
import numpy as np
import pandas as pd
import scaling_factors
import scaling_factors_reader
import build_composite_model
import paths
from applications_reader import Applications
"""

_SETUP = r"""
DELTA22 = paths.dataset_file("delta22", root=REPO)
XLSX = os.path.join(REPO, "data", "delta22", "delta22_experimental.xlsx")

def document_path(name):
    os.makedirs("documents", exist_ok=True)
    return os.path.join("documents", name)
"""

si_table_s10_s11_scaling = [
    md(r"""
# Tables S10 and S11: MagNET-Zero/MagNET-PCM scaling factors

Per-solvent linear coefficients (intercept, stationary, pcm) mapping MagNET-Zero shieldings plus a
MagNET-PCM correction to predicted shifts: ¹H (S10) and ¹³C (S11).
"""),
    code(_BOOTSTRAP),
    code(_IMPORTS),
    code(_SETUP),
    code(r"""
tables = scaling_factors.published_scaling_tables()   # {"H": Table S10, "C": Table S11}
print("Table S10 (1H):"); display(tables["H"])
print("Table S11 (13C):"); display(tables["C"])

out = document_path("si_table_s10_s11_scaling.xlsx")
with pd.ExcelWriter(out) as writer:
    tables["H"].reset_index().to_excel(writer, sheet_name="Table S10 (1H)", index=False)
    tables["C"].reset_index().to_excel(writer, sheet_name="Table S11 (13C)", index=False)
print("wrote", os.path.relpath(out, REPO))
"""),
    md("## Reproducibility check: re-derive from delta-22"),
    code(r"""
derived = scaling_factors.build_scaling_tables(DELTA22, XLSX)
for nucleus in ("H", "C"):
    p = tables[nucleus]
    d = derived[nucleus].reindex(p.index)[p.columns]
    max_dev = float(np.abs(p.values - d.values).max())
    print(f"{nucleus}: largest published-vs-rederived deviation = {max_dev:.2e}")
    assert max_dev < 1e-5, f"{nucleus} scaling table diverged from the published values"
print("both tables reproduce from delta-22")
"""),
    md(r"""
## Deployment tables with reflection symmetrization

The tables above match the published SI exactly. For serving new molecules, the recommended tables
average each prediction with its mirror image, correcting a reflection-parity error. Reproducing
them needs the model checkpoints; the shipped tables are shown below.
"""),
    code(r"""
symmetrized = scaling_factors_reader.load_symmetrized_tables()
print("Symmetrized deployment Table S10 (1H):"); display(symmetrized["H"])
print("Symmetrized deployment Table S11 (13C):"); display(symmetrized["C"])
"""),
    md(r"""
## Composite-model coefficients (Figure 5C/5D, SI S15)

The natural-products figures use a larger family of per-solvent coefficients fit on delta-22: OLS
fits, 1000-seed bootstrap resamples, their RMSE distributions, and the PCM conversion factors. These
regenerate from released inputs alone.
"""),
    code(r"""
reader = Applications(paths.dataset_file("applications", root=REPO))
regenerated = build_composite_model.regenerate(reader)
max_dev = build_composite_model.verify(reader, regenerated)
print(f"composite_model largest regenerated-vs-stored deviation = {max_dev:.2e}")
"""),
]


# name -> (cells, path relative to repo root)
NOTEBOOKS = {
    "si_table_s10_s11_scaling": (si_table_s10_s11_scaling,
                                 "analysis/si_tables/si_table_s10_s11_scaling.ipynb"),
}

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    names = sys.argv[1:]
    if not names:
        print("available names:", ", ".join(NOTEBOOKS))
        sys.exit(0)
    unknown = [n for n in names if n not in NOTEBOOKS]
    if unknown:
        raise SystemExit(f"unknown notebook name(s): {unknown}; available: {', '.join(NOTEBOOKS)}")
    for name in names:
        cells, relpath = NOTEBOOKS[name]
        _save(cells, os.path.join(repo, relpath))
