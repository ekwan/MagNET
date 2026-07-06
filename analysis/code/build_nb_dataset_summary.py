"""Source of truth for the Table S8 notebook (Dataset Summary Statistics). Edit the cell sources
here, then regenerate:

    python3 build_nb_dataset_summary.py si_table_s08_summary
    jupyter nbconvert --to notebook --execute --inplace ../si_tables/si_table_s08_summary.ipynb

Regenerating overwrites the .ipynb (clearing its execution outputs). All real code lives in
dataset_summary.py (the counting logic and the PUBLISHED_S8 reference values); the notebook only
runs it over the released data/ files. The notebook lives in analysis/si_tables/ (grouped by role), not here.

Run with no arguments to list the available notebook names.
"""
import os
import sys

from nb_build import md, code, save_notebook as _save


_BOOTSTRAP = r"""
import os, sys

# make the in-repo modules importable (not pip-installed)
REPO = os.path.abspath("../..")
for _p in ("analysis/code", "analysis/code/shared"):
    sys.path.insert(0, os.path.join(REPO, _p))
"""

_IMPORTS = r"""
import pandas as pd
import dataset_summary
"""

_SETUP = r"""
DATA_DIR = os.path.join(REPO, "data")

def document_path(name):
    os.makedirs("documents", exist_ok=True)
    return os.path.join("documents", name)
"""

si_table_s08_summary = [
    md(r"""
# Table S8: Dataset Summary Statistics

Molecule and ¹H/¹³C site counts for each training dataset (site counts from each HDF5's
`atomic_numbers`, ¹H=1/¹³C=6; MagNET-Zero combines both sigma-pepper rounds with sigma-concentrate).
"""),
    code(_BOOTSTRAP),
    code(_IMPORTS),
    code(_SETUP),
    code(r"""
table_s8 = dataset_summary.summary_table(DATA_DIR)
display(table_s8)

# write the table to this notebook's documents/ folder
out = document_path("si_table_s08_summary.xlsx")
with pd.ExcelWriter(out) as writer:
    table_s8.to_excel(writer, sheet_name="Table S8", index=False)
print("wrote", os.path.relpath(out, REPO))
"""),
    md("## Exact-reproduction check"),
    code(r"""
# every count should match the published SI Table S8 value exactly
for _, row in table_s8.iterrows():
    pub = dataset_summary.PUBLISHED_S8[row["dataset"]]
    got = (row["molecules"], row["n_1H_sites"], row["n_13C_sites"])
    assert got == pub, f"{row['dataset']}: {got} != published {pub}"
print("all rows match the published SI Table S8 exactly")
"""),
]


# name -> (cells, path relative to repo root)
NOTEBOOKS = {
    "si_table_s08_summary": (si_table_s08_summary, "analysis/si_tables/si_table_s08_summary.ipynb"),
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
