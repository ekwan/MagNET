"""Source of truth for the MagNET-benchmark table notebooks (Tables S3/S4 and Table S5). Edit the
cell sources here, then regenerate and re-execute ONLY the notebook you touched, e.g.:

    python3 build_nb_magnet_benchmark.py si_table_s03_s04_performance
    jupyter nbconvert --to notebook --execute --inplace ../si_tables/si_table_s03_s04_performance.ipynb

    python3 build_nb_magnet_benchmark.py si_table_s05_qcd
    jupyter nbconvert --to notebook --execute --inplace ../si_tables/si_table_s05_qcd.ipynb

Regenerating overwrites the .ipynb (clearing its execution outputs), so do not hand-edit the
notebooks and regenerate only what you changed. All real code lives in magnet_benchmark.py (the
PUBLISHED reference values and the exact-reproduction group mapping); the notebooks only read
data/magnet_test_predictions/ through it. The notebooks live in analysis/si_tables/ (grouped by
role), not here.

Run with no arguments to list the available notebook names.
"""
import os
import sys

from nb_build import md, code, save_notebook as _save


_BOOTSTRAP = r"""
import os, sys

# make the in-repo modules importable (not pip-installed)
REPO = os.path.abspath("../..")
for _p in ("data/magnet_test_predictions", "analysis/code", "analysis/code/shared"):
    sys.path.insert(0, os.path.join(REPO, _p))
"""

_IMPORTS = r"""
import pandas as pd
import magnet_test_predictions_reader
import magnet_benchmark
import paths
"""

_SETUP = r"""
PREDICTIONS = paths.dataset_file("magnet_test_predictions", root=REPO)

def document_path(name):
    # table/spreadsheet outputs go under this notebook's documents/ folder, created on first save
    os.makedirs("documents", exist_ok=True)
    return os.path.join("documents", name)
"""

si_table_s03_s04_performance = [
    md(r"""
# Tables S3 and S4: MagNET performance across geometries

How accurately MagNET reproduces its DFT training reference (PBE0/pcSseg-1) as geometries move from
stationary to vibrated to solvated, for the foundation model and the chloroform MagNET-x.
"""),
    code(_BOOTSTRAP),
    code(_IMPORTS),
    code(_SETUP),
    code(r"""
rows = magnet_benchmark.exact_stats_table(PREDICTIONS, magnet_test_predictions_reader)
res = pd.DataFrame(rows).set_index(["nucleus", "model", "test_set"])

table_rows = []
for nucleus in ("1H", "13C"):
    for model in magnet_benchmark.MODELS:
        for ts in magnet_benchmark.TEST_SETS:
            r = res.loc[(nucleus, model, ts)]
            pmed, pmae, prmse = magnet_benchmark.PUBLISHED[nucleus][(model, ts)]
            table_rows.append(dict(nucleus=nucleus, model=model, test_set=ts, n=int(r.n),
                                   median_repro=round(r.median_ae, 6), median_SI=round(pmed, 6),
                                   mae_repro=round(r.mae, 6), mae_SI=round(pmae, 6),
                                   rmse_repro=round(r.rmse, 5), rmse_SI=round(prmse, 5)))
table = pd.DataFrame(table_rows)
table_s3 = table[table.nucleus == "1H"].drop(columns="nucleus")
table_s4 = table[table.nucleus == "13C"].drop(columns="nucleus")
print("Table S3 (1H):"); display(table_s3)
print("Table S4 (13C):"); display(table_s4)

# write the reproduced tables to this notebook's documents/ folder, one sheet per SI table
out = document_path("si_table_s03_s04_performance.xlsx")
with pd.ExcelWriter(out) as writer:
    table_s3.to_excel(writer, sheet_name="Table S3 (1H)", index=False)
    table_s4.to_excel(writer, sheet_name="Table S4 (13C)", index=False)
print("wrote", os.path.relpath(out, REPO))
"""),
    md("## Exact-reproduction check"),
    code(r"""
# every row's reproduced median/MAE/RMSE should match the published SI value to a few parts per million
max_dev = (table[["median_repro", "median_SI"]].diff(axis=1).iloc[:, -1].abs().max(),
          table[["mae_repro", "mae_SI"]].diff(axis=1).iloc[:, -1].abs().max(),
          table[["rmse_repro", "rmse_SI"]].diff(axis=1).iloc[:, -1].abs().max())
print("largest reproduced-vs-published deviation (median, MAE, RMSE):", max_dev)
assert max(max_dev) < 1e-3, "a row diverged from the SI by more than float rounding"
"""),
]


si_table_s05_qcd = [
    md(r"""
# Table S5: Performance statistics for predicting QCD corrections

Foundation MagNET's accuracy at predicting the rovibrational (QCD) correction (stationary vs
trajectory-averaged shielding) over qcdtraj2500 (2500 molecules), ¹H and ¹³C, all shieldings at
PBE0/pcSseg-1.
"""),
    code(_BOOTSTRAP),
    code(_IMPORTS),
    code(_SETUP),
    code(r"""
rows = magnet_benchmark.qcd_stats_table(PREDICTIONS, magnet_test_predictions_reader)
table_rows = []
for r in rows:
    nucleus = r["model"].split("(")[1].rstrip(")")   # "1H" / "13C"
    pmed, pmae, prmse = magnet_benchmark.PUBLISHED_S5[nucleus]
    table_rows.append(dict(model=r["model"], n=int(r["n"]),
                           median_repro=round(r["median_ae"], 8), median_SI=round(pmed, 8),
                           mae_repro=round(r["mae"], 8), mae_SI=round(pmae, 8),
                           rmse_repro=round(r["rmse"], 8), rmse_SI=round(prmse, 8)))
table_s5 = pd.DataFrame(table_rows)
print("Table S5 (QCD corrections):"); display(table_s5)

# write the reproduced table to this notebook's documents/ folder
out = document_path("si_table_s05_qcd.xlsx")
with pd.ExcelWriter(out) as writer:
    table_s5.to_excel(writer, sheet_name="Table S5", index=False)
print("wrote", os.path.relpath(out, REPO))
"""),
    md("## Exact-reproduction check"),
    code(r"""
# every reproduced median/MAE/RMSE should match the published SI value to a few parts per million
dev = max((table_s5[["median_repro", "median_SI"]].diff(axis=1).iloc[:, -1].abs().max(),
           table_s5[["mae_repro", "mae_SI"]].diff(axis=1).iloc[:, -1].abs().max(),
           table_s5[["rmse_repro", "rmse_SI"]].diff(axis=1).iloc[:, -1].abs().max()))
print("largest reproduced-vs-published deviation:", dev)
assert dev < 1e-3, "a row diverged from the SI by more than float rounding"
"""),
]


# name -> (cells, path relative to repo root)
NOTEBOOKS = {
    "si_table_s03_s04_performance": (si_table_s03_s04_performance,
                                     "analysis/si_tables/si_table_s03_s04_performance.ipynb"),
    "si_table_s05_qcd": (si_table_s05_qcd, "analysis/si_tables/si_table_s05_qcd.ipynb"),
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
