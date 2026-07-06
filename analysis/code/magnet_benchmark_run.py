"""Produce the magnet_benchmark results table by running MagNET and MagNET-x over the four test sets
and write it to performance_results.csv.

Needs the checkpoints and datasets, so it does not run in CI; they ship in the repo (Git LFS via the
Hugging Face checkout) and resolve automatically. One row per (model, test set, nucleus) with median
AE, MAE, RMSE, and atom count. The three vibrated/stationary sets are sampled; isolated-chloroform is
run in full. The shipped CSV used --sample 500 --seed 0; re-running reproduces it (sampled rows within
tolerance, isolated rows exactly).

Usage:
    python magnet_benchmark_run.py --out performance_results.csv --n-passes 1 --sample 500 --seed 0
"""
import argparse
import csv
import os
import sys
import time

from paths import ensure_on_path, repo_root, dataset_file

HERE = os.path.dirname(os.path.abspath(__file__))
GH = repo_root(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ensure_on_path("analysis", "code", "shared", file=__file__)  # stats.py etc. live here
ensure_on_path("magnet", file=__file__)
for _d in ("sigma-shake", "gdb_qcd", "dft8k", "sigma-fresh"):
    ensure_on_path("data", _d, file=__file__)

import magnet_benchmark as M  # noqa: E402
from run_magnet import compute_MagNET_foundation_shieldings, _predict_with  # noqa: E402


def magnet_x_chloroform_shieldings(ans, geos, n_passes, device):
    """Absolute shieldings from the MagNET-x chloroform model (run as a plain predictor, no solvent),
    used to score MagNET-x on the same test sets as the foundation model."""
    return _predict_with("MagNET-x-chloroform_H", "MagNET-x-chloroform_C", ans, geos,
                         n_passes=n_passes, device=device)


MODEL_RUNNERS = {
    "MagNET": compute_MagNET_foundation_shieldings,
    "MagNET-x": magnet_x_chloroform_shieldings,
}


def assemble_all(paths, sample, seed):
    """Assemble the four test sets, returning {test_set: (ans, geos, dfts)}."""
    from decode_sigma_shake import SigmaShake
    from decode_gdb_qcd import GDBQCD
    from dft8k_reader import DFT8k
    from decode_sigma_fresh import SigmaFresh

    raw = {}
    with SigmaShake(paths["sigma_shake"]) as ss:
        raw["stationary_internal"] = M.cases_stationary_internal(ss, n_sample=sample, seed=seed)
    with GDBQCD(paths["gdb_qcd"]) as gq:
        raw["vibrated_internal"] = M.cases_vibrated_internal(gq, n_sample=sample // 3, seed=seed)
    with DFT8k(paths["dft8k"]) as d8:
        raw["vibrated_external"] = M.cases_vibrated_external(d8, n_sample=sample, seed=seed)
    with SigmaFresh(paths["sigma_fresh"]) as sf:
        # the full published test set (held-out solutes, frames 1..10 with an isolated calculation);
        # not sampled, so this row reproduces Tables S3/S4 exactly
        raw["isolated_chloroform"] = M.cases_isolated_chloroform(sf, scrub=False)
    # exclude molecules with elements outside MagNET's vocabulary (e.g. phosphorus in dft8k); the
    # model would refuse them anyway, and scoring them would not be a fair test
    cases = {}
    for ts, (ans, geos, dfts) in raw.items():
        a, g, d, dropped = M.filter_supported(ans, geos, dfts)
        if dropped:
            print(f"  {ts}: dropped {dropped} structure(s) with unsupported elements", flush=True)
        cases[ts] = (a, g, d)
    return cases


def main():
    """Assemble the four test sets, run both models over them, and write performance_results.csv."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "data", "magnet_benchmark", "performance_results.csv"))
    ap.add_argument("--n-passes", type=int, default=1)
    ap.add_argument("--sample", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sigma-shake", default=dataset_file("sigma-shake", root=GH))
    ap.add_argument("--gdb-qcd", default=dataset_file("gdb_qcd", root=GH))
    ap.add_argument("--dft8k", default=dataset_file("dft8k", root=GH))
    ap.add_argument("--sigma-fresh", default=dataset_file("sigma-fresh", root=GH))
    args = ap.parse_args()

    paths = {"sigma_shake": args.sigma_shake, "gdb_qcd": args.gdb_qcd,
             "dft8k": args.dft8k, "sigma_fresh": args.sigma_fresh}
    print(f"assembling test cases (sample={args.sample}, seed={args.seed})...", flush=True)
    cases = assemble_all(paths, args.sample, args.seed)
    for ts, (ans, _, _) in cases.items():
        print(f"  {ts}: {len(ans)} structures", flush=True)

    rows = []
    for model, runner in MODEL_RUNNERS.items():
        for ts in M.TEST_SETS:
            ans, geos, dfts = cases[ts]
            t = time.time()
            pred = runner(ans, geos, args.n_passes, args.device)
            tab = M.error_table(pred, dfts, ans)
            dt = time.time() - t
            for nucleus in ("1H", "13C"):
                s = tab[nucleus]
                rows.append(dict(model=model, test_set=ts, nucleus=nucleus,
                                 median_ae=s["median_ae"], mae=s["mae"], rmse=s["rmse"], n=s["n"]))
            print(f"{model:9s} {ts:20s} ({dt:.0f}s)  "
                  f"1H mae {tab['1H']['mae']:.4f}  13C mae {tab['13C']['mae']:.4f}", flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "test_set", "nucleus", "median_ae", "mae", "rmse", "n"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
