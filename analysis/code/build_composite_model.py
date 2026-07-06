"""Regenerate the composite_model/ group in applications.hdf5 from the shipped delta-22 data.

Replaces the old build_coefficients.py, which copied pre-fit CSVs out of a private tree via
environment variables. Everything here is fit in-repo from delta22.hdf5, so the natural-products
composite-model coefficients (Figure 5C/5D, SI S15) are reproducible from released inputs alone.

The stored group's structure (formula keys, parameter order, solvent order, seed count) is read
from the current applications.hdf5 and used as a template; only the numbers are regenerated.
"""
import os
import random
import sys

import h5py
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in ("analysis/code", "analysis/code/shared", "data/delta22", "data/applications"):
    sys.path.insert(0, os.path.join(REPO, _p))

import delta22 as D           # noqa: E402
import paths                  # noqa: E402

# The 22 delta-22 solutes in first-appearance order: the universe random.choices() draws from.
# nitromethane stays in the draw pool (index 14) but its rows are dropped from the fit data, exactly
# as the original pipeline did.
SOLUTE_ORDER = ["AcOH", "EtOAc", "EtOH", "THF", "acetone", "acetonitrile", "benzaldehyde", "dmf",
                "dmso", "ethane", "ethylene", "glycol", "imidazole", "isopropanol", "nitromethane",
                "propylene", "pyridine", "pyrrole", "pyrrolidine", "tbme", "toluene", "triethylamine"]
SEED_OFFSET = 100
BASE_FEATURES = ["experimental", "stationary", "qcd", "pcm", "desmond", "desmond_vib",
                 "openMM", "openMM_vib"]
ALIASES = {"des_vib": "desmond_vib", "op_vib": "openMM_vib"}


def _load_tables():
    """Three delta-22 NN tables (nitromethane rows removed) plus the pcm conversion factors:

      full_tab  - every row; the OLS point fits use it and drop only each formula's own missing
                  features, so pcm/qcd formulas fit all 12 solvents.
      fit_tab   - strict: all base features present (only the 4 explicit-solvent solvents survive);
                  the bootstrap resamples fit here, matching the stored bootstrap's 4 solvents.
      apply_tab - desmond or openMM present; the RMSE predictions are evaluated over it.
    """
    h5 = paths.dataset_file("delta22", root=REPO)
    xlsx = os.path.join(REPO, "data", "delta22", "delta22_experimental.xlsx")
    nn = D.load_query_df_nn(h5, xlsx, verbose=False)
    dft = D.load_query_df_dft(h5, xlsx, verbose=False)
    factors = {"H": D.pcm_conversion_factors(dft, "H"), "C": D.pcm_conversion_factors(dft, "C")}
    nn = nn[nn["solute"] != "nitromethane"].copy()
    # stationary_plus_pcm folds in the per-solvent conversion factor; every other _plus_ column is a
    # plain sum, built on demand by _ensure_column.
    fac = nn.apply(lambda r: factors[r["nucleus"]].get(r["solvent"], np.nan), axis=1)
    nn["stationary_plus_pcm"] = nn["stationary"] + nn["pcm"] * fac
    full_tab = nn.reset_index(drop=True)
    fit_tab = nn.dropna(subset=BASE_FEATURES).reset_index(drop=True)
    apply_tab = nn.dropna(subset=["desmond", "openMM"], how="all").reset_index(drop=True)
    return full_tab, fit_tab, apply_tab, factors


def _canonical(name):
    return ALIASES.get(name, name)


def _regressor_columns(formula_key):
    """The design-matrix regressors for a formula key. '_AND_' separates independent regressors;
    a '_plus_' name is one precomputed sum column. Returns the list of column names to fit on."""
    return [g for g in formula_key.split("_AND_")]


def _plus_parts(regressor):
    """Base features a '_plus_' regressor decomposes into (canonical names); [] if it is itself a
    base feature."""
    if "_plus_" not in regressor:
        return []
    return [_canonical(p) for p in regressor.split("_plus_") if p]


def _ensure_column(df, regressor):
    """Make sure a precomputed _plus_ regressor column exists on df (sum of its parts; the pcm part
    already carries the conversion factor via stationary_plus_pcm)."""
    if regressor in df.columns:
        return
    parts = regressor.split("_plus_")
    assert _canonical(parts[0]) == "stationary", f"unexpected base term in {regressor}"
    # build cumulatively so stationary_plus_pcm_plus_op_vib = stationary_plus_pcm + openMM_vib. The
    # pcm chains start from the scaled stationary_plus_pcm column; everything else from raw stationary.
    if parts[1:2] == ["pcm"]:
        col = df["stationary_plus_pcm"].to_numpy(float).copy()
        rest = parts[2:]
    else:
        col = df["stationary"].to_numpy(float).copy()
        rest = parts[1:]
    for p in rest:
        col = col + df[_canonical(p)].to_numpy(float)
    df[regressor] = col


def _fit(df, regressors):
    """OLS of experimental on the given regressor columns (with intercept); returns {name: coeff}
    keyed by 'Intercept' and each regressor. NaN rows are dropped, matching statsmodels."""
    y = pd.to_numeric(df["experimental"], errors="coerce").to_numpy(float)
    cols = [df[r].to_numpy(float) for r in regressors]
    X = np.column_stack([np.ones(len(df))] + cols)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    beta, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
    return dict(zip(["Intercept"] + list(regressors), beta))


def _decompose(coeffs, regressors, solvent, nucleus, factors):
    """Expand each _plus_ regressor's coefficient onto its base features (same value copied). The pcm
    row is scaled by the per-solvent conversion factor ONLY when the formula used a
    stationary_plus_pcm term (a free _AND_pcm regressor stays raw). Returns {base_feature: coeff}."""
    out = {"intercept": coeffs["Intercept"]}
    for reg, val in coeffs.items():
        if reg == "Intercept":
            continue
        parts = _plus_parts(reg)
        for base in (parts if parts else [_canonical(reg)]):
            out[base] = val
    uses_spp = any(_plus_parts(r)[1:2] == ["pcm"] for r in regressors)
    if uses_spp and "pcm" in out:
        out["pcm"] = out["pcm"] * float(factors[nucleus].get(solvent, np.nan))
    return out


def _bootstrap_sample(fit_tab, nucleus, solvent, seed):
    """The rows fit for one bootstrap draw: seed the RNG with seed+100, draw 22 solutes with
    replacement from SOLUTE_ORDER, keep that solvent/nucleus's rows for the drawn solutes."""
    random.seed(int(seed) + SEED_OFFSET)
    drawn = random.choices(SOLUTE_ORDER, k=22)
    return fit_tab[(fit_tab["nucleus"] == nucleus) & (fit_tab["solvent"] == solvent)
                   & (fit_tab["solute"].isin(drawn))]


def _rmse(fit_tab, apply_tab, display_formula, nucleus, solvent, seed, factors):
    """Bootstrap RMSE: fit the formula on one bootstrap draw, apply those raw regressor coefficients
    to every apply-table row for the solvent, and take the RMSE over labeled rows. The RMSE display
    names join regressors with ' + ' (each is a column, possibly a precomputed _plus_ one)."""
    regs = [t.strip() for t in display_formula.split(" + ")]
    for r in (fit_tab, apply_tab):
        for reg in regs:
            _ensure_column(r, reg)
    coeffs = _fit(_bootstrap_sample(fit_tab, nucleus, solvent, seed), regs)
    a = apply_tab[(apply_tab["nucleus"] == nucleus) & (apply_tab["solvent"] == solvent)]
    pred = coeffs["Intercept"] + sum(coeffs[r] * a[r].to_numpy(float) for r in regs)
    exp = pd.to_numeric(a["experimental"], errors="coerce").to_numpy(float)
    mask = np.isfinite(pred) & np.isfinite(exp)
    return float(np.sqrt(np.mean((pred[mask] - exp[mask]) ** 2)))


def _pcm_table(factors, nucleus):
    """pcm_conversion_factors subgroup for one nucleus: solvent, pcm_conversion_factor. Stored under
    the 'water' name rather than the loader's 'TIP4P'."""
    s = factors[nucleus].rename(index={"TIP4P": "water"})
    return pd.DataFrame({"solvent": s.index, "pcm_conversion_factor": s.to_numpy()})


def regenerate(reader):
    """Regenerate every composite_model subgroup from delta-22, keyed and ordered to match the stored
    group. Returns {"ols_coefficients"|"bootstrap_coefficients"|"rmse_distributions"|
    "pcm_conversion_factors": {key: DataFrame}}."""
    full_tab, fit_tab, apply_tab, factors = _load_tables()
    out = {k: {} for k in ("ols_coefficients", "bootstrap_coefficients",
                           "rmse_distributions", "pcm_conversion_factors")}

    for f in reader.composite_formulas("ols_coefficients"):
        regs = _regressor_columns(f)
        for reg in regs:
            _ensure_column(full_tab, reg)
        for nuc in ("H", "C"):
            try:
                template = reader.ols_coefficients(f, nuc)
            except KeyError:
                continue
            params = template["parameter"].tolist()
            solvents = [c for c in template.columns if c != "parameter"]
            data = {"parameter": params}
            for solvent in solvents:
                sub = full_tab[(full_tab["nucleus"] == nuc) & (full_tab["solvent"] == solvent)]
                dec = _decompose(_fit(sub, regs), regs, solvent, nuc, factors)
                data[solvent] = [dec[p] for p in params]
            out["ols_coefficients"][(f, nuc)] = pd.DataFrame(data)

    for f in reader.composite_formulas("bootstrap_coefficients"):
        regs = _regressor_columns(f)
        for reg in regs:
            _ensure_column(fit_tab, reg)
        for nuc in ("H", "C"):
            try:
                template = reader.bootstrap_coefficients(f, nuc)
            except KeyError:
                continue
            params = [c for c in template.columns if c not in ("solvent", "seed")]
            rows = []
            for solvent in dict.fromkeys(template["solvent"]):
                seeds = sorted(template[template["solvent"] == solvent]["seed"].unique())
                for seed in seeds:
                    dec = _decompose(_fit(_bootstrap_sample(fit_tab, nuc, solvent, seed), regs),
                                     regs, solvent, nuc, factors)
                    row = {"solvent": solvent, "seed": seed}
                    for p in params:
                        row[p] = dec["intercept"] if p == "Intercept" else dec[p]
                    rows.append(row)
            out["bootstrap_coefficients"][(f, nuc)] = pd.DataFrame(rows)[template.columns.tolist()]

    for nuc in ("H", "C"):
        template = reader.rmse_distribution(nuc)
        rows = []
        for f in dict.fromkeys(template["formula"]):
            sub = template[template["formula"] == f]
            for solvent in dict.fromkeys(sub["solvent"]):
                for seed in sorted(sub[sub["solvent"] == solvent]["seed"].unique()):
                    rows.append({"solvent": solvent, "nucleus": nuc, "formula": f, "seed": seed,
                                 "Bootstrap_RMSE": _rmse(fit_tab, apply_tab, f, nuc, solvent, seed, factors)})
        out["rmse_distributions"][nuc] = pd.DataFrame(rows)[template.columns.tolist()]
        out["pcm_conversion_factors"][nuc] = _pcm_table(factors, nuc)

    return out


def verify(reader, regenerated, atol=5e-3):
    """Compare regenerated numeric values against the stored group. Returns the global max abs diff.
    The floor is the fixed-point encoding granularity (~1e-3), far below any recipe error."""
    worst = 0.0
    for (f, nuc), df in regenerated["ols_coefficients"].items():
        st = reader.ols_coefficients(f, nuc)
        worst = max(worst, np.abs(df.drop(columns="parameter").to_numpy(float)
                                  - st.drop(columns="parameter").to_numpy(float)).max())
    for (f, nuc), df in regenerated["bootstrap_coefficients"].items():
        st = reader.bootstrap_coefficients(f, nuc).sort_values(["solvent", "seed"]).reset_index(drop=True)
        df = df.sort_values(["solvent", "seed"]).reset_index(drop=True)
        num = [c for c in df.columns if c not in ("solvent",)]
        worst = max(worst, np.abs(df[num].to_numpy(float) - st[num].to_numpy(float)).max())
    for nuc, df in regenerated["rmse_distributions"].items():
        st = reader.rmse_distribution(nuc).sort_values(["formula", "solvent", "seed"]).reset_index(drop=True)
        df = df.sort_values(["formula", "solvent", "seed"]).reset_index(drop=True)
        worst = max(worst, np.abs(df["Bootstrap_RMSE"].to_numpy(float)
                                  - st["Bootstrap_RMSE"].to_numpy(float)).max())
    for nuc, df in regenerated["pcm_conversion_factors"].items():
        st = reader.pcm_conversion_factors(nuc).set_index("solvent")["pcm_conversion_factor"]
        got = df.set_index("solvent")["pcm_conversion_factor"]
        worst = max(worst, np.abs(got - st.reindex(got.index)).max())
    return float(worst)


_NOTE = ("Linear composite-model coefficients fit on the delta-22 dataset and applied to the "
         "natural products (Figure 5C/5D, SI S15). Stored as CSV text; read via applications_reader. "
         "Regenerated in-repo from delta22.hdf5 by build_composite_model.py.")


def write(hdf5_path, regenerated):
    """Overwrite the composite_model/ group in applications.hdf5 with the regenerated CSV text."""
    with h5py.File(hdf5_path, "a") as h:
        if "composite_model" in h:
            del h["composite_model"]
        cm = h.create_group("composite_model")
        cm.attrs["note"] = _NOTE
        for (formula, nucleus), df in regenerated["ols_coefficients"].items():
            cm.require_group(f"ols_coefficients/{formula}").create_dataset(nucleus, data=df.to_csv(index=False))
        for (formula, nucleus), df in regenerated["bootstrap_coefficients"].items():
            cm.require_group(f"bootstrap_coefficients/{formula}").create_dataset(nucleus, data=df.to_csv(index=False))
        for nucleus, df in regenerated["rmse_distributions"].items():
            cm.require_group("rmse_distributions").create_dataset(nucleus, data=df.to_csv(index=False))
        for nucleus, df in regenerated["pcm_conversion_factors"].items():
            cm.require_group("pcm_conversion_factors").create_dataset(nucleus, data=df.to_csv(index=False))


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Regenerate composite_model/ in applications.hdf5 from delta-22.")
    ap.add_argument("--write", action="store_true",
                    help="overwrite the composite_model group in applications.hdf5 (default: verify only)")
    args = ap.parse_args()
    sys.path.insert(0, os.path.join(REPO, "data", "applications"))
    from applications_reader import Applications
    hdf5_path = paths.dataset_file("applications", root=REPO)
    reader = Applications(hdf5_path)
    regenerated = regenerate(reader)
    max_diff = verify(reader, regenerated)
    print(f"composite_model regenerated from delta-22; max abs diff vs stored = {max_diff:.2e}")
    if args.write:
        write(hdf5_path, regenerated)
        print(f"wrote composite_model/ into {hdf5_path}")


if __name__ == "__main__":
    main()
