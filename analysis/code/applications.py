"""Core analysis for the natural-products ("applications") figures (Figures 5C/5D, SI S8, S15). No plotting.

Takes an Applications loader (data/applications/applications_reader.py) and computes the numbers the
figure notebooks and tests share. A "composite model" predicts a shift as an OLS-weighted sum of
physics terms (gas-phase shielding + implicit correction + rovibrational correction); these figures
test how well weights fit on delta-22 carry to the natural-products test set.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from stats import rmse as _rmse
from spreadsheet import site_atom_indices
from applications_reader import SOLVENTS, SHELL_SIZES

# composite-model formula sets (column names refer to build_query_df_nn output)
FITTING_FORMULAS_H = [
    "stationary", "stationary_plus_pcm", "stationary + pcm", "stationary + pcm + qcd",
    "stationary_plus_qcd + pcm", "stationary_plus_pcm_plus_qcd", "stationary + openMM",
    "stationary + openMM + qcd", "stationary_plus_qcd + openMM", "stationary_plus_openMM_plus_qcd",
]
FITTING_FORMULAS_C = [
    "stationary", "stationary_plus_pcm", "stationary + pcm", "stationary + pcm + openMM_vib",
    "stationary_plus_op_vib + pcm", "stationary_plus_pcm_plus_op_vib", "stationary + openMM",
    "stationary + openMM + openMM_vib", "stationary_plus_op_vib + openMM", "stationary_plus_openMM_plus_op_vib",
]
FORMULA_REMAP = {
    "stationary_plus_pcm": "pcm2", "stationary + pcm": "pcm3", "stationary + pcm + qcd": "pcm4",
    "stationary + pcm + openMM_vib": "pcm4", "stationary_plus_qcd + pcm": "semi_parsimonious_pcm",
    "stationary_plus_op_vib + pcm": "semi_parsimonious_pcm", "stationary_plus_pcm_plus_qcd": "parsimonious_pcm",
    "stationary_plus_pcm_plus_op_vib": "parsimonious_pcm", "stationary + openMM": "expl3",
    "stationary + openMM + qcd": "expl4", "stationary + openMM + openMM_vib": "expl4",
    "stationary_plus_qcd + openMM": "semi_parsimonious_expl", "stationary_plus_op_vib + openMM": "semi_parsimonious_expl",
    "stationary_plus_openMM_plus_qcd": "parsimonious_expl", "stationary_plus_openMM_plus_op_vib": "parsimonious_expl",
}
# manuscript test-set membership, keyed by the canonical solute names build_query_df_nn emits
ALL_IN_ONE_BIN = {"Test Set": [
    "isomer_1E", "isomer_1Z", "isomer_2E", "isomer_2Z", "isomer_3E", "isomer_3Z", "isomer_4N", "isomer_4O",
    "vomicine", "prednisone", "peptide", "flavone", "dihydrotanshinone_I",
]}
# canonical solute -> manuscript display label (used as figure x-tick labels)
SOLUTE_DISPLAY = {
    "isomer_1E": "Isomer 1 (E)", "isomer_1Z": "Isomer 1 (Z)", "isomer_2E": "Isomer 2 (E)",
    "isomer_2Z": "Isomer 2 (Z)", "isomer_3E": "Isomer 3 (E)", "isomer_3Z": "Isomer 3 (Z)",
    "isomer_4N": "1,4-dimethylpyridin-\n2(1H)one", "isomer_4O": "2-methoxy-\n4-methylpyridine",
    "vomicine": "Vomicine", "prednisone": "Prednisone", "peptide": "Acetyl-L-alanyl-L-\nglutamine",
    "flavone": "Flavone", "dihydrotanshinone_I": "Dihydrotanshinone I",
}
# the peptide (and, for carbon, the pyridone isomer_4N) is highlighted green as the high-error outlier
PEPTIDE_HIGHLIGHT_H = {"peptide": "#B0FF40"}
PEPTIDE_HIGHLIGHT_C = {"isomer_4N": "#B0FF40", "peptide": "#B0FF40"}

# bootstrap-coefficient file token -> remapped formula name, per nucleus
_BOOTSTRAP_TOKENS = {
    "H": {"stationary_plus_pcm": "pcm2", "stationary_AND_pcm_AND_qcd": "pcm4",
          "stationary_plus_qcd_AND_pcm": "semi_parsimonious_pcm", "stationary_plus_pcm_plus_qcd": "parsimonious_pcm",
          "stationary_AND_openMM_AND_qcd": "expl4", "stationary_plus_qcd_AND_openMM": "semi_parsimonious_expl",
          "stationary_plus_openMM_plus_qcd": "parsimonious_expl"},
    "C": {"stationary_plus_pcm": "pcm2", "stationary_AND_pcm_AND_openMM_vib": "pcm4",
          "stationary_plus_op_vib_AND_pcm": "semi_parsimonious_pcm", "stationary_plus_pcm_plus_op_vib": "parsimonious_pcm",
          "stationary_AND_openMM_AND_openMM_vib": "expl4", "stationary_plus_op_vib_AND_openMM": "semi_parsimonious_expl",
          "stationary_plus_openMM_plus_op_vib": "parsimonious_expl"},
}


def shell_convergence_corrections(loader, solute, nucleus, solvents=SOLVENTS):
    """MagNET-X explicit-solvent correction (solvated minus isolated shielding) for each NMR
    site, averaged over the site's atoms and over the MD frames, as a function of solvent shell
    size. Backs SI Figure S8 panel A (shown for vomicine, proton sites).

    Returns a DataFrame indexed by (site, solvent) with one column per shell ("shell_50" ...
    "shell_650"). Shells absent for a given solute/solvent (e.g. the isomer_1Z/chloroform/250
    gap) come back as NaN.
    """
    sites = loader.site_atom_table(uuid_prefix_h_sites=True)
    sites = sites[(sites["solute"] == solute) & (sites["nucleus"] == nucleus)]
    rows = []
    for _, r in sites.iterrows():
        idx = site_atom_indices(r["atom_numbers"])
        for solvent in solvents:
            present = set(loader.available_shells(solute, solvent))
            vals = {"site": r["site"], "solvent": solvent}
            for shell in SHELL_SIZES:
                if shell not in present:
                    vals[f"shell_{shell}"] = np.nan
                    continue
                mx = loader.magnet_x(solute, solvent, shell)          # (frames, atoms, 2)
                corr = mx[:, idx, 1] - mx[:, idx, 0]                  # solvated - isolated
                corr = np.mean(corr, axis=1)                          # average over the site's atoms
                corr = corr[~np.isnan(corr)]                          # drop frames with missing data
                vals[f"shell_{shell}"] = float(np.mean(corr)) if len(corr) else np.nan
            rows.append(vals)
    return pd.DataFrame(rows).set_index(["site", "solvent"])


def experimental_stack(loader):
    """Experimental shifts as a long DataFrame indexed (solute, nucleus, site, solvent)."""
    exp = loader.experiment(uuid_prefix_h_sites=True).drop(columns=["atom_numbers"])
    exp = exp.set_index(["solute", "nucleus", "site"])
    stack = exp.stack(future_stack=True)
    stack = pd.DataFrame(stack, columns=["experimental"])
    stack.index.names = ["solute", "nucleus", "site", "solvent"]
    return stack.sort_index()


def _site_mean(arr, idx):
    return float(np.asarray(arr)[idx].mean())


def build_query_df_nn(loader):
    """Assemble the per-(solute, nucleus, site, solvent) NN feature table that the composite-model
    figures fit and plot (Figure 5C/5D, SI S15). Columns: experimental, stationary, pcm, qcd,
    openMM, openMM_vib, plus the composite columns added by add_composite_columns
    (stationary_plus_pcm/_qcd/_op_vib and their pcm+/openMM+ combinations). All inputs (MagNET
    shieldings and the PCM conversion factors) are read through the loader.
    """
    sites = loader.site_atom_table(uuid_prefix_h_sites=True)
    solvents = [c for c in loader.experiment().columns
                if c not in ("solute", "site", "nucleus", "atom_numbers")]

    # Cache the per-solute / per-(solute, solvent) loader reads. Each accessor opens the HDF5 file
    # fresh, and the loops below touch only ~13 solutes (or ~52 solute-solvent pairs) across
    # hundreds of sites, so without caching this re-opens the file thousands of times (~12x slower).
    _mz_cache, _qcd_cache, _shell_cache, _mx_cache, _mxs_cache = {}, {}, {}, {}, {}
    def get_mz(solute):
        if solute not in _mz_cache: _mz_cache[solute] = loader.magnet_zero(solute)
        return _mz_cache[solute]
    def get_qcd(solute):
        if solute not in _qcd_cache: _qcd_cache[solute] = loader.qcd(solute)
        return _qcd_cache[solute]
    def get_shells(solute, solvent):
        k = (solute, solvent)
        if k not in _shell_cache: _shell_cache[k] = loader.available_shells(solute, solvent)
        return _shell_cache[k]
    def get_mx(solute, solvent):
        k = (solute, solvent)
        if k not in _mx_cache: _mx_cache[k] = loader.magnet_x(solute, solvent, 650)
        return _mx_cache[k]
    def get_mxs(solute, solvent):
        k = (solute, solvent)
        if k not in _mxs_cache: _mxs_cache[k] = loader.magnet_x_stationary(solute, solvent)
        return _mxs_cache[k]

    # --- MagNET-Zero stationary + PCM (PCM predicted for chloroform, broadcast to all solvents) ---
    mz_rows = []
    for _, r in sites.iterrows():
        idx = site_atom_indices(r["atom_numbers"])
        mz = get_mz(r["solute"])
        mz_rows.append([r["solute"], r["nucleus"], r["site"],
                        _site_mean(mz["stationary"], idx), _site_mean(mz["pcm_correction"], idx)])
    mz = pd.DataFrame(mz_rows, columns=["solute", "nucleus", "site", "stationary", "pcm"])
    mz = pd.concat([mz.assign(solvent=s) for s in solvents], ignore_index=True)
    mz = mz.set_index(["solute", "nucleus", "site", "solvent"]).sort_index()

    # --- QCD correction (solvent-independent, tiled across solvents) ---
    qcd_rows = []
    for _, r in sites.iterrows():
        idx = site_atom_indices(r["atom_numbers"])
        q = get_qcd(r["solute"])
        corr = q["trajectories"].mean(0).mean(0)[idx, 3] - q["stationary"][idx, 3]
        qcd_rows.append([r["solute"], r["nucleus"], r["site"],
                         float(corr.mean()) if len(corr) else np.nan])
    qcd = pd.DataFrame(qcd_rows, columns=["solute", "nucleus", "site", "qcd"])
    qcd = pd.concat([qcd.assign(solvent=s) for s in solvents], ignore_index=True)
    qcd = qcd.set_index(["solute", "nucleus", "site", "solvent"]).sort_index()

    # --- MagNET-X explicit-solvent correction + vibrational correction (per solvent, 650-atom shell) ---
    ex_rows = []
    for _, r in sites.iterrows():
        idx = site_atom_indices(r["atom_numbers"])
        for solvent in solvents:
            shells = get_shells(r["solute"], solvent)
            if 650 not in shells:
                ex_rows.append([r["solute"], r["nucleus"], r["site"], solvent, np.nan, np.nan]); continue
            mx = get_mx(r["solute"], solvent)        # (frames, atoms, [isolated, solvated])
            corr = mx[:, idx, 1] - mx[:, idx, 0]
            corr = np.mean(corr, axis=1)
            corr = corr[~np.isnan(corr)]
            openmm = float(np.mean(corr)) if len(corr) else np.nan
            stat = get_mxs(r["solute"], solvent)
            vib = float(np.mean(mx.mean(0)[idx, 0] - np.asarray(stat)[idx]))
            ex_rows.append([r["solute"], r["nucleus"], r["site"], solvent, openmm, vib])
    ex = pd.DataFrame(ex_rows, columns=["solute", "nucleus", "site", "solvent", "openMM", "openMM_vib"])
    ex = ex.set_index(["solute", "nucleus", "site", "solvent"]).sort_index()

    # --- combine + composite columns ---
    comb = pd.concat([experimental_stack(loader), mz, ex, qcd], axis=1).reset_index()
    comb = comb.set_index(["solute", "nucleus", "site", "solvent"])
    # stationary_plus_pcm: scale the chloroform PCM correction to each solvent (factors from the loader,
    # which renames water -> TIP4P to match the explicit-solvent naming)
    conv = {}
    for nuc in ["H", "C"]:
        s = loader.pcm_conversion_factors(nuc).set_index("solvent")["pcm_conversion_factor"]
        conv[nuc] = s.rename({"water": "TIP4P"})
    nn = comb.reset_index()
    factors = nn.apply(lambda r: conv[r["nucleus"]].get(r["solvent"], np.nan), axis=1)
    nn["stationary_plus_pcm"] = nn["stationary"] + nn["pcm"] * factors
    nn = nn.set_index(comb.index.names)
    comb["stationary_plus_pcm"] = nn["stationary_plus_pcm"]
    comb["stationary_plus_qcd"] = comb["stationary"] + comb["qcd"]
    comb["stationary_plus_op_vib"] = comb["stationary"] + comb["openMM_vib"]
    comb["stationary_plus_pcm_plus_qcd"] = comb["stationary_plus_pcm"] + comb["qcd"]
    comb["stationary_plus_pcm_plus_op_vib"] = comb["stationary_plus_pcm"] + comb["openMM_vib"]
    comb["stationary_plus_openMM_plus_qcd"] = comb["stationary"] + comb["openMM"] + comb["qcd"]
    comb["stationary_plus_openMM_plus_op_vib"] = comb["stationary"] + comb["openMM"] + comb["openMM_vib"]
    return comb.reset_index()


# ----------------------------------------------------------------------------
# Composite-model fitting (Figure 5C/5D, SI S15). The coefficients applied at scale come from the
# loader's composite_model group (fit on delta-22).
# ----------------------------------------------------------------------------

def fit(fit_df, formula):
    """OLS of experimental shift on a composite formula; returns (RMSE, params).

    params is a pandas Series indexed by "Intercept" and the predictor names (matching statsmodels'
    naming, which the composite-model coefficient code downstream relies on). This does the ordinary
    least squares directly with numpy (np.linalg.lstsq), which is identical to the notebooks'
    statsmodels OLS but far faster in this fitting loop: statsmodels re-parses the formula with patsy
    on every call (~1000x slower per call). The `fit`-matches-`_fit_statsmodels` test guards the
    equivalence.

    Rows with a missing experimental value or a missing predictor are dropped before fitting and are
    not scored: statsmodels drops those rows when fitting too, and its `predict` returns NaN for a row
    with a missing predictor, so a single missing value never turns the whole RMSE into NaN.
    """
    terms = [term.strip() for term in formula.split("+") if term.strip()]
    response = fit_df["experimental"].to_numpy(dtype=float)
    columns = [fit_df[term].to_numpy(dtype=float) for term in terms]
    design = np.column_stack([np.ones(len(fit_df))] + columns)
    keep = np.isfinite(response) & np.all(np.isfinite(design), axis=1)
    design, response = design[keep], response[keep]
    beta, *_ = np.linalg.lstsq(design, response, rcond=None)
    params = pd.Series(beta, index=["Intercept"] + terms)
    return _rmse(design @ beta, response), params


def _fit_statsmodels(fit_df, formula):
    """The original statsmodels OLS, kept only as a test oracle for `fit` (see the equivalence test).
    Slow because statsmodels re-parses the formula with patsy on every call; do not use in the harness."""
    result = smf.ols(formula=f"experimental ~ {formula}", data=fit_df).fit()
    fit_df = fit_df.copy()
    fit_df["predicted"] = result.predict(fit_df)
    scored = fit_df[fit_df["experimental"].notna() & fit_df["predicted"].notna()]
    return _rmse(scored["predicted"], scored["experimental"]), result.params


def fit_formulas_per_solvent(df, formulas, solvents, formula_remap):
    """One OLS fit per (formula, solvent), pooled across all solutes; returns rmse and params rows."""
    rows = []
    for formula in formulas:
        for solvent in solvents:
            d = df[df["solvent"] == solvent].copy()
            rmse, params = fit(d, formula)
            rows.append({"formula": formula_remap.get(formula, formula), "solvent": solvent,
                         "rmse": rmse, "params": params})
    return pd.DataFrame(rows)


def fit_formulas_per_solvent_and_solute(df, formulas, solvents, solutes, formula_remap):
    """One OLS fit per (formula, solvent, solute); skips a group with under 2 rows or no
    experimental data, and records the exception instead of raising if a fit fails."""
    rows = []
    for formula in formulas:
        for solvent in solvents:
            for solute in solutes:
                d = df[(df["solvent"] == solvent) & (df["solute"] == solute)].copy()
                if len(d) < 2 or d["experimental"].isnull().all():
                    continue
                try:
                    rmse, params = fit(d, formula)
                    rows.append({"formula": formula_remap.get(formula, formula), "solvent": solvent,
                                 "solute": solute, "rmse": rmse, "params": params})
                except Exception as e:  # noqa: BLE001 - mirror source behaviour
                    rows.append({"formula": formula_remap.get(formula, formula), "solvent": solvent,
                                 "solute": solute, "rmse": np.nan, "params": str(e)})
    return pd.DataFrame(rows)


def per_solvent_fits(query_df_nn):
    """all_solute_fitting_results: full per-solvent OLS fits for both nuclei ('scaled to test set')."""
    out = {}
    formulas = {"H": FITTING_FORMULAS_H, "C": FITTING_FORMULAS_C}
    solvents = list(query_df_nn["solvent"].unique())
    for nuc in ["H", "C"]:
        d = query_df_nn[query_df_nn["nucleus"] == nuc].copy()
        out[nuc] = fit_formulas_per_solvent(d, formulas[nuc], solvents, FORMULA_REMAP)
    return out


def per_solute_fits(query_df_nn):
    """per_solute_fitting_results: per-solvent-and-solute OLS fits ('scaled to solute' baseline)."""
    out = {}
    formulas = {"H": FITTING_FORMULAS_H, "C": FITTING_FORMULAS_C}
    solvents = list(query_df_nn["solvent"].unique())
    solutes = list(query_df_nn["solute"].unique())
    for nuc in ["H", "C"]:
        d = query_df_nn[query_df_nn["nucleus"] == nuc].copy()
        out[nuc] = fit_formulas_per_solvent_and_solute(d, formulas[nuc], solvents, solutes, FORMULA_REMAP)
    return out


def build_bootstrap_seed_coeffs(loader):
    """Per-nucleus DataFrame of bootstrap coefficients with a remapped 'formula' column, read from
    the loader's composite_model group (the delta-22 bootstrap fits applied to the natural products)."""
    out = {}
    for nuc in ["H", "C"]:
        frames = []
        for token, name in _BOOTSTRAP_TOKENS[nuc].items():
            df = loader.bootstrap_coefficients(token, nuc).copy()
            df["formula"] = name
            frames.append(df)
        combined = pd.concat(frames, ignore_index=True, sort=False)
        front = [c for c in ["solvent", "formula", "seed"] if c in combined.columns]
        out[nuc] = combined[front + [c for c in combined.columns if c not in front]]
    return out


def apply_bootstrap_params_to_full_dataset(data_df, params_df, nucleus=None):
    """Apply each bootstrap coefficient set (per formula/seed/solvent) to the full NP dataset,
    returning predictions: solute, nucleus, site, solvent, formula, seed, experimental, predicted."""
    coeffs_df = params_df.copy()
    if nucleus is not None:
        data_df = data_df[data_df["nucleus"] == nucleus]
    keep = ["solute", "nucleus", "site", "solvent", "experimental"]
    keep += [c for c in data_df.columns if c in coeffs_df.columns and c not in keep]
    data_df = data_df[keep]
    merged = coeffs_df.merge(data_df, how="outer", on="solvent", suffixes=("_coeff", "_data"))
    idx = ["solute", "nucleus", "site", "solvent", "formula", "seed"]
    merged = merged[idx + [c for c in merged.columns if c not in idx]]
    # vectorized prediction: intercept (0 if missing) plus, for each parameter, coeff*data where the
    # coeff is present (a missing coeff contributes nothing; a present coeff with missing data
    # propagates NaN).
    pred = merged["Intercept"].astype(float).fillna(0.0) if "Intercept" in merged.columns else 0.0
    for cc in [c for c in merged.columns if c.endswith("_coeff")]:
        dc = cc.replace("_coeff", "_data")
        if dc not in merged.columns:
            continue
        term = merged[cc].astype(float) * merged[dc].astype(float)
        pred = pred + term.where(merged[cc].notna(), 0.0)
    merged["predicted"] = pred
    return merged[idx + ["experimental", "predicted"]]


def compute_solute_rmses(df, solute_col="solute"):
    """Per (solvent, nucleus, formula, seed, solute) RMSE over a solute's sites."""
    valid = df[df["experimental"].notnull() & df["predicted"].notnull()].copy()
    # RMSE is sqrt(mean(squared error)), so a single vectorized groupby-mean over a precomputed
    # squared-error column is identical to grouped.apply(rmse) but ~19x faster on the ~200k groups
    # of a full bootstrap table (stats.rmse is exactly np.sqrt(np.mean(np.square(...)))).
    valid["_squared_error"] = np.square(valid["predicted"] - valid["experimental"])
    result = (valid.groupby(["solvent", "nucleus", "formula", "seed", solute_col])["_squared_error"]
              .mean().pow(0.5).reset_index(name="Bootstrap_RMSE"))
    return result.rename(columns={solute_col: "solute"}) if solute_col != "solute" else result


def compute_grouped_rmse(df, solute_groups, solute_col="solute"):
    """Per (solvent, nucleus, formula, seed, solute_group) RMSE over all sites in the group."""
    lookup = {s: g for g, ss in solute_groups.items() for s in ss}
    df = df.copy()
    df["solute_group"] = df[solute_col].map(lookup).fillna("Unknown")
    valid = df[df["experimental"].notnull() & df["predicted"].notnull()].copy()
    # vectorized RMSE, same identity and speedup as compute_solute_rmses above
    valid["_squared_error"] = np.square(valid["predicted"] - valid["experimental"])
    return (valid.groupby(["solvent", "nucleus", "formula", "seed", "solute_group"])["_squared_error"]
            .mean().pow(0.5).reset_index(name="Bootstrap_RMSE"))


# ---------------------------------------------------------------------------------------------
# SI Figure S15's "Fitting RMSE Comparisons (chloroform, H/C)" bar chart: per test-set solute,
# three ways of choosing the composite-model coefficients -- "Scaled to Solute" (fit each solute
# on its own), "Scaled to Test Set" (one fit pooled across all 12 test-set solutes), "Extrapolated
# from delta22" (delta-22's bootstrap fits applied cold, never seeing the test set). Built from
# per_solute_fits (Scaled to Solute), per_solvent_fits (Scaled to Test Set, split back out per
# solute by scaled_to_test_set_per_solute_rmse below), and apply_bootstrap_params_to_full_dataset +
# compute_solute_rmses (Extrapolated from delta22, the same data Figure 5D's boxplot uses).

def scaled_to_test_set_per_solute_rmse(query_df_nn, all_solute_fits, nucleus, solvent, formula):
    """per_solvent_fits' output has one pooled rmse per (formula, solvent): the model is fit once
    across every test-set solute together. To get a comparable per-solute rmse (needed to sit
    "Scaled to Test Set" next to "Scaled to Solute" and "Extrapolated from delta22" in the same bar
    chart), this applies that one pooled fit's coefficients to each solute's own sites and scores
    them individually, reusing apply_bootstrap_params_to_full_dataset's prediction machinery by
    reshaping the fitted statsmodels Params Series into that function's expected wide
    "<term>_coeff" row format."""
    remapped = FORMULA_REMAP.get(formula, formula)
    fits = all_solute_fits[nucleus]
    match = fits[(fits["formula"] == remapped) & (fits["solvent"] == solvent)]
    if match.empty:
        raise KeyError(f"no per_solvent_fits row for formula={remapped!r} solvent={solvent!r}")
    params = match.iloc[0]["params"]
    # bare term names, matching data_df's own raw column names -- apply_bootstrap_params_to_full_
    # dataset's merge(..., suffixes=("_coeff", "_data")) is what appends "_coeff"/"_data" (only
    # triggered by the name COLLIDING with data_df's column of the same name); pre-suffixing here
    # would just create an unmatched "<term>_coeff" column no data_df column collides with.
    wide = {"Intercept": float(params.get("Intercept", 0.0)), "solvent": solvent,
           "formula": remapped, "seed": 0}
    for term in params.index:
        if term != "Intercept":
            wide[term] = float(params[term])
    params_df = pd.DataFrame([wide])
    # Restrict to this one solvent before the merge: apply_bootstrap_params_to_full_dataset's outer
    # join zero-fills a missing coefficient instead of raising, so an unfiltered call would
    # silently score every OTHER solvent's rows against this solvent's plane as if the missing
    # coefficients were all zero.
    solvent_only = query_df_nn[query_df_nn["solvent"] == solvent]
    preds = apply_bootstrap_params_to_full_dataset(solvent_only, params_df, nucleus=nucleus)
    return compute_solute_rmses(preds)


def fitting_rmse_comparison_table(query_df_nn, per_solute_fits_result, per_solvent_fits_result,
                                  bootstrap_rmses, nucleus, solvent, formula):
    """Assembles SI Figure S15's "Fitting RMSE Comparisons" table: one row per test-set solute,
    columns "Scaled to Solute", "Scaled to Test Set", "Extrapolated from delta22" (ppm RMSE), all
    at one nucleus/solvent/formula. bootstrap_rmses is compute_solute_rmses' output for `nucleus`
    (e.g. Figure 5D's bootstrap_rmses_h), averaged here over its bootstrap seeds."""
    remapped = FORMULA_REMAP.get(formula, formula)

    solute_fit = per_solute_fits_result[nucleus]
    scaled_to_solute = (solute_fit[(solute_fit["formula"] == remapped) & (solute_fit["solvent"] == solvent)]
                        .set_index("solute")["rmse"])

    scaled_to_test_set = (scaled_to_test_set_per_solute_rmse(query_df_nn, per_solvent_fits_result, nucleus, solvent, formula)
                          .query("solvent == @solvent and formula == @remapped")
                          .set_index("solute")["Bootstrap_RMSE"])

    boot = bootstrap_rmses[(bootstrap_rmses["nucleus"] == nucleus) & (bootstrap_rmses["solvent"] == solvent)
                           & (bootstrap_rmses["formula"] == remapped)]
    extrapolated = boot.groupby("solute")["Bootstrap_RMSE"].mean()

    table = pd.DataFrame({"Scaled to Solute": scaled_to_solute, "Scaled to Test Set": scaled_to_test_set,
                          "Extrapolated from delta22": extrapolated})
    return table.dropna(how="all")


def distribution_shift_by_solvent_table(per_solute_fits_result, per_solvent_fits_result,
                                        bootstrap_rmses, nucleus, formula, solvents=None):
    """SI Figure S15's per-solvent "distribution shift is minor" cross-check -- the solvent-averaged
    companion to fitting_rmse_comparison_table (which is per-solute for a single solvent). For one
    nucleus and formula, the mean test-set RMSE under the three ways of choosing the composite-model
    coefficients, one row per solvent: "Extrapolated from delta22" (delta-22's bootstrap
    coefficients applied cold), "Scaled to Test Set" (one fit pooled across the whole test set), and
    "Scaled to Solute" (a separate fit per solute). Where the three sit close, re-optimizing the
    coefficients barely helps, so they transfer and the residual error is physics; the one solvent
    where "Extrapolated" rises well above the refits is TIP4P (water), dominated by the peptide's
    conformer-population sensitivity, which is why S15 reports solvent-averaged performance.

    bootstrap_rmses is compute_solute_rmses' output for `nucleus` (e.g. the si_figure_s15 notebook's
    bootstrap_rmses["H"]); per_solute_fits_result / per_solvent_fits_result are the per_solute_fits /
    per_solvent_fits dicts keyed by nucleus. Averages over bootstrap seeds and over each solvent's
    solutes."""
    remapped = FORMULA_REMAP.get(formula, formula)
    boot = bootstrap_rmses[(bootstrap_rmses["nucleus"] == nucleus)
                           & (bootstrap_rmses["formula"] == remapped)]
    test_set = per_solvent_fits_result[nucleus]
    test_set = test_set[test_set["formula"] == remapped]
    solute = per_solute_fits_result[nucleus]
    solute = solute[solute["formula"] == remapped]
    if solvents is None:
        solvents = sorted(boot["solvent"].unique())
    rows = []
    for solvent in solvents:
        rows.append({
            "solvent": solvent,
            "Extrapolated from delta22": boot[boot["solvent"] == solvent]["Bootstrap_RMSE"].mean(),
            "Scaled to Test Set": test_set[test_set["solvent"] == solvent]["rmse"].mean(),
            "Scaled to Solute": solute[solute["solvent"] == solvent]["rmse"].mean(),
        })
    return pd.DataFrame(rows).set_index("solvent")


# ---------------------------------------------------------------------------------------------
# SI Figure S15's "Feature Space Coverage by Solvent" and "Residuals for Delta22 Fitting
# Coefficients" panels: do the test set and delta-22 cover the same region of feature space (so
# the test set's higher error can't be blamed on extrapolating outside what delta-22's fitting
# coefficients ever saw)? Both panels combine the test set (this module's query_df_nn) with
# delta-22's own per-site data at the same two features (delta22.py's query_df_nn, passed in by
# the caller so this module doesn't import delta22.py directly).

# Feature Space Coverage only compares the 4 solvents both datasets have explicit-solvent (OpenMM)
# coverage for -- delta-22 has Desmond+OpenMM MD for all 12 solvents, but the natural-products test
# set (sigma-fresh-derived) only has 4.
FEATURE_SPACE_SOLVENTS = ("chloroform", "benzene", "methanol", "TIP4P")


def _feature_space_config(nucleus):
    if nucleus == "H":
        return {"x_feature": "stationary_plus_qcd", "y_feature": "openMM"}
    if nucleus == "C":
        return {"x_feature": "stationary_plus_op_vib", "y_feature": "openMM"}
    raise ValueError(f"unsupported nucleus {nucleus!r}, expected 'H' or 'C'")


def _test_set_and_delta22(query_df_nn, delta22_query_df_nn, nucleus, cols):
    """The shared setup both panels below need: the test set's own rows (excludes any row
    literally named "delta22", a defensive filter) tagged "Test Set", concatenated with
    delta22_query_df_nn's rows (already delta-22-only) tagged "Delta22"."""
    test = query_df_nn[query_df_nn["nucleus"] == nucleus].copy()
    test = test[~test["solute"].astype(str).str.strip().str.lower().eq("delta22")]
    test["dataset"] = "Test Set"
    d22 = delta22_query_df_nn[delta22_query_df_nn["nucleus"] == nucleus].copy()
    d22["dataset"] = "Delta22"
    return pd.concat([test[cols + ["dataset"]], d22[cols + ["dataset"]]], ignore_index=True)


def feature_space_coverage_table(query_df_nn, delta22_query_df_nn, nucleus, solvents=FEATURE_SPACE_SOLVENTS):
    """SI Figure S15's "Feature Space Coverage by Solvent": test-set and delta-22 per-site feature
    values, mean-centered globally -- one scalar mean per feature, pooled over every solvent and
    both datasets (the notebook's separate fitting cell instead centers per solvent). Returns a
    tidy DataFrame: solvent, dataset ("Test Set"/"Delta22"), x (centered), y (centered)."""
    cfg = _feature_space_config(nucleus)
    x_feature, y_feature = cfg["x_feature"], cfg["y_feature"]
    combined = _test_set_and_delta22(query_df_nn, delta22_query_df_nn, nucleus, ["solvent", x_feature, y_feature])
    combined = combined[combined["solvent"].isin(solvents)].dropna(subset=[x_feature, y_feature])
    combined["x"] = combined[x_feature] - combined[x_feature].mean()
    combined["y"] = combined[y_feature] - combined[y_feature].mean()
    return combined[["solvent", "dataset", "x", "y"]].reset_index(drop=True)


def delta22_plane_residuals_table(query_df_nn, delta22_query_df_nn, nucleus, solvents=FEATURE_SPACE_SOLVENTS):
    """SI Figure S15's "Residuals for Delta22 Fitting Coefficients": fits a 2-feature OLS plane
    (experimental ~ x_feature + y_feature, the same two features feature_space_coverage_table
    plots) to delta-22's data alone, separately per solvent, then applies that one plane to both
    delta-22 and the test set, returning residual = experimental - predicted. For a linear model,
    minimizing RMSE directly (Nelder-Mead) is the same convex problem as ordinary least squares, so
    np.linalg.lstsq gives the identical plane. Returns a tidy DataFrame: solvent, dataset,
    experimental, residual."""
    cfg = _feature_space_config(nucleus)
    x_feature, y_feature = cfg["x_feature"], cfg["y_feature"]
    combined = _test_set_and_delta22(query_df_nn, delta22_query_df_nn, nucleus,
                                     ["solvent", x_feature, y_feature, "experimental"])
    combined = combined[combined["solvent"].isin(solvents)].dropna(subset=[x_feature, y_feature, "experimental"])

    rows = []
    for solvent in solvents:
        sdf = combined[combined["solvent"] == solvent]
        d22_sdf = sdf[sdf["dataset"] == "Delta22"]
        if len(d22_sdf) < 3:
            continue
        design = np.column_stack([np.ones(len(d22_sdf)), d22_sdf[x_feature], d22_sdf[y_feature]])
        c0, c1, c2 = np.linalg.lstsq(design, d22_sdf["experimental"].to_numpy(), rcond=None)[0]
        predicted = c0 + c1 * sdf[x_feature] + c2 * sdf[y_feature]
        out = sdf[["solvent", "dataset", "experimental"]].copy()
        out["residual"] = sdf["experimental"] - predicted
        rows.append(out)
    if not rows:
        return pd.DataFrame(columns=["solvent", "dataset", "experimental", "residual"])
    return pd.concat(rows, ignore_index=True).reset_index(drop=True)
