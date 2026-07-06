"""Core analysis for the delta-22 figures (no plotting).

Shared machinery behind the delta-22 figures (Figures 2A, 3, 4; SI S1, S4-S8, S11, S13, S16-S18):
a seeded per-solute train/test fitting harness, the Pareto computation, and correlation matrices.
Reads the released data via `delta22_reader`; the figure notebooks and tests import from here.

Splits are seeded with random.seed(seed + 100), shuffle, first n_test solutes are the test set, so
results are deterministic. `fit` uses numpy lstsq directly instead of statsmodels smf.ols in the
inner loop (patsy re-parsing makes that ~1000x slower); `_fit_statsmodels` is kept as a test oracle.
"""
import random

import numpy as np
import pandas as pd

from stats import rmse, mae, ols_fit, linear_fit_1d
from spreadsheet import site_atom_indices  # noqa: F401  re-exported for notebooks (D.site_atom_indices)
from delta22_reader import (
    load_delta22_dft_data,
    load_delta22_nn_data,
    load_solutes,
    load_perturbed_shieldings,
)

# The 12 solvents with explicit (Desmond) data, in a stable order.
DESMOND_SOLVENTS = [
    "tetrahydrofuran", "dichloromethane", "chloroform", "toluene", "benzene", "chlorobenzene",
    "acetone", "dimethylsulfoxide", "acetonitrile", "trifluoroethanol", "methanol", "TIP4P",
]

# the double hybrids (and DLPNO-MP2) have no implicit-solvent shielding of their own; the reader
# substitutes a B3LYP-D3(BJ) PCM correction for them, so their pcm column is not method-specific.
# Exclude them from any analysis that compares the PCM correction across methods.
DOUBLE_HYBRID_METHODS = ["B2GP_PLYP", "B2PLYP", "dlpno_mp2", "dsd_pbep86",
                         "mPW2PLYP", "revdsd_pbep86"]

STATIONARY_VS_PCM = ["stationary", "stationary + pcm"]


# ---------------------------------------------------------------------------
# loading / assembling the flat query table

def delta22_solutes(delta22_path):
    """The list of solute names stored in the file."""
    return load_solutes(delta22_path)


def build_query_df(combined_df):
    """Flatten a reader combined table (load_delta22_dft_data / load_delta22_nn_data output,
    which is multi-indexed) into a plain DataFrame with the index levels as columns. Fitting
    and split helpers operate on this flat table."""
    return combined_df.reset_index()


def load_query_df_dft(delta22_path, experimental_path, **kwargs):
    """Convenience: load the DFT combined table and flatten it in one call."""
    return build_query_df(load_delta22_dft_data(delta22_path, experimental_path, **kwargs))


def load_query_df_nn(delta22_path, experimental_path, **kwargs):
    """Convenience: load the MagNET (NN) combined table and flatten it in one call."""
    return build_query_df(load_delta22_nn_data(delta22_path, experimental_path, **kwargs))


# ---------------------------------------------------------------------------
# composite predictor columns

# the precomputed sum columns the formulas reference as single terms (e.g. "stationary_plus_pcm" is
# one predictor = stationary + pcm, distinct from the two-term "stationary + pcm"). For MagNET (NN)
# data, stationary_plus_pcm is precomputed by the loader using the solvent-specific PCM scaling
# factors; add_composite_columns leaves an existing stationary_plus_pcm column untouched.
# NOTE: the pcm three-term chains here (stationary_plus_pcm_plus_qcd etc.) are RAW sums with an
# unscaled pcm term. build_composite_model.py needs the pcm term carried at its scaled value, so it
# builds those columns itself on top of the scaled stationary_plus_pcm. Do not swap it to this
# helper: the results differ by order 1 ppm on every pcm chain.
_COMPOSITE_COLUMNS = {
    "stationary_plus_pcm": ("stationary", "pcm"),
    "stationary_plus_desmond": ("stationary", "desmond"),
    "stationary_plus_openMM": ("stationary", "openMM"),
    "stationary_plus_qcd": ("stationary", "qcd"),
    "stationary_plus_des_vib": ("stationary", "desmond_vib"),
    "stationary_plus_op_vib": ("stationary", "openMM_vib"),
    "stationary_plus_pcm_plus_qcd": ("stationary", "pcm", "qcd"),
    "stationary_plus_pcm_plus_des_vib": ("stationary", "pcm", "desmond_vib"),
    "stationary_plus_pcm_plus_op_vib": ("stationary", "pcm", "openMM_vib"),
    "stationary_plus_desmond_plus_qcd": ("stationary", "desmond", "qcd"),
    "stationary_plus_desmond_plus_des_vib": ("stationary", "desmond", "desmond_vib"),
    "stationary_plus_openMM_plus_qcd": ("stationary", "openMM", "qcd"),
    "stationary_plus_openMM_plus_op_vib": ("stationary", "openMM", "openMM_vib"),
}


def add_composite_columns(query_df, keep_existing=("stationary_plus_pcm",)):
    """Add the precomputed sum columns the composite formulas reference as single terms. Returns a
    copy. Columns named in keep_existing that are already present (e.g. an NN stationary_plus_pcm
    built with the solvent scaling factors) are not overwritten."""
    out = query_df.copy()
    for name, parts in _COMPOSITE_COLUMNS.items():
        if name in keep_existing and name in out.columns:
            continue
        out[name] = sum(out[part] for part in parts)
    return out


# ---------------------------------------------------------------------------
# the linear fitting harness

def _formula_terms(formula):
    """The additive predictor column names in 'a + b + c'. These figures use only additive main
    effects (no interactions), so a plain split on '+' is exact."""
    return [term.strip() for term in formula.split("+") if term.strip()]


def _design_matrix(df, terms):
    """Design matrix [1, term_1, term_2, ...] as float, plus the response, with any row that has a
    missing value in the response or a predictor dropped (statsmodels drops missing rows too)."""
    response = df["experimental"].to_numpy(dtype=float)
    columns = [df[term].to_numpy(dtype=float) for term in terms]
    design = np.column_stack([np.ones(len(df))] + columns)
    keep = np.isfinite(response) & np.all(np.isfinite(design), axis=1)
    return design[keep], response[keep]


def fit(train_df, test_df, formula, mode="rmse"):
    """Fit experimental ~ formula by ordinary least squares on train_df and score on both sets.

    Returns (train_error, test_error, params), where params is a pandas Series indexed by
    "Intercept" and the predictor names (matching statsmodels' naming). mode is "rmse" (default)
    or "mae". Uses numpy least squares directly (see the module docstring for why)."""
    terms = _formula_terms(formula)
    train_x, train_y = _design_matrix(train_df, terms)
    beta = ols_fit(train_x, train_y)
    params = pd.Series(beta, index=["Intercept"] + terms)

    test_x, test_y = _design_matrix(test_df, terms)
    train_pred = train_x @ beta
    test_pred = test_x @ beta
    score = rmse if mode == "rmse" else mae if mode == "mae" else None
    if score is None:
        raise ValueError(f"mode must be 'rmse' or 'mae', got {mode!r}")
    return score(train_pred, train_y), score(test_pred, test_y), params


def _fit_statsmodels(train_df, test_df, formula, mode="rmse"):
    """Statsmodels OLS, kept only as a test oracle for `fit`. Slow because statsmodels re-parses
    the formula with patsy on every call; do not use it in the harness."""
    import statsmodels.formula.api as smf
    result = smf.ols(formula=f"experimental ~ {formula}", data=train_df).fit()
    score = rmse if mode == "rmse" else mae
    train_err = score(result.predict(train_df).to_numpy(), train_df["experimental"].to_numpy())
    test_err = score(result.predict(test_df).to_numpy(), test_df["experimental"].to_numpy())
    return train_err, test_err, result.params


def generate_solute_splits(n_splits, query_df, solvent, solutes,
                           n_test=10, seed_offset=100):
    """Build n_splits seeded train/test partitions for one solvent, splitting on SOLUTE so a
    molecule is never in both sets. For split k: seed the RNG with k + seed_offset, shuffle the
    solute list, take the first n_test as the test set and the rest as the training set, then select
    that solvent's rows. Returns a list of (train_df, test_df) pairs."""
    solute_list = list(solutes)
    # filter this solvent once, then group its rows by solute, so each seed just concatenates the
    # relevant groups instead of masking the whole table again.
    solvent_df = query_df[query_df["solvent"] == solvent]
    by_solute = {solute: rows for solute, rows in solvent_df.groupby("solute")}
    empty = solvent_df.iloc[0:0]

    def _gather(molecules):
        frames = [by_solute[m] for m in molecules if m in by_solute]
        return pd.concat(frames) if frames else empty

    splits = []
    for seed in range(n_splits):
        random.seed(seed + seed_offset)
        molecules = list(solute_list)
        random.shuffle(molecules)
        splits.append((_gather(molecules[n_test:]), _gather(molecules[:n_test])))
    return splits


def _solute_partitions(n_splits, solutes, n_test, seed_offset=100):
    """The seeded (train_solutes, test_solutes) name partitions the harness fits over, matching
    generate_solute_splits exactly: for split k, seed random with k + seed_offset, shuffle the full
    solute list, test = the first n_test names, train = the rest. Row selection and solvent filtering
    are left to the caller (see _split_fit_scores), so these partitions are computed once and reused
    across formulas."""
    partitions = []
    base = list(solutes)
    for seed in range(n_splits):
        random.seed(seed + seed_offset)
        molecules = list(base)
        random.shuffle(molecules)
        partitions.append((molecules[n_test:], molecules[:n_test]))
    return partitions


def _split_fit_scores(group_df, solvent, formulas, partitions, mode="rmse", min_train=2, swap=False):
    """Fit each formula over the given seeded solute partitions for one group and solvent, returning
    {(formula, seed): (train_error, test_error)} for every split that was fit.

    The fast path behind run_fits / pareto_solvent_averaged_rmse: converts each column to numpy once
    and selects each split's rows by integer index, avoiding the per-split pd.concat that dominates
    runtime in generate_solute_splits + fit. Mathematically identical to fit() on the same rows.
    Splits whose train set is smaller than min_train, whose test set is empty, or that fail to solve
    are omitted from the result. swap=True trains on the first n_test solutes instead of the rest
    (Figure 2A)."""
    solvent_df = group_df[group_df["solvent"] == solvent]
    n_rows = len(solvent_df)
    score = rmse if mode == "rmse" else mae if mode == "mae" else None
    if score is None:
        raise ValueError(f"mode must be 'rmse' or 'mae', got {mode!r}")

    # row positions (0..n_rows-1 within solvent_df) for each solute, and the numeric columns as numpy
    solute_values = solvent_df["solute"].to_numpy()
    by_solute = {}
    for position, solute in enumerate(solute_values):
        by_solute.setdefault(solute, []).append(position)
    by_solute = {s: np.array(v, dtype=np.intp) for s, v in by_solute.items()}
    empty = np.array([], dtype=np.intp)

    def gather(names):
        arrays = [by_solute[m] for m in names if m in by_solute]
        return np.concatenate(arrays) if arrays else empty

    # each split's train/test row indices, computed once and shared across formulas
    split_indices = []
    for train_names, test_names in partitions:
        train_idx, test_idx = gather(train_names), gather(test_names)
        if swap:
            train_idx, test_idx = test_idx, train_idx
        split_indices.append((train_idx, test_idx))

    response = solvent_df["experimental"].to_numpy(dtype=float)
    term_columns = {term: solvent_df[term].to_numpy(dtype=float)
                    for formula in formulas for term in _formula_terms(formula)}

    scores = {}
    for formula in formulas:
        terms = _formula_terms(formula)
        design = np.column_stack([np.ones(n_rows)] + [term_columns[t] for t in terms])
        finite = np.isfinite(response) & np.all(np.isfinite(design), axis=1)
        for seed, (train_idx, test_idx) in enumerate(split_indices):
            if len(train_idx) < min_train or len(test_idx) < 1:
                continue
            train_fit = train_idx[finite[train_idx]]
            test_fit = test_idx[finite[test_idx]]
            try:
                beta = ols_fit(design[train_fit], response[train_fit])
                train_err = score(design[train_fit] @ beta, response[train_fit])
                test_err = score(design[test_fit] @ beta, response[test_fit])
            except Exception:
                continue
            scores[(formula, seed)] = (train_err, test_err)
    return scores


def run_fits(query_df, solvents, formulas, n_splits, solutes,
             group_cols=None, n_test=10, mode="rmse", min_train=2):
    """Run the fitting harness over every (group, solvent, formula, split) combination and return
    a tidy DataFrame with train_RMSE / test_RMSE per row.

    group_cols: optional list of columns to fit within separately (e.g.
    ["sap_nmr_method", "sap_basis", "sap_geometry_type"] for the per-method figures). When None,
    the whole query_df is used as a single group (the caller has pre-filtered it). Splits that are
    too small to fit, or that cannot be solved, are skipped.
    """
    rows = []
    if group_cols:
        group_iter = list(query_df.groupby(group_cols))
    else:
        group_iter = [((), query_df)]
    partitions = _solute_partitions(n_splits, solutes, n_test)
    for group_key, group_df in group_iter:
        meta = {}
        if group_cols:
            key = group_key if isinstance(group_key, tuple) else (group_key,)
            meta = dict(zip(group_cols, key))
        for solvent in solvents:
            scores = _split_fit_scores(group_df, solvent, formulas, partitions,
                                       mode=mode, min_train=min_train)
            for formula in formulas:
                for seed in range(n_splits):
                    got = scores.get((formula, seed))
                    if got is None:
                        continue
                    rows.append({**meta, "solvent": solvent, "formula": formula,
                                 "seed": seed, "train_RMSE": got[0], "test_RMSE": got[1]})
    return pd.DataFrame(rows)


def full_fit_coefficients(query_df, solvents, formula):
    """Fit `formula` on the full data (no train/test split) for each solvent and return a DataFrame
    indexed by parameter name ("Intercept" plus the predictors) with one column per solvent. This is
    the deterministic full-fit OLS whose coefficients are the composite model's coefficients, stored
    in applications.hdf5's composite_model/ols_coefficients group. Fitting the released delta-22 NN
    data here (with nitromethane dropped) reproduces those coefficients to within the data's
    integer-encoding precision."""
    columns = {}
    for solvent in solvents:
        sub = query_df[query_df["solvent"] == solvent]
        if len(sub) < 2:
            continue
        _, _, params = fit(sub, sub, formula)
        columns[solvent] = params
    return pd.DataFrame(columns)


def fit_coefficients(query_df, solvent, formula, solutes, n_splits,
                     n_test=10, seed_offset=100):
    """Fit one formula for one solvent across all seeded splits and return the per-split fitted
    coefficients as a DataFrame (one row per seed, columns = the statsmodels parameter names plus
    the seed). This is the building block for the bootstrap coefficient distributions and for
    verifying the composite-model coefficients fit on delta-22."""
    splits = generate_solute_splits(n_splits, query_df, solvent, solutes,
                                    n_test=n_test, seed_offset=seed_offset)
    records = []
    for seed, (train_df, test_df) in enumerate(splits):
        if len(train_df) < 2:
            continue
        _, _, params = fit(train_df, test_df, formula)
        record = {"solvent": solvent, "seed": seed}
        record.update({name: float(value) for name, value in params.items()})
        records.append(record)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Figure S8 B-D (Convergence of Explicit Solvent Corrections)

def frame_corrections(perturbed, atom_indices):
    """Per-frame solvent correction for one site: the mean over the site's atoms of
    (solvated - isolated), from a raw (n_frames, n_atoms, 2) perturbed-shielding array
    (load_perturbed_shieldings). Frames whose site atoms were not computed read as NaN."""
    site = (perturbed[:, :, 1] - perturbed[:, :, 0])[:, list(atom_indices)]
    valid = ~np.isnan(site)
    counts = valid.sum(axis=1)
    sums = np.where(valid, site, 0.0).sum(axis=1)
    return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def running_average(values):
    """Cumulative mean over frames ignoring NaN frames - the running value a correction converges to
    as more molecular-dynamics frames are included (Figure S8 panel B)."""
    values = np.asarray(values, dtype=float)
    valid = ~np.isnan(values)
    counts = np.cumsum(valid)
    sums = np.cumsum(np.where(valid, values, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / counts, np.nan)


def autocorrelation(values, max_lag=None):
    """Autocorrelation of the valid per-frame corrections at lags 0..max_lag (Figure S8 panel C).
    A fast drop to zero means frames are effectively independent."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    x = x - x.mean()
    n = len(x)
    if n < 2:
        return np.array([1.0])
    if max_lag is None:
        max_lag = n - 1
    denom = np.sum(x * x)
    return np.array([np.sum(x[:n - lag] * x[lag:]) / denom for lag in range(max_lag + 1)])


def frame_validity(perturbed):
    """Per frame, whether any shielding value was computed for that frame (a raw (n_frames,
    n_atoms, 2) perturbed-shielding array from load_perturbed_shieldings has NaN entries for atoms
    that were not computed). A frame with every entry NaN was skipped entirely by the DFT queue;
    Figure S8 panel D visualizes this mask across solvents for a representative solute."""
    return ~np.all(np.isnan(perturbed), axis=(1, 2))


def frame_validity_grid(hdf5_filepath, solute, solvents, explicit_solvation_model, shield_type="dft"):
    """Figure S8 panel D data: for one solute and one molecular-dynamics engine, the per-frame
    validity mask (frame_validity) for each solvent, stacked into a single boolean grid of shape
    (n_solvents, max_frames). Solvents with fewer trajectory frames than the longest one are padded
    with False (no data) past their own frame count, so every row lines up on a shared frame-index
    axis. Returns (grid, frame_counts), where frame_counts[i] is the true (unpadded) frame count for
    solvents[i], needed to normalize the frame-wise correction histograms in panel B."""
    masks = []
    for solvent in solvents:
        perturbed = load_perturbed_shieldings(hdf5_filepath, solute, solvent,
                                              explicit_solvation_model, shield_type)
        masks.append(frame_validity(perturbed))
    frame_counts = np.array([len(m) for m in masks])
    max_len = int(frame_counts.max()) if len(frame_counts) else 0
    grid = np.zeros((len(solvents), max_len), dtype=bool)
    for i, mask in enumerate(masks):
        grid[i, :len(mask)] = mask
    return grid, frame_counts


# ---------------------------------------------------------------------------
# Figures S11 / S13 (MagNET vs DFT corrections)

def compare_dft_nn(dft_df, nn_df, value_col, keys=("solute", "site", "nucleus")):
    """Merge the DFT and MagNET (NN) tables on the given keys and return, per site, the DFT and NN
    value of one correction column plus their difference (error = NN - DFT). Used for the
    rovibrational (qcd) comparison of Figure S11 and the explicit-solvent (desmond/openMM) comparison
    of Figure S13. The correction is tiled across NMR methods, so duplicate keyed rows are dropped
    before merging."""
    keys = list(keys)
    dft = dft_df[keys + [value_col]].drop_duplicates(subset=keys).dropna(subset=[value_col])
    nn = nn_df[keys + [value_col]].drop_duplicates(subset=keys).dropna(subset=[value_col])
    merged = dft.merge(nn, on=keys, suffixes=("_dft", "_nn"))
    merged["error"] = merged[f"{value_col}_nn"] - merged[f"{value_col}_dft"]
    return merged


def qcd_correction_by_site(dft_df, nn_df, nucleus="H"):
    """Per site, the DFT and MagNET (NN) rovibrational (QCD) correction for one nucleus, sorted by
    solute (SI Figure S11 panel C, the per-solute view behind panel A's aggregate scatter). QCD is a
    gas-phase rovibrational correction, so unlike the explicit-solvent per-site view (Figure S13 panel
    C) there is no solvent to split on: this is a plain wrapper around compare_dft_nn that keeps one
    nucleus and orders the rows for the per-solute strip plot."""
    merged = compare_dft_nn(dft_df, nn_df, "qcd", keys=("solute", "site", "nucleus"))
    sub = merged[merged["nucleus"] == nucleus]
    return sub.sort_values("solute", kind="stable").reset_index(drop=True)


def compare_dft_nn_by_engine(dft_df, nn_df, engines=("desmond", "openMM"),
                             keys=("solute", "site", "nucleus", "solvent")):
    """SI Figure S13 panels A and B data: compare_dft_nn run separately for each molecular-dynamics
    engine (Desmond and OpenMM), stacked into one table with a shared 'dft_value' / 'nn_value' /
    'error' column and an 'engine' label. compare_dft_nn alone cannot be stacked directly because it
    names its value columns after value_col (e.g. 'desmond_dft' vs 'openMM_dft'); this renames them
    to a common pair of columns first so both engines can share one axis."""
    frames = []
    for engine in engines:
        one = compare_dft_nn(dft_df, nn_df, engine, keys=keys)
        one = one.rename(columns={f"{engine}_dft": "dft_value", f"{engine}_nn": "nn_value"})
        one["engine"] = engine
        frames.append(one[list(keys) + ["dft_value", "nn_value", "error", "engine"]])
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Figure S5 (PCM Captures Bulk Solvent Effects)

# bulk solvent properties used to test what the PCM benefit tracks (TIP4P uses water's values)
SOLVENT_DIELECTRIC = {
    "chloroform": 4.81, "dichloromethane": 9.08, "methanol": 32.6, "TIP4P": 78.54,
    "trifluoroethanol": 26.7, "acetone": 21.01, "acetonitrile": 36.64, "dimethylsulfoxide": 47.0,
    "tetrahydrofuran": 7.52, "benzene": 2.28, "chlorobenzene": 5.69, "toluene": 2.38,
}
SOLVENT_POLARIZABILITY = {
    "chloroform": 8.5, "dichloromethane": 6.5, "methanol": 3.3, "TIP4P": 1.5,
    "trifluoroethanol": 4.5, "acetone": 6.4, "acetonitrile": 4.4, "dimethylsulfoxide": 7.9,
    "tetrahydrofuran": 8.0, "benzene": 10.3, "chlorobenzene": 12.4, "toluene": 12.3,
}


def pcm_benefit_per_solvent(query_df_dft, method, basis, geometry, solvents, n_splits, solutes,
                            nucleus="H", n_test=10):
    """Panel S5 data: per solvent, the percent reduction in mean test RMSE from adding the
    implicit-solvent (PCM) term, for one DFT method. Returns a Series indexed by solvent. A positive
    value means PCM lowered the error. Plotted against the bulk solvent properties above.

    The published Figure S5 aggregates the per-split test RMSEs with the mean (not the median); this
    is what reproduces the paper's Pearson R values exactly."""
    one = query_df_dft[(query_df_dft["sap_nmr_method"] == method)
                       & (query_df_dft["sap_basis"] == basis)
                       & (query_df_dft["sap_geometry_type"] == geometry)
                       & (query_df_dft["nucleus"] == nucleus)]
    one = add_composite_columns(one)
    results = run_fits(one, solvents, ["stationary", "stationary_plus_pcm"],
                       n_splits=n_splits, solutes=solutes, n_test=n_test)
    mean = results.groupby(["solvent", "formula"])["test_RMSE"].mean().unstack("formula")
    return -((mean["stationary_plus_pcm"] - mean["stationary"]) / mean["stationary"]) * 100.0


# ---------------------------------------------------------------------------
# Figure S7 (Explicit Solvent Corrections are Method-Independent)

def explicit_correction_pairs(combined_df, solvent, nucleus="H"):
    """Per site, the Desmond and OpenMM explicit-solvent corrections for one solvent (Figure S7).
    The explicit correction does not depend on the NMR method, so the same value is tiled across the
    method rows; duplicate site rows are dropped. Works on either the DFT or the NN combined table.
    A tight Desmond-vs-OpenMM agreement is the method-independence claim."""
    sub = combined_df[(combined_df["nucleus"] == nucleus) & (combined_df["solvent"] == solvent)]
    out = sub[["solute", "site", "desmond", "openMM"]].drop_duplicates(subset=["solute", "site"])
    return out.dropna(subset=["desmond", "openMM"]).reset_index(drop=True)


def explicit_correction_dft_nn_pairs(dft_df, nn_df, solvent, nucleus="H"):
    """Per site, the Desmond and OpenMM explicit-solvent corrections from both DFT and MagNET-x (NN)
    for one solvent and nucleus (Figure S13 panel C, the per-site view behind panel A's aggregate
    scatter). Built from two explicit_correction_pairs calls (one per source) merged on
    (solute, site), giving columns desmond_dft, openMM_dft, desmond_nn, openMM_nn."""
    dft_pairs = explicit_correction_pairs(dft_df, solvent, nucleus)
    nn_pairs = explicit_correction_pairs(nn_df, solvent, nucleus)
    return dft_pairs.merge(nn_pairs, on=["solute", "site"], suffixes=("_dft", "_nn"))


def si_s13d_fitting_accuracy(query_df_dft, query_df_nn, solvents, n_splits, solutes, nucleus="H",
                             method="dsd_pbep86", basis="pcSseg3", geometry="pbe0_tz", n_test=10):
    """Figure S13 panel D data: the semi-parsimonious composite model's (composite_models.py's
    best_semi formula) test RMSE against experiment, for every solvent, split by molecular-dynamics
    engine (Desmond vs OpenMM) and by the source of the explicit-solvent/vibrational term (DFT vs
    MagNET-x, "NN"). Protons use the QCD rovibrational correction (stationary_plus_qcd), which the
    SI text says is the same B3LYP/cc-pVDZ value for both DFT and NN bars; carbons use the classical
    vibrational correction computed from the same engine's trajectories (stationary_plus_desmond_vib
    or stationary_plus_openMM_vib), which does differ between the DFT and NN combined tables. Returns
    a tidy DataFrame with one row per (engine, source, solvent, seed) and a test_RMSE column."""
    engine_vib_column = {"desmond": "stationary_plus_des_vib", "openMM": "stationary_plus_op_vib"}
    rows = []
    for source_name, query_df in [("DFT", query_df_dft), ("NN", query_df_nn)]:
        sub = query_df[query_df["nucleus"] == nucleus]
        if source_name == "DFT":
            sub = sub[(sub["sap_nmr_method"] == method) & (sub["sap_basis"] == basis)
                     & (sub["sap_geometry_type"] == geometry)]
        sub = add_composite_columns(sub)
        for engine in ("desmond", "openMM"):
            base_term = "stationary_plus_qcd" if nucleus == "H" else engine_vib_column[engine]
            formula = f"{base_term} + {engine}"
            results = run_fits(sub, solvents, [formula], n_splits=n_splits, solutes=solutes,
                               n_test=n_test)
            results["engine"] = engine
            results["source"] = source_name
            rows.append(results)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Figure 2A / Figure S1 (the accuracy-vs-cost Pareto frontier)

# MagNET predicts the implicit-solvent (PCM) correction for chloroform only. To get a value for the
# other solvents, the chloroform correction is scaled by a per-solvent factor, fit on the delta-22
# DFT PCM corrections themselves: the slope (through the origin) of one method's PCM correction in
# that solvent against the reference method's PCM correction in chloroform. The reference (input)
# method is B3LYP-D3(BJ); the output method is the one MagNET was trained to reproduce (wP04 for
# proton, wB97X-D for carbon). These factors are also stored in applications.hdf5, but recomputing
# them here keeps Figure 2A self-contained within the delta-22 data.
MAGNET_PCM_INPUT_METHOD = "b3lyp_d3bj"
MAGNET_PCM_OUTPUT_METHODS = {"H": "wp04", "C": "wb97xd"}


def pcm_conversion_factors(query_df_dft, nucleus, output_method=None,
                           input_method=MAGNET_PCM_INPUT_METHOD,
                           geometry="aimnet2", basis="pcSseg2"):
    """The per-solvent factor that scales MagNET's chloroform PCM correction to each solvent, for
    one nucleus. Recomputed from the delta-22 DFT PCM corrections: for each solvent, the
    through-the-origin slope of the output method's PCM correction in that solvent against the input
    (reference) method's PCM correction in chloroform, over the shared sites. Returns a Series
    indexed by solvent in the reader's native naming (water is "TIP4P"). Matches the factors stored
    in applications.hdf5 to machine precision."""
    if output_method is None:
        output_method = MAGNET_PCM_OUTPUT_METHODS[nucleus]
    base = query_df_dft[(query_df_dft["sap_geometry_type"] == geometry)
                        & (query_df_dft["sap_basis"] == basis)
                        & (query_df_dft["nucleus"] == nucleus)]
    reference = (base[(base["sap_nmr_method"] == input_method) & (base["solvent"] == "chloroform")]
                 .set_index(["solute", "site"])["pcm"])
    output = base[base["sap_nmr_method"] == output_method]
    factors = {}
    for solvent in DESMOND_SOLVENTS:
        y = output[output["solvent"] == solvent].set_index(["solute", "site"])["pcm"]
        joined = pd.concat([reference.rename("x"), y.rename("y")], axis=1).dropna()
        xv = joined["x"].to_numpy(dtype=float)
        yv = joined["y"].to_numpy(dtype=float)
        factors[solvent] = float((xv * yv).sum() / (xv * xv).sum()) if len(xv) else np.nan
    return pd.Series(factors, name="pcm_conversion_factor")


def add_nn_stationary_plus_pcm(query_df_nn, factors_by_nucleus):
    """Add the MagNET stationary_plus_pcm column to a flat NN query table: the gas-phase shielding
    plus the chloroform PCM correction scaled by the per-solvent, per-nucleus factor from
    pcm_conversion_factors. factors_by_nucleus maps "H"/"C" to those Series. Returns a copy."""
    out = query_df_nn.copy()
    factor = out.apply(lambda r: factors_by_nucleus[r["nucleus"]].get(r["solvent"], np.nan), axis=1)
    out["stationary_plus_pcm"] = out["stationary"] + out["pcm"] * factor
    return out


# the columns the Pareto fit needs from each flat table, before the "sap_" prefix is stripped
_PARETO_COLUMNS = ["sap_geometry_type", "sap_nmr_method", "sap_basis", "nucleus", "site",
                   "solute", "solvent", "experimental", "stationary_plus_pcm"]


def assemble_pareto_fitting_df(query_df_dft, query_df_nn, conversion_factors=None,
                               exclude_solutes=("nitromethane",)):
    """Build the combined DFT + MagNET table the Pareto fit runs on: experimental shift against the
    single stationary_plus_pcm predictor, for every method/basis/geometry/nucleus/solvent. The DFT
    predictor is stationary + pcm; the MagNET predictor uses the scaled chloroform correction
    (conversion_factors, recomputed from query_df_dft if not given). TIP4P is renamed to water and
    the excluded solutes (nitromethane by default, an outlier) are removed. The DFT rows come first
    so the solute order is stable for the seeded splits."""
    if conversion_factors is None:
        conversion_factors = {nuc: pcm_conversion_factors(query_df_dft, nuc) for nuc in ("H", "C")}
    dft = query_df_dft.copy()
    dft["stationary_plus_pcm"] = dft["stationary"] + dft["pcm"]
    nn = add_nn_stationary_plus_pcm(query_df_nn, conversion_factors)
    combined = pd.concat([dft[_PARETO_COLUMNS], nn[_PARETO_COLUMNS]], ignore_index=True)
    combined = combined.rename(columns=lambda c: c.replace("sap_", ""))
    combined["solvent"] = combined["solvent"].replace("TIP4P", "water")
    combined["experimental"] = pd.to_numeric(combined["experimental"], errors="coerce")
    if exclude_solutes:
        combined = combined[~combined["solute"].isin(set(exclude_solutes))]
    return combined.reset_index(drop=True)


def pareto_solvent_averaged_rmse(fitting_df, solutes=None, n_splits=100, n_test=10,
                                 random_state=1, formula="stationary_plus_pcm"):
    """For every (geometry, method, basis, nucleus, solvent), the mean test RMSE of the
    experimental ~ stationary_plus_pcm fit over n_splits seeded per-solute splits, then a
    solvent-averaged row per (geometry, method, basis, nucleus). This is the accuracy axis of
    Figure 2A / Figure S1.

    Figure 2A splits differently from the other figures: it trains on the first n_test shuffled
    solutes and tests on the rest, seeding with random_state + split index (the other figures train
    on the remainder and seed with +100); swap=True on _split_fit_scores gets that train-on-first-
    n_test behavior. The per-split errors are averaged with the mean, not the median used elsewhere."""
    if solutes is None:
        solutes = fitting_df["solute"].unique().tolist()
    group_cols = ["geometry_type", "nmr_method", "basis", "nucleus", "solvent"]
    rows = []
    partitions = _solute_partitions(n_splits, solutes, n_test, seed_offset=random_state)
    for key, group_df in fitting_df.groupby(group_cols):
        solvent = key[group_cols.index("solvent")]
        scores = _split_fit_scores(group_df, solvent, [formula], partitions, swap=True)
        errors = [scores.get((formula, seed), (np.nan, np.nan))[1] for seed in range(n_splits)]
        with np.errstate(invalid="ignore"):
            mean_err = float(np.nanmean(errors)) if errors else np.nan
        rows.append({**dict(zip(group_cols, key)), "fitting_RMSE": mean_err})
    raw = pd.DataFrame(rows)
    averaged = (raw.groupby(["geometry_type", "nmr_method", "basis", "nucleus"])["fitting_RMSE"]
                .mean().reset_index())
    averaged["solvent"] = "solvent-averaged"
    return pd.concat([raw, averaged], ignore_index=True)


def load_pareto_timings(delta22_path):
    """The compute-time axis of Figure 2A: (dft_gas_timings, nn_timings). DFT timings are the
    gas-phase ('none' solvent) total time per (geometry, method, basis); MagNET timings sum the NMR
    time across its solvents and keep one geometry time. Reads the timing groups through the reader."""
    from delta22_reader import load_delta22_dft_timings, load_delta22_nn_timings
    dft = load_delta22_dft_timings(delta22_path)
    dft.index = dft.index.set_names([n.replace("sap_", "") for n in dft.index.names])
    dft = dft.droplevel("solvent_model")
    dft_gas = dft[dft.index.get_level_values("solvent") == "none"].droplevel("solvent")
    nn = load_delta22_nn_timings(delta22_path)
    nn.index = nn.index.set_names([n.replace("sap_", "") for n in nn.index.names])
    nn = nn.droplevel("solvent_model")
    nn = nn.groupby(["geometry_type", "nmr_method", "basis"]).agg(
        {"geometry_time": "first", "nmr_time": "sum"})
    nn["total_time"] = nn["geometry_time"] + nn["nmr_time"]
    return dft_gas, nn


def attach_pareto_timings(rmse_df, dft_gas_timings, nn_timings):
    """Join the gas-phase DFT timings onto the per-method RMSE rows and substitute the MagNET
    timings (which are not in the DFT gas table) for the MagNET rows. Returns rmse_df with
    geometry_time / nmr_time / total_time columns added."""
    out = rmse_df.join(dft_gas_timings, on=["geometry_type", "nmr_method", "basis"], how="left")
    magnet = out["nmr_method"] == "MagNET"
    nn_row = nn_timings.loc[("aimnet2", "MagNET", "N/A")]
    out.loc[magnet, ["geometry_time", "nmr_time", "total_time"]] = \
        nn_row[["geometry_time", "nmr_time", "total_time"]].to_numpy()
    return out


def fig2a_pareto_points(query_df_dft, query_df_nn, dft_gas_timings, nn_timings,
                        solutes=None, n_splits=100, n_test=10, random_state=1,
                        conversion_factors=None):
    """The full Figure 2A / Figure S1 table: one row per (geometry, method, basis, nucleus, solvent)
    plus the solvent-averaged rows, each carrying its mean test RMSE and its total compute time.
    Pass the flat DFT and NN query tables (build_query_df output) and the two timing tables from
    load_pareto_timings. The frontier for a given nucleus is pareto_frontier on the
    solvent-averaged rows with x="total_time", y="fitting_RMSE"."""
    fitting = assemble_pareto_fitting_df(query_df_dft, query_df_nn, conversion_factors)
    rmse = pareto_solvent_averaged_rmse(fitting, solutes, n_splits=n_splits, n_test=n_test,
                                        random_state=random_state)
    return attach_pareto_timings(rmse, dft_gas_timings, nn_timings)


def pareto_table_curated(points_df, nucleus, solvent="chloroform"):
    """SI Tables S1 (1H, nucleus="H") and S2 (13C, nucleus="C"): the curated 55-row subset of
    fig2a_pareto_points' full cross product that the source tables print. fig2a_pareto_points (and
    the SI Figure S1 scatter built from it) wants every aimnet2-geometry method/basis combination,
    each an unlabeled point on the plot; the printed tables instead keep only MagNET plus a single
    representative aimnet2-geometry DFT row (the same reference method each nucleus's MagNET-Zero
    output is trained against, MAGNET_PCM_OUTPUT_METHODS, at pcSseg2), alongside the full
    pbe0_tz-geometry grid (18 methods x 3 basis sets, minus mp2/pcSseg3 which was never computed, so
    53 rows). Sorted by total_time ascending to match the source table's row order."""
    sub = points_df[(points_df["nucleus"] == nucleus) & (points_df["solvent"] == solvent)]
    aimnet2_method = MAGNET_PCM_OUTPUT_METHODS[nucleus]
    keep = (
        (sub["geometry_type"] == "pbe0_tz")
        | (sub["nmr_method"] == "MagNET")
        | ((sub["geometry_type"] == "aimnet2") & (sub["nmr_method"] == aimnet2_method)
           & (sub["basis"] == "pcSseg2"))
    )
    return sub[keep].sort_values("total_time").reset_index(drop=True)


def pareto_frontier(points_df, x="total_time", y="test_RMSE"):
    """Return the rows of points_df on the lower-left Pareto frontier, i.e. the points for which
    no other point has both a smaller x (compute time) and a smaller y (error). Sorted by x.
    """
    ordered = points_df.sort_values([x, y]).reset_index(drop=True)
    frontier_rows = []
    best_y = np.inf
    for _, row in ordered.iterrows():
        if row[y] < best_y:
            frontier_rows.append(row)
            best_y = row[y]
    return pd.DataFrame(frontier_rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# correlation matrices (Figure 2B / Figure S4)

def correlation_matrix(query_df, value_col, sites_index, column_var):
    """Correlation matrix of value_col across the levels of column_var. Rows are the shared sites
    (sites_index, e.g. ["solute", "site"]); columns are the column_var values (e.g. "solvent" or
    "sap_nmr_method"). Returns a square DataFrame of Pearson correlations between the columns,
    computed on the sites where both columns have a value."""
    pivot = query_df.pivot_table(index=sites_index, columns=column_var, values=value_col)
    return pivot.corr()


# Composite-formula ablation workbook, "Correlations Between Features": correlations among the
# composite features within one nucleus and solvent.
FEATURE_CORRELATION_COLUMNS = ("stationary", "pcm", "desmond", "desmond_vib", "qcd")


def feature_correlation_matrix(query_df, nucleus, solvent, cols=FEATURE_CORRELATION_COLUMNS, squared=False):
    """Pearson correlation matrix (R^2 if squared=True) between the feature columns, within one
    nucleus and solvent, over sites where both features have a value. query_df must be filtered to a
    single (sap_nmr_method, sap_basis, sap_geometry_type); ablations_feature_correlations does that."""
    sub = query_df[(query_df["nucleus"] == nucleus) & (query_df["solvent"] == solvent)][list(cols)]
    sub = sub.dropna(how="all")
    if len(sub) < 2:
        return pd.DataFrame(np.nan, index=cols, columns=cols)
    corr = sub.corr(method="pearson")
    return corr ** 2 if squared else corr


def average_feature_correlation_matrix(query_df, nucleus, solvents, cols=FEATURE_CORRELATION_COLUMNS,
                                       squared=False):
    """Cell-by-cell nanmean of feature_correlation_matrix over the solvents (the ablation workbook's
    solvent-averaged matrix). nanmean so a solvent with too few sites for one feature pair drops out
    of that cell alone, not the whole average. Same single-level-of-theory requirement."""
    mats = [feature_correlation_matrix(query_df, nucleus, s, cols, squared) for s in solvents]
    stacked = np.stack([m.to_numpy() for m in mats])
    with np.errstate(invalid="ignore"):
        avg = np.nanmean(stacked, axis=0)
    return pd.DataFrame(avg, index=mats[0].index, columns=mats[0].columns)


def ablations_feature_correlations(query_df_dft, nucleus, solvents=None):
    """Solvent-averaged feature correlations for one nucleus from the unfiltered
    load_query_df_dft(...) output: filters to the reference level
    (MAGNET_PCM_OUTPUT_METHODS[nucleus], pcSseg2, aimnet2), then returns {"r": Pearson R matrix,
    "r2": R^2 matrix}. The single-level filter is required; pooling levels of theory gives wrong
    numbers."""
    solvents = list(solvents) if solvents is not None else sorted(query_df_dft["solvent"].unique())
    method = MAGNET_PCM_OUTPUT_METHODS[nucleus]
    sub = query_df_dft[(query_df_dft["nucleus"] == nucleus) & (query_df_dft["sap_nmr_method"] == method)
                       & (query_df_dft["sap_basis"] == "pcSseg2") & (query_df_dft["sap_geometry_type"] == "aimnet2")]
    return {"r": average_feature_correlation_matrix(sub, nucleus, solvents),
            "r2": average_feature_correlation_matrix(sub, nucleus, solvents, squared=True)}


def pcm_desmond_correlation_by_solvent(query_df_dft, solvents=None):
    """The ablation workbook's PCM-vs-Desmond table: per nucleus and solvent, the Pearson correlation
    between the PCM (implicit) and Desmond (explicit) corrections. Filters to the reference level
    per nucleus itself (same as ablations_feature_correlations). Returns a DataFrame indexed by
    nucleus, one column per solvent."""
    solvents = list(solvents) if solvents is not None else sorted(query_df_dft["solvent"].unique())
    rows = {}
    for nucleus in ("H", "C"):
        method = MAGNET_PCM_OUTPUT_METHODS[nucleus]
        nuc_df = query_df_dft[(query_df_dft["nucleus"] == nucleus) & (query_df_dft["sap_nmr_method"] == method)
                              & (query_df_dft["sap_basis"] == "pcSseg2") & (query_df_dft["sap_geometry_type"] == "aimnet2")]
        row = {}
        for solvent in solvents:
            sub = nuc_df[nuc_df["solvent"] == solvent][["pcm", "desmond"]].dropna()
            row[solvent] = float(sub["pcm"].corr(sub["desmond"])) if len(sub) > 1 else float("nan")
        rows[nucleus] = row
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# PCM benefit (Figure 3A / Figure S5): how much a correction reduces test error

def median_test_rmse(results_df, by):
    """Median test_RMSE grouped by the given columns (e.g. ["solvent", "formula"] or
    ["sap_nmr_method", "formula"]). A small convenience used by several figures."""
    return results_df.groupby(by)["test_RMSE"].median().reset_index()


# ---------------------------------------------------------------------------
# Figure 3 panel data (Explicit Solvation is Critical for Accuracy)

# the three solvent classes Figure 3 / S6 group by
SOLVENT_GROUPS = {
    "Polar Aprotic": ["chloroform", "dichloromethane", "tetrahydrofuran",
                      "acetonitrile", "dimethylsulfoxide", "acetone"],
    "Polar Protic": ["methanol", "TIP4P", "trifluoroethanol"],
    "Aromatic": ["benzene", "toluene", "chlorobenzene"],
}


def fig3a_pcm_benefit(query_df_dft, solvents, n_splits, solutes, nucleus="H", n_test=10):
    """Panel 3A data: for every DFT method/basis/geometry, the seeded test RMSE with and without the
    implicit-solvent term ("stationary" vs the single-column "stationary_plus_pcm"). The boxplot
    shows how much the PCM term lowers error across methods. One nucleus at a time (proton and
    carbon shieldings sit on different scales, so they are never pooled in a fit). Pass a query_df
    that has been through add_composite_columns. Returns the tidy run_fits results (one row per
    method/solvent/formula/seed)."""
    sub = query_df_dft[query_df_dft["nucleus"] == nucleus]
    return run_fits(sub, solvents, ["stationary", "stationary_plus_pcm"],
                    n_splits=n_splits, solutes=solutes,
                    group_cols=["sap_nmr_method", "sap_basis", "sap_geometry_type"], n_test=n_test)


def fig3a_pcm_benefit_by_solvent(query_df_dft, solvents, n_splits, solutes, nucleus="H", n_test=10):
    """Panel 3A data, split by solvent: for every DFT method and solvent, the per-split percent
    benefit of adding the implicit-solvent (PCM) term, 100 * (stationary - stationary_plus_pcm) /
    stationary. A positive value means PCM lowered test RMSE for that split; a negative value means
    PCM made it worse. Unlike fig3a_pcm_benefit (which reports raw RMSE pooled across solvents),
    this keeps one row per (method, solvent, seed), so a solvent such as benzene can be shown
    separately from chloroform and the aromatic-solvent-specific PCM penalty stays visible instead
    of being averaged away."""
    results = fig3a_pcm_benefit(query_df_dft, solvents, n_splits, solutes,
                                nucleus=nucleus, n_test=n_test)
    pivot = results.pivot_table(
        index=["sap_nmr_method", "sap_basis", "sap_geometry_type", "solvent", "seed"],
        columns="formula", values="test_RMSE").reset_index()
    pivot["percent_benefit"] = (100.0 * (pivot["stationary"] - pivot["stationary_plus_pcm"])
                                / pivot["stationary"])
    return pivot


def fig3b_shift_differences(query_df_dft, method, basis, geometry, nucleus="H",
                            x_solvent="methanol", y_solvent="benzene", reference="chloroform"):
    """Panel 3B data: per site, the experimental solvent-induced shift differences relative to a
    reference solvent (x = shift in x_solvent minus reference, y = shift in y_solvent minus
    reference) and the implicit-solvent (PCM) prediction of the same differences. One DFT method.
    The scatter compares whether PCM reproduces the measured solvent-to-solvent differences."""
    sub = query_df_dft[(query_df_dft["sap_nmr_method"] == method)
                       & (query_df_dft["sap_basis"] == basis)
                       & (query_df_dft["sap_geometry_type"] == geometry)
                       & (query_df_dft["nucleus"] == nucleus)]
    cols = ["solute", "site", "experimental", "pcm"]
    ref = sub[sub["solvent"] == reference][cols]
    xs = sub[sub["solvent"] == x_solvent][cols]
    ys = sub[sub["solvent"] == y_solvent][cols]
    merged = (xs.merge(ref, on=["solute", "site"], suffixes=("_x", "_ref"))
                .merge(ys, on=["solute", "site"]))
    merged.columns = ["solute", "site", "exp_x", "pcm_x", "exp_ref", "pcm_ref", "exp_y", "pcm_y"]
    merged["exp_diff_x"] = merged["exp_x"] - merged["exp_ref"]
    merged["exp_diff_y"] = merged["exp_y"] - merged["exp_ref"]
    merged["pcm_diff_x"] = merged["pcm_x"] - merged["pcm_ref"]
    merged["pcm_diff_y"] = merged["pcm_y"] - merged["pcm_ref"]
    return merged.dropna(subset=["exp_diff_x", "exp_diff_y"]).reset_index(drop=True)


def fig3d_formula_regressions(query_df_dft, method, basis, geometry, formulas, solvents,
                              n_splits, solutes, nucleus="H", n_test=10):
    """Panel 3D data: for one DFT method, the seeded test RMSE of an implicit-vs-explicit formula
    ladder per solvent. Pass a query_df that has been through add_composite_columns. Returns the
    tidy run_fits results (one row per solvent/formula/seed)."""
    one = query_df_dft[(query_df_dft["sap_nmr_method"] == method)
                       & (query_df_dft["sap_basis"] == basis)
                       & (query_df_dft["sap_geometry_type"] == geometry)
                       & (query_df_dft["nucleus"] == nucleus)]
    return run_fits(one, solvents, formulas, n_splits=n_splits, solutes=solutes, n_test=n_test)


# ---------------------------------------------------------------------------
# Figure 4 + SI S16/S17/S18 (solvent-induced shifts: experiment vs prediction)

def add_solvent_mean(query_df, group_cols=("solute", "site", "nucleus")):
    """Append a 'solvent_mean' pseudo-solvent: for each site, the mean over the real solvents of the
    numeric columns. Used as the solvent-averaged reference for Figure S18. Returns a new DataFrame
    with the extra rows."""
    means = query_df.groupby(list(group_cols)).mean(numeric_only=True).reset_index()
    means["solvent"] = "solvent_mean"
    return pd.concat([query_df, means], ignore_index=True)


def solvent_pair_differences(query_df, check_solvent, reference_solvent, nucleus="H",
                             explicit="desmond"):
    """Per (solute, site), the experimental and predicted correction differences between one solvent
    and a reference solvent (Figure 4): the experimental difference is reference minus check, and
    each correction difference (implicit PCM, explicit, and explicit+vibrations) is check minus
    reference. One DFT method's columns must already be selected."""
    a = query_df[(query_df["nucleus"] == nucleus) & (query_df["solvent"] == check_solvent)]
    b = query_df[(query_df["nucleus"] == nucleus) & (query_df["solvent"] == reference_solvent)]
    merged = a.merge(b, on=["solute", "site"], suffixes=(f"_{check_solvent}", f"_{reference_solvent}"))
    c, r, e = check_solvent, reference_solvent, explicit
    out = pd.DataFrame({
        "solute": merged["solute"], "site": merged["site"],
        "reference": reference_solvent, "solvent": check_solvent,
        "exp_diff": merged[f"experimental_{r}"] - merged[f"experimental_{c}"],
        "implicit_diff": merged[f"pcm_{c}"] - merged[f"pcm_{r}"],
        "explicit_diff": merged[f"{e}_{c}"] - merged[f"{e}_{r}"],
        "explicit_vib_diff": ((merged[f"{e}_{c}"] + merged[f"{e}_vib_{c}"])
                              - (merged[f"{e}_{r}"] + merged[f"{e}_vib_{r}"])),
    })
    return out.dropna(subset=["exp_diff"]).reset_index(drop=True)


def solvent_induced_shifts(query_df, reference, solvents, nucleus="H", explicit="desmond"):
    """The full set of solvent-vs-reference differences for one reference solvent (the data behind
    Figure S16 with chloroform, S17 with benzene, S18 with solvent_mean): every solvent except the
    reference, stacked. query_df should already include the solvent_mean rows if reference is
    'solvent_mean' (see add_solvent_mean)."""
    frames = [solvent_pair_differences(query_df, solvent, reference, nucleus, explicit)
              for solvent in solvents if solvent != reference]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fit_differences_to_experimental(diff_df, predicted_col):
    """Fit exp_diff = slope * predicted_col + intercept (the Figure 4C fit of the experimental
    solvent-induced shift differences to a correction's predicted differences) and return
    {slope, intercept, rmse, n}. Rows missing either value are dropped."""
    data = diff_df.dropna(subset=["exp_diff", predicted_col])
    x = data[predicted_col].to_numpy(dtype=float)
    y = data["exp_diff"].to_numpy(dtype=float)
    intercept, slope = linear_fit_1d(x, y)
    pred = intercept + slope * x
    return {"intercept": intercept, "slope": slope, "rmse": rmse(pred, y), "n": int(len(x))}
