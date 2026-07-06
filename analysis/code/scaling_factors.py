"""Recommended Linear Scaling Parameters for MagNET-Zero / MagNET-PCM (SI Tables S10 and S11).

Turns MagNET-Zero shieldings into chemical shifts via a per-solvent linear model:

    shift = c_intercept + c_stationary * sigma_zero + c_pcm * delta_pcm

where sigma_zero is the MagNET-Zero gas-phase shielding and delta_pcm the MagNET-PCM chloroform
correction. Proton (S10) uses a THREE-parameter model (all coefficients free) because for benzene and
toluene the PCM correction points the wrong way and only a free c_pcm can go positive. Carbon (S11)
uses a TWO-parameter model: the gas and PCM terms share one per-solvent slope (PCM scaled to solvent
by a conversion factor first), which fit held-out carbons slightly better; the reported `pcm` is that
slope times the conversion factor.

Everything reproduces from the released delta-22 data to the precision of its integer encoding
(`test_scaling_factors.py` locks it down). `build_scaling_tables(..., symmetrized=True)` instead
computes a separate deployment variant via reflection-symmetrized inference
(`scaling_factors_symmetrized.py`); it does not match these tables and is not what the SI reports.
"""
import os
import sys

import numpy as np
import pandas as pd

from paths import repo_root, ensure_on_path, dataset_file

# paths.py is the bootstrap at analysis/code/ root; the other shared utilities (stats, etc.) live in
# analysis/code/shared/, and the delta-22 reader/harness is reused from data/delta22. Put all of
# these on sys.path before importing them, so scaling_factors can be imported standalone (as its
# docstring invites), not only under pytest/notebooks that already pre-wire the paths.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = repo_root(__file__)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
ensure_on_path("analysis", "code", "shared", file=__file__)
ensure_on_path("data", "delta22", file=__file__)

from stats import linear_fit_1d  # noqa: E402
import delta22 as D  # noqa: E402

# The 12 solvents, in the reader's native naming (water is stored as "TIP4P").
SOLVENTS = tuple(D.DESMOND_SOLVENTS)

# Nitromethane is an outlier for both nuclei (its carbon shift is anomalous); the published tables
# reproduce only with it excluded throughout.
EXCLUDE_SOLUTES = ("nitromethane",)

# The model form recommended for each nucleus.
RECOMMENDED_MODEL = {"H": "three_parameter", "C": "two_parameter"}

_COLUMNS = ["intercept", "stationary", "pcm"]

# The published SI Tables S10 (proton) and S11 (carbon), shipped verbatim so you can convert
# shieldings to shifts WITHOUT the delta-22 data download. These are exactly what
# build_scaling_tables() reproduces from delta-22; test_scaling_factors.py asserts the two agree, so
# they cannot silently drift. Prefer published_scaling_tables() for a quick lookup, and
# build_scaling_tables() when you want to re-derive them from the raw data.
_PUBLISHED_TABLE_CSV = {
    "H": """solvent,intercept,stationary,pcm
tetrahydrofuran,31.321785,-0.980175,-0.786672
dichloromethane,31.377310,-0.979870,-0.808573
chloroform,31.294992,-0.975794,-0.852690
toluene,31.716957,-0.996733,1.944856
benzene,31.987569,-1.005289,2.236515
chlorobenzene,31.681438,-0.993846,0.830236
acetone,31.512157,-0.987261,-1.238414
dimethylsulfoxide,31.598698,-0.991103,-1.355248
acetonitrile,31.496110,-0.985716,-0.974465
trifluoroethanol,30.876004,-0.959997,-0.958852
methanol,31.256030,-0.976446,-1.250121
TIP4P,31.359134,-0.978809,-1.478298
""",
    "C": """solvent,intercept,stationary,pcm
tetrahydrofuran,171.054488,-0.919000,-1.069509
dichloromethane,171.509389,-0.921167,-1.115015
chloroform,171.728797,-0.924231,-0.936804
toluene,171.690907,-0.924987,-0.618410
benzene,171.967177,-0.927033,-0.593716
chlorobenzene,171.075910,-0.921238,-0.997641
acetone,171.308023,-0.919610,-1.239503
dimethylsulfoxide,170.598425,-0.918674,-1.296501
acetonitrile,171.879839,-0.922629,-1.287664
trifluoroethanol,174.237671,-0.939012,-1.290073
methanol,172.426161,-0.927566,-1.288847
TIP4P,173.696379,-0.937854,-1.342668
""",
}


def published_scaling_tables():
    """The published SI scaling tables, {"H": Table S10, "C": Table S11}, as solvent-indexed
    DataFrames. No data download needed; use these with predict_shift to turn MagNET-Zero/PCM
    shieldings into chemical shifts."""
    import io
    return {nucleus: pd.read_csv(io.StringIO(csv)).set_index("solvent")
            for nucleus, csv in _PUBLISHED_TABLE_CSV.items()}


def proton_scaling_table(query_df_nn, solvents=SOLVENTS, exclude_solutes=EXCLUDE_SOLUTES):
    """SI Table S10: per-solvent [intercept, stationary, pcm] for the proton three-parameter model.

    Fits experimental ~ stationary + pcm freely for each solvent on the MagNET-Zero shielding and the
    MagNET-PCM (chloroform) correction. Returns a DataFrame indexed by solvent.
    """
    nn = query_df_nn[(query_df_nn["nucleus"] == "H")
                     & ~query_df_nn["solute"].isin(set(exclude_solutes))]
    coeffs = D.full_fit_coefficients(nn, list(solvents), "stationary + pcm")
    table = coeffs.T.rename(columns={"Intercept": "intercept"})[_COLUMNS]
    table.index.name = "solvent"
    return table


def carbon_conversion_factors(query_df_dft):
    """The per-solvent PCM conversion factor for carbon (delta22.pcm_conversion_factors): for each
    solvent, the through-the-origin slope of that solvent's PCM correction (at the level MagNET-Zero
    reproduces, wB97X-D) against chloroform's B3LYP-D3(BJ) correction (the level MagNET-PCM
    reproduces), from the delta-22 DFT PCM data. So one factor converts both chloroform to the solvent
    and B3LYP to wB97X-D. A Series indexed by solvent."""
    return D.pcm_conversion_factors(query_df_dft, "C")


def carbon_scaling_table(query_df_nn, query_df_dft=None, solvents=SOLVENTS,
                         exclude_solutes=EXCLUDE_SOLUTES, conversion_factors=None):
    """SI Table S11: per-solvent [intercept, stationary, pcm] for the carbon two-parameter model.

    For each solvent, fits experimental ~ (stationary + factor * pcm) with a single shared slope,
    where `factor` is the per-solvent PCM conversion factor. The reported `stationary` is that slope
    and the reported `pcm` is slope * factor, so the prediction equation in the module docstring
    applies with MagNET-PCM's chloroform correction. The factors come from query_df_dft
    (carbon_conversion_factors) unless passed in as `conversion_factors` (a Series or dict). A solvent
    with fewer than two usable sites, or an undefined conversion factor, gets a row of NaN rather than
    a fake zero fit. Returns a DataFrame indexed by solvent.
    """
    if conversion_factors is None:
        if query_df_dft is None:
            raise ValueError("provide query_df_dft or conversion_factors")
        conversion_factors = carbon_conversion_factors(query_df_dft)
    nn = query_df_nn[(query_df_nn["nucleus"] == "C")
                     & ~query_df_nn["solute"].isin(set(exclude_solutes))]
    rows = {}
    for solvent in solvents:
        factor = float(conversion_factors[solvent])
        sub = nn[nn["solvent"] == solvent]
        y = pd.to_numeric(sub["experimental"], errors="coerce").to_numpy(float)
        x = sub["stationary"].to_numpy(float) + factor * sub["pcm"].to_numpy(float)
        keep = np.isfinite(x) & np.isfinite(y)
        if not np.isfinite(factor) or keep.sum() < 2:
            rows[solvent] = {"intercept": np.nan, "stationary": np.nan, "pcm": np.nan}
            continue
        intercept, slope = linear_fit_1d(x[keep], y[keep])
        rows[solvent] = {"intercept": intercept, "stationary": slope, "pcm": slope * factor}
    table = pd.DataFrame.from_dict(rows, orient="index")[_COLUMNS]
    table.index.name = "solvent"
    return table


def build_scaling_tables(delta22_path, experimental_path, symmetrized=False, n_passes=10):
    """Reproduce both recommended-scaling tables from the released delta-22 data.

    symmetrized=False (default) reproduces the published SI Tables S10/S11 exactly, using the
    HDF5's stored MagNET-Zero/PCM shieldings -- this is what test_scaling_factors.py checks against
    and what published_scaling_tables() ships.

    symmetrized=True instead computes the shieldings via live, symmetrized inference
    (predict_shieldings(..., n_passes=n_passes, symmetrize=True), see
    scaling_factors_symmetrized.py), which corrects a reflection-parity bug in the stored HDF5
    values (they come from a single unsymmetrized forward pass). Use this for a deployment-quality
    scaling table -- e.g. serving MagNET-Zero/MagNET-PCM on new "everyday" molecules -- rather than
    for reproducing the SI. Requires the magnet package and released model checkpoints; slower
    (re-runs inference on all 22 solutes instead of reading pre-baked values).

    Returns {"H": Table S10 DataFrame, "C": Table S11 DataFrame}.
    """
    if symmetrized:
        import scaling_factors_symmetrized as _S
        override_df = _S.compute_symmetrized_nn_shieldings_df(delta22_path, n_passes=n_passes,
                                                               verbose=False)
        nn = D.load_query_df_nn(delta22_path, experimental_path, verbose=False,
                                nn_shieldings_override_df=override_df)
    else:
        nn = D.load_query_df_nn(delta22_path, experimental_path, verbose=False)
    dft = D.load_query_df_dft(delta22_path, experimental_path, verbose=False)
    return {"H": proton_scaling_table(nn), "C": carbon_scaling_table(nn, dft)}


def predict_shift(table, solvent, magnet_zero_shielding, magnet_pcm_chloroform_correction):
    """Apply a scaling table to MagNET-Zero / MagNET-PCM outputs to predict a chemical shift.

    table: a proton or carbon scaling table (DataFrame indexed by solvent). Get one from
        published_scaling_tables()[nucleus] (no data download) or build_scaling_tables(...)[nucleus]
        (re-derived from delta-22).
    solvent: the solvent name (water is "TIP4P").
    magnet_zero_shielding: the MagNET-Zero gas-phase shielding (scalar or array).
    magnet_pcm_chloroform_correction: MagNET-PCM's chloroform correction (scalar or array).

    The same equation serves both nuclei because the stored `pcm` coefficient already folds in the
    per-solvent scaling, so you always pass the chloroform correction.

    End to end, from a geometry (the H/C shieldings each model returns are per-atom arrays):

        from magnet.run_magnet import compute_MagNET_Zero_shieldings, compute_MagNET_PCM_corrections
        zero = compute_MagNET_Zero_shieldings([Z], [xyz])[0]          # MagNET-Zero shieldings
        pcm  = compute_MagNET_PCM_corrections([Z], [xyz])[0]          # MagNET-PCM chloroform correction
        tables = published_scaling_tables()
        carbons = Z == 6
        shifts_13C = predict_shift(tables["C"], "benzene", zero[carbons], pcm[carbons])
    """
    row = table.loc[solvent]
    return (row["intercept"]
            + row["stationary"] * np.asarray(magnet_zero_shielding, dtype=float)
            + row["pcm"] * np.asarray(magnet_pcm_chloroform_correction, dtype=float))


if __name__ == "__main__":
    h5 = dataset_file("delta22", root=_REPO)
    xlsx = os.path.join(_REPO, "data", "delta22", "delta22_experimental.xlsx")
    tables = build_scaling_tables(h5, xlsx)
    for nucleus, label in (("H", "Table S10 (1H)"), ("C", "Table S11 (13C)")):
        print(f"\n=== {label}: {RECOMMENDED_MODEL[nucleus]} model ===")
        print(tables[nucleus].round(6).to_string())
