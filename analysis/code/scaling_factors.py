"""Recommended Linear Scaling Parameters for MagNET-Zero / MagNET-PCM (SI Tables S10 and S11).

Turns MagNET-Zero shieldings into chemical shifts via a per-solvent linear model:

    shift = c_intercept + c_stationary * sigma_zero + c_pcm * delta_pcm

where sigma_zero is the MagNET-Zero gas-phase shielding and delta_pcm the MagNET-PCM chloroform
correction. Proton (S10) uses a THREE-parameter model (all coefficients free) because for benzene and
toluene the PCM correction points the wrong way and only a free c_pcm can go positive. Carbon (S11)
uses a TWO-parameter model: the gas and PCM terms share one per-solvent slope (PCM scaled to solvent
by a conversion factor first), which fit held-out carbons slightly better; the reported `pcm` is that
slope times the conversion factor.

The published tables are REFLECTION-SYMMETRIZED: each shielding is averaged over 20 forward passes
with the geometry mirrored for half (n_passes=10, symmetrize=True), cancelling the SO(3)-only model's
reflection-parity error. They ship verbatim in
data/scaling_factors/scaling_factors_symmetrized_{H,C}.csv; published_scaling_tables() returns them,
and build_scaling_tables(symmetrized=True) reproduces them from live inference (needs checkpoints).
build_scaling_tables(symmetrized=False) fits the raw single-pass shieldings in delta22.hdf5 instead
and lands ~0.01 ppm off. test_scaling_factors.py locks this down.
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

# published_scaling_tables() reads the shipped SI tables from here (provenance in the module docstring).
_SYMMETRIZED_CSV_DIR = os.path.join(_REPO, "data", "scaling_factors")


def published_scaling_tables():
    """The published SI scaling tables, {"H": Table S10, "C": Table S11}, as solvent-indexed
    DataFrames (columns intercept / stationary / pcm). Read from the shipped CSVs, so no delta-22
    download is needed; pass to predict_shift to turn MagNET-Zero/PCM shieldings into shifts. See the
    module docstring for provenance."""
    tables = {}
    for nucleus in ("H", "C"):
        path = os.path.join(_SYMMETRIZED_CSV_DIR, f"scaling_factors_symmetrized_{nucleus}.csv")
        tables[nucleus] = pd.read_csv(path).set_index("solvent")[_COLUMNS]
    return tables


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
    """Re-derive both recommended-scaling tables from the released delta-22 data.

    symmetrized=True reproduces the published SI Tables S10/S11 the way they were generated: live,
    reflection-symmetrized inference (scaling_factors_symmetrized.py). Needs the magnet package and
    model checkpoints, and is slow (re-runs inference on all 22 solutes).

    symmetrized=False (default) fits the raw single-pass MagNET-Zero/PCM shieldings stored in the
    HDF5, which carry the reflection-parity error the published tables correct, so it lands ~0.01 ppm
    off the published SI. Use it for a checkpoint-free re-derivation.

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
    # symmetrized=False: the checkpoint-free raw-data fit, within ~0.01 ppm of the published SI
    # (published_scaling_tables() returns the exact symmetrized SI values).
    tables = build_scaling_tables(h5, xlsx)
    for nucleus, label in (("H", "Table S10 (1H)"), ("C", "Table S11 (13C)")):
        print(f"\n=== {label} (unsymmetrized re-derivation): {RECOMMENDED_MODEL[nucleus]} model ===")
        print(tables[nucleus].round(6).to_string())
