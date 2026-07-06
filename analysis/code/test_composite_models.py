"""Tests for analysis/code/composite_models.py.

The fitting side is exercised on a small synthetic delta-22-shaped query table (composite_models
adds no new fitting logic of its own -- it calls delta22.py's harness directly, which has its own
tests). What's specific to this module is the pivot/reindex shape and the openpyxl table-writing
layer, so those get direct, hermetic tests here: the formula lookup helpers, the %benefit formula
writer, and the full workbook structure (sheet names, header cells, RMSE cell placement) on
synthetic data. An opt-in test checks the real data reproduces a plausible RMSE range.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "data", "delta22"))  # delta22_reader
sys.path.insert(0, HERE)
import paths

import composite_models as CM  # noqa: E402


def _synthetic_query_df_dft(solutes, lots, seed=0):
    """A query_df_dft-shaped table with all the columns composite_models' formulas reference, at
    each of the given (method, basis, geometry) triples, for both nuclei."""
    rng = np.random.default_rng(seed)
    rows = []
    for solute in solutes:
        for solvent in CM.ORDERED_SOLVENTS:
            for nucleus in ["H", "C"]:
                for method, basis, geometry in lots:
                    stationary = rng.normal(30 if nucleus == "H" else 150, 5)
                    pcm = rng.normal(0, 0.5)
                    desmond = rng.normal(0, 0.5)
                    qcd = rng.normal(0, 0.2)
                    desmond_vib = rng.normal(0, 0.3)
                    openMM = rng.normal(0, 0.5)
                    openMM_vib = rng.normal(0, 0.3)
                    experimental = stationary + 0.5 * pcm + 0.3 * desmond + rng.normal(0, 0.05)
                    rows.append(dict(
                        solute=solute, solvent=solvent, nucleus=nucleus,
                        sap_nmr_method=method, sap_basis=basis, sap_geometry_type=geometry,
                        stationary=stationary, pcm=pcm, desmond=desmond, qcd=qcd,
                        desmond_vib=desmond_vib, openMM=openMM, openMM_vib=openMM_vib,
                        experimental=experimental))
    return pd.DataFrame(rows)


SOLUTES = [f"m{i}" for i in range(16)]
DSD_LOT = ("dsd_pbep86", "pcSseg3", "pbe0_tz")
WP04_LOT = ("wp04", "pcSseg2", "aimnet2")
WB97XD_LOT = ("wb97xd", "pcSseg2", "aimnet2")


def test_canonical_formula_name_ignores_case_spaces_underscores():
    assert CM._canonical_formula_name("stationary + PCM") == CM._canonical_formula_name("stationary+pcm")
    assert CM._canonical_formula_name("stationary_plus_pcm") != CM._canonical_formula_name("stationary + pcm")


def test_get_rmse_looks_up_by_canonical_name():
    df = pd.DataFrame({"chloroform": [0.123]}, index=["stationary + pcm"])
    lookup = CM._build_formula_lookup(df)
    assert CM._get_rmse(df, lookup, "stationary+PCM", "chloroform") == pytest.approx(0.123)


def test_get_rmse_missing_formula_raises_keyerror():
    df = pd.DataFrame({"chloroform": [0.1]}, index=["stationary"])
    lookup = CM._build_formula_lookup(df)
    with pytest.raises(KeyError):
        CM._get_rmse(df, lookup, "not_a_real_formula", "chloroform")


def test_ablation_rmse_table_shape_and_exact_recovery():
    # experimental = stationary + 0.5*pcm + 0.3*desmond exactly (no noise), so a formula containing
    # both pcm and qcd should fit near-perfectly (tiny test RMSE) while "stationary" alone
    # (missing both real predictors) should not. "stationary + pcm + qcd" is a real formula in
    # CM.FORMULAS; there is no formula combining pcm and desmond together, so the ground truth is
    # built from pcm and qcd instead.
    rng = np.random.default_rng(1)
    rows = []
    for solute in SOLUTES:
        for solvent in CM.ORDERED_SOLVENTS:
            stationary = rng.normal(30, 5)
            pcm = rng.normal(0, 0.5)
            desmond = rng.normal(0, 0.5)
            qcd = rng.normal(0, 0.2)
            desmond_vib = rng.normal(0, 0.3)
            openMM = rng.normal(0, 0.5)
            openMM_vib = rng.normal(0, 0.3)
            experimental = stationary + 0.5 * pcm + 0.3 * qcd
            rows.append(dict(solute=solute, solvent=solvent, nucleus="H",
                             sap_nmr_method="dsd_pbep86", sap_basis="pcSseg3",
                             sap_geometry_type="pbe0_tz", stationary=stationary, pcm=pcm,
                             desmond=desmond, qcd=qcd, desmond_vib=desmond_vib, openMM=openMM,
                             openMM_vib=openMM_vib, experimental=experimental))
    query_df_dft = pd.DataFrame(rows)

    table = CM.ablation_rmse_table(query_df_dft, "H", "dsd_pbep86", "pcSseg3", "pbe0_tz", SOLUTES,
                                   n_splits=10)
    assert list(table.index) == CM.FORMULAS
    assert list(table.columns) == CM.ORDERED_SOLVENTS + ["Mean Test RMSE"]
    assert table.loc["stationary + pcm + qcd", "Mean Test RMSE"] < 0.05
    assert table.loc["stationary", "Mean Test RMSE"] > table.loc["stationary + pcm + qcd", "Mean Test RMSE"]


def test_write_nucleus_table_cell_placement():
    query_df_dft = _synthetic_query_df_dft(SOLUTES, [DSD_LOT])
    table = CM.ablation_rmse_table(query_df_dft, "H", *DSD_LOT, SOLUTES, n_splits=5)

    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    next_row = CM.write_nucleus_table(
        ws=ws, start_row=1, nucleus_name="Proton", stationary_desc="DSD-PBEP86/pcSseg3",
        pcm_desc="B3LYP-D3BJ/pcSseg3", test_rmse_df=table, vmin=0.06, vmax=0.28,
        formula_config=CM.PROTON_FORMULA_CONFIG)

    assert ws["A1"].value == "Proton"
    assert ws["A6"].value == "formula"
    assert ws["A9"].value == "stationary"
    assert ws["B9"].value == pytest.approx(table.loc["stationary", "chloroform"])
    # the "Mean Test RMSE" column is the last of the 13 data columns (B..N)
    assert ws["N5"].value == "Mean Test RMSE"
    assert ws["N9"].value == pytest.approx(table.loc["stationary", "Mean Test RMSE"])
    # %benefit rows carry a live Excel formula, not a precomputed number
    assert str(ws["B14"].value).startswith("=IFERROR(")
    assert next_row == 1 + 45


def test_build_ablations_workbook_single_level_two_sheets():
    query_df_dft = _synthetic_query_df_dft(SOLUTES, [DSD_LOT])
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "ablations.xlsx")
        CM.build_ablations_workbook(query_df_dft, SOLUTES, out, reference_levels=("dsd",),
                                    n_splits=5, verbose=False)
        from openpyxl import load_workbook
        wb = load_workbook(out)
        assert wb.sheetnames == ["Proton", "Carbon"]


def test_build_ablations_workbook_both_levels_four_sheets_matching_shipped_names():
    query_df_dft = _synthetic_query_df_dft(SOLUTES, [DSD_LOT, WP04_LOT, WB97XD_LOT])
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "ablations.xlsx")
        CM.build_ablations_workbook(query_df_dft, SOLUTES, out, n_splits=5, verbose=False)
        from openpyxl import load_workbook
        wb = load_workbook(out)
        # matches wolford/ablations_updated 2.xlsx's exact sheet names
        assert wb.sheetnames == ["Proton (DSD)", "Carbon (DSD)", "Proton (WP04)", "Carbon (wB97XD)"]


# --- opt-in: reproduce a plausible ablation table from the real delta-22 data ---------------------

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
REAL_H5 = paths.dataset_file("delta22", root=REPO)
REAL_XLSX = os.path.join(REPO, "data", "delta22", "delta22_experimental.xlsx")


@pytest.mark.skipif(not (os.path.exists(REAL_H5) and os.path.exists(REAL_XLSX)),
                    reason="real delta22.hdf5 / experimental xlsx not present")
def test_ablation_rmse_table_reproduces_plausible_range_from_real_data():
    import delta22 as D
    query_df_dft = D.add_composite_columns(D.load_query_df_dft(REAL_H5, REAL_XLSX, verbose=False))
    solutes = D.delta22_solutes(REAL_H5)
    level = CM.REFERENCE_LEVELS["dsd"]
    table = CM.ablation_rmse_table(query_df_dft, "H", level["method_h"], level["basis_h"],
                                   level["geometry_h"], solutes, n_splits=20)
    vmin, vmax = CM._VMIN_VMAX["H"]
    # the shipped workbook's fixed color-scale range is itself a real-data-derived sanity bound:
    # every formula's mean test RMSE should land inside it
    assert (table["Mean Test RMSE"] >= vmin * 0.5).all()
    assert (table["Mean Test RMSE"] <= vmax * 1.5).all()
    # the fullest model should not be worse than the bare stationary reference
    assert table.loc["stationary + desmond + qcd", "Mean Test RMSE"] < table.loc["stationary", "Mean Test RMSE"]
