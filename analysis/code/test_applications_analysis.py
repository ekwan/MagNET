"""Tests for analysis/code/applications.py (the no-plotting analysis module).

Builds a tiny synthetic applications.hdf5 + spreadsheet so the correction math can be checked
in CI without the multi-hundred-megabyte release files.
"""
import os
import sys

import numpy as np
import pandas as pd
import h5py
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "data", "applications"))
sys.path.insert(0, HERE)
import paths

from applications_reader import Applications, SHELL_SIZES, SOLVENTS  # noqa: E402
import applications as A  # noqa: E402

SCALE = 1e4
MISS = np.int32(-2147483648)


def test_distribution_shift_by_solvent_table_averages_per_solvent():
    """distribution_shift_by_solvent_table averages the three coefficient-choice RMSEs per solvent,
    and surfaces the water column as the biggest extrapolation gap (the S15 story)."""
    remapped = A.FORMULA_REMAP["stationary_plus_qcd + openMM"]
    per_solute = {"H": pd.DataFrame([
        {"formula": remapped, "solvent": "chloroform", "solute": "a", "rmse": 0.10},
        {"formula": remapped, "solvent": "chloroform", "solute": "b", "rmse": 0.20},
        {"formula": remapped, "solvent": "TIP4P", "solute": "a", "rmse": 0.30},
        {"formula": remapped, "solvent": "TIP4P", "solute": "b", "rmse": 0.50},
    ])}
    per_solvent = {"H": pd.DataFrame([
        {"formula": remapped, "solvent": "chloroform", "rmse": 0.16},
        {"formula": remapped, "solvent": "TIP4P", "rmse": 0.45},
    ])}
    bootstrap = pd.DataFrame([
        {"nucleus": "H", "formula": remapped, "solvent": "chloroform", "solute": "a", "seed": 0, "Bootstrap_RMSE": 0.14},
        {"nucleus": "H", "formula": remapped, "solvent": "chloroform", "solute": "b", "seed": 0, "Bootstrap_RMSE": 0.18},
        {"nucleus": "H", "formula": remapped, "solvent": "TIP4P", "solute": "a", "seed": 0, "Bootstrap_RMSE": 0.60},
        {"nucleus": "H", "formula": remapped, "solvent": "TIP4P", "solute": "b", "seed": 0, "Bootstrap_RMSE": 0.80},
    ])
    table = A.distribution_shift_by_solvent_table(per_solute, per_solvent, bootstrap, "H",
                                                  "stationary_plus_qcd + openMM")
    assert list(table.index) == ["TIP4P", "chloroform"]  # sorted; uppercase sorts before lowercase
    assert list(table.columns) == ["Extrapolated from delta22", "Scaled to Test Set", "Scaled to Solute"]
    assert table.loc["chloroform", "Scaled to Solute"] == pytest.approx(0.15)
    assert table.loc["chloroform", "Scaled to Test Set"] == pytest.approx(0.16)
    assert table.loc["chloroform", "Extrapolated from delta22"] == pytest.approx(0.16)
    assert table.loc["TIP4P", "Scaled to Solute"] == pytest.approx(0.40)
    assert table.loc["TIP4P", "Extrapolated from delta22"] == pytest.approx(0.70)
    chl_gap = table.loc["chloroform", "Extrapolated from delta22"] - table.loc["chloroform", "Scaled to Solute"]
    water_gap = table.loc["TIP4P", "Extrapolated from delta22"] - table.loc["TIP4P", "Scaled to Solute"]
    assert water_gap > chl_gap


def _enc(a):
    a = np.asarray(a, dtype=np.float64)
    nan = np.isnan(a)
    out = np.round(np.where(nan, 0.0, a) * SCALE).astype(np.int32)
    out[nan] = MISS
    return out


@pytest.fixture
def loader(tmp_path):
    """One solute (vomicine), 3 atoms, one H site over atoms 1+2, one C site over atom 3.
    magnet_x is set so the solvated-minus-isolated correction is a known constant per shell."""
    h5 = tmp_path / "applications.hdf5"
    xlsx = tmp_path / "exp.xlsx"
    with h5py.File(h5, "w") as f:
        sg = f.create_group("solvents")
        for sv in SOLVENTS:
            sg.create_group(sv).create_dataset("atomic_numbers", data=np.array([6], np.int32))
        g = f.create_group("vomicine")
        g.create_dataset("atomic_numbers", data=np.array([1, 1, 6], np.int32))
        mx = g.create_group("magnet_x")
        for sv in SOLVENTS:
            svg = mx.create_group(sv)
            for k, shell in enumerate(SHELL_SIZES):
                arr = np.zeros((4, 3, 2), dtype=np.float64)
                arr[:, :, 0] = 10.0                 # isolated
                arr[:, :, 1] = 10.0 + 0.1 * (k + 1)  # solvated -> correction = 0.1*(k+1)
                svg.create_dataset(f"shell_{shell}", data=_enc(arr))
            svg.create_dataset("stationary", data=_enc(np.full(3, 10.0)))
        mz = g.create_group("magnet_zero")
        mz.create_dataset("stationary", data=_enc(np.array([100.0, 101.0, 50.0])))
        mz.create_dataset("pcm_correction", data=_enc(np.array([-1.0, -1.0, -2.0])))
        q = g.create_group("qcd")
        q.create_dataset("stationary", data=_enc(np.zeros((3, 4))))
        q.create_dataset("trajectories", data=_enc(np.zeros((2, 3, 3, 4))))  # corr -> 0
        cm = f.create_group("composite_model")
        cmc = cm.create_group("pcm_conversion_factors")
        # the real conversion table uses 'water' (renamed to TIP4P by build_query_df_nn), never both
        rows = "solvent,pcm_conversion_factor\n" + "".join(
            f"{s},1.0\n" for s in ["chloroform", "benzene", "methanol", "water"])
        cmc.create_dataset("H", data=rows)
        cmc.create_dataset("C", data=rows)
    with pd.ExcelWriter(xlsx) as w:
        pd.DataFrame({
            "site": ["H", "1 C"], "nucleus": ["H", "C"],
            "atom_numbers": ["1,2", "3"],
            "chloroform": [1.0, 2.0], "benzene": [1.0, 2.0],
            "methanol": [1.0, 2.0], "water": [1.0, 2.0],
        }).to_excel(w, sheet_name="vomicine", index=False)
    return Applications(h5, xlsx)


def test_shell_convergence_values(loader):
    df = A.shell_convergence_corrections(loader, "vomicine", "H")
    # one H site (uuid-prefixed) x 4 solvents
    assert df.index.get_level_values("solvent").nunique() == len(SOLVENTS)
    assert df.index.get_level_values("site").unique().tolist() == ["01 H"]
    # correction for shell k (0-based) is exactly 0.1*(k+1) within the int32 quantum
    row = df.xs("chloroform", level="solvent").iloc[0]
    for k, shell in enumerate(SHELL_SIZES):
        assert row[f"shell_{shell}"] == pytest.approx(0.1 * (k + 1), abs=5e-5)


def test_only_requested_nucleus(loader):
    df = A.shell_convergence_corrections(loader, "vomicine", "H")
    # C site must not appear in an H-nucleus convergence table
    sites = df.index.get_level_values("site").unique().tolist()
    assert all("C" not in s for s in sites)


def test_site_atom_indices():
    assert A.site_atom_indices("1,2") == [0, 1]
    assert A.site_atom_indices("21") == [20]


def test_fit_recovers_known_line():
    # experimental = 2*stationary + 5 exactly -> intercept ~5, slope ~2, RMSE ~0
    df = pd.DataFrame({"stationary": [1.0, 2.0, 3.0, 4.0], "experimental": [7.0, 9.0, 11.0, 13.0]})
    rmse, params = A.fit(df, "stationary")
    assert rmse < 1e-9
    assert params["Intercept"] == pytest.approx(5.0, abs=1e-6)
    assert params["stationary"] == pytest.approx(2.0, abs=1e-6)


def test_fit_ignores_rows_with_no_experimental_value():
    # one site has no experimental measurement (common in the real natural-products spreadsheet);
    # it must not turn the whole RMSE into NaN, only be left out of the score
    df = pd.DataFrame({
        "stationary": [1.0, 2.0, 3.0, 4.0],
        "experimental": [7.0, 9.0, 11.0, np.nan],
    })
    rmse, params = A.fit(df, "stationary")
    assert rmse < 1e-9
    assert params["stationary"] == pytest.approx(2.0, abs=1e-6)


def test_fit_matches_statsmodels():
    # the numpy fit must be numerically identical to the original statsmodels OLS oracle, across
    # multi-term formulas and with missing values present (rows dropped identically by both).
    rng = np.random.default_rng(7)
    n = 60
    df = pd.DataFrame({
        "stationary": rng.normal(100, 20, n), "pcm": rng.normal(0, 2, n),
        "qcd": rng.normal(0, 1, n), "openMM": rng.normal(0, 2, n),
    })
    df["experimental"] = (2.0 * df["stationary"] + 0.5 * df["pcm"] - 0.3 * df["qcd"]
                          + 1.0 + rng.normal(0, 1, n))
    df.loc[[3, 17, 42], "experimental"] = np.nan   # missing experimental values
    df.loc[[8, 25], "pcm"] = np.nan                # missing predictor values
    for formula in ["stationary", "stationary + pcm", "stationary + pcm + qcd",
                    "stationary + pcm + qcd + openMM"]:
        rmse, params = A.fit(df, formula)
        s_rmse, s_params = A._fit_statsmodels(df, formula)
        assert rmse == pytest.approx(s_rmse, abs=1e-8), formula
        for name in s_params.index:
            assert params[name] == pytest.approx(s_params[name], abs=1e-6), f"{formula}:{name}"


def test_build_query_df_nn(loader):
    q = A.build_query_df_nn(loader)
    expected = {"solute", "nucleus", "site", "solvent", "experimental", "stationary", "pcm",
                "openMM", "openMM_vib", "qcd", "stationary_plus_pcm", "stationary_plus_qcd",
                "stationary_plus_op_vib"}
    assert expected.issubset(q.columns)
    # H site covers atoms 1+2: stationary = mean(100, 101) = 100.5; pcm = mean(-1, -1) = -1;
    # conversion factor 1.0 -> stationary_plus_pcm = 99.5
    h = q[(q["nucleus"] == "H") & (q["solvent"] == "chloroform")].iloc[0]
    assert h["stationary"] == pytest.approx(100.5, abs=1e-3)
    assert h["stationary_plus_pcm"] == pytest.approx(99.5, abs=1e-3)
    assert h["qcd"] == pytest.approx(0.0, abs=1e-3)


def test_apply_bootstrap_nan_handling():
    # a missing coeff (openMM=NaN) contributes nothing; a present coeff with missing data -> NaN
    coeffs = pd.DataFrame([{"solvent": "chloroform", "formula": "pcm2", "seed": 0,
                            "Intercept": 5.0, "stationary": 2.0, "openMM": np.nan}])
    data = pd.DataFrame([
        {"solute": "m", "nucleus": "H", "site": "a", "solvent": "chloroform",
         "experimental": 7.0, "stationary": 1.0, "openMM": 9.0},                 # -> 5 + 2*1 = 7
        {"solute": "m", "nucleus": "H", "site": "b", "solvent": "chloroform",
         "experimental": np.nan, "stationary": np.nan, "openMM": 9.0},           # stationary data NaN -> NaN
    ])
    out = A.apply_bootstrap_params_to_full_dataset(data, coeffs, nucleus="H").set_index("site")
    assert out.loc["a", "predicted"] == pytest.approx(7.0)
    assert np.isnan(out.loc["b", "predicted"])


def test_apply_bootstrap_and_solute_rmse():
    # one formula/seed/solvent: predicted = 5 + 2*stationary; experimental matches -> RMSE 0
    coeffs = pd.DataFrame({"solvent": ["chloroform"], "formula": ["pcm2"], "seed": [0],
                           "Intercept": [5.0], "stationary": [2.0]})
    data = pd.DataFrame({
        "solute": ["m", "m"], "nucleus": ["H", "H"], "site": ["01 H", "02 H"],
        "solvent": ["chloroform", "chloroform"], "experimental": [7.0, 9.0], "stationary": [1.0, 2.0],
    })
    preds = A.apply_bootstrap_params_to_full_dataset(data, coeffs, nucleus="H")
    assert np.allclose(preds["predicted"], preds["experimental"])
    rmses = A.compute_solute_rmses(preds)
    assert rmses["Bootstrap_RMSE"].iloc[0] == pytest.approx(0.0, abs=1e-9)


# --- SI Figure S15's "Fitting RMSE Comparisons" -----------------------------------------------

def _rmse_comparison_fixture():
    """Two test-set solutes ("m", "n"), one solvent, formula "stationary + openMM". Each solute has
    only 2 points against 3 free parameters, so a per-solute-only fit is always exact (RMSE 0) --
    that part needs no special data. The 4 points are deliberately NOT co-planar in (stationary,
    openMM, experimental) space (verified: an earlier version of this fixture picked co-planar
    points by accident, which made the pooled "Scaled to Test Set" fit ALSO exact and silently
    hid the "formula" column bug this fixture now catches -- see the FORMULA_REMAP note below), so
    the pooled fit has nonzero RMSE and "Scaled to Test Set" != "Scaled to Solute". The bootstrap
    coefficients are also chosen far from both fits, so all three columns differ."""
    query_df_nn = pd.DataFrame([
        {"solute": "m", "nucleus": "H", "site": "01", "solvent": "chloroform", "experimental": 10.0, "stationary": 5.0, "openMM": 1.0},
        {"solute": "m", "nucleus": "H", "site": "02", "solvent": "chloroform", "experimental": 12.0, "stationary": 6.0, "openMM": 1.0},
        {"solute": "n", "nucleus": "H", "site": "01", "solvent": "chloroform", "experimental": 20.0, "stationary": 5.0, "openMM": 2.0},
        {"solute": "n", "nucleus": "H", "site": "02", "solvent": "chloroform", "experimental": 23.0, "stationary": 7.0, "openMM": 2.0},
    ])
    formula = "stationary + openMM"
    per_solute = {"H": A.fit_formulas_per_solvent_and_solute(query_df_nn, [formula], ["chloroform"],
                                                             ["m", "n"], A.FORMULA_REMAP)}
    all_solute = {"H": A.fit_formulas_per_solvent(query_df_nn, [formula], ["chloroform"], A.FORMULA_REMAP)}
    # The real pipeline (applications.build_bootstrap_seed_coeffs) always stores the REMAPPED
    # formula name in its "formula" column, same as per_solute_fits/per_solvent_fits above --
    # fitting_rmse_comparison_table looks up bootstrap_rmses by the remapped name, so a coeffs
    # frame built with the raw formula string would never match and "Extrapolated from delta22"
    # would silently come out all-NaN instead of raising. Match that convention here.
    remapped = A.FORMULA_REMAP[formula]
    coeffs = pd.DataFrame({"solvent": ["chloroform"], "formula": [remapped], "seed": [0],
                           "Intercept": [0.0], "stationary": [1.0], "openMM": [1.0]})
    preds = A.apply_bootstrap_params_to_full_dataset(query_df_nn, coeffs, nucleus="H")
    bootstrap_rmses = A.compute_solute_rmses(preds)
    return query_df_nn, per_solute, all_solute, bootstrap_rmses, formula


def test_scaled_to_test_set_per_solute_rmse_applies_one_pooled_fit_per_solute():
    query_df_nn, _, all_solute, _, formula = _rmse_comparison_fixture()
    out = A.scaled_to_test_set_per_solute_rmse(query_df_nn, all_solute, "H", "chloroform", formula)
    assert set(out["solute"]) == {"m", "n"}
    # both solutes score under the SAME pooled coefficients, so their RMSEs need not be equal
    assert out["Bootstrap_RMSE"].notna().all()


def test_fitting_rmse_comparison_table_has_all_three_columns_and_solutes():
    query_df_nn, per_solute, all_solute, bootstrap_rmses, formula = _rmse_comparison_fixture()
    table = A.fitting_rmse_comparison_table(query_df_nn, per_solute, all_solute, bootstrap_rmses,
                                            "H", "chloroform", formula)
    assert set(table.columns) == {"Scaled to Solute", "Scaled to Test Set", "Extrapolated from delta22"}
    assert set(table.index) == {"m", "n"}
    assert table.notna().all().all()   # a formula-remapping mismatch would leave a column all-NaN
    # "Scaled to Solute" fits each solute alone with 2 free params on 2 points -> exact, RMSE 0
    assert table["Scaled to Solute"].max() < 1e-8
    # the fixture's points are deliberately non-coplanar (see _rmse_comparison_fixture), so the
    # pooled fit is NOT exact and genuinely differs from the per-solute fit
    assert table["Scaled to Test Set"].max() > 1e-3
    # the bootstrap coefficients are far from either fit, so this column differs from both
    assert (table["Extrapolated from delta22"] - table["Scaled to Solute"]).abs().min() > 0.1


@pytest.mark.skipif(
    not os.path.exists(paths.dataset_file("applications", file=__file__)),
    reason="real applications.hdf5 not present")
def test_fitting_rmse_comparison_table_reproduces_published_chloroform_values():
    """Every published "Fitting RMSE Comparisons (chloroform, H)" value (SI Figure S15), read
    directly off the SI, reproduced from the released data. Cross-checks the delta-22-bootstrap
    ("Extrapolated") column at a slightly looser tolerance since it averages over random seeds."""
    real_data_dir = os.path.join(HERE, "..", "..", "data", "applications")
    loader = Applications(paths.dataset_file("applications", file=__file__),
                          os.path.join(real_data_dir, "applications_experimental.xlsx"))
    query_df_nn = A.build_query_df_nn(loader)
    seed = A.build_bootstrap_seed_coeffs(loader)
    per_solute = A.per_solute_fits(query_df_nn)
    all_solute = A.per_solvent_fits(query_df_nn)
    preds_h = A.apply_bootstrap_params_to_full_dataset(query_df_nn, seed["H"], nucleus="H")
    bootstrap_rmses_h = A.compute_solute_rmses(preds_h)
    table = A.fitting_rmse_comparison_table(query_df_nn, per_solute, all_solute, bootstrap_rmses_h,
                                            "H", "chloroform", "stationary_plus_qcd + openMM")
    # (solute, column) -> published value, read off the SI Figure S15 bar chart
    published = {
        ("isomer_1E", "Scaled to Solute"): 0.138, ("isomer_1E", "Scaled to Test Set"): 0.138,
        ("vomicine", "Scaled to Solute"): 0.173, ("vomicine", "Scaled to Test Set"): 0.183,
        ("prednisone", "Scaled to Solute"): 0.130, ("prednisone", "Scaled to Test Set"): 0.153,
        ("flavone", "Scaled to Solute"): 0.021, ("flavone", "Scaled to Test Set"): 0.062,
    }
    for (solute, col), expected in published.items():
        assert table.loc[solute, col] == pytest.approx(expected, abs=3e-3), f"{solute}/{col}"
    published_extrapolated = {"isomer_1E": 0.146, "vomicine": 0.180, "prednisone": 0.166, "flavone": 0.060}
    for solute, expected in published_extrapolated.items():
        assert table.loc[solute, "Extrapolated from delta22"] == pytest.approx(expected, abs=0.02), solute


@pytest.mark.skipif(
    not os.path.exists(paths.dataset_file("applications", file=__file__)),
    reason="real applications.hdf5 not present")
def test_distribution_shift_by_solvent_table_flags_water_on_real_data():
    """The per-solvent distribution-shift check on real data: for protons, TIP4P (water) must show
    the largest 'Extrapolated from delta22' minus 'Scaled to Solute' gap -- the exact claim the
    function's docstring and SI Figure S15's narrative make (the peptide dominates the water column).
    Also spot-checks two published per-solvent RMSE values."""
    real_data_dir = os.path.join(HERE, "..", "..", "data", "applications")
    loader = Applications(paths.dataset_file("applications", file=__file__),
                          os.path.join(real_data_dir, "applications_experimental.xlsx"))
    query_df_nn = A.build_query_df_nn(loader)
    seed = A.build_bootstrap_seed_coeffs(loader)
    per_solute = A.per_solute_fits(query_df_nn)
    all_solute = A.per_solvent_fits(query_df_nn)
    bootstrap_h = A.compute_solute_rmses(
        A.apply_bootstrap_params_to_full_dataset(query_df_nn, seed["H"], nucleus="H"))
    table = A.distribution_shift_by_solvent_table(per_solute, all_solute, bootstrap_h, "H",
                                                  "stationary_plus_qcd + openMM")
    gap = table["Extrapolated from delta22"] - table["Scaled to Solute"]
    assert gap.idxmax() == "TIP4P", f"expected water to have the largest extrapolation gap, got {gap.idxmax()}"
    assert table.loc["TIP4P", "Extrapolated from delta22"] == pytest.approx(0.336, abs=0.01)
    assert table.loc["chloroform", "Scaled to Solute"] == pytest.approx(0.107, abs=0.01)


# --- SI Figure S15's "Feature Space Coverage by Solvent" / "Residuals for Delta22 Fitting Coefficients" ---

def _feature_space_fixture():
    """A test-set solute ("m") and a delta-22 solute, both nucleus H, in chloroform and benzene
    (TIP4P/methanol deliberately absent from "m" to check the solvent filter drops rows cleanly).
    stationary_plus_qcd/openMM are chosen so delta-22's chloroform plane is exact-fittable
    (3 points, 3 free parameters -> zero residual)."""
    query_df_nn = pd.DataFrame([
        {"solute": "m", "nucleus": "H", "solvent": "chloroform", "stationary_plus_qcd": 10.0, "openMM": 1.0, "experimental": 12.0},
        {"solute": "m", "nucleus": "H", "solvent": "benzene", "stationary_plus_qcd": 11.0, "openMM": 2.0, "experimental": 15.0},
        {"solute": "m", "nucleus": "C", "solvent": "chloroform", "stationary_plus_op_vib": 100.0, "openMM": 3.0, "experimental": 120.0},
    ])
    delta22_query_df_nn = pd.DataFrame([
        {"solute": "delta22", "nucleus": "H", "solvent": "chloroform", "stationary_plus_qcd": 12.0, "openMM": 0.0, "experimental": 12.0},
        {"solute": "delta22", "nucleus": "H", "solvent": "chloroform", "stationary_plus_qcd": 14.0, "openMM": 1.0, "experimental": 15.0},
        {"solute": "delta22", "nucleus": "H", "solvent": "chloroform", "stationary_plus_qcd": 16.0, "openMM": 2.0, "experimental": 18.0},
        {"solute": "delta22", "nucleus": "H", "solvent": "benzene", "stationary_plus_qcd": 9.0, "openMM": 1.5, "experimental": 13.0},
    ])
    return query_df_nn, delta22_query_df_nn


def test_feature_space_coverage_table_combines_and_centers_both_datasets():
    query_df_nn, delta22_query_df_nn = _feature_space_fixture()
    out = A.feature_space_coverage_table(query_df_nn, delta22_query_df_nn, "H", solvents=("chloroform", "benzene"))
    assert set(out["dataset"]) == {"Test Set", "Delta22"}
    assert set(out["solvent"]) == {"chloroform", "benzene"}
    # global centering: the pooled x column (both datasets, both solvents) averages to 0
    assert out["x"].mean() == pytest.approx(0.0, abs=1e-9)
    assert out["y"].mean() == pytest.approx(0.0, abs=1e-9)


def test_feature_space_coverage_table_drops_solvents_test_set_lacks():
    query_df_nn, delta22_query_df_nn = _feature_space_fixture()
    out = A.feature_space_coverage_table(query_df_nn, delta22_query_df_nn, "H",
                                         solvents=("chloroform", "benzene", "methanol", "TIP4P"))
    # "m" (the only test-set H solute) has no methanol/TIP4P rows, so those solvents drop out
    assert set(out["solvent"]) == {"chloroform", "benzene"}


def test_delta22_plane_residuals_table_delta22_residuals_are_near_zero():
    # delta-22's own 3 chloroform points exactly determine a 3-parameter plane, so delta-22's own
    # residuals against that plane must be ~0 (an OLS fit's residuals average to 0 with an
    # intercept; with exactly as many points as parameters, every residual is individually ~0)
    query_df_nn, delta22_query_df_nn = _feature_space_fixture()
    out = A.delta22_plane_residuals_table(query_df_nn, delta22_query_df_nn, "H", solvents=("chloroform",))
    d22 = out[out["dataset"] == "Delta22"]
    assert len(d22) == 3
    assert np.allclose(d22["residual"], 0.0, atol=1e-6)
    # the test-set row scores against that SAME plane and is not forced to be exact
    test_row = out[out["dataset"] == "Test Set"]
    assert len(test_row) == 1


def test_delta22_plane_residuals_table_skips_solvents_with_too_few_delta22_points():
    # benzene has only 1 delta-22 point (< 3 needed to fit a plane) -> dropped entirely, no crash
    query_df_nn, delta22_query_df_nn = _feature_space_fixture()
    out = A.delta22_plane_residuals_table(query_df_nn, delta22_query_df_nn, "H", solvents=("chloroform", "benzene"))
    assert set(out["solvent"]) == {"chloroform"}


@pytest.mark.skipif(
    not os.path.exists(paths.dataset_file("applications", file=__file__)),
    reason="real applications.hdf5 not present")
def test_feature_space_and_residuals_tables_sane_on_real_data():
    """Not an exact-number reproduction (these are scatter plots, not summary statistics) --
    structural sanity checks: delta-22's own residuals against its own fitted plane average to
    (near) zero every solvent, and the known H/TIP4P data-scarcity caveat from the SI text ("the
    available experimental data for water in the test set is limited to one solute") shows up as a
    small test-set count for that one solvent."""
    real_data_dir = os.path.join(HERE, "..", "..", "data", "applications")
    delta22_dir = os.path.join(HERE, "..", "..", "data", "delta22")
    sys.path.insert(0, delta22_dir)
    import delta22 as D
    loader = Applications(paths.dataset_file("applications", file=__file__),
                          os.path.join(real_data_dir, "applications_experimental.xlsx"))
    query_df_nn = A.build_query_df_nn(loader)
    delta22_query_df_nn = D.add_composite_columns(D.load_query_df_nn(
        paths.dataset_file("delta22", file=__file__), os.path.join(delta22_dir, "delta22_experimental.xlsx"),
        verbose=False))
    for nucleus in ["H", "C"]:
        residuals = A.delta22_plane_residuals_table(query_df_nn, delta22_query_df_nn, nucleus)
        d22_means = residuals[residuals["dataset"] == "Delta22"].groupby("solvent")["residual"].mean()
        assert (d22_means.abs() < 1e-6).all()
        test_counts = residuals[residuals["dataset"] == "Test Set"].groupby("solvent").size()
        assert test_counts["chloroform"] > 50   # plenty of chloroform test-set data
        # SI-documented data scarcity: "the available experimental data for water in the test set
        # is limited to one solute" -- the residuals table (which requires an experimental value,
        # unlike the feature-space-coverage table) sees far fewer TIP4P test-set rows than chloroform
        assert 0 < test_counts.get("TIP4P", 0) < test_counts["chloroform"] / 5
