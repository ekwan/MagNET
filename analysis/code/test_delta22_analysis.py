"""Tests for analysis/code/delta22.py (the no-plotting analysis module).

The analysis functions operate on flat query DataFrames, so these tests build small synthetic
tables with a known linear relationship and check the fitting harness, the seeded splits, the
Pareto frontier, and the correlation matrix. One test proves the fast numpy `fit` gives the same
coefficients and RMSE as the original statsmodels fit.
"""
import os
import random
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "data", "delta22"))  # delta22_reader
sys.path.insert(0, HERE)
import paths

import delta22 as D  # noqa: E402


def make_query_df(n_solutes=12, solvents=("chloroform", "benzene"), noise=0.0, seed=0):
    """A flat query table whose experimental value is exactly 2*stationary + 0.5*pcm + 1 (plus
    optional noise), so a model with both terms fits near-perfectly and 'stationary' alone leaves
    the 0.5*pcm residual."""
    rng = np.random.default_rng(seed)
    rows = []
    for si in range(n_solutes):
        solute = f"m{si:02d}"
        for solvent in solvents:
            for k in range(3):
                stationary = rng.normal(100, 20)
                pcm = rng.normal(0, 2)
                experimental = 2.0 * stationary + 0.5 * pcm + 1.0 + rng.normal(0, noise)
                rows.append({
                    "solute": solute, "solvent": solvent, "nucleus": "H",
                    "site": f"{solute}_{solvent}_{k}", "experimental": experimental,
                    "stationary": stationary, "pcm": pcm,
                    "desmond": rng.normal(0, 2), "openMM": rng.normal(0, 2),
                    "qcd": rng.normal(0, 1), "desmond_vib": rng.normal(0, 1),
                    "openMM_vib": rng.normal(0, 1),
                })
    return pd.DataFrame(rows)


def test_fit_recovers_known_line():
    df = make_query_df(noise=0.0)
    train_rmse, test_rmse, params = D.fit(df, df, "stationary + pcm")
    assert params["Intercept"] == pytest.approx(1.0, abs=1e-6)
    assert params["stationary"] == pytest.approx(2.0, abs=1e-6)
    assert params["pcm"] == pytest.approx(0.5, abs=1e-6)
    assert train_rmse == pytest.approx(0.0, abs=1e-6)
    assert test_rmse == pytest.approx(0.0, abs=1e-6)


def test_fit_matches_statsmodels():
    df = make_query_df(noise=1.0, seed=3)
    train = df[df["solute"] <= "m07"]
    test = df[df["solute"] > "m07"]
    for formula in ["stationary", "stationary + pcm", "stationary + pcm + qcd"]:
        tr, te, params = D.fit(train, test, formula)
        s_tr, s_te, s_params = D._fit_statsmodels(train, test, formula)
        assert tr == pytest.approx(s_tr, abs=1e-8)
        assert te == pytest.approx(s_te, abs=1e-8)
        for name in s_params.index:
            assert params[name] == pytest.approx(s_params[name], abs=1e-6)


def test_fit_drops_missing_like_statsmodels():
    # fitting must drop rows with a missing response or predictor, exactly as statsmodels does,
    # and recover the same coefficients. (Scoring is checked on a clean test set; statsmodels would
    # otherwise NaN-poison an RMSE computed over rows that have missing predictors.)
    clean = make_query_df(noise=0.5, seed=5).reset_index(drop=True)
    train = clean.copy()
    train.loc[0, "experimental"] = np.nan      # missing response
    train.loc[1, "pcm"] = np.nan               # missing predictor
    _train_err, te, params = D.fit(train, clean, "stationary + pcm")
    s_tr, s_te, s_params = D._fit_statsmodels(train, clean, "stationary + pcm")
    for name in s_params.index:
        assert params[name] == pytest.approx(s_params[name], abs=1e-6)
    assert te == pytest.approx(s_te, abs=1e-8)


def test_generate_solute_splits_seeded_and_disjoint():
    df = make_query_df()
    solutes = [f"m{i:02d}" for i in range(12)]
    splits = D.generate_solute_splits(5, df, "chloroform", solutes, n_test=4)
    assert len(splits) == 5
    # reproduce the exact seed logic for split 0
    random.seed(0 + 100)
    expected = list(solutes)
    random.shuffle(expected)
    train_df, test_df = splits[0]
    assert set(test_df["solute"].unique()) == set(expected[:4])
    assert set(train_df["solute"].unique()) == set(expected[4:])
    # a solute is never in both halves, and only the requested solvent is present
    assert set(test_df["solute"]).isdisjoint(set(train_df["solute"]))
    assert set(test_df["solvent"].unique()) == {"chloroform"}
    # deterministic: same call gives the same partition
    again = D.generate_solute_splits(5, df, "chloroform", solutes, n_test=4)
    assert set(again[0][1]["solute"].unique()) == set(test_df["solute"].unique())


def test_run_fits_more_terms_lower_error():
    df = make_query_df(noise=0.3, seed=1)
    solutes = sorted(df["solute"].unique())
    results = D.run_fits(df, ["chloroform", "benzene"], D.STATIONARY_VS_PCM,
                         n_splits=5, solutes=solutes, n_test=4)
    assert set(results.columns) >= {"solvent", "formula", "seed", "train_RMSE", "test_RMSE"}
    med = D.median_test_rmse(results, ["formula"]).set_index("formula")["test_RMSE"]
    assert med["stationary + pcm"] < med["stationary"]   # the pcm term genuinely helps


def test_run_fits_group_cols():
    df = make_query_df(noise=0.3, seed=2)
    df = pd.concat([df.assign(sap_nmr_method="wp04"),
                    df.assign(sap_nmr_method="dsd_pbep86")], ignore_index=True)
    solutes = sorted(df["solute"].unique())
    results = D.run_fits(df, ["chloroform"], ["stationary + pcm"], n_splits=3,
                         solutes=solutes, group_cols=["sap_nmr_method"], n_test=4)
    assert "sap_nmr_method" in results.columns
    assert set(results["sap_nmr_method"].unique()) == {"wp04", "dsd_pbep86"}


def test_fit_coefficients_per_seed():
    df = make_query_df(noise=0.4, seed=7)
    solutes = sorted(df["solute"].unique())
    coeffs = D.fit_coefficients(df, "chloroform", "stationary + pcm", solutes,
                                n_splits=6, n_test=4)
    assert len(coeffs) == 6
    assert {"solvent", "seed", "Intercept", "stationary", "pcm"}.issubset(coeffs.columns)
    # the recovered slope should sit near the true value of 2
    assert coeffs["stationary"].mean() == pytest.approx(2.0, abs=0.1)


def test_full_fit_coefficients_shape_and_values():
    df = make_query_df(noise=0.0, solvents=("chloroform", "benzene"))
    coeffs = D.full_fit_coefficients(df, ["chloroform", "benzene"], "stationary + pcm")
    assert list(coeffs.columns) == ["chloroform", "benzene"]
    assert set(coeffs.index) == {"Intercept", "stationary", "pcm"}
    # the data is exactly 2*stationary + 0.5*pcm + 1, recovered per solvent
    for solvent in ["chloroform", "benzene"]:
        assert coeffs.loc["Intercept", solvent] == pytest.approx(1.0, abs=1e-6)
        assert coeffs.loc["stationary", solvent] == pytest.approx(2.0, abs=1e-6)
        assert coeffs.loc["pcm", solvent] == pytest.approx(0.5, abs=1e-6)


# ---- opt-in: checked against the STORED composite-model coefficients ----
# These coefficients live in applications.hdf5's composite_model/ols_coefficients group. Re-fitting
# the released delta-22 data with the fast numpy harness (nitromethane dropped) must reproduce them
# to within the data's integer-encoding precision (~5e-4 after error propagation through the small
# corrections).
REAL_H5 = paths.dataset_file("delta22", file=__file__)
REAL_XLSX = os.path.join(HERE, "..", "..", "data", "delta22", "delta22_experimental.xlsx")
APPLICATIONS_H5 = paths.dataset_file("applications", file=__file__)


@pytest.mark.skipif(
    not (os.path.exists(REAL_H5) and os.path.exists(REAL_XLSX) and os.path.exists(APPLICATIONS_H5)),
    reason="real delta22.hdf5 / experimental xlsx / applications.hdf5 not present")
def test_full_fit_reproduces_stored_composite_coefficients():
    import io
    import h5py

    query = D.load_query_df_nn(REAL_H5, REAL_XLSX, exclude_solutes=["nitromethane"], verbose=False)
    worst = 0.0
    with h5py.File(APPLICATIONS_H5, "r") as f:
        ols = f["composite_model"]["ols_coefficients"]
        for nucleus in ["H", "C"]:
            sub = query[query["nucleus"] == nucleus]
            for formula in ["stationary", "stationary + openMM", "stationary + openMM + qcd"]:
                tag = "_AND_".join(term.strip() for term in formula.split("+"))
                raw = np.array(ols[tag][nucleus]).item()
                raw = raw.decode() if hasattr(raw, "decode") else raw
                stored = pd.read_csv(io.StringIO(raw)).set_index("parameter")
                mine = D.full_fit_coefficients(sub, list(stored.columns), formula)
                mine = mine.rename(index={"Intercept": "intercept"})
                for solvent in stored.columns:
                    for param in stored.index:
                        worst = max(worst, abs(mine.loc[param, solvent] - stored.loc[param, solvent]))
    assert worst < 5e-4, f"worst coefficient diff {worst:.2e} exceeds the int32 precision floor"


def make_dft_query_df(methods=("wp04", "dsd_pbep86"), solvents=("chloroform", "methanol", "benzene"),
                      n_solutes=12, seed=0):
    """A synthetic DFT query table with method/basis/geometry columns. Experimental is
    31 - (stationary + pcm) + tiny noise, i.e. a single shared slope on the gas shielding plus its
    implicit-solvent correction (the real structure), so the single-column 'stationary_plus_pcm'
    model fits well and 'stationary' alone leaves the pcm residual. desmond/openMM/qcd are
    independent noise that does not help."""
    rng = np.random.default_rng(seed)
    rows = []
    for method in methods:
        for si in range(n_solutes):
            solute = f"m{si:02d}"
            for solvent in solvents:
                for k in range(2):
                    stationary = rng.normal(100, 20)
                    pcm = rng.normal(0, 2)
                    rows.append({
                        "solute": solute, "solvent": solvent, "nucleus": "H",
                        "site": f"{solute}_{k}", "sap_nmr_method": method,
                        "sap_basis": "pcSseg2", "sap_geometry_type": "aimnet2",
                        "experimental": 31.0 - (stationary + pcm) + rng.normal(0, 0.05),
                        "stationary": stationary, "pcm": pcm,
                        "desmond": rng.normal(0, 1), "openMM": rng.normal(0, 1),
                        "qcd": rng.normal(0, 1), "desmond_vib": rng.normal(0, 1),
                        "openMM_vib": rng.normal(0, 1),
                    })
    return pd.DataFrame(rows)


def test_add_composite_columns():
    df = make_query_df()
    out = D.add_composite_columns(df)
    assert np.allclose(out["stationary_plus_pcm"], out["stationary"] + out["pcm"])
    assert np.allclose(out["stationary_plus_pcm_plus_qcd"],
                       out["stationary"] + out["pcm"] + out["qcd"])
    # an existing NN-style stationary_plus_pcm (with scaling factors) is preserved
    nn = df.copy()
    nn["stationary_plus_pcm"] = 999.0
    kept = D.add_composite_columns(nn)
    assert np.allclose(kept["stationary_plus_pcm"], 999.0)


def test_fig3b_shift_differences_exact():
    # one method, three solvents; experimental/pcm chosen so the differences are known
    rows = []
    vals = {"chloroform": (1.0, 0.1), "methanol": (1.5, 0.4), "benzene": (0.7, -0.2)}
    for solvent, (exp, pcm) in vals.items():
        rows.append({"solute": "m0", "site": "s0", "nucleus": "H", "solvent": solvent,
                     "sap_nmr_method": "wp04", "sap_basis": "pcSseg2",
                     "sap_geometry_type": "aimnet2", "experimental": exp, "pcm": pcm})
    df = pd.DataFrame(rows)
    out = D.fig3b_shift_differences(df, "wp04", "pcSseg2", "aimnet2", nucleus="H",
                                    x_solvent="methanol", y_solvent="benzene", reference="chloroform")
    assert len(out) == 1
    assert out["exp_diff_x"].iloc[0] == pytest.approx(1.5 - 1.0)
    assert out["exp_diff_y"].iloc[0] == pytest.approx(0.7 - 1.0)
    assert out["pcm_diff_x"].iloc[0] == pytest.approx(0.4 - 0.1)
    assert out["pcm_diff_y"].iloc[0] == pytest.approx(-0.2 - 0.1)


def test_fig3a_and_fig3d_pcm_helps():
    df = D.add_composite_columns(make_dft_query_df(seed=4))
    solutes = sorted(df["solute"].unique())
    solvents = ["chloroform", "methanol", "benzene"]
    a = D.fig3a_pcm_benefit(df, solvents, n_splits=4, solutes=solutes, n_test=4)
    assert {"sap_nmr_method", "solvent", "formula", "test_RMSE"}.issubset(a.columns)
    med = a.groupby("formula")["test_RMSE"].median()
    assert med["stationary_plus_pcm"] < med["stationary"]   # PCM lowers error
    d = D.fig3d_formula_regressions(df, "wp04", "pcSseg2", "aimnet2",
                                    ["stationary + pcm", "stationary + desmond"], solvents,
                                    n_splits=4, solutes=solutes, n_test=4)
    assert set(d["formula"].unique()) == {"stationary + pcm", "stationary + desmond"}


def test_fig3a_pcm_benefit_by_solvent_keeps_solvents_separate():
    # chloroform: pcm carries a real (though not perfectly scaled) part of the signal, so adding it
    # to the fit lowers test error. benzene: pcm is unrelated noise on top of an already-explained
    # response, so adding it as a predictor only overfits and raises test error. A solvent-pooled
    # average would blend these two directions away; fig3a_pcm_benefit_by_solvent must keep one row
    # per solvent so both signs are visible.
    rng = np.random.default_rng(11)
    rows = []
    n_solutes = 16
    for si in range(n_solutes):
        solute = f"m{si:02d}"
        for solvent in ["chloroform", "benzene"]:
            for k in range(3):
                stationary = rng.normal(100, 20)
                if solvent == "chloroform":
                    pcm = rng.normal(0, 2)
                    experimental = 31.0 - stationary - 0.8 * pcm + rng.normal(0, 0.05)
                else:
                    pcm = rng.normal(0, 8)   # large noise unrelated to experimental
                    experimental = 31.0 - stationary + rng.normal(0, 0.05)
                rows.append({
                    "solute": solute, "solvent": solvent, "nucleus": "H",
                    "site": f"{solute}_{solvent}_{k}", "sap_nmr_method": "wp04",
                    "sap_basis": "pcSseg2", "sap_geometry_type": "aimnet2",
                    "experimental": experimental, "stationary": stationary, "pcm": pcm,
                    "desmond": 0.0, "openMM": 0.0, "qcd": 0.0, "desmond_vib": 0.0, "openMM_vib": 0.0,
                })
    df = D.add_composite_columns(pd.DataFrame(rows))
    solutes = sorted(df["solute"].unique())
    out = D.fig3a_pcm_benefit_by_solvent(df, ["chloroform", "benzene"], n_splits=10,
                                         solutes=solutes, n_test=6)
    assert {"sap_nmr_method", "solvent", "seed", "percent_benefit"}.issubset(out.columns)
    chloro_benefit = out[out["solvent"] == "chloroform"]["percent_benefit"].median()
    benz_benefit = out[out["solvent"] == "benzene"]["percent_benefit"].median()
    assert chloro_benefit > 0    # PCM helps in chloroform
    assert benz_benefit < 0      # PCM hurts in benzene


def test_fig3a_does_not_pool_nuclei():
    # H sites (~30 ppm) and C sites (~150 ppm) with different relationships. If fig3a pooled both
    # nuclei into one fit the test RMSE would blow up; filtering to one nucleus keeps it small.
    rng = np.random.default_rng(0)
    rows = []
    for si in range(12):
        for solvent in ["chloroform", "methanol", "benzene"]:
            for nucleus, base, slope, offset in [("H", 30, 1.0, 31.0), ("C", 150, 1.05, 200.0)]:
                stationary = rng.normal(base, base * 0.1)
                pcm = rng.normal(0, 1)
                rows.append({
                    "solute": f"m{si:02d}", "site": f"m{si:02d}_{nucleus}", "nucleus": nucleus,
                    "solvent": solvent, "sap_nmr_method": "wp04", "sap_basis": "pcSseg2",
                    "sap_geometry_type": "aimnet2", "stationary": stationary, "pcm": pcm,
                    "experimental": offset - slope * (stationary + pcm) + rng.normal(0, 0.05),
                    "desmond": 0.0, "openMM": 0.0, "qcd": 0.0, "desmond_vib": 0.0, "openMM_vib": 0.0,
                })
    df = D.add_composite_columns(pd.DataFrame(rows))
    solutes = sorted(df["solute"].unique())
    res = D.fig3a_pcm_benefit(df, ["chloroform", "methanol", "benzene"], n_splits=4,
                              solutes=solutes, nucleus="H", n_test=4)
    assert res["test_RMSE"].median() < 1.0   # would be tens of ppm if C were pooled in


def test_add_solvent_mean():
    df = pd.DataFrame({
        "solute": ["m0", "m0"], "site": ["s0", "s0"], "nucleus": ["H", "H"],
        "solvent": ["chloroform", "benzene"], "experimental": [2.0, 4.0], "pcm": [0.2, 0.6],
    })
    out = D.add_solvent_mean(df)
    mean_row = out[out["solvent"] == "solvent_mean"]
    assert len(mean_row) == 1
    assert mean_row["experimental"].iloc[0] == pytest.approx(3.0)   # mean(2, 4)
    assert mean_row["pcm"].iloc[0] == pytest.approx(0.4)            # mean(0.2, 0.6)


def test_solvent_pair_differences_exact():
    def row(solvent, exp, pcm, des, dvib):
        return {"solute": "m0", "site": "s0", "nucleus": "H", "solvent": solvent,
                "experimental": exp, "pcm": pcm, "desmond": des, "desmond_vib": dvib}
    df = pd.DataFrame([row("methanol", 2.0, 0.4, 0.5, 0.1),    # check solvent
                       row("chloroform", 1.0, 0.1, 0.2, 0.05)])  # reference
    out = D.solvent_pair_differences(df, "methanol", "chloroform", nucleus="H", explicit="desmond")
    assert len(out) == 1
    r = out.iloc[0]
    assert r["exp_diff"] == pytest.approx(1.0 - 2.0)          # reference - check
    assert r["implicit_diff"] == pytest.approx(0.4 - 0.1)     # check - reference
    assert r["explicit_diff"] == pytest.approx(0.5 - 0.2)
    assert r["explicit_vib_diff"] == pytest.approx((0.5 + 0.1) - (0.2 + 0.05))


def test_solvent_induced_shifts_stacks_solvents():
    rng = np.random.default_rng(0)
    rows = []
    for solvent in ["chloroform", "methanol", "benzene"]:
        for si in range(3):
            rows.append({"solute": f"m{si}", "site": f"m{si}_s", "nucleus": "H", "solvent": solvent,
                         "experimental": rng.normal(), "pcm": rng.normal(), "desmond": rng.normal(),
                         "desmond_vib": rng.normal()})
    df = pd.DataFrame(rows)
    out = D.solvent_induced_shifts(df, "chloroform", ["chloroform", "methanol", "benzene"],
                                   nucleus="H", explicit="desmond")
    # two non-reference solvents x 3 sites = 6 rows, reference column constant
    assert len(out) == 6
    assert set(out["solvent"].unique()) == {"methanol", "benzene"}
    assert (out["reference"] == "chloroform").all()


def test_fit_differences_to_experimental():
    # exp_diff = -2 * predicted + 0.3 exactly
    pred = np.linspace(-1, 1, 20)
    df = pd.DataFrame({"exp_diff": -2.0 * pred + 0.3, "explicit_diff": pred})
    fit = D.fit_differences_to_experimental(df, "explicit_diff")
    assert fit["slope"] == pytest.approx(-2.0, abs=1e-9)
    assert fit["intercept"] == pytest.approx(0.3, abs=1e-9)
    assert fit["rmse"] == pytest.approx(0.0, abs=1e-9)
    assert fit["n"] == 20


def test_explicit_correction_pairs_dedups_and_drops_nan():
    rows = []
    for method in ["wp04", "dsd_pbep86"]:    # same site tiled across methods
        rows.append({"solute": "m0", "site": "s0", "nucleus": "H", "solvent": "chloroform",
                     "sap_nmr_method": method, "desmond": 0.5, "openMM": 0.4})
    rows.append({"solute": "m1", "site": "s1", "nucleus": "H", "solvent": "chloroform",
                 "sap_nmr_method": "wp04", "desmond": np.nan, "openMM": 0.2})   # dropped (NaN)
    out = D.explicit_correction_pairs(pd.DataFrame(rows), "chloroform", "H")
    assert len(out) == 1
    assert out.iloc[0]["desmond"] == pytest.approx(0.5)
    assert out.iloc[0]["openMM"] == pytest.approx(0.4)


def test_compare_dft_nn():
    # method-tiled correction; dedup by keys, then error = NN - DFT
    dft = pd.DataFrame([
        {"solute": "m0", "site": "s0", "nucleus": "H", "sap_nmr_method": "wp04", "qcd": 0.10},
        {"solute": "m0", "site": "s0", "nucleus": "H", "sap_nmr_method": "mp2", "qcd": 0.10},
        {"solute": "m1", "site": "s1", "nucleus": "C", "sap_nmr_method": "wp04", "qcd": 0.50},
    ])
    nn = pd.DataFrame([
        {"solute": "m0", "site": "s0", "nucleus": "H", "sap_nmr_method": "MagNET", "qcd": 0.13},
        {"solute": "m1", "site": "s1", "nucleus": "C", "sap_nmr_method": "MagNET", "qcd": 0.40},
    ])
    out = D.compare_dft_nn(dft, nn, "qcd", keys=("solute", "site", "nucleus"))
    assert len(out) == 2
    h = out[out["nucleus"] == "H"].iloc[0]
    assert h["qcd_dft"] == pytest.approx(0.10)
    assert h["qcd_nn"] == pytest.approx(0.13)
    assert h["error"] == pytest.approx(0.13 - 0.10)


def test_qcd_correction_by_site_filters_nucleus_and_sorts_by_solute():
    # three H sites across two solutes (out of alphabetical order) plus one C site that must be
    # dropped; check the nucleus filter, the solute sort, and that the DFT/NN values pass through.
    dft = pd.DataFrame([
        {"solute": "zzz", "site": "s0", "nucleus": "H", "sap_nmr_method": "wp04", "qcd": -0.70},
        {"solute": "AcOH", "site": "s1", "nucleus": "H", "sap_nmr_method": "wp04", "qcd": -0.65},
        {"solute": "AcOH", "site": "s2", "nucleus": "H", "sap_nmr_method": "wp04", "qcd": -0.80},
        {"solute": "AcOH", "site": "s3", "nucleus": "C", "sap_nmr_method": "wp04", "qcd": -4.0},
    ])
    nn = pd.DataFrame([
        {"solute": "zzz", "site": "s0", "nucleus": "H", "sap_nmr_method": "MagNET", "qcd": -0.72},
        {"solute": "AcOH", "site": "s1", "nucleus": "H", "sap_nmr_method": "MagNET", "qcd": -0.64},
        {"solute": "AcOH", "site": "s2", "nucleus": "H", "sap_nmr_method": "MagNET", "qcd": -0.79},
        {"solute": "AcOH", "site": "s3", "nucleus": "C", "sap_nmr_method": "MagNET", "qcd": -4.1},
    ])
    out = D.qcd_correction_by_site(dft, nn, nucleus="H")
    assert len(out) == 3                              # the C row is dropped
    assert (out["nucleus"] == "H").all()
    assert list(out["solute"]) == ["AcOH", "AcOH", "zzz"]   # sorted, "zzz" sorts after "AcOH"
    row = out[out["site"] == "s1"].iloc[0]
    assert row["qcd_dft"] == pytest.approx(-0.65)
    assert row["qcd_nn"] == pytest.approx(-0.64)
    assert row["error"] == pytest.approx(-0.64 - (-0.65))


def test_compare_dft_nn_by_engine_stacks_both_engines():
    # one shared site, a different desmond/openMM value each, so the two engines land in separate
    # rows with a common dft_value/nn_value pair of columns instead of engine-specific column names
    dft = pd.DataFrame([
        {"solute": "m0", "site": "s0", "nucleus": "H", "solvent": "chloroform",
         "sap_nmr_method": "wp04", "desmond": 0.10, "openMM": 0.20},
    ])
    nn = pd.DataFrame([
        {"solute": "m0", "site": "s0", "nucleus": "H", "solvent": "chloroform",
         "sap_nmr_method": "MagNET", "desmond": 0.13, "openMM": 0.17},
    ])
    out = D.compare_dft_nn_by_engine(dft, nn)
    assert len(out) == 2
    assert set(out["engine"]) == {"desmond", "openMM"}
    assert list(out.columns) == ["solute", "site", "nucleus", "solvent", "dft_value", "nn_value",
                                 "error", "engine"]
    desmond_row = out[out["engine"] == "desmond"].iloc[0]
    assert desmond_row["dft_value"] == pytest.approx(0.10)
    assert desmond_row["nn_value"] == pytest.approx(0.13)
    assert desmond_row["error"] == pytest.approx(0.13 - 0.10)
    openmm_row = out[out["engine"] == "openMM"].iloc[0]
    assert openmm_row["dft_value"] == pytest.approx(0.20)
    assert openmm_row["nn_value"] == pytest.approx(0.17)
    assert openmm_row["error"] == pytest.approx(0.17 - 0.20)


def test_explicit_correction_dft_nn_pairs_merges_sources():
    dft = pd.DataFrame([
        {"solute": "m0", "site": "s0", "nucleus": "H", "solvent": "chloroform",
         "sap_nmr_method": "wp04", "desmond": 0.5, "openMM": 0.4},
        {"solute": "m1", "site": "s1", "nucleus": "H", "solvent": "chloroform",
         "sap_nmr_method": "wp04", "desmond": -0.2, "openMM": -0.3},
    ])
    nn = pd.DataFrame([
        {"solute": "m0", "site": "s0", "nucleus": "H", "solvent": "chloroform",
         "sap_nmr_method": "MagNET", "desmond": 0.55, "openMM": 0.42},
        # site s1 has no NN row, so it should not appear in the merged (inner-join) result
    ])
    out = D.explicit_correction_dft_nn_pairs(dft, nn, "chloroform", "H")
    assert len(out) == 1
    row = out.iloc[0]
    assert row["solute"] == "m0"
    assert row["desmond_dft"] == pytest.approx(0.5)
    assert row["openMM_dft"] == pytest.approx(0.4)
    assert row["desmond_nn"] == pytest.approx(0.55)
    assert row["openMM_nn"] == pytest.approx(0.42)


def test_si_s13d_fitting_accuracy_selects_formula_by_nucleus_and_engine():
    # stationary + qcd + desmond exactly reproduces experimental for H; stationary + desmond_vib +
    # openMM exactly reproduces it for C. Any other combination of (nucleus, engine) leaves a
    # nonzero residual from the term the formula does NOT include, so a near-zero test RMSE proves
    # si_s13d_fitting_accuracy picked the QCD-based formula for H and the vib-based one for C, and
    # picked the desmond/openMM explicit term matching the "engine" column.
    rng = np.random.default_rng(5)
    solvents = ["chloroform", "methanol", "TIP4P", "benzene"]
    rows = []
    for si in range(14):
        solute = f"m{si:02d}"
        for solvent in solvents:
            for k in range(2):
                stationary = rng.normal(50, 5)
                qcd = rng.normal(0, 1)
                desmond = rng.normal(0, 1)
                desmond_vib = rng.normal(0, 1)
                openMM = rng.normal(0, 1)
                openMM_vib = rng.normal(0, 1)
                rows.append({
                    "solute": solute, "solvent": solvent, "site": f"{solute}_{k}",
                    "sap_nmr_method": "dsd_pbep86", "sap_basis": "pcSseg3",
                    "sap_geometry_type": "pbe0_tz",
                    "stationary": stationary, "qcd": qcd, "pcm": 0.0,
                    "desmond": desmond, "openMM": openMM,
                    "desmond_vib": desmond_vib, "openMM_vib": openMM_vib,
                    "nucleus": "H",
                    "experimental": stationary + qcd + desmond,
                })
                rows.append({
                    "solute": solute, "solvent": solvent, "site": f"{solute}_{k}",
                    "sap_nmr_method": "dsd_pbep86", "sap_basis": "pcSseg3",
                    "sap_geometry_type": "pbe0_tz",
                    "stationary": stationary, "qcd": qcd, "pcm": 0.0,
                    "desmond": desmond, "openMM": openMM,
                    "desmond_vib": desmond_vib, "openMM_vib": openMM_vib,
                    "nucleus": "C",
                    "experimental": stationary + openMM_vib + openMM,
                })
    dft = pd.DataFrame(rows)
    nn = dft.copy()   # same synthetic values for both sources; the exact-fit check is source-agnostic
    solutes = sorted(dft["solute"].unique())

    h_results = D.si_s13d_fitting_accuracy(dft, nn, solvents, n_splits=3, solutes=solutes,
                                           nucleus="H")
    h_desmond = h_results[h_results["engine"] == "desmond"]["test_RMSE"]
    assert (h_desmond < 1e-8).all()
    h_openmm = h_results[h_results["engine"] == "openMM"]["test_RMSE"]
    assert (h_openmm > 0.1).all()   # missing the desmond term used to build "experimental"

    c_results = D.si_s13d_fitting_accuracy(dft, nn, solvents, n_splits=3, solutes=solutes,
                                           nucleus="C")
    c_openmm = c_results[c_results["engine"] == "openMM"]["test_RMSE"]
    assert (c_openmm < 1e-8).all()
    c_desmond = c_results[c_results["engine"] == "desmond"]["test_RMSE"]
    assert (c_desmond > 0.1).all()   # missing the openMM term used to build "experimental"


def test_pcm_benefit_per_solvent_positive():
    df = make_dft_query_df(seed=3)   # experimental = 31 - (stationary + pcm), so PCM genuinely helps
    solutes = sorted(df["solute"].unique())
    benefit = D.pcm_benefit_per_solvent(df, "wp04", "pcSseg2", "aimnet2",
                                        ["chloroform", "methanol", "benzene"], n_splits=4,
                                        solutes=solutes, nucleus="H", n_test=4)
    assert set(benefit.index) == {"chloroform", "methanol", "benzene"}
    assert (benefit > 0).all()   # adding PCM lowers the error in every solvent


def test_frame_convergence_math():
    # 3 atoms, 5 frames; site = atoms 0 and 2; solvated - isolated = a known per-frame value
    n_frames = 5
    isolated = np.zeros((n_frames, 3))
    solvated = np.zeros((n_frames, 3))
    per_frame_truth = np.array([0.2, 0.4, np.nan, 0.6, 0.8])
    for f in range(n_frames):
        solvated[f, 0] = per_frame_truth[f]
        solvated[f, 2] = per_frame_truth[f]
    solvated[2, :] = np.nan        # frame 2 invalid
    perturbed = np.stack([isolated, solvated], axis=-1)   # (5, 3, 2)
    fc = D.frame_corrections(perturbed, [0, 2])
    assert fc[0] == pytest.approx(0.2)
    assert np.isnan(fc[2])
    ra = D.running_average(fc)
    # running average over the valid frames 0.2, 0.4, (skip), 0.6, 0.8
    assert ra[0] == pytest.approx(0.2)
    assert ra[1] == pytest.approx(0.3)
    assert ra[2] == pytest.approx(0.3)             # invalid frame does not move the average
    assert ra[4] == pytest.approx((0.2 + 0.4 + 0.6 + 0.8) / 4)


def test_autocorrelation():
    ac = D.autocorrelation(np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0]), max_lag=2)
    assert ac[0] == pytest.approx(1.0)             # lag 0 is always 1
    assert ac[1] < 0                               # alternating -> negative lag-1 autocorrelation


def test_frame_validity():
    # 4 frames, 2 atoms: frame 0 fully computed, frame 1 only the isolated value, frame 2 all NaN
    # (skipped entirely), frame 3 fully computed. A frame counts as valid if anything was computed.
    perturbed = np.full((4, 2, 2), np.nan)
    perturbed[0] = [[1.0, 2.0], [3.0, 4.0]]
    perturbed[1, 0, 0] = 5.0
    perturbed[3] = [[1.0, 2.0], [3.0, 4.0]]
    valid = D.frame_validity(perturbed)
    assert list(valid) == [True, True, False, True]


def test_frame_validity_grid_pads_to_the_longest_solvent(monkeypatch):
    # two "solvents" with different trajectory lengths; the grid must pad the shorter one with
    # False past its own frame count, and frame_counts must report each solvent's true length.
    fake_data = {
        "acetone": np.array([[[1.0, 2.0]], [[np.nan, np.nan]], [[1.0, 2.0]]]),   # 3 frames, 1 invalid
        "benzene": np.array([[[1.0, 2.0]], [[1.0, 2.0]]]),                        # 2 frames, all valid
    }

    def fake_load_perturbed_shieldings(hdf5_filepath, solute, solvent, explicit_solvation_model,
                                       shield_type):
        return fake_data[solvent]

    monkeypatch.setattr(D, "load_perturbed_shieldings", fake_load_perturbed_shieldings)
    grid, frame_counts = D.frame_validity_grid("unused.hdf5", "solute", ["acetone", "benzene"],
                                               "openMM")
    assert grid.shape == (2, 3)
    assert list(frame_counts) == [3, 2]
    assert list(grid[0]) == [True, False, True]
    assert list(grid[1]) == [True, True, False]        # padded with False past benzene's 2 frames


def test_pareto_frontier():
    points = pd.DataFrame({
        "label": ["a", "b", "c", "d", "e"],
        "total_time": [1.0, 2.0, 3.0, 4.0, 5.0],
        "test_RMSE": [5.0, 3.0, 4.0, 1.0, 2.0],
    })
    front = D.pareto_frontier(points, x="total_time", y="test_RMSE")
    assert list(front["label"]) == ["a", "b", "d"]   # c and e are dominated


def test_correlation_matrix():
    # two solvents whose per-site pcm corrections are perfectly correlated
    base = np.array([0.1, 0.5, -0.3, 0.7, -0.2, 0.4])
    rows = []
    for i, value in enumerate(base):
        rows.append({"solute": f"m{i}", "site": f"s{i}", "solvent": "chloroform", "pcm": value})
        rows.append({"solute": f"m{i}", "site": f"s{i}", "solvent": "benzene", "pcm": 2 * value + 1})
    df = pd.DataFrame(rows)
    corr = D.correlation_matrix(df, "pcm", ["solute", "site"], "solvent")
    assert corr.loc["chloroform", "benzene"] == pytest.approx(1.0, abs=1e-9)


# --- Figure 2A / S1 (Pareto) ----------------------------------------------------------------

def test_pcm_conversion_factors_recovers_known_slope():
    # build DFT rows where the output method's PCM in each solvent is a known multiple of the input
    # method's chloroform PCM, so the recomputed factor must equal that multiple.
    rng = np.random.default_rng(0)
    multiples = {"chloroform": 1.0, "benzene": 0.65, "methanol": 1.4}
    rows = []
    for si in range(8):
        ref = rng.normal(0, 2)   # the input method's chloroform PCM for this site
        for solvent, mult in multiples.items():
            common = {"solute": f"m{si}", "site": f"m{si}_s", "nucleus": "H",
                      "sap_geometry_type": "aimnet2", "sap_basis": "pcSseg2", "solvent": solvent}
            # input (reference) method: chloroform carries `ref`, other solvents irrelevant
            rows.append({**common, "sap_nmr_method": D.MAGNET_PCM_INPUT_METHOD,
                         "pcm": ref if solvent == "chloroform" else rng.normal()})
            # output method: this solvent's PCM is mult * the chloroform reference
            rows.append({**common, "sap_nmr_method": "wp04", "pcm": mult * ref})
    df = pd.DataFrame(rows)
    factors = D.pcm_conversion_factors(df, "H", output_method="wp04")
    for solvent, mult in multiples.items():
        assert factors[solvent] == pytest.approx(mult, abs=1e-9)


def test_add_nn_stationary_plus_pcm():
    df = pd.DataFrame({
        "solute": ["m0", "m0"], "site": ["a", "b"], "nucleus": ["H", "C"],
        "solvent": ["benzene", "benzene"], "stationary": [100.0, 50.0], "pcm": [2.0, 4.0],
    })
    factors = {"H": pd.Series({"benzene": 0.5}), "C": pd.Series({"benzene": 3.0})}
    out = D.add_nn_stationary_plus_pcm(df, factors)
    assert out["stationary_plus_pcm"].iloc[0] == pytest.approx(100.0 + 2.0 * 0.5)
    assert out["stationary_plus_pcm"].iloc[1] == pytest.approx(50.0 + 4.0 * 3.0)


def _pareto_synthetic():
    """A tiny DFT+NN pair where a 'fast' method is also more accurate, so it must sit on the
    frontier and a 'slow, worse' method must not."""
    rng = np.random.default_rng(1)
    solvents = ["chloroform", "benzene", "TIP4P"]
    dft_rows, nn_rows = [], []
    for si in range(12):
        solute = f"m{si:02d}"
        for solvent in solvents:
            stat = rng.normal(100, 20)
            pcm = rng.normal(0, 2)
            exp = 31.0 - (stat + pcm)
            for method, noise in [("good_fast", 0.02), ("bad_slow", 1.5)]:
                dft_rows.append({
                    "solute": solute, "site": f"{solute}_s", "nucleus": "H", "solvent": solvent,
                    "sap_nmr_method": method, "sap_basis": "pcSseg2", "sap_geometry_type": "aimnet2",
                    "experimental": exp + rng.normal(0, noise), "stationary": stat, "pcm": pcm})
            nn_rows.append({
                "solute": solute, "site": f"{solute}_s", "nucleus": "H", "solvent": solvent,
                "sap_nmr_method": "MagNET", "sap_basis": "N/A", "sap_geometry_type": "aimnet2",
                "experimental": exp + rng.normal(0, 0.05), "stationary": stat, "pcm": pcm})
    return pd.DataFrame(dft_rows), pd.DataFrame(nn_rows)


def test_assemble_pareto_fitting_df_and_rmse():
    dft, nn = _pareto_synthetic()
    factors = {"H": pd.Series({s: 1.0 for s in ["chloroform", "benzene", "TIP4P"]}),
               "C": pd.Series(dtype=float)}
    fitting = D.assemble_pareto_fitting_df(dft, nn, conversion_factors=factors, exclude_solutes=())
    # TIP4P renamed to water, all three methods present, single predictor column built
    assert "water" in set(fitting["solvent"]) and "TIP4P" not in set(fitting["solvent"])
    assert set(fitting["nmr_method"]) == {"good_fast", "bad_slow", "MagNET"}
    assert "stationary_plus_pcm" in fitting.columns
    rmse = D.pareto_solvent_averaged_rmse(fitting, n_splits=6, n_test=6)
    assert "solvent-averaged" in set(rmse["solvent"])
    sav = rmse[rmse["solvent"] == "solvent-averaged"].set_index("nmr_method")["fitting_RMSE"]
    assert sav["good_fast"] < sav["bad_slow"]     # the accurate method has the lower RMSE


def test_attach_pareto_timings_substitutes_magnet():
    rmse = pd.DataFrame({
        "geometry_type": ["aimnet2", "aimnet2"], "nmr_method": ["bad_slow", "MagNET"],
        "basis": ["pcSseg2", "N/A"], "nucleus": ["H", "H"], "solvent": ["solvent-averaged"] * 2,
        "fitting_RMSE": [1.0, 0.2],
    })
    dft_gas = pd.DataFrame({"geometry_time": [10.0], "nmr_time": [990.0], "total_time": [1000.0]},
                           index=pd.MultiIndex.from_tuples([("aimnet2", "bad_slow", "pcSseg2")],
                                 names=["geometry_type", "nmr_method", "basis"]))
    nn = pd.DataFrame({"geometry_time": [5.0], "nmr_time": [15.0], "total_time": [20.0]},
                      index=pd.MultiIndex.from_tuples([("aimnet2", "MagNET", "N/A")],
                            names=["geometry_type", "nmr_method", "basis"]))
    out = D.attach_pareto_timings(rmse, dft_gas, nn)
    slow = out[out["nmr_method"] == "bad_slow"].iloc[0]
    magnet = out[out["nmr_method"] == "MagNET"].iloc[0]
    assert slow["total_time"] == pytest.approx(1000.0)
    assert magnet["total_time"] == pytest.approx(20.0)     # substituted from the NN timings
    # MagNET dominates: faster and more accurate, so it is on the frontier and slow is not
    front = D.pareto_frontier(out, x="total_time", y="fitting_RMSE")
    assert list(front["nmr_method"]) == ["MagNET"]


def test_pareto_table_curated_keeps_only_the_documented_rows():
    points = pd.DataFrame([
        # kept: MagNET itself
        {"nucleus": "H", "solvent": "chloroform", "geometry_type": "aimnet2", "nmr_method": "MagNET", "basis": "N/A", "total_time": 1.0},
        # kept: the one representative aimnet2 row (H's reference method wp04, pcSseg2)
        {"nucleus": "H", "solvent": "chloroform", "geometry_type": "aimnet2", "nmr_method": "wp04", "basis": "pcSseg2", "total_time": 2.0},
        # dropped: aimnet2 + wp04 but the WRONG basis
        {"nucleus": "H", "solvent": "chloroform", "geometry_type": "aimnet2", "nmr_method": "wp04", "basis": "pcSseg1", "total_time": 3.0},
        # dropped: aimnet2 + pcSseg2 but the WRONG method (not H's reference)
        {"nucleus": "H", "solvent": "chloroform", "geometry_type": "aimnet2", "nmr_method": "hf", "basis": "pcSseg2", "total_time": 4.0},
        # kept: part of the full pbe0_tz grid, any method/basis
        {"nucleus": "H", "solvent": "chloroform", "geometry_type": "pbe0_tz", "nmr_method": "hf", "basis": "pcSseg1", "total_time": 5.0},
        # dropped: right nucleus/geometry, WRONG solvent
        {"nucleus": "H", "solvent": "benzene", "geometry_type": "pbe0_tz", "nmr_method": "hf", "basis": "pcSseg1", "total_time": 6.0},
        # dropped: right solvent/geometry, WRONG nucleus
        {"nucleus": "C", "solvent": "chloroform", "geometry_type": "pbe0_tz", "nmr_method": "hf", "basis": "pcSseg1", "total_time": 7.0},
    ])
    out = D.pareto_table_curated(points, "H", solvent="chloroform")
    assert len(out) == 3
    assert set(out["total_time"]) == {1.0, 2.0, 5.0}
    assert list(out["total_time"]) == sorted(out["total_time"])   # sorted by total_time ascending


def test_pareto_table_curated_uses_the_carbon_reference_method():
    # C's reference method is wb97xd (MAGNET_PCM_OUTPUT_METHODS), not H's wp04
    points = pd.DataFrame([
        {"nucleus": "C", "solvent": "chloroform", "geometry_type": "aimnet2", "nmr_method": "wp04", "basis": "pcSseg2", "total_time": 1.0},
        {"nucleus": "C", "solvent": "chloroform", "geometry_type": "aimnet2", "nmr_method": "wb97xd", "basis": "pcSseg2", "total_time": 2.0},
    ])
    out = D.pareto_table_curated(points, "C", solvent="chloroform")
    assert list(out["nmr_method"]) == ["wb97xd"]


# Published Figure 2A values to reproduce from the released data (from the original
# fig2a_pareto_results.csv). MagNET sits at the cheap end of a flat frontier.
FIG2A_MAGNET_SOLVENT_AVERAGED = {"H": 0.1647203288369508, "C": 1.7268703504301826}
FIG2A_CONVERSION_FACTORS_H_BENZENE = 0.6513139306748533   # from magnet_pcm_conversion_factors_H.csv


@pytest.mark.skipif(
    not (os.path.exists(REAL_H5) and os.path.exists(REAL_XLSX)),
    reason="real delta22.hdf5 / experimental xlsx not present")
def test_fig2a_reproduces_published_magnet_point():
    """Recompute the MagNET Pareto point and conversion factors from the released data and check
    them against the published Figure 2A numbers (the int32 encoding sets the precision floor)."""
    dft = D.load_query_df_dft(REAL_H5, REAL_XLSX, verbose=False)
    nn = D.load_query_df_nn(REAL_H5, REAL_XLSX, verbose=False)
    # conversion factors recomputed from the DFT PCM data match the stored CSV value
    cf_h = D.pcm_conversion_factors(dft, "H")
    assert cf_h["benzene"] == pytest.approx(FIG2A_CONVERSION_FACTORS_H_BENZENE, abs=1e-6)
    # build only the MagNET fitting rows and average over solvents (fast; avoids the full DFT sweep)
    factors = {nuc: D.pcm_conversion_factors(dft, nuc) for nuc in ("H", "C")}
    nn_fit = D.add_nn_stationary_plus_pcm(nn, factors).rename(columns=lambda c: c.replace("sap_", ""))
    nn_fit["solvent"] = nn_fit["solvent"].replace("TIP4P", "water")
    nn_fit["experimental"] = pd.to_numeric(nn_fit["experimental"], errors="coerce")
    nn_fit = nn_fit[nn_fit["solute"] != "nitromethane"]
    # the split solute order must match the full combined table (DFT solutes, in file order)
    solutes = D.assemble_pareto_fitting_df(dft, nn, conversion_factors=factors)["solute"].unique().tolist()
    rmse = D.pareto_solvent_averaged_rmse(nn_fit, solutes=solutes)
    sav = rmse[(rmse["nmr_method"] == "MagNET") & (rmse["solvent"] == "solvent-averaged")]
    for nucleus, expected in FIG2A_MAGNET_SOLVENT_AVERAGED.items():
        got = sav[sav["nucleus"] == nucleus]["fitting_RMSE"].iloc[0]
        assert got == pytest.approx(expected, abs=1e-4), f"{nucleus}: {got} vs {expected}"


# --- SI Figure S15's "Correlations Between Features" ---------------------------------------------

def test_feature_correlation_matrix_synthetic():
    # a and b perfectly correlated, c independent noise uncorrelated with either
    n = 200
    rng = np.random.default_rng(0)
    a = rng.normal(size=n)
    df = pd.DataFrame({
        "nucleus": ["H"] * n, "solvent": ["chloroform"] * n,
        "stationary": a, "pcm": a * 2 + 1,  # perfectly correlated with stationary
        "desmond": rng.normal(size=n), "desmond_vib": rng.normal(size=n), "qcd": rng.normal(size=n),
    })
    corr = D.feature_correlation_matrix(df, "H", "chloroform")
    assert corr.loc["stationary", "pcm"] == pytest.approx(1.0, abs=1e-9)
    corr2 = D.feature_correlation_matrix(df, "H", "chloroform", squared=True)
    assert corr2.loc["stationary", "pcm"] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
def test_average_feature_correlation_matrix_averages_across_solvents():
    df = pd.DataFrame({
        "nucleus": ["H"] * 4, "solvent": ["chloroform", "chloroform", "benzene", "benzene"],
        "stationary": [1.0, 2.0, 1.0, 3.0], "pcm": [1.0, 2.0, 3.0, 1.0],
        "desmond": [0.5, 0.5, 0.5, 0.5], "desmond_vib": [0.1, 0.2, 0.1, 0.2], "qcd": [0, 0, 0, 0],
    })
    avg = D.average_feature_correlation_matrix(df, "H", ["chloroform", "benzene"])
    chloroform_only = D.feature_correlation_matrix(df, "H", "chloroform")
    benzene_only = D.feature_correlation_matrix(df, "H", "benzene")
    expected = (chloroform_only.loc["stationary", "pcm"] + benzene_only.loc["stationary", "pcm"]) / 2
    assert avg.loc["stationary", "pcm"] == pytest.approx(expected)


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
def test_average_feature_correlation_matrix_ignores_a_solvent_with_too_few_sites():
    # benzene has only 1 site -> feature_correlation_matrix returns all-NaN for benzene alone
    # (< 2 sites). A plain sum()/len() average would make the WHOLE averaged cell NaN even though
    # chloroform (4 sites) has a perfectly good value; nanmean should use chloroform alone instead.
    df = pd.DataFrame({
        "nucleus": ["H"] * 5,
        "solvent": ["chloroform", "chloroform", "chloroform", "chloroform", "benzene"],
        "stationary": [1.0, 2.0, 3.0, 4.0, 1.0], "pcm": [2.0, 4.0, 6.0, 8.0, 5.0],
        "desmond": [0.5] * 5, "desmond_vib": [0.1] * 5, "qcd": [0.0] * 5,
    })
    avg = D.average_feature_correlation_matrix(df, "H", ["chloroform", "benzene"])
    assert not np.isnan(avg.loc["stationary", "pcm"])
    assert avg.loc["stationary", "pcm"] == pytest.approx(1.0)   # chloroform alone: perfectly correlated


@pytest.mark.skipif(
    not (os.path.exists(REAL_H5) and os.path.exists(REAL_XLSX)),
    reason="real delta22.hdf5 / experimental xlsx not present")
def test_si_s15_feature_correlations_reproduces_published_matrices():
    """Every entry of both nuclei's published Pearson R and R^2 "Correlations Between Features"
    matrices (SI Figure S15), reproduced from the released data to 3 decimal places. Guards against
    pooling all DFT methods/bases/geometries together instead of filtering to the MagNET-Zero
    reference level, which throws entries off by up to 0.03."""
    dft = D.load_query_df_dft(REAL_H5, REAL_XLSX, verbose=False)
    # R^2 here is the average of each solvent's r^2 (Jensen's inequality: != the square of the
    # averaged r), so it is hardcoded separately from published_r, not derived from it.
    published_r = {
        "H": {("pcm", "stationary"): 0.298, ("desmond", "stationary"): 0.262, ("desmond", "pcm"): 0.399,
              ("desmond_vib", "stationary"): -0.170, ("desmond_vib", "pcm"): -0.219, ("desmond_vib", "desmond"): -0.141,
              ("qcd", "stationary"): -0.851, ("qcd", "pcm"): -0.362, ("qcd", "desmond"): -0.394, ("qcd", "desmond_vib"): 0.405},
        "C": {("pcm", "stationary"): 0.584, ("desmond", "stationary"): 0.106, ("desmond", "pcm"): 0.667,
              ("desmond_vib", "stationary"): 0.531, ("desmond_vib", "pcm"): 0.369, ("desmond_vib", "desmond"): 0.075,
              ("qcd", "stationary"): -0.616, ("qcd", "pcm"): -0.635, ("qcd", "desmond"): -0.418, ("qcd", "desmond_vib"): -0.138},
    }
    published_r2 = {
        "H": {("pcm", "stationary"): 0.089, ("desmond", "stationary"): 0.093, ("desmond", "pcm"): 0.386,
              ("desmond_vib", "stationary"): 0.032, ("desmond_vib", "pcm"): 0.049, ("desmond_vib", "desmond"): 0.025,
              ("qcd", "stationary"): 0.724, ("qcd", "pcm"): 0.131, ("qcd", "desmond"): 0.184, ("qcd", "desmond_vib"): 0.167},
        "C": {("pcm", "stationary"): 0.341, ("desmond", "stationary"): 0.047, ("desmond", "pcm"): 0.463,
              ("desmond_vib", "stationary"): 0.284, ("desmond_vib", "pcm"): 0.139, ("desmond_vib", "desmond"): 0.019,
              ("qcd", "stationary"): 0.380, ("qcd", "pcm"): 0.404, ("qcd", "desmond"): 0.189, ("qcd", "desmond_vib"): 0.021},
    }
    for nucleus in ("H", "C"):
        corr = D.si_s15_feature_correlations(dft, nucleus)
        for (a, b), expected in published_r[nucleus].items():
            assert corr["r"].loc[a, b] == pytest.approx(expected, abs=5e-4), f"{nucleus} r[{a},{b}]"
        for (a, b), expected in published_r2[nucleus].items():
            assert corr["r2"].loc[a, b] == pytest.approx(expected, abs=5e-4), f"{nucleus} r2[{a},{b}]"


@pytest.mark.skipif(
    not (os.path.exists(REAL_H5) and os.path.exists(REAL_XLSX)),
    reason="real delta22.hdf5 / experimental xlsx not present")
def test_pcm_desmond_correlation_by_solvent_reproduces_published_table():
    dft = D.load_query_df_dft(REAL_H5, REAL_XLSX, verbose=False)
    table = D.pcm_desmond_correlation_by_solvent(dft)
    published = {("H", "chloroform"): 0.739, ("H", "TIP4P"): 0.728, ("H", "benzene"): -0.454,
                ("C", "chloroform"): 0.408, ("C", "TIP4P"): 0.840, ("C", "benzene"): 0.594}
    for (nucleus, solvent), expected in published.items():
        assert table.loc[nucleus, solvent] == pytest.approx(expected, abs=5e-4), f"{nucleus}/{solvent}"


@pytest.mark.skipif(
    not (os.path.exists(REAL_H5) and os.path.exists(REAL_XLSX)),
    reason="real delta22.hdf5 / experimental xlsx not present")
def test_pcm_benefit_reproduces_published_figure_s5_correlations():
    # Figure S5 aggregates the per-split PCM benefit with the mean across splits, not the median;
    # only the mean reproduces the published Pearson R values.
    dft = D.load_query_df_dft(REAL_H5, REAL_XLSX, verbose=False)
    solutes = sorted(dft["solute"].unique())

    def pearson_r(benefit, prop, exclude=()):
        s = benefit.drop(index=[x for x in exclude if x in benefit.index])
        x = np.array([prop[k] for k in s.index])
        return float(np.corrcoef(x, s.values)[0, 1])

    aromatics_tfe = ["benzene", "toluene", "chlorobenzene", "trifluoroethanol"]
    benefit_h = D.pcm_benefit_per_solvent(dft, D.MAGNET_PCM_OUTPUT_METHODS["H"], "pcSseg2", "aimnet2",
                                          D.DESMOND_SOLVENTS, n_splits=250, solutes=solutes, nucleus="H")
    benefit_c = D.pcm_benefit_per_solvent(dft, D.MAGNET_PCM_OUTPUT_METHODS["C"], "pcSseg2", "aimnet2",
                                          D.DESMOND_SOLVENTS, n_splits=250, solutes=solutes, nucleus="C")
    published = {  # (panel, axis): published Pearson R
        "S5A 1H dielectric": (pearson_r(benefit_h, D.SOLVENT_DIELECTRIC), 0.617),
        "S5A 1H polarizability": (pearson_r(benefit_h, D.SOLVENT_POLARIZABILITY), -0.801),
        "S5B 13C dielectric": (pearson_r(benefit_c, D.SOLVENT_DIELECTRIC), 0.722),
        "S5B 13C polarizability": (pearson_r(benefit_c, D.SOLVENT_POLARIZABILITY), -0.743),
        "S5C 1H dielectric": (pearson_r(benefit_h, D.SOLVENT_DIELECTRIC, aromatics_tfe), 0.884),
        "S5C 1H polarizability": (pearson_r(benefit_h, D.SOLVENT_POLARIZABILITY, aromatics_tfe), -0.587),
    }
    for name, (got, expected) in published.items():
        assert got == pytest.approx(expected, abs=2e-3), f"{name}: {got} != {expected}"
