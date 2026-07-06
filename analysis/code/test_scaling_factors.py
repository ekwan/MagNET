"""Tests for analysis/code/scaling_factors.py.

Synthetic tests exercise the two table builders and the prediction equation without any large file,
so the core math runs in CI. The opt-in real-data test reproduces the published SI numbers (Tables
S10 and S11) from the released delta-22 data when it is present.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import scaling_factors as S  # noqa: E402
import paths as P  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
REAL_H5 = P.dataset_file("delta22", root=REPO)
REAL_XLSX = os.path.join(REPO, "data", "delta22", "delta22_experimental.xlsx")


# --------------------------------------------------------------------------- synthetic builders

def _proton_nn(coeffs, n=6, seed=0):
    """Synthetic proton table where experimental = a + b*stationary + c*pcm exactly, per solvent.
    coeffs maps solvent -> (a, b, c)."""
    rng = np.random.default_rng(seed)
    rows = []
    for solvent, (a, b, c) in coeffs.items():
        for i in range(n):
            stat = float(rng.normal(30, 3))
            pcm = float(rng.normal(0, 0.5))
            rows.append(dict(nucleus="H", solute=f"m{i}", site=f"m{i}_0", solvent=solvent,
                             stationary=stat, pcm=pcm, experimental=a + b * stat + c * pcm))
    return pd.DataFrame(rows)


def _carbon_nn(coeffs, factors, n=6, seed=1):
    """Synthetic carbon table where experimental = a + b*(stationary + factor*pcm) exactly, per
    solvent. coeffs maps solvent -> (a, b); factors maps solvent -> conversion factor."""
    rng = np.random.default_rng(seed)
    rows = []
    for solvent, (a, b) in coeffs.items():
        factor = factors[solvent]
        for i in range(n):
            stat = float(rng.normal(100, 20))
            pcm = float(rng.normal(0, 0.5))
            rows.append(dict(nucleus="C", solute=f"m{i}", site=f"m{i}_0", solvent=solvent,
                             stationary=stat, pcm=pcm, experimental=a + b * (stat + factor * pcm)))
    return pd.DataFrame(rows)


def _nitromethane_rows(nucleus, solvents):
    """One garbage nitromethane row per solvent; the fits must exclude it."""
    return pd.DataFrame([dict(nucleus=nucleus, solute="nitromethane", site="x", solvent=s,
                              stationary=0.0, pcm=0.0, experimental=999.0) for s in solvents])


# --------------------------------------------------------------------------- synthetic tests

def test_recommended_model_forms():
    # Proton is the three-parameter model, carbon the two-parameter model (John's fitting trials).
    assert S.RECOMMENDED_MODEL == {"H": "three_parameter", "C": "two_parameter"}


def test_predict_shift_equation():
    table = pd.DataFrame(
        {"intercept": [31.0], "stationary": [-0.95], "pcm": [-0.85]},
        index=pd.Index(["chloroform"], name="solvent"),
    )
    got = S.predict_shift(table, "chloroform", magnet_zero_shielding=25.0,
                          magnet_pcm_chloroform_correction=-0.1)
    assert got == pytest.approx(31.0 - 0.95 * 25.0 - 0.85 * -0.1)
    got_vec = S.predict_shift(table, "chloroform", [25.0, 26.0], [-0.1, 0.2])
    assert np.allclose(got_vec, [31.0 - 0.95 * 25.0 - 0.85 * -0.1,
                                 31.0 - 0.95 * 26.0 - 0.85 * 0.2])


def test_proton_table_recovers_known_line_and_drops_nitromethane():
    coeffs = {"chloroform": (31.0, -0.97, -0.85), "benzene": (32.0, -1.00, 2.20)}
    nn = pd.concat([_proton_nn(coeffs), _nitromethane_rows("H", coeffs)], ignore_index=True)
    table = S.proton_scaling_table(nn, solvents=list(coeffs))
    assert list(table.columns) == ["intercept", "stationary", "pcm"]
    assert table.index.name == "solvent"
    for solvent, (a, b, c) in coeffs.items():
        row = table.loc[solvent]
        # exact recovery (and the nitromethane garbage row did not perturb it -> it was excluded)
        assert row["intercept"] == pytest.approx(a, abs=1e-6)
        assert row["stationary"] == pytest.approx(b, abs=1e-6)
        assert row["pcm"] == pytest.approx(c, abs=1e-6)


def test_carbon_table_reconstruction_and_drops_nitromethane():
    coeffs = {"chloroform": (171.0, -0.92), "benzene": (172.0, -0.93)}
    factors = {"chloroform": 1.0, "benzene": 0.63}
    nn = pd.concat([_carbon_nn(coeffs, factors), _nitromethane_rows("C", coeffs)], ignore_index=True)
    table = S.carbon_scaling_table(nn, solvents=list(coeffs), conversion_factors=factors)
    assert list(table.columns) == ["intercept", "stationary", "pcm"]
    for solvent, (a, b) in coeffs.items():
        row = table.loc[solvent]
        assert row["intercept"] == pytest.approx(a, abs=1e-6)
        assert row["stationary"] == pytest.approx(b, abs=1e-6)
        # the reported pcm is the shared slope times the conversion factor
        assert row["pcm"] == pytest.approx(b * factors[solvent], abs=1e-6)
        assert row["pcm"] == pytest.approx(row["stationary"] * factors[solvent], abs=1e-12)


def test_carbon_empty_or_nan_factor_gives_nan_row():
    # chloroform has data and a finite factor; benzene has no rows and a NaN factor.
    nn = _carbon_nn({"chloroform": (171.0, -0.92)}, {"chloroform": 1.0})
    table = S.carbon_scaling_table(nn, solvents=["chloroform", "benzene"],
                                   conversion_factors={"chloroform": 1.0, "benzene": np.nan})
    assert np.isfinite(table.loc["chloroform", "intercept"])
    # a degenerate solvent must be NaN, not a fake (0, 0, 0) fit
    assert table.loc["benzene"].isna().all()


def test_carbon_requires_factors_or_dft():
    with pytest.raises(ValueError):
        S.carbon_scaling_table(_carbon_nn({"chloroform": (171.0, -0.92)}, {"chloroform": 1.0}),
                               solvents=["chloroform"])


# --------------------------------------------------------------------------- opt-in real-data test

# Published SI values: parameter tuples are (intercept, stationary, pcm) per solvent, all 12 solvents.
# These also pin the shipped published_scaling_tables() copy (no-data test below): shipped CSV ==
# this dict == build_scaling_tables() from delta-22 (real-data test). The chain keeps all three in sync.
PUBLISHED_S10_H = {
    "chloroform": (31.294997, -0.9757947, -0.8526936),
    "tetrahydrofuran": (31.321805, -0.9801755, -0.786672),
    "dichloromethane": (31.3773285, -0.979871, -0.8085722),
    "acetone": (31.512167, -0.9872612, -1.2383371),
    "acetonitrile": (31.4961183, -0.9857168, -0.9744366),
    "dimethylsulfoxide": (31.5987046, -0.9911032, -1.3551614),
    "trifluoroethanol": (30.8759924, -0.9599975, -0.9589483),
    "methanol": (31.2560285, -0.9764463, -1.2500843),
    "TIP4P": (31.3591125, -0.978808, -1.4782606),
    "benzene": (31.9876742, -1.0052929, 2.23638523),
    "toluene": (31.7170517, -0.9967364, 1.94472899),
    "chlorobenzene": (31.6814988, -0.9938483, 0.83014615),
}
PUBLISHED_S11_C = {
    "chloroform": (171.728792, -0.9242313, -0.9368043),
    "tetrahydrofuran": (171.054483, -0.9190001, -1.069509),
    "dichloromethane": (171.509383, -0.9211671, -1.1150154),
    "acetone": (171.308017, -0.9196101, -1.2395026),
    "acetonitrile": (171.879832, -0.9226291, -1.2876643),
    "dimethylsulfoxide": (170.598418, -0.9186745, -1.2965007),
    "trifluoroethanol": (174.237665, -0.9390125, -1.2900728),
    "methanol": (172.426154, -0.9275656, -1.2888469),
    "TIP4P": (173.696364, -0.937854, -1.3426678),
    "benzene": (171.967174, -0.9270331, -0.5937164),
    "toluene": (171.690904, -0.9249871, -0.6184099),
    "chlorobenzene": (171.075905, -0.921238, -0.997641),
}


def test_shipped_published_tables_match_si_values():
    """The shipped published_scaling_tables() copy (no data download needed) equals the published SI
    values for all 12 solvents and both nuclei. Runs in CI without delta-22; the real-data test below
    ties those same SI values back to a fit on the raw data, so the shipped copy cannot drift."""
    tables = S.published_scaling_tables()
    for nucleus, published in (("H", PUBLISHED_S10_H), ("C", PUBLISHED_S11_C)):
        table = tables[nucleus]
        assert list(table.columns) == ["intercept", "stationary", "pcm"]
        assert sorted(table.index) == sorted(published)
        for solvent, (intercept, stationary, pcm) in published.items():
            row = table.loc[solvent]
            assert row["intercept"] == pytest.approx(intercept, abs=5e-4), f"{nucleus} {solvent} int"
            assert row["stationary"] == pytest.approx(stationary, abs=5e-4), f"{nucleus} {solvent} stat"
            assert row["pcm"] == pytest.approx(pcm, abs=5e-4), f"{nucleus} {solvent} pcm"


@pytest.mark.skipif(not (os.path.exists(REAL_H5) and os.path.exists(REAL_XLSX)),
                    reason="real delta22.hdf5 / experimental xlsx not present")
def test_reproduces_published_si_tables():
    """Both recommended-scaling tables reproduce the published SI numbers for all 12 solvents. The
    tolerance is set above the int32-encoding floor (worst case ~1.3e-4, the benzene proton pcm) but
    tight enough to catch a real regression."""
    tables = S.build_scaling_tables(REAL_H5, REAL_XLSX)
    for nucleus, published in (("H", PUBLISHED_S10_H), ("C", PUBLISHED_S11_C)):
        table = tables[nucleus]
        assert list(table.columns) == ["intercept", "stationary", "pcm"]
        assert sorted(table.index) == sorted(published)
        for solvent, (intercept, stationary, pcm) in published.items():
            row = table.loc[solvent]
            assert row["intercept"] == pytest.approx(intercept, abs=5e-4), f"{nucleus} {solvent} int"
            assert row["stationary"] == pytest.approx(stationary, abs=5e-4), f"{nucleus} {solvent} stat"
            assert row["pcm"] == pytest.approx(pcm, abs=5e-4), f"{nucleus} {solvent} pcm"


# --------------------------------------------------------------------------- symmetrized-inference smoke test

CKPT_ROOT = P.checkpoints_root() or ""  # "" (deposit env unset) -> os.path.exists(...) below is False
CKPT_ZERO = os.path.join(CKPT_ROOT, "MagNET-Zero")
CKPT_PCM = os.path.join(CKPT_ROOT, "MagNET-PCM")
_HAS_CHECKPOINTS = os.path.exists(CKPT_ZERO) and os.path.exists(CKPT_PCM)


@pytest.mark.skipif(not (os.path.exists(REAL_H5) and os.path.exists(REAL_XLSX) and _HAS_CHECKPOINTS),
                    reason="real delta22.hdf5 / experimental xlsx / model checkpoints not present")
def test_symmetrized_build_scaling_tables_runs_and_is_close_to_published():
    """symmetrized=True re-derives the tables from live, reflection-symmetrized inference instead
    of the HDF5's stored (unsymmetrized) shieldings. This is a deployment-quality table (e.g. for
    serving MagNET-Zero/PCM on new molecules), not a replacement for the published SI numbers --
    it should be close (delta-22 solutes are small, so the correction is modest) but need not match
    to the same tight tolerance as test_reproduces_published_si_tables."""
    pytest.importorskip("torch")
    tables = S.build_scaling_tables(REAL_H5, REAL_XLSX, symmetrized=True, n_passes=2)
    max_abs_delta = 0.0
    for nucleus, published in (("H", PUBLISHED_S10_H), ("C", PUBLISHED_S11_C)):
        table = tables[nucleus]
        assert list(table.columns) == ["intercept", "stationary", "pcm"]
        assert sorted(table.index) == sorted(published)
        for solvent, (intercept, stationary, pcm) in published.items():
            row = table.loc[solvent]
            assert np.isfinite(row["intercept"]) and np.isfinite(row["stationary"]) and np.isfinite(row["pcm"])
            # loose tolerance: this is a methodology check (does it run and land in the right
            # ballpark), not a bit-for-bit reproduction -- see module docstring for measured deltas
            assert row["intercept"] == pytest.approx(intercept, abs=0.05), f"{nucleus} {solvent} int"
            assert row["stationary"] == pytest.approx(stationary, abs=0.01), f"{nucleus} {solvent} stat"
            assert row["pcm"] == pytest.approx(pcm, abs=0.1), f"{nucleus} {solvent} pcm"
            max_abs_delta = max(max_abs_delta, abs(row["intercept"] - intercept),
                                abs(row["stationary"] - stationary), abs(row["pcm"] - pcm))
    # symmetrized inference must actually change something -- if nn_shieldings_override_df were
    # silently dropped and this fell back to the exact HDF5 path, every delta above would be 0.0
    # and still pass the loose tolerances; this catches that failure mode specifically
    assert max_abs_delta > 1e-4, "symmetrized=True produced numbers identical to published -- the override path did not run"


@pytest.mark.skipif(not (os.path.exists(REAL_H5) and os.path.exists(REAL_XLSX) and _HAS_CHECKPOINTS),
                    reason="real delta22.hdf5 / experimental xlsx / model checkpoints not present")
def test_shipped_symmetrized_csvs_reproduce_from_live_inference():
    """Value-traceability for the deployment CSVs in data/scaling_factors/: they must reproduce from
    a live symmetrized build at the n_passes they were generated with (10), not merely parse. A
    swapped column, wrong solvent order, or stale hand-edit of the shipped CSVs fails here. The
    tolerance absorbs the model's pass-to-pass inference noise (a single forward pass is not
    deterministic) but is far tighter than any structural error."""
    pytest.importorskip("torch")
    sys.path.insert(0, os.path.join(REPO, "data", "scaling_factors"))
    import scaling_factors_reader as R  # noqa: E402
    shipped = R.load_symmetrized_tables()
    fresh = S.build_scaling_tables(REAL_H5, REAL_XLSX, symmetrized=True, n_passes=10)
    for nucleus in ("H", "C"):
        s, f = shipped[nucleus], fresh[nucleus]
        assert list(s.columns) == list(f.columns) == ["intercept", "stationary", "pcm"]
        assert sorted(s.index) == sorted(f.index)
        for solvent in s.index:
            assert s.loc[solvent, "intercept"] == pytest.approx(f.loc[solvent, "intercept"], abs=0.03), f"{nucleus} {solvent} intercept"
            assert s.loc[solvent, "stationary"] == pytest.approx(f.loc[solvent, "stationary"], abs=0.01), f"{nucleus} {solvent} stationary"
            assert s.loc[solvent, "pcm"] == pytest.approx(f.loc[solvent, "pcm"], abs=0.02), f"{nucleus} {solvent} pcm"


def test_magnet_package_scaling_matches_canonical_tables():
    """The tiny copy shipped in the importable package (magnet.scaling, used internally by
    magnet.predict_shifts) must equal the canonical SI tables here, so the two can never drift."""
    pytest.importorskip("torch")   # importing the magnet package pulls in the torch stack
    from magnet import scaling as MS
    canonical = S.published_scaling_tables()      # DataFrames indexed by solvent
    tiny = MS.published_scaling_tables()          # {solvent: {column: value}} dicts
    for nucleus in ("H", "C"):
        assert set(tiny[nucleus]) == set(canonical[nucleus].index)
        for solvent, coeffs in tiny[nucleus].items():
            for column in ("intercept", "stationary", "pcm"):
                assert coeffs[column] == pytest.approx(float(canonical[nucleus].loc[solvent, column]))
    # the two predict_shift implementations give the same shift
    assert float(MS.predict_shift(tiny["C"], "chloroform", 170.0, -0.3)) == pytest.approx(
        float(S.predict_shift(canonical["C"], "chloroform", 170.0, -0.3)))
