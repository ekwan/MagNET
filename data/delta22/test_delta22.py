"""
Tests for the delta22 reader (delta22_reader.py).

These build a tiny synthetic dataset in the *exact* on-disk format (so the fixture also
serves as an executable spec of the layout) plus a matching experimental spreadsheet, then
run the public loaders against them. No dependency on the multi-GB real file. One opt-in
test runs against the real `delta22.hdf5` if it is present next to this file.

Run:  pytest test_delta22.py -q
Requires: pytest, numpy, h5py, pandas, openpyxl, tqdm.
"""
import os
import numpy as np
import pandas as pd
import h5py
import pytest

import delta22_reader as R
from delta22_reader import (
    _decode_fixed_point,
    _MISSING_MARKER,
    load_delta22_dft_data,
    load_delta22_nn_data,
    load_delta22_dft_timings,
    load_delta22_nn_timings,
)

SCALE = 1e4


def _enc(arr):
    """Encode floats the way the real file does: int32 = round(value * 1e4), NaN -> marker."""
    a = np.asarray(arr, dtype=np.float64)
    mask = np.isnan(a)
    out = np.zeros(a.shape, dtype=np.int32)
    out[~mask] = np.rint(a[~mask] * SCALE).astype(np.int32)
    out[mask] = _MISSING_MARKER
    return out


# ---- known synthetic content -------------------------------------------------
# One solute "moltest", 4 atoms (H, C, H, C); one solvent (chloroform); one method
# (b3lyp_d3bj/pcSseg2, which is also the default PCM reference) present as a gas entry
# and a chloroform-PCM entry; two geometries (aimnet2, pbe0_tz).
SOLUTE = "moltest"
Z = np.array([1, 6, 1, 6], np.uint8)          # 1-based H at 1,3 ; C at 2,4
METHOD_NAMES = [
    'b3lyp_d3bj,pcSseg2,gas,"none"',
    'b3lyp_d3bj,pcSseg2,pcm,"chloroform"',
]
# conventional_shieldings[method, geometry, atom]
GAS = np.array([[30.0, 150.0, 31.0, 151.0],    # aimnet2
                [30.2, 150.4, 31.2, 151.4]])   # pbe0_tz
PCM = np.array([[30.5, 151.0, 31.5, 152.0],    # aimnet2  (+0.5 H, +1.0 C vs gas)
                [30.7, 151.4, 31.7, 152.4]])   # pbe0_tz
CONV = np.stack([GAS, PCM])                     # (2 methods, 2 geom, 4 atoms)

NN_GAS = np.array([30.1, 150.1, 31.1, 151.1])
NN_PCM = np.array([0.4, 0.9, 0.4, 0.9])         # chloroform - gas

# Expected stationary shift for site Ha_H (atoms 1,3) on the aimnet2 geometry:
EXP_STATIONARY_HA_AIMNET2 = np.mean([30.0, 31.0])   # 30.5
# Expected PCM correction for Ha_H aimnet2 = mean(PCM 1,3) - mean(GAS 1,3):
EXP_PCM_HA_AIMNET2 = np.mean([30.5, 31.5]) - np.mean([30.0, 31.0])  # 0.5


def _write_solute(grp):
    grp.create_dataset("atomic_numbers", data=Z)

    sap = grp.create_group("stationary_and_pcm")
    sap.create_dataset("conventional_shieldings", data=_enc(CONV))
    sap.create_dataset("geometries", data=_enc(np.zeros((2, 4, 3))))
    sap.create_dataset("conventional_nmr_timings", data=np.ones((2, 2)) * 1.5)
    sap.create_dataset("geometry_optimization_timings", data=np.array([2.0, 9.0]))
    sap.create_dataset("nn_gas_shieldings", data=_enc(NN_GAS))     # (4,) flat
    sap.create_dataset("nn_pcm_corrections", data=_enc(NN_PCM))    # (4,) flat
    sap.create_dataset("nn_nmr_timings", data=np.array([0.1, 0.2]))

    qcd = grp.create_group("qcd").create_group("gas_phase")
    qcd.create_dataset("unperturbed_dft_shieldings", data=_enc(GAS[0]))
    qcd.create_dataset("unperturbed_nn_shieldings", data=_enc(NN_GAS))
    # (n_trajectories=2, n_frames=2, n_atoms=4); means equal the unperturbed -> 0 correction
    qcd.create_dataset("perturbed_dft_shieldings",
                       data=_enc(np.broadcast_to(GAS[0], (2, 2, 4)).copy()))
    qcd.create_dataset("perturbed_nn_shieldings",
                       data=_enc(np.broadcast_to(NN_GAS, (2, 2, 4)).copy()))
    # geometry companions (coordinates, never consumed by the shielding loaders)
    qcd.create_dataset("unperturbed_geometry", data=_enc(np.full((4, 3), 0.5)))
    qcd.create_dataset("perturbed_geometries", data=_enc(np.full((2, 2, 4, 3), 0.5)))

    for engine in ("desmond", "openMM"):
        eg = grp.create_group(engine)
        gp = eg.create_group("gas_phase")
        gp.create_dataset("unperturbed_solute_geometry", data=_enc(np.zeros((4, 3))))
        gp.create_dataset("unperturbed_dft_shieldings", data=_enc(GAS[0]))
        gp.create_dataset("unperturbed_nn_shieldings", data=_enc(NN_GAS))
        sv = eg.create_group("chloroform")
        # (n_frames=3, n_atoms=4, 2) -> [:, :, 0] isolated, [:, :, 1] solvated
        iso = np.broadcast_to(GAS[0], (3, 4)).copy()
        solv = iso + 0.2                                  # +0.2 ppm solvent shift
        solv[2, 0] = np.nan                               # exercise the missing-value path
        dft = np.stack([iso, solv], axis=-1)
        nn = np.stack([np.broadcast_to(NN_GAS, (3, 4)).copy(),
                       np.broadcast_to(NN_GAS, (3, 4)).copy() + 0.2], axis=-1)
        sv.create_dataset("perturbed_dft_shieldings", data=_enc(dft))
        sv.create_dataset("perturbed_nn_shieldings", data=_enc(nn))
        # solvated-cluster coordinates: 4 solute atoms + one 3-atom solvent block = 7
        sv.create_dataset("perturbed_ensemble_geometries", data=_enc(np.full((3, 7, 3), 1.25)))


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    d = tmp_path_factory.mktemp("d22")
    hdf5 = str(d / "mini.hdf5")
    xlsx = str(d / "mini_experimental.xlsx")
    with h5py.File(hdf5, "w") as f:
        f.create_dataset("conventional_nmr_method_names",
                         data=np.array(METHOD_NAMES, dtype=h5py.string_dtype()))
        _write_solute(f.create_group("solutes").create_group(SOLUTE))
        # top-level solvent topology: one 3-atom solvent block per engine ordering
        cf = f.create_group("solvents").create_group("chloroform")
        cf.create_dataset("desmond_atomic_numbers", data=np.array([6, 1, 17], np.uint8))
        cf.create_dataset("openMM_atomic_numbers", data=np.array([6, 1, 17], np.uint8))
    pd.DataFrame({
        "solute": [SOLUTE, SOLUTE],
        "site": ["Ha_H", "Ca_C"],
        "atom_numbers": ["1,3", "2,4"],
        "chloroform": [2.10, 30.0],
    }).to_excel(xlsx, index=False)
    return hdf5, xlsx


# ---- the encode/decode round-trip (the one new piece of logic) ----
def test_decode_scaling_and_missing():
    enc = np.array([2070700, _MISSING_MARKER, -150000], np.int32)
    out = _decode_fixed_point(enc)
    assert out[0] == pytest.approx(207.07)
    assert np.isnan(out[1])
    assert out[2] == pytest.approx(-15.0)


def test_decode_passes_floats_through():
    f = np.array([1.5, 2.5], np.float64)
    np.testing.assert_array_equal(_decode_fixed_point(f), f)


# ---- the full reader against the synthetic file ----
EXPECTED_COLUMNS = ["experimental", "stationary", "qcd", "pcm",
                    "desmond", "openMM", "desmond_vib", "openMM_vib"]
EXPECTED_INDEX = ["solute", "sap_geometry_type", "sap_nmr_method",
                  "sap_basis", "nucleus", "site", "solvent"]


def test_dft_loader_shape_and_decode(fixture):
    hdf5, xlsx = fixture
    df = load_delta22_dft_data(hdf5, xlsx, verbose=False)
    assert list(df.index.names) == EXPECTED_INDEX
    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) > 0
    row = df.reset_index().query(
        "site == 'Ha_H' and solvent == 'chloroform' and sap_geometry_type == 'aimnet2'"
    )
    assert len(row) == 1
    assert row["stationary"].iloc[0] == pytest.approx(EXP_STATIONARY_HA_AIMNET2, abs=1e-3)
    assert row["pcm"].iloc[0] == pytest.approx(EXP_PCM_HA_AIMNET2, abs=1e-3)
    # explicit correction is +0.2 ppm; the seeded missing frame must not break it
    assert row["desmond"].iloc[0] == pytest.approx(0.2, abs=1e-3)
    assert np.isfinite(row["desmond"].iloc[0])


def test_nn_loader_shape(fixture):
    hdf5, xlsx = fixture
    df = load_delta22_nn_data(hdf5, xlsx, verbose=False)
    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) > 0
    assert df.index.get_level_values("sap_nmr_method").unique().tolist() == ["MagNET"]


def test_load_perturbed_shieldings(fixture):
    hdf5, _ = fixture
    raw = R.load_perturbed_shieldings(hdf5, SOLUTE, "chloroform", "openMM", "dft")
    assert raw.shape == (3, 4, 2)                 # (n_frames, n_atoms, [isolated, solvated])
    # the fixture set solvated = isolated + 0.2 with one missing value seeded
    corr = raw[:, :, 1] - raw[:, :, 0]
    assert np.nanmax(np.abs(corr - 0.2)) < 1e-3
    assert np.isnan(raw[2, 0, 1])                 # the seeded missing solvated value


def test_nn_loader_exclude_solutes(fixture):
    hdf5, xlsx = fixture
    full = load_delta22_nn_data(hdf5, xlsx, verbose=False)
    assert SOLUTE in full.index.get_level_values("solute")
    excluded = load_delta22_nn_data(hdf5, xlsx, exclude_solutes=[SOLUTE], verbose=False)
    assert SOLUTE not in excluded.index.get_level_values("solute")
    assert len(excluded) == 0   # the fixture has only the one solute


def test_timings_loaders(fixture):
    hdf5, _ = fixture
    dft = load_delta22_dft_timings(hdf5)
    nn = load_delta22_nn_timings(hdf5)
    for t in (dft, nn):
        assert list(t.columns) == ["geometry_time", "nmr_time", "total_time"]
        assert (t["total_time"] >= 0).all()


# ---- geometry + topology accessors (the raw stored arrays the loaders don't expose) ----
def test_solutes_listing(fixture):
    hdf5, _ = fixture
    assert R.load_solutes(hdf5) == [SOLUTE]


def test_geometry_accessors_shape_and_decode(fixture):
    hdf5, _ = fixture
    sg = R.load_stationary_geometries(hdf5, SOLUTE)
    assert set(sg) == {"aimnet2", "pbe0_tz"}
    assert sg["aimnet2"].shape == (4, 3)
    q = R.load_qcd_geometries(hdf5, SOLUTE)
    assert q["unperturbed"].shape == (4, 3)
    assert q["perturbed"].shape == (2, 2, 4, 3)
    assert q["unperturbed"][0, 0] == pytest.approx(0.5)        # coordinates decode from fixed point
    assert q["perturbed"][0, 0, 0, 0] == pytest.approx(0.5)
    e = R.load_explicit_ensemble_geometries(hdf5, SOLUTE, "chloroform", "openMM")
    assert e.shape == (3, 7, 3)                                # 4 solute + 3 solvent atoms
    assert e[0, 0, 0] == pytest.approx(1.25)
    gs = R.load_explicit_solute_geometry(hdf5, SOLUTE, "desmond")
    assert gs.shape == (4, 3)


def test_solvent_atomic_numbers(fixture):
    hdf5, _ = fixture
    for engine in ("desmond", "openMM"):
        an = R.load_solvent_atomic_numbers(hdf5, "chloroform", engine)
        assert list(an) == [6, 1, 17]


# ---------- opt-in: real file, if present ----------
REAL = os.path.join(os.path.dirname(__file__), "delta22.hdf5")
REAL_XLSX = os.path.join(os.path.dirname(__file__), "delta22_experimental.xlsx")


@pytest.mark.skipif(not (os.path.exists(REAL) and os.path.exists(REAL_XLSX)),
                    reason="real delta22.hdf5 / experimental xlsx not present")
def test_real_file_smoke():
    with h5py.File(REAL, "r") as f:
        assert len(f["solutes"].keys()) == 22
        assert len(f["conventional_nmr_method_names"]) == 473
        any_solute = next(iter(f["solutes"].keys()))
        desmond = [k for k in f["solutes"][any_solute]["desmond"].keys() if k != "gas_phase"]
        assert len(desmond) == 12
    df = load_delta22_dft_data(REAL, REAL_XLSX, verbose=False)
    assert len(df) > 0
    assert list(df.columns) == EXPECTED_COLUMNS
    # the geometry + topology accessors reach the real stored arrays and decode to angstroms
    sol = R.load_solutes(REAL)[0]
    sg = R.load_stationary_geometries(REAL, sol)
    assert sg["aimnet2"].ndim == 2 and sg["aimnet2"].shape[1] == 3
    assert np.nanmax(np.abs(sg["aimnet2"])) < 100.0
    an = R.load_solvent_atomic_numbers(REAL, "chloroform", "openMM")
    assert an.ndim == 1 and len(an) > 0
