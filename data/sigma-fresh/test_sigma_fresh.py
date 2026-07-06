"""
Tests for the sigma-fresh reader (decode_sigma_fresh.SigmaFresh).

A tiny synthetic dataset is built in the *exact* on-disk format (so the fixture also
serves as an executable spec of the layout) and the decoder is exercised against it.
One opt-in test runs against the real `sigma-fresh.hdf5` if present next to this file.

Run:  pytest test_sigma_fresh.py -q
Requires: pytest, numpy, h5py.
"""
import os
import numpy as np
import h5py
import pytest

from decode_sigma_fresh import SigmaFresh, SENTINEL

SCALE = 1e-4


def _qc(x):
    return np.round(np.asarray(x, float) / SCALE).astype(np.int32)


def _qs(x):
    x = np.asarray(x, float)
    out = np.where(np.isnan(x), SENTINEL, np.round(x / SCALE)).astype(np.int32)
    return out


def _solute(g, name, *, idx, n_solute, znums, coords, iso, solv, status, nsolv, prad, frad, charge=0, typ="train"):
    s = g.create_group(f"solute_{idx:05d}")
    s.attrs["name"] = name
    s.attrs["type"] = typ
    s.attrs["n_solute_atoms"] = n_solute
    s.attrs["solute_charge"] = charge
    s.create_dataset("atomic_numbers", data=np.asarray(znums, np.int8))
    s.create_dataset("coordinates", data=_qc(coords))
    s.create_dataset("shielding_isolated", data=_qs(iso))
    s.create_dataset("shielding_solvated", data=_qs(solv))
    s.create_dataset("status", data=np.asarray(status, np.uint8))
    s.create_dataset("n_solvents_partial", data=np.asarray(nsolv, np.int32))
    s.create_dataset("partial_radius", data=_qc(prad))
    s.create_dataset("full_radius", data=_qc(frad))


@pytest.fixture(scope="module")
def mini_path(tmp_path_factory):
    p = str(tmp_path_factory.mktemp("sf") / "mini.hdf5")
    with h5py.File(p, "w") as f:
        f.attrs["dataset"] = "sigma-fresh"
        f.attrs["solvents"] = "chloroform,benzene"
        f.attrs["sentinel"] = int(SENTINEL)
        f.attrs["scale"] = SCALE
        f.attrs["level_of_theory"] = "PBE0/pcSseg-1 (solute), PBE0/MIDI! (solvent)"
        # chloroform: solute_00001 has 3 frames; solute (2 atoms) + 1 chloroform (5 atoms) = 7
        gc = f.create_group("chloroform"); gc.attrs["n_solvent_atoms"] = 5
        Z = [6, 1, 6, 1, 17, 17, 17]                 # 2 solute + 5 solvent
        n_solute = 2
        # 4 frames: complete, geometry-only (all NaN), complete, partial (isolated finite, solvated NaN)
        coords = np.zeros((4, 7, 3), float)
        for fr in range(4):
            coords[fr] = np.arange(7 * 3).reshape(7, 3) * 0.1 + fr   # raw, frame-shifted (uncentered)
        iso = np.array([[150.0, 30.0], [np.nan, np.nan], [151.0, 31.0], [152.0, 32.0]])
        solv = np.array([[149.5, 29.8], [np.nan, np.nan], [150.6, 30.7], [np.nan, np.nan]])
        _solute(gc, "GLY000001", idx=1, n_solute=n_solute, znums=Z, coords=coords,
                iso=iso, solv=solv, status=[3, 1, 3, 2], nsolv=[1, 1, 1, 1],
                prad=[3.1, 3.1, 3.2, 3.2], frad=[9.9, 9.9, 9.9, 9.9], typ="train")
        # benzene: a test solute, 1 frame complete
        gb = f.create_group("benzene"); gb.attrs["n_solvent_atoms"] = 12
        Zb = [8, 1]                                   # 2-atom solute, 0 solvent (degenerate but valid)
        _solute(gb, "MOL000009", idx=9, n_solute=2, znums=Zb,
                coords=np.array([[[0, 0, 0], [1.0, 0, 0]]], float),
                iso=np.array([[280.0, 32.0]]), solv=np.array([[279.0, 31.5]]),
                status=[3], nsolv=[0], prad=[0.0], frad=[0.0], typ="test")
    return p


@pytest.fixture
def ds(mini_path):
    with SigmaFresh(mini_path) as d:
        yield d


def test_catalogue(ds):
    assert ds.solvents == ["chloroform", "benzene"]
    assert ds.n_solutes("chloroform") == 1 and ds.n_solutes("benzene") == 1
    assert ds.solutes("chloroform") == ["solute_00001"]


def test_pose_values_roundtrip(ds):
    p = ds.pose("chloroform", 1, 1)        # 1-based frame
    assert p["name"] == "GLY000001" and p["type"] == "train" and p["status"] == "complete"
    np.testing.assert_array_equal(p["atomic_numbers"], [6, 1, 6, 1, 17, 17, 17])
    np.testing.assert_allclose(p["shielding_isolated"], [150.0, 30.0], atol=5e-5)
    np.testing.assert_allclose(p["shielding_solvated"], [149.5, 29.8], atol=5e-5)
    assert p["n_solvents_partial"] == 1
    np.testing.assert_allclose(p["partial_radius"], 3.1, atol=5e-5)


def test_shieldings_solute_only(ds):
    p = ds.pose("chloroform", 1, 1)
    # shieldings have one entry per solute atom, not per cluster atom
    assert p["shielding_isolated"].shape == (2,)
    assert p["coordinates"].shape == (7, 3)


def test_geometry_only_frame_is_nan(ds):
    p = ds.pose("chloroform", 1, 2)        # frame 2 = geometry-only
    assert p["status"] == "geometries_only"
    assert np.isnan(p["shielding_isolated"]).all()
    assert np.isnan(p["shielding_solvated"]).all()
    # geometry is still valid on a geometry-only frame
    assert np.isfinite(p["coordinates"]).all()


def test_raw_coordinates_not_centered(ds):
    # frames are stored raw: distinct frames have distinct centroids (no per-frame centering)
    c1 = ds.pose("chloroform", 1, 1)["coordinates"].mean(0)
    c3 = ds.pose("chloroform", 1, 3)["coordinates"].mean(0)
    assert not np.allclose(c1, c3)


def test_solute_bundle_and_mask(ds):
    s = ds.solute("chloroform", 1)
    assert s["coordinates"].shape == (4, 7, 3)
    assert s["shielding_isolated"].shape == (4, 2)
    assert s["solute_mask"].tolist() == [True, True, False, False, False, False, False]
    assert list(s["status"]) == ["complete", "geometries_only", "complete", "partial_shieldings"]
    # correction is finite on complete frames, NaN on the geometry-only and partial frames
    corr = s["shielding_solvated"] - s["shielding_isolated"]
    assert np.isfinite(corr[0]).all() and np.isnan(corr[1]).all() and np.isnan(corr[3]).all()


def test_partial_status_frame(ds):
    # status==2: isolated shielding present, solvated not computed (sentinel -> NaN)
    p = ds.pose("chloroform", 1, 4)
    assert p["status"] == "partial_shieldings"
    assert np.isfinite(p["shielding_isolated"]).all()
    assert np.isnan(p["shielding_solvated"]).all()


def test_test_solute_type(ds):
    p = ds.pose("benzene", 9, 1)
    assert p["type"] == "test" and p["name"] == "MOL000009"


def test_bad_lookups(ds):
    with pytest.raises(IndexError):
        ds.pose("chloroform", 1, 0)        # frames are 1-based
    with pytest.raises(IndexError):
        ds.pose("chloroform", 1, 99)
    with pytest.raises(IndexError):
        ds.pose("chloroform", 999, 1)
    with pytest.raises(KeyError):
        ds.pose("acetone", 1, 1)


def test_context_manager_closes(mini_path):
    d = SigmaFresh(mini_path)
    d.close()
    assert not d.f.id.valid


# ---------- opt-in: real file, if present ----------
REAL = os.path.join(os.path.dirname(__file__), "sigma-fresh.hdf5")


@pytest.mark.skipif(not os.path.exists(REAL), reason="real sigma-fresh.hdf5 not present")
def test_real_file():
    with SigmaFresh(REAL) as ds:
        assert set(ds.solvents) == {"chloroform", "benzene", "methanol", "TIP4P"}
        for s in ds.solvents:
            assert ds.n_solutes(s) > 9000
        p = ds.pose("chloroform", 3, 1)
        assert p["coordinates"].shape[0] == p["atomic_numbers"].shape[0]
        assert p["shielding_isolated"].shape == (p["n_solute_atoms"],)
        # at least one complete frame exists for this solute, with finite shieldings
        s = ds.solute("chloroform", 3)
        complete = s["status_code"] == 3
        assert complete.any()
        assert np.isfinite(s["shielding_solvated"][complete]).any()
