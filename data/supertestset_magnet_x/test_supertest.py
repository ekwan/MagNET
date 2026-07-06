"""Tests for data/supertestset_magnet_x/decode_supertest.py.

A tiny synthetic file in the stored int32 format exercises the reader, the two-level block
reconstruction (full system vs solute prefix), the isolated/solvated pairing, and the
explicit-solvent correction, without the real file. An opt-in smoke test runs against the real
file when present.
"""
import os
import sys

import numpy as np
import pytest
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import decode_supertest as D  # noqa: E402

SCALE = 1e-4
MARKER = -2147483648
REAL = os.path.join(HERE, "supertestset_magnet_x.hdf5")


def _encode(values):
    v = np.asarray(values, dtype=np.float64)
    out = np.round(v / SCALE).astype(np.int64)
    out[~np.isfinite(v)] = MARKER
    return out.astype(np.int32)


def _build_synthetic(path):
    """One molecule, one solvent, one conformer: an isolated record (3 solute atoms) and a
    solvated record (3 solute + 4 solvent = 7 full atoms)."""
    rng = np.random.default_rng(0)
    solute_an = np.array([6, 1, 8], dtype=np.uint8)
    iso_sh = np.array([100.0, 30.0, 250.0])
    solv_sh = iso_sh + np.array([-0.3, 0.1, -0.5])     # known correction
    full_an_solv = np.concatenate([solute_an, [1, 1, 1, 1]]).astype(np.uint8)
    records = [
        # name, solvent, conf, is_isolated, full_an, full_xyz, n_solute, solute_sh
        ("MOLX", "benzene", 0, 1, solute_an, rng.normal(0, 2, (3, 3)), 3, iso_sh),
        ("MOLX", "benzene", 0, 0, full_an_solv, rng.normal(0, 2, (7, 3)), 3, solv_sh),
    ]
    str_dt = h5py.string_dtype("utf-8")
    opts = dict(compression="gzip", shuffle=True)
    names, solvents, confs, iso, n_sol, n_full = [], [], [], [], [], []
    an_b, co_b, sh_b = [], [], []
    for name, solv, conf, is_iso, fan, fxyz, ns, ssh in records:
        names.append(name); solvents.append(solv); confs.append(conf); iso.append(is_iso)
        n_sol.append(ns); n_full.append(len(fan))
        an_b.append(fan); co_b.append(_encode(fxyz)); sh_b.append(_encode(ssh))
    with h5py.File(path, "w") as f:
        f.attrs["n_records"] = len(records)
        f.attrs["n_catalog"] = 1
        f.attrs["solvents"] = "benzene"
        f.attrs["scale"] = SCALE
        f.attrs["missing_value_marker"] = MARKER
        f.create_dataset("catalog_names", data=np.array(["MOLX"], dtype=object), dtype=str_dt)
        f.create_dataset("catalog_smiles", data=np.array(["CCO"], dtype=object), dtype=str_dt)
        f.create_dataset("record_names", data=np.array(names, dtype=object), dtype=str_dt)
        f.create_dataset("record_solvents", data=np.array(solvents, dtype=object), dtype=str_dt)
        f.create_dataset("conformer", data=np.asarray(confs, np.int32), **opts)
        f.create_dataset("is_isolated", data=np.asarray(iso, np.uint8), **opts)
        f.create_dataset("n_solute_atoms", data=np.asarray(n_sol, np.int32), **opts)
        f.create_dataset("n_full_atoms", data=np.asarray(n_full, np.int32), **opts)
        f.create_dataset("full_atomic_numbers", data=np.concatenate(an_b), **opts)
        f.create_dataset("full_coordinates", data=np.concatenate(co_b), **opts)
        f.create_dataset("solute_shieldings", data=np.concatenate(sh_b), **opts)


def test_record_reconstruction_and_solute_prefix(tmp_path):
    path = str(tmp_path / "tiny.hdf5")
    _build_synthetic(path)
    with D.SuperTestMagNETX(path) as ds:
        assert len(ds) == 2
        iso = ds.record(0)
        solv = ds.record(1)
        assert iso["is_isolated"] and not solv["is_isolated"]
        # isolated full system is just the solute
        assert len(iso["full_atomic_numbers"]) == 3
        # solvated full system is solute (first 3) + solvent (4)
        assert len(solv["full_atomic_numbers"]) == 7
        assert np.array_equal(solv["solute_atomic_numbers"], solv["full_atomic_numbers"][:3])
        assert solv["solute_shieldings"].shape == (3,)


def test_explicit_correction(tmp_path):
    path = str(tmp_path / "tiny.hdf5")
    _build_synthetic(path)
    with D.SuperTestMagNETX(path) as ds:
        corr = ds.explicit_correction("MOLX", "benzene", 0)
        assert np.allclose(corr, [-0.3, 0.1, -0.5], atol=1e-4)
        assert ds.catalog() == {"MOLX": "CCO"}
        with pytest.raises(KeyError):
            ds.find("MOLX", "water", 0, is_isolated=True)


def test_decode_marker_to_nan():
    out = D._decode(np.array([0, 10000, MARKER], dtype=np.int32))
    assert out[1] == pytest.approx(1.0) and np.isnan(out[2])


@pytest.mark.skipif(not os.path.exists(REAL), reason="real supertestset_magnet_x.hdf5 not present")
def test_real_file_smoke():
    with D.SuperTestMagNETX(REAL) as ds:
        assert len(ds) == 318
        assert len(ds.catalog()) == 50
        corr = ds.explicit_correction("MOL176459", "benzene", 0)
        assert corr.shape[0] == 39 and np.isfinite(corr).all()
