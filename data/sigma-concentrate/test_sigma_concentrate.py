"""
Tests for the sigma-concentrate reader (decode_sigma_concentrate.SigmaConcentrate).

These build a tiny synthetic dataset in the *exact* on-disk format (so the fixture also serves as
an executable spec of the layout) and exercise the decoder against it. No dependency on the real
file. One opt-in test runs against the real `sigma-concentrate.hdf5` if it is present next to this
file.

Run:  pytest test_sigma_concentrate.py -q
Requires: pytest, numpy, h5py.
"""
import os
import numpy as np
import h5py
import pytest

from decode_sigma_concentrate import SigmaConcentrate

SCALE = 1e-4


def _i32(x):
    return np.round(np.asarray(x, float) / SCALE).astype(np.int32)


def _write(f, *, n_atoms, znums, sids, coords, gas, pcm, failed):
    assert len(znums) == int(np.sum(n_atoms))
    f.attrs["dataset"] = "sigma-concentrate"
    f.attrs["level_of_theory"] = "B3LYP/pcSseg-2"
    f.attrs["geometry"] = "AIMNet2 stationary"
    f.attrs["solvent"] = "gas + PCM chloroform"
    f.attrs["n_molecules"] = len(n_atoms)
    f.attrs["n_atoms"] = int(np.sum(n_atoms))
    f.create_dataset("atomic_numbers", data=np.asarray(znums, np.int8))
    f.create_dataset("n_atoms", data=np.asarray(n_atoms, np.uint8))
    f.create_dataset("sigma_shake_id", data=np.asarray(sids, np.int32))
    d = f.create_dataset("coordinates", data=_i32(coords)); d.attrs["scale"] = SCALE; d.attrs["units"] = "angstrom"
    d = f.create_dataset("shielding_b3lyp_gas", data=_i32(gas)); d.attrs["scale"] = SCALE; d.attrs["units"] = "ppm"
    d = f.create_dataset("shielding_b3lyp_pcm", data=_i32(pcm)); d.attrs["scale"] = SCALE; d.attrs["units"] = "ppm"
    f.create_dataset("failed_indices", data=np.asarray(failed, np.int32))


# 3 molecules (2,1,3 atoms = 6); heteroatoms exercise the passthrough; molecule 1 failed
NATOMS = [2, 1, 3]
Z = [6, 1,   8,   6, 7, 1]
SIDS = [1000265, 42, 999808]
COORDS = np.array([[0, 0, 0], [1.2345, 0, 0],
                   [5, 5, 5],
                   [3, 0, 0], [4.1, 0, 0], [3, 1.5, 0]], float)
GAS = np.array([180.1234, 31.0,   0.0,   150.0, 151.5, 29.9])   # mol 1 (atom 2) = sentinel 0
PCM = np.array([180.4567, 30.8,   0.0,   149.6, 151.2, 29.7])


@pytest.fixture(scope="module")
def mini_path(tmp_path_factory):
    p = str(tmp_path_factory.mktemp("sc") / "mini.hdf5")
    with h5py.File(p, "w") as f:
        _write(f, n_atoms=NATOMS, znums=Z, sids=SIDS, coords=COORDS, gas=GAS, pcm=PCM, failed=[1])
    return p


@pytest.fixture
def ds(mini_path):
    with SigmaConcentrate(mini_path) as d:
        yield d


def test_dimensions(ds):
    assert ds.n_molecules == 3 and ds.n_atoms_total == 6
    assert ds.shieldings == ["shielding_b3lyp_gas", "shielding_b3lyp_pcm"]
    assert len(ds) == 3


def test_offsets_slice_each_molecule(ds):
    assert len(ds.molecule(0)["atomic_numbers"]) == 2
    assert len(ds.molecule(1)["atomic_numbers"]) == 1
    assert len(ds.molecule(2)["atomic_numbers"]) == 3


def test_values_roundtrip_and_id(ds):
    m = ds.molecule(2)
    assert m["sigma_shake_id"] == 999808
    np.testing.assert_array_equal(m["atomic_numbers"], [6, 7, 1])   # heteroatom passthrough
    np.testing.assert_allclose(m["coordinates"], COORDS[3:6], atol=5e-5)
    np.testing.assert_allclose(m["shielding_b3lyp_gas"], GAS[3:6], atol=5e-5)
    np.testing.assert_allclose(m["shielding_b3lyp_pcm"], PCM[3:6], atol=5e-5)
    assert not m["failed"]


def test_both_shieldings_distinct(ds):
    m = ds.molecule(0)
    # gas and pcm differ atom-by-atom (the solvent correction is the signal)
    assert not np.allclose(m["shielding_b3lyp_gas"], m["shielding_b3lyp_pcm"])


def test_sigma_shake_id_accessor(ds):
    assert ds.sigma_shake_id(0) == 1000265
    assert ds.sigma_shake_id(1) == 42


def test_failed_flag(ds):
    assert ds.molecule(1)["failed"] is True
    assert ds.molecule(0)["failed"] is False
    f = ds.molecule(1)
    # failed molecule's shieldings are sentinel zeros; geometry/id still valid
    assert np.all(f["shielding_b3lyp_gas"] == 0) and np.all(f["shielding_b3lyp_pcm"] == 0)
    assert f["sigma_shake_id"] == 42
    np.testing.assert_allclose(f["coordinates"], COORDS[2:3], atol=5e-5)


def test_empty_failed_indices(tmp_path):
    p = str(tmp_path / "nofail.hdf5")
    with h5py.File(p, "w") as f:
        _write(f, n_atoms=[2], znums=[6, 1], sids=[7],
               coords=np.array([[0, 0, 0], [1.0, 0, 0]], float),
               gas=np.array([150.0, 30.0]), pcm=np.array([149.5, 29.8]), failed=[])
    with SigmaConcentrate(p) as d:
        assert d.molecule(0)["failed"] is False


def test_index_out_of_range_raises(ds):
    with pytest.raises(IndexError):
        ds.molecule(-1)
    with pytest.raises(IndexError):
        ds.molecule(3)


def test_context_manager_closes(mini_path):
    d = SigmaConcentrate(mini_path)
    d.close()
    assert not d.f.id.valid


# ---------- opt-in: real file, if present ----------
REAL = os.path.join(os.path.dirname(__file__), "sigma-concentrate.hdf5")


@pytest.mark.skipif(not os.path.exists(REAL), reason="real sigma-concentrate.hdf5 not present")
def test_real_file():
    with SigmaConcentrate(REAL) as ds:
        assert ds.n_molecules == 50_000 and ds.n_atoms_total == 1_413_134
        assert len(ds.f["failed_indices"]) == 0
        # gas and pcm are distinct columns; PCM differs from gas on carbon
        Z = ds.f["atomic_numbers"][:500_000]
        gas = ds.f["shielding_b3lyp_gas"][:500_000].astype(float) * SCALE
        pcm = ds.f["shielding_b3lyp_pcm"][:500_000].astype(float) * SCALE
        C = Z == 6
        assert gas[C].mean() != pcm[C].mean()
        # sigma_shake_id is populated and plausibly a sigma-shake id (positive)
        assert ds.sigma_shake_id(0) > 0


@pytest.mark.skipif(not os.path.exists(REAL), reason="real sigma-concentrate.hdf5 not present")
def test_real_file_reproduces_table_s8_site_counts():
    """SI Table S8's 'sigma-concentrate' row is this file exactly, and (together with
    sigma-pepper's wp04_wb97xd_pcSseg2 group -- see test_sigma_pepper.py's matching test) these
    50,000 reused molecules make up the other half of the 'MagNET-Zero (fine-tuning)' row:
    3,287,698 + 754,437 = 4,042,135 H sites; 2,028,642 + 528,569 = 2,557,211 C sites."""
    with SigmaConcentrate(REAL) as ds:
        z = ds.f["atomic_numbers"][:]
        assert int((z == 1).sum()) == 754_437
        assert int((z == 6).sum()) == 528_569
