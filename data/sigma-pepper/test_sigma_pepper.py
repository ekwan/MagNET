"""
Tests for the sigma-pepper reader (decode_sigma_pepper.SigmaPepper).

These build a tiny synthetic two-group dataset in the *exact* on-disk format (so the fixture
also serves as an executable spec of the layout) and exercise the decoder against it. No
dependency on the multi-hundred-MB real file. One opt-in test runs against the real
`sigma-pepper.hdf5` if it is present next to this file.

Run:  pytest test_sigma_pepper.py -q
Requires: pytest, numpy, h5py.
"""
import os
import numpy as np
import h5py
import pytest

from decode_sigma_pepper import SigmaPepper

SCALE = 1e-4


def _i32(x):
    return np.round(np.asarray(x, float) / SCALE).astype(np.int32)


def _write_group(f, name, *, n_atoms, znums, smiles, coords, shieldings, failed, attrs):
    """shieldings: dict name->per-atom array; failed: list of molecule indices."""
    g = f.create_group(name)
    g.attrs.update(attrs)
    g.attrs["n_molecules"] = len(n_atoms)
    g.attrs["n_atoms"] = int(np.sum(n_atoms))
    assert len(znums) == int(np.sum(n_atoms))
    g.create_dataset("atomic_numbers", data=np.asarray(znums, np.int8))
    g.create_dataset("n_atoms", data=np.asarray(n_atoms, np.uint8))
    d = g.create_dataset("coordinates", data=_i32(coords)); d.attrs["scale"] = SCALE; d.attrs["units"] = "angstrom"
    for sname, arr in shieldings.items():
        d = g.create_dataset(sname, data=_i32(arr)); d.attrs["scale"] = SCALE; d.attrs["units"] = "ppm"
    g.create_dataset("failed_indices", data=np.asarray(failed, np.int32))
    enc = [s.encode("utf-8") for s in smiles]
    off = np.concatenate([[0], np.cumsum([len(e) for e in enc])]).astype(np.int64)
    g.create_dataset("smiles_buffer", data=np.frombuffer(b"".join(enc), np.uint8))
    g.create_dataset("smiles_offsets", data=off)


# group A: 3 molecules (2,1,3 atoms = 6), one shielding, molecule 1 failed
A_NATOMS = [2, 1, 3]
A_Z = [6, 1,   8,   6, 7, 1]            # heteroatoms exercise the passthrough
A_SMILES = ["CC", "O", "CCC"]
A_COORDS = np.array([[0, 0, 0], [1.2345, 0, 0],
                     [5, 5, 5],
                     [3, 0, 0], [4.1, 0, 0], [3, 1.5, 0]], float)
A_PBE0 = np.array([180.1234, 31.0, 0.0, 150.0, 151.5, 29.9])   # mol 1 (atom idx 2) = sentinel 0
# group B: 2 molecules (3,2 atoms = 5), two shieldings, molecule 0 failed
B_NATOMS = [3, 2]
B_Z = [6, 8, 1,   6, 7]
B_SMILES = ["CCO", "CN"]
B_COORDS = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0],
                     [9, 9, 9], [9.5, 9, 9]], float)
B_WB97 = np.array([0.0, 0.0, 0.0, 110.0, 28.0])    # mol 0 failed (all zero)
B_WP04 = np.array([0.0, 0.0, 0.0, 98.0, 28.3])


@pytest.fixture(scope="module")
def mini_path(tmp_path_factory):
    p = str(tmp_path_factory.mktemp("sp") / "mini.hdf5")
    with h5py.File(p, "w") as f:
        _write_group(f, "pbe0_pcSseg1", n_atoms=A_NATOMS, znums=A_Z, smiles=A_SMILES, coords=A_COORDS,
                     shieldings={"shielding_pbe0": A_PBE0}, failed=[1],
                     attrs=dict(level_of_theory="PBE0/pcSseg-1, gas",
                                geometry="AIMNet2 stationary (fmax=0.01)", gdb_range="1-10 heavy"))
        _write_group(f, "wp04_wb97xd_pcSseg2", n_atoms=B_NATOMS, znums=B_Z, smiles=B_SMILES, coords=B_COORDS,
                     shieldings={"shielding_wb97xd": B_WB97, "shielding_wp04": B_WP04}, failed=[0],
                     attrs=dict(level_of_theory="WP04 + wB97X-D /pcSseg-2, gas",
                                geometry="AIMNet2 stationary (fmax=1e-4)", gdb_range="1-9 heavy"))
    return p


@pytest.fixture
def ds(mini_path):
    with SigmaPepper(mini_path) as d:
        yield d


def test_group_names(ds):
    assert set(ds.group_names) == {"pbe0_pcSseg1", "wp04_wb97xd_pcSseg2"}


def test_component_dimensions(ds):
    a = ds["pbe0_pcSseg1"]
    assert a.n_molecules == 3 and a.n_atoms == 6
    b = ds["wp04_wb97xd_pcSseg2"]
    assert b.n_molecules == 2 and b.n_atoms == 5
    assert b.shieldings == ["shielding_wb97xd", "shielding_wp04"]


def test_offsets_slice_each_molecule(ds):
    a = ds["pbe0_pcSseg1"]
    assert len(a.molecule(0)["atomic_numbers"]) == 2
    assert len(a.molecule(1)["atomic_numbers"]) == 1
    assert len(a.molecule(2)["atomic_numbers"]) == 3


def test_values_roundtrip_and_smiles(ds):
    m = ds["pbe0_pcSseg1"].molecule(2)
    assert m["smiles"] == "CCC"
    np.testing.assert_array_equal(m["atomic_numbers"], [6, 7, 1])   # heteroatom passthrough
    np.testing.assert_allclose(m["coordinates"], A_COORDS[3:6], atol=5e-5)
    np.testing.assert_allclose(m["shielding_pbe0"], A_PBE0[3:6], atol=5e-5)
    assert not m["failed"]


def test_two_shieldings_present(ds):
    m = ds["wp04_wb97xd_pcSseg2"].molecule(1)
    assert m["smiles"] == "CN"
    np.testing.assert_allclose(m["shielding_wp04"], B_WP04[3:5], atol=5e-5)
    np.testing.assert_allclose(m["shielding_wb97xd"], B_WB97[3:5], atol=5e-5)


def test_failed_flag(ds):
    assert ds["pbe0_pcSseg1"].molecule(1)["failed"] is True
    assert ds["pbe0_pcSseg1"].molecule(0)["failed"] is False
    b0 = ds["wp04_wb97xd_pcSseg2"].molecule(0)
    assert b0["failed"] is True
    # failed molecule's shieldings are sentinel zeros (the whole point of the flag)
    assert np.all(b0["shielding_wp04"] == 0) and np.all(b0["shielding_wb97xd"] == 0)
    # geometry/smiles still valid for a failed molecule
    assert b0["smiles"] == "CCO"
    np.testing.assert_allclose(b0["coordinates"], B_COORDS[0:3], atol=5e-5)


def test_empty_failed_indices(tmp_path):
    p = str(tmp_path / "nofail.hdf5")
    with h5py.File(p, "w") as f:
        _write_group(f, "pbe0_pcSseg1", n_atoms=[2], znums=[6, 1], smiles=["CO"],
                     coords=np.array([[0, 0, 0], [1.0, 0, 0]], float),
                     shieldings={"shielding_pbe0": np.array([150.0, 30.0])}, failed=[],
                     attrs=dict(level_of_theory="x", geometry="x", gdb_range="x"))
    with SigmaPepper(p) as d:
        assert d["pbe0_pcSseg1"].molecule(0)["failed"] is False


def test_smiles_accessor(ds):
    assert ds["pbe0_pcSseg1"].smiles(0) == "CC"
    assert ds["wp04_wb97xd_pcSseg2"].smiles(1) == "CN"


def test_index_out_of_range_raises(ds):
    with pytest.raises(IndexError):
        ds["pbe0_pcSseg1"].molecule(-1)
    with pytest.raises(IndexError):
        ds["pbe0_pcSseg1"].molecule(3)


def test_context_manager_closes(mini_path):
    d = SigmaPepper(mini_path)
    d.close()
    assert not d.f.id.valid


# ---------- opt-in: real file, if present ----------
REAL = os.path.join(os.path.dirname(__file__), "sigma-pepper.hdf5")


@pytest.mark.skipif(not os.path.exists(REAL), reason="real sigma-pepper.hdf5 not present")
def test_real_file():
    with SigmaPepper(REAL) as ds:
        a = ds["pbe0_pcSseg1"]
        b = ds["wp04_wb97xd_pcSseg2"]
        assert a.n_molecules == 2_167_210 and a.n_atoms == 45_418_890
        assert b.n_molecules == 320_105 and b.n_atoms == 6_113_101
        assert len(a.g["failed_indices"]) == 1
        assert len(b.g["failed_indices"]) == 472
        # column identity: WP04 must be lower than wB97XD on carbon
        Z = b.g["atomic_numbers"][:2_000_000]
        wp = b.g["shielding_wp04"][:2_000_000].astype(float) * SCALE
        wb = b.g["shielding_wb97xd"][:2_000_000].astype(float) * SCALE
        C = (Z == 6) & (wp != 0) & (wb != 0)
        assert wp[C].mean() < wb[C].mean()


@pytest.mark.skipif(not os.path.exists(REAL), reason="real sigma-pepper.hdf5 not present")
def test_real_file_reproduces_table_s8_site_counts():
    """SI Table S8's 'MagNET-Zero (initial round)' row is this file's pbe0_pcSseg1 group exactly.
    The 'MagNET-Zero (fine-tuning)' row is this group's wp04_wb97xd_pcSseg2 PLUS sigma-concentrate
    (the 50,000 reused molecules) -- see test_sigma_concentrate.py's matching test for that half."""
    with SigmaPepper(REAL) as ds:
        a = ds["pbe0_pcSseg1"]
        z_a = a.g["atomic_numbers"][:]
        assert int((z_a == 1).sum()) == 24_047_999
        assert int((z_a == 6).sum()) == 15_330_361

        b = ds["wp04_wb97xd_pcSseg2"]
        z_b = b.g["atomic_numbers"][:]
        # this group alone is not a Table S8 row (fine-tuning adds sigma-concentrate's atoms too),
        # but its counts are the fixed half of that sum, so lock them down here too.
        assert int((z_b == 1).sum()) == 3_287_698
        assert int((z_b == 6).sum()) == 2_028_642
