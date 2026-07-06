"""
Tests for the dft8k reader (dft8k_reader.py).

Most tests build a tiny synthetic dft8k.hdf5 in the exact on-disk format (so the fixture
doubles as a spec of what the reader expects) and need no large data. One opt-in test runs
against the real dft8k.hdf5 if it is present (skipped otherwise, e.g. in CI).

Run:  pytest test_dft8k.py -q
Requires: pytest, numpy, h5py.
"""
import os
import numpy as np
import h5py
import pytest

from dft8k_reader import DFT8k, _decode, _MARKER

SCALE = 1e4


def _enc(arr):
    """Encode floats the way the build script does: int32 = round(value * 1e4), NaN -> marker."""
    a = np.asarray(arr, dtype=np.float64)
    nan = np.isnan(a)
    out = np.rint(np.where(nan, 0.0, a) * SCALE).astype(np.int32)
    out[nan] = _MARKER
    return out


# ---------------------------------------------------------------------------
# fixed-point decoding
# ---------------------------------------------------------------------------
def test_decode_round_trip_and_marker():
    values = np.array([0.0, 207.07, -355.5, np.nan, 875.0], dtype=np.float64)
    decoded = _decode(_enc(values))
    assert np.isnan(decoded[3])
    np.testing.assert_allclose(decoded[~np.isnan(decoded)],
                               values[~np.isnan(values)], atol=5e-5)


def test_decode_passes_through_float():
    # a plain-decimal array must read back unchanged (decode is a no-op on floats)
    values = np.array([1.0, 2.5, -7.25], dtype=np.float64)
    np.testing.assert_array_equal(_decode(values), values)


def test_decode_handles_int64():
    # any integer dtype decodes the same way as int32
    vals = np.array([10000, -3550000, _MARKER], dtype=np.int64)
    out = _decode(vals)
    np.testing.assert_allclose(out[:2], [1.0, -355.0])
    assert np.isnan(out[2])


# ---------------------------------------------------------------------------
# synthetic dft8k.hdf5 in the real two-group layout
# ---------------------------------------------------------------------------
# two molecules: ids 100 and 200, with 2 and 3 atoms (atoms are H, C, ...)
A_IDS = [100, 200]
A_COUNTS = [2, 3]
A_Z = [1, 6, 6, 1, 7]          # mol100: H C ; mol200: C H N
# per-atom DFT shieldings (concatenated): pbe0, wp04, wb97xd
A_PBE0 = [30.0, 150.0, 151.0, 31.0, 200.0]
A_WP04 = [30.5, 150.5, 151.5, 31.5, 200.5]   # the 1H reference
A_WB97 = [30.7, 150.7, 151.7, 31.7, 200.7]   # the 13C reference
# MagNET predicts H with the WP04 model and C with the wB97X-D model; other atoms blank
A_NN_WP04 = [30.6, np.nan, np.nan, 31.6, np.nan]    # only at H atoms (index 0, 3)
A_NN_WB97 = [np.nan, 150.6, 151.6, np.nan, np.nan]  # only at C atoms (index 1, 2)


def _make_synthetic_dft8k(path):
    with h5py.File(path, "w") as f:
        g = f.create_group("aimnet2_wp04_wb97xd_pcsseg2")
        g.attrs["geometry"] = "AIMNet2"
        g.attrs["level_of_theory"] = "PBE0/pcSseg-1; WP04/pcSseg-2; wB97X-D/pcSseg-2"
        g.attrs["n_molecules"] = len(A_IDS)
        g.attrs["n_atoms"] = sum(A_COUNTS)
        g.attrs["scale"] = 1e-4
        g.create_dataset("molecule_ids", data=np.array(A_IDS, np.uint32))
        g.create_dataset("n_atoms", data=np.array(A_COUNTS, np.uint8))
        # one SMILES per molecule; mol100 has a real one, mol200 carries the literal "none"
        g.create_dataset("smiles", data=np.array(["CC", "none"], dtype=object),
                         dtype=h5py.string_dtype(encoding="utf-8"))
        g.create_dataset("atomic_numbers", data=np.array(A_Z, np.int8))
        coords = np.arange(sum(A_COUNTS) * 3, dtype=np.float64).reshape(-1, 3) * 0.1
        g.create_dataset("coordinates", data=_enc(coords))
        g.create_dataset("shielding_pbe0_pcSseg1", data=_enc(A_PBE0))
        g.create_dataset("shielding_wp04_pcSseg2", data=_enc(A_WP04))
        g.create_dataset("shielding_wb97xd_pcSseg2", data=_enc(A_WB97))
        g.create_dataset("nn_shielding_wp04_pcSseg2", data=_enc(A_NN_WP04))
        g.create_dataset("nn_shielding_wb97xd_pcSseg2", data=_enc(A_NN_WB97))

        # second group, SAME ids but reversed row order, three geometries per molecule
        b = f.create_group("b3lyp_pbe0_pcsseg1")
        b.attrs["geometry"] = "B3LYP/pcSseg-1"
        b.attrs["level_of_theory"] = "PBE0/pcSseg-1 on B3LYP/pcSseg-1 geometries"
        b.attrs["n_molecules"] = 2
        b.attrs["n_atoms"] = 5
        b.attrs["n_geometries"] = 3
        b.attrs["scale"] = 1e-4
        b.create_dataset("molecule_ids", data=np.array([200, 100], np.uint32))  # reversed
        b.create_dataset("n_atoms", data=np.array([3, 2], np.uint8))
        b.create_dataset("atomic_numbers", data=np.array([6, 1, 7, 1, 6], np.int8))
        bcoords = np.arange(3 * 5 * 3, dtype=np.float64).reshape(3, 5, 3) * 0.01
        b.create_dataset("coordinates", data=_enc(bcoords))
        bshield = np.arange(3 * 5, dtype=np.float64).reshape(3, 5) + 100.0
        b.create_dataset("shielding", data=_enc(bshield))
        b.create_dataset("weights", data=np.array([1, 1, 1, 0.5, 0.5], np.float32))
        # symmetry groups are 0-based atom indices, local to each molecule.
        # mol200 (first row, 3 atoms): two groups [[1, 2]] and [[0]];
        # mol100 (second row, 2 atoms): one group [[0, 1]].
        b.create_dataset("symmetric_atoms_n_groups", data=np.array([2, 1], np.int32))
        b.create_dataset("symmetric_atoms_group_lengths", data=np.array([2, 1, 2], np.int32))
        b.create_dataset("symmetric_atoms_members", data=np.array([1, 2, 0, 0, 1], np.int32))


def test_aimnet2_group(tmp_path):
    p = tmp_path / "dft8k.hdf5"
    _make_synthetic_dft8k(str(p))
    with DFT8k(str(p)) as ds:
        assert len(ds.aimnet2) == 2
        m = ds.aimnet2.molecule(0)
        assert m["id"] == 100
        assert m["smiles"] == "CC"
        assert ds.aimnet2.smiles(1) == "none"   # mol200 has no SMILES
        np.testing.assert_array_equal(m["atomic_numbers"], [1, 6])
        np.testing.assert_allclose(m["wp04_pcSseg2"], [30.5, 150.5], atol=5e-5)
        # H atom (index 0) has a WP04 prediction; C atom (index 1) does not
        assert not np.isnan(m["nn_wp04_pcSseg2"][0])
        assert np.isnan(m["nn_wp04_pcSseg2"][1])
        # C atom has a wB97X-D prediction; H atom does not
        assert np.isnan(m["nn_wb97xd_pcSseg2"][0])
        assert not np.isnan(m["nn_wb97xd_pcSseg2"][1])
        np.testing.assert_allclose(m["nn_wb97xd_pcSseg2"][1], 150.6, atol=5e-5)


def test_all_shieldings_flat(tmp_path):
    p = tmp_path / "dft8k.hdf5"
    _make_synthetic_dft8k(str(p))
    with DFT8k(str(p)) as ds:
        flat = ds.aimnet2.all_shieldings()
        # one entry per atom across both molecules, in stored order
        np.testing.assert_array_equal(flat["atomic_numbers"], [1, 6, 6, 1, 7])
        np.testing.assert_allclose(flat["wp04_pcSseg2"], A_WP04, atol=5e-5)
        # MagNET prediction is present only at the atoms its model covers
        assert np.isnan(flat["nn_wp04_pcSseg2"]).tolist() == [False, True, True, False, True]
        # the 1H residual (DFT minus MagNET) at the two H atoms is -0.1
        Z = flat["atomic_numbers"]
        h = (Z == 1) & np.isfinite(flat["nn_wp04_pcSseg2"])
        np.testing.assert_allclose(flat["wp04_pcSseg2"][h] - flat["nn_wp04_pcSseg2"][h],
                                   [-0.1, -0.1], atol=5e-5)


def test_b3lyp_group_and_symmetry(tmp_path):
    p = tmp_path / "dft8k.hdf5"
    _make_synthetic_dft8k(str(p))
    with DFT8k(str(p)) as ds:
        b = ds.b3lyp
        assert b.n_geometries == 3
        m = b.molecule(0)               # first row is id 200
        assert m["id"] == 200
        assert m["coordinates"].shape == (3, 3, 3)   # (geometries, atoms, xyz)
        assert m["shielding"].shape == (3, 3)
        assert m["symmetric_atoms"] == [[1, 2], [0]]
        m2 = b.molecule(1)              # id 100
        assert m2["symmetric_atoms"] == [[0, 1]]
        np.testing.assert_allclose(m2["weights"], [0.5, 0.5])
        # the symmetry indices are valid 0-based positions into this molecule's atoms
        for group in m["symmetric_atoms"] + m2["symmetric_atoms"]:
            for atom_index in group:
                assert 0 <= atom_index < len(m["atomic_numbers"]) or \
                       0 <= atom_index < len(m2["atomic_numbers"])


def test_match_across_groups_by_id(tmp_path):
    p = tmp_path / "dft8k.hdf5"
    _make_synthetic_dft8k(str(p))
    with DFT8k(str(p)) as ds:
        # rows are ordered differently in the two groups; matching by id must still work
        a = ds.aimnet2.molecule_by_id(200)
        b = ds.b3lyp.molecule_by_id(200)
        assert a["id"] == b["id"] == 200
        np.testing.assert_array_equal(a["atomic_numbers"], b["atomic_numbers"])


# ---------------------------------------------------------------------------
# opt-in test against the real file (skipped when absent, e.g. in CI)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REAL = os.path.join(_HERE, "dft8k.hdf5")


@pytest.mark.skipif(not os.path.exists(_REAL), reason="real dft8k.hdf5 not present")
def test_real_dft8k():
    with DFT8k(_REAL) as ds:
        assert len(ds.aimnet2) == 7111
        assert len(ds.b3lyp) == 7111
        assert ds.b3lyp.n_geometries == 3
        # the two groups carry the same id set
        assert set(int(x) for x in ds.aimnet2.molecule_ids) == \
               set(int(x) for x in ds.b3lyp.molecule_ids)
        m = ds.aimnet2.molecule(0)
        # SMILES: 6780 of the 7111 molecules have one; the rest carry the literal "none"
        n_real_smiles = sum(1 for i in range(len(ds.aimnet2)) if ds.aimnet2.smiles(i) != "none")
        assert n_real_smiles == 6780
        assert isinstance(m["smiles"], str) and m["smiles"] != "none"
        z = m["atomic_numbers"]
        h = z == 1
        c = z == 6
        other = ~(h | c)
        # the WP04 model predicts every hydrogen and nothing else; the wB97X-D model
        # predicts every carbon and nothing else; non-target atoms read back blank
        assert not np.isnan(m["nn_wp04_pcSseg2"][h]).any()
        assert np.isnan(m["nn_wp04_pcSseg2"][~h]).all()
        assert not np.isnan(m["nn_wb97xd_pcSseg2"][c]).any()
        assert np.isnan(m["nn_wb97xd_pcSseg2"][~c]).all()
        assert other.sum() == 0 or np.isnan(m["nn_wp04_pcSseg2"][other]).all()
        # the same molecule resolves in the other group by id
        b = ds.b3lyp.molecule_by_id(m["id"])
        np.testing.assert_array_equal(m["atomic_numbers"], b["atomic_numbers"])
        # every symmetry index is a valid 0-based position into this molecule's atoms
        for group in b["symmetric_atoms"]:
            for atom_index in group:
                assert 0 <= atom_index < len(b["atomic_numbers"])
