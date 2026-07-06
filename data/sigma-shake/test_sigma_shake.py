"""
Tests for the sigma-shake reader (decode_sigma_shake.SigmaShake).

These build a tiny synthetic dataset in the *exact* on-disk format (so the fixture also
serves as an executable spec of the layout) and exercise the decoder against it. No
dependency on the multi-GB real file. One opt-in test runs against the real
`sigma-shake.hdf5` if it is present next to this file.

Run:  pytest test_sigma_shake.py -q
Requires: pytest, numpy, h5py.
"""
import os
import numpy as np
import h5py
import pytest

from decode_sigma_shake import SigmaShake

SCALE = 1e-4

# ---- known synthetic content: 3 molecules, 9 atoms; molecule 1 is incomplete ----
MOL_ID = np.array([100, 200, 300], np.int32)
ATOM_START = np.array([0, 3, 5], np.int32)
ATOM_END = np.array([3, 5, 9], np.int32)
Z = np.array([6, 1, 1,   8, 1,   6, 6, 1, 1], np.int8)
STAT_XYZ = np.array([
    [0.0000, 0.0000, 0.0000], [1.0000, 0.0000, 0.0000], [0.0000, 1.0000, 0.0000],   # mol 100
    [2.0000, 2.0000, 2.0000], [2.5000, 2.0000, 2.0000],                             # mol 200 (incomplete)
    [3.0000, 0.0000, 0.0000], [4.0000, 0.0000, 0.0000], [3.0000, 1.0000, 0.0000], [4.0000, 1.0000, 0.0000],
], np.float64)
STAT_SIG = np.array([150.1234, 30.5678, 31.0000,   200.0000, 25.0000,   140.0000, 141.0000, 29.0000, 29.5000])
PERT_XYZ = np.array([
    [0.0100, 0.0000, 0.0000], [-0.0050, 0.0000, 0.0000], [0.0000, 0.0200, 0.0000],  # mol 100
    [0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000],                             # mol 200 (zeroed)
    [0.0030, 0.0000, 0.0000], [-0.0030, 0.0000, 0.0000], [0.0000, 0.0050, 0.0000], [0.0000, -0.0050, 0.0000],
], np.float64)
PERT_SIG = np.array([151.0000, 30.9999, 31.5000,   0.0, 0.0,   140.5000, 141.5000, 29.2000, 29.7000])
INCOMPLETE_MASK = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0], bool)
SPLIT = {"train": np.array([100, 200], np.int32), "val": np.array([300], np.int32),
         "test": np.array([], np.int32)}
REMOVE = np.array([200], np.int32)
INCOMPLETE_IDS = np.array([200], np.int32)


def _i32(x):
    return np.round(np.asarray(x) / SCALE).astype(np.int32)


@pytest.fixture(scope="module")
def mini_path(tmp_path_factory):
    p = str(tmp_path_factory.mktemp("ss") / "mini.hdf5")
    with h5py.File(p, "w") as f:
        f.attrs["n_atoms"] = len(Z)
        f.attrs["n_molecules"] = len(MOL_ID)
        for name, arr in [("stationary_coords", STAT_XYZ), ("perturbation", PERT_XYZ),
                          ("shielding_stationary", STAT_SIG), ("shielding_perturbed", PERT_SIG)]:
            d = f.create_dataset(name, data=_i32(arr))
            d.attrs["scale"] = SCALE
        f.create_dataset("atomic_numbers", data=Z)
        f.create_dataset("incomplete_mask", data=INCOMPLETE_MASK)
        f.create_dataset("molecule_id", data=MOL_ID)
        f.create_dataset("atom_start", data=ATOM_START)
        f.create_dataset("atom_end", data=ATOM_END)
        f.create_dataset("remove_molecule_ids", data=REMOVE)
        f.create_dataset("incomplete_molecule_ids", data=INCOMPLETE_IDS)
        for k, v in SPLIT.items():
            f.create_dataset(f"split_{k}_ids", data=v)
    return p


@pytest.fixture
def ds(mini_path):
    with SigmaShake(mini_path) as d:
        yield d


def test_dimensions(ds):
    assert ds.n_molecules == 3
    assert ds.n_atoms == 9


def test_complete_molecule_values(ds):
    m = ds.molecule(0)
    assert m["molecule_id"] == 100
    assert not m["incomplete"]
    np.testing.assert_array_equal(m["atomic_numbers"], [6, 1, 1])
    np.testing.assert_allclose(m["stationary_coords"], STAT_XYZ[0:3], atol=5e-5)
    np.testing.assert_allclose(m["shielding_stationary"], STAT_SIG[0:3], atol=5e-5)
    np.testing.assert_allclose(m["shielding_perturbed"], PERT_SIG[0:3], atol=5e-5)


def test_perturbed_geometry_reconstruction(ds):
    m = ds.molecule(0)
    np.testing.assert_allclose(m["perturbed_coords"], STAT_XYZ[0:3] + PERT_XYZ[0:3], atol=1e-4)


def test_incomplete_molecule_omits_perturbed_keys(ds):
    m = ds.molecule(1)
    assert m["molecule_id"] == 200
    assert m["incomplete"] is True
    for k in ("perturbation", "perturbed_coords", "shielding_perturbed"):
        assert k not in m
    # stationary data still valid
    np.testing.assert_allclose(m["stationary_coords"], STAT_XYZ[3:5], atol=5e-5)
    np.testing.assert_allclose(m["shielding_stationary"], STAT_SIG[3:5], atol=5e-5)


def test_scale_roundtrip_is_within_tolerance(ds):
    # every stored value should reconstruct to <= half the LSB (5e-5)
    m = ds.molecule(2)
    assert np.abs(m["stationary_coords"] - STAT_XYZ[5:9]).max() <= 5e-5
    assert np.abs(m["shielding_stationary"] - STAT_SIG[5:9]).max() <= 5e-5


def test_by_id(ds):
    assert ds.by_id(100)["molecule_id"] == 100
    assert ds.by_id(300)["molecule_id"] == 300


def test_by_id_unknown_raises(ds):
    with pytest.raises(KeyError):
        ds.by_id(999999)


def test_index_out_of_range_raises(ds):
    with pytest.raises(IndexError):
        ds.molecule(-1)
    with pytest.raises(IndexError):
        ds.molecule(3)


def test_split_effective_filters_remove_ids(ds):
    # raw train has 100 and 200; effective drops 200 (in remove_molecule_ids)
    np.testing.assert_array_equal(np.sort(ds.split("train", effective=False)), [100, 200])
    np.testing.assert_array_equal(ds.split("train"), [100])
    np.testing.assert_array_equal(ds.split("val"), [300])
    assert len(ds.split("test")) == 0


def test_context_manager_closes(mini_path):
    d = SigmaShake(mini_path)
    d.close()
    assert not d.f.id.valid


# ---------- opt-in: real file, if present ----------
REAL = os.path.join(os.path.dirname(__file__), "sigma-shake.hdf5")


@pytest.mark.skipif(not os.path.exists(REAL), reason="real sigma-shake.hdf5 not present")
def test_real_file_matches_si():
    with SigmaShake(REAL) as ds:
        assert ds.n_molecules == 4_787_816
        assert ds.n_atoms == 135_219_803
        assert len(ds.split("train")) == 3_730_572
        assert len(ds.split("val")) == 329_154
        assert len(ds.split("test")) == 329_163
        m = ds.molecule(0)
        assert m["molecule_id"] == 24553
        # spot round-trip: perturbed = stationary + perturbation, bond-length-scale accuracy
        assert m["perturbed_coords"].shape == m["stationary_coords"].shape
