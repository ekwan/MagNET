"""Tests for data/gdb_qcd/decode_gdb_qcd.py.

A tiny synthetic file in the stored int32 format exercises the reader, the per-molecule block
reconstruction (variable atoms and trajectories, fixed 33 frames), and the fixed-point decode,
all without the multi-hundred-MB real file. An opt-in smoke test runs against the real file when
it is present.
"""
import os
import sys

import numpy as np
import pytest
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import decode_gdb_qcd as D  # noqa: E402

SCALE = 1e-4
MARKER = -2147483648
N_FRAMES = 33
REAL = os.path.join(HERE, "gdb_qcd.hdf5")


def _encode(values):
    out = np.round(np.asarray(values, dtype=np.float64) / SCALE).astype(np.int64)
    out[~np.isfinite(values)] = MARKER
    return out.astype(np.int32)


def _build_synthetic(path):
    """Two molecules: 3 atoms / 2 trajectories, and 4 atoms / 1 trajectory."""
    rng = np.random.default_rng(0)
    specs = [(3, 2), (4, 1)]  # (n_atoms, n_trajectories)
    mol_ids = np.array([11, 22], dtype=np.int64)
    n_atoms = np.array([s[0] for s in specs], dtype=np.int32)
    n_traj = np.array([s[1] for s in specs], dtype=np.int32)
    atomic, coord_blocks, shield_blocks, truth = [], [], [], []
    for na, nt in specs:
        an = rng.integers(1, 9, size=na).astype(np.uint8)
        coords = rng.normal(0, 2, size=(nt, N_FRAMES, na, 3))
        shields = rng.normal(100, 30, size=(nt, N_FRAMES, na))
        atomic.append(an)
        coord_blocks.append(_encode(coords.reshape(-1, 3)))
        shield_blocks.append(_encode(shields.reshape(-1)))
        truth.append((an, coords, shields))
    opts = dict(compression="gzip", shuffle=True)
    with h5py.File(path, "w") as f:
        f.attrs["n_molecules"] = 2
        f.attrs["n_trajectories"] = int(n_traj.sum())
        f.attrs["n_frames"] = N_FRAMES
        f.attrs["nmr_level_of_theory"] = "PBE0/pcSseg-1"
        f.attrs["scale"] = SCALE
        f.attrs["missing_value_marker"] = MARKER
        f.create_dataset("molecule_ids", data=mol_ids, **opts)
        f.create_dataset("n_atoms", data=n_atoms, **opts)
        f.create_dataset("n_trajectories", data=n_traj, **opts)
        f.create_dataset("atomic_numbers", data=np.concatenate(atomic), **opts)
        f.create_dataset("coordinates", data=np.concatenate(coord_blocks), **opts)
        f.create_dataset("shieldings", data=np.concatenate(shield_blocks), **opts)
    return mol_ids, specs, truth


def test_reader_reconstructs_blocks(tmp_path):
    path = str(tmp_path / "tiny.hdf5")
    mol_ids, specs, truth = _build_synthetic(path)
    with D.GDBQCD(path) as ds:
        assert len(ds) == 2
        assert ds.n_frames == 33
        assert list(ds.molecule_ids) == list(mol_ids)
        for i, (na, nt) in enumerate(specs):
            m = ds.molecule(i)
            an, coords, shields = truth[i]
            assert m["id"] == int(mol_ids[i])
            assert ds.n_trajectories(i) == nt
            assert np.array_equal(m["atomic_numbers"], an)
            assert m["coordinates"].shape == (nt, 33, na, 3)
            assert m["shieldings"].shape == (nt, 33, na)
            # fixed-point round-trip within the 5e-5 floor
            assert np.abs(m["coordinates"] - coords).max() <= 5e-5
            assert np.abs(m["shieldings"] - shields).max() <= 5e-5


def test_lookup_by_id_and_errors(tmp_path):
    path = str(tmp_path / "tiny.hdf5")
    mol_ids, _, _ = _build_synthetic(path)
    with D.GDBQCD(path) as ds:
        assert ds.molecule_by_id(22)["id"] == 22
        assert ds.index_of(11) == 0
        with pytest.raises(KeyError):
            ds.index_of(999)
        with pytest.raises(IndexError):
            ds.molecule(5)


def test_decode_marker_to_nan():
    vals = np.array([0, 10000, MARKER, -25000], dtype=np.int32)
    out = D._decode(vals)
    assert out[0] == 0.0 and out[1] == pytest.approx(1.0) and out[3] == pytest.approx(-2.5)
    assert np.isnan(out[2])


@pytest.mark.skipif(not os.path.exists(REAL), reason="real gdb_qcd.hdf5 not present")
def test_real_file_smoke():
    with D.GDBQCD(REAL) as ds:
        assert len(ds) == 2461
        assert int(ds.f.attrs["n_trajectories"]) == 56702
        m = ds.molecule(0)
        assert m["coordinates"].shape[1] == 33
        assert m["coordinates"].shape[0] == ds.n_trajectories(0)
        assert np.isfinite(m["shieldings"]).all()
