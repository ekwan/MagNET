"""Tests for data/delta50/decode_delta50.py.

A tiny synthetic file in the stored int32 format exercises the reader, the per-molecule slicing,
the fixed-point decode, and the PCM-correction helper, without the real file. An opt-in smoke test
runs against the real file when present.
"""
import os
import sys

import numpy as np
import pytest
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import decode_delta50 as D  # noqa: E402

SCALE = 1e-4
MARKER = -2147483648
REAL = os.path.join(HERE, "delta50.hdf5")


def _encode(values):
    v = np.asarray(values, dtype=np.float64)
    out = np.round(v / SCALE).astype(np.int64)
    out[~np.isfinite(v)] = MARKER
    return out.astype(np.int32)


def _build_synthetic(path):
    """Two molecules (3 and 5 atoms) in the stored int32 format, plus the decoded truth to check against."""
    rng = np.random.default_rng(0)
    names = ["alpha", "beta"]
    specs = [3, 5]  # n_atoms
    n_atoms = np.array(specs, dtype=np.int32)
    an, coords, zero, b3, pcm, truth = [], [], [], [], [], []
    for na in specs:
        a = rng.integers(1, 9, size=na).astype(np.uint8)
        c = rng.normal(0, 2, size=(na, 3))
        z = rng.normal(100, 30, size=na)
        g = rng.normal(100, 30, size=na)
        p = g + rng.normal(0, 0.2, size=na)
        an.append(a); coords.append(_encode(c)); zero.append(_encode(z))
        b3.append(_encode(g)); pcm.append(_encode(p))
        truth.append((a, c, z, g, p))
    opts = dict(compression="gzip", shuffle=True)
    with h5py.File(path, "w") as f:
        f.attrs["n_molecules"] = 2
        f.attrs["scale"] = SCALE
        f.attrs["missing_value_marker"] = MARKER
        f.create_dataset("molecule_names", data=np.array(names, dtype=object),
                         dtype=h5py.string_dtype("utf-8"))
        f.create_dataset("n_atoms", data=n_atoms, **opts)
        f.create_dataset("atomic_numbers", data=np.concatenate(an), **opts)
        f.create_dataset("coordinates", data=np.concatenate(coords), **opts)
        f.create_dataset("nn_magnet_zero", data=np.concatenate(zero), **opts)
        f.create_dataset("nn_b3lyp", data=np.concatenate(b3), **opts)
        f.create_dataset("nn_b3lyp_pcm", data=np.concatenate(pcm), **opts)
    return names, specs, truth


def test_reader_and_decode(tmp_path):
    path = str(tmp_path / "tiny.hdf5")
    names, specs, truth = _build_synthetic(path)
    with D.Delta50(path) as ds:
        assert len(ds) == 2
        assert ds.molecule_names == names
        for i, na in enumerate(specs):
            m = ds.molecule(i)
            a, c, z, g, p = truth[i]
            assert m["name"] == names[i]
            assert np.array_equal(m["atomic_numbers"], a)
            assert m["coordinates"].shape == (na, 3)
            assert np.abs(m["coordinates"] - c).max() <= 5e-5
            assert np.abs(m["nn_magnet_zero"] - z).max() <= 5e-5
            assert np.allclose(ds.pcm_correction(i), p - g, atol=1e-4)


def test_lookup_and_errors(tmp_path):
    path = str(tmp_path / "tiny.hdf5")
    names, _, _ = _build_synthetic(path)
    with D.Delta50(path) as ds:
        assert ds.molecule_by_name("beta")["name"] == "beta"
        with pytest.raises(KeyError):
            ds.index_of("missing")
        with pytest.raises(IndexError):
            ds.molecule(9)


def test_decode_marker_to_nan():
    out = D._decode(np.array([0, 10000, MARKER], dtype=np.int32))
    assert out[1] == pytest.approx(1.0) and np.isnan(out[2])


@pytest.mark.skipif(not os.path.exists(REAL), reason="real delta50.hdf5 not present")
def test_real_file_smoke():
    with D.Delta50(REAL) as ds:
        assert len(ds) == 50
        assert "nitromethane" in ds.molecule_names
        m = ds.molecule_by_name("nitromethane")
        assert np.isfinite(m["nn_magnet_zero"]).all()
        assert ds.pcm_correction(ds.index_of("nitromethane")).shape == m["atomic_numbers"].shape
