"""
decode_delta50.py - reader/decoder for the delta50 HDF5 dataset.

DELTA50 is a small benchmark of 50 common organic molecules (reference 47 in the paper). This
file holds MagNET's predicted NMR shieldings for each molecule on its AIMNet2 geometry, at three
model chemistries:

  nn_magnet_zero  MagNET-Zero (WP04 for 1H, wB97X-D for 13C; pcSseg-2; 20-pass), the gas-phase
                  triple-zeta-quality prediction
  nn_b3lyp        the gas-phase component of the MagNET-PCM correction (B3LYP/pcSseg-2)
  nn_b3lyp_pcm    the chloroform component of the MagNET-PCM correction (B3LYP/pcSseg-2 + PCM)

The MagNET-PCM implicit-solvent correction for a molecule is nn_b3lyp_pcm minus nn_b3lyp. The
paper uses this set to probe outliers (nitromethane, nitroethane, and t-butyl nitrate are the
notable ones).

Storage format:
  - Per-atom arrays are concatenated across molecules. Molecule i has `n_atoms[i]` atoms and owns
    the rows [start[i], start[i+1]) where start is the cumulative sum of `n_atoms`.
  - coordinates and shieldings are int32 fixed point: physical = stored * 1e-4, so reconstruction
    error is <= 5e-5 (Angstrom or ppm). A missing value would be the marker -2147483648 -> NaN.
  - molecules are identified by name (`molecule_names`).

Requires: Python >= 3.7, numpy, h5py. No other dependencies.
"""
from __future__ import annotations
import numpy as np
import h5py

_SCALE = 1e-4
_MARKER = -2147483648


def _decode(values: np.ndarray) -> np.ndarray:
    """int32 fixed-point -> float64 physical units; marker -> NaN. No-op on float input."""
    values = np.asarray(values)
    if not np.issubdtype(values.dtype, np.integer):
        return np.asarray(values, dtype=np.float64)
    out = values.astype(np.float64) * _SCALE
    out[values == _MARKER] = np.nan
    return out


class Delta50:
    """Reader for delta50.hdf5. Use as a context manager; index molecules by position or by name."""

    def __init__(self, path: str):
        self.f = h5py.File(path, "r")
        self.molecule_names = [str(n) for n in self.f["molecule_names"].asstr()[:]]
        self.n_molecules = int(self.f.attrs["n_molecules"])
        self._n_atoms = self.f["n_atoms"][:].astype(np.int64)
        self._start = np.empty(self.n_molecules + 1, np.int64)
        self._start[0] = 0
        np.cumsum(self._n_atoms, out=self._start[1:])
        self._name_to_index = {n: i for i, n in enumerate(self.molecule_names)}

    def _check(self, index: int):
        if not 0 <= index < self.n_molecules:
            raise IndexError(f"molecule index {index} out of range [0, {self.n_molecules})")

    def index_of(self, name: str) -> int:
        """Position of the molecule named `name`; raises KeyError if absent."""
        if name not in self._name_to_index:
            raise KeyError(f"molecule {name!r} not in delta50")
        return self._name_to_index[name]

    def molecule(self, index: int) -> dict:
        """Return one molecule's data (physical units) by position.

        Keys: name; atomic_numbers (n,); coordinates (n, 3 Angstrom); and the three predicted
        shieldings (nn_magnet_zero, nn_b3lyp, nn_b3lyp_pcm), each (n,) ppm.
        """
        self._check(index)
        sl = slice(int(self._start[index]), int(self._start[index + 1]))
        return {
            "name": self.molecule_names[index],
            "atomic_numbers": self.f["atomic_numbers"][sl],
            "coordinates": _decode(self.f["coordinates"][sl]),
            "nn_magnet_zero": _decode(self.f["nn_magnet_zero"][sl]),
            "nn_b3lyp": _decode(self.f["nn_b3lyp"][sl]),
            "nn_b3lyp_pcm": _decode(self.f["nn_b3lyp_pcm"][sl]),
        }

    def molecule_by_name(self, name: str) -> dict:
        """Return one molecule's data (see `molecule`) by name."""
        return self.molecule(self.index_of(name))

    def pcm_correction(self, index: int) -> np.ndarray:
        """The MagNET-PCM implicit-solvent correction (nn_b3lyp_pcm minus nn_b3lyp), (n,) ppm."""
        m = self.molecule(index)
        return m["nn_b3lyp_pcm"] - m["nn_b3lyp"]

    def close(self):
        """Close the underlying HDF5 file."""
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def __len__(self):
        return self.n_molecules

    def __repr__(self):
        return f"<Delta50: {self.n_molecules} molecules>"


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "delta50.hdf5"
    with Delta50(path) as ds:
        print(ds)
        m = ds.molecule(ds.index_of("nitromethane") if "nitromethane" in ds.molecule_names else 0)
        c = m["atomic_numbers"] == 6
        print(f"  {m['name']}: {len(m['atomic_numbers'])} atoms")
        print(f"    13C MagNET-Zero[:3] = {np.round(m['nn_magnet_zero'][c][:3], 3)} ppm")
