"""
decode_sigma_pepper.py - reader/decoder for the sigma-pepper HDF5 dataset.

sigma-pepper holds computed gas-phase NMR shieldings on AIMNet2 stationary geometries for
GDB-derived molecules; it is the data behind the MagNET-Zero model. The file has TWO groups,
one per level of theory:

  /pbe0_pcSseg1          GDB 1-10 heavy atoms, PBE0/pcSseg-1, one shielding per atom
  /wp04_wb97xd_pcSseg2   GDB 1-9  heavy atoms, two shieldings per atom; the `shieldings` list is
                         sorted alphabetically, so the order is shielding_wb97xd (13C reference)
                         then shielding_wp04 (1H reference)

Storage format:
  - Per-atom arrays are concatenated across all molecules in a group. Each molecule's atom
    count is in `n_atoms`; the row range for molecule i is [start[i], start[i+1]) where
    start = cumulative sum of n_atoms (reconstructed once, on open).
  - coordinates and shieldings are int32 fixed-point: physical = stored * scale (scale = 1e-4),
    so reconstruction error is <= 5e-5 (Angstrom or ppm). Coordinates are mean-centered per
    molecule (centroid at the origin), a translation only; interatomic distances are unchanged.
  - SMILES are stored as a single concatenated UTF-8 buffer (`smiles_buffer`) plus
    `smiles_offsets`; molecule i's SMILES is buffer[offsets[i]:offsets[i+1]].
  - The molecule's array index IS its identifier (row order is fixed to the source).
  - `failed_indices` lists molecules whose shielding calculation failed: their shieldings are
    sentinel zeros and must be ignored. Geometry and SMILES remain valid.

Requires: Python >= 3.7, numpy, h5py. No other dependencies.
"""
from __future__ import annotations
import numpy as np
import h5py


class Component:
    """One group of sigma-pepper (a single level of theory)."""

    def __init__(self, grp: h5py.Group):
        self.g = grp
        self.name = grp.name.lstrip("/")
        self.n_molecules = int(grp.attrs["n_molecules"])
        self.n_atoms = int(grp.attrs["n_atoms"])
        self.level_of_theory = str(grp.attrs["level_of_theory"])
        self.geometry = str(grp.attrs["geometry"])
        self.gdb_range = str(grp.attrs["gdb_range"])
        # atom row offsets: start[i]..start[i+1] for molecule i (one cumsum, cached)
        na = grp["n_atoms"][:].astype(np.int64)
        self._start = np.empty(self.n_molecules + 1, np.int64)
        self._start[0] = 0
        np.cumsum(na, out=self._start[1:])
        # smiles buffer (small) + offsets, read once
        self._sbuf = grp["smiles_buffer"][:].tobytes()
        self._soff = grp["smiles_offsets"][:]
        self._failed = set(int(x) for x in grp["failed_indices"][:])
        self.shieldings = sorted(k for k in grp.keys() if k.startswith("shielding_"))

    def _check(self, index: int):
        if not 0 <= index < self.n_molecules:
            raise IndexError(f"molecule index {index} out of range [0, {self.n_molecules})")

    def smiles(self, index: int) -> str:
        """Return molecule `index`'s SMILES string."""
        self._check(index)
        return self._sbuf[self._soff[index]:self._soff[index + 1]].decode("utf-8")

    def _scaled(self, name: str, sl: slice) -> np.ndarray:
        d = self.g[name]
        return d[sl].astype(np.float64) * float(d.attrs["scale"])

    def molecule(self, index: int) -> dict:
        """Return one molecule's decoded data (physical units) by array index.

        Keys: index, smiles, atomic_numbers, coordinates (n,3 Angstrom), one entry per
        shielding dataset (n, ppm), and `failed`. If `failed` is True the shieldings are
        sentinel zeros and should be ignored; geometry and SMILES are still valid.
        """
        self._check(index)
        a, b = int(self._start[index]), int(self._start[index + 1])
        sl = slice(a, b)
        out = {
            "index": index,
            "smiles": self.smiles(index),
            "atomic_numbers": self.g["atomic_numbers"][sl],
            "coordinates": self._scaled("coordinates", sl),   # (n,3) Angstrom
            "failed": index in self._failed,
        }
        for s in self.shieldings:
            out[s] = self._scaled(s, sl)                       # (n,) ppm
        return out

    def __len__(self):
        return self.n_molecules

    def __repr__(self):
        return (f"<Component /{self.name}: {self.n_molecules:,} molecules, "
                f"{self.n_atoms:,} atoms, {self.level_of_theory}>")


class SigmaPepper:
    """Reader for a sigma-pepper HDF5 file; exposes each level-of-theory group as a Component."""

    def __init__(self, path: str):
        self.f = h5py.File(path, "r")
        self.components = {name: Component(self.f[name]) for name in self.f.keys()
                          if isinstance(self.f[name], h5py.Group)}

    @property
    def group_names(self):
        """List of level-of-theory group names in the file."""
        return list(self.components)

    def __getitem__(self, name: str) -> Component:
        return self.components[name]

    def close(self):
        """Close the underlying HDF5 file."""
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "sigma-pepper.hdf5"
    with SigmaPepper(path) as ds:
        for name in ds.group_names:
            c = ds[name]
            print(c)
            m = c.molecule(0)
            extra = "  ".join(f"{s}={np.round(m[s][:3], 4)}" for s in c.shieldings)
            print(f"  mol 0: {m['smiles']}  Z={list(m['atomic_numbers'][:6])}  failed={m['failed']}")
            print(f"         {extra}")
