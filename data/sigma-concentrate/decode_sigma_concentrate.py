"""
decode_sigma_concentrate.py - reader/decoder for the sigma-concentrate HDF5 dataset.

sigma-concentrate holds B3LYP/pcSseg-2 NMR shieldings computed both in the gas phase and with
an implicit chloroform solvent (PCM), on AIMNet2 stationary geometries for 50,000 molecules
sampled from sigma-shake. It is the data behind the MagNET-PCM implicit-solvent model, which
learns the solvent correction shielding_b3lyp_pcm - shielding_b3lyp_gas.

The file is flat (a single molecule set, one geometry per molecule):

  shielding_b3lyp_gas    isotropic sigma, B3LYP/pcSseg-2, gas phase
  shielding_b3lyp_pcm    isotropic sigma, B3LYP/pcSseg-2, PCM, solvent = chloroform

Storage format:
  - Per-atom arrays are concatenated across all molecules. Each molecule's atom count is in
    `n_atoms`; the row range for molecule i is [start[i], start[i+1]) where start = cumulative
    sum of n_atoms (reconstructed once, on open).
  - coordinates and shieldings are int32 fixed-point: physical = stored * scale (scale = 1e-4),
    so reconstruction error is <= 5e-5 (Angstrom or ppm). Coordinates are mean-centered per
    molecule (centroid at the origin), a translation only; interatomic distances are unchanged.
  - The molecule's array index IS its identifier (row order is fixed to the source). `sigma_shake_id`
    additionally gives each molecule's id in the sigma-shake dataset (same molecule there, at a
    different level of theory), so the two datasets can be joined.
  - `failed_indices` lists molecules whose shielding calculation failed (none in this release):
    their shieldings would be sentinel zeros and must be ignored. There are no SMILES.

Requires: Python >= 3.7, numpy, h5py. No other dependencies.
"""
from __future__ import annotations
import numpy as np
import h5py


class SigmaConcentrate:
    """Reader for the sigma-concentrate HDF5 file: gas/PCM shieldings on 50,000 sigma-shake molecules."""

    def __init__(self, path: str):
        self.f = h5py.File(path, "r")
        self.n_molecules = int(self.f.attrs["n_molecules"])
        self.n_atoms_total = int(self.f.attrs["n_atoms"])
        self.level_of_theory = str(self.f.attrs["level_of_theory"])
        self.geometry = str(self.f.attrs["geometry"])
        # atom row offsets: start[i]..start[i+1] for molecule i (one cumsum, cached)
        na = self.f["n_atoms"][:].astype(np.int64)
        self._start = np.empty(self.n_molecules + 1, np.int64)
        self._start[0] = 0
        np.cumsum(na, out=self._start[1:])
        self._sid = self.f["sigma_shake_id"][:]
        self._failed = set(int(x) for x in self.f["failed_indices"][:])
        self.shieldings = sorted(k for k in self.f.keys() if k.startswith("shielding_"))

    def _check(self, index: int):
        if not 0 <= index < self.n_molecules:
            raise IndexError(f"molecule index {index} out of range [0, {self.n_molecules})")

    def _scaled(self, name: str, sl: slice) -> np.ndarray:
        d = self.f[name]
        return d[sl].astype(np.float64) * float(d.attrs["scale"])

    def sigma_shake_id(self, index: int) -> int:
        """Return molecule `index`'s id in the sigma-shake dataset."""
        self._check(index)
        return int(self._sid[index])

    def molecule(self, index: int) -> dict:
        """Return one molecule's decoded data (physical units) by array index.

        Keys: index, sigma_shake_id, atomic_numbers, coordinates (n,3 Angstrom),
        shielding_b3lyp_gas (n, ppm), shielding_b3lyp_pcm (n, ppm), and `failed`. If `failed`
        is True the shieldings are sentinel zeros and should be ignored; geometry is still valid.
        """
        self._check(index)
        a, b = int(self._start[index]), int(self._start[index + 1])
        sl = slice(a, b)
        out = {
            "index": index,
            "sigma_shake_id": int(self._sid[index]),
            "atomic_numbers": self.f["atomic_numbers"][sl],
            "coordinates": self._scaled("coordinates", sl),   # (n,3) Angstrom
            "failed": index in self._failed,
        }
        for s in self.shieldings:
            out[s] = self._scaled(s, sl)                       # (n,) ppm
        return out

    def __len__(self):
        return self.n_molecules

    def close(self):
        """Close the underlying HDF5 file."""
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def __repr__(self):
        return (f"<SigmaConcentrate: {self.n_molecules:,} molecules, "
                f"{self.n_atoms_total:,} atoms, {self.level_of_theory}>")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "sigma-concentrate.hdf5"
    with SigmaConcentrate(path) as ds:
        print(ds)
        m = ds.molecule(0)
        print(f"  mol 0: sigma_shake_id={m['sigma_shake_id']}  Z={list(m['atomic_numbers'][:6])}  "
              f"failed={m['failed']}")
        print(f"    gas={np.round(m['shielding_b3lyp_gas'][:3], 4)}  "
              f"pcm={np.round(m['shielding_b3lyp_pcm'][:3], 4)}")
