"""
decode_sigma_shake.py - reader/decoder for the sigma-shake HDF5 dataset.

sigma-shake stores PBE0/pcSseg-1 isotropic NMR shieldings for the stationary and
vibrationally-perturbed geometries of ~4.79M GDB-derived molecules. It is the training
data for the MagNET foundation model.

Storage format:
  - Per-atom arrays are concatenated across all molecules; molecule i occupies
    rows [atom_start[i], atom_end[i]).
  - Coordinates, perturbation, and shieldings are stored as int32 fixed-point.
    Physical value = stored_integer * scale, where `scale` (= 1e-4) is an attribute
    on each dataset. Reconstruction error is <= 5e-5 (Angstrom or ppm).
  - The perturbed geometry is NOT stored directly; it is
        perturbed_coords = stationary_coords + perturbation
  - Where `incomplete_mask` is True, the molecule has no perturbed calculation:
    `perturbation` and `shielding_perturbed` are sentinel zeros and must be ignored.

Requires: Python >= 3.7, numpy, h5py.  No other dependencies.
"""
from __future__ import annotations
import numpy as np
import h5py


class SigmaShake:
    """Reader for the sigma-shake HDF5 file: stationary and vibrationally-perturbed shieldings."""

    def __init__(self, path: str):
        self.f = h5py.File(path, "r")
        self._id = self.f["molecule_id"][:]
        self._start = self.f["atom_start"][:]
        self._end = self.f["atom_end"][:]
        self._id_to_row = None  # built lazily on first by_id()

    # ---- scalars / metadata ----
    @property
    def n_molecules(self) -> int:
        """Total number of molecules in the file."""
        return self._id.shape[0]

    @property
    def n_atoms(self) -> int:
        """Total number of atoms across all molecules."""
        return int(self.f.attrs["n_atoms"])

    def _scaled(self, name: str, sl: slice) -> np.ndarray:
        d = self.f[name]
        return d[sl].astype(np.float64) * float(d.attrs["scale"])

    # ---- molecule access ----
    def molecule(self, index: int) -> dict:
        """Return one molecule's decoded data by ROW index (0..n_molecules-1).

        For an incomplete molecule (no perturbed calculation) the returned dict OMITS the
        'perturbation', 'perturbed_coords', and 'shielding_perturbed' keys.
        """
        if not 0 <= index < self.n_molecules:
            raise IndexError(f"molecule index {index} out of range [0, {self.n_molecules})")
        a, b = int(self._start[index]), int(self._end[index])
        sl = slice(a, b)
        Z = self.f["atomic_numbers"][sl]
        stat = self._scaled("stationary_coords", sl)
        incomplete = bool(self.f["incomplete_mask"][a])  # constant within a molecule
        out = {
            "molecule_id": int(self._id[index]),
            "atomic_numbers": Z,
            "stationary_coords": stat,                # (n,3) Angstrom
            "shielding_stationary": self._scaled("shielding_stationary", sl),  # (n,) ppm
            "incomplete": incomplete,
        }
        if not incomplete:
            pert = self._scaled("perturbation", sl)
            out["perturbed_coords"] = stat + pert     # reconstruct absolute geometry
            out["perturbation"] = pert
            out["shielding_perturbed"] = self._scaled("shielding_perturbed", sl)
        return out

    def by_id(self, molecule_id: int) -> dict:
        """Return one molecule's decoded data (see `molecule`) looked up by molecule_id."""
        if self._id_to_row is None:
            self._id_to_row = {int(v): i for i, v in enumerate(self._id)}
        try:
            row = self._id_to_row[int(molecule_id)]
        except KeyError:
            raise KeyError(f"unknown molecule_id {molecule_id}") from None
        return self.molecule(row)

    # ---- splits ----
    def split(self, which: str, effective: bool = True) -> np.ndarray:
        """Return molecule_id values for 'train'/'val'/'test'.
        effective=True drops remove_molecule_ids (the load-time filter), reproducing the
        sizes reported in the SI (train 3,730,572 / val 329,154 / test 329,163)."""
        ids = self.f[f"split_{which}_ids"][:]
        if effective:
            rm = self.f["remove_molecule_ids"][:]
            ids = ids[~np.isin(ids, rm)]
        return ids

    def close(self):
        """Close the underlying HDF5 file."""
        self.f.close()

    def __enter__(self): return self
    def __exit__(self, *a): self.close()


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "sigma-shake.hdf5"
    with SigmaShake(path) as ds:
        print(f"molecules: {ds.n_molecules:,}   atoms: {ds.n_atoms:,}")
        for w in ("train", "val", "test"):
            print(f"  effective {w:5s}: {len(ds.split(w)):,}")
        m = ds.molecule(0)
        print(f"\nmolecule 0 (id={m['molecule_id']}): {len(m['atomic_numbers'])} atoms, "
              f"incomplete={m['incomplete']}")
        print("  Z         :", m["atomic_numbers"][:8])
        print("  stat xyz  :", np.round(m["stationary_coords"][0], 4), "...")
        print("  sigma_stat:", np.round(m["shielding_stationary"][:4], 4), "...")
        if not m["incomplete"]:
            print("  pert xyz  :", np.round(m["perturbed_coords"][0], 4), "...")
            print("  sigma_pert:", np.round(m["shielding_perturbed"][:4], 4), "...")
