"""
decode_supertest.py - reader/decoder for the supertestset_magnet_x HDF5 dataset.

This is the explicit-solvent test set for MagNET-x. It holds DFT (Gaussian) NMR shieldings for
natural-product solutes computed two ways: with the solute isolated, and with the solute sitting
in an explicit box of solvent molecules sampled from an OpenMM molecular-dynamics simulation. The
explicit-solvent correction at each atom is the solvated shielding minus the isolated shielding,
which is what MagNET-x learns to predict.

Each record is one computed structure for a (molecule, solvent, conformer) combination, in either
the isolated or the solvated state:

  isolated   the full system is just the solute
  solvated   the full system is the solute (always the first atoms) followed by solvent molecules

The shieldings are stored only for the solute atoms (the solvent carries no NMR signal of interest).
There are 318 records covering 8 molecules across 4 solvents and 5 conformers. A separate catalog
lists all 50 candidate molecules of the test set with their SMILES.

Storage format:
  - Per-atom arrays are concatenated across records. Record i has `n_full_atoms[i]` atoms in its
    full system and `n_solute_atoms[i]` solute atoms (the leading atoms of the full system). Its
    full atomic numbers and coordinates occupy `n_full_atoms[i]` rows; its solute shieldings occupy
    `n_solute_atoms[i]` rows. Offsets are reconstructed once, on open.
  - coordinates and shieldings are int32 fixed point: physical = stored * 1e-4, reconstruction
    error <= 5e-5. A missing value would be the marker -2147483648 -> NaN.

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


class SuperTestMagNETX:
    """Reader for supertestset_magnet_x.hdf5. Use as a context manager."""

    def __init__(self, path: str):
        self.f = h5py.File(path, "r")
        self.n_records = int(self.f.attrs["n_records"])
        self.record_names = [str(s) for s in self.f["record_names"].asstr()[:]]
        self.record_solvents = [str(s) for s in self.f["record_solvents"].asstr()[:]]
        self._conf = self.f["conformer"][:]
        self._iso = self.f["is_isolated"][:].astype(bool)
        self._n_sol = self.f["n_solute_atoms"][:].astype(np.int64)
        self._n_full = self.f["n_full_atoms"][:].astype(np.int64)
        self._fstart = np.empty(self.n_records + 1, np.int64)
        self._fstart[0] = 0
        np.cumsum(self._n_full, out=self._fstart[1:])
        self._sstart = np.empty(self.n_records + 1, np.int64)
        self._sstart[0] = 0
        np.cumsum(self._n_sol, out=self._sstart[1:])

    # ------------------------------------------------------------------ catalog
    def catalog(self) -> dict:
        """The 50-molecule candidate catalog as a {name: SMILES} dict."""
        names = self.f["catalog_names"].asstr()[:]
        smiles = self.f["catalog_smiles"].asstr()[:]
        return {str(n): str(s) for n, s in zip(names, smiles)}

    # ------------------------------------------------------------------ records
    def _check(self, index: int):
        if not 0 <= index < self.n_records:
            raise IndexError(f"record index {index} out of range [0, {self.n_records})")

    def record(self, index: int) -> dict:
        """Return one record (physical units) by position.

        Keys: name, solvent, conformer (int), is_isolated (bool); full_atomic_numbers (n_full,),
        full_coordinates (n_full, 3 Angstrom); solute_atomic_numbers (n_solute,), solute_shieldings
        (n_solute,) ppm. The solute atoms are the leading n_solute atoms of the full system.
        """
        self._check(index)
        fsl = slice(int(self._fstart[index]), int(self._fstart[index + 1]))
        ssl = slice(int(self._sstart[index]), int(self._sstart[index + 1]))
        ns = int(self._n_sol[index])
        full_an = self.f["full_atomic_numbers"][fsl]
        return {
            "name": self.record_names[index],
            "solvent": self.record_solvents[index],
            "conformer": int(self._conf[index]),
            "is_isolated": bool(self._iso[index]),
            "full_atomic_numbers": full_an,
            "full_coordinates": _decode(self.f["full_coordinates"][fsl]),
            "solute_atomic_numbers": full_an[:ns],
            "solute_shieldings": _decode(self.f["solute_shieldings"][ssl]),
        }

    def find(self, name: str, solvent: str, conformer: int, is_isolated: bool) -> int:
        """Index of the record matching (name, solvent, conformer, is_isolated), or raise KeyError."""
        for i in range(self.n_records):
            if (self.record_names[i] == name and self.record_solvents[i] == solvent
                    and int(self._conf[i]) == conformer and bool(self._iso[i]) == is_isolated):
                return i
        raise KeyError(f"no record for {name!r}, {solvent!r}, conformer {conformer}, "
                       f"isolated={is_isolated}")

    def explicit_correction(self, name: str, solvent: str, conformer: int) -> np.ndarray:
        """The explicit-solvent correction (solvated minus isolated solute shieldings), (n_solute,)
        ppm, for one (molecule, solvent, conformer)."""
        solvated = self.record(self.find(name, solvent, conformer, is_isolated=False))
        isolated = self.record(self.find(name, solvent, conformer, is_isolated=True))
        return solvated["solute_shieldings"] - isolated["solute_shieldings"]

    def close(self):
        """Close the underlying HDF5 file."""
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def __len__(self):
        return self.n_records

    def __repr__(self):
        n_iso = int(self._iso.sum())
        return (f"<SuperTestMagNETX: {self.n_records} records "
                f"({n_iso} isolated, {self.n_records - n_iso} solvated), "
                f"solvents {self.f.attrs['solvents']}>")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "supertestset_magnet_x.hdf5"
    with SuperTestMagNETX(path) as ds:
        print(ds)
        print(f"  catalog: {len(ds.catalog())} molecules")
        r = ds.record(ds.find(ds.record_names[0], ds.record_solvents[0], int(ds._conf[0]), bool(ds._iso[0])))
        corr = ds.explicit_correction(r["name"], r["solvent"], r["conformer"])
        print(f"  {r['name']} in {r['solvent']} (conformer {r['conformer']}): "
              f"{len(r['solute_atomic_numbers'])} solute atoms")
        print(f"    explicit-solvent correction[:3] = {np.round(corr[:3], 4)} ppm")
