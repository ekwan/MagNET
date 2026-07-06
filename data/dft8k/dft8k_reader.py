"""
dft8k_reader.py - reader/decoder for the dft8k HDF5 dataset.

DFT8k is an external NMR shielding benchmark of 7111 organic molecules (Guan, Paton and
co-workers; reference 14 in the paper), used to evaluate the MagNET models out of sample.
The file has TWO groups, one per level of theory. Both groups cover the SAME 7111 molecules
and use the SAME integer molecule ids, but in different row order, so match a molecule across
the two groups by id (use `molecule_by_id`), not by row position.

  /aimnet2_wp04_wb97xd_pcsseg2   AIMNet2 geometries; per atom, three DFT shieldings plus the
                                 two MagNET predictions. This is the file behind the paper's
                                 DFT8k panels.
  /b3lyp_pbe0_pcsseg1            B3LYP/pcSseg-1 geometries; three geometries per molecule with
                                 one PBE0/pcSseg-1 shielding each, plus symmetry-equivalence
                                 groups and per-atom averaging weights. This is the format the
                                 model's own training/eval loader consumes.

Storage format:
  - Per-atom arrays are concatenated across the molecules in a group. Each molecule's atom
    count is in `n_atoms`; molecule i owns the rows [start[i], start[i+1]) where start is the
    cumulative sum of `n_atoms` (reconstructed once, on open).
  - coordinates and shieldings are int32 fixed-point: physical = stored * scale (scale = 1e-4),
    so reconstruction error is <= 5e-5 (Angstrom or ppm). A missing value (an atom a model does
    not predict, or an absent shielding) is the marker -2147483648 and decodes to NaN.
  - The molecule's integer `molecule_ids` entry is its identifier.

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


class _Group:
    """Common molecule-indexing logic shared by both groups."""

    def __init__(self, grp: h5py.Group):
        self.g = grp
        self.name = grp.name.lstrip("/")
        self.molecule_ids = grp["molecule_ids"][:]
        self.n_molecules = int(grp.attrs["n_molecules"])
        self.n_atoms = int(grp.attrs["n_atoms"])
        self.level_of_theory = str(grp.attrs["level_of_theory"])
        self.geometry = str(grp.attrs["geometry"])
        na = grp["n_atoms"][:].astype(np.int64)
        self._start = np.empty(self.n_molecules + 1, np.int64)
        self._start[0] = 0
        np.cumsum(na, out=self._start[1:])
        self._id_to_index = {int(m): i for i, m in enumerate(self.molecule_ids)}

    def _check(self, index: int):
        if not 0 <= index < self.n_molecules:
            raise IndexError(f"molecule index {index} out of range [0, {self.n_molecules})")

    def _rows(self, index: int) -> slice:
        return slice(int(self._start[index]), int(self._start[index + 1]))

    def index_of(self, molecule_id: int) -> int:
        """Return the row position of a molecule id within this group."""
        if int(molecule_id) not in self._id_to_index:
            raise KeyError(f"molecule id {molecule_id} not in group {self.name}")
        return self._id_to_index[int(molecule_id)]

    def molecule_by_id(self, molecule_id: int) -> dict:
        """Look up a molecule by its id instead of its row position."""
        return self.molecule(self.index_of(molecule_id))

    def __len__(self):
        return self.n_molecules


class AIMNet2Group(_Group):
    """/aimnet2_wp04_wb97xd_pcsseg2 - AIMNet2 geometries, DFT and MagNET shieldings per atom."""

    def __init__(self, grp: h5py.Group):
        super().__init__(grp)
        # one SMILES per molecule. 6780 of 7111 are real; the rest are the literal "none".
        # Older files may lack the field, so guard for it.
        self._smiles = grp["smiles"].asstr()[:] if "smiles" in grp else None

    def smiles(self, index: int) -> str:
        """Return molecule i's SMILES string (the literal "none" if it was not provided)."""
        self._check(index)
        if self._smiles is None:
            return "none"
        return str(self._smiles[index])

    def molecule(self, index: int) -> dict:
        """Return one molecule's atom data (physical units) by position.

        Keys: id, smiles; atomic_numbers, coordinates (n,3 Angstrom); the three DFT shieldings
        (pbe0_pcSseg1, wp04_pcSseg2, wb97xd_pcSseg2; each (n,) ppm); and the two MagNET
        predictions (nn_wp04_pcSseg2, nn_wb97xd_pcSseg2; (n,) ppm). A MagNET prediction is
        NaN at atoms that model does not predict (WP04 is the 1H model, wB97X-D the 13C).
        """
        self._check(index)
        sl = self._rows(index)
        return {
            "id": int(self.molecule_ids[index]),
            "smiles": self.smiles(index),
            "atomic_numbers": self.g["atomic_numbers"][sl],
            "coordinates": _decode(self.g["coordinates"][sl]),
            "pbe0_pcSseg1": _decode(self.g["shielding_pbe0_pcSseg1"][sl]),
            "wp04_pcSseg2": _decode(self.g["shielding_wp04_pcSseg2"][sl]),
            "wb97xd_pcSseg2": _decode(self.g["shielding_wb97xd_pcSseg2"][sl]),
            "nn_wp04_pcSseg2": _decode(self.g["nn_shielding_wp04_pcSseg2"][sl]),
            "nn_wb97xd_pcSseg2": _decode(self.g["nn_shielding_wb97xd_pcSseg2"][sl]),
        }

    def all_shieldings(self) -> dict:
        """Every atom in the dataset as flat arrays (physical units), for whole-dataset analyses
        like the MagNET-vs-DFT residual histograms.

        Keys: atomic_numbers (n_atoms_total,); the three DFT shieldings (pbe0_pcSseg1,
        wp04_pcSseg2, wb97xd_pcSseg2) and the two MagNET predictions (nn_wp04_pcSseg2,
        nn_wb97xd_pcSseg2), each (n_atoms_total,) ppm. A MagNET prediction is NaN at atoms that
        model does not predict (WP04 is the 1H model, wB97X-D the 13C). To get one nucleus' residual,
        difference the DFT and MagNET column for that model at that element and drop the NaNs.
        """
        return {
            "atomic_numbers": self.g["atomic_numbers"][:],
            "pbe0_pcSseg1": _decode(self.g["shielding_pbe0_pcSseg1"][:]),
            "wp04_pcSseg2": _decode(self.g["shielding_wp04_pcSseg2"][:]),
            "wb97xd_pcSseg2": _decode(self.g["shielding_wb97xd_pcSseg2"][:]),
            "nn_wp04_pcSseg2": _decode(self.g["nn_shielding_wp04_pcSseg2"][:]),
            "nn_wb97xd_pcSseg2": _decode(self.g["nn_shielding_wb97xd_pcSseg2"][:]),
        }

    def __repr__(self):
        return (f"<AIMNet2Group /{self.name}: {self.n_molecules:,} molecules, "
                f"{self.n_atoms:,} atoms>")


class B3LYPGroup(_Group):
    """/b3lyp_pbe0_pcsseg1 - three geometries per molecule, symmetry groups and weights."""

    def __init__(self, grp: h5py.Group):
        super().__init__(grp)
        self.n_geometries = int(grp.attrs["n_geometries"])
        # symmetry groups are stored flat: n_groups per molecule, each group's length, then
        # all member atom indices. Reconstruct the per-molecule offsets once, on open.
        self._sym_n = grp["symmetric_atoms_n_groups"][:].astype(np.int64)
        self._sym_len = grp["symmetric_atoms_group_lengths"][:].astype(np.int64)
        self._sym_mem = grp["symmetric_atoms_members"][:]
        self._group_start = np.empty(self.n_molecules + 1, np.int64)
        self._group_start[0] = 0
        np.cumsum(self._sym_n, out=self._group_start[1:])
        self._member_start = np.empty(self._sym_len.shape[0] + 1, np.int64)
        self._member_start[0] = 0
        np.cumsum(self._sym_len, out=self._member_start[1:])

    def symmetric_atoms(self, index: int) -> list:
        """Return molecule i's symmetry-equivalence groups as a list of lists of atom indices.

        The indices are 0-based positions into this molecule's atom arrays (atomic_numbers,
        coordinates), so a group can be used to index those arrays directly.
        """
        self._check(index)
        groups = []
        for gi in range(int(self._group_start[index]), int(self._group_start[index + 1])):
            a, b = int(self._member_start[gi]), int(self._member_start[gi + 1])
            groups.append([int(x) for x in self._sym_mem[a:b]])
        return groups

    def molecule(self, index: int) -> dict:
        """Return one molecule's data (physical units) by position.

        Keys: id, atomic_numbers; coordinates (n_geometries, n, 3 Angstrom); shielding
        (n_geometries, n, ppm), where geometry 0 is a reference and 1, 2 are perturbations;
        weights (n,) per-atom symmetry-averaging weight; symmetric_atoms (list of lists of
        equivalent atoms, each a 0-based index into this molecule's atom arrays).
        """
        self._check(index)
        sl = self._rows(index)
        return {
            "id": int(self.molecule_ids[index]),
            "atomic_numbers": self.g["atomic_numbers"][sl],
            "coordinates": _decode(self.g["coordinates"][:, sl, :]),
            "shielding": _decode(self.g["shielding"][:, sl]),
            "weights": self.g["weights"][sl].astype(np.float64),
            "symmetric_atoms": self.symmetric_atoms(index),
        }

    def __repr__(self):
        return (f"<B3LYPGroup /{self.name}: {self.n_molecules:,} molecules, "
                f"{self.n_atoms:,} atoms, {self.n_geometries} geometries>")


class DFT8k:
    """Reader for dft8k.hdf5. Use as a context manager; access the two groups by attribute."""

    def __init__(self, path: str):
        self.f = h5py.File(path, "r")
        self.aimnet2 = AIMNet2Group(self.f["aimnet2_wp04_wb97xd_pcsseg2"])
        self.b3lyp = B3LYPGroup(self.f["b3lyp_pbe0_pcsseg1"])

    def close(self):
        """Close the underlying HDF5 file."""
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def __repr__(self):
        return f"<DFT8k: {self.aimnet2!r}, {self.b3lyp!r}>"


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "dft8k.hdf5"
    with DFT8k(path) as ds:
        print(ds)
        m = ds.aimnet2.molecule(0)
        h = m["atomic_numbers"] == 1
        print(f"  aimnet2 mol id={m['id']}: {len(m['atomic_numbers'])} atoms  smiles={m['smiles']}")
        print(f"    1H WP04 DFT[:3]={np.round(m['wp04_pcSseg2'][h][:3], 3)}  "
              f"MagNET[:3]={np.round(m['nn_wp04_pcSseg2'][h][:3], 3)}")
        b = ds.b3lyp.molecule(0)
        print(f"  b3lyp   mol id={b['id']}: coordinates {b['coordinates'].shape}, "
              f"{len(b['symmetric_atoms'])} symmetry groups")
