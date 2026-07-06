"""
decode_gdb_qcd.py - reader/decoder for the gdb_qcd HDF5 dataset.

gdb_qcd is a quasiclassical-dynamics benchmark: replicate molecular-dynamics trajectories for
2461 small molecules drawn from GDB (the MagNET / sigma-shake test set). It is used to benchmark
rovibrational corrections, which are computed by comparing NMR shieldings on the moving
(displaced) geometries against the resting (stationary) geometry.

For each molecule there are several trajectories; each trajectory has 33 frames; each frame gives
every atom's position and its PBE0/pcSseg-1 NMR shielding. Trajectory counts vary by molecule
(about 23 on average); the frame count is always 33. The resting (stationary) reference shieldings
are not included here: they are the test-set entries of the MagNET training data (sigma-shake),
matched by the same molecule id.

Storage format:
  - Per-atom arrays are concatenated across molecules. Molecule i has `n_atoms[i]` atoms and
    `n_trajectories[i]` trajectories. Its atomic numbers occupy `n_atoms[i]` rows; its coordinates
    and shieldings occupy `n_trajectories[i] * 33 * n_atoms[i]` rows, which reshape to
    (n_trajectories, 33, n_atoms, ...). Offsets are reconstructed once, on open.
  - coordinates and shieldings are int32 fixed point: physical = stored * 1e-4, so reconstruction
    error is <= 5e-5 (Angstrom or ppm). A missing value would be the marker -2147483648 -> NaN
    (none occur in this dataset).
  - `molecule_ids` are integers and match the sigma-shake test-set ids.

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


class GDBQCD:
    """Reader for gdb_qcd.hdf5. Use as a context manager; index molecules by position or by id."""

    def __init__(self, path: str):
        self.f = h5py.File(path, "r")
        self.molecule_ids = self.f["molecule_ids"][:]
        self.n_molecules = int(self.f.attrs["n_molecules"])
        self.n_frames = int(self.f.attrs["n_frames"])
        self.nmr_level_of_theory = str(self.f.attrs["nmr_level_of_theory"])
        self._n_atoms = self.f["n_atoms"][:].astype(np.int64)
        self._n_traj = self.f["n_trajectories"][:].astype(np.int64)
        # atom-row offsets (atomic_numbers) and coordinate/shielding-row offsets
        self._astart = np.empty(self.n_molecules + 1, np.int64)
        self._astart[0] = 0
        np.cumsum(self._n_atoms, out=self._astart[1:])
        block = self._n_traj * self.n_frames * self._n_atoms
        self._cstart = np.empty(self.n_molecules + 1, np.int64)
        self._cstart[0] = 0
        np.cumsum(block, out=self._cstart[1:])
        self._id_to_index = {int(m): i for i, m in enumerate(self.molecule_ids)}

    def _check(self, index: int):
        if not 0 <= index < self.n_molecules:
            raise IndexError(f"molecule index {index} out of range [0, {self.n_molecules})")

    def index_of(self, molecule_id: int) -> int:
        """Position of the molecule with this id; raises KeyError if absent."""
        if int(molecule_id) not in self._id_to_index:
            raise KeyError(f"molecule id {molecule_id} not in gdb_qcd")
        return self._id_to_index[int(molecule_id)]

    def n_trajectories(self, index: int) -> int:
        """How many trajectories molecule i has."""
        self._check(index)
        return int(self._n_traj[index])

    def molecule(self, index: int) -> dict:
        """Return one molecule's trajectory data (physical units) by position.

        Keys:
          id               - integer molecule id (matches the sigma-shake test set)
          atomic_numbers   - (n_atoms,) uint8
          coordinates      - (n_trajectories, 33, n_atoms, 3) Angstrom
          shieldings       - (n_trajectories, 33, n_atoms) ppm, PBE0/pcSseg-1
        """
        self._check(index)
        na = int(self._n_atoms[index])
        nt = int(self._n_traj[index])
        asl = slice(int(self._astart[index]), int(self._astart[index + 1]))
        csl = slice(int(self._cstart[index]), int(self._cstart[index + 1]))
        coords = _decode(self.f["coordinates"][csl]).reshape(nt, self.n_frames, na, 3)
        shields = _decode(self.f["shieldings"][csl]).reshape(nt, self.n_frames, na)
        return {
            "id": int(self.molecule_ids[index]),
            "atomic_numbers": self.f["atomic_numbers"][asl],
            "coordinates": coords,
            "shieldings": shields,
        }

    def molecule_by_id(self, molecule_id: int) -> dict:
        """Return one molecule's trajectory data (see `molecule`) by id."""
        return self.molecule(self.index_of(molecule_id))

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
        return (f"<GDBQCD: {self.n_molecules:,} molecules, "
                f"{int(self.f.attrs['n_trajectories']):,} trajectories, "
                f"{self.n_frames} frames each, {self.nmr_level_of_theory}>")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "gdb_qcd.hdf5"
    with GDBQCD(path) as ds:
        print(ds)
        m = ds.molecule(0)
        print(f"  molecule id={m['id']}: {len(m['atomic_numbers'])} atoms, "
              f"coordinates {m['coordinates'].shape}, shieldings {m['shieldings'].shape}")
        print(f"    trajectory 0, frame 0, atom 0: xyz={np.round(m['coordinates'][0,0,0], 4)}  "
              f"shielding={m['shieldings'][0,0,0]:.4f} ppm")
