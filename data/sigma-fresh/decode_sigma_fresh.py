"""
Reader for the sigma-fresh explicit-solvent dataset (numpy + h5py only).

sigma-fresh holds PBE0/pcSseg-1 (solute) / PBE0/MIDI! (solvent) NMR shieldings on
explicitly-solvated MD snapshots, for ~10,000 solutes in each of four solvents
(chloroform, benzene, methanol, TIP4P water). It trains the MagNET-x explicit-solvent
model, which predicts the per-atom correction `shielding_solvated - shielding_isolated`.

Layout: one HDF5 with a group per solvent, and a subgroup `solute_NNNNN` per solute.
Each solute stores all 102 trajectory frames of the full solvent cluster. Only the
solute atoms carry shieldings; many frames are geometry-only (shieldings not computed),
marked by `status` and by a sentinel value in the shielding arrays.

    from decode_sigma_fresh import SigmaFresh
    with SigmaFresh("sigma-fresh.hdf5") as ds:
        print(ds.solvents)                       # ['chloroform','benzene','methanol','TIP4P']
        print(len(ds.solutes("chloroform")))     # ~10084
        p = ds.pose("chloroform", 3, 5)          # solvent, solute_index (1-based), frame_index (1-based)
        print(p["name"], p["type"], p["status"]) # e.g. GLY182705 train complete
        corr = p["shielding_solvated"] - p["shielding_isolated"]   # NaN where not computed

Requires: numpy, h5py.
"""
import numpy as np
import h5py

SCALE = 1e-4
SENTINEL = np.int32(-2147483648)
_STATUS = {0: "incomplete", 1: "geometries_only", 2: "partial_shieldings",
           3: "complete", 4: "error"}


class SigmaFresh:
    """Reader for a sigma-fresh HDF5 file: per-solvent, per-solute explicit-solvation trajectories."""

    def __init__(self, path):
        self.f = h5py.File(path, "r")
        self.scale = float(self.f.attrs.get("scale", SCALE))
        self.sentinel = np.int32(self.f.attrs.get("sentinel", SENTINEL))
        self.solvents = [s for s in self.f.attrs.get("solvents", "").split(",") if s] \
            or sorted(self.f.keys())

    # ---- catalogue ----
    def solutes(self, solvent):
        """Sorted list of solute subgroup names present for a solvent."""
        return sorted(self.f[solvent].keys())

    def n_solutes(self, solvent):
        """Number of solutes present for a solvent."""
        return len(self.f[solvent].keys())

    def _grp(self, solvent, solute_index):
        if solvent not in self.f:
            raise KeyError(f"no such solvent: {solvent!r}")
        name = f"solute_{int(solute_index):05d}"
        if name not in self.f[solvent]:
            raise IndexError(f"{solvent}/{name} not in dataset")
        return self.f[solvent][name]

    def _decode_shield(self, raw):
        out = raw.astype(np.float64) * self.scale
        out[raw == self.sentinel] = np.nan
        return out

    # ---- access ----
    def solute(self, solvent, solute_index):
        """All frames for one solute as a dict (coordinates (102,A,3); shieldings (102,n_solute))."""
        g = self._grp(solvent, solute_index)
        Z = g["atomic_numbers"][()]
        ns = int(g.attrs["n_solute_atoms"])
        return {
            "solvent": solvent,
            "solute_index": int(solute_index),
            "name": g.attrs["name"],
            "type": g.attrs["type"],
            "solute_charge": int(g.attrs["solute_charge"]),
            "n_solute_atoms": ns,
            "n_solvent_atoms": int(self.f[solvent].attrs["n_solvent_atoms"]),
            "atomic_numbers": Z,
            "solute_mask": np.arange(Z.shape[0]) < ns,
            "coordinates": g["coordinates"][()].astype(np.float64) * self.scale,
            "shielding_isolated": self._decode_shield(g["shielding_isolated"][()]),
            "shielding_solvated": self._decode_shield(g["shielding_solvated"][()]),
            "status": np.array([_STATUS[s] for s in g["status"][()]], dtype=object),
            "status_code": g["status"][()],
            "n_solvents_partial": g["n_solvents_partial"][()],
            "partial_radius": g["partial_radius"][()].astype(np.float64) * self.scale,
            "full_radius": g["full_radius"][()].astype(np.float64) * self.scale,
        }

    def pose(self, solvent, solute_index, frame_index):
        """One (solute, frame) pose. frame_index is 1-based (1..102), matching the source/CSVs.

        coordinates: (A,3) full solvent cluster, raw (uncentered).
        shielding_isolated / shielding_solvated: (n_solute,), NaN where not computed.
        For type='train' solutes the solvated shielding was computed on the partial pose
        (the first `n_solvents_partial` solvent molecules); for type='test' it used the
        full cluster.
        """
        g = self._grp(solvent, solute_index)
        n_frames = g["coordinates"].shape[0]
        if not (1 <= int(frame_index) <= n_frames):
            raise IndexError(f"frame_index {frame_index} out of range 1..{n_frames}")
        fi = int(frame_index) - 1
        ns = int(g.attrs["n_solute_atoms"])
        return {
            "solvent": solvent,
            "solute_index": int(solute_index),
            "frame_index": int(frame_index),
            "name": g.attrs["name"],
            "type": g.attrs["type"],
            "n_solute_atoms": ns,
            "atomic_numbers": g["atomic_numbers"][()],
            "coordinates": g["coordinates"][fi].astype(np.float64) * self.scale,
            "shielding_isolated": self._decode_shield(g["shielding_isolated"][fi]),
            "shielding_solvated": self._decode_shield(g["shielding_solvated"][fi]),
            "status": _STATUS[int(g["status"][fi])],
            "n_solvents_partial": int(g["n_solvents_partial"][fi]),
            "partial_radius": float(g["partial_radius"][fi]) * self.scale,
            "full_radius": float(g["full_radius"][fi]) * self.scale,
        }

    # ---- lifecycle ----
    def close(self):
        """Close the underlying HDF5 file."""
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def __repr__(self):
        return f"<SigmaFresh solvents={self.solvents}>"


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "sigma-fresh.hdf5"
    with SigmaFresh(path) as ds:
        print(ds)
        for s in ds.solvents:
            print(f"  {s}: {ds.n_solutes(s)} solutes")
        p = ds.pose(ds.solvents[0], 3, 1)
        print("pose chloroform/solute_00003/frame_1:", p["name"], p["type"], p["status"])
        print("  isolated[:4]:", p["shielding_isolated"][:4])
        print("  solvated[:4]:", p["shielding_solvated"][:4])
