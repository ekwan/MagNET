"""Loader for the natural-products ("applications") release.

applications.hdf5 stores shieldings and coordinates as whole numbers equal to the real value
times 10,000 (scale 1e-4), with -2147483648 marking values that were never computed. Energies
are stored as ordinary floating-point numbers. This module turns the stored numbers back into
real values (leaving blanks where data is missing) and exposes one accessor per data layer, so
notebooks never touch the raw file format. It is the single place where decoding happens.

Solute names: vomicine, prednisone, peptide, flavone, dihydrotanshinone_I, and
isomer_1E/1Z/2E/2Z/3E/3Z/4N/4O.
Solvents: TIP4P (water), benzene, chloroform, methanol.
"""
import h5py
import io
import os

import numpy as np
import pandas as pd

_SCALE = 1e4
_MISSING_MARKER = -2147483648

SOLVENTS = ["TIP4P", "benzene", "chloroform", "methanol"]
# shell label = approximate number of heavy (non-hydrogen) atoms in the solute-plus-solvent cluster
SHELL_SIZES = [50, 150, 250, 350, 450, 550, 650]


def _decode(values):
    """Turn stored whole numbers back into real values; blanks where data is missing."""
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.integer):
        out = values.astype(np.float64) / _SCALE
        out[values == _MISSING_MARKER] = np.nan
        return out
    return np.asarray(values, dtype=np.float64)


class Applications:
    """Read-only accessor for applications.hdf5 (+ the experimental spreadsheet)."""

    def __init__(self, hdf5_path, experimental_xlsx=None, md_geometries_path=None):
        self.hdf5_path = str(hdf5_path)
        self.experimental_xlsx = str(experimental_xlsx) if experimental_xlsx else None
        # the large MD-geometry companion is optional; default to the sibling file if present
        if md_geometries_path is None:
            sibling = self.hdf5_path.replace("applications.hdf5", "applications_md_geometries.hdf5")
            md_geometries_path = sibling if (sibling != self.hdf5_path and os.path.exists(sibling)) else None
        self.md_geometries_path = str(md_geometries_path) if md_geometries_path else None

    def _open(self):
        return h5py.File(self.hdf5_path, "r")

    def _open_md(self):
        if not self.md_geometries_path:
            raise ValueError(
                "MD-geometry companion not available. Download applications_md_geometries.hdf5 "
                "next to applications.hdf5, or pass md_geometries_path=...")
        return h5py.File(self.md_geometries_path, "r")

    # ---- structure ----
    _NON_SOLUTE_GROUPS = ("solvents", "composite_model")

    def solutes(self):
        """List the solute names stored in the release."""
        with self._open() as f:
            return [k for k in f.keys() if k not in self._NON_SOLUTE_GROUPS]

    def atomic_numbers(self, solute):
        """Return a solute's per-atom atomic numbers."""
        with self._open() as f:
            return np.asarray(f[solute]["atomic_numbers"][()])

    # ---- MagNET-Zero (gas) + implicit-solvent (PCM) ----
    def magnet_zero(self, solute):
        """Boltzmann-weighted MagNET-Zero stationary shieldings and the PCM correction."""
        with self._open() as f:
            g = f[solute]["magnet_zero"]
            return {
                "stationary": _decode(g["stationary"][()]),
                "pcm_correction": _decode(g["pcm_correction"][()]),
            }

    def magnet_zero_conformers(self, solute):
        """Per-conformer dict: {conformer_name: {energy, geometry, magnet_zero, pcm}}."""
        out = {}
        with self._open() as f:
            cg = f[solute]["magnet_zero"]["conformers"]
            for ck in cg.keys():
                c = cg[ck]
                out[ck] = {
                    "energy": float(c["energy"][()]),
                    "geometry": _decode(c["geometry"][()]),
                    "atomic_numbers": np.asarray(c["atomic_numbers"][()]),
                    "magnet_zero": _decode(c["magnet_zero"][()]),
                    "pcm": _decode(c["pcm"][()]),
                }
        return out

    # ---- QCD (rovibrational) ----
    def qcd(self, solute, shielding_only=False):
        """qcd stationary (n_atoms,4)=[x,y,z,shielding] and trajectories (replica,frame,n_atoms,4).
        With shielding_only=True returns just the shielding channel (column 3)."""
        with self._open() as f:
            g = f[solute]["qcd"]
            stat = _decode(g["stationary"][()])
            traj = _decode(g["trajectories"][()])
        if shielding_only:
            return {"stationary": stat[..., 3], "trajectories": traj[..., 3]}
        return {"stationary": stat, "trajectories": traj}

    # ---- MagNET-X (explicit solvent) ----
    def magnet_x(self, solute, solvent, shell):
        """Explicit-solvent shieldings (frames, n_atoms, 2)=[isolated, solvated] for one shell."""
        with self._open() as f:
            return _decode(f[solute]["magnet_x"][solvent][f"shell_{shell}"][()])

    def magnet_x_stationary(self, solute, solvent):
        """Explicit-solvent stationary-geometry shieldings for one solute/solvent pair."""
        with self._open() as f:
            return _decode(f[solute]["magnet_x"][solvent]["stationary"][()])

    def available_shells(self, solute, solvent):
        """List the shell sizes stored for one solute/solvent pair."""
        with self._open() as f:
            return sorted(
                int(k.split("_")[1])
                for k in f[solute]["magnet_x"][solvent].keys()
                if k.startswith("shell_")
            )

    # ---- DFT reference (provided-only; PBE0(PBE1PBE)/pcSseg-3 GIAO, PCM per solvent) ----
    def dft_reference(self, solute, solvent):
        """Reference DFT shieldings for one solvent. Solvent keys here are 'gas', 'water',
        'benzene', 'chloroform', 'methanol' -- note 'water' is the CPCM continuum-water model
        (not the explicit 'TIP4P' key used by magnet_x), and 'gas' has no MagNET-x counterpart."""
        with self._open() as f:
            return _decode(f[solute]["dft_reference"][solvent]["shieldings"][()])

    def dft_reference_geometries(self, solute):
        """The DFT conformer geometries the reference shieldings were computed on, as a dict
        {conformer_name: {coordinates (n_atoms, 3), atomic_numbers (n_atoms,), energy}}."""
        out = {}
        with self._open() as f:
            g = f[solute]["dft_reference"].get("geometries")
            if g is None:
                return out
            for ck in g.keys():
                c = g[ck]
                out[ck] = {
                    "coordinates": _decode(c["coordinates"][()]),
                    "atomic_numbers": np.asarray(c["atomic_numbers"][()]),
                    "energy": float(c["energy"][()]),
                }
        return out

    # ---- composite-model coefficients (fit on delta-22, applied to the natural products) ----
    def _composite_csv(self, *path):
        key = "composite_model/" + "/".join(path)
        with self._open() as f:
            if key not in f:
                raise KeyError(f"{key} not in {self.hdf5_path}")
            txt = f[key][()]
        if isinstance(txt, bytes):
            txt = txt.decode("utf-8")
        return pd.read_csv(io.StringIO(txt))

    def ols_coefficients(self, formula, nucleus):
        """OLS composite-model coefficients (parameter rows x solvent columns) for one formula/nucleus."""
        return self._composite_csv("ols_coefficients", formula, nucleus)

    def bootstrap_coefficients(self, formula, nucleus):
        """Bootstrap-resampled composite-model coefficients (one row per solvent x seed)."""
        return self._composite_csv("bootstrap_coefficients", formula, nucleus)

    def rmse_distribution(self, nucleus):
        """delta-22 bootstrap RMSE distributions (solvent, nucleus, formula, seed, Bootstrap_RMSE)."""
        return self._composite_csv("rmse_distributions", nucleus)

    def pcm_conversion_factors(self, nucleus):
        """Per-solvent scale factors applied to the chloroform-only PCM correction."""
        return self._composite_csv("pcm_conversion_factors", nucleus)

    def composite_formulas(self, kind="ols_coefficients"):
        """List the formula keys available under a composite_model subgroup."""
        with self._open() as f:
            g = f.get(f"composite_model/{kind}")
            return sorted(g.keys()) if g is not None else []

    # ---- canonical solvent atom ordering ----
    def solvent_atomic_numbers(self, solvent):
        """Return one solvent molecule's atomic numbers, in the canonical atom order."""
        with self._open() as f:
            return np.asarray(f["solvents"][solvent]["atomic_numbers"][()])

    # ---- MD solvent-shell geometries (the large companion file) ----
    def available_md_shells(self, solute, solvent):
        """List the shell sizes with MD geometries for one solute/solvent pair."""
        with self._open_md() as f:
            return sorted(int(k.split("_")[1]) for k in f[solute][solvent].keys()
                          if k.startswith("geometries_"))

    def md_geometry(self, solute, solvent, shell):
        """Solvent-shell snapshot coordinates (frames, n_shell_atoms, 3) for one shell. The frame
        axis aligns 1:1 with magnet_x(solute, solvent, shell)."""
        with self._open_md() as f:
            return _decode(f[solute][solvent][f"geometries_{shell}"][()])

    def md_geometry_atomic_numbers(self, solute, solvent, shell):
        """Atomic numbers for one MD solvent-shell snapshot's atoms."""
        with self._open_md() as f:
            return np.asarray(f[solute][solvent][f"atomic_numbers_{shell}"][()])

    def md_stationary_geometry(self, solute):
        """The solute's stationary-geometry coordinates from the OpenMM setup."""
        with self._open_md() as f:
            return _decode(f[solute]["openMM_stationary_geometry"][()])

    # ---- experiment ----
    @staticmethod
    def _prefix_h_sites(df):
        """Give each proton site a unique label per solute. The spreadsheet labels every H
        site simply "H"; this prepends an incrementing two-digit number in row order per
        solute ("01 H", "02 H", ...), matching the figure notebooks so per-site H rows are
        distinguishable. Carbon sites already carry distinct labels and are left untouched."""
        df = df.copy()
        for solute in df["solute"].unique():
            n = 1
            mask = (df["solute"] == solute) & (df["nucleus"] == "H")
            for idx in df[mask].index:
                df.at[idx, "site"] = f"{n:02d} {df.at[idx, 'site']}"
                n += 1
        return df

    def experiment(self, rename_water_to_tip4p=True, uuid_prefix_h_sites=False):
        """Flat long table from the per-solute-sheet xlsx: one row per (solute, site).
        With uuid_prefix_h_sites=True, proton sites are made unique per solute (see
        _prefix_h_sites) as the figure notebooks require."""
        if not self.experimental_xlsx:
            raise ValueError("experimental_xlsx path not provided")
        sheets = pd.read_excel(self.experimental_xlsx, sheet_name=None)
        frames = []
        for solute, df in sheets.items():
            df = df.copy()
            df.insert(0, "solute", solute)
            frames.append(df)
        out = pd.concat(frames, ignore_index=True)
        if rename_water_to_tip4p and "water" in out.columns:
            out = out.rename(columns={"water": "TIP4P"})
        if uuid_prefix_h_sites:
            out = self._prefix_h_sites(out)
        return out

    def site_atom_table(self, uuid_prefix_h_sites=False):
        """Mapping of each (solute, nucleus, site) to its atom_numbers, used to aggregate
        per-atom shieldings up to NMR sites. `atom_numbers` is returned as the raw string from
        the spreadsheet (e.g. "17,19" or "21"); callers split/offset it themselves (atoms are
        1-based in the spreadsheet, so subtract 1 to index the shielding arrays)."""
        exp = self.experiment(rename_water_to_tip4p=False, uuid_prefix_h_sites=uuid_prefix_h_sites)
        cols = ["solute", "site", "nucleus", "atom_numbers"]
        return exp[cols].drop_duplicates().reset_index(drop=True)
