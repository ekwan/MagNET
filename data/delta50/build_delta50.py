"""Provenance/build script for delta50.hdf5.

Regenerates the shipped delta50.hdf5 from three raw sources (none of which are shipped, exactly as
for dft8k):

  1. the previous delta50.hdf5, for the MagNET predictions already packaged into it
     (nn_magnet_zero, nn_b3lyp, nn_b3lyp_pcm; these come from MagNET inference and are carried
     forward unchanged), plus the atom order, geometries, and atomic numbers;
  2. a directory of Gaussian NMR output files, one per molecule, each with two link jobs at the
     MagNET-Zero reference levels (WP04 for 1H, wB97X-D for 13C; pcSseg-2; gas), computed on the
     stored AIMNet2 geometries. Parsed with cctk. These become the per-atom DFT reference
     shieldings shielding_wp04_pcSseg2 and shielding_wb97xd_pcSseg2, so the MagNET-Zero-vs-DFT
     residual (Table S3/S4-style, the DELTA50 outlier check in the paper) reproduces from the file.
  3. DELTA50_benchmark.xlsx from the DELTA50 Supporting Information (Cohen et al., Molecules 2023,
     28, 2449; reference 47; CC BY 4.0), for the experimental 1H and 13C chemical shifts.

The experimental shifts are mapped onto this file's own atom order and stored per atom as
experimental_shift. DELTA50 numbers its atoms against independently built structures; those geometries
were recomputed here, so the numbering only sometimes lines up. Each molecule is mapped by DELTA50's
own atom numbering when that numbering is element-consistent and covers every H/C atom (47 of 50);
the three whose atom order differs (cyclohexanone, methyl acetate, 2-methyl-2-nitropropane) are
mapped by matching RDKit topological-symmetry classes to DELTA50 environments in order of the
DFT-predicted shift. The mapping is validated: coverage is 100% of H and C atoms, and the large
scaled-DFT-vs-experiment residuals are all chemically hard sp2 carbons (carbonyl, nitro, aromatic).

Usage:
    python3 build_delta50.py [SRC_HDF5] [OUT_DIR] [XLSX] [DST_HDF5]

with the defaults below. Requires numpy, h5py, pandas, cctk, and rdkit.
"""
from __future__ import annotations
import os
import re
import sys
from collections import defaultdict

import numpy as np
import h5py

SCALE = 1e-4
MARKER = -2147483648

_HOME = os.path.expanduser("~")
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = os.path.join(_HERE, "delta50.hdf5")
DEFAULT_OUT_DIR = os.path.join(_HOME, "research", "magnet", "delta50", "delta50",
                               "output", "wp04_wb97xd_pcSseg2")
DEFAULT_XLSX = os.path.join(_HOME, "research", "magnet", "delta50", "DELTA50_benchmark.xlsx")

# Molecules whose canonical name here differs from the one in the previous file and the raw inputs.
# DELTA50 (and the old file) mislabel 2-methyl-2-nitropropane, a C-nitro compound, as a nitrate ester.
RENAMES = {"t-butyl nitrate": "2-methyl-2-nitropropane"}

# DELTA50 spreadsheet compound names that differ from our molecule_names. "3,3-dimethyl-1-butene"
# is split across two cells ("(3,3-Dimethyl-" then "1-butene)"), so both fragments are aliased.
NAME_ALIASES = {
    "dmf": "n,n-dimethylformamide", "thf": "tetrahydrofuran", "dmac": "n,n-dimethylacetamide",
    "2cyanopropane": "isobutyronitrile", "mtbe": "methyl t-butyl ether",
    "cyclopentenone": "cyclopent-2-en-1-one", "cyclohexenone": "cyclohex-2-en-1-one",
    "pbenzoquinone": "1,4-benzoquinone", "thp": "tetrahydropyran",
    "33dimethyl": "t-butylethylene", "1butene": "t-butylethylene",
    "tbutylnitrate": "2-methyl-2-nitropropane",
}


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _encode(values):
    v = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(v)
    out = np.full(v.shape, MARKER, dtype=np.int64)
    out[finite] = np.round(v[finite] / SCALE).astype(np.int64)
    return out.astype(np.int32)


def _parse_atom_labels(s):
    """DELTA50 atom-number strings: comma lists ("5,6,7"), ranges ("7-12"), singles, and mixtures."""
    out = []
    for tok in re.split(r"[,\s]+", str(s).strip()):
        m = re.match(r"^(\d+)-(\d+)$", tok)
        if m:
            out += list(range(int(m.group(1)), int(m.group(2)) + 1))
        elif tok.isdigit():
            out.append(int(tok))
    return out


def load_predictions(src_hdf5):
    with h5py.File(src_hdf5, "r") as f:
        return {
            "molecule_names": [str(n) for n in f["molecule_names"].asstr()[:]],
            "n_atoms": f["n_atoms"][:].astype(np.int32),
            "atomic_numbers": f["atomic_numbers"][:],
            "coordinates": f["coordinates"][:],
            "nn_magnet_zero": f["nn_magnet_zero"][:],
            "nn_b3lyp": f["nn_b3lyp"][:],
            "nn_b3lyp_pcm": f["nn_b3lyp_pcm"][:],
        }


def load_dft_reference(out_dir, file_names, out_names, atomic_numbers, starts):
    """Per-atom WP04 and wB97X-D isotropic shieldings from the Gaussian outputs, aligned to the
    stored atom order (checked). Files are matched by `file_names` (the raw-input naming) but the
    returned per-molecule dict is keyed by `out_names` (the canonical output naming).
    Returns (wp04, wb97xd, per_molecule_shieldings)."""
    import cctk
    files = {_norm(f[len("delta50-"):-len(".out")]): os.path.join(out_dir, f)
             for f in os.listdir(out_dir) if f.endswith(".out")}
    # the Gaussian files keep the pre-rename naming; fall back through RENAMES so re-running the
    # build against its own (already-renamed) output still finds each file.
    reverse = {v: k for k, v in RENAMES.items()}

    def _find(name):
        for candidate in (name, reverse.get(name, name)):
            if _norm(candidate) in files:
                return files[_norm(candidate)]
        raise KeyError(f"no Gaussian output for {name!r} in {out_dir}")

    wp04 = np.full(starts[-1], np.nan)
    wb97xd = np.full(starts[-1], np.nan)
    per_mol = {}
    for i, (file_name, out_name) in enumerate(zip(file_names, out_names)):
        objs = cctk.GaussianFile.read_file(_find(file_name))
        by_level = {}
        for g in objs:
            ens = g.ensemble
            iso = np.asarray(ens.get_properties_dict(ens.molecules[-1])["isotropic_shielding"],
                             dtype=float)
            z = np.asarray(ens.molecules[-1].atomic_numbers)
            by_level["wb97xd" if "wb97xd" in g.route_card.lower() else "wp04"] = (z, iso)
        sl = slice(int(starts[i]), int(starts[i + 1]))
        z_ref = atomic_numbers[sl]
        for level, (z, iso) in by_level.items():
            if not np.array_equal(z, z_ref):
                raise ValueError(f"{out_name}: {level} atom order does not match the stored geometry")
        wp04[sl] = by_level["wp04"][1]
        wb97xd[sl] = by_level["wb97xd"][1]
        per_mol[out_name] = {"wp04": by_level["wp04"][1], "wb97xd": by_level["wb97xd"][1]}
    return wp04, wb97xd, per_mol


def _parse_experimental_sheets(xlsx, valid_names):
    """{molecule_name: {"H": [(atoms, shift)], "C": [...]}} from the two DELTA50 functional sheets."""
    import pandas as pd
    valid = {_norm(n): n for n in valid_names}
    out = {n: {"H": [], "C": []} for n in valid_names}
    for sheet, nucleus in (("1H Functional", "H"), ("13C Functional", "C")):
        df = pd.read_excel(xlsx, sheet_name=sheet, header=None)
        hdr = next(r for r in range(df.shape[0]) if str(df.iat[r, 0]).strip() == "Compound")
        current = None
        for r in range(hdr + 1, df.shape[0]):
            compound, label, shift = df.iat[r, 0], df.iat[r, 1], df.iat[r, 2]
            if isinstance(compound, str) and compound.strip():
                k = _norm(compound)
                current = valid.get(_norm(NAME_ALIASES.get(k, compound)))
            if current is None or pd.isna(label) or pd.isna(shift):
                continue
            out[current][nucleus].append((_parse_atom_labels(label), float(shift)))
    return out


def _symmetry_classes(atomic_numbers, coordinates):
    """RDKit topological-symmetry classes for one molecule: {rank: [atom indices]}. Bonds are
    perceived from the geometry."""
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    pt = Chem.GetPeriodicTable()
    lines = [str(len(atomic_numbers)), ""]
    for z, xyz in zip(atomic_numbers, coordinates):
        lines.append(f"{pt.GetElementSymbol(int(z))} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}")
    mol = Chem.MolFromXYZBlock("\n".join(lines))
    rdDetermineBonds.DetermineBonds(mol, charge=0)
    classes = defaultdict(list)
    for atom_idx, rank in enumerate(Chem.CanonicalRankAtoms(mol, breakTies=False)):
        classes[rank].append(atom_idx)
    return classes


def map_experimental(xlsx, molecule_names, atomic_numbers, coordinates, starts, dft_per_mol):
    """Map DELTA50 experimental shifts onto our atom order; per-atom array (ppm, NaN where absent).

    Direct: use DELTA50's atom numbering where it is element-consistent and covers every H/C atom.
    Fallback: for molecules whose atom order differs, match RDKit symmetry classes to DELTA50
    environments by DFT-predicted shift order. Asserts full H/C coverage."""
    experimental = np.full(starts[-1], np.nan)
    fallback = []
    exp = _parse_experimental_sheets(xlsx, molecule_names)
    for i, name in enumerate(molecule_names):
        sl = slice(int(starts[i]), int(starts[i + 1]))
        z = atomic_numbers[sl]
        n = len(z)
        envs = {1: exp[name]["H"], 6: exp[name]["C"]}
        consistent = all(1 <= a <= n and z[a - 1] == zz
                         for zz in (1, 6) for atoms, _ in envs[zz] for a in atoms)
        # direct: use DELTA50's numbering. conflict = two environments claim one atom with different
        # shifts (a numbering error, e.g. overlapping ranges); route those to the fallback rather
        # than let a later write silently win.
        direct = np.full(n, np.nan)
        conflict = False
        if consistent:
            for zz in (1, 6):
                for atoms, shift in envs[zz]:
                    for a in atoms:
                        if np.isfinite(direct[a - 1]) and direct[a - 1] != shift:
                            conflict = True
                        direct[a - 1] = shift
        if consistent and not conflict and np.isfinite(direct[(z == 1) | (z == 6)]).all():
            experimental[sl] = direct
            continue
        # fallback: RDKit symmetry classes matched to DELTA50 environments by DFT-shift order
        fallback.append(name)
        classes = _symmetry_classes(z, coordinates[sl])
        assigned = np.full(n, np.nan)
        sigma = {1: dft_per_mol[name]["wp04"], 6: dft_per_mol[name]["wb97xd"]}
        for zz in (1, 6):
            element_classes = [atoms for atoms in classes.values() if z[atoms[0]] == zz]
            if len(element_classes) != len(envs[zz]):
                raise ValueError(f"{name}: {len(element_classes)} symmetry classes vs "
                                 f"{len(envs[zz])} DELTA50 environments for Z={zz}")
            by_sigma = sorted(element_classes, key=lambda a: np.mean([sigma[zz][j] for j in a]))
            by_shift = sorted(envs[zz], key=lambda e: -e[1])  # sigma ascending <-> shift descending
            for atoms, (_, shift) in zip(by_sigma, by_shift):
                for j in atoms:
                    assigned[j] = shift
        experimental[sl] = assigned

    hc = (atomic_numbers == 1) | (atomic_numbers == 6)
    missing = int((~np.isfinite(experimental[hc])).sum())
    if missing:
        raise ValueError(f"{missing} H/C atoms have no experimental shift")
    return experimental, fallback


def build(src_hdf5, out_dir, xlsx, dst_hdf5):
    pred = load_predictions(src_hdf5)
    file_names = pred["molecule_names"]                         # raw-input naming (files, sheets)
    names = [RENAMES.get(n, n) for n in file_names]             # canonical output naming
    starts = np.zeros(len(names) + 1, np.int64)
    np.cumsum(pred["n_atoms"], out=starts[1:])

    coords = pred["coordinates"].astype(np.float64) * SCALE  # int32 fixed point -> Angstrom
    wp04, wb97xd, dft_per_mol = load_dft_reference(out_dir, file_names, names,
                                                   pred["atomic_numbers"], starts)
    experimental, fallback = map_experimental(xlsx, names, pred["atomic_numbers"], coords, starts,
                                              dft_per_mol)
    print(f"experimental mapped by symmetry fallback: {fallback}")

    string_dt = h5py.string_dtype("utf-8")
    opts = dict(compression="gzip", shuffle=True)
    with h5py.File(dst_hdf5, "w") as f:
        f.attrs["dataset"] = "delta50"
        f.attrs["description"] = (
            "The 50 small molecules of the DELTA50 benchmark (Cohen et al., Molecules 2023, 28, "
            "2449; CC BY 4.0) on AIMNet2 geometries. Per-atom arrays: MagNET predictions, the DFT "
            "reference shieldings MagNET-Zero targets (WP04 for 1H, wB97X-D for 13C, pcSseg-2, gas), "
            "and the published experimental shifts mapped onto this atom order. The MagNET-Zero-vs-"
            "DFT residual and the experiment comparison both reproduce from this file.")
        f.attrs["n_molecules"] = len(names)
        f.attrs["geometry"] = "AIMNet2"
        f.attrs["scale"] = SCALE
        f.attrs["missing_value_marker"] = MARKER
        f.attrs["nn_magnet_zero"] = "MagNET-Zero: WP04 (1H) / wB97X-D (13C), pcSseg-2, 20-pass"
        f.attrs["nn_b3lyp"] = "MagNET-PCM gas-phase component: B3LYP/pcSseg-2"
        f.attrs["nn_b3lyp_pcm"] = "MagNET-PCM chloroform component: B3LYP/pcSseg-2 + chloroform PCM"
        f.attrs["shielding_wp04_pcSseg2"] = "DFT reference: WP04/pcSseg-2 gas (the 1H MagNET-Zero level)"
        f.attrs["shielding_wb97xd_pcSseg2"] = "DFT reference: wB97X-D/pcSseg-2 gas (the 13C MagNET-Zero level)"
        f.attrs["experimental_shift"] = (
            "Experimental 1H/13C shift (ppm, CDCl3, TMS reference) from DELTA50 (Cohen et al., "
            "Molecules 2023, 28, 2449; CC BY 4.0), mapped onto this atom order. NaN at atoms with no "
            "reported shift (heteroatoms).")

        f.create_dataset("molecule_names", data=np.array(names, dtype=object), dtype=string_dt)
        f.create_dataset("n_atoms", data=pred["n_atoms"], **opts)
        f.create_dataset("atomic_numbers", data=pred["atomic_numbers"], **opts)
        f.create_dataset("coordinates", data=pred["coordinates"], **opts)
        f.create_dataset("nn_magnet_zero", data=pred["nn_magnet_zero"], **opts)
        f.create_dataset("nn_b3lyp", data=pred["nn_b3lyp"], **opts)
        f.create_dataset("nn_b3lyp_pcm", data=pred["nn_b3lyp_pcm"], **opts)
        f.create_dataset("shielding_wp04_pcSseg2", data=_encode(wp04), **opts)
        f.create_dataset("shielding_wb97xd_pcSseg2", data=_encode(wb97xd), **opts)
        f.create_dataset("experimental_shift", data=_encode(experimental), **opts)

    return dst_hdf5


if __name__ == "__main__":
    args = sys.argv[1:]
    src = args[0] if len(args) > 0 else DEFAULT_SRC
    out_dir = args[1] if len(args) > 1 else DEFAULT_OUT_DIR
    xlsx = args[2] if len(args) > 2 else DEFAULT_XLSX
    dst = args[3] if len(args) > 3 else DEFAULT_SRC
    print(f"src predictions : {src}")
    print(f"DFT outputs     : {out_dir}")
    print(f"experimental    : {xlsx}")
    print(f"writing         : {dst}")
    build(src, out_dir, xlsx, dst)
    print("done.")
