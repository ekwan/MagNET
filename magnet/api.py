"""The MagNET user-facing API.

Four entry points, each taking one molecule (`atomic_numbers` shaped `(N,)` and `coordinates` shaped
`(N, 3)`) or a list of molecules, and returning the matching shape:

| function | what you get |
|---|---|
| `predict_shifts` | <sup>1</sup>H and <sup>13</sup>C chemical shifts (ppm) in a specific solvent using MagNET-Zero/MagNET-PCM |
| `predict_shieldings` | <sup>1</sup>H and <sup>13</sup>C shieldings (ppm) in the gas phase |
| `implicit_solvent_correction` | shieldings(PCM=chloroform) - shieldings(gas phase) |
| `explicit_solvent_correction` | shieldings(solute+solvents) - shieldings(solute) |

**Shared options** (all four functions):

- `n_passes` (default `10`): average out equivariance error over `n_passes` forward passes
- `symmetrize` (default `True`): if True, average over `n_passes` on the input geometry and `n_passes` on the mirror image of the input geometry
- `device` (default `None`): where to run, a torch device or a string like `"cpu"` or `"cuda"`; uses GPU if available
- `checkpoints_dir` (default `None`): directory holding the released weights, if they are not in the current directory or the repo; may be the `model_checkpoints/` folder or its parent, and takes precedence over both
"""
__docformat__ = "google"

from collections import Counter

import numpy as np

from . import run_magnet
from . import scaling

_MODELS = {
    "MagNET": run_magnet.compute_MagNET_foundation_shieldings,
    "MagNET-Zero": run_magnet.compute_MagNET_Zero_shieldings,
}

# Atomic-number counts in one molecule of each MagNET-x solvent, used to validate that the
# non-solute atoms really are a whole number of molecules of the declared solvent.
_SOLVENT_COMPOSITION = {
    "chloroform": Counter({6: 1, 1: 1, 17: 3}),        # CHCl3
    "benzene": Counter({6: 6, 1: 6}),                  # C6H6
    "methanol": Counter({6: 1, 8: 1, 1: 4}),           # CH3OH
    "water": Counter({8: 1, 1: 2}),                    # H2O (run as TIP4P)
}


def _check_passes(n_passes):
    if not (isinstance(n_passes, (int, np.integer)) and n_passes >= 1):
        raise ValueError(f"n_passes must be a positive integer; got {n_passes!r}")


def _as_batch(atomic_numbers, coordinates):
    """Normalize a single molecule or a list of molecules to parallel lists. Returns
    (atomic_numbers_list, coordinates_list, was_single). Validates matching atom counts."""
    if len(coordinates) == 0:
        raise ValueError("empty input; pass one molecule ((N,) and (N, 3)) or a non-empty list")
    single = np.ndim(np.asarray(coordinates[0])) == 1     # (N, 3): first row is length-3 -> single
    if single:
        an_list = [np.asarray(atomic_numbers)]
        xyz_list = [np.asarray(coordinates, dtype=float)]
    else:
        an_list = [np.asarray(a) for a in atomic_numbers]
        xyz_list = [np.asarray(c, dtype=float) for c in coordinates]
    if len(an_list) != len(xyz_list):
        raise ValueError(f"got {len(an_list)} atomic_numbers array(s) but {len(xyz_list)} geometry(ies)")
    for a, c in zip(an_list, xyz_list):
        if a.ndim != 1 or c.ndim != 2 or c.shape[1] != 3 or a.shape[0] != c.shape[0]:
            raise ValueError("each molecule needs atomic_numbers shaped (N,) and coordinates shaped "
                             f"(N, 3) with the same N; got {a.shape} and {c.shape}")
    return an_list, xyz_list, single


def predict_shifts(atomic_numbers, coordinates, solvent="chloroform", n_passes=10, symmetrize=True,
                   device=None, return_components=False, checkpoints_dir=None):
    """Predict the <sup>1</sup>H and <sup>13</sup>C chemical shifts of a molecule in a solvent.

    This is the main entry point. It runs the MagNET-Zero and MagNET-PCM models and applies the paper's
    per-solvent calibration, so the output is directly comparable to an experimental spectrum. Needs an
    AIMNet2-optimized geometry.

    Args:
        atomic_numbers: element numbers, shape `(N,)`, for one molecule; or a list of such arrays.
        coordinates: xyz positions in Angstrom, shape `(N, 3)`; or a list of them.
        solvent: one of the 12 calibrated solvents: `"tetrahydrofuran"`, `"dichloromethane"`,
            `"chloroform"`, `"toluene"`, `"benzene"`, `"chlorobenzene"`, `"acetone"`,
            `"dimethylsulfoxide"`, `"acetonitrile"`, `"trifluoroethanol"`, `"methanol"`, `"water"`.
        n_passes: average out equivariance error over `n_passes` forward passes (default `10`).
        symmetrize: if True, also average over the mirror image of the input geometry (default `True`),
            doubling the passes.
        device: where to run, a torch device or a string like `"cpu"` or `"cuda"` (default `None`);
            uses GPU if available.
        return_components: return the numbers behind each shift (see Returns) instead of just the shifts.
        checkpoints_dir: directory holding the released weights, if they are not in the current
            directory or the repo (default `None`); may be the `model_checkpoints/` folder or its
            parent, and takes precedence over both defaults.

    Returns:
        One chemical shift in ppm per atom, shape `(N,)`, with NaN at any atom that is not
        <sup>1</sup>H or <sup>13</sup>C. Pass a list of molecules and you get a list of arrays back.

        With `return_components=True` you get a dict instead (or a list of dicts):

        - `shifts`: the shifts, as above.
        - `zero_shielding`: the MagNET-Zero gas-phase shielding at each atom.
        - `pcm_correction`: the MagNET-PCM solvent correction at each atom.
        - `coefficients`: the calibration used, as
          `{"H": {"intercept": ..., "stationary": ..., "pcm": ...}, "C": {...}}`, so that
          `shift = intercept + stationary * zero_shielding + pcm * pcm_correction`.
    """
    _check_passes(n_passes)
    tables = scaling.published_scaling_tables()
    key = "TIP4P" if solvent == "water" else solvent
    if key not in tables["C"] or key not in tables["H"]:
        options = ["water" if s == "TIP4P" else s for s in tables["C"]]
        raise ValueError(f"unknown solvent {solvent!r}; choose one of {options}")

    an_list, xyz_list, single = _as_batch(atomic_numbers, coordinates)
    zero = run_magnet.compute_MagNET_Zero_shieldings(an_list, xyz_list, n_passes=n_passes,
                                                     symmetrize=symmetrize, device=device,
                                                     checkpoints_dir=checkpoints_dir)
    pcm = run_magnet.compute_MagNET_PCM_corrections(an_list, xyz_list, n_passes=n_passes,
                                                    symmetrize=symmetrize, device=device,
                                                    checkpoints_dir=checkpoints_dir)
    results = []
    for atoms, sigma, delta in zip(an_list, zero, pcm):
        out = np.full(atoms.shape, np.nan, dtype=float)
        for nucleus, atomic_number in (("H", 1), ("C", 6)):
            mask = atoms == atomic_number
            if mask.any():
                out[mask] = scaling.predict_shift(tables[nucleus], key, sigma[mask], delta[mask])
        if return_components:
            results.append({"shifts": out, "zero_shielding": sigma, "pcm_correction": delta,
                            "coefficients": {"H": dict(tables["H"][key]), "C": dict(tables["C"][key])}})
        else:
            results.append(out)
    return results[0] if single else results


def predict_shieldings(atomic_numbers, coordinates, model="MagNET", n_passes=10, symmetrize=True,
                       device=None, checkpoints_dir=None):
    """Predict gas-phase NMR shielding constants for a molecule.

    A shielding constant is what the network outputs directly, before it is calibrated into a chemical
    shift; if you want shifts, use `predict_shifts`. For solvent effects use
    `implicit_solvent_correction` or `explicit_solvent_correction`.

    Args:
        atomic_numbers: element numbers, shape `(N,)`; or a list of such arrays.
        coordinates: xyz positions in Angstrom, shape `(N, 3)`; or a list of them.
        model: `"MagNET"`, the general foundation model, or `"MagNET-Zero"`, which is more accurate but
            expects an AIMNet2-optimized geometry.
        n_passes: average out equivariance error over `n_passes` forward passes (default `10`).
        symmetrize: if True, also average over the mirror image of the input geometry (default `True`),
            doubling the passes.
        device: where to run, a torch device or a string like `"cpu"` or `"cuda"` (default `None`);
            uses GPU if available.
        checkpoints_dir: directory holding the released weights, if they are not in the current
            directory or the repo (default `None`); may be the `model_checkpoints/` folder or its
            parent, and takes precedence over both defaults.

    Returns:
        One shielding constant in ppm per atom, shape `(N,)`. The models are trained on <sup>1</sup>H
        and <sup>13</sup>C, so the values are meaningful only at hydrogen and carbon atoms; atoms of
        other elements come back as `0.0`, not a prediction. (`predict_shifts` instead returns `NaN`
        there.) Pass a list of molecules and you get a list of arrays back.
    """
    redirect = {"MagNET-PCM": "implicit_solvent_correction", "MagNET-x": "explicit_solvent_correction"}
    if model in redirect:
        raise ValueError(f"{model!r} is a correction model; use magnet.{redirect[model]}(...), "
                         f"not predict_shieldings().")
    if model not in _MODELS:
        raise ValueError(f"model must be one of {list(_MODELS)}; got {model!r}")
    _check_passes(n_passes)
    an_list, xyz_list, single = _as_batch(atomic_numbers, coordinates)
    out = _MODELS[model](an_list, xyz_list, n_passes=n_passes, symmetrize=symmetrize, device=device,
                         checkpoints_dir=checkpoints_dir)
    return out[0] if single else out


def implicit_solvent_correction(atomic_numbers, coordinates, n_passes=10, symmetrize=True, device=None,
                                checkpoints_dir=None):
    """Predict how a solvent changes a molecule's shieldings, using a fast continuum-solvent model.

    The result is a per-atom shielding change: add it to a `predict_shieldings(..., model="MagNET-Zero")`
    value to get the solvated shielding, or just use `predict_shifts`, which does this for you. Uses
    MagNET-PCM, which only predicts the chloroform correction, so there is no solvent argument here;
    `predict_shifts` is what reuses this one correction for other solvents, through its per-solvent
    calibration. Needs an AIMNet2-optimized geometry.

    Args:
        atomic_numbers: element numbers, shape `(N,)`; or a list of such arrays.
        coordinates: xyz positions in Angstrom, shape `(N, 3)`; or a list of them.
        n_passes: average out equivariance error over `n_passes` forward passes (default `10`).
        symmetrize: if True, also average over the mirror image of the input geometry (default `True`),
            doubling the passes.
        device: where to run, a torch device or a string like `"cpu"` or `"cuda"` (default `None`);
            uses GPU if available.
        checkpoints_dir: directory holding the released weights, if they are not in the current
            directory or the repo (default `None`); may be the `model_checkpoints/` folder or its
            parent, and takes precedence over both defaults.

    Returns:
        One shielding change in ppm per atom, shape `(N,)`. Meaningful only at hydrogen and carbon
        atoms; atoms of other elements come back as `0.0`. Pass a list of molecules and you get a list
        of arrays back.
    """
    _check_passes(n_passes)
    an_list, xyz_list, single = _as_batch(atomic_numbers, coordinates)
    out = run_magnet.compute_MagNET_PCM_corrections(an_list, xyz_list, n_passes=n_passes,
                                                    symmetrize=symmetrize, device=device,
                                                    checkpoints_dir=checkpoints_dir)
    return out[0] if single else out


def _validate_solvent(solvent, solvent_atomic_numbers):
    """Confirm the non-solute atoms really are whole molecules of `solvent`, one after another.

    The solvent atoms must be listed one molecule at a time: all atoms of the first molecule, then
    all atoms of the next, and so on. This splits them into consecutive groups of the solvent's
    molecule size and checks each group has the right number of each element (the order of atoms
    within a group does not matter). Listing them this way is required by the model, which finds the
    solvent molecules by splitting the atom list into equal-size groups the same way, and uses that
    to drop solvent molecules that sit far from the solute."""
    if solvent not in _SOLVENT_COMPOSITION:
        raise ValueError(f"unknown solvent {solvent!r}; expected one of {sorted(_SOLVENT_COMPOSITION)}")
    atoms = np.asarray(solvent_atomic_numbers).tolist()
    if not atoms:
        raise ValueError("no solvent atoms; solute_atoms cannot name every atom")
    per_molecule = _SOLVENT_COMPOSITION[solvent]
    size = run_magnet.N_ATOMS_PER_SOLVENT[solvent]
    if len(atoms) % size != 0:
        raise ValueError(f"got {len(atoms)} non-solute atoms, not a whole number of {solvent} "
                         f"molecules ({size} atoms each); check solute_atoms and solvent")
    for start in range(0, len(atoms), size):
        group = Counter(atoms[start:start + size])
        if group != per_molecule:
            raise ValueError(f"the non-solute atoms are not whole {solvent} molecules listed one at "
                             f"a time: the group of {size} atoms starting at position {start} has "
                             f"element counts {dict(group)}, expected {dict(per_molecule)}. List each "
                             f"solvent molecule's atoms together.")


def _explicit_one(atoms, xyz, solute_atoms, solvent):
    """Validate solute_atoms and the solvent composition, then reorder one snapshot to
    solute-first-then-solvent. Returns (solute_atomic_numbers, reordered_atomic_numbers,
    reordered_coordinates)."""
    n = len(atoms)
    solute = np.asarray(solute_atoms, dtype=int)
    solute = np.where(solute < 0, solute + n, solute)          # normalize negative indices
    if solute.size == 0:
        raise ValueError("solute_atoms is empty; name at least one solute atom")
    if np.any(solute < 0) or np.any(solute >= n):
        raise ValueError(f"solute_atoms has an index out of range for a {n}-atom system")
    if len(set(solute.tolist())) != solute.size:
        raise ValueError("solute_atoms has duplicate indices")
    solute_set = set(solute.tolist())
    solvent_idx = np.array([i for i in range(n) if i not in solute_set], dtype=int)
    _validate_solvent(solvent, atoms[solvent_idx])
    order = np.concatenate([solute, solvent_idx])              # solute first, then solvent blocks
    return atoms[solute], atoms[order], xyz[order]


def explicit_solvent_correction(atomic_numbers, coordinates, solute_atoms, solvent="chloroform",
                                n_passes=10, symmetrize=True, device=None,
                                solvent_distance_threshold=12.0, checkpoints_dir=None):
    """Predict a solvent's effect on shieldings from an MD snapshot with explicit solvent molecules.

    Uses MagNET-x. Corrections are returned for solute atoms only. Corrections should be averaged
    over multiple frames.

    The expected atom ordering is [solute, n x solvents].

    Args:
        atomic_numbers: element numbers for the whole solute + solvent system, shape `(N,)`; or a list
            of snapshots.
        coordinates: xyz positions in Angstrom, shape `(N, 3)`; or a list of them.
        solute_atoms: which atoms are the solute, as 0-based indices (a list or array of ints;
            negative indices count from the end, like Python lists); the rest are treated as solvent.
        solvent: `"chloroform"`, `"benzene"`, `"methanol"`, or `"water"`, the four MagNET-x supports
            (default `"chloroform"`).
        n_passes: average out equivariance error over `n_passes` forward passes (default `10`).
        symmetrize: if True, also average over the mirror image of the input geometry (default `True`),
            doubling the passes.
        device: where to run, a torch device or a string like `"cpu"` or `"cuda"` (default `None`);
            uses GPU if available.
        solvent_distance_threshold: solvent molecules whose nearest atom is farther than this many
            Angstrom from the solute are dropped automatically (default 12.0), so extra solvent in the
            snapshot is harmless.
        checkpoints_dir: directory holding the released weights, if they are not in the current
            directory or the repo (default `None`); may be the `model_checkpoints/` folder or its
            parent, and takes precedence over both defaults.

    Returns:
        One correction in ppm per solute atom, in the order you listed them in `solute_atoms`. Pass a
        list of snapshots and you get a list of arrays back.
    """
    if solvent not in run_magnet.N_ATOMS_PER_SOLVENT:
        raise ValueError(f"solvent must be one of {sorted(run_magnet.N_ATOMS_PER_SOLVENT)}; "
                         f"got {solvent!r}")
    _check_passes(n_passes)
    an_list, xyz_list, single = _as_batch(atomic_numbers, coordinates)
    solute_an_list, full_an_list, full_xyz_list = [], [], []
    for atoms, xyz in zip(an_list, xyz_list):
        solute_an, full_an, full_xyz = _explicit_one(atoms, xyz, solute_atoms, solvent)
        solute_an_list.append(solute_an)
        full_an_list.append(full_an)
        full_xyz_list.append(full_xyz)
    out = run_magnet.compute_MagNET_x_corrections(
        solvent, solute_an_list, full_an_list, full_xyz_list,
        n_passes=n_passes, symmetrize=symmetrize, device=device,
        solvent_distance_threshold=solvent_distance_threshold, checkpoints_dir=checkpoints_dir)
    return out[0] if single else out
