"""Worked examples for the MagNET model family.

Each helper loads the released checkpoints for one model and runs `predict_shieldings`. All of them
expose the two prediction options:

  - `n_passes`: average over this many forward passes to remove the per-pass frame noise (the MagNET
    paper uses 20).
  - `mirror_average`: also average over the molecule and its mirror image. Isotropic shielding is
    parity-even, but the SO(3)-only network does not enforce that, so symmetrizing removes a spurious
    reflection error that grows to ~0.3 ppm on 13C for large molecules. It costs one extra evaluation.
    The released reference shieldings are NOT mirror-averaged, so leave it False to reproduce them and turn
    it on for best accuracy on large molecules.

Geometries are the AIMNet2 (machine-learning-optimized) stationary geometries, in Angstrom.
"""
import os

import numpy as np
import torch

from magnet.model import MagNET_Lightning
from magnet.inference import (predict_shieldings, predict_shieldings_batch,
                              resolve_mirror_average)

MODEL_CHECKPOINTS = {
    # foundation model (predicts the gas-phase shielding the rovibrational/QCD analysis builds on)
    "MagNET_H": "model_checkpoints/MagNET/MagNET_1H.ckpt",
    "MagNET_C": "model_checkpoints/MagNET/MagNET_13C.ckpt",
    # MagNET-Zero: gas-phase shielding (WP04 for 1H, wB97X-D for 13C, pcSseg-2)
    "MagNET-Zero_H": "model_checkpoints/MagNET-Zero/MagNET-Zero_1H.ckpt",
    "MagNET-Zero_C": "model_checkpoints/MagNET-Zero/MagNET-Zero_13C.ckpt",
    # MagNET-PCM: B3LYP-D3(BJ)/pcSseg-2 gas-phase and chloroform-PCM shieldings; the correction is
    # their difference
    "MagNET-PCM-withPCM_H": "model_checkpoints/MagNET-PCM/MagNET-PCM-withPCM_1H.ckpt",
    "MagNET-PCM-withPCM_C": "model_checkpoints/MagNET-PCM/MagNET-PCM-withPCM_13C.ckpt",
    "MagNET-PCM-withoutPCM_H": "model_checkpoints/MagNET-PCM/MagNET-PCM-withoutPCM_1H.ckpt",
    "MagNET-PCM-withoutPCM_C": "model_checkpoints/MagNET-PCM/MagNET-PCM-withoutPCM_13C.ckpt",
    # MagNET-x: explicit-solvent shieldings, one model per solvent (see compute_MagNET_x_corrections).
    # The released checkpoint folder is "MagNET-x" (lowercase x); paths are case-sensitive on Linux.
    "MagNET-x-chloroform_H": "model_checkpoints/MagNET-x/MagNET-x-chloroform_1H.ckpt",
    "MagNET-x-chloroform_C": "model_checkpoints/MagNET-x/MagNET-x-chloroform_13C.ckpt",
    "MagNET-x-benzene_H": "model_checkpoints/MagNET-x/MagNET-x-benzene_1H.ckpt",
    "MagNET-x-benzene_C": "model_checkpoints/MagNET-x/MagNET-x-benzene_13C.ckpt",
    "MagNET-x-methanol_H": "model_checkpoints/MagNET-x/MagNET-x-methanol_1H.ckpt",
    "MagNET-x-methanol_C": "model_checkpoints/MagNET-x/MagNET-x-methanol_13C.ckpt",
    "MagNET-x-water_H": "model_checkpoints/MagNET-x/MagNET-x-water_1H.ckpt",
    "MagNET-x-water_C": "model_checkpoints/MagNET-x/MagNET-x-water_13C.ckpt",
}


def _default_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _resolve_checkpoint(path, checkpoints_dir=None):
    """Locate a checkpoint file given a MODEL_CHECKPOINTS-relative path like
    "model_checkpoints/MagNET/MagNET_1H.ckpt". Resolution order for a relative path:
      1. under `checkpoints_dir`, if given (it may be the model_checkpoints/ folder itself, or a
         directory that contains one); this takes precedence over the defaults below,
      2. as given (relative to the current directory),
      3. relative to the release repo root (the parent of this magnet/ folder), where
         model_checkpoints/ sits in the integrated Hugging Face checkout.
    Absolute paths pass through unchanged. If nothing is found, return the original so the loader
    raises a clear error on it."""
    if checkpoints_dir:
        # the stored path is "model_checkpoints/<rest>"; accept a checkpoints_dir that IS the
        # model_checkpoints/ folder (join <rest>) or one that CONTAINS it (join the whole path)
        rest = path[len("model_checkpoints/"):] if path.startswith("model_checkpoints/") else path
        for candidate in (os.path.join(checkpoints_dir, rest), os.path.join(checkpoints_dir, path)):
            if os.path.exists(candidate):
                return candidate
    if os.path.exists(path):
        return path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # parent of magnet/
    candidate = os.path.join(repo_root, path)
    if os.path.exists(candidate):
        return candidate
    return path


def load_model_to_device(path, device=None, checkpoints_dir=None):
    """Load one checkpoint onto `device`. `checkpoints_dir`, if given, is where to find the released
    weights: it may point at the model_checkpoints/ folder itself or at a directory that contains one,
    and takes precedence over the current directory and the repo root (see _resolve_checkpoint). Pass
    it when the weights live outside the repo; leave it None to use the in-repo/CWD defaults."""
    if device is None:
        device = _default_device()
    path = _resolve_checkpoint(path, checkpoints_dir=checkpoints_dir)
    return MagNET_Lightning.load_from_checkpoint(checkpoint_path=path, map_location=device).to(device).eval()


def _predict_with(key_H, key_C, atomic_numbers_list, geometries_list,
                  n_passes=20, mirror_average=False, symmetrize=None, device=None, checkpoints_dir=None, **predict_kwargs):
    """Load one model pair and run predict_shieldings over a list of molecules. Returns a list of
    per-atom shielding arrays. Extra keyword arguments are forwarded to predict_shieldings (used by
    the explicit-solvent path for the solute/solvent split)."""
    mirror_average = resolve_mirror_average(mirror_average, symmetrize, "_predict_with")
    if device is None:
        device = _default_device()
    model_H = load_model_to_device(MODEL_CHECKPOINTS[key_H], device, checkpoints_dir=checkpoints_dir)
    model_C = load_model_to_device(MODEL_CHECKPOINTS[key_C], device, checkpoints_dir=checkpoints_dir)
    # one batch across the molecules as well as their passes, rather than a call per molecule
    return [np.atleast_1d(one.squeeze()) for one in predict_shieldings_batch(
        model_H, model_C, list(atomic_numbers_list), list(geometries_list),
        device=device, n_passes=n_passes, mirror_average=mirror_average, **predict_kwargs)]


def compute_MagNET_foundation_shieldings(atomic_numbers_list, geometries_list,
                                         n_passes=20, mirror_average=False, symmetrize=None, device=None, checkpoints_dir=None):
    """MagNET foundation-model gas-phase shieldings (1H and 13C) for AIMNet2 geometries."""
    mirror_average = resolve_mirror_average(mirror_average, symmetrize, "compute_MagNET_foundation_shieldings")
    return _predict_with("MagNET_H", "MagNET_C", atomic_numbers_list, geometries_list,
                         n_passes=n_passes, mirror_average=mirror_average, device=device,
                         checkpoints_dir=checkpoints_dir)


def compute_MagNET_Zero_shieldings(atomic_numbers_list, geometries_list,
                                   n_passes=20, mirror_average=False, symmetrize=None, device=None, checkpoints_dir=None):
    """MagNET-Zero gas-phase shieldings (1H and 13C) for a list of AIMNet2 geometries.

    Args:
        atomic_numbers_list: list of (N,) integer atomic-number arrays.
        geometries_list: list of (N, 3) coordinate arrays, in Angstrom.
        n_passes, mirror_average: prediction options (see module docstring).
        checkpoints_dir: optional directory holding the released weights (see load_model_to_device).

    Returns a list of (N,) shielding arrays (1H at H atoms, 13C at C atoms).
    """
    mirror_average = resolve_mirror_average(mirror_average, symmetrize, "compute_MagNET_Zero_shieldings")
    return _predict_with("MagNET-Zero_H", "MagNET-Zero_C", atomic_numbers_list, geometries_list,
                         n_passes=n_passes, mirror_average=mirror_average, device=device,
                         checkpoints_dir=checkpoints_dir)


def compute_MagNET_PCM_corrections(atomic_numbers_list, geometries_list,
                                   n_passes=20, mirror_average=False, symmetrize=None, device=None, checkpoints_dir=None):
    """MagNET-PCM implicit-solvent (chloroform) corrections (1H and 13C) for AIMNet2 geometries.

    The correction is the difference between the PCM-corrected shielding and the gas-phase shielding,
    both from the B3LYP-D3(BJ)/pcSseg-2 MagNET-PCM checkpoints.

    Returns a list of (N,) correction arrays.
    """
    mirror_average = resolve_mirror_average(mirror_average, symmetrize, "compute_MagNET_PCM_corrections")
    gas = _predict_with("MagNET-PCM-withoutPCM_H", "MagNET-PCM-withoutPCM_C",
                        atomic_numbers_list, geometries_list,
                        n_passes=n_passes, mirror_average=mirror_average, device=device,
                        checkpoints_dir=checkpoints_dir)
    pcm = _predict_with("MagNET-PCM-withPCM_H", "MagNET-PCM-withPCM_C",
                        atomic_numbers_list, geometries_list,
                        n_passes=n_passes, mirror_average=mirror_average, device=device,
                        checkpoints_dir=checkpoints_dir)
    return [pcm[i] - gas[i] for i in range(len(pcm))]


# Atoms per solvent molecule, for the explicit-solvent split (water is stored/run as TIP4P).
N_ATOMS_PER_SOLVENT = {"chloroform": 5, "benzene": 12, "methanol": 6, "water": 3}


def compute_MagNET_x_corrections(solvent, solute_atomic_numbers_list, atomic_numbers_list,
                                 geometries_list, n_passes=20, mirror_average=False, symmetrize=None, device=None,
                                 solvent_distance_threshold=12.0, checkpoints_dir=None):
    """MagNET-x explicit-solvent corrections (solvated minus isolated) for one solvent.

    Args:
        solvent: 'chloroform' | 'benzene' | 'methanol' | 'water' (water uses the TIP4P model).
        solute_atomic_numbers_list: list of (n_solute,) solute atomic-number arrays.
        atomic_numbers_list: list of (n_total,) atomic numbers for the whole solute+solvent system.
            The solute atoms must come first, followed by complete solvent molecules in contiguous
            blocks of N_ATOMS_PER_SOLVENT[solvent] atoms.
        geometries_list: list of (n_total, 3) coordinate arrays (same atom order), in Angstrom.
        n_passes, mirror_average: prediction options (see module docstring).
        solvent_distance_threshold: drop solvent molecules whose nearest atom is beyond this many
            Angstrom of the solute (the released pipeline used 12.0). The isolated pass uses 0.0
            internally, which strips all solvent.
        checkpoints_dir: optional directory holding the released weights (see load_model_to_device).

    Returns a list of (n_solute,) correction arrays. The published pipeline averages this over MD
    frames; a single-frame value is only an estimate.
    """
    mirror_average = resolve_mirror_average(mirror_average, symmetrize, "compute_MagNET_x_corrections")
    if solvent not in N_ATOMS_PER_SOLVENT:
        raise ValueError(f"solvent must be one of {sorted(N_ATOMS_PER_SOLVENT)}; got {solvent!r}")
    n_per = N_ATOMS_PER_SOLVENT[solvent]
    if device is None:
        device = _default_device()
    model_H = load_model_to_device(MODEL_CHECKPOINTS[f"MagNET-x-{solvent}_H"], device, checkpoints_dir=checkpoints_dir)
    model_C = load_model_to_device(MODEL_CHECKPOINTS[f"MagNET-x-{solvent}_C"], device, checkpoints_dir=checkpoints_dir)
    out = []
    for solute_Z, full_Z, geom in zip(solute_atomic_numbers_list, atomic_numbers_list, geometries_list):
        kw = dict(model_H=model_H, model_C=model_C, solute_atomic_numbers=solute_Z, geometry=geom,
                  atomic_numbers=full_Z, N_atoms_per_solvent=n_per, device=device,
                  n_passes=n_passes, mirror_average=mirror_average)
        solvated = np.atleast_1d(predict_shieldings(solvent_distance_threshold=solvent_distance_threshold, **kw).squeeze())
        isolated = np.atleast_1d(predict_shieldings(solvent_distance_threshold=0.0, **kw).squeeze())
        out.append(solvated - isolated)
    return out


if __name__ == "__main__":
    atomic_numbers = np.array([6, 6, 6, 8, 8, 6, 1, 1, 1, 1, 1, 1])
    coordinates = np.array(
        [
            [-2.07273936, 0.71782219, -0.27169511],
            [-1.51154256, -0.47834334, -0.05051489],
            [-0.06133485, -0.72471339, 0.10392854],
            [0.38785097, -1.84332836, 0.30459866],
            [0.65684897, 0.41875002, -0.00514369],
            [2.0648613, 0.23182768, 0.13713977],
            [-1.48494101, 1.6266042, -0.3569932],
            [-3.14927745, 0.81001383, -0.37304503],
            [-2.12480974, -1.36963022, 0.03013895],
            [2.44585443, -0.42836204, -0.64841408],
            [2.30088663, -0.16782625, 1.12843716],
            [2.5483427, 1.20718563, 0.03357037],
        ]
    )

    foundation = compute_MagNET_foundation_shieldings([atomic_numbers], [coordinates])
    zero = compute_MagNET_Zero_shieldings([atomic_numbers], [coordinates])
    pcm = compute_MagNET_PCM_corrections([atomic_numbers], [coordinates])
    print("foundation gas-phase shieldings:", foundation[0])
    print("MagNET-Zero gas-phase shieldings:", zero[0])
    print("MagNET-PCM chloroform corrections:", pcm[0])
