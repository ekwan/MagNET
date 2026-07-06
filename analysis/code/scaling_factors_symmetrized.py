"""Computes delta-22's MagNET-Zero / MagNET-PCM shieldings via live, symmetrized inference,
instead of reading the pre-baked (unsymmetrized, single-pass) values stored in delta22.hdf5.

Why: `data/delta22/delta22_reader.py::load_stationary_and_pcm_nn_shieldings` decodes
`nn_gas_shieldings`/`nn_pcm_corrections` straight out of the HDF5. Those were computed with a
single, unsymmetrized forward pass (see `magnet/run_magnet.py`'s module docstring: "The released
reference shieldings are NOT symmetrized"). Isotropic shielding is parity-even, so the SO(3)-only
backbone's un-symmetrized predictions carry a spurious reflection error that is not always
negligible, even on small molecules (delta-22's solutes are 6-22 atoms). `predict_shieldings(...,
symmetrize=True)` fixes this for free by averaging the prediction on the original geometry and its
mirror image.

This module produces a scaling-factor table meant for deployment (e.g. serving MagNET-Zero/PCM on
arbitrary "everyday" molecules), not a replacement for the SI's published Tables S10/S11: it is a
separate, corrected-methodology table for production use, computed the same way
`scaling_factors.build_scaling_tables(symmetrized=True)` does.
"""
import contextlib
import os
import time

import h5py
import numpy as np
import pandas as pd

from paths import repo_root, ensure_on_path, checkpoints_root

_REPO = repo_root(__file__)
# _REPO itself must be on sys.path too: run_magnet.py does `from magnet import ...`, which needs
# the magnet/ package's PARENT directory on sys.path, not just magnet/ itself. Without this, the
# import only works by accident when something else (e.g. `pip install -e .`, or pytest's rootdir
# handling) has already put _REPO on sys.path.
ensure_on_path(root=_REPO)
ensure_on_path("magnet", file=__file__)
ensure_on_path("data", "delta22", file=__file__)

import run_magnet  # noqa: E402
from run_magnet import compute_MagNET_Zero_shieldings, compute_MagNET_PCM_corrections  # noqa: E402
from delta22_reader import load_solutes, load_stationary_geometries  # noqa: E402

@contextlib.contextmanager
def _resolved_checkpoints():
    """Temporarily points run_magnet.MODEL_CHECKPOINTS at absolute paths, restoring the original
    (relative-path) dict afterward -- run_magnet.MODEL_CHECKPOINTS is a shared module-level dict,
    and other code importing run_magnet in the same process expects the original relative paths.
    Its values already start with "model_checkpoints/", so join each against the repo root."""
    base = os.path.dirname(checkpoints_root(required=True))
    original = run_magnet.MODEL_CHECKPOINTS
    run_magnet.MODEL_CHECKPOINTS = {
        key: os.path.join(base, rel_path) for key, rel_path in original.items()
    }
    try:
        yield
    finally:
        run_magnet.MODEL_CHECKPOINTS = original

_NN_COLUMNS = ["solute", "sap_geometry_type", "sap_nmr_method", "sap_basis",
               "solvent_model", "solvent", "shieldings", "geometry_time", "nmr_time"]


def _load_atomic_numbers(delta22_path, solute):
    with h5py.File(delta22_path, "r") as f:
        return np.array(f["solutes"][solute]["atomic_numbers"])


def _load_aimnet2_geometry_time(delta22_path, solute):
    with h5py.File(delta22_path, "r") as f:
        return float(np.array(f["solutes"][solute]["stationary_and_pcm"]["geometry_optimization_timings"])[0])


def compute_symmetrized_nn_shieldings_df(delta22_path, n_passes=10, device=None, verbose=True):
    """Same row shape as load_stationary_and_pcm_nn_shieldings (one "gas" + one "pcm_correction"
    row per solute), but the shieldings come from live predict_shieldings(symmetrize=True)
    inference instead of the HDF5's stored, unsymmetrized values.

    n_passes=10 with symmetrize=True means 10 forward passes on the original geometry and 10 on
    its mirror image, all 20 averaged together.

    Returns a DataFrame with the same 9 columns load_stationary_and_pcm_nn_shieldings produces;
    pass it as `nn_shieldings_override_df` to delta22_reader.load_delta22_nn_data.
    """
    solutes = load_solutes(delta22_path)
    atomic_numbers_list = [_load_atomic_numbers(delta22_path, s) for s in solutes]
    geometries_list = [load_stationary_geometries(delta22_path, s)["aimnet2"] for s in solutes]

    if verbose:
        print(f"computing symmetrized MagNET-Zero/PCM shieldings for {len(solutes)} solutes "
              f"(n_passes={n_passes}, symmetrize=True)...", flush=True)

    t0 = time.time()
    with _resolved_checkpoints():
        zero_shieldings = compute_MagNET_Zero_shieldings(
            atomic_numbers_list, geometries_list, n_passes=n_passes, symmetrize=True, device=device)
        t1 = time.time()
        pcm_corrections = compute_MagNET_PCM_corrections(
            atomic_numbers_list, geometries_list, n_passes=n_passes, symmetrize=True, device=device)
    t2 = time.time()

    if verbose:
        print(f"  MagNET-Zero: {t1-t0:.1f}s total, MagNET-PCM: {t2-t1:.1f}s total", flush=True)

    zero_time_per_solute = (t1 - t0) / len(solutes)
    pcm_time_per_solute = (t2 - t1) / len(solutes)

    rows = []
    for solute, gas, pcm in zip(solutes, zero_shieldings, pcm_corrections):
        geom_time = _load_aimnet2_geometry_time(delta22_path, solute)
        rows.append([solute, "aimnet2", "MagNET", "N/A", "gas", "none",
                     np.asarray(gas, dtype=float), geom_time, zero_time_per_solute])
        rows.append([solute, "aimnet2", "MagNET", "N/A", "pcm_correction", "chloroform",
                     np.asarray(pcm, dtype=float), geom_time, pcm_time_per_solute])

    return pd.DataFrame(rows, columns=_NN_COLUMNS)
