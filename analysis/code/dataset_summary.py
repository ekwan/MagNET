"""SI Table S8 (Dataset Summary Statistics): molecule and 1H/13C site counts for the training datasets.

Site counts from each file's flat ``atomic_numbers`` (1H = 1, 13C = 6); molecule counts from each
file's per-molecule index. Vectorized reads, so seconds even on the multi-million-molecule files. The
MagNET-Zero rows combine the two sigma-pepper groups (initial PBE0/pcSseg-1, WP04/wB97X-D/pcSseg-2
fine-tuning) with sigma-concentrate.
"""
import os

import h5py
import pandas as pd

import paths

# Published SI Table S8 values, for the exact-reproduction check: label -> (molecules, 1H, 13C).
PUBLISHED_S8 = {
    "sigma-shake": (4_787_816, 72_175_813, 50_542_154),
    "MagNET-Zero (initial round)": (2_167_210, 24_047_999, 15_330_361),
    "MagNET-Zero (fine-tuning)": (370_105, 4_042_135, 2_557_211),
    "MagNET-Zero (both rounds combined)": (2_537_315, 28_090_134, 17_887_572),
    "sigma-concentrate": (50_000, 754_437, 528_569),
}


def _counts(node):
    """(molecules, 1H sites, 13C sites) for an open h5py file or group."""
    z = node["atomic_numbers"][:]
    n_1h = int((z == 1).sum())
    n_13c = int((z == 6).sum())
    if "molecule_id" in node:
        n_mol = node["molecule_id"].shape[0]
    elif "smiles_offsets" in node:            # offsets are n_molecules + 1
        n_mol = node["smiles_offsets"].shape[0] - 1
    elif "sigma_shake_id" in node:
        n_mol = node["sigma_shake_id"].shape[0]
    else:
        raise KeyError("node has no recognized per-molecule key "
                       "(molecule_id / smiles_offsets / sigma_shake_id)")
    return n_mol, n_1h, n_13c


def counts(path, group=None):
    """(molecules, 1H sites, 13C sites) for one HDF5 file, optionally within one group."""
    with h5py.File(path, "r") as f:
        return _counts(f[group] if group else f)


def summary_table(data_dir):
    """SI Table S8 as a DataFrame with columns dataset / molecules / n_1H_sites / n_13C_sites.

    data_dir is the release repo's ``data/`` directory; the three source files are read from their
    dataset subfolders. The two MagNET-Zero training rounds and their combination are assembled from
    the sigma-pepper groups plus sigma-concentrate.
    """
    # data_dir is the repo's data/ folder; its parent is the repo root the deposit fallback needs.
    root = os.path.dirname(os.path.normpath(data_dir))
    shake = paths.dataset_file("sigma-shake", root=root)
    pepper = paths.dataset_file("sigma-pepper", root=root)
    concentrate = paths.dataset_file("sigma-concentrate", root=root)

    sigma_shake = counts(shake)
    initial = counts(pepper, "pbe0_pcSseg1")                 # MagNET-Zero initial round
    pepper_ft = counts(pepper, "wp04_wb97xd_pcSseg2")        # fine-tuning: sigma-pepper part
    sigma_concentrate = counts(concentrate)                  # fine-tuning: sigma-concentrate part
    fine_tuning = tuple(a + b for a, b in zip(pepper_ft, sigma_concentrate))
    combined = tuple(a + b for a, b in zip(initial, fine_tuning))

    rows = [
        ("sigma-shake", sigma_shake),
        ("MagNET-Zero (initial round)", initial),
        ("MagNET-Zero (fine-tuning)", fine_tuning),
        ("MagNET-Zero (both rounds combined)", combined),
        ("sigma-concentrate", sigma_concentrate),
    ]
    return pd.DataFrame(
        [{"dataset": label, "molecules": m, "n_1H_sites": h, "n_13C_sites": c}
         for label, (m, h, c) in rows]
    )
