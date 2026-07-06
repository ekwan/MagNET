"""Tables S3 and S4: MagNET shielding accuracy across stationary, vibrated, and solvated geometries.

How well MagNET reproduces its DFT training reference (PBE0/pcSseg-1) for the foundation model and the
chloroform MagNET-x, on four test sets:

  stationary_internal   held-out GDB resting geometries         (sigma-shake test split)
  vibrated_internal     MD frames of those molecules            (gdb_qcd)
  vibrated_external     perturbed outside-benchmark geometries  (dft8k)
  isolated_chloroform   isolated-solute chloroform MD snapshots (sigma-fresh)

Two reproduction paths; the shipped notebook uses the first:
1. EXACT, from `data/magnet_test_predictions/`: raw per-atom predictions over the FULL test sets,
   computed once from the released checkpoints. `exact_stats_table` maps each (nucleus, model, test
   set) to its group and calls the reader's `stats()`; median/mean/RMSE reproduce exactly, no
   checkpoints needed.
2. FROM SCRATCH: `cases_*` assembles (atomic numbers, geometry, DFT shielding) test cases and
   `magnet_benchmark_run.py` runs a model over them. Needs the checkpoints and only samples each set.

RMSE note (path 2): median and mean AE reproduce from a modest sample; RMSE does not. For the vibrated
sets it is dominated by ~1-in-10,000 extreme MD frames (the CO-style breakdown of Figure S12), so the
published 13C RMSE (7.14 ppm) needs the full ~1.9M-frame set; path 1 sidesteps this. The
isolated-chloroform row runs in full even under path 2; its foundation 1H RMSE (0.30 ppm) vs MAE
(0.07 ppm) reflects a broad heavy tail of real deshielded geometries (not DFT failures), kept as-is.
The Figure S10 stationary-only ablation checkpoint is not released, so that figure is not reproduced.
"""
import numpy as np

from stats import summarize_errors

# --------------------------------------------------------------------------- published SI values
# (median absolute error, mean absolute error, RMSE), per (model, test set). From SI Tables S3/S4.
# Keys: model in {"MagNET", "MagNET-x"}; test set as named above.
PUBLISHED = {
    "1H": {
        ("MagNET", "stationary_internal"): (0.020946503, 0.030467307, 0.05941977),
        ("MagNET", "vibrated_internal"): (0.027399063, 0.039091997, 0.10485571),
        ("MagNET", "vibrated_external"): (0.03241539, 0.04796361, 0.11397923),
        ("MagNET", "isolated_chloroform"): (0.03216362, 0.071752764, 0.29437021),
        ("MagNET-x", "stationary_internal"): (0.02163124, 0.031185, 0.06033938),
        ("MagNET-x", "vibrated_internal"): (0.027683258, 0.039436683, 0.10467214),
        ("MagNET-x", "vibrated_external"): (0.023479462, 0.034395833, 0.06877531),
        ("MagNET-x", "isolated_chloroform"): (0.022384644, 0.030986413, 0.04612577),
    },
    "13C": {
        ("MagNET", "stationary_internal"): (0.23955917, 0.36742094, 1.98238429),
        ("MagNET", "vibrated_internal"): (0.31917763, 0.47851510, 7.13765463),
        ("MagNET", "vibrated_external"): (0.38428497, 0.57037073, 0.98834883),
        ("MagNET", "isolated_chloroform"): (0.36051940, 0.58715980, 1.22315871),
        ("MagNET-x", "stationary_internal"): (0.25761414, 0.39035153, 1.98721069),
        ("MagNET-x", "vibrated_internal"): (0.33805847, 0.50241500, 7.15089956),
        ("MagNET-x", "vibrated_external"): (0.32779312, 0.47457147, 0.81721424),
        ("MagNET-x", "isolated_chloroform"): (0.30544280, 0.43592300, 0.72233414),
    },
}

NUCLEUS_Z = {"1H": 1, "13C": 6}
TEST_SETS = ["stationary_internal", "vibrated_internal", "vibrated_external", "isolated_chloroform"]
MODELS = ["MagNET", "MagNET-x"]

# Table S5: QCD-correction statistics for the foundation MagNET, one row per nucleus. Computed over
# the qcdtraj2500 test set (2500 molecules) at PBE0/pcSseg-1. Published (median AE, MAE, RMSE):
PUBLISHED_S5 = {
    "1H": (0.01526165, 0.023047674, 0.04020983),
    "13C": (0.17198181, 0.26223338, 0.50198762),
}
QCD_GROUP = {"1H": "qcdtraj2500_H_pretrained_H", "13C": "qcdtraj2500_C_pretrained_C"}


def qcd_stats_table(predictions_path, decode_module):
    """Table S5: (median AE, MAE, RMSE, n) of the QCD-correction errors for the foundation MagNET,
    one row per nucleus, read from data/magnet_test_predictions/ via qcd_correction_stats."""
    rows = []
    for nucleus in ("1H", "13C"):
        med, mae, rmse, n = decode_module.qcd_correction_stats(predictions_path, QCD_GROUP[nucleus])
        rows.append(dict(model=f"MagNET ({nucleus})", median_ae=med, mae=mae, rmse=rmse, n=n))
    return rows

# --------------------------------------------------------------------------- exact reproduction
# from data/magnet_test_predictions/ (see the module docstring, path 1). That dataset names its
# groups after its own test-set families (gasphaseinternal, gasphasedft8k, solutesmd500isolated),
# which don't match the four TEST_SETS names above one-to-one, so this is the mapping between them.
TEST_PREDICTIONS_MODEL_GROUP = {"MagNET": "pretrained", "MagNET-x": "chloroform"}
TEST_PREDICTIONS_NUCLEUS = {"1H": "H", "13C": "C"}


def test_predictions_group(model, test_set, nucleus):
    """The data/magnet_test_predictions/ group name holding the raw predictions for one Table S3/S4
    row. `model` is "MagNET" or "MagNET-x", `test_set` is one of TEST_SETS, `nucleus` is "1H" or
    "13C". See the TEST_PREDICTIONS_* maps below for the underlying group-naming scheme."""
    n = TEST_PREDICTIONS_NUCLEUS[nucleus]
    m = TEST_PREDICTIONS_MODEL_GROUP[model]
    if test_set == "stationary_internal":
        return f"gasphaseinternal_{n}_{m}_{n}_vib0"
    if test_set == "vibrated_internal":
        return f"gasphaseinternal_{n}_{m}_{n}_vib1"
    if test_set == "vibrated_external":
        return f"gasphasedft8k_{n}_{m}_{n}_vib1"
    if test_set == "isolated_chloroform":
        return f"solutesmd500isolated_chloroform_{n}_{m}_{n}"
    raise ValueError(f"unknown test set {test_set!r}")


def exact_stats_table(predictions_path, decode_module):
    """One row per (nucleus, model, test set), with (median_ae, mae, rmse, n) read directly from the
    full released MagNET predictions -- no sampling, no live inference, so RMSE reproduces exactly
    too (see the module docstring, path 1). `decode_module` is
    data/magnet_test_predictions/magnet_test_predictions_reader.py; the caller imports and passes it
    in, so this module carries no path dependency on data/magnet_test_predictions/."""
    rows = []
    for nucleus in NUCLEUS_Z:
        for model in MODELS:
            for test_set in TEST_SETS:
                group = test_predictions_group(model, test_set, nucleus)
                med, mae, rmse, n = decode_module.stats(predictions_path, group)
                rows.append({"nucleus": nucleus, "model": model, "test_set": test_set,
                             "median_ae": med, "mae": mae, "rmse": rmse, "n": n})
    return rows

# The elements MagNET was trained on (matches SUPPORTED_ELEMENTS in the magnet package). Structures
# with any other element (e.g. the phosphorus in dft8k) are excluded from the test sets: the model
# refuses them, and they cannot be fairly scored anyway.
SUPPORTED_ELEMENTS = frozenset({1, 6, 7, 8, 9, 16, 17})


def filter_supported(atomic_numbers_list, geometries_list, dft_list):
    """Drop any structure whose atoms are not all in SUPPORTED_ELEMENTS. Returns the three lists,
    filtered in lockstep, plus the number dropped."""
    keep = [i for i, an in enumerate(atomic_numbers_list)
            if set(np.unique(an).tolist()) <= SUPPORTED_ELEMENTS]
    dropped = len(atomic_numbers_list) - len(keep)
    return ([atomic_numbers_list[i] for i in keep],
            [geometries_list[i] for i in keep],
            [dft_list[i] for i in keep], dropped)


# --------------------------------------------------------------------------- error statistics

def collect_abs_errors(predicted_list, dft_list, atomic_numbers_list, z):
    """Concatenate the absolute errors |predicted - dft| over all atoms with atomic number `z`,
    across a list of test cases. Atoms where either value is NaN (not predicted or not computed) are
    dropped."""
    out = []
    for pred, dft, an in zip(predicted_list, dft_list, atomic_numbers_list):
        pred = np.asarray(pred, dtype=np.float64)
        dft = np.asarray(dft, dtype=np.float64)
        an = np.asarray(an)
        mask = (an == z) & np.isfinite(pred) & np.isfinite(dft)
        if mask.any():
            out.append(np.abs(pred[mask] - dft[mask]))
    return np.concatenate(out) if out else np.array([])


def summarize(abs_errors):
    """Median absolute error, mean absolute error, RMSE, and count for an array of absolute errors."""
    e = np.asarray(abs_errors, dtype=np.float64)
    if e.size == 0:
        return {"median_ae": float("nan"), "mae": float("nan"), "rmse": float("nan"), "n": 0}
    return summarize_errors(e, np.zeros_like(e))


def error_table(predicted_list, dft_list, atomic_numbers_list):
    """Summaries for both nuclei: {"1H": {...}, "13C": {...}}."""
    return {
        nucleus: summarize(collect_abs_errors(predicted_list, dft_list, atomic_numbers_list, z))
        for nucleus, z in NUCLEUS_Z.items()
    }


# --------------------------------------------------------------------------- test-case assembly
# Each function returns three parallel lists: atomic_numbers, geometry (n_atoms, 3), and the DFT
# reference shielding (n_atoms,), one entry per sampled structure. A model is then run over the
# geometries and compared with the DFT shieldings.

def cases_stationary_internal(sigma_shake, n_sample=2000, seed=0):
    """Resting geometries of held-out GDB molecules (sigma-shake test split)."""
    test_ids = sigma_shake.split("test")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(test_ids, size=min(n_sample, len(test_ids)), replace=False)
    ans, geos, dfts = [], [], []
    for mid in chosen:
        m = sigma_shake.by_id(int(mid))
        ans.append(m["atomic_numbers"])
        geos.append(m["stationary_coords"])
        dfts.append(m["shielding_stationary"])
    return ans, geos, dfts


def cases_vibrated_internal(gdb_qcd, n_sample=2000, frames_per_molecule=4, seed=0):
    """Molecular-dynamics frames of held-out GDB molecules (gdb_qcd)."""
    rng = np.random.default_rng(seed)
    mol_idx = rng.choice(gdb_qcd.n_molecules, size=min(n_sample, gdb_qcd.n_molecules), replace=False)
    ans, geos, dfts = [], [], []
    for i in mol_idx:
        m = gdb_qcd.molecule(int(i))
        nt, nf = m["coordinates"].shape[0], m["coordinates"].shape[1]
        for _ in range(frames_per_molecule):
            t, f = int(rng.integers(nt)), int(rng.integers(nf))
            ans.append(m["atomic_numbers"])
            geos.append(m["coordinates"][t, f])
            dfts.append(m["shieldings"][t, f])
    return ans, geos, dfts


def cases_vibrated_external(dft8k, n_sample=2000, seed=0):
    """Perturbed (non-stationary) geometries of the dft8k benchmark; geometry 0 is the reference, 1
    and 2 are perturbations, so the vibrated set uses geometries 1 and 2."""
    rng = np.random.default_rng(seed)
    n = dft8k.b3lyp.n_molecules
    chosen = rng.choice(n, size=min(n_sample, n), replace=False)
    ans, geos, dfts = [], [], []
    for i in chosen:
        m = dft8k.b3lyp.molecule(int(i))
        for g in (1, 2):
            ans.append(m["atomic_numbers"])
            geos.append(m["coordinates"][g])
            dfts.append(m["shielding"][g])
    return ans, geos, dfts


# The isolated-chloroform test set is defined entirely from the released sigma-fresh file, so the
# table is reproducible without any external file. The foundation model's RMSE on this row is large
# (0.30 ppm 1H) because the error is heavy-tailed, not because of a few bad frames: the optional
# scrub in isolated_chloroform_testset removes the most strongly-deshielded frames and the RMSE
# barely moves (0.31 -> 0.30). Those frames are real geometries, not DFT failures, so the default
# test set keeps them.

CHLOROFORM_OUTLIER_SIGMA = 5   # cutoff used by the MagNET-x training pipeline (train_MagNET_X.py)


def _chloroform_test_solutes(sigma_fresh):
    """Solute indices flagged as the held-out test split in the released sigma-fresh file. This
    matches the paper's chloroform test split exactly (534 solutes)."""
    group = sigma_fresh.f["chloroform"]
    out = []
    for name in group.keys():
        kind = group[name].attrs["type"]
        if isinstance(kind, bytes):
            kind = kind.decode()
        if kind == "test":
            out.append(int(name.split("_")[1]))
    return sorted(out)


def isolated_chloroform_testset(sigma_fresh, max_frame=10, n_sigma=CHLOROFORM_OUTLIER_SIGMA):
    """Assemble the chloroform isolated-solute test set for Tables S3/S4, from released data only.

    Each case is one isolated-solute snapshot: the solute geometry from a chloroform MD frame and the
    isolated (no-solvent) DFT shielding computed for it. The set is defined as:

      * test solutes : attrs['type'] == 'test' (the paper's held-out split; evaluating MagNET-x on
                       its training solutes would be data leakage).
      * frames       : frame_index 1..max_frame where the isolated shielding was actually computed.

    The published Tables S3/S4 use the full set (ans_all/geos_all/dfts_all). `flagged` marks the most
    strongly deshielded frames (~16-17 ppm isolated shielding, far-downfield O-H / N-H), found with the
    same +/- n_sigma cut the MagNET-x training pipeline applies (train_MagNET_X.py), recomputed here
    from the dataset's own shieldings. Their shieldings vary smoothly along each MD trajectory and sit
    on a continuous low-shielding tail, so this split is a diagnostic for documentation, not a
    correction: ans/geos/dfts (with those frames removed) exist only to show how much of the RMSE they
    carry.

    Returns a dict:
      ans, geos, dfts             -- test set with the deshielded-tail frames removed (diagnostic only)
      ans_all, geos_all, dfts_all -- the full published test set (reproduces Tables S3/S4)
      flagged                     -- sorted [(solute_index, frame_index), ...] in the deshielded tail
      bands                       -- {'isolated': (lo, hi), 'solvated': (lo, hi)} n_sigma bands (ppm)
    """
    frames = []
    iso_H, solv_H = [], []
    for si in _chloroform_test_solutes(sigma_fresh):
        s = sigma_fresh.solute("chloroform", si)
        ns = s["n_solute_atoms"]
        an = s["atomic_numbers"][:ns]                  # solute is the leading ns atoms
        is_h = an == 1
        iso = s["shielding_isolated"]                  # (n_frames, n_solute) ppm, NaN where not done
        solv = s["shielding_solvated"]
        coords = s["coordinates"]
        for row in range(min(max_frame, iso.shape[0])):    # row r -> frame_index r+1 (1-based)
            sh = iso[row]
            if not np.isfinite(sh).any():
                continue
            solv_row = solv[row]
            frames.append({"solute": si, "frame": row + 1, "an": an,
                           "geo": coords[row][:ns], "iso": sh, "solv": solv_row, "is_h": is_h})
            h = sh[is_h]
            iso_H.append(h[np.isfinite(h)])
            h = solv_row[is_h]
            solv_H.append(h[np.isfinite(h)])
    iso_H = np.concatenate(iso_H)
    solv_H = np.concatenate(solv_H)
    iso_band = (iso_H.mean() - n_sigma * iso_H.std(), iso_H.mean() + n_sigma * iso_H.std())
    solv_band = (solv_H.mean() - n_sigma * solv_H.std(), solv_H.mean() + n_sigma * solv_H.std())

    def is_outlier(fr):
        h = fr["iso"][fr["is_h"]]
        h = h[np.isfinite(h)]
        if ((h < iso_band[0]) | (h > iso_band[1])).any():
            return True
        h = fr["solv"][fr["is_h"]]
        h = h[np.isfinite(h)]
        return bool(h.size and ((h < solv_band[0]) | (h > solv_band[1])).any())

    out = {"ans": [], "geos": [], "dfts": [], "ans_all": [], "geos_all": [], "dfts_all": [],
           "flagged": [], "bands": {"isolated": iso_band, "solvated": solv_band}}
    for fr in frames:
        out["ans_all"].append(fr["an"])
        out["geos_all"].append(fr["geo"])
        out["dfts_all"].append(fr["iso"])
        if is_outlier(fr):
            out["flagged"].append((fr["solute"], fr["frame"]))
        else:
            out["ans"].append(fr["an"])
            out["geos"].append(fr["geo"])
            out["dfts"].append(fr["iso"])
    out["flagged"].sort()
    return out


def cases_isolated_chloroform(sigma_fresh, scrub=False, max_frame=10):
    """Isolated-solute snapshots from chloroform MD (sigma-fresh), as (ans, geos, dfts).

    Default (scrub=False) is the full published test set, which reproduces Tables S3/S4. scrub=True
    drops the strongly-deshielded-proton frames; that is a diagnostic, not a correction (those frames
    are real geometries, not DFT failures). See isolated_chloroform_testset.
    """
    t = isolated_chloroform_testset(sigma_fresh, max_frame=max_frame)
    return (t["ans"], t["geos"], t["dfts"]) if scrub else (t["ans_all"], t["geos_all"], t["dfts_all"])
