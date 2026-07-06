"""Reader for magnet_test_predictions.hdf5 - MagNET's raw per-atom predictions and DFT targets
on every benchmark test set.

This is the inference behind SI Tables S3 and S4 (proton and carbon performance of MagNET) and
SI Table S5 (quasi-classical-dynamics corrections). Each test set was run through several models and
geometries; this file stores, for every molecule, the predicted shieldings and the DFT reference so
the accuracy statistics can be recomputed exactly.

The file is a set of groups, one per original result file. List them with `available()`. Group names
carry a solvent field only for the solvated MD sets:
    <testset>_<nucleus>_<model>_<extra>                   (gas-phase and QCD sets)
    <testset>_<solvent>_<nucleus>_<model>_<extra>         (solutesmd500 / solutesmd500isolated)
for example  gasphaseinternal_C_pretrained_C_vib1  or  solutesmd500isolated_chloroform_H_chloroform_H.

Test sets:
    gasphaseinternal      internal GDB test set, gas phase
    gasphasedft8k         external DFT8K benchmark (Guan/Paton), gas phase
    qcdtraj2500           ~2500 molecules with quasi-classical dynamics trajectories
    solutesmd500          500 solutes, solvated poses from classical MD
    solutesmd500isolated  the same 500 solutes, isolated (single-molecule) poses
    solutesmd50super      50 "super" test solutes (solvent-correction outputs)
    solutesmd8superopenmm 8 "super" solutes (OpenMM)

Models:
    pretrained            MagNET foundation model (trained on vibrated geometries)
    pretrained_stationary MagNET trained only on stationary geometries (the Figure S10 ablation)
    chloroform/benzene/methanol/tip4p   MagNET-x solvent-specific models

Geometry suffix on the gas-phase sets:
    vib0                  stationary (equilibrium) test geometry
    vib1                  vibrated test geometry

Each group loads back with `load()` to the original {molecule_key: tuple} object. The tuple layout
differs by family (this matters for computing errors correctly):
    gasphaseinternal, gasphasedft8k : (prediction, target, atomic_numbers, mask)   mask selects the
                                      scored nucleus; the error is over prediction[mask]-target[mask].
    solutesmd500, solutesmd500isolated : (target, prediction)   already one nucleus, no mask.
    solutesmd50super : (molecule_name, smiles, prediction, target)   these are solvent CORRECTIONS.
    solutesmd8superopenmm : four float arrays of shape (n_frames, n_atoms); pairing is not documented,
                                      so use load() and interpret them yourself.
    qcdtraj2500 : (stationary_prediction, stationary_target, trajectory_prediction, trajectory_target)
                                      where the trajectory arrays are (n_replicates, n_frames, n_scored).
                                      The QCD *correction* (SI Table S5) is the trajectory mean minus the
                                      stationary value; use qcd_correction_stats(), NOT stats().

`stats(group)` gives the (median, MAE, RMSE, n) of the per-atom errors for the gas-phase and MD-solute
families (SI Tables S3/S4). For the QCD family use `qcd_correction_stats(group)` (SI Table S5). Both
raise on a family they do not handle rather than return a wrong number.
"""
import numpy as np
import h5py

SCALE = 1e4
NAN_SENTINEL = np.iinfo(np.int32).min


def _dec_float(a):
    a = np.asarray(a)
    return np.where(a == NAN_SENTINEL, np.nan, a / SCALE).astype(np.float32)


def available(path):
    """Return the sorted list of test-set group names in the file."""
    with h5py.File(path, "r") as h:
        return sorted(h.keys())


def _fetch(h, g, name):
    """Read dataset g/name, following a content-addressed dedup reference if present."""
    if name in g:
        return g[name][()]
    ref = g.attrs.get(name + "_ref")
    if ref is not None:
        return h[ref][()]
    raise KeyError(f"{name} missing in {g.name}")


def _decode_group(h, g):
    n = int(g.attrs["n"])
    tlen = int(g.attrs["tuple_len"])
    kk = g.attrs["key_kind"]
    if kk == "str":
        raw = g["keys"][()]
        keys = [k.decode() if isinstance(k, bytes) else str(k) for k in raw]
    else:
        raw = _fetch(h, g, "keys")
        if kk == "tuple":
            keys = [tuple(int(x) for x in row) for row in raw]
        else:
            keys = [int(x) for x in raw]

    cols = []
    for p in range(tlen):
        if f"p{p}_str" in g:
            s = g[f"p{p}_str"][()]
            cols.append(("str", [x.decode() if isinstance(x, bytes) else str(x) for x in s]))
            continue
        kind = g.attrs[f"p{p}_kind"]
        data = _fetch(h, g, f"p{p}_data")
        shapes = _fetch(h, g, f"p{p}_shapes")
        offs = _fetch(h, g, f"p{p}_offsets")
        if kind == "float":
            data = _dec_float(data)
        elif kind == "bool":
            data = data.astype(bool)
        else:
            data = data.astype(np.int64)
        vals = [data[offs[i]:offs[i + 1]].reshape(tuple(int(x) for x in shapes[i])) for i in range(n)]
        cols.append((kind, vals))
    return {keys[i]: tuple(cols[p][1][i] for p in range(tlen)) for i in range(n)}


def load(path, group):
    """Load one test-set group back to its original {molecule_key: tuple} dict."""
    with h5py.File(path, "r") as h:
        if group not in h:
            raise KeyError(f"{group!r} not in file; see available()")
        return _decode_group(h, h[group])


def _require_aligned(h, g, group, *positions):
    """The flat computation below pairs elements across columns by position, which is only equivalent
    to the per-molecule form when the columns share the same per-molecule boundaries. Every column
    stores those boundaries as `p{p}_offsets`; require them to match so malformed input (a molecule
    whose prediction/target/mask arrays are not the same length) is refused loudly instead of pairing
    the wrong atoms silently."""
    offs = [_fetch(h, g, f"p{p}_offsets") for p in positions]
    if not all(np.array_equal(offs[0], o) for o in offs[1:]):
        raise ValueError(f"{group} has misaligned per-molecule lengths across columns "
                         f"{positions}; the stored arrays are inconsistent. Use load() and inspect.")


def _flat_error_array(h, g, group):
    """Per-atom |prediction - target| for every scored atom in a group, computed straight from the
    flat column arrays without rebuilding the per-molecule dict. Each column is stored concatenated in
    molecule order with matching per-molecule boundaries (checked by _require_aligned), so a
    whole-column subtraction (plus the stored mask, for the gas-phase family) gives the per-atom errors
    for every recognized tuple layout. Returns None for an unrecognized layout, so the caller can
    refuse rather than guess."""
    tlen = int(g.attrs["tuple_len"])
    if tlen == 2:
        # (target, prediction), already one nucleus, no mask
        _require_aligned(h, g, group, 0, 1)
        target = _dec_float(_fetch(h, g, "p0_data"))
        pred = _dec_float(_fetch(h, g, "p1_data"))
        return np.abs(pred.ravel() - target.ravel())
    if tlen == 4 and "p3_str" not in g and g.attrs.get("p3_kind") == "bool":
        # (prediction, target, atomic_numbers, mask); the mask selects the scored nucleus
        _require_aligned(h, g, group, 0, 1, 3)
        pred = _dec_float(_fetch(h, g, "p0_data"))
        target = _dec_float(_fetch(h, g, "p1_data"))
        mask = _fetch(h, g, "p3_data").astype(bool)
        return np.abs(pred[mask] - target[mask])
    if tlen == 4 and "p0_str" in g:
        # (name, smiles, prediction, target) - solvent-correction predictions
        _require_aligned(h, g, group, 2, 3)
        pred = _dec_float(_fetch(h, g, "p2_data"))
        target = _dec_float(_fetch(h, g, "p3_data"))
        return np.abs(pred.ravel() - target.ravel())
    return None


def errors(path, group):
    """Flat array of per-atom absolute errors |prediction - target| for a per-atom group
    (gas-phase and MD-solute families - SI Tables S3/S4).

    Refuses the QCD family (raise), whose accuracy is a trajectory *correction*, not a per-atom
    shielding error - use qcd_correction_errors() for that. Also refuses any tuple layout it does not
    recognize (for example solutesmd8superopenmm, whose four float arrays have no documented pairing);
    load() that group and interpret it yourself.
    """
    if group.startswith("qcdtraj"):
        raise ValueError(f"{group} is a QCD trajectory set; its accuracy is a correction "
                         f"(SI Table S5). Use qcd_correction_errors()/qcd_correction_stats().")
    with h5py.File(path, "r") as h:
        if group not in h:
            raise KeyError(f"{group!r} not in file; see available()")
        e = _flat_error_array(h, h[group], group)
    if e is None:
        raise ValueError(f"{group} has an unrecognized tuple layout for per-atom errors; "
                         f"use load() and compute directly.")
    return e


def stats(path, group):
    """(median_ae, mae, rmse, n) over all scored atoms in a per-atom group - the SI Table S3/S4
    quantities. Raises for the QCD family (use qcd_correction_stats)."""
    d = errors(path, group)
    return float(np.median(d)), float(d.mean()), float((d ** 2).mean() ** 0.5), int(d.size)


def qcd_correction_errors(path, group):
    """Flat per-atom absolute errors in the QCD *correction* for a qcdtraj2500 group - the quantity
    behind SI Table S5. The correction is the trajectory-averaged shielding minus the stationary
    shielding; the error is |predicted_correction - DFT_correction|, per scored atom.

    Each molecule stores (stationary_prediction, stationary_target, trajectory_prediction,
    trajectory_target); the trajectory arrays are (n_replicates, n_frames, n_scored).
    """
    if not group.startswith("qcdtraj"):
        raise ValueError(f"{group} is not a QCD trajectory set; use errors()/stats().")
    d = load(path, group)
    parts = []
    for stat_pred, stat_target, traj_pred, traj_target in d.values():
        pred_corr = np.asarray(traj_pred).mean(axis=(0, 1)) - np.asarray(stat_pred)
        dft_corr = np.asarray(traj_target).mean(axis=(0, 1)) - np.asarray(stat_target)
        parts.append(np.abs(pred_corr - dft_corr))
    return np.concatenate(parts)


def qcd_correction_stats(path, group):
    """(median_ae, mae, rmse, n) of the QCD correction errors for a qcdtraj2500 group - SI Table S5."""
    e = qcd_correction_errors(path, group)
    return float(np.median(e)), float(e.mean()), float((e ** 2).mean() ** 0.5), int(e.size)


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "magnet_test_predictions.hdf5"
    names = available(p)
    print(f"{len(names)} groups.")
    print("Tables S3/S4 (per-atom error):")
    for g in ("gasphaseinternal_C_pretrained_C_vib1", "solutesmd500isolated_chloroform_H_chloroform_H"):
        if g in names:
            print(f"  {g}: median/MAE/RMSE/n =", stats(p, g))
    print("Table S5 (QCD correction):")
    for g in ("qcdtraj2500_H_pretrained_H", "qcdtraj2500_C_pretrained_C"):
        if g in names:
            print(f"  {g}: median/MAE/RMSE/n =", qcd_correction_stats(p, g))
