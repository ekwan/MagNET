"""MagNET-Zero vs DFT residuals on the DFT8K benchmark (~7,000 organics, out of sample). No plotting.

Each nucleus is compared against the level MagNET-Zero reproduces: protons vs WP04/pcSseg-2, carbons
vs wB97X-D/pcSseg-2, gas phase on AIMNet2 geometries; residual = DFT shielding minus MagNET. Read via
data/dft8k/dft8k_reader.py. The figure notebook and tests import from here.
"""
import numpy as np

from stats import summarize_errors

# the DFT level each MagNET-Zero model reproduces, and the element it predicts
NUCLEI = {
    "H": {"atomic_number": 1, "dft": "wp04_pcSseg2", "nn": "nn_wp04_pcSseg2"},
    "C": {"atomic_number": 6, "dft": "wb97xd_pcSseg2", "nn": "nn_wb97xd_pcSseg2"},
}


def load_shieldings(dft8k_path):
    """The flat per-atom shielding arrays from the DFT8K AIMNet2 group, through the released
    reader. Returns the reader's all_shieldings dict (atomic_numbers plus the DFT and MagNET
    shieldings, each ppm, NaN where not computed)."""
    from dft8k_reader import DFT8k
    with DFT8k(dft8k_path) as ds:
        return ds.aimnet2.all_shieldings()


def residuals(shieldings, nucleus):
    """The DFT-minus-MagNET shielding residuals (ppm) for one nucleus, over the atoms of that
    element where MagNET makes a prediction. shieldings is the load_shieldings dict; nucleus is
    "H" or "C". Atoms the model does not predict (its prediction is NaN) are dropped."""
    spec = NUCLEI[nucleus]
    z = np.asarray(shieldings["atomic_numbers"])
    dft = np.asarray(shieldings[spec["dft"]], dtype=float)
    nn = np.asarray(shieldings[spec["nn"]], dtype=float)
    keep = (z == spec["atomic_number"]) & np.isfinite(dft) & np.isfinite(nn)
    return dft[keep] - nn[keep]


def residual_stats(errors, small_threshold=0.1):
    """Summary statistics of a residual array: count, RMSE, MAE, the 95th percentile of the
    absolute error, and the fraction of atoms whose absolute error is below small_threshold ppm
    (the paper reports >95% of proton sites below 0.1 ppm)."""
    errors = np.asarray(errors, dtype=float)
    abs_err = np.abs(errors)
    if not errors.size:
        return {"n": 0, "rmse": float("nan"), "mae": float("nan"), "median_ae": float("nan"),
                "abs_p95": float("nan"), "frac_below": float("nan")}
    stats = summarize_errors(errors, np.zeros_like(errors))
    stats["abs_p95"] = float(np.quantile(abs_err, 0.95))
    stats["frac_below"] = float(np.mean(abs_err < small_threshold))
    return stats


def residual_summary(dft8k_path, small_threshold=0.1):
    """Load the DFT8K shieldings and return {nucleus: residual_stats} for both nuclei."""
    shieldings = load_shieldings(dft8k_path)
    return {nucleus: residual_stats(residuals(shieldings, nucleus), small_threshold)
            for nucleus in NUCLEI}


# ---------------------------------------------------------------------------------------------
# Figure 5B's molecule-level panels: the two extreme-residual callouts and the six functional-
# group boxes. Unlike the whole-dataset histogram above, these need each molecule's SMILES (to
# find/classify structures), so they read the reader's per-molecule API instead of all_shieldings.

def iter_molecule_residuals(dft8k_path, nucleus):
    """Yield (molecule_index, molecule_id, smiles, residuals) for every DFT8K molecule, where
    residuals is that molecule's DFT-minus-MagNET residual array (ppm) for its atoms of the given
    nucleus (empty if the molecule has none of that element, or MagNET made no prediction there).
    Molecules with no SMILES on file (the literal "none") are still yielded; callers that need a
    parseable structure should skip those themselves."""
    from dft8k_reader import DFT8k
    spec = NUCLEI[nucleus]
    with DFT8k(dft8k_path) as ds:
        g = ds.aimnet2
        # One bulk read of the flat per-atom arrays, then slice per molecule using the group's own
        # precomputed atom offsets (g._start). Calling g.molecule(i) per molecule instead does
        # several small h5py reads each for all ~7,000 molecules, ~260x slower for identical data.
        flat = g.all_shieldings()
        z_all, dft_all, nn_all = flat["atomic_numbers"], flat[spec["dft"]], flat[spec["nn"]]
        start = g._start
        for i in range(g.n_molecules):
            sl = slice(int(start[i]), int(start[i + 1]))
            z, dft, nn = z_all[sl], dft_all[sl], nn_all[sl]
            keep = (z == spec["atomic_number"]) & np.isfinite(dft) & np.isfinite(nn)
            yield i, int(g.molecule_ids[i]), g.smiles(i), (dft[keep] - nn[keep])


def find_extreme_residual(dft8k_path, nucleus, sign="max"):
    """The single most extreme DFT-minus-MagNET residual anywhere in the DFT8K set for one
    nucleus: the largest positive residual if sign="max", or the most negative if sign="min". The
    exact extremum over the full released test set (not a random sample), so it is deterministic.
    For 1H, sign="max" reproduces the published figure's positive-tail callout exactly (a porphyrin
    ring, +17.62 ppm). sign="min" gives a different, chemically unremarkable ester outlier at
    -4.06 ppm rather than the figure's negative-tail callout (a sulfonium zwitterion, -1.04 ppm):
    that callout is a hand-picked illustrative example, not the strict extremum. See
    molecule_by_id/find_extreme_zwitterion_residual for how that specific example was located.
    Returns a dict with the parent molecule's id/smiles and the residual value, or None if no atom
    of that nucleus has a prediction anywhere in the set."""
    if sign not in ("max", "min"):
        raise ValueError(f"sign must be 'max' or 'min', got {sign!r}")
    best = None
    for i, mol_id, smiles, res in iter_molecule_residuals(dft8k_path, nucleus):
        if res.size == 0:
            continue
        val = float(res.max()) if sign == "max" else float(res.min())
        if best is None or (sign == "max" and val > best["residual"]) or (sign == "min" and val < best["residual"]):
            best = {"molecule_index": i, "molecule_id": int(mol_id), "smiles": smiles, "residual": val}
    return best


def molecule_by_id(dft8k_path, nucleus, molecule_id):
    """The residual data for one specific DFT8K molecule, by its integer id. Used to look up
    Figure 5B's sulfonium-zwitterion callout (id 88779, identified by brute-force scanning the
    most-negative 1H residuals for a match to the published -1.040 ppm value -- see
    find_extreme_zwitterion_residual for the automated version of that search) once its id is
    known, without re-scanning the whole dataset. Returns the same dict shape as
    find_extreme_residual (residual is that molecule's single most negative residual for the given
    nucleus), or None if the molecule has no atom of that nucleus with a prediction."""
    for i, mol_id, smiles, res in iter_molecule_residuals(dft8k_path, nucleus):
        if mol_id == molecule_id:
            if res.size == 0:
                return None
            return {"molecule_index": i, "molecule_id": int(mol_id), "smiles": smiles,
                    "residual": float(res.min())}
    return None


def find_extreme_zwitterion_residual(dft8k_path, nucleus, sign="min"):
    """Like find_extreme_residual, but restricted to molecules containing BOTH a formally
    positively-charged sulfur ([S+]) and a formally negatively-charged atom ([-]) -- i.e.
    sulfonium zwitterions. Even this narrower filter does not land on the same molecule as the
    published callout: a different, more negative sulfonium zwitterion exists in the set. Provided
    for exploration, not as an exact reproduction. The published example itself is available via
    molecule_by_id(..., molecule_id=88779)."""
    from rdkit import Chem
    if sign not in ("max", "min"):
        raise ValueError(f"sign must be 'max' or 'min', got {sign!r}")
    pos_s = Chem.MolFromSmarts("[S+]")
    neg = Chem.MolFromSmarts("[-]")
    best = None
    for i, mol_id, smiles, res in iter_molecule_residuals(dft8k_path, nucleus):
        if res.size == 0 or smiles in _UNPARSEABLE_SMILES:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or not (mol.HasSubstructMatch(pos_s) and mol.HasSubstructMatch(neg)):
            continue
        val = float(res.max()) if sign == "max" else float(res.min())
        if best is None or (sign == "max" and val > best["residual"]) or (sign == "min" and val < best["residual"]):
            best = {"molecule_index": i, "molecule_id": int(mol_id), "smiles": smiles, "residual": val}
    return best


# A handful of DFT8K molecules have no usable SMILES: some store the literal "none" (documented in
# dft8k_reader.py), and a few others store the literal "nan" (an apparent upstream data quirk). Both
# are skipped before ever reaching RDKit, which would otherwise print a harmless but noisy
# parse-error message to stderr for each one.
_UNPARSEABLE_SMILES = {"none", "nan", ""}


# SMARTS patterns for the six functional groups the published figure highlights. These identify
# whether a molecule CONTAINS the group at all (a molecule-level classification), not which atom
# specifically carries it: the released dft8k.hdf5 stores atoms in whatever order the original DFT
# input used, not SMILES parse order, so a matched substructure cannot be reliably mapped back to
# one specific stored atom without extra data. Averaging over every atom of the nucleus in every
# matching molecule is the robust alternative used here.
FUNCTIONAL_GROUP_SMARTS = {
    "Carbonyls": "[CX3]=[OX1]",
    "Amines": "[NX3;H2,H1,H0;!$(NC=O);!$(N=[!#6]);!a]",
    "Sulfonyl": "[SX4](=[OX1])(=[OX1])",
    "Pyridines": "[n;r6]",
    "Furans": "[o;r5]",
    "Nitroso": "[NX2]=[OX1]",
}


def functional_group_errors(dft8k_path, nucleus, patterns=None):
    """Mean absolute DFT-minus-MagNET residual (ppm), for every atom of `nucleus`, in every DFT8K
    molecule whose SMILES matches each functional-group SMARTS pattern (default
    FUNCTIONAL_GROUP_SMARTS). A molecule can match more than one group (e.g. a pyridine bearing a
    carbonyl counts in both), matching the natural reading of the published panel's per-group boxes
    as independent slices of the same dataset, not a mutually exclusive partition.

    Returns {group_name: {"mean_abs_error": ppm, "n_molecules": matching molecule count,
    "n_atoms": total atoms of this nucleus pooled across those molecules}}. A group with no matches
    in the released set gets NaN/0/0.
    """
    from rdkit import Chem
    if patterns is None:
        patterns = FUNCTIONAL_GROUP_SMARTS
    compiled = {name: Chem.MolFromSmarts(smarts) for name, smarts in patterns.items()}
    for name, patt in compiled.items():
        if patt is None:
            raise ValueError(f"invalid SMARTS for group {name!r}: {patterns[name]!r}")
    pooled_residuals = {name: [] for name in patterns}
    n_molecules = {name: 0 for name in patterns}
    for _, _, smiles, res in iter_molecule_residuals(dft8k_path, nucleus):
        if smiles in _UNPARSEABLE_SMILES or res.size == 0:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        for name, patt in compiled.items():
            if mol.HasSubstructMatch(patt):
                pooled_residuals[name].append(res)
                n_molecules[name] += 1
    out = {}
    for name in patterns:
        chunks = pooled_residuals[name]
        if chunks:
            all_res = np.concatenate(chunks)
            out[name] = {"mean_abs_error": float(np.mean(np.abs(all_res))),
                        "n_molecules": n_molecules[name], "n_atoms": int(all_res.size)}
        else:
            out[name] = {"mean_abs_error": float("nan"), "n_molecules": 0, "n_atoms": 0}
    return out
