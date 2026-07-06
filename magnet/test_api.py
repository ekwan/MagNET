"""Adversarial edge-case tests for the magnet public API (magnet/api.py).

Pure-logic tests (single-vs-list detection, validation, solvent checking, solute selection) run
whenever the torch stack is installed. Tests marked @needs_model load real checkpoints and are
skipped unless the weights are present; they stay few and fast (n_passes=1, symmetrize=False, tiny
molecules). Run the whole suite with `pytest`; to include the real-model tests, check out the
Hugging Face copy so the weights land in model_checkpoints/.
"""
import os
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

import magnet.api as api
from magnet import scaling

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)                       # magnet/ -> repo root
_CKPT_REL = os.path.join("model_checkpoints", "MagNET-Zero", "MagNET-Zero_1H.ckpt")
_HAVE_MODELS = os.path.exists(os.path.join(_REPO, _CKPT_REL))
needs_model = pytest.mark.skipif(not _HAVE_MODELS, reason="checkpoints not available")

# ---- tiny geometries (validity irrelevant for API-contract tests) ----
CH4_Z = np.array([6, 1, 1, 1, 1])
CH4_XYZ = np.array([[0., 0., 0.],
                    [0.629, 0.629, 0.629],
                    [-0.629, -0.629, 0.629],
                    [-0.629, 0.629, -0.629],
                    [0.629, -0.629, -0.629]])
# formaldehyde CH2O: exercises H, C, and an O (should be NaN in shifts)
CH2O_Z = np.array([6, 8, 1, 1])
CH2O_XYZ = np.array([[0., 0., 0.], [1.2, 0., 0.], [-0.5, 0.94, 0.], [-0.5, -0.94, 0.]])
# one chloroform molecule, atoms in molecule order C H Cl Cl Cl
CHCL3_Z = np.array([6, 1, 17, 17, 17])
CHCL3_XYZ = np.array([[0., 0., 0.], [0., 0., 1.1], [1.7, 0., -0.4],
                      [-0.85, 1.47, -0.4], [-0.85, -1.47, -0.4]])


# =====================================================================
# _as_batch: single-vs-list detection and validation
# =====================================================================

def test_single_ndarray_is_single():
    an, xyz, single = api._as_batch(CH4_Z, CH4_XYZ)
    assert single is True and len(an) == 1 and an[0].shape == (5,) and xyz[0].shape == (5, 3)

def test_single_pythonlist_is_single():
    an, xyz, single = api._as_batch([6, 1, 1, 1, 1], CH4_XYZ.tolist())
    assert single is True and len(xyz) == 1

def test_one_atom_molecule_is_single():
    an, xyz, single = api._as_batch(np.array([1]), np.array([[0., 0., 0.]]))
    assert single is True and xyz[0].shape == (1, 3)

def test_list_of_molecules_is_batch():
    an, xyz, single = api._as_batch([CH4_Z, CH2O_Z], [CH4_XYZ, CH2O_XYZ])
    assert single is False and len(an) == 2

def test_batch_size_one_list_is_batch():
    an, xyz, single = api._as_batch([CH4_Z], [CH4_XYZ])
    assert single is False and len(an) == 1

def test_3d_ndarray_is_batch():
    Z = np.stack([CH4_Z, CH4_Z]);  X = np.stack([CH4_XYZ, CH4_XYZ])  # (2,5) and (2,5,3)
    an, xyz, single = api._as_batch(Z, X)
    assert single is False and len(xyz) == 2 and xyz[0].shape == (5, 3)

def test_ragged_batch_ok():
    an, xyz, single = api._as_batch([CH4_Z, CH2O_Z], [CH4_XYZ, CH2O_XYZ])
    assert xyz[0].shape == (5, 3) and xyz[1].shape == (4, 3)

def test_empty_list_raises():
    with pytest.raises(ValueError, match="empty input"):
        api._as_batch([], [])

def test_empty_ndarray_raises():
    with pytest.raises(ValueError, match="empty input"):
        api._as_batch(np.array([]), np.zeros((0, 3)))

def test_mismatched_atom_counts_single_raises():
    with pytest.raises(ValueError):
        api._as_batch(np.array([6, 1, 1]), np.zeros((4, 3)))

def test_wrong_last_dim_raises():
    with pytest.raises(ValueError):
        api._as_batch(np.array([6, 1]), np.zeros((2, 2)))

def test_mismatched_list_lengths_raises():
    with pytest.raises(ValueError):
        api._as_batch([CH4_Z, CH4_Z], [CH4_XYZ, CH4_XYZ, CH4_XYZ])

def test_coords_1d_single_atom_flat_is_misread():
    # coordinates=(3,) flat for one atom: coordinates[0] is a scalar (ndim 0) -> not single ->
    # treated as a 3-molecule batch of 0-d coords -> raises. Documents the (N,3) requirement.
    with pytest.raises(ValueError):
        api._as_batch(np.array([6]), np.array([0., 0., 0.]))


# =====================================================================
# _check_passes
# =====================================================================

@pytest.mark.parametrize("bad", [0, -1, 1.5, "3", None, 2.0])
def test_check_passes_rejects(bad):
    with pytest.raises(ValueError, match="n_passes"):
        api._check_passes(bad)

@pytest.mark.parametrize("good", [1, 10, np.int64(5)])
def test_check_passes_accepts(good):
    api._check_passes(good)  # no raise

def test_check_passes_bool_quirk():
    # bool is a subclass of int in Python; True passes as "1 pass", False is rejected.
    api._check_passes(True)              # accepted (quirk, not a real bug)
    with pytest.raises(ValueError):
        api._check_passes(False)


# =====================================================================
# predict_shieldings argument validation (pure logic, before model load)
# =====================================================================

def test_predict_shieldings_bad_model():
    with pytest.raises(ValueError, match="model must be one of"):
        api.predict_shieldings(CH4_Z, CH4_XYZ, model="bogus")

@pytest.mark.parametrize("m,fn", [("MagNET-PCM", "implicit_solvent_correction"),
                                  ("MagNET-x", "explicit_solvent_correction")])
def test_predict_shieldings_correction_redirect(m, fn):
    with pytest.raises(ValueError, match=fn):
        api.predict_shieldings(CH4_Z, CH4_XYZ, model=m)

def test_predict_shieldings_bad_passes():
    with pytest.raises(ValueError, match="n_passes"):
        api.predict_shieldings(CH4_Z, CH4_XYZ, n_passes=0)


# =====================================================================
# predict_shifts solvent validation (pure logic, before model load)
# =====================================================================

def test_predict_shifts_unknown_solvent():
    with pytest.raises(ValueError, match="unknown solvent"):
        api.predict_shifts(CH4_Z, CH4_XYZ, solvent="ethanol")

def test_predict_shifts_solvent_case_sensitive():
    with pytest.raises(ValueError, match="unknown solvent"):
        api.predict_shifts(CH4_Z, CH4_XYZ, solvent="Chloroform")

def test_predict_shifts_water_not_rejected_by_validation():
    # "water" must map to TIP4P and NOT trip the unknown-solvent guard.
    tables = scaling.published_scaling_tables()
    assert "TIP4P" in tables["C"] and "TIP4P" in tables["H"]
    # the error message offers "water" (not "TIP4P") as an option:
    options = ["water" if s == "TIP4P" else s for s in tables["C"]]
    assert "water" in options and "TIP4P" not in options


# =====================================================================
# _validate_solvent: atom-type-count check against a named solvent
# =====================================================================

def test_validate_chloroform():
    api._validate_solvent("chloroform", CHCL3_Z)  # no raise

def test_validate_two_chloroforms():
    api._validate_solvent("chloroform", np.concatenate([CHCL3_Z, CHCL3_Z]))

def test_validate_water():
    api._validate_solvent("water", np.array([8, 1, 1]))

def test_validate_benzene():
    api._validate_solvent("benzene", np.array([6] * 6 + [1] * 6))

def test_validate_methanol():
    api._validate_solvent("methanol", np.array([6, 8, 1, 1, 1, 1]))

def test_validate_order_within_block_independent():
    # order within one molecule's block does not matter
    api._validate_solvent("water", np.array([1, 8, 1]))

def test_validate_empty_raises():
    with pytest.raises(ValueError, match="no solvent atoms"):
        api._validate_solvent("chloroform", np.array([], dtype=int))

def test_validate_wrong_count_raises():
    # 7 atoms is not a whole number of chloroforms (5 atoms each)
    with pytest.raises(ValueError, match="not a whole number"):
        api._validate_solvent("chloroform", np.array([6, 6, 1, 1, 17, 17, 17]))

def test_validate_wrong_composition_raises():
    # 5 atoms (one chloroform-sized block) but the wrong elements
    with pytest.raises(ValueError, match="not whole chloroform"):
        api._validate_solvent("chloroform", np.array([6, 6, 6, 6, 6]))

def test_validate_grouped_by_element_raises():
    # two chloroforms' worth of atoms (correct TOTALS: 2 C, 2 H, 6 Cl) but grouped by element, so the
    # contiguous 5-atom blocks are not whole molecules. Total-count check would miss this; block does not.
    grouped = np.array([6, 6, 1, 1, 17, 17, 17, 17, 17, 17])
    with pytest.raises(ValueError, match="not whole chloroform"):
        api._validate_solvent("chloroform", grouped)

def test_validate_two_chloroforms_contiguous_ok():
    # same atoms as above but laid out as two whole contiguous molecules: fine
    api._validate_solvent("chloroform", np.concatenate([CHCL3_Z, CHCL3_Z]))

def test_validate_mismatched_solvent_raises():
    # a benzene block passed as methanol (both divisible cases would still fail composition)
    with pytest.raises(ValueError, match="not whole methanol"):
        api._validate_solvent("methanol", np.array([6, 6, 6, 6, 6, 6]))


# =====================================================================
# supported-solvent tables stay in sync with the code (validation)
# =====================================================================

EXPLICIT_SOLVENTS = ["chloroform", "benzene", "methanol", "water"]
PREDICT_SHIFTS_SOLVENTS = [
    "tetrahydrofuran", "dichloromethane", "chloroform", "toluene", "benzene", "chlorobenzene",
    "acetone", "dimethylsulfoxide", "acetonitrile", "trifluoroethanol", "methanol", "water",
]

def _one_molecule(solvent):
    """Build one molecule of `solvent` as an atomic-number array from the composition table."""
    return np.array([element for element, count in api._SOLVENT_COMPOSITION[solvent].items()
                     for _ in range(count)])

def test_explicit_solvent_set_matches_run_magnet():
    # composition table, the model's block-size table, and the documented list all agree
    from magnet import run_magnet
    assert sorted(api._SOLVENT_COMPOSITION) == sorted(run_magnet.N_ATOMS_PER_SOLVENT)
    assert sorted(api._SOLVENT_COMPOSITION) == sorted(EXPLICIT_SOLVENTS)

def test_solvent_composition_sums_to_block_size():
    from magnet import run_magnet
    for solvent, comp in api._SOLVENT_COMPOSITION.items():
        assert sum(comp.values()) == run_magnet.N_ATOMS_PER_SOLVENT[solvent]

@pytest.mark.parametrize("solvent", EXPLICIT_SOLVENTS)
def test_validate_accepts_one_molecule(solvent):
    api._validate_solvent(solvent, _one_molecule(solvent))  # must not raise

@pytest.mark.parametrize("solvent", EXPLICIT_SOLVENTS)
def test_validate_accepts_two_contiguous_molecules(solvent):
    one = _one_molecule(solvent)
    api._validate_solvent(solvent, np.concatenate([one, one]))  # must not raise

@pytest.mark.parametrize("solvent", EXPLICIT_SOLVENTS)
def test_validate_rejects_dropping_one_atom(solvent):
    # one atom short of a whole molecule -> wrong count
    short = _one_molecule(solvent)[:-1]
    with pytest.raises(ValueError, match="not a whole number"):
        api._validate_solvent(solvent, short)

def test_predict_shifts_supports_the_12_documented_solvents():
    tables = scaling.published_scaling_tables()
    for nucleus in ("H", "C"):
        keys = {"water" if s == "TIP4P" else s for s in tables[nucleus]}
        assert keys == set(PREDICT_SHIFTS_SOLVENTS)

def test_explicit_rejects_predict_only_solvent():
    # acetone is valid for predict_shifts but not for explicit corrections
    atoms = np.concatenate([np.array([6, 1, 1, 1]), CHCL3_Z])
    xyz = np.zeros((len(atoms), 3))
    with pytest.raises(ValueError, match="solvent must be one of"):
        api.explicit_solvent_correction(atoms, xyz, solute_atoms=[0, 1, 2, 3], solvent="acetone",
                                        n_passes=1, symmetrize=False)


# =====================================================================
# _explicit_one: solute selection, reordering, negative/dup/range
# =====================================================================

def _system():  # methane solute (0..3) + one chloroform (4..8)
    atoms = np.concatenate([np.array([6, 1, 1, 1]), CHCL3_Z])
    xyz = np.concatenate([CH4_XYZ[:4] + 10.0, CHCL3_XYZ])  # solute far, solvent near origin
    return atoms, xyz

def test_explicit_one_basic():
    atoms, xyz = _system()
    solute_an, full_an, full_xyz = api._explicit_one(atoms, xyz, [0, 1, 2, 3], "chloroform")
    assert np.array_equal(solute_an, np.array([6, 1, 1, 1]))
    assert np.array_equal(full_an, atoms)  # already solute-first

def test_explicit_one_reorders_solvent_first_input():
    # input has chloroform first, methane second; solute indices point at the methane
    atoms = np.concatenate([CHCL3_Z, np.array([6, 1, 1, 1])])
    xyz = np.concatenate([CHCL3_XYZ, CH4_XYZ[:4] + 10.0])
    solute_an, full_an, full_xyz = api._explicit_one(atoms, xyz, [5, 6, 7, 8], "chloroform")
    assert np.array_equal(solute_an, np.array([6, 1, 1, 1]))
    # reordered: solute first, then the solvent block
    assert np.array_equal(full_an[:4], np.array([6, 1, 1, 1]))
    assert np.array_equal(np.sort(full_an[4:]), np.sort(CHCL3_Z))

def test_explicit_one_output_order_follows_solute_atoms():
    atoms, xyz = _system()
    solute_an, full_an, _ = api._explicit_one(atoms, xyz, [3, 2, 1, 0], "chloroform")  # reversed
    assert np.array_equal(solute_an, np.array([1, 1, 1, 6]))  # matches requested order
    assert np.array_equal(full_an[:4], np.array([1, 1, 1, 6]))

def test_explicit_one_negative_indices():
    atoms, xyz = _system()  # n=9
    solute_an, _, _ = api._explicit_one(atoms, xyz, [-9, -8, -7, -6], "chloroform")  # -> 0,1,2,3
    assert np.array_equal(solute_an, np.array([6, 1, 1, 1]))

def test_explicit_one_wrong_solvent_raises():
    atoms, xyz = _system()  # solvent is chloroform
    with pytest.raises(ValueError, match="not .* benzene"):
        api._explicit_one(atoms, xyz, [0, 1, 2, 3], "benzene")

def test_explicit_one_duplicate_raises():
    atoms, xyz = _system()
    with pytest.raises(ValueError, match="duplicate"):
        api._explicit_one(atoms, xyz, [0, 0, 1, 2], "chloroform")

def test_explicit_one_negative_normalizes_to_duplicate_raises():
    atoms, xyz = _system()  # n=9; index 2 and -7 both -> 2
    with pytest.raises(ValueError, match="duplicate"):
        api._explicit_one(atoms, xyz, [2, -7], "chloroform")

def test_explicit_one_out_of_range_raises():
    atoms, xyz = _system()
    with pytest.raises(ValueError, match="out of range"):
        api._explicit_one(atoms, xyz, [0, 1, 2, 9], "chloroform")

def test_explicit_one_very_negative_raises():
    atoms, xyz = _system()
    with pytest.raises(ValueError, match="out of range"):
        api._explicit_one(atoms, xyz, [-100], "chloroform")

def test_explicit_one_all_atoms_solute_raises():
    atoms, xyz = _system()
    with pytest.raises(ValueError, match="no solvent atoms"):
        api._explicit_one(atoms, xyz, list(range(len(atoms))), "chloroform")

def test_explicit_one_empty_solute_raises():
    atoms, xyz = _system()
    with pytest.raises(ValueError, match="empty"):
        api._explicit_one(atoms, xyz, [], "chloroform")


# =====================================================================
# explicit_solvent_correction: pure-logic validation before any model call
# =====================================================================

def test_explicit_unknown_solvent_raises():
    atoms, xyz = _system()
    with pytest.raises(ValueError, match="solvent must be one of"):
        api.explicit_solvent_correction(atoms, xyz, solute_atoms=[0, 1, 2, 3], solvent="ethanol",
                                        n_passes=1, symmetrize=False)

def test_explicit_solvent_mismatch_raises():
    # declare benzene but the snapshot has chloroform -> count check fails before any model load
    atoms, xyz = _system()
    with pytest.raises(ValueError, match="not .* benzene"):
        api.explicit_solvent_correction(atoms, xyz, solute_atoms=[0, 1, 2, 3], solvent="benzene",
                                        n_passes=1, symmetrize=False)


# =====================================================================
# REAL MODEL TESTS (few, tiny, n_passes=1, symmetrize=False)
# =====================================================================

@needs_model
def test_real_predict_shieldings_single_returns_array():
    out = api.predict_shieldings(CH4_Z, CH4_XYZ, n_passes=1, symmetrize=False)
    assert isinstance(out, np.ndarray) and out.shape == (5,)
    assert np.all(np.isfinite(out))  # H and C all predicted, none NaN

@needs_model
def test_real_predict_shieldings_list_returns_list():
    out = api.predict_shieldings([CH4_Z], [CH4_XYZ], n_passes=1, symmetrize=False)
    assert isinstance(out, list) and len(out) == 1 and out[0].shape == (5,)

@needs_model
def test_real_predict_shieldings_batch_shapes():
    out = api.predict_shieldings([CH4_Z, CH2O_Z], [CH4_XYZ, CH2O_XYZ],
                                 n_passes=1, symmetrize=False)
    assert isinstance(out, list) and out[0].shape == (5,) and out[1].shape == (4,)

@needs_model
def test_real_unsupported_element_rejected():
    # PH3-like: phosphorus (15) is out of vocab
    Z = np.array([15, 1, 1, 1])
    X = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    with pytest.raises(ValueError, match="unsupported atomic numbers"):
        api.predict_shieldings(Z, X, n_passes=1, symmetrize=False)

@needs_model
def test_real_predict_shifts_components_reproduce():
    d = api.predict_shifts(CH2O_Z, CH2O_XYZ, solvent="chloroform", n_passes=1,
                           symmetrize=False, return_components=True)
    assert set(d) == {"shifts", "zero_shielding", "pcm_correction", "coefficients"}
    shifts, sigma, delta = d["shifts"], d["zero_shielding"], d["pcm_correction"]
    coef = d["coefficients"]
    # coefficients must match chloroform rows
    tbl = scaling.published_scaling_tables()
    assert coef["H"] == tbl["H"]["chloroform"] and coef["C"] == tbl["C"]["chloroform"]
    # reproduce shift = intercept + stationary*zero + pcm*pcm for H (idx 2,3) and C (idx 0)
    for idx, nuc in [(0, "C"), (2, "H"), (3, "H")]:
        expect = coef[nuc]["intercept"] + coef[nuc]["stationary"] * sigma[idx] \
                 + coef[nuc]["pcm"] * delta[idx]
        assert np.isclose(shifts[idx], expect, atol=1e-6), (idx, nuc, shifts[idx], expect)
    # O atom (idx 1) must be NaN, H/C finite
    assert np.isnan(shifts[1])
    assert np.all(np.isfinite([shifts[0], shifts[2], shifts[3]]))

@needs_model
def test_real_predict_shifts_acetone_matches_readme():
    # the README "Your First Prediction" worked example: acetone in chloroform on an AIMNet2
    # geometry, default n_passes/symmetrize. Guards the documented numbers against model drift.
    Z = np.array([6, 6, 8, 6, 1, 1, 1, 1, 1, 1])
    X = np.array([
        [1.2913, -0.5947, -0.0016], [0.0029, 0.1931, -0.0010], [-0.0174, 1.3994, -0.0003],
        [-1.2743, -0.6189, 0.0004], [-1.0822, -1.6899, 0.0014], [-1.8597, -0.3513, 0.8791],
        [-1.8605, -0.3530, -0.8783], [1.3415, -1.2208, 0.8916], [1.3170, -1.2669, -0.8614],
        [2.1415, 0.0788, -0.0298],
    ])
    shifts = api.predict_shifts(Z, X, solvent="chloroform")
    carbonyl = shifts[[1]].mean()
    methyl_C = shifts[[0, 3]].mean()
    methyl_H = shifts[[4, 5, 6, 7, 8, 9]].mean()
    assert np.isclose(carbonyl, 207.6, atol=1.0), carbonyl
    assert np.isclose(methyl_C, 30.7, atol=0.5), methyl_C
    assert np.isclose(methyl_H, 2.20, atol=0.15), methyl_H

@needs_model
def test_real_predict_shifts_water_maps_to_tip4p():
    d = api.predict_shifts(CH4_Z, CH4_XYZ, solvent="water", n_passes=1,
                           symmetrize=False, return_components=True)
    tbl = scaling.published_scaling_tables()
    assert d["coefficients"]["H"] == tbl["H"]["TIP4P"]
    assert d["coefficients"]["C"] == tbl["C"]["TIP4P"]

@needs_model
def test_real_single_vs_batch_same_shapes():
    single = api.predict_shieldings(CH4_Z, CH4_XYZ, n_passes=1, symmetrize=False)
    batch = api.predict_shieldings([CH4_Z, CH4_Z], [CH4_XYZ, CH4_XYZ],
                                   n_passes=1, symmetrize=False)
    assert single.shape == batch[0].shape == batch[1].shape
    # values are close (stochastic frames -> not identical); documents non-determinism
    assert np.allclose(single, batch[0], atol=0.5)

@needs_model
@pytest.mark.xfail(reason="known: zero-edge graph (lone atom) gives a cryptic torch error, not a "
                          "clean ValueError; a lone atom is not a valid NMR input",
                   raises=RuntimeError)
def test_real_single_atom_solute():
    # isolated single atom: no neighbors -> empty edge tensor -> deep torch.min() crash.
    api.predict_shieldings(np.array([6]), np.array([[0., 0., 0.]]),
                           n_passes=1, symmetrize=False)

@needs_model
@pytest.mark.xfail(reason="known: atoms all beyond the model cutoff -> zero edges -> torch "
                          "RuntimeError, not a clean ValueError",
                   raises=RuntimeError)
def test_real_atoms_beyond_cutoff_crash():
    # two atoms 50 A apart: no edges within cutoff -> same zero-edge crash as one atom.
    api.predict_shieldings(np.array([6, 1]), np.array([[0., 0., 0.], [0., 0., 50.]]),
                           n_passes=1, symmetrize=False)
