"""Tests for analysis/code/delta50_residuals.py.

Synthetic tests check residual selection and statistics on a hand-built shieldings dict. Opt-in
tests run against the real delta50.hdf5 and reproduce the paper's DELTA50 observation: MagNET-Zero
tracks its DFT reference to well under 0.5 ppm (1H) and a few ppm (13C), with nitromethane,
nitroethane, and 2-methyl-2-nitropropane the significant 13C outliers (up to ~7 ppm).
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "data", "delta50"))  # decode_delta50
sys.path.insert(0, HERE)
import paths  # noqa: E402

import delta50_residuals as D  # noqa: E402


def make_shieldings():
    """Flat shieldings dict like the reader's all_atoms: H, C, C, H, N. MagNET-Zero is one array
    (WP04 quality at H, wB97X-D quality at C); the DFT reference is split by nucleus."""
    return {
        "atomic_numbers": np.array([1, 6, 6, 1, 7]),
        "shielding_wp04_pcSseg2": np.array([30.5, 150.5, 151.5, 31.5, 200.5]),
        "shielding_wb97xd_pcSseg2": np.array([30.7, 150.7, 151.7, 31.7, 200.7]),
        "nn_magnet_zero": np.array([30.6, 150.6, 151.6, 31.3, np.nan]),
    }


def test_residuals_select_nucleus_against_the_right_dft_level():
    s = make_shieldings()
    h = D.residuals(s, "H")  # DFT WP04 minus MagNET at the two H atoms (0, 3)
    np.testing.assert_allclose(sorted(h), sorted([30.5 - 30.6, 31.5 - 31.3]))
    c = D.residuals(s, "C")  # DFT wB97X-D minus MagNET at the two C atoms (1, 2)
    np.testing.assert_allclose(sorted(c), sorted([150.7 - 150.6, 151.7 - 151.6]))
    assert len(h) == 2 and len(c) == 2


def test_residual_stats():
    errors = np.array([-0.1, 0.2, 0.05, -0.05])
    st = D.residual_stats(errors)
    assert st["n"] == 4
    assert st["rmse"] == pytest.approx(np.sqrt(np.mean(errors ** 2)))
    assert st["mae"] == pytest.approx(np.mean(np.abs(errors)))
    assert st["max_ae"] == pytest.approx(0.2)


def test_residual_stats_empty():
    st = D.residual_stats(np.array([]))
    assert st["n"] == 0 and np.isnan(st["rmse"])


# --- opt-in: reproduce the DELTA50 residual numbers from the released data -----------------------

REAL = paths.dataset_file("delta50", file=__file__)


@pytest.mark.skipif(not os.path.exists(REAL), reason="real delta50.hdf5 not present")
def test_reproduces_delta50_magnet_zero_vs_dft_residuals():
    summary = D.residual_summary(REAL)
    # 1H: MagNET-Zero tracks WP04 to ~0.05 ppm RMSE, max residual ~0.47 ppm ("up to 0.4 ppm")
    assert summary["H"]["rmse"] == pytest.approx(0.055, abs=0.01)
    assert summary["H"]["max_ae"] == pytest.approx(0.469, abs=0.02)
    # 13C: RMSE ~0.61 ppm, max residual ~6.65 ppm ("up to 7 ppm")
    assert summary["C"]["rmse"] == pytest.approx(0.607, abs=0.03)
    assert summary["C"]["max_ae"] == pytest.approx(6.654, abs=0.05)


@pytest.mark.skipif(not os.path.exists(REAL), reason="real delta50.hdf5 not present")
def test_named_13c_outliers_are_the_three_nitroalkanes():
    per_mol = D.per_molecule_max_residual(REAL)
    top3 = sorted(per_mol, key=lambda nm: -per_mol[nm]["C"])[:3]
    assert set(top3) == {"nitromethane", "nitroethane", "2-methyl-2-nitropropane"}
    # nitromethane is the single largest 13C outlier, ~6.65 ppm
    assert per_mol["nitromethane"]["C"] == pytest.approx(6.654, abs=0.05)
    # nitrobenzene, by contrast, is accurately predicted (paper's explicit contrast)
    assert per_mol["nitrobenzene"]["C"] < 0.5


# Golden experimental shifts (ppm) for the molecules whose atom order is most likely to be mismapped:
# the three symmetry-fallback cases and the near-degenerate distinct environments (amide N-methyls,
# diastereotopic methyls). Values are the published DELTA50 shifts; a mapping regression changes them.
_GOLDEN_MULTISET = {
    "cyclohexanone": {"C": [25.02, 27.03, 27.03, 42.0, 42.0, 212.15]},
    "methyl acetate": {"C": [20.71, 51.62, 171.55]},
    "2-methyl-2-nitropropane": {"C": [27.87, 27.87, 27.87, 85.06]},
    "n,n-dimethylformamide": {"C": [31.44, 36.48, 162.52]},
    "n,n-dimethylacetamide": {"C": [21.58, 35.2, 38.05, 170.66]},
    "2-methyl-2-butene": {"C": [13.41, 17.32, 25.63, 118.44, 132.1]},
}
# (0-based carbon index -> shift): pins each shift to a specific atom, so a swap of two same-value-set
# environments (e.g. the two DMF N-methyls) fails even though the multiset is unchanged.
_GOLDEN_ATOM = {
    "n,n-dimethylformamide": {0: 162.52, 3: 31.44, 4: 36.48},
    "n,n-dimethylacetamide": {1: 170.66, 3: 35.2, 4: 38.05, 5: 21.58},
    "2-methyl-2-butene": {0: 17.32, 1: 132.1, 2: 118.44, 3: 13.41, 4: 25.63},
    "cyclohexanone": {1: 212.15, 2: 42.0, 3: 42.0, 4: 27.03, 5: 27.03, 6: 25.02},
}


@pytest.mark.skipif(not os.path.exists(REAL), reason="real delta50.hdf5 not present")
def test_experimental_multiset_matches_published_delta50():
    from decode_delta50 import Delta50
    with Delta50(REAL) as ds:
        for name, expected in _GOLDEN_MULTISET.items():
            m = ds.molecule_by_name(name)
            got = sorted(round(float(x), 2) for x in m["experimental_shift"][m["atomic_numbers"] == 6]
                         if np.isfinite(x))
            assert got == sorted(expected["C"]), name


@pytest.mark.skipif(not os.path.exists(REAL), reason="real delta50.hdf5 not present")
def test_experimental_swap_sensitive_carbons_pinned_to_atoms():
    from decode_delta50 import Delta50
    with Delta50(REAL) as ds:
        for name, atoms in _GOLDEN_ATOM.items():
            m = ds.molecule_by_name(name)
            for idx, shift in atoms.items():
                assert m["atomic_numbers"][idx] == 6, (name, idx)
                assert m["experimental_shift"][idx] == pytest.approx(shift, abs=1e-3), (name, idx)


@pytest.mark.skipif(not os.path.exists(REAL), reason="real delta50.hdf5 not present")
def test_experimental_shift_present_and_agrees_with_dft():
    from decode_delta50 import Delta50
    with Delta50(REAL) as ds:
        atoms = ds.all_atoms()
    z = atoms["atomic_numbers"]
    exp = atoms["experimental_shift"]
    # experimental is present for every H and C atom, absent for heteroatoms
    assert np.isfinite(exp[(z == 1) | (z == 6)]).all()
    assert np.isnan(exp[(z != 1) & (z != 6)]).all()
    # A correct atom mapping makes scaled DFT track experiment; a scrambled one would not. These
    # tolerances are loose (our DFT is gas phase on AIMNet2 geometries, not DELTA50's PCM level) but
    # far below the ppm-scale RMSE a wrong assignment would produce.
    for zz, key, tol in ((1, "shielding_wp04_pcSseg2", 0.5), (6, "shielding_wb97xd_pcSseg2", 5.0)):
        m = (z == zz) & np.isfinite(exp)
        sigma = atoms[key][m]
        A = np.vstack([sigma, np.ones(m.sum())]).T
        coef, *_ = np.linalg.lstsq(A, exp[m], rcond=None)
        rmse = np.sqrt(np.mean((A @ coef - exp[m]) ** 2))
        assert rmse < tol, f"Z={zz} scaled-DFT-vs-exp RMSE {rmse:.3f} too large"
