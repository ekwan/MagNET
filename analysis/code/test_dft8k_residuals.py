"""Tests for analysis/code/dft8k_residuals.py.

The synthetic tests check the residual selection and statistics on a hand-built shieldings dict.
One opt-in test runs against the real dft8k.hdf5 and asserts the published DFT8K residual numbers
(MagNET-Zero is a useful surrogate for DFT): 1H RMSE ~0.11 ppm, 13C RMSE ~0.81 ppm, with >95% of
proton sites below 0.1 ppm error.
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "data", "dft8k"))  # dft8k_reader
sys.path.insert(0, HERE)
import paths

import dft8k_residuals as D  # noqa: E402


def make_shieldings():
    """A flat shieldings dict like the reader's all_shieldings: H, C, C, H, N. MagNET predicts H
    with the WP04 model and C with the wB97X-D model; other atoms are NaN. The 1H residuals are
    -0.1 and +0.2; the 13C residuals are 0.1 and 0.1."""
    return {
        "atomic_numbers": np.array([1, 6, 6, 1, 7]),
        "wp04_pcSseg2": np.array([30.5, 150.5, 151.5, 31.5, 200.5]),
        "wb97xd_pcSseg2": np.array([30.7, 150.7, 151.7, 31.7, 200.7]),
        "nn_wp04_pcSseg2": np.array([30.6, np.nan, np.nan, 31.3, np.nan]),
        "nn_wb97xd_pcSseg2": np.array([np.nan, 150.6, 151.6, np.nan, np.nan]),
    }


def test_residuals_select_nucleus_and_drop_unpredicted():
    s = make_shieldings()
    h = D.residuals(s, "H")
    # only the two H atoms (indices 0, 3), residual = DFT - MagNET
    np.testing.assert_allclose(sorted(h), sorted([30.5 - 30.6, 31.5 - 31.3]))
    c = D.residuals(s, "C")
    np.testing.assert_allclose(sorted(c), sorted([150.7 - 150.6, 151.7 - 151.6]))
    # the nitrogen atom is in neither nucleus' residuals
    assert len(h) == 2 and len(c) == 2


def test_residual_stats():
    errors = np.array([-0.1, 0.2, 0.05, -0.05])
    st = D.residual_stats(errors, small_threshold=0.1)
    assert st["n"] == 4
    assert st["rmse"] == pytest.approx(np.sqrt(np.mean(errors ** 2)))
    assert st["mae"] == pytest.approx(np.mean(np.abs(errors)))
    # two of four are strictly below 0.1 ppm absolute (0.05 and 0.05); 0.1 and 0.2 are not
    assert st["frac_below"] == pytest.approx(0.5)


def test_residual_stats_empty():
    st = D.residual_stats(np.array([]))
    assert st["n"] == 0 and np.isnan(st["rmse"])


# --- opt-in: reproduce the published DFT8K residual numbers from the released data --------------

REAL = paths.dataset_file("dft8k", file=__file__)


@pytest.mark.skipif(not os.path.exists(REAL), reason="real dft8k.hdf5 not present")
def test_reproduces_published_dft8k_residuals():
    summary = D.residual_summary(REAL)
    # 1H: RMSE ~0.11 ppm, >95% of sites below 0.1 ppm error (paper Figure 5 / DFT8K text)
    assert summary["H"]["rmse"] == pytest.approx(0.105, abs=0.01)
    assert summary["H"]["mae"] == pytest.approx(0.035, abs=0.01)
    assert summary["H"]["frac_below"] >= 0.95
    # 13C: RMSE ~0.81 ppm, MAE ~0.41 ppm
    assert summary["C"]["rmse"] == pytest.approx(0.807, abs=0.02)
    assert summary["C"]["mae"] == pytest.approx(0.409, abs=0.02)


@pytest.mark.skipif(not os.path.exists(REAL), reason="real dft8k.hdf5 not present")
def test_extreme_residual_reproduces_the_published_porphyrin_callout():
    # Figure 5B's positive-tail outlier callout ("Porphyrin Rings", error +17.620 ppm) is the true
    # global maximum 1H residual, so this reproduces it exactly.
    extreme = D.find_extreme_residual(REAL, "H", sign="max")
    assert extreme["residual"] == pytest.approx(17.6197, abs=1e-3)
    assert extreme["molecule_id"] == 20170716


@pytest.mark.skipif(not os.path.exists(REAL), reason="real dft8k.hdf5 not present")
def test_extreme_residual_min_is_not_the_published_zwitterion_callout():
    # The negative-tail callout ("Sulfonium Zwitterion", -1.040 ppm) is NOT the true global
    # minimum -- a different, chemically unremarkable molecule is more negative. This test locks
    # down that documented discrepancy so it is not silently "fixed" by a future data update.
    extreme = D.find_extreme_residual(REAL, "H", sign="min")
    assert extreme["residual"] < -1.5
    assert extreme["molecule_id"] != 88779


@pytest.mark.skipif(not os.path.exists(REAL), reason="real dft8k.hdf5 not present")
def test_molecule_by_id_reproduces_the_published_zwitterion_callout():
    zwitterion = D.molecule_by_id(REAL, "H", molecule_id=88779)
    assert zwitterion["residual"] == pytest.approx(-1.0403, abs=1e-3)
    assert "[S+]" in zwitterion["smiles"] and "[O-]" in zwitterion["smiles"]


@pytest.mark.skipif(not os.path.exists(REAL), reason="real dft8k.hdf5 not present")
def test_functional_group_errors_reproduces_published_values():
    # published (Figure 5B, mean |1H residual| per group): Carbonyls 0.034, Amines 0.036,
    # Sulfonyl 0.039, Pyridines 0.042, Furans 0.034, Nitroso 0.088 ppm.
    published = {"Carbonyls": 0.034, "Amines": 0.036, "Sulfonyl": 0.039,
                 "Pyridines": 0.042, "Furans": 0.034, "Nitroso": 0.088}
    group_errors = D.functional_group_errors(REAL, "H")
    for name, expected in published.items():
        assert group_errors[name]["mean_abs_error"] == pytest.approx(expected, abs=2e-3), name
        assert group_errors[name]["n_molecules"] > 0


@pytest.mark.skipif(not os.path.exists(REAL), reason="real dft8k.hdf5 not present")
def test_functional_group_errors_respects_an_explicit_empty_patterns_dict():
    # patterns=None means "use the default set"; patterns={} is a real, distinct request for "no
    # groups" and must return {} rather than silently falling back to the default (an `x or
    # DEFAULT` check would wrongly treat an empty dict as falsy and do exactly that).
    assert D.functional_group_errors(REAL, "H", patterns={}) == {}


def test_find_extreme_zwitterion_residual_requires_a_valid_sign():
    with pytest.raises(ValueError):
        D.find_extreme_zwitterion_residual(REAL, "H", sign="sideways")


@pytest.mark.skipif(not os.path.exists(REAL), reason="real dft8k.hdf5 not present")
def test_find_extreme_zwitterion_residual_finds_a_real_sulfonium_zwitterion():
    # not asserted to match the published callout exactly (see molecule_by_id's docstring: the
    # published example is a hand-picked one, not the strict extremum among zwitterions) -- this
    # just checks the search itself works and returns a genuine sulfonium zwitterion.
    result = D.find_extreme_zwitterion_residual(REAL, "H", sign="min")
    assert result is not None
    assert "[S+]" in result["smiles"] and "[O-]" in result["smiles"]
    assert result["residual"] < 0
