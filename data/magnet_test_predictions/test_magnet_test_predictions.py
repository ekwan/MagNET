"""Tests for the magnet_test_predictions reader.

The fixture builds a tiny archive with the real encoder (`build_magnet_test_predictions.build_file`)
covering every tuple family, both key kinds, string columns, multi-dimensional trajectory arrays, and
the content-addressed dedup path - so it is a faithful spec of the on-disk format, and the reader is
tested end to end against it. Optional tests run against the real file when it is present.
"""
import os
import pickle
import numpy as np
import pytest

import magnet_test_predictions_reader as R
import build_magnet_test_predictions as B


def _make_source(dirpath):
    """Write tiny pickles, one per family, mimicking the real result files (names drive the reader)."""
    files = {}

    # gas-phase 4-tuple (prediction, target, atomic_numbers, mask); NaN in a prediction
    files["results_gasphaseinternal_H_pretrained_H_vib1.pickle"] = {
        10: (np.array([1.0, 2.0, 3.0], np.float32), np.array([1.05, 2.10, 2.90], np.float32),
             np.array([1, 1, 6]), np.array([True, True, False])),
        11: (np.array([4.0, np.nan], np.float32), np.array([4.20, 5.00], np.float32),
             np.array([1, 8]), np.array([True, False])),
    }
    # a SECOND gas-phase model on the SAME test set: identical target/atoms/mask -> must dedup
    files["results_gasphaseinternal_H_chloroform_H_vib1.pickle"] = {
        10: (np.array([1.1, 2.1, 3.1], np.float32), np.array([1.05, 2.10, 2.90], np.float32),
             np.array([1, 1, 6]), np.array([True, True, False])),
        11: (np.array([4.1, np.nan], np.float32), np.array([4.20, 5.00], np.float32),
             np.array([1, 8]), np.array([True, False])),
    }
    # MD-solute 2-tuple (target, prediction), tuple keys
    files["results_solutesmd500isolated_chloroform_C_chloroform_C.pickle"] = {
        (1, 1): (np.array([10.0, 20.0], np.float32), np.array([10.1, 19.8], np.float32)),
        (1, 2): (np.array([30.0], np.float32), np.array([30.5], np.float32)),
    }
    # super-corrections 4-tuple (name, smiles, prediction, target), 2-D arrays, int keys
    files["results_solutesmd50super_corrections_TIP4P_C_tip4p_C.pickle"] = {
        7: ("MOL1", "CCO", np.array([[-1.0, -2.0]], np.float32), np.array([[-1.2, -1.9]], np.float32)),
    }
    # QCD 4-tuple (stat_pred, stat_target, traj_pred(reps,frames,scored), traj_target)
    tp = np.zeros((2, 3, 2), np.float32); tt = np.zeros((2, 3, 2), np.float32)
    tp[:] = np.array([100.0, 50.0]); tt[:] = np.array([100.0, 50.0])
    tp += 0.4  # predicted trajectory shifted +0.4 from stationary
    tt += 0.3  # DFT trajectory shifted +0.3 -> correction error 0.1 per atom
    files["results_qcdtraj2500_C_pretrained_C.pickle"] = {
        100: (np.array([100.0, 50.0], np.float32), np.array([100.0, 50.0], np.float32), tp, tt),
    }
    # str-key family
    files["results_solutesmd8superopenmm_TIP4P_C_tip4p_C.pickle"] = {
        "MOLX": tuple(np.full((2, 2), v, np.float32) for v in (1.0, 1.1, 2.0, 2.1)),
    }
    for fn, d in files.items():
        pickle.dump(d, open(os.path.join(dirpath, fn), "wb"))


@pytest.fixture
def archive(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    _make_source(str(src))
    out = str(tmp_path / "magnet_test_predictions.hdf5")
    nfiles, ndedup = B.build_file(str(src), out)
    assert nfiles == 6
    assert ndedup > 0, "the two gas-phase models share a target/atoms/mask; dedup must trigger"
    return out


def test_available(archive):
    got = R.available(archive)
    assert "gasphaseinternal_H_pretrained_H_vib1" in got
    assert "qcdtraj2500_C_pretrained_C" in got
    assert len(got) == 6


def test_roundtrip_all_families(archive):
    """Every family decodes back to its source pickle exactly (floats within a quantum)."""
    src = os.path.join(os.path.dirname(archive), "src")
    for fn in os.listdir(src):
        group = fn[len("results_"):-len(".pickle")]
        orig = pickle.load(open(os.path.join(src, fn), "rb"))
        dec = R.load(archive, group)
        assert set(dec) == set(orig)
        for k in orig:
            for a, b in zip(orig[k], dec[k]):
                if isinstance(a, str):
                    assert a == b
                else:
                    a = np.asarray(a); b = np.asarray(b)
                    assert a.shape == b.shape
                    if a.dtype == bool:
                        assert np.array_equal(a, b.astype(bool))
                    elif np.issubdtype(a.dtype, np.integer):
                        assert np.array_equal(a, b)
                    else:
                        m = np.isfinite(a)
                        assert np.array_equal(np.isnan(a), np.isnan(np.asarray(b, float)))
                        if m.any():
                            assert np.abs(a[m] - np.asarray(b)[m]).max() < 1e-4


def test_dedup_reference_resolves_to_identical_bytes(archive):
    """The deduped target of the second model must equal the first model's target exactly."""
    import h5py
    with h5py.File(archive, "r") as h:
        # exactly one of the two same-target groups stores p1_data; the other references it
        g1 = h["gasphaseinternal_H_pretrained_H_vib1"]
        g2 = h["gasphaseinternal_H_chloroform_H_vib1"]
        has = [("p1_data" in g) for g in (g1, g2)]
        ref = [("p1_data_ref" in g.attrs) for g in (g1, g2)]
        assert sum(has) == 1 and sum(ref) == 1 and has != ref
    a = R.load(archive, "gasphaseinternal_H_pretrained_H_vib1")
    b = R.load(archive, "gasphaseinternal_H_chloroform_H_vib1")
    for k in a:
        assert np.array_equal(np.asarray(a[k][1]), np.asarray(b[k][1]))  # identical DFT target
        assert not np.array_equal(np.asarray(a[k][0]), np.asarray(b[k][0]))  # different predictions


def test_errors_apply_mask_and_stats(archive):
    e = R.errors(archive, "gasphaseinternal_H_pretrained_H_vib1")
    assert np.allclose(np.sort(e), [0.05, 0.10, 0.20], atol=1e-4)  # mol10 two atoms + mol11 one
    med, mae, rmse, n = R.stats(archive, "gasphaseinternal_H_pretrained_H_vib1")
    assert n == 3 and abs(mae - 0.35 / 3) < 1e-4


def test_tuple_keys_roundtrip(archive):
    d = R.load(archive, "solutesmd500isolated_chloroform_C_chloroform_C")
    assert set(d) == {(1, 1), (1, 2)}


def test_qcd_correction(archive):
    # constructed so every atom's correction error is 0.1
    med, mae, rmse, n = R.qcd_correction_stats(archive, "qcdtraj2500_C_pretrained_C")
    assert abs(mae - 0.1) < 1e-3 and abs(rmse - 0.1) < 1e-3


def test_stats_refuses_qcd(archive):
    with pytest.raises(ValueError):
        R.stats(archive, "qcdtraj2500_C_pretrained_C")


def test_qcd_stats_refuses_nonqcd(archive):
    with pytest.raises(ValueError):
        R.qcd_correction_stats(archive, "gasphaseinternal_H_pretrained_H_vib1")


def test_errors_refuses_unpaired_family(archive):
    # solutesmd8superopenmm is four float arrays with no documented pairing -> must refuse, not guess
    with pytest.raises(ValueError):
        R.errors(archive, "solutesmd8superopenmm_TIP4P_C_tip4p_C")


# ----- optional real-file checks (run locally when the archive is downloaded) -----
REAL = os.path.join(os.path.dirname(__file__), "magnet_test_predictions.hdf5")


@pytest.mark.skipif(not os.path.exists(REAL), reason="real hdf5 not present (runs locally)")
def test_real_tables_s3_s4():
    # foundation 13C vibrated internal (SI: 0.31917763 / 0.47851510 / 7.13765463)
    med, mae, rmse, n = R.stats(REAL, "gasphaseinternal_C_pretrained_C_vib1")
    assert abs(med - 0.3192) < 1e-3 and abs(mae - 0.4785) < 1e-3 and abs(rmse - 7.1377) < 1e-2
    # foundation 1H isolated chloroform (SI: 0.03216 / 0.07175 / 0.29437)
    med, mae, rmse, n = R.stats(REAL, "solutesmd500isolated_chloroform_H_pretrained_H")
    assert abs(med - 0.0322) < 1e-3 and abs(mae - 0.0718) < 1e-3 and abs(rmse - 0.2944) < 1e-2


@pytest.mark.skipif(not os.path.exists(REAL), reason="real hdf5 not present (runs locally)")
def test_real_table_s5_qcd():
    # SI Table S5: MagNET 1H 0.01526165/0.023047674/0.04020983 ; 13C 0.17198181/0.26223338/0.50198762
    med, mae, rmse, n = R.qcd_correction_stats(REAL, "qcdtraj2500_H_pretrained_H")
    assert abs(med - 0.0152617) < 1e-3 and abs(mae - 0.0230477) < 1e-3 and abs(rmse - 0.0402098) < 1e-3
    med, mae, rmse, n = R.qcd_correction_stats(REAL, "qcdtraj2500_C_pretrained_C")
    assert abs(med - 0.1719818) < 2e-3 and abs(mae - 0.2622334) < 1e-3 and abs(rmse - 0.5019876) < 1e-3
