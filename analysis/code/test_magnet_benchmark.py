"""Tests for analysis/code/magnet_benchmark.py.

The error statistics and the published-table structure are checked synthetically, so they run in CI
without any model or large dataset. Two more tests are opt-in, gated on real data being present
locally: test_shipped_results_reproduce_published_median_and_mae (the sampled, checkpoint-based
reproduction; see magnet_benchmark_run.py) and test_exact_stats_table_reproduces_published_values_
exactly (the full, unsampled reproduction from data/magnet_test_predictions/, which is what the
shipped notebook actually uses -- see the module docstring's two reproduction paths).
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths

import magnet_benchmark as M  # noqa: E402


def test_collect_abs_errors_filters_by_element_and_nan():
    an = [np.array([1, 6, 1, 8])]
    pred = [np.array([10.0, 100.0, 11.0, np.nan])]
    dft = [np.array([10.1, 102.0, 10.9, 50.0])]
    h = M.collect_abs_errors(pred, dft, an, z=1)
    c = M.collect_abs_errors(pred, dft, an, z=6)
    assert np.allclose(np.sort(h), [0.1, 0.1])      # the two hydrogens
    assert np.allclose(c, [2.0])                    # the one carbon
    # oxygen has a NaN prediction -> dropped, so no oxygen errors survive
    assert M.collect_abs_errors(pred, dft, an, z=8).size == 0


def test_summarize_matches_numpy():
    e = np.array([0.1, 0.2, 0.3, 0.4])
    s = M.summarize(e)
    assert s["median_ae"] == pytest.approx(0.25)
    assert s["mae"] == pytest.approx(0.25)
    assert s["rmse"] == pytest.approx(np.sqrt(np.mean(e ** 2)))
    assert s["n"] == 4
    empty = M.summarize(np.array([]))
    assert empty["n"] == 0 and np.isnan(empty["mae"])


def test_error_table_both_nuclei():
    an = [np.array([1, 6])]
    pred = [np.array([10.0, 100.0])]
    dft = [np.array([10.5, 98.0])]
    t = M.error_table(pred, dft, an)
    assert t["1H"]["mae"] == pytest.approx(0.5)
    assert t["13C"]["mae"] == pytest.approx(2.0)


def test_filter_supported_drops_out_of_vocabulary():
    # second structure contains phosphorus (15), which MagNET was never trained on
    ans = [np.array([6, 1, 1]), np.array([6, 1, 15]), np.array([8, 1, 17])]
    geos = [np.zeros((3, 3))] * 3
    dfts = [np.zeros(3)] * 3
    a, g, d, dropped = M.filter_supported(ans, geos, dfts)
    assert dropped == 1
    assert len(a) == len(g) == len(d) == 2
    assert all(set(np.unique(x).tolist()) <= M.SUPPORTED_ELEMENTS for x in a)


RESULTS = os.path.join(HERE, "..", "..", "data", "magnet_benchmark", "performance_results.csv")


@pytest.mark.skipif(not os.path.exists(RESULTS), reason="performance_results.csv not present")
def test_shipped_results_reproduce_published_median_and_mae():
    """The shipped reproduced numbers match the published median and mean absolute error (the robust
    statistics). RMSE is intentionally not checked: it is outlier-dominated for the vibrated sets and
    needs the full test set (see the module docstring)."""
    import csv
    # tolerance per nucleus: 1H errors are ~0.03 ppm, 13C ~0.4 ppm, so allow a sampling margin
    TOL = {"1H": 0.015, "13C": 0.12}
    rows = list(csv.DictReader(open(RESULTS)))
    assert len(rows) == 2 * len(M.MODELS) * len(M.TEST_SETS)   # 16 rows
    for r in rows:
        pub_median, pub_mae, _ = M.PUBLISHED[r["nucleus"]][(r["model"], r["test_set"])]
        tol = TOL[r["nucleus"]]
        tag = f"{r['model']}/{r['test_set']}/{r['nucleus']}"
        assert abs(float(r["median_ae"]) - pub_median) < tol, f"{tag} median {r['median_ae']} vs {pub_median}"
        assert abs(float(r["mae"]) - pub_mae) < tol, f"{tag} mae {r['mae']} vs {pub_mae}"


def test_predictions_group_matches_documented_naming_scheme():
    # spot-check against the exact group names defined in magnet_benchmark.test_predictions_group
    assert M.test_predictions_group("MagNET", "vibrated_external", "13C") == "gasphasedft8k_C_pretrained_C_vib1"
    assert M.test_predictions_group("MagNET", "isolated_chloroform", "1H") == "solutesmd500isolated_chloroform_H_pretrained_H"
    assert M.test_predictions_group("MagNET-x", "stationary_internal", "1H") == "gasphaseinternal_H_chloroform_H_vib0"
    with pytest.raises(ValueError):
        M.test_predictions_group("MagNET", "not_a_real_test_set", "1H")


PREDICTIONS_H5 = paths.dataset_file("magnet_test_predictions", file=__file__)


@pytest.mark.skipif(not os.path.exists(PREDICTIONS_H5), reason="magnet_test_predictions.hdf5 not present")
def test_exact_stats_table_reproduces_published_values_exactly():
    """Unlike the sampled test above, this reads the FULL released test sets (no sampling), so median,
    MAE, and RMSE should all match the published SI values to float rounding, not just a sampling
    tolerance -- this is what the shipped si_table_s03_s04_performance.ipynb notebook actually asserts."""
    sys.path.insert(0, os.path.join(HERE, "..", "..", "data", "magnet_test_predictions"))
    import magnet_test_predictions_reader as R
    rows = M.exact_stats_table(PREDICTIONS_H5, R)
    assert len(rows) == 2 * len(M.MODELS) * len(M.TEST_SETS)
    for r in rows:
        pub_median, pub_mae, pub_rmse = M.PUBLISHED[r["nucleus"]][(r["model"], r["test_set"])]
        tag = f"{r['model']}/{r['test_set']}/{r['nucleus']}"
        assert r["median_ae"] == pytest.approx(pub_median, abs=1e-3), tag
        assert r["mae"] == pytest.approx(pub_mae, abs=1e-3), tag
        assert r["rmse"] == pytest.approx(pub_rmse, abs=1e-3), tag


def test_published_table_structure():
    # both nuclei, both models, all four test sets, triples of finite numbers
    for nucleus in ("1H", "13C"):
        for model in M.MODELS:
            for ts in M.TEST_SETS:
                triple = M.PUBLISHED[nucleus][(model, ts)]
                assert len(triple) == 3
                assert all(np.isfinite(v) and v > 0 for v in triple)
    # the 13C vibrated-internal RMSE is the documented outlier-dominated value
    assert M.PUBLISHED["13C"][("MagNET", "vibrated_internal")][2] == pytest.approx(7.1377, abs=1e-3)
