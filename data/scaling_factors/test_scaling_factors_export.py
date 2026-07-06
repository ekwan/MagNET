"""Structural checks for the shipped symmetrized scaling tables (data/scaling_factors/).

These run on the small CSVs that ship in git, so they need no download. The file is named
test_scaling_factors_export.py (not test_scaling_factors.py) on purpose: analysis/code/ already has a
test_scaling_factors.py, and pytest's default import mode cannot collect two test modules that share
a basename.
"""
import os
import sys

import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import scaling_factors_reader as R  # noqa: E402


def test_both_tables_load_with_expected_shape_and_columns():
    tables = R.load_symmetrized_tables()
    assert set(tables) == {"H", "C"}
    for nucleus, df in tables.items():
        assert df.shape == (12, 3), nucleus
        assert list(df.columns) == ["intercept", "stationary", "pcm"], nucleus
        assert "chloroform" in df.index
        assert df.notnull().all().all(), nucleus


def test_slopes_are_negative_and_intercepts_in_range():
    tables = R.load_symmetrized_tables()
    # the gas-phase (stationary) slope is a shielding-to-shift conversion, always near -1
    for nucleus, df in tables.items():
        assert (df["stationary"] < 0).all(), nucleus
    # proton reference intercept sits near a bare-proton shielding (~31 ppm), carbon much higher
    assert 25 < tables["H"]["intercept"].mean() < 35
    assert 150 < tables["C"]["intercept"].mean() < 200


def test_predict_shift_uses_the_documented_linear_formula():
    # a synthetic one-row table with round coefficients, so the expected value is hand-computed
    # independently of predict_shift's own arithmetic (this catches a sign flip or a swap of the
    # stationary/pcm coefficients, which recomputing the same formula inline would not):
    #   10 + (-1)*5 + 2*3 = 11
    table = pd.DataFrame({"intercept": [10.0], "stationary": [-1.0], "pcm": [2.0]}, index=["x"])
    table.index.name = "solvent"
    assert R.predict_shift(table, "x", shielding=5.0, correction=3.0) == pytest.approx(11.0)


def test_reader_rejects_unknown_nucleus():
    with pytest.raises(ValueError):
        R._csv_path("X")
