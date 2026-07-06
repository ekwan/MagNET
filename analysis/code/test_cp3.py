"""Tests for analysis/code/cp3.py. The fraction-below-DFT statistic is a quantitative claim in the
paper (36% of proton sites), so it is checked against the shipped Goodman CP3 data, along with the
zone logic on a synthetic array."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cp3  # noqa: E402


def test_fraction_below_dft_synthetic():
    # 10 values: 2 below the experimental floor, 5 in [0.02, 0.10), 3 at/above 0.10. Equal-width
    # bins make the area fraction in [0.02, 0.10] equal to the count fraction there.
    x = np.array([0.005, 0.015, 0.03, 0.04, 0.05, 0.06, 0.07, 0.12, 0.2, 0.3])
    frac = cp3.fraction_below_dft(x, bins=100)
    assert frac == pytest.approx(0.5, abs=0.02)   # 5 of 10 in [0.02, 0.10)


def test_zone_boundaries():
    assert cp3._zone(0.01) == "below_experimental"
    assert cp3._zone(0.05) == "below_dft"
    assert cp3._zone(0.2) == "dft_zone"
    assert cp3._zone(0.4) == "large"


REAL = os.path.join(HERE, "..", "..", "data", "cp3", "goodman2009_cp3.xlsx")


@pytest.mark.skipif(not os.path.exists(REAL), reason="goodman2009_cp3.xlsx not present")
def test_reproduces_published_36_percent():
    variations = cp3.load_variations(REAL, nucleus="H")
    assert len(variations) == 168                      # the 1H sites in the CP3 set
    frac = cp3.fraction_below_dft(variations)
    assert frac == pytest.approx(0.36, abs=0.01)       # paper: 36% of proton sites below DFT
