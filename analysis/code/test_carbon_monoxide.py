"""Tests for analysis/code/carbon_monoxide.py. The figure's claim is that MagNET tracks
DFT near the equilibrium bond length but drifts in the non-equilibrium tails; that is checked against
the shipped scan data, along with the loaders and the histogram helper."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import carbon_monoxide as co  # noqa: E402


def test_comparison_loads_and_is_sorted():
    df = co.load_comparison()
    assert len(df) == 81
    assert list(df.columns) == ["bond_length", "gaussian_c", "magnet_c", "delta_c",
                                "gaussian_o", "magnet_o", "delta_o"]
    assert df["bond_length"].is_monotonic_increasing
    assert df["bond_length"].min() == pytest.approx(0.70, abs=1e-3)
    assert df["bond_length"].max() == pytest.approx(1.50, abs=1e-3)
    # delta is exactly MagNET minus DFT
    assert np.allclose(df["delta_c"], df["magnet_c"] - df["gaussian_c"])


def test_histogram_helper():
    centers, counts, bin_width = co.load_bond_length_histogram()
    assert len(centers) == len(counts) == 100
    assert bin_width > 0
    assert counts.sum() > 0           # there is real bond-length support behind the error panel


def test_error_grows_away_from_equilibrium():
    """The published claim: the carbon error is small near the equilibrium bond length and much
    larger in the stretched/compressed tails the model never trained on."""
    near, tails = co.equilibrium_vs_extreme_error()
    assert near < 30.0                # near-equilibrium error stays modest (about 19 ppm)
    assert tails > 100.0              # tails blow up (about 185 ppm)
    assert tails > 5 * near           # the tails are far worse, which is the point of the figure
