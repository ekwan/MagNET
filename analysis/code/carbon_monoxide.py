"""Figure S12: MagNET vs DFT carbon-13 shielding of carbon monoxide across bond lengths.

The CO bond is stretched/compressed; at each length the 13C shielding is computed by DFT
(PBE1PBE/pcSseg-1, gas, Gaussian) and by the MagNET foundation model. This module holds only the
loading; the plot is drawn inline in the figure notebook.

Two files ship alongside:
- `co_magnet_vs_gaussian.csv` - one row per bond length: DFT and MagNET shieldings and their
  difference, for carbon and oxygen.
- `co_bond_lengths_histogram.csv` - carbon-oxygen bond-length distribution across real molecules,
  as (bin left edge, count) rows, for the shaded histogram.
"""
import os

import pandas as pd

from stats import mae

HERE = os.path.dirname(os.path.abspath(__file__))
COMPARISON_CSV = os.path.join(HERE, "..", "..", "data", "carbon_monoxide", "co_magnet_vs_gaussian.csv")
HISTOGRAM_CSV = os.path.join(HERE, "..", "..", "data", "carbon_monoxide", "co_bond_lengths_histogram.csv")


def load_comparison(path=COMPARISON_CSV):
    """The bond-length scan as a DataFrame, sorted by bond length. Columns: `bond_length` (angstrom),
    and for each of carbon and oxygen the `gaussian`, `magnet`, and `delta` (MagNET minus DFT)
    shieldings in ppm."""
    raw = pd.read_csv(path)
    return pd.DataFrame(
        {
            "bond_length": raw["bond_length_angstrom"].astype(float),
            "gaussian_c": raw["gaussian_shielding_c_ppm"].astype(float),
            "magnet_c": raw["magnet_shielding_c_ppm"].astype(float),
            "delta_c": raw["delta_c_ppm"].astype(float),
            "gaussian_o": raw["gaussian_shielding_o_ppm"].astype(float),
            "magnet_o": raw["magnet_shielding_o_ppm"].astype(float),
            "delta_o": raw["delta_o_ppm"].astype(float),
        }
    ).sort_values("bond_length", ignore_index=True)


def load_bond_length_histogram(path=HISTOGRAM_CSV):
    """The sampled carbon-oxygen bond-length distribution as `(centers, counts, bin_width)`: the bin
    centers in angstrom, the count in each bin, and the (uniform) bin width. The stored table holds
    left bin edges and counts."""
    table = pd.read_csv(path)
    left_edges = table["bin_left_angstrom"].to_numpy(dtype=float)
    counts = table["count"].to_numpy(dtype=float)
    bin_width = float(left_edges[1] - left_edges[0]) if len(left_edges) > 1 else 0.01
    centers = left_edges + bin_width / 2.0
    return centers, counts, bin_width


def equilibrium_vs_extreme_error(comparison=None, window=(1.0, 1.3), extreme=(0.9, 1.4)):
    """The figure's claim as two numbers: the mean absolute carbon error (ppm) for bond lengths
    inside the near-equilibrium `window`, and for the non-equilibrium tails outside `extreme`. The
    second is far larger, which is the whole point of the panel."""
    df = load_comparison() if comparison is None else comparison
    near = df[(df["bond_length"] >= window[0]) & (df["bond_length"] <= window[1])]
    tails = df[(df["bond_length"] < extreme[0]) | (df["bond_length"] > extreme[1])]
    return mae(near["delta_c"], 0), mae(tails["delta_c"], 0)


