"""Figure 1A data and analysis: informative 1H shift differences are too small for DFT to resolve.

Loads per-site 1H shift spread across the Goodman CP3 stereoisomers and classifies each site against
the experimental noise floor (0.02 ppm) and DFT's resolving power (0.10 ppm); ~36% fall in between.
Data: `goodman2009_cp3.xlsx` (Goodman, J. Org. Chem. 2009, 74, 4597), shipped alongside. Plotting is
in the figure notebook.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CP3_XLSX = os.path.join(HERE, "..", "..", "data", "cp3", "goodman2009_cp3.xlsx")

# accuracy thresholds in ppm: the experimental noise floor, the DFT resolving limit, and the cutoff
# above which a variation counts as large. These define the zones a site's variation falls into.
EXPERIMENTAL_LIMIT = 0.02
DFT_LIMIT = 0.10
LARGE_VARIATION = 0.30


def load_variations(path=CP3_XLSX, nucleus="H"):
    """The per-site chemical-shift standard deviations across the CP3 stereoisomers for one nucleus
    ("H" or "C"), as a 1D array of finite values (ppm)."""
    df = pd.read_excel(path)
    x = df[df["nucleus"] == nucleus]["stdev"].to_numpy(dtype=float)
    return x[np.isfinite(x)]


def _zone(value):
    """Classify one variation (ppm) into its accuracy zone: within experimental noise, informative
    but below the DFT resolving limit, resolvable by DFT, or large."""
    if value < EXPERIMENTAL_LIMIT:
        return "below_experimental"
    if value < DFT_LIMIT:
        return "below_dft"
    if value < LARGE_VARIATION:
        return "dft_zone"
    return "large"


def fraction_below_dft(variations, lo=EXPERIMENTAL_LIMIT, hi=DFT_LIMIT, bins=30):
    """The fraction of the variation histogram's area between the experimental floor and the DFT
    limit: the sites whose shift variation is informative but too small for DFT to resolve (the
    paper's 36%). Computed as histogram area in [lo, hi] over total area, splitting the bins that
    straddle a boundary, exactly as the figure does."""
    edges = np.linspace(0.0, float(variations.max()), bins + 1)
    counts, edge = np.histogram(variations, bins=edges)
    total = window = 0.0
    for height, left, right in zip(counts, edge[:-1], edge[1:]):
        total += height * (right - left)
        overlap = max(0.0, min(right, hi) - max(left, lo))
        window += height * overlap
    return window / total if total > 0 else float("nan")
