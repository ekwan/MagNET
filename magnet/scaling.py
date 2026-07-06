"""Turn MagNET-Zero / MagNET-PCM shieldings into the chemical shifts a chemist measures, using the
paper's published per-solvent linear scaling (SI Table S10 for proton, S11 for carbon).

This is a tiny, self-contained copy (numpy only), so `from magnet import published_scaling_tables,
predict_shift` works with no data download and no extra path setup. The full derivation of these
tables from the delta-22 data lives in analysis/code/scaling_factors.py, and a test there checks the
two copies agree so they cannot drift.

The prediction equation, for one nucleus and one solvent, is

    shift = intercept + stationary * sigma_zero + pcm * delta_pcm

where sigma_zero is the MagNET-Zero gas-phase shielding (compute_MagNET_Zero_shieldings) and delta_pcm
is the MagNET-PCM chloroform correction (compute_MagNET_PCM_corrections). You always pass the
chloroform correction; the per-solvent scaling is already folded into the `pcm` coefficient.
"""
import csv
import io

import numpy as np

# Published SI Tables S10 (proton) and S11 (carbon), verbatim (reflection-symmetrized: n_passes=10,
# symmetrize=True). Columns: intercept, stationary, pcm. Kept byte-for-byte in sync with
# data/scaling_factors/scaling_factors_symmetrized_{H,C}.csv by test_scaling_factors.py.
_PUBLISHED_TABLE_CSV = {
    "H": """solvent,intercept,stationary,pcm
tetrahydrofuran,31.319664,-0.980082,-0.781373
dichloromethane,31.374485,-0.979760,-0.806215
chloroform,31.291983,-0.975682,-0.851715
toluene,31.714067,-0.996613,1.948607
benzene,31.984616,-1.005170,2.239317
chlorobenzene,31.677950,-0.993715,0.830908
acetone,31.510823,-0.987188,-1.229850
dimethylsulfoxide,31.597261,-0.991031,-1.348009
acetonitrile,31.494061,-0.985628,-0.969617
trifluoroethanol,30.873172,-0.959885,-0.955808
methanol,31.253627,-0.976349,-1.246680
TIP4P,31.356622,-0.978710,-1.475907
""",
    "C": """solvent,intercept,stationary,pcm
tetrahydrofuran,171.052274,-0.918997,-1.069506
dichloromethane,171.507154,-0.921165,-1.115013
chloroform,171.726919,-0.924228,-0.936801
toluene,171.689276,-0.924983,-0.618407
benzene,171.965715,-0.927029,-0.593714
chlorobenzene,171.073820,-0.921235,-0.997638
acetone,171.305850,-0.919608,-1.239500
dimethylsulfoxide,170.596065,-0.918672,-1.296497
acetonitrile,171.877492,-0.922626,-1.287661
trifluoroethanol,174.235231,-0.939011,-1.290071
methanol,172.423799,-0.927563,-1.288843
TIP4P,173.692750,-0.937848,-1.342659
""",
}


def published_scaling_tables():
    """The published scaling tables as {"H": table, "C": table}. Each table maps a solvent name to a
    {"intercept", "stationary", "pcm"} dict of coefficients. Pass one table and a solvent name to
    predict_shift. Water is named "TIP4P"."""
    tables = {}
    for nucleus, text in _PUBLISHED_TABLE_CSV.items():
        table = {}
        for row in csv.DictReader(io.StringIO(text.strip())):
            solvent = row.pop("solvent")
            table[solvent] = {name: float(value) for name, value in row.items()}
        tables[nucleus] = table
    return tables


def predict_shift(table, solvent, magnet_zero_shielding, magnet_pcm_chloroform_correction):
    """Chemical shift(s) from MagNET-Zero / MagNET-PCM outputs, for one nucleus and solvent.

    table: one nucleus's table from published_scaling_tables(). solvent: a solvent name (water is
    "TIP4P"). magnet_zero_shielding and magnet_pcm_chloroform_correction are scalars or arrays.
    """
    row = table[solvent]
    return (row["intercept"]
            + row["stationary"] * np.asarray(magnet_zero_shielding, dtype=float)
            + row["pcm"] * np.asarray(magnet_pcm_chloroform_correction, dtype=float))
