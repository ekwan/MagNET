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

# Published SI Tables S10 (proton) and S11 (carbon), verbatim. Columns: intercept, stationary, pcm.
_PUBLISHED_TABLE_CSV = {
    "H": """solvent,intercept,stationary,pcm
tetrahydrofuran,31.321785,-0.980175,-0.786672
dichloromethane,31.377310,-0.979870,-0.808573
chloroform,31.294992,-0.975794,-0.852690
toluene,31.716957,-0.996733,1.944856
benzene,31.987569,-1.005289,2.236515
chlorobenzene,31.681438,-0.993846,0.830236
acetone,31.512157,-0.987261,-1.238414
dimethylsulfoxide,31.598698,-0.991103,-1.355248
acetonitrile,31.496110,-0.985716,-0.974465
trifluoroethanol,30.876004,-0.959997,-0.958852
methanol,31.256030,-0.976446,-1.250121
TIP4P,31.359134,-0.978809,-1.478298
""",
    "C": """solvent,intercept,stationary,pcm
tetrahydrofuran,171.054488,-0.919000,-1.069509
dichloromethane,171.509389,-0.921167,-1.115015
chloroform,171.728797,-0.924231,-0.936804
toluene,171.690907,-0.924987,-0.618410
benzene,171.967177,-0.927033,-0.593716
chlorobenzene,171.075910,-0.921238,-0.997641
acetone,171.308023,-0.919610,-1.239503
dimethylsulfoxide,170.598425,-0.918674,-1.296501
acetonitrile,171.879839,-0.922629,-1.287664
trifluoroethanol,174.237671,-0.939012,-1.290073
methanol,172.426161,-0.927566,-1.288847
TIP4P,173.696379,-0.937854,-1.342668
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
