"""Reader for the published MagNET-Zero/MagNET-PCM linear-scaling tables in this folder.

These two CSV files (one per nucleus) ARE Supporting Information Tables S10 (proton) and S11 (carbon),
verbatim: the reflection-symmetrized (n_passes=10, symmetrize=True) per-solvent coefficients that turn
a MagNET-Zero gas-phase shielding plus a MagNET-PCM chloroform correction into a predicted shift.
analysis/code/scaling_factors.py derives them; see its module docstring for the method.

Each row is one solvent; columns are intercept, stationary (the MagNET-Zero slope), and pcm (the
MagNET-PCM-correction slope). Predicted shift = intercept + stationary * shielding + pcm * correction.
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NUCLEI = ("H", "C")


def _csv_path(nucleus):
    if nucleus not in NUCLEI:
        raise ValueError(f"nucleus must be one of {NUCLEI}, got {nucleus!r}")
    return os.path.join(HERE, f"scaling_factors_symmetrized_{nucleus}.csv")


def load_symmetrized_tables():
    """Returns {"H": DataFrame, "C": DataFrame}, each indexed by solvent with columns
    intercept / stationary / pcm."""
    return {n: pd.read_csv(_csv_path(n)).set_index("solvent") for n in NUCLEI}


def predict_shift(table, solvent, shielding, correction):
    """Chemical shift for one site: intercept + stationary * shielding + pcm * correction, using
    `table` (one nucleus' DataFrame from load_symmetrized_tables) and its `solvent` row."""
    row = table.loc[solvent]
    return float(row["intercept"] + row["stationary"] * shielding + row["pcm"] * correction)
