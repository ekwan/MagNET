"""Tests for build_composite_model.py: the composite_model/ group in applications.hdf5 reproduces
from the shipped delta-22 data, so the natural-products coefficients need no private tree or
environment variables. Skipped when the large hdf5 files are absent."""
import os
import random
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in ("analysis/code", "analysis/code/shared", "data/delta22", "data/applications"):
    sys.path.insert(0, os.path.join(REPO, _p))

import build_composite_model as B  # noqa: E402
import paths  # noqa: E402

DELTA22 = paths.dataset_file("delta22", root=REPO)
APPLICATIONS = paths.dataset_file("applications", root=REPO)
_HAVE_DATA = os.path.exists(DELTA22) and os.path.exists(APPLICATIONS)

# The fixed-point integer encoding (int32 x 1e4) sets the reproduction floor; recipe errors are ~1.
ENCODING_TOL = 5e-3


def _reader():
    from applications_reader import Applications
    return Applications(APPLICATIONS)


@pytest.mark.skipif(not _HAVE_DATA, reason="delta22.hdf5 / applications.hdf5 not present")
def test_ols_and_pcm_reproduce():
    """Every OLS formula (all solvents, both nuclei) and both PCM-conversion tables reproduce."""
    reader = _reader()
    full_tab, _fit_tab, _apply_tab, factors = B._load_tables()
    worst = 0.0
    for formula in reader.composite_formulas("ols_coefficients"):
        regressors = B._regressor_columns(formula)
        for reg in regressors:
            B._ensure_column(full_tab, reg)
        for nucleus in ("H", "C"):
            stored = reader.ols_coefficients(formula, nucleus).set_index("parameter")
            for solvent in stored.columns:
                sub = full_tab[(full_tab["nucleus"] == nucleus) & (full_tab["solvent"] == solvent)]
                got = B._decompose(B._fit(sub, regressors), regressors, solvent, nucleus, factors)
                worst = max(worst, max(abs(float(stored.loc[p, solvent]) - got[p]) for p in stored.index))
    assert worst < ENCODING_TOL, f"OLS max diff {worst}"

    for nucleus in ("H", "C"):
        stored = reader.pcm_conversion_factors(nucleus).set_index("solvent")["pcm_conversion_factor"]
        got = B._pcm_table(factors, nucleus).set_index("solvent")["pcm_conversion_factor"]
        assert np.abs(got - stored.reindex(got.index)).max() < 1e-9


@pytest.mark.skipif(not _HAVE_DATA, reason="delta22.hdf5 / applications.hdf5 not present")
def test_bootstrap_and_rmse_reproduce():
    """Bootstrap coefficients and RMSE distributions reproduce on a sample of seeds (the fixed-seed
    resampler makes every seed deterministic, so a sample is representative). This is the fast CI
    check; build_composite_model.main() re-fits all 1000 seeds and is the authoritative full pass."""
    reader = _reader()
    _full_tab, fit_tab, apply_tab, factors = B._load_tables()
    seeds = range(0, 1000, 97)

    worst = 0.0
    for formula in reader.composite_formulas("bootstrap_coefficients"):
        regressors = B._regressor_columns(formula)
        for reg in regressors:
            B._ensure_column(fit_tab, reg)
        for nucleus in ("H", "C"):
            try:
                stored = reader.bootstrap_coefficients(formula, nucleus)
            except KeyError:
                continue
            params = [c for c in stored.columns if c not in ("solvent", "seed")]
            for solvent in stored["solvent"].unique():
                for seed in seeds:
                    row = stored[(stored["solvent"] == solvent) & (stored["seed"] == seed)]
                    if not len(row):
                        continue
                    got = B._decompose(B._fit(B._bootstrap_sample(fit_tab, nucleus, solvent, seed),
                                              regressors), regressors, solvent, nucleus, factors)
                    for p in params:
                        key = "intercept" if p == "Intercept" else p
                        worst = max(worst, abs(float(row[p].iloc[0]) - got[key]))
    assert worst < ENCODING_TOL, f"bootstrap max diff {worst}"

    worst = 0.0
    for nucleus in ("H", "C"):
        stored = reader.rmse_distribution(nucleus)
        for formula in stored["formula"].unique():
            sub = stored[stored["formula"] == formula]
            for solvent in sub["solvent"].unique():
                for seed in seeds:
                    row = sub[(sub["solvent"] == solvent) & (sub["seed"] == seed)]
                    if not len(row):
                        continue
                    got = B._rmse(fit_tab, apply_tab, formula, nucleus, solvent, seed, factors)
                    worst = max(worst, abs(float(row["Bootstrap_RMSE"].iloc[0]) - got))
    assert worst < ENCODING_TOL, f"rmse max diff {worst}"


@pytest.mark.skipif(not _HAVE_DATA, reason="delta22.hdf5 / applications.hdf5 not present")
def test_bootstrap_is_deterministic():
    """The resampler reseeds internally, so a given seed draws the same solutes no matter the
    external RNG state going in."""
    fit_tab = B._load_tables()[1]
    random.seed(123)
    a = B._bootstrap_sample(fit_tab, "C", "chloroform", 7)
    random.seed(456)
    b = B._bootstrap_sample(fit_tab, "C", "chloroform", 7)
    assert list(a["solute"]) == list(b["solute"])


@pytest.mark.skipif(not _HAVE_DATA, reason="delta22.hdf5 / applications.hdf5 not present")
def test_write_roundtrips(tmp_path):
    """write() serializes the regenerated tables into an hdf5 file that reads back unchanged through
    Applications, so --write cannot silently corrupt the shipped file's composite_model group."""
    from applications_reader import Applications
    regenerated = B.regenerate(_reader())
    scratch = str(tmp_path / "cm.hdf5")
    B.write(scratch, regenerated)
    worst = B.verify(Applications(scratch), regenerated)
    assert worst < 1e-6, f"write/read round-trip diff {worst}"
