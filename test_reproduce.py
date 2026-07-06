"""Reproduction check: every analysis notebook executes headless.

This follows the repo's real-data test pattern: it SKIPS when the datasets are not present (a fresh
CI checkout has none of the big `.hdf5` files) and RUNS locally once they are downloaded. So it never
slows or breaks CI, but a `pytest` on a machine with the data verifies the whole figure/table pipeline.
The human-facing entrypoint is reproduce.py; this is the pytest wrapper.
"""
import glob
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOKS = sorted(glob.glob(os.path.join(HERE, "analysis", "*", "*.ipynb")))

# Sentinel for "the datasets have been downloaded": delta22 is read by many notebooks. os.path.exists
# follows symlinks, so a working in-repo symlink into the big-file source counts as present, while a
# fresh CI checkout (no file, or a dangling link) does not.
_HAVE_DATA = os.path.exists(os.path.join(HERE, "data", "delta22", "delta22.hdf5"))


@pytest.mark.reproduce
@pytest.mark.skipif(not _HAVE_DATA, reason="datasets not downloaded (hf download ekwan16/MagNET)")
@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=[os.path.relpath(p, HERE) for p in NOTEBOOKS])
def test_notebook_reproduces(notebook):
    nbformat = pytest.importorskip("nbformat")
    nbconvert = pytest.importorskip("nbconvert.preprocessors")  # requires Jupyter
    executor = nbconvert.ExecutePreprocessor(timeout=900)
    nb = nbformat.read(notebook, as_version=4)
    # run from the notebook's own directory so its in-repo data and figures/ paths resolve
    executor.preprocess(nb, {"metadata": {"path": os.path.dirname(notebook)}})
