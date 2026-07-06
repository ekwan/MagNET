"""Render the MagNET API docs into api_docs/ with pdoc.

    python build_api_docs.py

Documents the magnet package's public modules and every analysis module under analysis/code,
skipping test files. The analysis modules import their siblings and the dataset readers by bare
name, so their directories are added to sys.path first.
"""
import glob
import os
import sys

import pdoc

HERE = os.path.dirname(os.path.abspath(__file__))


def _import_dirs():
    """Directories the analysis modules import from: their own folder, shared/, and each
    data/<dataset>/ folder that holds a reader module."""
    dirs = [os.path.join(HERE, "analysis", "code"),
            os.path.join(HERE, "analysis", "code", "shared")]
    dirs += sorted(os.path.dirname(p) for p in glob.glob(os.path.join(HERE, "data", "*", "*_reader.py")))
    return dirs


def _modules():
    """The .py files to document: the six public magnet modules, then the analysis modules."""
    mods = [os.path.join(HERE, "magnet", name + ".py")
            for name in ("__init__", "api", "inference", "model", "run_magnet", "scaling")]
    analysis = glob.glob(os.path.join(HERE, "analysis", "code", "*.py"))
    analysis += glob.glob(os.path.join(HERE, "analysis", "code", "shared", "*.py"))
    for path in sorted(analysis):
        name = os.path.basename(path)
        if name.startswith("test_"):
            continue
        mods.append(path)
    return mods


def main():
    for d in _import_dirs():
        sys.path.insert(0, d)
    pdoc.render.configure(template_directory=os.path.join(HERE, "pdoc_templates"))
    pdoc.pdoc(*_modules(), output_directory=os.path.join(HERE, "api_docs"))
    print("api docs written to", os.path.join(HERE, "api_docs", "index.html"))


if __name__ == "__main__":
    main()
