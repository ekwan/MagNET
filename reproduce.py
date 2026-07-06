#!/usr/bin/env python3
"""Regenerate every figure and table in the paper from the committed data.

Runs each analysis notebook headless (no display, no widgets). Every notebook reads the in-repo
data/<name>/ files and writes its figures to <notebook>/figures/ and its tables to
<notebook>/documents/. The notebooks are stored without outputs, so this script is how the figures
and tables are produced. No environment variables and no network are needed.

Usage:
    python reproduce.py                 # run all notebooks
    python reproduce.py fig3 s10        # run only notebooks whose path contains one of these strings
"""
import argparse
import glob
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("filters", nargs="*", help="only run notebooks whose path contains one of these")
    ap.add_argument("--timeout", type=int, default=900, help="per-cell timeout in seconds (default 900)")
    args = ap.parse_args()

    try:
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
    except ImportError:
        sys.exit("reproduce.py needs the analysis dependencies: pip install -r requirements.txt")

    notebooks = sorted(glob.glob(os.path.join(HERE, "analysis", "*", "*.ipynb")))
    if args.filters:
        notebooks = [p for p in notebooks if any(f in p for f in args.filters)]
    if not notebooks:
        sys.exit("no matching notebooks")

    print("running {} notebooks headless".format(len(notebooks)))
    n_ok = n_fail = 0
    for path in notebooks:
        rel = os.path.relpath(path, HERE)
        start = time.time()
        notebook = nbformat.read(path, as_version=4)
        # run each notebook from its own directory so its in-repo data and figures/ paths resolve
        executor = ExecutePreprocessor(timeout=args.timeout)
        try:
            executor.preprocess(notebook, {"metadata": {"path": os.path.dirname(path)}})
            print("  OK   ({:5.0f}s) {}".format(time.time() - start, rel))
            n_ok += 1
        except Exception as error:
            print("  FAIL ({:5.0f}s) {}: {}: {}".format(
                time.time() - start, rel, type(error).__name__, str(error)[:120]))
            n_fail += 1

    print("\n{} ok, {} failed".format(n_ok, n_fail))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
