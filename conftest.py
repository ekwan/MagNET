# The magnet/ model package needs the heavy PyTorch / PyTorch Geometric stack
# (see magnet/requirements.txt), which the default dataset-reader test environment does not
# install. When torch is absent, skip collecting the magnet tests entirely so the lightweight
# CI stays green. They run locally once the torch stack from magnet/requirements.txt is present.
#
# This is needed (not just the importorskip guards inside test_magnet.py) because that test file
# lives inside the magnet package, so pytest imports magnet/__init__.py during collection, which
# imports torch before any in-test skip can run.
import os
import sys

# The analysis code lives in analysis/code/ (importable modules) and analysis/code/shared/ (the
# shared utilities). Put both on sys.path so tests anywhere resolve bare imports like `import
# delta22` and `from stats import rmse`, no matter which folder the test file itself sits in.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "analysis", "code"),
           os.path.join(_HERE, "analysis", "code", "shared")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

collect_ignore = []
collect_ignore_glob = []
try:
    import torch  # noqa: F401
except ImportError:
    collect_ignore = ["magnet"]
    collect_ignore_glob = ["magnet/*"]


# The reproduce tests (test_reproduce.py) execute every notebook and take minutes; they are opt-in so
# a plain `pytest` stays fast. Run them with `pytest --run-notebooks` (or `pytest -m reproduce`).
def pytest_addoption(parser):
    parser.addoption("--run-notebooks", action="store_true", default=False,
                     help="run the reproduce tests that execute every analysis notebook")


def pytest_configure(config):
    config.addinivalue_line("markers", "reproduce: executes a full analysis notebook (slow, opt-in)")


def pytest_collection_modifyitems(config, items):
    import pytest
    if config.getoption("--run-notebooks"):
        return
    skip = pytest.mark.skip(reason="notebook reproduce test; pass --run-notebooks to run")
    for item in items:
        if "reproduce" in item.keywords:
            item.add_marker(skip)
