"""Shared helpers for the ``build_nb_*.py`` notebook generators.

Each ``build_nb_<x>.py`` assembles a notebook from a list of cells built with :func:`md` and
:func:`code`, then writes it with :func:`save_notebook`. The notebook metadata written here is
intentionally minimal -- ``jupyter nbconvert --execute`` fills in the full ``language_info``/kernel
block when the notebook is run.
"""
import nbformat


def md(source):
    """Return a markdown cell from a string, with leading/trailing blank lines stripped."""
    return nbformat.v4.new_markdown_cell(source.strip("\n"))


def code(source):
    """Return a code cell from a string, with leading/trailing blank lines stripped."""
    return nbformat.v4.new_code_cell(source.strip("\n"))


def save_notebook(cells, path):
    """Write a notebook made of ``cells`` to ``path`` and print a confirmation line."""
    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {"language_info": {"name": "python"}}
    with open(path, "w") as f:
        nbformat.write(nb, f)
    print("wrote", path)
