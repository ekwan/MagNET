"""Repo-root and sys.path resolution for the analysis code.

Bootstrap-only: imports nothing from analysis/code/ (it is what makes that folder importable), so it
stays at analysis/code/ root. repo_root(file) assumes the caller sits two dirs below the repo root
(analysis/code/x.py -> ../..), which holds for every module here.
"""
import os
import sys

def repo_root(file):
    """The release repo root, given a module's __file__."""
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(file)), "..", ".."))


def dataset_file(name, filename=None, *, file=None, root=None):
    """Absolute path to a large released dataset file, in the repo at data/<name>/<filename>.

    `filename` defaults to "<name>.hdf5". Give `root` (the repo root) or `file` (a module __file__,
    from which the repo root is derived) so the file can be located. Only the large, git-ignored
    data files go through here; small companion files (readers, spreadsheets) always live in the
    repo and are addressed directly. The Hugging Face checkout carries these files via Git LFS.
    """
    if filename is None:
        filename = "{}.hdf5".format(name)
    if root is None:
        if file is None:
            raise ValueError(
                "dataset_file needs root=<repo root> or file=__file__ to locate {!r}".format(filename)
            )
        root = repo_root(file)
    return os.path.join(root, "data", name, filename)


def checkpoints_root(*, required=False):
    """Absolute path to the in-repo model_checkpoints/ folder holding the trained weights.

    Returns the path when it exists, else None so callers can skip checkpoint-dependent work, unless
    required=True, which raises. The Hugging Face checkout carries the weights via Git LFS; they are
    git-ignored on GitHub, so this is None in CI.
    """
    in_repo = os.path.join(repo_root(__file__), "model_checkpoints")
    if os.path.exists(in_repo):
        return in_repo
    if required:
        raise RuntimeError(
            "No model_checkpoints/ found. It ships in the repo via Git LFS in the Hugging Face "
            "checkout (hf download ekwan16/MagNET --include 'model_checkpoints/*')."
        )
    return None


def ensure_on_path(*relative_parts, file=None, root=None):
    """Inserts os.path.join(root, *relative_parts) at the front of sys.path, if not already present.

    Pass either `file` (typically __file__, resolved via repo_root) or `root` directly.
    """
    if root is None:
        if file is None:
            raise ValueError("ensure_on_path needs either file=__file__ or root=<repo root>")
        root = repo_root(file)
    target = os.path.join(root, *relative_parts)
    if target not in sys.path:
        sys.path.insert(0, target)
    return target
