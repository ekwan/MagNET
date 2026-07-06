import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import paths as P  # noqa: E402


def test_repo_root_resolves_this_file_two_levels_up():
    # requirements.txt lives only at the repo root, so its presence is what proves repo_root landed
    # in the right place; the checkout folder's name is not part of the contract (a CI checkout, for
    # example, names it after the GitHub repo, not this developer's local folder name).
    root = P.repo_root(__file__)
    assert os.path.isfile(os.path.join(root, "requirements.txt"))
    assert os.path.basename(root) != "code"  # sanity: didn't stop one level too early


def test_repo_root_from_synthetic_tree(tmp_path):
    fake_module = tmp_path / "analysis" / "code" / "x.py"
    fake_module.parent.mkdir(parents=True)
    fake_module.write_text("")
    assert P.repo_root(str(fake_module)) == str(tmp_path)


def test_ensure_on_path_inserts_at_front_and_dedupes():
    # sys.path is a single shared, mutable, process-global list -- other test modules in the same
    # pytest session insert their own entries into the REAL sys.path (some also at index 0), so
    # asserting against the real list is flaky no matter how carefully this test brackets its own
    # calls. Swap in a private list for the duration of the test instead, so ensure_on_path's
    # "insert at index 0, don't duplicate" contract can be checked hermetically.
    real_sys_path = sys.path
    sys.path = ["/some/other/existing/entry"]
    try:
        target = os.path.join(P.repo_root(__file__), "data", "delta22")
        P.ensure_on_path("data", "delta22", file=__file__)
        assert sys.path[0] == target  # inserts at the FRONT, so local modules can shadow the rest
        assert sys.path.count(target) == 1
        # inserting the same target again must not duplicate it
        P.ensure_on_path("data", "delta22", file=__file__)
        assert sys.path.count(target) == 1
        assert sys.path[0] == target
    finally:
        sys.path = real_sys_path


def test_ensure_on_path_needs_file_or_root():
    try:
        P.ensure_on_path("data")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_dataset_file_resolves_in_repo(tmp_path):
    got = P.dataset_file("delta22", root=str(tmp_path))
    assert got == os.path.join(str(tmp_path), "data", "delta22", "delta22.hdf5")


def test_dataset_file_derives_root_from_file(tmp_path):
    # the form most rewired modules use: dataset_file("<name>", file=__file__). A module at
    # analysis/code/x.py must resolve the file two directories up, at <repo>/data/...
    fake = tmp_path / "analysis" / "code" / "x.py"
    fake.parent.mkdir(parents=True)
    fake.write_text("")
    got = P.dataset_file("delta22", file=str(fake))
    assert got == os.path.join(str(tmp_path), "data", "delta22", "delta22.hdf5")


def test_dataset_file_custom_filename_and_needs_a_root():
    got = P.dataset_file("applications", "applications_md_geometries.hdf5", root="/r")
    assert got == os.path.join("/r", "data", "applications", "applications_md_geometries.hdf5")
    try:
        P.dataset_file("delta22")   # no root, no file
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_checkpoints_root(monkeypatch, tmp_path):
    # point the resolver at an empty tree, so its model_checkpoints/ is absent (as in CI)
    empty = tmp_path / "repo"
    empty.mkdir()
    monkeypatch.setattr(P, "repo_root", lambda _file: str(empty))
    assert P.checkpoints_root() is None
    try:
        P.checkpoints_root(required=True)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
    # in-repo location: model_checkpoints/ present at the repo root
    (empty / "model_checkpoints").mkdir()
    assert P.checkpoints_root() == os.path.join(str(empty), "model_checkpoints")
    assert P.checkpoints_root(required=True) == os.path.join(str(empty), "model_checkpoints")
