"""Tests for the magnet package.

The synthetic tests build a tiny EquiformerV2 model (no checkpoint, no dataset) and run a
forward pass, so they exercise the whole package wiring and run in CI. The tests that need the
released checkpoints / reference data are gated with skipif and run only when those files are
present in the repo (model_checkpoints/ and data/<name>/, carried by the Hugging Face checkout).
"""
import os
import numpy as np
import pytest

# The model needs the heavy PyTorch / PyG stack (see requirements.txt). When it is not
# installed (e.g. the dataset-reader CI, which installs only numpy/h5py/pandas), skip the whole
# module cleanly instead of erroring at import.
pytest.importorskip("torch")
pytest.importorskip("torch_scatter")
pytest.importorskip("torch_cluster")
pytest.importorskip("torch_geometric")
pytest.importorskip("magnet")

from magnet.model import MagNET_Lightning
from magnet.inference import predict_shieldings

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
_CKPT_ROOT = os.path.join(_REPO, "model_checkpoints")  # absent on GitHub (git-ignored); every skipif below is then False
CKPT = os.path.join(_CKPT_ROOT, "MagNET-Zero")
# The natural-products reference shieldings are a tiny (54 KB) committed test fixture (test_data/), so
# the reproduce/symmetrize tests need only the released checkpoints, not a separate reference download.
NP_H5 = os.path.join(_HERE, "test_data", "magnet_zero_and_pcm_shieldings.h5")


def _dataset_file(name):
    """Resolve a released dataset hdf5 the way analysis/code/paths.py does, but without importing it
    (this package is standalone): the in-repo data/<name>/<name>.hdf5, carried by the Hugging Face
    checkout via Git LFS."""
    return os.path.join(_REPO, "data", name, name + ".hdf5")

TINY_PARAMS = dict(
    model_type="EquiformerV2", max_neighbors=20, max_radius=5.0, max_num_elements=20,
    num_layers=1, sphere_channels=8, attn_hidden_channels=8, num_heads=1,
    attn_alpha_channels=8, attn_value_channels=8, ffn_hidden_channels=16,
    lmax_list=[2], mmax_list=[2], grid_resolution=14, weight_init="uniform", lr=1e-4,
)

# a small tetrahedral-ish CH4 so the graph has edges
METHANE_Z = np.array([6, 1, 1, 1, 1])
METHANE_XYZ = np.array([[0, 0, 0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63],
                        [-0.63, 0.63, -0.63], [0.63, -0.63, -0.63]], dtype=float)


def test_predict_rejects_unsupported_elements():
    """MagNET only knows H, C, N, O, F, S, Cl; predicting on anything else must raise, not guess.
    The check runs before the model, so dummy models are fine."""
    # phosphorus (15) in the solute
    with pytest.raises(ValueError, match="unsupported atomic numbers"):
        predict_shieldings(None, None, np.array([6, 1, 1, 15]), np.zeros((4, 3)))
    # an unsupported atom in the solvent block (the full atomic_numbers, MagNET-x path)
    with pytest.raises(ValueError, match="unsupported"):
        predict_shieldings(None, None, np.array([6, 1]), np.zeros((4, 3)),
                           atomic_numbers=np.array([6, 1, 35, 1]))   # bromine
    # a supported-only molecule passes the element check (it may fail later for lack of a real
    # model, but not with an "unsupported" element error)
    try:
        predict_shieldings(None, None, METHANE_Z, METHANE_XYZ)
    except ValueError as e:
        assert "unsupported" not in str(e)
    except Exception:
        pass


def test_resolve_checkpoint_honors_checkpoints_dir(tmp_path):
    """checkpoints_dir points at the released weights when they live outside the repo. It may be the
    model_checkpoints/ folder itself or a directory that contains one, and takes precedence."""
    from magnet.run_magnet import _resolve_checkpoint
    rel = "model_checkpoints/MagNET-Zero/MagNET-Zero_1H.ckpt"

    # (a) checkpoints_dir IS the model_checkpoints/ folder -> the leading path segment is stripped
    mc = tmp_path / "mc"
    (mc / "MagNET-Zero").mkdir(parents=True)
    inside = mc / "MagNET-Zero" / "MagNET-Zero_1H.ckpt"
    inside.write_bytes(b"")
    assert _resolve_checkpoint(rel, checkpoints_dir=str(mc)) == str(inside)

    # (b) checkpoints_dir CONTAINS a model_checkpoints/ folder (an `hf download --local-dir` target)
    parent = tmp_path / "download"
    contained = parent / "model_checkpoints" / "MagNET-Zero" / "MagNET-Zero_1H.ckpt"
    contained.parent.mkdir(parents=True)
    contained.write_bytes(b"")
    assert _resolve_checkpoint(rel, checkpoints_dir=str(parent)) == str(contained)

    # (c) a checkpoints_dir without the file is not fabricated (falls through to the defaults)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert not _resolve_checkpoint(rel, checkpoints_dir=str(empty)).startswith(str(empty))

    # (d) an absolute path resolves regardless of checkpoints_dir
    assert _resolve_checkpoint(str(inside), checkpoints_dir=str(empty)) == str(inside)


@pytest.fixture(scope="module")
def tiny_model():
    return MagNET_Lightning(TINY_PARAMS).eval()


def test_predict_shape_and_finite(tiny_model):
    out = predict_shieldings(tiny_model, tiny_model, solute_atomic_numbers=METHANE_Z,
                             geometry=METHANE_XYZ, device="cpu")
    out = np.asarray(out)
    assert out.shape == (METHANE_Z.shape[0],)
    assert np.isfinite(out).all()


def test_n_passes_averages(tiny_model):
    out = predict_shieldings(tiny_model, tiny_model, solute_atomic_numbers=METHANE_Z,
                             geometry=METHANE_XYZ, device="cpu", n_passes=3)
    assert np.asarray(out).shape == (METHANE_Z.shape[0],)
    assert np.isfinite(out).all()


def test_symmetrize_runs_and_is_reflection_consistent(tiny_model):
    # the symmetrized prediction averages over G and its mirror image, so evaluating it on G
    # and on reflect(G) must give the same per-atom values (up to per-pass frame noise).
    refl = METHANE_XYZ.copy(); refl[:, 0] = -refl[:, 0]
    a = predict_shieldings(tiny_model, tiny_model, solute_atomic_numbers=METHANE_Z,
                           geometry=METHANE_XYZ, device="cpu", n_passes=8, symmetrize=True)
    b = predict_shieldings(tiny_model, tiny_model, solute_atomic_numbers=METHANE_Z,
                           geometry=refl, device="cpu", n_passes=8, symmetrize=True)
    # both are reflection-symmetrized estimates of the same molecule
    assert np.allclose(np.asarray(a), np.asarray(b), atol=0.5)


def test_solvent_path_runs():
    """Regression test for the explicit-solvent (MagNET-x) code path. That branch runs only when
    filter_solvent_edges=True (the MagNET-x checkpoints set it), so the gas-phase tests never reach
    it; a torch_geometric >= 2.6 change (data.keys became a method) once broke it. Build a tiny
    solvent-aware model and feed a solute plus two solvent molecules."""
    model = MagNET_Lightning(dict(TINY_PARAMS, filter_solvent_edges=True)).eval()
    solute_Z = np.array([6, 1, 1, 1, 1])
    solvent_mol = np.array([[0.0, 0.0, 0.0], [0.0, 0.96, 0.0], [0.93, -0.24, 0.0]])  # 3-atom solvent
    full_Z = np.concatenate([solute_Z, [8, 1, 1], [8, 1, 1]])
    geom = np.concatenate([METHANE_XYZ, solvent_mol + [3.5, 0, 0], solvent_mol + [0, 3.5, 0]])
    out = np.asarray(predict_shieldings(model, model, solute_atomic_numbers=solute_Z, geometry=geom,
                                        atomic_numbers=full_Z, N_atoms_per_solvent=3,
                                        solvent_distance_threshold=12.0, device="cpu"))
    assert out.shape == (solute_Z.shape[0],)
    assert np.isfinite(out).all()


@pytest.mark.skipif(not os.path.exists(CKPT), reason="released MagNET-Zero checkpoints not present")
def test_real_magnet_zero_reproduces_reference():
    import h5py
    dev = "cpu"
    mH = MagNET_Lightning.load_from_checkpoint(os.path.join(CKPT, "MagNET-Zero_1H.ckpt"), map_location=dev).eval()
    mC = MagNET_Lightning.load_from_checkpoint(os.path.join(CKPT, "MagNET-Zero_13C.ckpt"), map_location=dev).eval()
    with h5py.File(NP_H5, "r") as f:
        name = list(f.keys())[0]
        g = f[name]
        z = g["atomic_numbers"][:]; xyz = g["geometry"][:].astype(np.float64)
        # This reference file predates the MagNET casing standardization, so its dataset is stored
        # under the old "MagNet_..." spelling. Read whichever spelling the file actually uses.
        ref_key = "MagNET_Zero_shieldings" if "MagNET_Zero_shieldings" in g else "MagNet_Zero_shieldings"
        ref = g[ref_key][:]
    pred = predict_shieldings(mH, mC, solute_atomic_numbers=z, geometry=xyz, device=dev, n_passes=20)
    tgt = (z == 1) | (z == 6)
    # un-symmetrized 20-pass reproduces the (un-symmetrized) released numbers well under 0.1 ppm
    assert np.abs(pred[tgt] - ref[tgt]).mean() < 0.05


@pytest.mark.skipif(not os.path.exists(CKPT), reason="released MagNET-Zero checkpoints not present")
def test_real_symmetrize_changes_13C_on_large_molecule():
    import h5py
    dev = "cpu"
    mH = MagNET_Lightning.load_from_checkpoint(os.path.join(CKPT, "MagNET-Zero_1H.ckpt"), map_location=dev).eval()
    mC = MagNET_Lightning.load_from_checkpoint(os.path.join(CKPT, "MagNET-Zero_13C.ckpt"), map_location=dev).eval()
    with h5py.File(NP_H5, "r") as f:
        # pick a large molecule (>=50 atoms) where the parity error is significant
        big = [k for k in f.keys() if f[k]["atomic_numbers"].shape[0] >= 50][0]
        g = f[big]
        z = g["atomic_numbers"][:]; xyz = g["geometry"][:].astype(np.float64)
    plain = predict_shieldings(mH, mC, solute_atomic_numbers=z, geometry=xyz, device=dev, n_passes=12)
    sym = predict_shieldings(mH, mC, solute_atomic_numbers=z, geometry=xyz, device=dev, n_passes=12, symmetrize=True)
    c = z == 6
    # symmetrization removes a real 13C parity error: it should shift carbons by > 0.05 ppm RMS
    assert np.sqrt(((plain[c] - sym[c]) ** 2).mean()) > 0.05


CKPT_FOUNDATION = os.path.join(_CKPT_ROOT, "MagNET")
CKPT_PCM = os.path.join(_CKPT_ROOT, "MagNET-PCM")

# a small, valid molecule (methyl acetate) to smoke-run the foundation and PCM checkpoints
_SMOKE_Z = np.array([6, 6, 6, 8, 8, 6, 1, 1, 1, 1, 1, 1])
_SMOKE_XYZ = np.array([
    [-2.07273936, 0.71782219, -0.27169511], [-1.51154256, -0.47834334, -0.05051489],
    [-0.06133485, -0.72471339, 0.10392854], [0.38785097, -1.84332836, 0.30459866],
    [0.65684897, 0.41875002, -0.00514369], [2.0648613, 0.23182768, 0.13713977],
    [-1.48494101, 1.6266042, -0.3569932], [-3.14927745, 0.81001383, -0.37304503],
    [-2.12480974, -1.36963022, 0.03013895], [2.44585443, -0.42836204, -0.64841408],
    [2.30088663, -0.16782625, 1.12843716], [2.5483427, 1.20718563, 0.03357037]])


@pytest.mark.skipif(not os.path.exists(CKPT_FOUNDATION),
                    reason="released MagNET (foundation) checkpoints not present")
def test_real_foundation_loads_and_runs():
    """The foundation checkpoints load and produce finite per-atom shieldings of the right shape.
    Regression for the checkpoint-family load/run path (the MagNET-X case bug was this class)."""
    dev = "cpu"
    mH = MagNET_Lightning.load_from_checkpoint(os.path.join(CKPT_FOUNDATION, "MagNET_1H.ckpt"), map_location=dev).eval()
    mC = MagNET_Lightning.load_from_checkpoint(os.path.join(CKPT_FOUNDATION, "MagNET_13C.ckpt"), map_location=dev).eval()
    pred = predict_shieldings(mH, mC, solute_atomic_numbers=_SMOKE_Z, geometry=_SMOKE_XYZ, device=dev, n_passes=2)
    assert pred.shape == (_SMOKE_Z.shape[0],)
    assert np.isfinite(pred[(_SMOKE_Z == 1) | (_SMOKE_Z == 6)]).all()


@pytest.mark.skipif(not os.path.exists(CKPT_PCM),
                    reason="released MagNET-PCM checkpoints not present")
def test_real_pcm_correction_is_finite_and_nonzero():
    """The MagNET-PCM gas and with-PCM checkpoints load, run, and genuinely differ: the chloroform
    correction (with-PCM minus gas) is finite and not identically zero over the H/C atoms."""
    dev = "cpu"
    mH = MagNET_Lightning.load_from_checkpoint(os.path.join(CKPT_PCM, "MagNET-PCM-withoutPCM_1H.ckpt"), map_location=dev).eval()
    mC = MagNET_Lightning.load_from_checkpoint(os.path.join(CKPT_PCM, "MagNET-PCM-withoutPCM_13C.ckpt"), map_location=dev).eval()
    pH = MagNET_Lightning.load_from_checkpoint(os.path.join(CKPT_PCM, "MagNET-PCM-withPCM_1H.ckpt"), map_location=dev).eval()
    pC = MagNET_Lightning.load_from_checkpoint(os.path.join(CKPT_PCM, "MagNET-PCM-withPCM_13C.ckpt"), map_location=dev).eval()
    gas = predict_shieldings(mH, mC, solute_atomic_numbers=_SMOKE_Z, geometry=_SMOKE_XYZ, device=dev, n_passes=4)
    pcm = predict_shieldings(pH, pC, solute_atomic_numbers=_SMOKE_Z, geometry=_SMOKE_XYZ, device=dev, n_passes=4)
    correction = pcm - gas
    hc = (_SMOKE_Z == 1) | (_SMOKE_Z == 6)
    assert np.isfinite(correction[hc]).all()
    assert np.abs(correction[hc]).max() > 1e-3   # the two checkpoints are not the same model


CKPT_X = os.path.join(_CKPT_ROOT, "MagNET-x")
SIGMA_FRESH = _dataset_file("sigma-fresh")  # in-repo data/sigma-fresh/, git-ignored, via the Hugging Face checkout


@pytest.mark.skipif(not (os.path.exists(CKPT_X) and os.path.exists(SIGMA_FRESH)),
                    reason="MagNET-x checkpoints or sigma-fresh data not present")
def test_real_magnet_x_reproduces_dft_correction():
    """The MagNET-x explicit-solvent correction (solvated minus isolated) should approximate the DFT
    correction stored in sigma-fresh for a single solute+solvent pose."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sigma-fresh"))
    from decode_sigma_fresh import SigmaFresh
    dev = "cpu"
    mH = MagNET_Lightning.load_from_checkpoint(os.path.join(CKPT_X, "MagNET-x-chloroform_1H.ckpt"), map_location=dev).eval()
    mC = MagNET_Lightning.load_from_checkpoint(os.path.join(CKPT_X, "MagNET-x-chloroform_13C.ckpt"), map_location=dev).eval()
    with SigmaFresh(SIGMA_FRESH) as ds:
        p = ds.pose("chloroform", 3, 1)
    ns = p["n_solute_atoms"]; nper = 5; cut = ns + p["n_solvents_partial"] * nper
    Zf = p["atomic_numbers"][:cut]; geom = p["coordinates"][:cut]; Zs = p["atomic_numbers"][:ns]
    ref = (p["shielding_solvated"] - p["shielding_isolated"])[:ns]

    def run(threshold):
        return np.asarray(predict_shieldings(
            mH, mC, solute_atomic_numbers=Zs, geometry=geom, atomic_numbers=Zf,
            N_atoms_per_solvent=nper, solvent_distance_threshold=threshold, device=dev, n_passes=8)).squeeze()

    correction = run(12.0) - run(0.0)
    keep = ((Zs == 1) | (Zs == 6)) & np.isfinite(ref)
    rms = np.sqrt(np.mean((correction[keep] - ref[keep]) ** 2))
    assert rms < 0.3   # measured ~0.085 ppm on this pose; model error, not exact
