"""
Tests for the leveling-effect analysis core (leveling.py).

Most tests build tiny synthetic inputs in the exact on-disk formats (so the fixtures
double as a spec of what the loaders expect) and need no large data. Two opt-in tests
run against the real delta22.hdf5 and the Kaupp NS372 spreadsheet if they are present.

Run:  pytest test_leveling.py -q
Requires: pytest, numpy, h5py, pandas, openpyxl, scikit-learn.
"""
import os
import numpy as np
import h5py
import openpyxl
import pytest

from fixed_point import MISSING_MARKER as _MISSING_MARKER
from leveling import (
    _decode_fixed_point,
    analyze_nucleus,
    read_nucleus,
    load_delta22,
    load_ns372,
    load_ns372_1h_convergence,
    convergence_statistics,
    DELTA22_REF,
)

SCALE = 1e4


def _enc(arr):
    """Encode floats the way the real file does: int32 = round(value * 1e4), NaN -> marker."""
    a = np.asarray(arr, dtype=np.float64)
    mask = np.isnan(a)
    out = np.zeros(a.shape, dtype=np.int32)
    out[~mask] = np.rint(a[~mask] * SCALE).astype(np.int32)
    out[mask] = _MISSING_MARKER
    return out


# ---------------------------------------------------------------------------
# fixed-point decoding
# ---------------------------------------------------------------------------
def test_decode_round_trip_and_marker():
    values = np.array([0.0, 207.07, -3.5, np.nan, 12345.6789], dtype=np.float64)
    decoded = _decode_fixed_point(_enc(values))
    assert np.isnan(decoded[3])
    np.testing.assert_allclose(decoded[~np.isnan(decoded)],
                               values[~np.isnan(values)], atol=1e-4)


def test_decode_passes_through_float():
    # an older plain-decimal copy must read unchanged (decode is a no-op on floats)
    values = np.array([1.0, 2.5, -7.25], dtype=np.float64)
    np.testing.assert_array_equal(_decode_fixed_point(values), values)


def test_decode_handles_int64_and_no_marker():
    # a re-encode at int64 must decode like int32 (any integer dtype), and an
    # integer array with no missing marker must round-trip with no spurious blanks
    vals = np.array([10000, 20000, -35000], dtype=np.int64)   # = 1.0, 2.0, -3.5
    out = _decode_fixed_point(vals)
    np.testing.assert_allclose(out, [1.0, 2.0, -3.5])
    assert not np.isnan(out).any()


# ---------------------------------------------------------------------------
# leveling analysis on a synthetic, perfectly-leveled matrix
# ---------------------------------------------------------------------------
def test_analyze_nucleus_detects_leveling():
    # build methods that are affine transforms of one base signal plus tiny noise:
    # after scaling they are interchangeable, so PC1 should dominate and every
    # pairwise correlation should be near 1.
    rng = np.random.RandomState(0)
    base = rng.normal(size=60)
    slopes = np.linspace(0.8, 1.3, 8)
    offsets = np.linspace(-5, 5, 8)
    cols = [a * base + b + 0.01 * rng.normal(size=60)
            for a, b in zip(slopes, offsets)]
    M = np.column_stack(cols)
    methods = [f"m{i}" for i in range(8)]
    res = analyze_nucleus(M, methods, ref=M[:, 0])
    assert res['ev'][0] > 0.99            # PC1 carries essentially all variance
    assert res['r_min'] > 0.99            # worst pairwise correlation still near 1
    assert set(res['scaled_rmse']) == set(methods)
    assert 'r2' in res['par'] and 'theta' in res['par']


# ---------------------------------------------------------------------------
# delta22 loader against a synthetic int32 HDF5 in the real layout
# ---------------------------------------------------------------------------
DELTA22_FUNCS = ['hf', 'blyp_d3bj', 'bp86_d3bj', 'b97d3_d3bj', 'tpsstpss_d3bj',
                 'b3lyp_d3bj', 'pbe0_d3bj', 'm062x_d3', 'wb97xd', 'wp04', 'wc04',
                 'B2PLYP', 'mPW2PLYP', 'B2GP_PLYP', 'dsd_pbep86', 'revdsd_pbep86',
                 'mp2', 'dlpno_mp2']


def _make_synthetic_delta22(path):
    # one gas method label per functional. most sit at pcSseg3, but mp2 is only at
    # pcSseg2 and hf only at pcSseg1, so the loader's largest-basis fallback ladder
    # (pcSseg3 then pcSseg2 then pcSseg1) is exercised.
    basis_for = {f: "pcSseg3" for f in DELTA22_FUNCS}
    basis_for["mp2"] = "pcSseg2"
    basis_for["hf"] = "pcSseg1"
    names = [f"{f},{basis_for[f]},gas,none" for f in DELTA22_FUNCS]
    n_methods = len(names)
    with h5py.File(path, "w") as f:
        f.create_dataset("conventional_nmr_method_names",
                         data=np.array(names, dtype=h5py.string_dtype()))
        # one solute, atoms H, C, H, C, N  (N is ignored: not 1H/13C)
        z = np.array([1, 6, 1, 6, 7], np.uint8)
        g = f.create_group("solutes/mol1")
        g.create_dataset("atomic_numbers", data=z)
        # conventional_shieldings[method, geometry, atom]; geometry 1 = PBE0/cc-pVTZ
        base = np.array([30.0, 150.0, 31.0, 151.0, 200.0])
        cs = np.zeros((n_methods, 2, len(z)))
        for mi in range(n_methods):
            cs[mi, 0] = base + 0.5 * mi          # aimnet2 geometry (unused)
            cs[mi, 1] = base + 0.01 * mi         # pbe0_tz geometry (used)
        g.create_dataset("stationary_and_pcm/conventional_shieldings", data=_enc(cs))
    return base


def test_load_delta22_synthetic(tmp_path):
    p = tmp_path / "delta22.hdf5"
    base = _make_synthetic_delta22(str(p))
    out = load_delta22(str(p))
    assert set(out) == {"1H", "13C"}
    # 1 solute has two H sites (atoms 0,2) and two C sites (atoms 1,3)
    assert out["1H"]["M"].shape == (2, len(DELTA22_FUNCS))
    assert out["13C"]["M"].shape == (2, len(DELTA22_FUNCS))
    assert DELTA22_REF in out["1H"]["methods"]
    # every functional was selected via the basis-fallback ladder, so no method is missing
    assert not np.isnan(out["1H"]["M"]).any()
    assert not np.isnan(out["13C"]["M"]).any()
    # decoded values must match what we wrote on the pbe0_tz geometry; pick the
    # DSD-PBEP86 method column for the first H site (atom 0)
    ref_method_index = DELTA22_FUNCS.index("dsd_pbep86")
    expected = base[0] + 0.01 * ref_method_index
    col = out["1H"]["methods"].index(DELTA22_REF)
    np.testing.assert_allclose(out["1H"]["M"][0, col], expected, atol=1e-4)


# ---------------------------------------------------------------------------
# NS372 sheet parsing against a synthetic Kaupp-style worksheet
# ---------------------------------------------------------------------------
def _make_synthetic_kaupp_sheet(path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "S6 - 1H Shieldings"
    # row 1: functional headers from column 7; row 2: tau sub-labels
    # two methods: SVWN (no tau) and TPSS (tau = tauC), matching leveling.METHODS keys
    ws.cell(1, 7, "SVWN")
    ws.cell(1, 8, "TPSS"); ws.cell(2, 8, "τC")
    # data rows from row 4: col1 nucleus (sticky), col2 molecule, col4 CCSD(T) ref
    rows = [("1H", "molA", 30.0, 30.1, 30.2),
            ("1H", "molB", 28.0, 28.2, 28.1),
            ("1H", "molC", 25.0, 25.3, 25.1)]
    for i, (nuc, mol, ref, svwn, tpss) in enumerate(rows):
        r = 4 + i
        ws.cell(r, 1, nuc)
        ws.cell(r, 2, mol)
        ws.cell(r, 4, ref)
        ws.cell(r, 7, svwn)
        ws.cell(r, 8, tpss)
    wb.save(path)


def test_read_nucleus_synthetic(tmp_path):
    p = tmp_path / "kaupp.xlsx"
    _make_synthetic_kaupp_sheet(str(p))
    wb = openpyxl.load_workbook(str(p), data_only=True)
    ref, data = read_nucleus(wb, "S6 - 1H Shieldings")
    np.testing.assert_allclose(ref, [30.0, 28.0, 25.0])
    assert ("SVWN", None) in data
    assert ("TPSS", "τC") in data
    np.testing.assert_allclose(data[("SVWN", None)], [30.1, 28.2, 25.3])
    np.testing.assert_allclose(data[("TPSS", "τC")], [30.2, 28.1, 25.1])


# ---------------------------------------------------------------------------
# opt-in tests against the real files (skipped when absent, e.g. in CI)
# ---------------------------------------------------------------------------
def _find(*candidates):
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None

_HERE = os.path.dirname(os.path.abspath(__file__))
import paths
_REAL_DELTA22 = _find(
    paths.dataset_file("delta22", file=__file__),   # deposit copy, then in-repo data/delta22/
    os.path.join(_HERE, "delta22.hdf5"),
    os.path.join(_HERE, "delta22", "delta22.hdf5"),
)
_REAL_KAUPP = _find(
    os.path.join(_HERE, "..", "..", "data", "ns372", "ct1c00919_si_002.xlsx"),
)


@pytest.mark.skipif(_REAL_DELTA22 is None, reason="real delta22.hdf5 not present")
def test_real_delta22_reproduces_leveling():
    out = load_delta22(_REAL_DELTA22)
    assert out["1H"]["M"].shape == (145, 18)
    assert out["13C"]["M"].shape == (76, 18)
    res = analyze_nucleus(out["1H"]["M"], out["1H"]["methods"], out["1H"]["ref"])
    assert res['ev'][0] > 0.99            # PC1 dominance on the real proton data


@pytest.mark.skipif(_REAL_KAUPP is None, reason="real Kaupp NS372 spreadsheet not present")
def test_real_ns372_loads_all_nuclei():
    out = load_ns372(_REAL_KAUPP)
    assert set(out) == {"1H", "11B", "13C", "15N", "17O", "19F", "31P", "33S"}
    res = analyze_nucleus(out["13C"]["M"], out["13C"]["methods"], out["13C"]["ref"])
    assert res['ev'][0] > 0.99


@pytest.mark.skipif(_REAL_KAUPP is None, reason="real Kaupp NS372 spreadsheet not present")
def test_real_ns372_1h_convergence_stats():
    # Figure 2D input: the 1H slice reduces to 44 methods vs the CCSD(T) reference,
    # and each method's slope/scaled RMSE lands in the expected physical range.
    cc = load_ns372_1h_convergence(_REAL_KAUPP)
    assert cc.shape == (124, 45)                       # 44 methods + CCSD(T)
    assert "CCSD(T)" in cc.columns and "HF" in cc.columns   # 'Hartree-Fock' header renamed
    res = convergence_statistics(cc).set_index("Method")
    assert len(res) == 44
    assert (res["Slope"].between(0.9, 1.15)).all()     # near the y=x limit
    assert (res["RMSE (scaled)"] > 0).all()
    # the double hybrid sits closest to the electronic-structure limit
    assert res["RMSE (scaled)"].idxmin() == "DSD-PBEP86"
    assert res.loc["DSD-PBEP86", "RMSE (scaled)"] < 0.1
