"""Tests for the applications (natural-products) loader.

These build a tiny synthetic applications.hdf5 + experimental spreadsheet in a temp dir that
mimic the real schema, so the loader and its decoding can be checked in CI without the multi-
hundred-megabyte release files.
"""
import numpy as np
import pandas as pd
import h5py
import pytest

from applications_reader import Applications, _decode, SHELL_SIZES, SOLVENTS

SCALE = 1e4
MISS = np.int32(-2147483648)


def _enc(a):
    a = np.asarray(a, dtype=np.float64)
    nan = np.isnan(a)
    out = np.where(nan, 0.0, a)
    out = np.round(out * SCALE).astype(np.int32)
    out[nan] = MISS
    return out


@pytest.fixture
def fake_release(tmp_path):
    """A 2-solute synthetic release: vomicine (with a missing shell) + isomer_1E."""
    h5 = tmp_path / "applications.hdf5"
    xlsx = tmp_path / "applications_experimental.xlsx"
    rng = np.random.default_rng(0)

    def mk_solute(g, n_atoms, atomic_numbers, shells):
        g.create_dataset("atomic_numbers", data=np.asarray(atomic_numbers, dtype=np.int32))
        mz = g.create_group("magnet_zero")
        # MagNET predicts H/C only -> heteroatoms read 0 (mirror the real data)
        hetero = ~np.isin(np.asarray(atomic_numbers), [1, 6])
        stat = rng.normal(100, 30, n_atoms); stat[hetero] = 0.0
        pcm = rng.normal(0, 2, n_atoms); pcm[hetero] = 0.0
        mz.create_dataset("stationary", data=_enc(stat))
        mz.create_dataset("pcm_correction", data=_enc(pcm))
        cf = mz.create_group("conformers").create_group("conformer_0")
        cf.create_dataset("energy", data=np.float64(-1234.5))   # float, not encoded
        cf.create_dataset("geometry", data=_enc(rng.normal(0, 5, (n_atoms, 3))))
        cf.create_dataset("atomic_numbers", data=np.asarray(atomic_numbers, dtype=np.int32))
        cf.create_dataset("magnet_zero", data=_enc(rng.normal(100, 30, n_atoms)))
        cf.create_dataset("pcm", data=_enc(rng.normal(0, 2, n_atoms)))
        q = g.create_group("qcd")
        q.create_dataset("stationary", data=_enc(rng.normal(0, 50, (n_atoms, 4))))
        q.create_dataset("trajectories", data=_enc(rng.normal(0, 50, (3, 4, n_atoms, 4))))
        mx = g.create_group("magnet_x")
        for sv in SOLVENTS:
            svg = mx.create_group(sv)
            for sh in shells:
                svg.create_dataset(f"shell_{sh}", data=_enc(rng.normal(100, 30, (5, n_atoms, 2))))
            svg.create_dataset("stationary", data=_enc(rng.normal(100, 30, n_atoms)))
        dr = g.create_group("dft_reference")
        # dft_reference uses gas + CPCM-continuum 'water' (not the explicit 'TIP4P'), like the real data
        for sv in ["gas", "water", "benzene", "chloroform", "methanol"]:
            dr.create_group(sv).create_dataset("shieldings", data=_enc(rng.normal(100, 40, n_atoms)))
        gg = dr.create_group("geometries").create_group("conformer_001")
        gg.create_dataset("coordinates", data=_enc(rng.normal(0, 5, (n_atoms, 3))))
        gg.create_dataset("atomic_numbers", data=np.asarray(atomic_numbers, dtype=np.int32))
        gg.create_dataset("energy", data=np.float64(-100.0))

    with h5py.File(h5, "w") as f:
        sg = f.create_group("solvents")
        for sv in SOLVENTS:
            sg.create_group(sv).create_dataset("atomic_numbers", data=np.array([6, 1, 1], dtype=np.int32))
        # vomicine: drop one shell (mimic the isomer_1Z/chloroform/250 corruption)
        mk_solute(f.create_group("vomicine"), 4, [8, 7, 6, 1], [s for s in SHELL_SIZES if s != 250])
        mk_solute(f.create_group("isomer_1E"), 3, [6, 1, 1], SHELL_SIZES)
        # composite_model group: fit tables stored verbatim as CSV text
        cm = f.create_group("composite_model")
        cm.create_group("pcm_conversion_factors").create_dataset(
            "H", data="solvent,pcm_conversion_factor\nchloroform,1.0\nbenzene,0.6\n")
        cm.create_group("ols_coefficients").create_group("stationary_plus_pcm").create_dataset(
            "H", data="parameter,chloroform,benzene\nintercept,31.2,31.3\nstationary,-0.98,-0.99\n")
        cm.create_group("bootstrap_coefficients").create_group("stationary_plus_pcm").create_dataset(
            "H", data="solvent,seed,Intercept,stationary,pcm\nchloroform,0,31.2,-0.98,-1.0\nbenzene,0,31.3,-0.99,-0.6\n")
        cm.create_group("rmse_distributions").create_dataset(
            "H", data="solvent,nucleus,formula,seed,Bootstrap_RMSE\nchloroform,H,pcm2,0,0.07\nbenzene,H,pcm2,0,0.21\n")

    # experiment xlsx: one sheet per solute
    with pd.ExcelWriter(xlsx) as w:
        for sol in ["vomicine", "isomer_1E"]:
            pd.DataFrame({
                "site": ["a_C", "b_H"], "nucleus": ["C", "H"], "atom_numbers": ["3", "4"],
                "chloroform": [1.0, 2.0], "benzene": [1.1, 2.1],
                "methanol": [1.2, 2.2], "water": [1.3, 2.3],
            }).to_excel(w, sheet_name=sol, index=False)
    return Applications(h5, xlsx)


def test_decode_roundtrip_and_missing():
    vals = np.array([1.2345, -50.0, 0.0], dtype=np.float64)
    enc = _enc(vals)
    out = _decode(enc)
    assert np.allclose(out, vals, atol=5e-5)
    enc_with_miss = np.append(enc, MISS).astype(np.int32)
    assert np.isnan(_decode(enc_with_miss)[-1])


def test_decode_int64_is_still_scaled():
    # a rebuild that stored the columns as int64 (any integer width) must decode the same way as
    # int32; the old exact `dtype == np.int32` check fell through and returned raw scaled integers.
    vals = np.array([1.2345, -50.0, 0.0], dtype=np.float64)
    enc = _enc(vals).astype(np.int64)
    out = _decode(enc)
    assert np.allclose(out, vals, atol=5e-5)
    enc_with_miss = np.append(enc, np.int64(-2147483648))
    assert np.isnan(_decode(enc_with_miss)[-1])


def test_solutes_and_solvents_group_excluded(fake_release):
    assert set(fake_release.solutes()) == {"vomicine", "isomer_1E"}


def test_magnet_zero_heteroatoms_read_zero(fake_release):
    # vomicine atomic_numbers [8, 7, 6, 1]: the O and N positions must decode to 0 (MagNET does H/C only)
    mz = fake_release.magnet_zero("vomicine")
    hetero = ~np.isin(fake_release.atomic_numbers("vomicine"), [1, 6])
    assert np.allclose(mz["stationary"][hetero], 0.0)
    assert np.allclose(mz["pcm_correction"][hetero], 0.0)
    assert np.any(mz["stationary"][~hetero] != 0.0)  # H/C carry real values


def test_qcd_shielding_only_drops_xyz(fake_release):
    full = fake_release.qcd("isomer_1E")
    just = fake_release.qcd("isomer_1E", shielding_only=True)
    assert full["stationary"].shape == (3, 4)
    assert just["stationary"].shape == (3,)
    assert just["trajectories"].shape == (3, 4, 3)
    assert np.allclose(just["stationary"], full["stationary"][..., 3], atol=5e-5)


def test_magnet_x_and_shell_guard(fake_release):
    mx = fake_release.magnet_x("isomer_1E", "benzene", 650)
    assert mx.shape == (5, 3, 2)
    # vomicine is missing shell 250 -> available_shells must exclude it, not crash
    assert 250 not in fake_release.available_shells("vomicine", "chloroform")
    assert fake_release.available_shells("isomer_1E", "chloroform") == SHELL_SIZES


def test_dft_reference_and_conformers(fake_release):
    assert fake_release.dft_reference("vomicine", "gas").shape == (4,)
    conf = fake_release.magnet_zero_conformers("vomicine")
    assert conf["conformer_0"]["energy"] == pytest.approx(-1234.5)
    assert conf["conformer_0"]["geometry"].shape == (4, 3)


def test_experiment_water_rename(fake_release):
    df = fake_release.experiment()
    assert "TIP4P" in df.columns and "water" not in df.columns
    assert set(df["solute"].unique()) == {"vomicine", "isomer_1E"}


def test_site_atom_table(fake_release):
    tbl = fake_release.site_atom_table()
    assert list(tbl.columns) == ["solute", "site", "nucleus", "atom_numbers"]
    assert set(tbl["solute"].unique()) == {"vomicine", "isomer_1E"}
    assert set(str(v) for v in tbl["atom_numbers"]) == {"3", "4"}


def test_composite_model_accessors(fake_release):
    conv = fake_release.pcm_conversion_factors("H")
    assert list(conv.columns) == ["solvent", "pcm_conversion_factor"]
    assert conv.set_index("solvent").loc["chloroform", "pcm_conversion_factor"] == 1.0
    ols = fake_release.ols_coefficients("stationary_plus_pcm", "H")
    assert "parameter" in ols.columns and "chloroform" in ols.columns
    assert fake_release.composite_formulas("ols_coefficients") == ["stationary_plus_pcm"]
    boot = fake_release.bootstrap_coefficients("stationary_plus_pcm", "H")
    assert {"solvent", "seed", "Intercept", "stationary", "pcm"}.issubset(boot.columns)
    rd = fake_release.rmse_distribution("H")
    assert {"solvent", "formula", "Bootstrap_RMSE"}.issubset(rd.columns)


def test_dft_reference_uses_water_not_tip4p(fake_release):
    # the DFT reference is CPCM-continuum 'water'; the explicit 'TIP4P' key does not exist there
    assert fake_release.dft_reference("vomicine", "water").shape == (4,)
    with pytest.raises(KeyError):
        fake_release.dft_reference("vomicine", "TIP4P")


def test_dft_reference_geometries(fake_release):
    geo = fake_release.dft_reference_geometries("vomicine")
    assert "conformer_001" in geo
    c = geo["conformer_001"]
    assert c["coordinates"].shape == (4, 3)
    assert c["energy"] == pytest.approx(-100.0)
    assert list(c["atomic_numbers"]) == [8, 7, 6, 1]


def test_md_geometry_companion(tmp_path):
    md = tmp_path / "applications_md_geometries.hdf5"
    with h5py.File(md, "w") as f:
        g = f.create_group("vomicine")
        g.create_dataset("openMM_stationary_geometry", data=_enc(np.zeros((4, 3))))
        sv = g.create_group("chloroform")
        sv.create_dataset("geometries_50", data=_enc(np.ones((5, 7, 3))))
        sv.create_dataset("atomic_numbers_50", data=np.arange(7, dtype=np.int32))
    app = Applications(tmp_path / "applications.hdf5", md_geometries_path=md)
    assert app.available_md_shells("vomicine", "chloroform") == [50]
    assert app.md_geometry("vomicine", "chloroform", 50).shape == (5, 7, 3)
    assert app.md_geometry_atomic_numbers("vomicine", "chloroform", 50).shape == (7,)
    assert app.md_stationary_geometry("vomicine").shape == (4, 3)


def test_md_geometry_absent_raises_clearly(fake_release):
    # the big companion is optional; accessing it without it gives a clear, actionable error
    with pytest.raises(ValueError, match="companion"):
        fake_release.available_md_shells("vomicine", "chloroform")


def test_uuid_prefix_h_sites(fake_release):
    # each solute's single H site (fixture site "b_H") becomes "01 b_H"; C sites untouched
    tbl = fake_release.site_atom_table(uuid_prefix_h_sites=True)
    h_sites = tbl[tbl["nucleus"] == "H"]["site"].tolist()
    assert all(s.startswith("01 ") for s in h_sites)
    c_sites = tbl[tbl["nucleus"] == "C"]["site"].tolist()
    assert all(not s[:2].isdigit() for s in c_sites)
