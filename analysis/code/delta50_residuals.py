"""MagNET-Zero vs DFT residuals on the DELTA50 benchmark (50 common organics). No plotting.

MagNET-Zero is compared against the DFT level it reproduces, on the same AIMNet2 geometries: protons
vs WP04/pcSseg-2, carbons vs wB97X-D/pcSseg-2, gas phase; residual = DFT shielding minus MagNET.
This is the comparison behind the paper's DELTA50 outlier observation (nitromethane, nitroethane, and
2-methyl-2-nitropropane are the significant outliers). Read via data/delta50/decode_delta50.py.
"""
import numpy as np

from stats import summarize_errors

# the DFT level MagNET-Zero reproduces for each nucleus; MagNET-Zero is a single per-atom array
NUCLEI = {
    "H": {"atomic_number": 1, "dft": "shielding_wp04_pcSseg2", "nn": "nn_magnet_zero"},
    "C": {"atomic_number": 6, "dft": "shielding_wb97xd_pcSseg2", "nn": "nn_magnet_zero"},
}


def load_shieldings(delta50_path):
    """Flat per-atom arrays from the released reader: atomic_numbers, the MagNET predictions, the
    DFT reference shieldings, and experimental_shift (all ppm, NaN where absent)."""
    from decode_delta50 import Delta50
    with Delta50(delta50_path) as ds:
        return ds.all_atoms()


def residuals(shieldings, nucleus):
    """DFT-minus-MagNET-Zero shielding residuals (ppm) for one nucleus ("H" or "C"), over that
    element's atoms where both values are present."""
    spec = NUCLEI[nucleus]
    z = np.asarray(shieldings["atomic_numbers"])
    dft = np.asarray(shieldings[spec["dft"]], dtype=float)
    nn = np.asarray(shieldings[spec["nn"]], dtype=float)
    keep = (z == spec["atomic_number"]) & np.isfinite(dft) & np.isfinite(nn)
    return dft[keep] - nn[keep]


def residual_stats(errors):
    """count / RMSE / MAE / median AE / max absolute error for a residual array."""
    errors = np.asarray(errors, dtype=float)
    if not errors.size:
        return {"n": 0, "rmse": float("nan"), "mae": float("nan"), "median_ae": float("nan"),
                "max_ae": float("nan")}
    stats = summarize_errors(errors, np.zeros_like(errors))
    stats["max_ae"] = float(np.abs(errors).max())
    return stats


def residual_summary(delta50_path):
    """{nucleus: residual_stats} for both nuclei, from the released file."""
    shieldings = load_shieldings(delta50_path)
    return {nucleus: residual_stats(residuals(shieldings, nucleus)) for nucleus in NUCLEI}


def per_molecule_max_residual(delta50_path):
    """{molecule_name: {"H": max|residual|, "C": max|residual|}} - the DFT-vs-MagNET-Zero outlier
    magnitudes per molecule (nitromethane, nitroethane, 2-methyl-2-nitropropane rank highest for C)."""
    from decode_delta50 import Delta50
    out = {}
    with Delta50(delta50_path) as ds:
        for i in range(len(ds)):
            m = ds.molecule(i)
            z = m["atomic_numbers"]
            ref = np.where(z == 1, m["shielding_wp04_pcSseg2"],
                          np.where(z == 6, m["shielding_wb97xd_pcSseg2"], np.nan))
            r = ref - m["nn_magnet_zero"]
            res = {}
            for nucleus, zz in (("H", 1), ("C", 6)):
                sel = (z == zz) & np.isfinite(r)
                res[nucleus] = float(np.abs(r[sel]).max()) if sel.any() else float("nan")
            out[m["name"]] = res
    return out
